"""Calib-only Head budgets and paired summaries for the fixed 6-bit comparison."""

from pathlib import Path

import numpy as np

from experiments.head_analysis.common.io import atomic_file, read_json, write_json
from experiments.head_analysis.runtime.kv_quantization import (
    GROUP_SIZE,
    PROTECTED_TOKENS,
    HeadBitPlan,
    head_plan_byte_ledger,
)


POLICIES = tuple(f"{side}_{method}" for side in ("key", "value") for method in ("uniform6", "head_aware", "shifted"))
PAIR_POLICIES = tuple(f"{side}_{method}" for side in ("key", "value") for method in ("uniform6", "fixed", "shifted"))
LOCAL_GATE_THRESHOLDS = {
    "minimum_improvement": 0.05,
    "minimum_improved_inputs": 4,
    "maximum_condition_regression": 0.10,
}
COMBINATION_GATE_THRESHOLDS = {
    **LOCAL_GATE_THRESHOLDS,
    "absolute_kl_floor": 1e-12,
    "background_multiplier": 5,
}


def choose_local_pairs(cost):
    """Minimize [bits=4/6/8,layer,head] additive cost over all-6 or one 4/8 pair.

    Ascending donor/receiver traversal implements the frozen exact-tie rule.
    Negative/nonmonotone gains are retained; the all-6 state is always feasible.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 3 or cost.shape[0] != 3 or cost.shape[2] < 2 or not np.isfinite(cost).all():
        raise ValueError("pair fitting requires finite 4/6/8-bit costs for at least two Heads")
    _, layers, heads = cost.shape
    bits = np.full((layers, heads), 6, dtype=np.uint8)
    for layer in range(layers):
        best = 0.0
        for donor in range(heads):
            penalty = cost[0, layer, donor] - cost[1, layer, donor]
            for receiver in range(heads):
                if donor == receiver:
                    continue
                gain = cost[1, layer, receiver] - cost[2, layer, receiver] - penalty
                if gain > best:
                    best, pair = gain, (donor, receiver)
        if best > 0:
            bits[layer, pair[0]], bits[layer, pair[1]] = 4, 8
    return bits


def local_plan_cost(cost, bits):
    cost, bits = np.asarray(cost), np.asarray(bits)
    return float(np.take_along_axis(cost, ((bits.astype(int) - 4) // 2)[None], axis=0).mean())


def analyze_local_pairs(costs, sample_ids, conditions, old_plans):
    """First gate only: fixed Calib plan versus per-input local Oracle, K/V apart."""
    costs = np.asarray(costs, dtype=np.float64)
    if (
        costs.ndim != 5
        or costs.shape[:3] != (len(sample_ids), 2, 3)
        or len(set(sample_ids)) != len(sample_ids)
        or len(conditions) != len(sample_ids)
        or len(set(conditions)) != len(conditions)
        or not np.isfinite(costs).all()
        or np.any(costs < 0)
    ):
        raise ValueError("local gate requires complete, nonnegative costs and one Calib input per condition")
    inputs, _, _, layers, heads = costs.shape
    plans, sides, per_input = {}, {}, [{"sample_id": sid, "condition": c} for sid, c in zip(sample_ids, conditions)]
    fixed_bits = np.empty((2, layers, heads), np.uint8)
    oracle_bits = np.empty((inputs, 2, layers, heads), np.uint8)
    objectives = {name: np.empty((inputs, 2)) for name in ("uniform6", "fixed", "oracle", "old_half")}
    old_pair_gains = np.empty((2, layers))
    for si, side in enumerate(("key", "value")):
        mean = costs[:, si].mean(0)
        bits = choose_local_pairs(mean)
        fixed_bits[si] = bits
        old = old_plans[f"{side}_head_aware"]
        if np.asarray(old.bits).shape != (layers, heads):
            raise ValueError("old allocation geometry differs from the local cost table")
        plans[f"{side}_uniform6"] = HeadBitPlan(side, tuple((6,) * heads for _ in range(layers)), old.head_dim)
        plans[f"{side}_fixed"] = HeadBitPlan(side, tuple(map(tuple, bits.tolist())), old.head_dim)
        old_scores = mean[0] - mean[2]
        for layer in range(layers):
            receiver = int(np.argsort(-old_scores[layer], kind="stable")[0])
            donor = next(int(h) for h in np.argsort(old_scores[layer], kind="stable") if h != receiver)
            old_pair_gains[si, layer] = (
                mean[1, layer, receiver] - mean[2, layer, receiver] - mean[0, layer, donor] + mean[1, layer, donor]
            )
        for i, cost in enumerate(costs[:, si]):
            oracle = choose_local_pairs(cost)
            oracle_bits[i, si] = oracle
            values = {
                "uniform6": float(cost[1].mean()),
                "fixed": local_plan_cost(cost, bits),
                "oracle": local_plan_cost(cost, oracle),
                "old_half": local_plan_cost(cost, old.bits),
            }
            tolerance = 1e-12 + 1e-10 * abs(values["uniform6"])
            if values["oracle"] > min(values["uniform6"], values["fixed"]) + tolerance:
                raise FloatingPointError("local Oracle is worse than a feasible fixed/uniform plan")
            per_input[i][side] = values | {
                "fixed_minus_oracle": values["fixed"] - values["oracle"],
                "oracle_difference_is_roundoff": values["oracle"] > values["uniform6"],
            }
            for name, value in values.items():
                objectives[name][i, si] = value
        u, f, o = (objectives[name][:, si] for name in ("uniform6", "fixed", "oracle"))
        u_mean, f_mean, o_mean = map(float, (u.mean(), f.mean(), o.mean()))
        tolerance = 1e-12 + 1e-10 * abs(u_mean)
        if f_mean > u_mean + tolerance:
            raise FloatingPointError("Calib-fitted local plan is worse than its feasible uniform baseline")
        defined = u_mean > tolerance
        fixed_gain = 1 - f_mean / u_mean if defined else None
        oracle_gain = 1 - o_mean / u_mean if defined else None
        input_tolerance = 1e-12 + 1e-10 * np.abs(u)
        wins = int(np.sum(u - f > input_tolerance))
        bad = np.flatnonzero(f - u > LOCAL_GATE_THRESHOLDS["maximum_condition_regression"] * u + input_tolerance)
        if not defined:
            decision = "stop_undefined_uniform_reference"
        elif oracle_gain < LOCAL_GATE_THRESHOLDS["minimum_improvement"]:
            decision = "stop_insufficient_local_opportunity"
        elif (
            fixed_gain < LOCAL_GATE_THRESHOLDS["minimum_improvement"]
            or wins < LOCAL_GATE_THRESHOLDS["minimum_improved_inputs"]
            or len(bad)
        ):
            decision = "stop_fixed_profile_not_stable"
        else:
            decision = "eligible_for_combination_review"
        sides[side] = {
            "mean_J": {name: float(values[:, si].mean()) for name, values in objectives.items()},
            "fixed_improvement": fixed_gain,
            "oracle_improvement": oracle_gain,
            "improved_input_count": wins,
            "inputs_over_regression_limit": [sample_ids[i] for i in bad],
            "layers_with_exchange": int(np.sum((bits == 4).any(1))),
            "old_score_positive_pair_layers": int(np.sum(old_pair_gains[si] > 0)),
            "gate_decision": decision,
        }
    report = {
        "input_count": inputs,
        "bit_widths": [4, 6, 8],
        "thresholds": LOCAL_GATE_THRESHOLDS,
        "sides": sides,
        "per_input": per_input,
        "next_stage_started": False,
        "scope": "Calib primary-point local additive objective only; Oracle is not a final-KL upper bound",
    }
    arrays = {
        "output_nmse": costs,
        "fixed_bits": fixed_bits,
        "oracle_bits": oracle_bits,
        "old_score_pair_gain": old_pair_gains,
        **{f"J_{name}": values for name, values in objectives.items()},
    }
    return report, arrays, plans


def fit_budget_plans(source_calib):
    """Exactly half of each layer's Heads get 8 bits; no Validation/Test reads."""
    source = Path(source_calib)
    state, cfg = read_json(source / "run_metadata.json"), read_json(source / "config.json")
    ids = [row["sample_id"] for row in cfg["calib_samples"]]
    if not ids or state["status"] != "completed" or state["cohort"] != "calib" or state["completed_sample_ids"] != ids:
        raise ValueError("budget fitting requires completed frozen Calib samples")
    protocol = {
        "bit_widths": [4, 8],
        "sides": ["key", "value"],
        "group_size": GROUP_SIZE,
        "protected_prompt_tokens_each_end": PROTECTED_TOKENS,
        "key_representation": "post_rope",
        "dtype": "bfloat16",
    }
    if any(cfg.get(key) != value for key, value in protocol.items()):
        raise ValueError("Calib profile differs from the existing grouped K_post/V protocol")
    geometry = state["geometry"]
    layers, heads, dim = (geometry[k] for k in ("num_layers", "kv_heads", "head_dim"))
    if heads % 2:
        raise ValueError("the equal-byte 4/8 versus 6-bit comparison requires an even Head count")
    with np.load(source / "calib_profile.npz") as archive:
        cost = archive["local_output_nmse"].copy()
        count = archive["local_output_nmse_valid_condition_count"]
        if (
            cost.shape != (2, 2, layers, heads)
            or count.shape != cost.shape
            or not np.isfinite(cost).all()
            or not np.all(count == len(ids))
        ):
            raise ValueError("Calib local NMSE must cover both sides/widths and all layer/Heads")
    if len({row["condition"] for row in cfg["calib_samples"]}) != len(ids):
        raise ValueError("the source profile must retain equal weighting of the six conditions")
    plans, scores = {}, cost[:, 0] - cost[:, 1]
    for side_id, side in enumerate(("key", "value")):
        uniform = np.full((layers, heads), 6, np.uint8)
        aware = np.full((layers, heads), 4, np.uint8)
        # With two equal-cost bit-width choices this minimizes the additive
        # 4/8-bit Calib cost. It makes no prediction about the uniform 6-bit error.
        for layer in range(layers):
            selected = np.argsort(-scores[side_id, layer], kind="stable")[: heads // 2]
            aware[layer, selected] = 8
        for method, matrix in (("uniform6", uniform), ("head_aware", aware), ("shifted", np.roll(aware, 1, axis=1))):
            plans[f"{side}_{method}"] = HeadBitPlan(side, tuple(map(tuple, matrix.tolist())), dim)
        a = head_plan_byte_ledger(plans[f"{side}_uniform6"], 192, 0)
        for method in ("head_aware", "shifted"):
            b = head_plan_byte_ledger(plans[f"{side}_{method}"], 192, 0)
            for field in ("code_bytes", "statistic_bytes", "request_bytes", "model_bytes", "total_bytes"):
                if a[field] != b[field]:
                    raise ValueError(f"this geometry cannot provide exactly equal encoded budgets: {field}")
    config = {
        "schema_version": "headkv-head-budget-v1",
        "source_calib_directory": str(source.resolve()),
        "source_config": cfg,
        "geometry": geometry,
        "calib_sample_ids": ids,
        "validation_sample_ids": [row["sample_id"] for row in cfg["validation_samples"]],
        "rule": "per layer top H/2 by mean Calib local Output NMSE(4)-NMSE(8); ties: ascending Head ID",
        "identity_control": "shift each layer's fitted bit row by +1 Head; no refit or offset search",
        "scores": scores.tolist(),
        "plans": {
            name: {"side": plan.side, "bits": [list(row) for row in plan.bits], "head_dim": plan.head_dim}
            for name, plan in plans.items()
        },
        "scope": "exploratory combination replay on the existing Calib/Validation inputs; Test is not used",
    }
    return config, plans


def write_budget_plans(directory, config, plans):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "allocation.json", config)
    for name, plan in plans.items():
        with atomic_file(directory / (name + ".bin"), binary=True) as handle:
            handle.write(plan.to_bytes())


def read_budget_plans(directory, *, policies=POLICIES):
    directory = Path(directory)
    config = read_json(directory / "allocation.json")
    plans = {}
    for name in policies:
        row = config["plans"][name]
        plan = HeadBitPlan.from_bytes((directory / (name + ".bin")).read_bytes())
        expected = HeadBitPlan(row["side"], tuple(map(tuple, row["bits"])), row["head_dim"])
        if plan != expected:
            raise ValueError("serialized Head allocation differs from its frozen definition")
        plans[name] = plan
    return config, plans


def read_pair_replay_plans(local_gate_directory):
    """Read frozen gate-one descriptors and derive only the prescribed +1 control."""
    root = Path(local_gate_directory)
    state, summary = read_json(root / "run_metadata.json"), read_json(root / "summary.json")
    names = tuple(f"{s}_{m}" for s in ("key", "value") for m in ("uniform6", "fixed"))
    allocation, plans = read_budget_plans(root / "allocation", policies=names)
    if (
        state["status"] != "completed"
        or state["cohort"] != "calib"
        or state["completed_sample_ids"] != allocation["calib_sample_ids"]
        or state["target_sample_ids"] != allocation["calib_sample_ids"]
        or allocation["source_config"] != read_json(root / "config.json")["source_config"]
        or allocation["thresholds"] != LOCAL_GATE_THRESHOLDS
        or set(allocation["eligible_sides"]) != {"key", "value"}
        or any(summary["sides"][s]["gate_decision"] != "eligible_for_combination_review" for s in ("key", "value"))
    ):
        raise ValueError("pair replay requires completed and passed frozen K/V local gates")
    for side in ("key", "value"):
        fixed = plans[f"{side}_fixed"]
        shifted = np.roll(np.asarray(fixed.bits), 1, axis=1)
        plans[f"{side}_shifted"] = HeadBitPlan(side, tuple(map(tuple, shifted.tolist())), fixed.head_dim)
    frozen = {
        **allocation,
        "schema_version": "headkv-pair-combination-v1",
        "local_gate_directory": str(root),
        "shift": 1,
        "combination_thresholds": COMBINATION_GATE_THRESHOLDS,
        "plans": {
            name: {"side": p.side, "bits": [list(row) for row in p.bits], "head_dim": p.head_dim}
            for name, p in plans.items()
        },
    }
    return frozen, plans


def summarize_budget_samples(samples, *, fixed_method="head_aware"):
    """First average positions within each input, then equally weight conditions.

    PG19 lengths are reported as separate conditions but remain the same source;
    these descriptive means/win counts are not independent-source significance tests.
    """
    fields = ("prediction_kl", "delta_nll", "top1_match")
    policies = POLICIES if fixed_method == "head_aware" else PAIR_POLICIES
    short = "aware" if fixed_method == "head_aware" else fixed_method
    per_input = []
    for sample in samples:
        values = {}
        for policy in policies:
            values[policy] = {
                precision: {
                    field: float(np.mean([point[policy][precision][field] for point in sample["checkpoints"]]))
                    for field in fields
                }
                for precision in ("native_fp64", "projection_fp32")
            }
        per_input.append({"sample_id": sample["sample_id"], "condition": sample["condition"], "values": values})
    conditions = [row["condition"] for row in per_input]
    overall = {}
    for side in ("key", "value"):
        overall[side] = {}
        for precision in ("native_fp64", "projection_fp32"):
            pairs = [
                (row["values"][f"{side}_uniform6"][precision], row["values"][f"{side}_{fixed_method}"][precision])
                for row in per_input
            ]
            differences = np.array([u["prediction_kl"] - a["prediction_kl"] for u, a in pairs])

            # A condition with more rows does not receive more weight.
            def mean(values):
                return float(
                    np.mean(
                        [
                            np.mean([v for v, c in zip(values, conditions) if c == condition])
                            for condition in dict.fromkeys(conditions)
                        ]
                    )
                )

            overall[side][precision] = {
                "uniform6_mean_kl": mean([u["prediction_kl"] for u, a in pairs]),
                f"{fixed_method}_mean_kl": mean([a["prediction_kl"] for u, a in pairs]),
                f"uniform_minus_{short}_kl": mean(differences),
                f"uniform_minus_{short}_delta_nll": mean([u["delta_nll"] - a["delta_nll"] for u, a in pairs]),
                "uniform6_top1_match_fraction": mean([u["top1_match"] for u, a in pairs]),
                f"{fixed_method}_top1_match_fraction": mean([a["top1_match"] for u, a in pairs]),
                f"inputs_with_lower_{fixed_method}_kl": int((differences > 0).sum()),
                "inputs_with_equal_kl": int((differences == 0).sum()),
                f"worst_{short}_minus_uniform_input_kl": float(-differences.min()),
                f"shifted_minus_{short}_mean_kl": mean(
                    [
                        row["values"][f"{side}_shifted"][precision]["prediction_kl"] - a["prediction_kl"]
                        for row, (u, a) in zip(per_input, pairs)
                    ]
                ),
            }
    return {
        "per_input": per_input,
        "condition_equal_means": overall,
        "input_count": len(samples),
        "scope": "paired fixed-trajectory checkpoint evidence; no continuous-compression/task-quality/system claim",
    }


def summarize_pair_budget_samples(samples):
    """Frozen second gate; numerical background and primary/late scopes stay separate."""
    if len({s["condition"] for s in samples}) != len(samples):
        raise ValueError("pair gate expects one frozen Calib input per condition")
    report = summarize_budget_samples(samples, fixed_method="fixed")
    thresholds = COMBINATION_GATE_THRESHOLDS
    background = {}
    for precision in ("native_fp64", "projection_fp32"):
        eta = max(point["dense_after"][precision]["prediction_kl"] for s in samples for point in s["checkpoints"])
        background[precision] = {
            "eta": eta,
            "absolute_threshold": max(thresholds["absolute_kl_floor"], thresholds["background_multiplier"] * eta),
        }
    decisions = {}
    for side in ("key", "value"):
        means = report["condition_equal_means"][side]
        f = means["projection_fp32"]["fixed_mean_kl"]
        u = means["projection_fp32"]["uniform6_mean_kl"]
        s = f + means["projection_fp32"]["shifted_minus_fixed_mean_kl"]
        eps = background["projection_fp32"]["absolute_threshold"]
        gain_u = 1 - f / u if u > eps else None
        gain_s = 1 - f / s if s > eps else None
        wins, bad = 0, []
        for row in report["per_input"]:
            values = row["values"]
            ui = values[f"{side}_uniform6"]["projection_fp32"]["prediction_kl"]
            fi = values[f"{side}_fixed"]["projection_fp32"]["prediction_kl"]
            wins += ui - fi > eps
            if fi - ui > max(thresholds["maximum_condition_regression"] * ui, eps):
                bad.append(row["sample_id"])
        native = means["native_fp64"]
        native_eps = background["native_fp64"]["absolute_threshold"]
        native_consistent = (
            native["fixed_mean_kl"] - native["uniform6_mean_kl"] <= native_eps
            and -native["shifted_minus_fixed_mean_kl"] <= native_eps
        )
        new_top1 = []
        for sample in samples:
            for step, point in zip(sample["steps"], sample["checkpoints"]):
                for precision in ("native_fp64", "projection_fp32"):
                    if (
                        point[f"{side}_uniform6"][precision]["top1_match"]
                        and not point[f"{side}_fixed"][precision]["top1_match"]
                    ):
                        new_top1.append({"sample_id": sample["sample_id"], "decode_step": step, "precision": precision})
        checks = {
            "gain_vs_uniform": gain_u is not None and gain_u >= thresholds["minimum_improvement"] and u - f > eps,
            "gain_vs_shifted": gain_s is not None and gain_s >= thresholds["minimum_improvement"] and s - f > eps,
            "improved_input_count": wins >= thresholds["minimum_improved_inputs"],
            "no_large_condition_regression": not bad,
            "native_direction_consistent": bool(native_consistent),
            "no_new_top1_mismatch": not new_top1,
        }
        decisions[side] = {
            "fixed_improvement_vs_uniform": gain_u,
            "fixed_improvement_vs_shifted": gain_s,
            "improved_input_count": int(wins),
            "inputs_over_regression_limit": bad,
            "new_top1_mismatches": new_top1,
            "checks": checks,
            "gate_decision": "eligible_for_validation_review" if all(checks.values()) else "stop_combination_gate",
        }
    primary = [{**s, "checkpoints": s["checkpoints"][:1]} for s in samples]
    late = [{**s, "checkpoints": s["checkpoints"][1:]} for s in samples if len(s["checkpoints"]) > 1]
    report.update(
        thresholds=thresholds,
        dense_background=background,
        gate_decisions=decisions,
        primary_only=summarize_budget_samples(primary, fixed_method="fixed"),
        late_only=summarize_budget_samples(late, fixed_method="fixed") if late else None,
        validation_started=False,
    )
    return report
