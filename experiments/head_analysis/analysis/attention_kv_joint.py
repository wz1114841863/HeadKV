"""D1/D2: published V2 representation features versus aligned B1 Page RCO."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.head_analysis.analysis.b2_relationships import correlate_profiles
from experiments.head_analysis.analysis.head_identity import decompose_profiles, reference_bootstrap
from experiments.head_analysis.analysis.kv_representation import _json_value, describe
from experiments.head_analysis.common.io import read_json, write_json, write_npz
from experiments.head_analysis.common.statistics import average_ranks, finite_mean, pearson, percentile_ci


RCO_ID = "page_rco:prompt_only:k32:d32"
PRIMARY_FEATURES = ("full_k_post_middle_effective_rank", "full_k_post_middle_E16",
                    "full_v_middle_effective_rank", "full_v_middle_E16", "cka_v_head_observed")
D2_METRICS = ("fixed_profile_spearman", "pooled_response_spearman", "mean_input_response_spearman")


def _spearman(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    return pearson(average_ranks(np.where(valid, x, np.nan)), average_ranks(np.where(valid, y, np.nan)))[0]


def load_inputs(bundle, reference_bundle, b2_bundle):
    """Read the old FP64 relation features and verify every saved RCO/count row."""
    b2_bundle = Path(b2_bundle)
    index = read_json(b2_bundle / "sample_index.json")["samples"]
    lookup = {row["sample_id"]: row for row in index}
    with np.load(b2_bundle / "stages/stage4/stage4_joint_profiles.npz", allow_pickle=False) as a:
        b1_ids = a["sample_ids"].tolist()
        j = a["y_ids"].tolist().index(RCO_ID)
        b1_rco = a["sample_y"][:, j].astype(np.float64)
        b1_count = a["sample_y_source_count_max"][:, j]
        if not np.array_equal(a["sample_y_source_count_min"][:, j], b1_count):
            raise ValueError("Page RCO must use physical KV Heads without Query-Head aggregation")
        if not np.array_equal(a["sample_y_valid_source_heads"][:, j] > 0, b1_count > 0):
            raise ValueError("B1 RCO validity and Step pair counts differ")
    if len(set(b1_ids)) != len(b1_ids) or set(b1_ids) != set(lookup):
        raise ValueError("B1 sample coordinates differ from the sample index")
    ids, samples, values, rco, counts, reference_flags = [], [], [], [], [], []
    layers = heads = names = definitions = None
    for directory, is_reference in ((Path(reference_bundle), True), (Path(bundle), False)):
        source = read_json(directory / "summary.json")
        if source["status"] != "completed" or source["sample_ids"] != [r["sample_id"] for r in source["samples"]]:
            raise ValueError("incomplete or misaligned V2 package")
        if definitions is None:
            definitions = source["definitions"]
        elif definitions != source["definitions"]:
            raise ValueError("Pilot and extension metric definitions differ")
        for row in source["samples"]:
            sid = row["sample_id"]
            record = lookup[sid]
            if any(row[k] != record[k] for k in ("dataset", "pair_id", "prompt_length")):
                raise ValueError("V2 sample metadata differs from the B1/B2 input index")
            with np.load(directory / row["numeric_file"], allow_pickle=False) as a:
                if layers is None:
                    layers, heads = a["layer_ids"], a["kv_head_ids"]
                    names = a["relation_feature_names"].tolist()
                if not (np.array_equal(layers, a["layer_ids"]) and np.array_equal(heads, a["kv_head_ids"])
                        and names == a["relation_feature_names"].tolist()):
                    raise ValueError("V2 Layer/KV Head/feature coordinates differ")
                feature = a["relation_feature_values"].astype(np.float64)
                norm = np.stack([a[f"full_{rep}_middle_norm_summary"][..., 0] for rep in ("k_post", "v")])
                j = b1_ids.index(sid)
                if not (np.array_equal(a["b1_page_rco_raw_mean"], b1_rco[j], equal_nan=True)
                        and np.array_equal(a["b1_page_rco_valid_pair_count"], b1_count[j])):
                    raise ValueError(f"V2 and B1 RCO/count values differ: {sid}")
            ids.append(sid); samples.append(record); values.append(np.concatenate((feature, norm)))
            rco.append(b1_rco[j]); counts.append(b1_count[j]); reference_flags.append(is_reference)
    values, rco, counts = np.stack(values, 1), np.stack(rco), np.stack(counts)
    if len(set(ids)) != len(ids) or set(ids) != set(b1_ids):
        raise ValueError("V2 Pilot plus extension must cover the B1 sample set exactly once")
    if not (np.array_equal(layers, np.arange(rco.shape[1])) and np.array_equal(heads, np.arange(rco.shape[2]))
            and values.shape[2:] == rco.shape[1:]):
        raise ValueError("B1 canonical Layer/KV Head axes and V2 coordinates differ")
    if not np.array_equal(np.isfinite(rco), counts > 0):
        raise ValueError("RCO finite support and Step pair counts differ")
    datasets = np.asarray([r["dataset"] for r in samples])
    conditions = np.asarray([f"{r['dataset']}_{r['target_prompt_length']}" for r in samples])
    groups = np.asarray([f"pg19:{r['pair_id']}" if r["dataset"] == "pg19"
                         else f"{r['dataset']}:{r['sample_id']}" for r in samples])
    reference_flags = np.asarray(reference_flags)
    if set(groups[reference_flags]) & set(groups[~reference_flags]):
        raise ValueError("Pilot and extension source groups overlap")
    return {"sample_ids": np.asarray(ids), "feature_names": np.asarray(names + ["full_k_post_middle_mean_norm", "full_v_middle_mean_norm"]),
        "values": values, "rco": rco, "rco_valid_pair_count": counts, "layer_ids": layers, "kv_head_ids": heads,
        "datasets": datasets, "conditions": conditions, "source_groups": groups, "is_reference": reference_flags}


def decompose_condition(x, y):
    """Paired common coordinates; separate fixed Head contrasts from input response."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x).all(1) & np.isfinite(y).all(0)[None]
    valid &= (valid.sum(-1) >= 3)[..., None]
    numeric = {}
    for key, values in (("x", x), ("y", np.broadcast_to(y, x.shape))):
        values = np.where(valid[:, None], values, np.nan)
        rows = [decompose_profiles(feature) for feature in values]
        for field in ("head_effect", "residual", "per_head_input_std", "per_head_relative_std", "relative_fixed_share",
                      "ss_total", "ss_head", "ss_input", "ss_interaction", "per_layer_relative_fixed_share"):
            numeric[f"{key}_{field}"] = np.stack([row[field] for row in rows])
        numeric[f"{key}_values"] = values
        numeric[f"{key}_head_effect"], numeric[f"{key}_residual"] = _relative_parts(values)
    rx, ry = numeric["x_residual"], numeric["y_residual"]
    numeric["valid_mask"] = valid
    numeric["per_layer_fixed_rho"] = _spearman(numeric["x_head_effect"], numeric["y_head_effect"])
    numeric["per_input_response_rho"] = _spearman(rx.reshape(rx.shape[:2] + (-1,)), ry.reshape(ry.shape[:2] + (-1,)))
    numeric["per_layer_response_rho"] = _spearman(rx, ry)
    numeric["within_head_response_rho"] = _spearman(rx.transpose(0, 2, 3, 1), ry.transpose(0, 2, 3, 1))
    numeric["metrics"] = _response_metrics(numeric["x_head_effect"], numeric["y_head_effect"], rx, ry)
    return numeric


def _relative_parts(values):
    """Stable double centering on [...,Input,Layer,Head], shared by point/Bootstrap.

    Subtract a valid reference Head, then the first input, before reductions;
    these offsets cancel algebraically and preserve exact zero responses.
    """
    first = np.argmax(np.isfinite(values), axis=-1)[..., None]
    relative = values - np.take_along_axis(values, first, axis=-1)
    anchor = relative[..., 0, :, :]
    change = relative - anchor[..., None, :, :]
    change = change - finite_mean(change, -1)[..., None]
    mean_change = finite_mean(change, -3)
    fixed = anchor - finite_mean(anchor, -1)[..., None] + mean_change
    return fixed, change - mean_change[..., None, :, :]


def _response_metrics(hx, hy, rx, ry):
    fixed = _spearman(hx.reshape(hx.shape[:-2] + (-1,)), hy.reshape(hy.shape[:-2] + (-1,)))
    pooled = _spearman(rx.reshape(rx.shape[:-3] + (-1,)), ry.reshape(ry.shape[:-3] + (-1,)))
    per_input = _spearman(rx.reshape(rx.shape[:-2] + (-1,)), ry.reshape(ry.shape[:-2] + (-1,)))
    return np.stack((fixed, pooled, finite_mean(per_input, -1)), -1)


def response_bootstrap(decomposition, draws, batch_size=16):
    """Refit condition Head means on each source resample, including PG19 multiplicity."""
    x, y = decomposition["x_values"], decomposition["y_values"]
    output = []
    for start in range(0, len(draws), batch_size):
        selected = draws[start:start + batch_size]
        hx, rx = _relative_parts(x[:, selected].transpose(1, 0, 2, 3, 4))
        hy, ry = _relative_parts(y[:, selected].transpose(1, 0, 2, 3, 4))
        output.append(_response_metrics(hx, hy, rx, ry))
    return np.concatenate(output)


def run_analysis(bundle, reference_bundle, output_dir, *, b2_bundle, bootstrap_replicates=2000, seed=20261112):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    data = load_inputs(bundle, reference_bundle, b2_bundle)
    x, y, datasets, conditions, groups, ref = (data[k] for k in
        ("values", "rco", "datasets", "conditions", "source_groups", "is_reference"))
    eligible = datasets != "ruler"
    primary = np.isin(data["feature_names"], PRIMARY_FEATURES)
    # Keep all 100 rows and coordinates, but never substitute a short RULER interval.
    y = np.where(eligible[:, None, None], y, np.nan)
    relation = correlate_profiles(x, y[None])
    rho = relation["values"][..., 1, 0]
    pilot_sign = np.full_like(rho, np.nan)
    for label in sorted(set(conditions[eligible])):
        pilot = ref & (conditions == label)
        if pilot.sum() != 1:
            raise ValueError("each eligible condition requires one fixed Pilot reference")
        pilot_sign[:, conditions == label] = np.sign(rho[:, pilot])
    agreement = np.where(np.isfinite(rho) & np.isfinite(pilot_sign) & (pilot_sign != 0), np.sign(rho) == pilot_sign, np.nan)
    positive = np.where(np.isfinite(rho), rho > 0, np.nan)
    negative = np.where(np.isfinite(rho), rho < 0, np.nan)
    scores = np.stack((rho, positive, negative, agreement), -1)
    ext = eligible & ~ref
    bootstrap = reference_bootstrap(scores[:, ext].transpose(1, 0, 2), conditions[ext], datasets[ext], groups[ext],
                                    replicates=bootstrap_replicates, seed=seed)
    result = {"schema_version": "headkv-v2-attention-kv-joint-d-v1", "status": "running",
        "source_bundle": str(Path(bundle).resolve()), "reference_bundle": str(Path(reference_bundle).resolve()),
        "b2_bundle": str(Path(b2_bundle).resolve()), "sample_ids": data["sample_ids"].tolist(),
        "feature_names": data["feature_names"].tolist(), "primary_features": list(PRIMARY_FEATURES),
        "eligible_sample_count": int(eligible.sum()), "extension_eligible_sample_count": int(ext.sum()),
        "extension_source_count": len(set(groups[ext])), "seed": seed, "bootstrap_replicates": bootstrap_replicates,
        "definitions": {
            "alignment": "100 V2 IDs joined to existing B2 copy of B1 Page RCO; every RCO/count row cross-checked against V2; canonical physical Layer/KV Head axes",
            "rco": RCO_ID + ", rco_raw_mean over original full B1 Decode; actual valid Step pair counts retained",
            "features": "14 published FP64 relation features unchanged, plus two existing full-middle Norm features (published FP32)",
            "primary": "full-middle K_post/V rank and E16, V mean-neighbor Observed CKA; others supplementary",
            "D1": "pairwise finite support; center both sides by each input Layer's Head mean, then Spearman across Layer/Head coordinates; retain per-layer results",
            "direction": "positive/negative input fractions and agreement with the condition's single fixed Pilot sign; undefined reference sign stays missing",
            "D2": "within each task/length condition, paired common-finite coordinates: X=input-layer mean + fixed Head contrast + input×Head response; same for RCO",
            "fixed_profile": "Spearman between the condition mean centered X and RCO profiles; descriptive, not a predictor",
            "response": "correlate residuals after removing input-layer means and condition Head means; pooled and mean per-input Spearman, per-layer and within-Head input correlations saved",
            "D2_bootstrap": "source resampling within condition; re-estimate Head means for every draw; PG19 lengths share source draws; common coordinates fixed before resampling",
            "shares": "existing descriptive sums of squares, not population variance components; relative fixed share excludes input-layer shifts",
            "weighting": "condition-equal source means; 56 extension rows/47 sources primary, all 60 long rows supplementary; RULER excluded from long-interval relation",
            "scope": "post-hoc association on previously studied tasks/trajectories, not causal or predictive evidence or compression benefit",
        }, "D1": {}, "D2": []}
    output_dir.mkdir(parents=True)
    write_json(output_dir / "summary.json", result)
    write_npz(output_dir / "profiles.npz", {**data, "eligible": eligible, "primary": primary})
    write_npz(output_dir / "d1_relations.npz", {"feature_names": data["feature_names"], "sample_ids": data["sample_ids"],
        "rho": rho, "per_layer_rho": relation["layer_values"][..., 0],
        "valid_head_count": relation["valid_heads"][..., 1], "per_layer_valid_head_count": relation["layer_valid_heads"],
        "pilot_direction": pilot_sign, "pilot_direction_agreement": agreement})
    write_npz(output_dir / "d1_bootstrap.npz", {"feature_names": data["feature_names"],
        "metric_names": np.asarray(["mean_rho", "positive_fraction", "negative_fraction", "pilot_direction_agreement"]), **bootstrap})
    result["D1"]["bootstrap"] = {key: value for key, value in bootstrap.items()
        if key.startswith(("condition_", "equal_condition_")) and "bootstrap_means" not in key}
    result["D1"]["cohorts"] = {name: {str(feature): describe(rho[f, mask]) for f, feature in enumerate(data["feature_names"])}
        for name, mask in (("pilot", eligible & ref), ("extension", ext), ("all_long", eligible))}
    d2_draws, d2_points = [], []
    for population, include in (("extension", ext), ("all", eligible)):
        for label in bootstrap["condition_names"]:
            mask = include & (conditions == label)
            indices = np.flatnonzero(mask)
            indices = indices[np.argsort(groups[indices])]
            if len(set(groups[indices])) != len(indices):
                raise ValueError("D2 expects one input per source within each length condition")
            decomposition = decompose_condition(x[:, indices], y[indices])
            row = {"population": population, "condition": str(label), "sample_count": len(indices),
                "sample_ids": data["sample_ids"][indices].tolist(), "metric_names": list(D2_METRICS),
                "metrics": decomposition["metrics"], "x_relative_fixed_share": decomposition["x_relative_fixed_share"],
                "y_relative_fixed_share": decomposition["y_relative_fixed_share"],
                "x_input_shift_share": np.divide(decomposition["x_ss_input"], decomposition["x_ss_total"],
                    out=np.full(len(primary), np.nan), where=decomposition["x_ss_total"] > 0),
                "within_head_response_rho": [describe(v) for v in decomposition["within_head_response_rho"]]}
            numeric = {"feature_names": data["feature_names"], "sample_indices": indices,
                "sample_ids": data["sample_ids"][indices], "layer_ids": data["layer_ids"], "kv_head_ids": data["kv_head_ids"],
                "metric_names": np.asarray(D2_METRICS),
                **{key: value for key, value in decomposition.items() if key not in ("x_values", "y_values")}}
            if population == "extension":
                stratum = "pg19" if label.startswith("pg19_") else str(label)
                if not np.array_equal(groups[indices], bootstrap[f"group_ids_{stratum}"]):
                    raise ValueError("D2 source ordering differs from the Bootstrap group coordinates")
                boot = response_bootstrap(decomposition, bootstrap[f"draws_{stratum}"])
                ci, valid = percentile_ci(boot)
                numeric.update(bootstrap_metrics=boot, ci95=ci, valid_replicates=valid)
                row.update(ci95=ci, valid_replicates=valid)
                d2_draws.append(boot); d2_points.append(decomposition["metrics"])
            filename = f"d2_{population}_{label}.npz"
            write_npz(output_dir / filename, numeric)
            row["numeric_file"] = filename
            result["D2"].append(row)
            print(f"V2 D2 completed: {population}/{label}", flush=True)
    equal_draws = finite_mean(np.stack(d2_draws, 1), 1)
    equal_ci, equal_valid = percentile_ci(equal_draws)
    result["D2_equal_conditions"] = {"metric_names": list(D2_METRICS), "mean": finite_mean(np.stack(d2_points), 0),
        "ci95": equal_ci, "valid_replicates": equal_valid}
    write_npz(output_dir / "d2_equal_conditions.npz", {"feature_names": data["feature_names"],
        "metric_names": np.asarray(D2_METRICS), "condition_names": bootstrap["condition_names"],
        "condition_metrics": np.stack(d2_points), "equal_condition_bootstrap": equal_draws,
        "equal_condition_mean": result["D2_equal_conditions"]["mean"], "ci95": equal_ci, "valid_replicates": equal_valid})
    result = _json_value(result)
    result["status"] = "completed"
    write_json(output_dir / "summary.json", result, overwrite=True)
    return result
