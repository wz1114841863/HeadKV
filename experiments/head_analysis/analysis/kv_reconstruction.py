"""V3-A preparation, Calib/Validation decisions, then a separately opened Test."""
from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np

from experiments.head_analysis.analysis.kv_representation import _json_value, describe
from experiments.head_analysis.analysis.pca_reconstruction import (
    RANKS, apply_position_rope, condition_weights, fit_basis, fit_calibration,
    oracle_errors, projection_errors, rope_coefficients, sample_encoding_errors,
)
from experiments.head_analysis.analysis.kv_representation_budget import allocate_budget, byte_ledger, combine_ledgers
from experiments.head_analysis.common.io import read_json, write_json, write_npz
from experiments.head_analysis.common.statistics import bootstrap_draws, bootstrap_means, finite_mean, percentile_ci
from experiments.head_analysis.storage.representation_inputs import discover_samples, freeze_split, preflight, read_sample


REPRESENTATIONS = ("k_post", "v")
SHARINGS = ("head", "layer", "global")
METHODS = ("oracle_same_mean", "oracle_classic", *SHARINGS,
           *(f"head_aware_{kind}_{r}" for r in (32, 64, 96) for kind in ("equal", "strict")))
METRICS = ("normalized_mse", "head_error_p90", "relative_frobenius", "frobenius", "raw_mse")


def _save(path, value):
    write_json(path, _json_value(value))


def prepare(run_dirs, fixed_inputs, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    records = discover_samples(run_dirs, fixed_inputs)
    output.mkdir(parents=True)
    split = freeze_split(records, output / "split.json")
    # Metadata/shape/finite checks may read all sets, but compute no reconstruction error.
    check = preflight(split["samples"])
    _save(output / "preflight.json", check)
    model = Path(records[0]["model"])
    _save(output / "rope_config.json", read_json(model / "config.json"))
    _save(output / "protocol.json", {"status": "frozen_before_errors", "seed": 20261113,
        "ranks": RANKS, "primary_rank": 64, "representations": REPRESENTATIONS,
        "test_methods": METHODS, "scope": "full middle [64,T-64); BF16 sample checks on same 448 positions",
        "precision": "FP64 ideal; FP32 bases/means, BF16 coefficients; CPU only",
        "A1_mean_limit": 1.10, "A1_condition_limit": 1.25, "A2_mean_limit": 1.10,
        "A2_condition_limit": 1.25, "A3_minimum_improvement": .05,
        "A3_maximum_condition_degradation": .05, "A3_maximum_tail_degradation": .05,
        "bootstrap_replicates": 2000, "inference_path_modified": False})
    return check


def _load_moments(records):
    means, covariances = [], []
    for record in records:
        data = read_sample(record)
        means.append(data["mean"]); covariances.append(data["covariance"])
    return np.stack(means), np.stack(covariances)


def _load_models(output):
    with np.load(Path(output) / "calibration.npz", allow_pickle=False) as a:
        return {key: a[key] for key in a.files}


def _basis(models, sharing):
    shape = models["head_basis"].shape
    if sharing == "head":
        return models["head_basis"]
    if sharing == "layer":
        return np.broadcast_to(models["layer_basis"][..., :, None, :, :], shape)
    return np.broadcast_to(models["global_basis"][..., None, None, :, :], shape)


def fit_models(records, output):
    means, covs = _load_moments(records)
    labels = [r["condition"] for r in records]
    fit = fit_calibration(means, covs, labels, [r["source_group"] for r in records])
    models = {"mean": fit["mean"], "scale": fit["scale"], "G": fit["G"],
              "calibration_weights": fit["weights"], "calibration_sample_ids": np.asarray([r["sample_id"] for r in records])}
    for sharing in SHARINGS:
        basis = fit_basis(fit["G"], fit["scale"], sharing)
        if np.max(basis["orthogonality_error"]) > 1e-10:
            raise ValueError("basis orthogonality check failed")
        models[f"{sharing}_basis"] = basis["compact_basis"]
    model_basis = models["head_basis"]
    error = projection_errors(fit["mean"], fit["G"], fit["mean"], model_basis)
    costs = error / fit["scale"][..., None]
    for rep in range(2):
        for rank in (32, 64, 96):
            allocation = allocate_budget(costs[rep], rank)
            for kind, field in (("equal", "equal_coefficient_ranks"), ("strict", "strict_ranks")):
                models[f"ranks_{rep}_{kind}_{rank}"] = allocation[field]
            _save(Path(output) / f"allocation_{REPRESENTATIONS[rep]}_{rank}.json", allocation)
    write_npz(Path(output) / "calibration.npz", models)
    # Release Calib-only full moments before raw-vector RoPE calibration.
    del means, covs
    return models


def _gather(curve, rank_matrix):
    indices = np.broadcast_to(np.searchsorted(RANKS, rank_matrix), curve.shape[:-1])
    return np.take_along_axis(curve, indices[..., None], -1)[..., 0]


def _summary_scores(error, energy, scale, lengths):
    normalized = np.divide(error, scale[..., None], out=np.full_like(error, np.nan), where=scale[..., None] > 0)
    relative = np.sqrt(np.divide(error, energy[..., None], out=np.full_like(error, np.nan), where=energy[..., None] > 0))
    # [Method,Sample,Representation,Layer,Head,Rank] -> [Sample,Method,Representation,Rank,Metric]
    flat = normalized.transpose(0, 1, 2, 5, 3, 4).reshape(-1, normalized.shape[3] * normalized.shape[4])
    usable = np.isfinite(flat).any(-1)
    p90 = np.full(len(flat), np.nan)
    p90[usable] = np.nanquantile(flat[usable], .9, axis=-1)
    p90 = p90.reshape(normalized.shape[:3] + (normalized.shape[-1],))
    score = np.stack((finite_mean(normalized, (3, 4)), p90,
        finite_mean(relative, (3, 4)), finite_mean(np.sqrt(error * lengths[None, :, None, None, None, None]), (3, 4)),
        finite_mean(error, (3, 4))), -1)
    return score.transpose(1, 0, 2, 3, 4)


def summarize(records, ideal, actual, energy, models, sample_energy=None):
    conditions = np.asarray([r["condition"] for r in records])
    lengths = np.asarray([r["prompt_length"] - 128 for r in records])
    scores = _summary_scores(ideal, energy, models["scale"], lengths)
    actual_scores = _summary_scores(actual, energy if sample_energy is None else sample_energy, models["scale"], np.full(len(records), 448))
    labels = sorted(set(conditions))
    condition_scores = np.stack([finite_mean(scores[conditions == label], 0) for label in labels])
    actual_condition_scores = np.stack([finite_mean(actual_scores[conditions == label], 0) for label in labels])
    return {"condition_names": np.asarray(labels), "condition_scores": condition_scores,
        "equal_condition_scores": finite_mean(condition_scores, 0), "input_scores": scores,
        "bf16_condition_scores": actual_condition_scores,
        "bf16_equal_condition_scores": finite_mean(actual_condition_scores, 0), "bf16_input_scores": actual_scores}


def _export_model(output, method, rep, models, ranks, sharing):
    """An actual contiguous model payload; its file size must equal the ledger."""
    basis = _basis(models, sharing)[rep]
    layers, heads, dimension, _ = basis.shape
    ledger = byte_ledger(ranks, [129], strategy="per_head" if sharing == "head" else sharing, dimension=dimension)
    path = Path(output) / "deployed_models" / f"{REPRESENTATIONS[rep]}_{method}.bin"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("xb") as handle:
        layout = 1 | (2 if ledger["rank_encoding"] == "per_head" else 0)
        handle.write(struct.pack("<HHHBB", layers, heads, dimension, SHARINGS.index(sharing), layout))
        encoded_rank = np.asarray(ranks.flat[0] if ledger["rank_encoding"] == "uniform" else ranks, dtype="<u2")
        handle.write(encoded_rank.tobytes())
        handle.write(models["mean"][rep].astype("<f4").tobytes())
        if sharing == "head":
            for l, h in np.ndindex(ranks.shape):
                handle.write(basis[l, h, :, :int(ranks[l, h])].astype("<f4").tobytes())
        elif sharing == "layer":
            for l in range(layers):
                handle.write(basis[l, 0, :, :int(ranks[l].max())].astype("<f4").tobytes())
        else:
            handle.write(basis[0, 0, :, :int(ranks.max())].astype("<f4").tobytes())
    if path.stat().st_size != ledger["model_total_bytes"]:
        raise ValueError("serialized deployment model differs from the byte ledger")


def evaluate(records, output, models, phase):
    """Evaluate one authorized split; no fitting or method selection occurs here."""
    means, covs = _load_moments(records)
    energy = np.trace(covs, axis1=-2, axis2=-1) + np.square(means).sum(-1)
    shape = (len(METHODS), len(records), *models["scale"].shape, len(RANKS))
    ideal = np.full(shape, np.nan); actual = np.full(shape, np.nan); sampled_ideal = np.full(shape, np.nan)
    oracle = oracle_errors(means, covs, models["mean"])
    ideal[0], ideal[1] = oracle["same_mean"], oracle["classic"]
    for sharing in SHARINGS:
        ideal[METHODS.index(sharing)] = projection_errors(means, covs, models["mean"], _basis(models, sharing))
    full = ideal[METHODS.index("head")]
    for mi, method in enumerate(METHODS[5:], 5):
        _, _, kind, r = method.split("_")
        for rep in range(2):
            ideal[mi, :, rep, ..., list(RANKS).index(int(r))] = _gather(full[:, rep], models[f"ranks_{rep}_{kind}_{r}"])
    # Check the entire ideal curves before computing BF16 sample errors.
    scale = models["scale"][None]
    if np.nanmax(ideal[:5, ..., -1] / scale) > 1e-10 or np.nanmax(np.diff(ideal[:5], axis=-1) / scale[..., None]) > 1e-10:
        raise ValueError("full-rank or nested-rank correctness check failed")
    if np.any(ideal[1] > ideal[0] + 1e-10 * scale[..., None]) or np.any(ideal[0][None] > ideal[2:5] + 1e-10 * scale[..., None]):
        raise ValueError("Oracle ordering check failed")
    del covs, means
    sample_energy = []
    for si, record in enumerate(records):
        raw = read_sample(record, include_vectors=True)
        vectors = raw["vectors"][[1, 2]].astype(np.float64)
        sample_energy.append(np.square(vectors).sum(-1).mean(-1))
        for sharing in SHARINGS:
            mi = METHODS.index(sharing)
            result = sample_encoding_errors(vectors, models["mean"], _basis(models, sharing))
            actual[mi, si] = result["bf16_mse"]; sampled_ideal[mi, si] = result["ideal_mse"]
            expected = 2 * 448 * np.prod(models["scale"].shape) * np.asarray(RANKS, np.uint64)
            if not np.array_equal(result["coefficient_bytes_total"], expected):
                raise ValueError("actual BF16 coefficient arrays differ from the payload budget")
            # Exact-scope formula validation on this sample's own FP64 moments.
            mu = vectors.mean(-2); centered = vectors - mu[..., None, :]
            cov = centered.swapaxes(-1, -2) @ centered / vectors.shape[-2]
            analytic = projection_errors(mu, cov, models["mean"], _basis(models, sharing))
            if np.max(np.abs(analytic - result["ideal_mse"]) / np.maximum(models["scale"][..., None], 1e-300)) > 1e-10:
                raise ValueError("direct sampled reconstruction and FP64 moment formula differ")
        for mi, method in enumerate(METHODS[5:], 5):
            _, _, kind, r = method.split("_"); ri = list(RANKS).index(int(r))
            for rep in range(2):
                ranks = models[f"ranks_{rep}_{kind}_{r}"]
                actual[mi, si, rep, ..., ri] = _gather(actual[2, si, rep], ranks)
                sampled_ideal[mi, si, rep, ..., ri] = _gather(sampled_ideal[2, si, rep], ranks)
        print(f"V3 {phase} sample {si+1}/{len(records)}: {record['sample_id']}", flush=True)
    sample_energy = np.stack(sample_energy)
    summary = summarize(records, ideal, actual, energy, models, sample_energy)
    sample_scores = _summary_scores(sampled_ideal, sample_energy, models["scale"], np.full(len(records), 448))
    conds = np.asarray([r["condition"] for r in records])
    sample_conditions = np.stack([finite_mean(sample_scores[conds == label], 0) for label in summary["condition_names"]])
    summary.update(sample_ideal_input_scores=sample_scores, sample_ideal_condition_scores=sample_conditions,
                   sample_ideal_equal_condition_scores=finite_mean(sample_conditions, 0))
    write_npz(Path(output) / f"{phase}_errors.npz", {"sample_ids": np.asarray([r["sample_id"] for r in records]),
        "representations": np.asarray(REPRESENTATIONS), "methods": np.asarray(METHODS), "ranks": RANKS,
        "metric_names": np.asarray(METRICS), "middle_count": np.asarray([r["prompt_length"]-128 for r in records]),
        "full_middle_energy": energy, "sample_energy": sample_energy, "ideal_mse": ideal,
        "sample_ideal_mse": sampled_ideal, "bf16_mse": actual, **summary})
    return summary, ideal, sampled_ideal, actual


def _relative_change(candidate, reference):
    return None if not np.isfinite(reference) or reference <= 1e-10 else float(candidate / reference - 1)


def _changes(candidate, reference):
    candidate, reference = np.broadcast_arrays(candidate, reference)
    result = np.full(reference.shape, np.nan, dtype=np.float64)
    np.divide(candidate, reference, out=result, where=np.isfinite(reference) & (reference > 1e-10))
    return result - 1


def _acceptable(candidate, reference, increase):
    change = _changes(candidate, reference)
    equivalent = (np.abs(reference) <= 1e-10) & (np.abs(candidate - reference) <= 1e-10)
    return (change <= increase) | equivalent


def decide(summary, models, *, select=True):
    """Validation-only choices at the predeclared primary budget; no Test argument."""
    ri = list(RANKS).index(64)
    eq, cond = summary["equal_condition_scores"], summary["condition_scores"]
    decisions = {}
    for rep, name in enumerate(REPRESENTATIONS):
        ph = eq[2, rep, ri, 0]; oracle = eq[0, rep, ri, 0]
        a1_ratio = _relative_change(ph, oracle)
        a1_condition = _changes(cond[:, 2, rep, ri, 0], cond[:, 0, rep, ri, 0])
        shared = {}
        for sharing in ("layer", "global"):
            mi = METHODS.index(sharing)
            ratio = _relative_change(eq[mi, rep, ri, 0], ph)
            c_ratios = _changes(cond[:, mi, rep, ri, 0], cond[:, 2, rep, ri, 0])
            actual = summary["bf16_equal_condition_scores"]
            sample = summary.get("sample_ideal_equal_condition_scores", actual)
            ac = summary["bf16_condition_scores"]
            actual_pass = bool(_acceptable(actual[mi, rep, ri, 0], actual[2, rep, ri, 0], .10)) and bool(np.all(
                _acceptable(ac[:, mi, rep, ri, 0], ac[:, 2, rep, ri, 0], .25)))
            shared[sharing] = {"mean_increase": ratio, "condition_increases": c_ratios,
                "passed": bool(_acceptable(eq[mi, rep, ri, 0], ph, .10)) and bool(np.all(
                    _acceptable(cond[:, mi, rep, ri, 0], cond[:, 2, rep, ri, 0], .25))),
                "sample_ideal_mean_increase": _relative_change(sample[mi, rep, ri, 0], sample[2, rep, ri, 0]),
                "bf16_sample_mean_increase": _relative_change(actual[mi, rep, ri, 0], actual[2, rep, ri, 0]),
                "bf16_sample_threshold_passed": actual_pass}
        selected = next((sharing for sharing in ("global", "layer") if shared[sharing]["passed"]), None) if select else None
        selected_actual = next((sharing for sharing in ("global", "layer")
            if shared[sharing]["passed"] and shared[sharing]["bf16_sample_threshold_passed"]), None) if select else None
        decisions[name] = {"A1": {"mean_increase_over_same_mean_oracle": a1_ratio,
            "condition_increases": a1_condition,
            "passed": bool(_acceptable(ph, oracle, .10)) and bool(np.all(
                _acceptable(cond[:, 2, rep, ri, 0], cond[:, 0, rep, ri, 0], .25)))},
            "A2": {"candidates": shared, "selected_sharing": selected,
                   "selected_bf16_sample_sharing": selected_actual}, "A3": {}}
    return decisions


def comparisons(records, summary, output, phase):
    """Source Bootstrap, with both lengths of each PG19 Pair sharing draws."""
    scores = summary["input_scores"]; conds = np.asarray([r["condition"] for r in records])
    groups = np.asarray([r["source_group"] for r in records]); labels = summary["condition_names"]
    samples, point, numeric = [], [], {}
    for condition in labels:
        idx = np.flatnonzero(conds == condition); source_ids = sorted(set(groups[idx]))
        key = "pg19" if condition.startswith("pg19_") else str(condition)
        draw = bootstrap_draws(20261113, f"v3:{key}", 2000, len(source_ids))
        numeric[f"draws_{key}"] = draw; numeric[f"sources_{key}"] = np.asarray(source_ids)
        grouped = np.stack([finite_mean(scores[idx[groups[idx] == group]], 0) for group in source_ids])
        samples.append(bootstrap_means(grouped, draw)); point.append(finite_mean(grouped, 0))
    boots = np.stack(samples, 1); points = np.stack(point); equal = finite_mean(boots, 1)
    records_out = {}
    for rank in (32, 64, 96):
        ri = list(RANKS).index(rank); mi = METHODS.index(f"head_aware_strict_{rank}")
        difference = equal[:, mi, :, ri, 0] - equal[:, 2, :, ri, 0]
        ratio = -_changes(equal[:, mi, :, ri, 0], equal[:, 2, :, ri, 0])
        ci, valid = percentile_ci(ratio)
        eqpoint = finite_mean(points, 0)
        improvement = -_changes(eqpoint[mi, :, ri, 0], eqpoint[2, :, ri, 0])
        condition_degradation = _changes(points[:, mi, :, ri, 0], points[:, 2, :, ri, 0])
        tail_degradation = _changes(eqpoint[mi, :, ri, 1], eqpoint[2, :, ri, 1])
        sample = summary.get("sample_ideal_equal_condition_scores", summary["bf16_equal_condition_scores"])
        bf16 = summary["bf16_equal_condition_scores"]
        sample_improvement = -_changes(sample[mi, :, ri, 0], sample[2, :, ri, 0])
        actual_improvement = -_changes(bf16[mi, :, ri, 0], bf16[2, :, ri, 0])
        sample_preserved = (sample_improvement > 0) & (actual_improvement > 0)
        records_out[str(rank)] = {"improvement": improvement, "ci95": ci, "valid_replicates": valid,
            "condition_degradation": condition_degradation, "tail_degradation": tail_degradation,
            "sample_ideal_improvement": sample_improvement, "bf16_sample_improvement": actual_improvement,
            "sample_advantage_preserved": sample_preserved,
            "passed": (improvement >= .05) & (ci[:, 0] > 0) & np.all(
                _acceptable(points[:, mi, :, ri, 0], points[:, 2, :, ri, 0], .05), 0)
                & _acceptable(eqpoint[mi, :, ri, 1], eqpoint[2, :, ri, 1], .05)}
        numeric[f"improvement_{rank}"] = ratio; numeric[f"difference_{rank}"] = difference
    write_npz(Path(output) / f"{phase}_bootstrap.npz", {**numeric, "condition_names": labels,
        "condition_means": points, "equal_condition_mean": finite_mean(points, 0)})
    return records_out


def budget_report(records, models, output, phase):
    ledgers = {}; lengths = [r["prompt_length"] for r in records]
    for rep, name in enumerate(REPRESENTATIONS):
        ledgers[name] = {}
        for sharing in SHARINGS:
            for rank in (32, 64, 96):
                ranks = np.full(models["scale"].shape[-2:], rank)
                key = f"{sharing}_{rank}"
                ledger = byte_ledger(ranks, lengths, strategy="per_head" if sharing == "head" else sharing)
                ledgers[name][key] = ledger
                if phase == "validation":
                    _export_model(output, key, rep, models, ranks, sharing)
        for rank in (32, 64, 96):
            base = ledgers[name][f"head_{rank}"]
            for kind in ("equal", "strict"):
                ranks = models[f"ranks_{rep}_{kind}_{rank}"]
                key = f"head_aware_{kind}_{rank}"
                ledger = byte_ledger(ranks, lengths)
                if kind == "strict" and (ledger["model_total_bytes"] > base["model_total_bytes"] or
                    np.any(ledger["cache_total_bytes"] > base["cache_total_bytes"])):
                    raise ValueError("A3 exceeded model/request byte budget")
                ledgers[name][key] = ledger
                if phase == "validation":
                    _export_model(output, key, rep, models, ranks, "head")
    total = {key: combine_ledgers(ledgers["k_post"][key], ledgers["v"][key]) for key in ledgers["k_post"]}
    _save(Path(output) / f"{phase}_byte_ledger.json", {"sample_ids": [r["sample_id"] for r in records],
        "model_payloads_are_serialized": True, "full_cache_bytes_are_budget_model": True,
        "cache_encoding": "BF16 middle coefficients; BF16 first/last64; T:uint32; offsets derived",
        "ledger": ledgers, "k_plus_v": total})


def _report_rows(records, models, ideal, sample_ideal, actual):
    result = {}
    weights = condition_weights([r["condition"] for r in records])
    for rep, name in enumerate(REPRESENTATIONS):
        result[name] = {}
        for mi, method in enumerate(METHODS):
            ri = list(RANKS).index(int(method.rsplit("_", 1)[1])) if mi >= 5 else list(RANKS).index(64)
            normalized = ideal[mi, :, rep, ..., ri] / models["scale"][rep]
            valid = np.isfinite(normalized)
            if not valid.any():
                continue
            mean = np.einsum("s,slh->lh", weights, np.where(valid, normalized, 0))
            l, h = np.unravel_index(np.argmax(mean), mean.shape)
            sample_i, sl, sh = np.unravel_index(np.nanargmax(normalized), normalized.shape)
            sample_diff = (actual[mi, :, rep, ..., ri] - sample_ideal[mi, :, rep, ..., ri]) / models["scale"][rep]
            result[name][method] = {"nominal_rank": int(RANKS[ri]), "worst_layer": int(l), "worst_head": int(h),
                "worst_head_mean_normalized_mse": float(mean[l, h]),
                "worst_input_head": {"sample_id": records[sample_i]["sample_id"], "layer": int(sl), "head": int(sh),
                                     "normalized_mse": float(normalized[sample_i, sl, sh])},
                "normalized_head_error": describe(normalized), "bf16_minus_ideal_sample_error": describe(sample_diff)}
            bf16 = actual[mi, :, rep, ..., ri] / models["scale"][rep]
            if np.isfinite(bf16).any():
                mean_bf16 = np.einsum("s,slh->lh", weights, bf16)
                bl, bh = np.unravel_index(np.argmax(mean_bf16), mean_bf16.shape)
                bi, bsl, bsh = np.unravel_index(np.nanargmax(bf16), bf16.shape)
                result[name][method]["bf16_worst_head"] = {"layer": int(bl), "head": int(bh),
                    "normalized_mse": float(mean_bf16[bl, bh]), "single_input": records[bi]["sample_id"],
                    "single_layer": int(bsl), "single_head": int(bsh), "single_normalized_mse": float(bf16[bi, bsl, bsh])}
    return result


def run_validation(output):
    output = Path(output)
    if (output / "validation_freeze.json").exists() or (output / "calibration.npz").exists():
        raise FileExistsError("validation/calibration already exists; use a new output directory")
    if read_json(output / "preflight.json").get("status") not in ("passed", "completed"):
        raise ValueError("source preflight has not passed")
    records = read_json(output / "split.json")["samples"]
    calib = [r for r in records if r["split"] == "calib"]
    validation = [r for r in records if r["split"] == "validation"]
    models = fit_models(calib, output)
    summary, ideal, sampled, actual = evaluate(validation, output, models, "validation")
    decisions = decide(summary, models)
    comparison = comparisons(validation, summary, output, "validation")
    for rep, name in enumerate(REPRESENTATIONS):
        decisions[name]["A3"] = {rank: {key: value[rep] if key != "condition_degradation" else value[:, rep]
            for key, value in values.items()} for rank, values in comparison.items()}
    budget_report(validation, models, output, "validation")
    rope = run_rope(calib, validation, output, "validation")
    _save(output / "validation_report.json", {"status": "completed", "decisions": decisions,
        "condition_names": summary["condition_names"], "metric_names": METRICS,
        "methods": METHODS, "condition_scores": summary["condition_scores"],
        "equal_condition_scores": summary["equal_condition_scores"],
        "bf16_condition_scores": summary["bf16_condition_scores"],
        "bf16_equal_condition_scores": summary["bf16_equal_condition_scores"],
        "sample_ideal_equal_condition_scores": summary["sample_ideal_equal_condition_scores"],
        "worst_heads": _report_rows(validation, models, ideal, sampled, actual), "rope": rope})
    # This publication is the sole gate for Test. All candidate choices are already on disk.
    _save(output / "validation_freeze.json", {"status": "frozen_before_test", "frozen_at_unix": time.time(),
        "calibration_file": "calibration.npz", "rope_calibration_file": "rope_calibration.npz",
        "split_file": "split.json", "decisions": decisions, "test_methods": METHODS,
        "test_role": "failed Validation methods remain prespecified diagnostics and cannot be promoted by Test",
        "protocol_file": "protocol.json", "bf16_dtype_path": "FP32 matmul, BF16 coefficients, FP32 decode"})
    return decisions


def run_test(output):
    output = Path(output)
    frozen = read_json(output / "validation_freeze.json")
    if frozen["status"] != "frozen_before_test" or (output / "test_errors.npz").exists():
        raise ValueError("Test requires frozen Validation and a new Test result")
    records = read_json(output / "split.json")["samples"]
    test = [r for r in records if r["split"] == "test"]
    models = _load_models(output)
    summary, ideal, sampled, actual = evaluate(test, output, models, "test")
    budget_report(test, models, output, "test")
    comparison = comparisons(test, summary, output, "test")
    assessment = decide(summary, models, select=False)
    for name in REPRESENTATIONS:
        selected = frozen["decisions"][name]["A2"]["selected_sharing"]
        assessment[name]["A1"]["confirmed_candidate"] = (
            frozen["decisions"][name]["A1"]["passed"] and assessment[name]["A1"]["passed"])
        assessment[name]["A2"]["frozen_selected_sharing"] = selected
        assessment[name]["A2"]["confirmed_ideal_candidate"] = (
            selected is not None and assessment[name]["A2"]["candidates"][selected]["passed"])
        actual_selected = frozen["decisions"][name]["A2"].get("selected_bf16_sample_sharing")
        assessment[name]["A2"]["frozen_bf16_sample_sharing"] = actual_selected
        assessment[name]["A2"]["confirmed_bf16_sample_candidate"] = (
            actual_selected is not None and assessment[name]["A2"]["candidates"][actual_selected]["passed"]
            and assessment[name]["A2"]["candidates"][actual_selected]["bf16_sample_threshold_passed"])
    for rank, values in comparison.items():
        validation_pass = np.asarray([frozen["decisions"][name]["A3"][rank]["passed"] for name in REPRESENTATIONS])
        validation_sample_pass = np.asarray([frozen["decisions"][name]["A3"][rank]["sample_advantage_preserved"] for name in REPRESENTATIONS])
        values["confirmed_ideal_candidate"] = validation_pass & values["passed"]
        values["confirmed_bf16_sample_candidate"] = values["confirmed_ideal_candidate"] & validation_sample_pass & values["sample_advantage_preserved"]
    rope = run_rope([], test, output, "test")
    _save(output / "test_report.json", {"status": "completed", "validation_decisions_unchanged": frozen["decisions"],
        "frozen_candidate_assessment": assessment,
        "condition_names": summary["condition_names"], "metric_names": METRICS, "methods": METHODS,
        "condition_scores": summary["condition_scores"], "equal_condition_scores": summary["equal_condition_scores"],
        "bf16_condition_scores": summary["bf16_condition_scores"],
        "bf16_equal_condition_scores": summary["bf16_equal_condition_scores"],
        "sample_ideal_equal_condition_scores": summary["sample_ideal_equal_condition_scores"],
        "A3_prespecified_test_diagnostics": comparison,
        "worst_heads": _report_rows(test, models, ideal, sampled, actual), "rope": rope})
    _save(output / "summary.json", {"status": "completed", "scope": "V3-A only; no model inference, V3-B or V3-C",
        "validation_report": "validation_report.json", "test_report": "test_report.json",
        "test_opened_after_validation_freeze": True, "split_counts": read_json(output / "split.json")["counts"]})
    return summary


def run_rope(calibration, evaluation, output, phase):
    """Three sampled paths to the same original K_post target; a numerical supplement."""
    config = read_json(Path(output) / "rope_config.json")

    def vectors(record):
        raw = read_sample(record, include_vectors=True)
        pre, post = raw["vectors"][:2].astype(np.float64)
        c, s = rope_coefficients(config, raw["positions"], pre.shape[-1])
        inverse = apply_position_rope(post, c, s, inverse=True)
        return np.stack((pre, post, inverse)), c, s

    path = Path(output) / "rope_calibration.npz"
    if phase == "validation":
        means, covs = [], []
        for record in calibration:
            x, _, _ = vectors(record); mu = x.mean(-2); centered = x - mu[..., None, :]
            means.append(mu); covs.append(centered.swapaxes(-1, -2) @ centered / 448)
        fit = fit_calibration(np.stack(means), np.stack(covs), [r["condition"] for r in calibration],
                              [r["source_group"] for r in calibration])
        fitted = fit_basis(fit["G"], fit["scale"], "head")
        write_npz(path, {"mean": fit["mean"], "basis": fitted["basis"], "scale": fit["scale"][1]})
    with np.load(path) as a:
        mu, basis, scale = a["mean"], a["basis"], a["scale"]
    ideal, actual, base_ideal, base_bf16 = [], [], [], []
    import torch
    for record in evaluation:
        x, c, s = vectors(record); target = x[1]; ir, ar = [], []
        for rank in RANKS:
            u = basis[..., :int(rank)]
            decoded = ((x - mu[..., None, :]) @ u) @ u.swapaxes(-1, -2) + mu[..., None, :]
            x32 = torch.from_numpy(x.astype(np.float32)); m32 = torch.from_numpy(mu.astype(np.float32))
            u32 = torch.from_numpy(np.array(u, dtype=np.float32, copy=True))
            z = ((x32 - m32.unsqueeze(-2)) @ u32).to(torch.bfloat16)
            bf = (z.float() @ u32.transpose(-1, -2) + m32.unsqueeze(-2)).numpy().astype(np.float64)
            for pi in (0, 2):
                decoded[pi] = apply_position_rope(decoded[pi], c, s)
                bf[pi] = apply_position_rope(bf[pi], c, s, bf16=True)
            ir.append(np.square(decoded - target).sum(-1).mean(-1))
            ar.append(np.square(bf - target).sum(-1).mean(-1))
        ideal.append(np.stack(ir, -1)); actual.append(np.stack(ar, -1))
        base_ideal.append(np.stack([np.square(apply_position_rope(x[pi], c, s) - target).sum(-1).mean(-1)
                                    if pi != 1 else np.zeros_like(scale) for pi in range(3)]))
        base_bf16.append(np.stack([np.square(apply_position_rope(x[pi], c, s, bf16=True) - target).sum(-1).mean(-1)
                                   if pi != 1 else np.zeros_like(scale) for pi in range(3)]))
        print(f"V3 RoPE {phase}: {record['sample_id']}", flush=True)
    ideal, actual = np.stack(ideal), np.stack(actual)
    weights = condition_weights([r["condition"] for r in evaluation])
    write_npz(Path(output) / f"{phase}_rope.npz", {"sample_ids": np.asarray([r["sample_id"] for r in evaluation]),
        "paths": np.asarray(["k_pre_then_rope", "k_post_direct", "inverse_rope_then_rope"]),
        "ranks": RANKS, "scale": scale, "ideal_mse": ideal, "bf16_mse": actual,
        "transform_only_ideal_mse": np.stack(base_ideal), "transform_only_bf16_mse": np.stack(base_bf16)})
    return {"scope": "same 448 middle positions; CPU coefficients; not independent mechanisms or full-cache evidence",
        "ideal_normalized_mean": np.einsum("s,splhr->pr", weights, ideal / scale[None, None, ..., None]) / scale.size,
        "bf16_normalized_mean": np.einsum("s,splhr->pr", weights, actual / scale[None, None, ..., None]) / scale.size,
        "transform_only_ideal": describe(np.stack(base_ideal)), "transform_only_bf16": describe(np.stack(base_bf16))}
