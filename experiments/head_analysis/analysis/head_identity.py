"""Describe fixed Head contrasts and input dependence in existing V2 profiles."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.head_analysis.analysis.kv_representation import _json_value, correlate_heads, describe
from experiments.head_analysis.common.io import iter_jsonl, read_json, write_json, write_npz
from experiments.head_analysis.common.statistics import (
    average_ranks, bootstrap_draws, bootstrap_means, correlation_1d, finite_mean,
    pearson, percentile_ci,
)


FEATURE_SPECS = (
    ("full_k_post_middle_effective_rank", "full_k_post_middle_effective_rank", None),
    ("full_v_middle_effective_rank", "full_v_middle_effective_rank", None),
    ("full_k_post_middle_E16", "full_k_post_middle_variance_coverage", 1),
    ("full_v_middle_E16", "full_v_middle_variance_coverage", 1),
    ("full_k_post_middle_mean_norm", "full_k_post_middle_norm_summary", 0),
    ("full_v_middle_mean_norm", "full_v_middle_norm_summary", 0),
    *((f"sample_{rep}_middle_effective_rank", f"sample_{rep}_middle_effective_rank", None)
      for rep in ("k_pre", "k_post", "v")),
    *((f"cka_{rep}_head_{field}", f"cka_{rep}_head_{field}", None)
      for rep in ("k_pre", "k_post", "v") for field in ("observed", "excess")),
    ("rope_middle_rank_post_pre_ratio", "rope_middle_rank_post_pre_ratio", None),
)

A_FEATURE_SPECS = FEATURE_SPECS[:6] + tuple(
    (name.replace("full_", "sample_", 1), key.replace("full_", "sample_", 1), column)
    for name, key, column in FEATURE_SPECS[:6]
)


def _common_values(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2 or not all(values.shape[1:]):
        raise ValueError("profiles require [Sample,Layer,Head] with at least two samples")
    valid = np.isfinite(values).all(axis=0)
    valid &= (valid.sum(axis=-1) >= 2)[:, None]
    return np.where(valid[None], values, np.nan), valid, valid.sum(axis=-1)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else float("nan")


def decompose_profiles(values):
    """Balanced descriptive sums of squares, using shared finite Head coordinates.

    X = layer baseline + input-layer shift + fixed Head contrast + input×Head
    response. The response also contains sampling/length variation; it is not a
    separate estimate of measurement noise or a population random-effect ICC.
    """
    x, valid, counts = _common_values(values)
    samples, layers, _ = x.shape
    # Shift before reduction so an exactly constant decimal profile stays constant.
    first = np.argmax(valid, axis=-1)
    origin = np.where(counts > 0, x[0, np.arange(layers), first], 0.)
    shifted = x - origin[None, :, None]
    mean = finite_mean(shifted, axis=(0, 2))
    input_effect = finite_mean(shifted, axis=-1) - mean[None]
    head_effect = finite_mean(shifted, axis=0) - mean[:, None]
    centered = shifted - finite_mean(shifted, axis=-1)[..., None]
    residual = centered - head_effect[None]
    by_layer = {
        "total": np.nansum((shifted - mean[None, :, None]) ** 2, axis=(0, 2)),
        "head": samples * np.nansum(head_effect ** 2, axis=-1),
        "input": counts * np.nansum(input_effect ** 2, axis=0),
        "interaction": np.nansum(residual ** 2, axis=(0, 2)),
    }
    sums = {key: float(value.sum()) for key, value in by_layer.items()}
    layer_shares = {}
    for key, source in (("fixed_head", "head"), ("input", "input"), ("interaction", "interaction")):
        layer_shares[key] = np.divide(by_layer[source], by_layer["total"],
                                       out=np.full(layers, np.nan), where=by_layer["total"] > 0)
    relative_denominator = by_layer["head"] + by_layer["interaction"]
    per_head_input_std = np.sqrt(finite_mean((shifted - finite_mean(shifted, 0)[None]) ** 2, 0))
    return {
        "layer_baseline": origin + mean, "input_effect": input_effect,
        "head_effect": head_effect, "residual": residual, "centered": centered,
        "valid_mask": valid, "valid_head_count": counts.astype(np.uint32),
        **{f"ss_{key}": value for key, value in sums.items()},
        **{f"per_layer_ss_{key}": value for key, value in by_layer.items()},
        "shares": {key: _ratio(sums[source], sums["total"])
                   for key, source in (("fixed_head", "head"), ("input", "input"), ("interaction", "interaction"))},
        "relative_fixed_share": _ratio(sums["head"], sums["head"] + sums["interaction"]),
        "per_layer_shares": layer_shares,
        "per_layer_relative_fixed_share": np.divide(by_layer["head"], relative_denominator,
            out=np.full(layers, np.nan), where=relative_denominator > 0),
        "between_head_rms": np.sqrt(_ratio(sums["head"], samples * valid.sum())),
        "within_head_input_rms": np.sqrt(_ratio(sums["input"] + sums["interaction"], samples * valid.sum())),
        "per_head_input_std": per_head_input_std,
        "per_head_relative_std": np.sqrt(finite_mean(residual ** 2, 0)),
    }


def pairwise_and_holdout(values, source_groups):
    """Compare relative Head shapes; exclude the entire source group when held out."""
    decomposition = decompose_profiles(values)
    x = np.where(decomposition["valid_mask"][None], np.asarray(values, dtype=np.float64), np.nan)
    z = decomposition["centered"]
    groups = np.asarray(source_groups)
    samples, layers, _ = x.shape
    if groups.shape != (samples,) or len(set(groups.tolist())) < 2:
        raise ValueError("at least two source groups aligned with samples are required")
    pairwise = np.full((samples, samples), np.nan)
    per_layer = np.full((samples, samples, layers), np.nan)
    pair_count = np.zeros((samples, samples), dtype=np.uint32)
    per_layer_count = np.zeros((samples, samples, layers), dtype=np.uint32)
    for i in range(samples):
        for j in range(i + 1, samples):
            correlation = correlate_heads(x[i], x[j])
            pairwise[i, j] = pairwise[j, i] = correlation["layer_centered_spearman"]
            per_layer[i, j] = per_layer[j, i] = correlation["per_layer_spearman"]
            pair_count[i, j] = pair_count[j, i] = correlation["layer_centered_valid_head_count"]
            per_layer_count[i, j] = per_layer_count[j, i] = correlation["per_layer_valid_head_count"]
    prediction = np.full_like(z, np.nan)
    rho = np.full(samples, np.nan)
    skill = np.full(samples, np.nan)
    error = np.zeros(samples)
    baseline = np.zeros(samples)
    train_count = np.zeros(samples, dtype=np.uint32)
    for i in range(samples):
        train = groups != groups[i]
        train_count[i] = train.sum()
        prediction[i] = finite_mean(z[train], 0)
        rho[i] = correlation_1d(prediction[i].ravel(), z[i].ravel())[1]
        error[i] = np.nansum((prediction[i] - z[i]) ** 2)
        baseline[i] = np.nansum(z[i] ** 2)
        skill[i] = 1 - _ratio(error[i], baseline[i])
    return {"pairwise_spearman": pairwise, "pairwise_per_layer": per_layer,
            "pairwise_valid_count": pair_count, "pairwise_per_layer_valid_count": per_layer_count,
            "predictions": prediction, "holdout_rho": rho, "holdout_skill": skill,
            "holdout_train_count": train_count, "holdout_squared_error": error,
            "holdout_baseline_squared_error": baseline,
            "pooled_skill": 1 - _ratio(error.sum(), baseline.sum())}


def _cohorts(samples):
    result = [("all_six", np.arange(len(samples)))]
    long_text = np.asarray([i for i, sample in enumerate(samples) if sample["dataset"] != "ruler"])
    if len(long_text) >= 2:
        result.append(("long_text", long_text))
    pairs = defaultdict(list)
    for i, sample in enumerate(samples):
        if sample["dataset"] == "pg19":
            if not sample.get("pair_id"):
                raise ValueError("PG19 profiles require their source pair_id")
            pairs[sample["pair_id"]].append(i)
    if any(len(indices) > 1 for indices in pairs.values()):
        for label, reverse in (("short", False), ("long", True)):
            drop = set()
            for indices in pairs.values():
                ordered = sorted(indices, key=lambda i: samples[i]["prompt_length"], reverse=reverse)
                drop.update(ordered[1:])
            result.append((f"pg19_{label}_representative", np.asarray([i for i in range(len(samples)) if i not in drop])))
    return result


def load_profiles(bundle: Path, feature_specs=FEATURE_SPECS):
    """Read published features with their sample and physical Head coordinates."""
    bundle = Path(bundle)
    source = read_json(bundle / "summary.json")
    samples = source["samples"]
    if source["status"] != "completed":
        raise ValueError("V2 numeric bundle is incomplete")
    ids = [sample["sample_id"] for sample in samples]
    if len(ids) != len(set(ids)) or ids != source["sample_ids"]:
        raise ValueError("V2 sample IDs are duplicated or differ from the sample summaries")
    values = []
    layers = heads = None
    for sample in samples:
        with np.load(bundle / sample["numeric_file"], allow_pickle=False) as arrays:
            if layers is None:
                layers, heads = arrays["layer_ids"].copy(), arrays["kv_head_ids"].copy()
            elif not (np.array_equal(layers, arrays["layer_ids"]) and np.array_equal(heads, arrays["kv_head_ids"])):
                raise ValueError("Layer/KV Head coordinates differ across samples")
            values.append(np.stack([arrays[key] if index is None else arrays[key][..., index]
                                    for _, key, index in feature_specs]).astype(np.float64))
    profiles = np.stack(values, axis=1)
    if profiles.shape[2:] != (len(layers), len(heads)):
        raise ValueError("feature arrays differ from their Layer/KV Head coordinates")
    return source, profiles, layers, heads


def run_analysis(bundle: Path, output_dir: Path):
    """Read the completed V2 numeric package without model, raw K/V or new CKA."""
    bundle, output_dir = Path(bundle), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    source, profiles, layers, heads = load_profiles(bundle)
    samples, ids = source["samples"], source["sample_ids"]
    datasets = np.asarray([sample["dataset"] for sample in samples])
    groups = np.asarray([f"pg19:{sample['pair_id']}" if sample["dataset"] == "pg19"
                         else f"{sample['dataset']}:{sample['sample_id']}" for sample in samples])
    feature_names = [name for name, _, _ in FEATURE_SPECS]
    result = {"schema_version": "headkv-v2-head-identity-v1", "status": "running",
              "source_bundle": str(bundle.resolve()), "sample_ids": ids, "feature_names": feature_names,
              "source_groups": groups.tolist(), "cohorts": [],
              "definitions": {
                  "values": "published feature arrays converted losslessly to FP64; no new spectrum or CKA computation",
                  "axes": "feature,sample,layer,physical KV Head; same common-finite coordinates per cohort/feature",
                  "decomposition": "X = layer baseline + input-layer shift + fixed Head contrast + input×Head response",
                  "sums_of_squares": "SS_input=H*sum A²; SS_head=S*sum B²; SS_response=sum R²; per-layer shared H",
                  "shares": "descriptive SS shares of within-layer total; not population variance components or ICC",
                  "relative_fixed_share": "SS_head/(SS_head+SS_response), after removing each input-layer mean",
                  "response": "includes input-specific Head behavior, sampling and length effects; not pure noise",
                  "pairwise": "subtract input-layer Head means before pooled Spearman; also retain each Layer's Spearman",
                  "holdout": "leave one source group out; PG19 pair excluded together; training mean centered profile",
                  "skill": "1-SSE_prediction/SSE_uniform_Head; uniform centered baseline=0; negative skill retained",
                  "holdout_scope": "relative Head-shape comparison uses test input-layer mean; not deployable absolute/new-data prediction",
                  "inference": "six already-studied inputs; PG19 paired; no p-values, CI or independent-Head inference",
              }}
    output_dir.mkdir(parents=True)
    write_json(output_dir / "summary.json", result)
    write_npz(output_dir / "profiles.npz", {"values": profiles, "feature_names": np.asarray(feature_names),
        "sample_ids": np.asarray(ids), "datasets": datasets, "source_groups": groups,
        "layer_ids": layers, "kv_head_ids": heads})
    for name, indices in _cohorts(samples):
        numeric = defaultdict(list)
        summary = {"name": name, "sample_ids": [ids[i] for i in indices],
                   "numeric_file": f"{name}.npz", "features": []}
        pair_i, pair_j = np.triu_indices(len(indices), 1)
        cross_dataset = datasets[indices][pair_i] != datasets[indices][pair_j]
        for feature, data in zip(feature_names, profiles[:, indices]):
            decomposition = decompose_profiles(data)
            comparison = pairwise_and_holdout(data, groups[indices])
            for key, value in {**decomposition, **comparison}.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        numeric[f"{key}_{subkey}"].append(subvalue)
                else:
                    numeric[key].append(value)
            summary["features"].append({"feature": feature,
                "valid_head_coordinates": int(decomposition["valid_mask"].sum()),
                "valid_layers": int((decomposition["valid_head_count"] > 0).sum()),
                "shares": decomposition["shares"], "relative_fixed_share": decomposition["relative_fixed_share"],
                "between_head_rms": decomposition["between_head_rms"],
                "within_head_input_rms": decomposition["within_head_input_rms"],
                "within_to_between_rms_ratio": _ratio(decomposition["within_head_input_rms"], decomposition["between_head_rms"]),
                "per_layer_fixed_head_share": describe(decomposition["per_layer_shares"]["fixed_head"]),
                "per_layer_relative_fixed_share": describe(decomposition["per_layer_relative_fixed_share"]),
                "pairwise_spearman": describe(comparison["pairwise_spearman"][pair_i, pair_j]),
                "cross_dataset_pairwise_spearman": describe(comparison["pairwise_spearman"][pair_i[cross_dataset], pair_j[cross_dataset]]),
                "per_layer_pairwise_spearman": describe(comparison["pairwise_per_layer"][pair_i, pair_j]),
                "holdout_rho": describe(comparison["holdout_rho"]),
                "holdout_skill": describe(comparison["holdout_skill"]),
                "pooled_skill": comparison["pooled_skill"]})
        write_npz(output_dir / summary["numeric_file"], {
            "feature_names": np.asarray(feature_names), "sample_indices": indices,
            "sample_ids": np.asarray(summary["sample_ids"]),
            **{key: np.asarray(value) for key, value in numeric.items()}})
        result["cohorts"].append(_json_value(summary))
    result["status"] = "completed"
    write_json(output_dir / "summary.json", result, overwrite=True)
    return result


def fixed_reference_comparison(reference, targets):
    """Freeze the reference Head contrasts; targets only determine evaluation support."""
    frozen = decompose_profiles(reference)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 3 or targets.shape[1:] != frozen["head_effect"].shape:
        raise ValueError("reference and targets require matching Layer/KV Head axes")
    valid = np.isfinite(targets) & np.isfinite(frozen["head_effect"])[None]
    x = np.where(valid, targets, np.nan)
    prediction = np.where(valid, frozen["head_effect"][None], np.nan)
    layer_mean = finite_mean(x, -1)
    centered = x - layer_mean[..., None]
    prediction = prediction - finite_mean(prediction, -1)[..., None]
    per_layer_rho, per_layer_count = pearson(average_ranks(prediction), average_ranks(centered))
    samples = len(x)
    rho, count = pearson(average_ranks(prediction.reshape(samples, -1)),
                         average_ranks(centered.reshape(samples, -1)))
    error = np.nansum((prediction - centered) ** 2, axis=-1)
    baseline = np.nansum(centered ** 2, axis=-1)
    per_layer_skill = np.full_like(error, np.nan)
    np.divide(error, baseline, out=per_layer_skill, where=baseline > 0)
    skill = np.full(samples, np.nan)
    np.divide(error.sum(-1), baseline.sum(-1), out=skill, where=baseline.sum(-1) > 0)
    return {"reference_profile": frozen["head_effect"], "reference_layer_mean": frozen["layer_baseline"],
            "target_layer_mean": layer_mean, "rho": rho, "per_layer_rho": per_layer_rho,
            "valid_count": count, "per_layer_valid_count": per_layer_count,
            "skill": 1 - skill, "per_layer_skill": 1 - per_layer_skill,
            "per_layer_error": error, "per_layer_baseline": baseline}


def input_pair_correlations(values):
    """Batched upper-triangle correlations with pairwise finite Head support."""
    values = np.asarray(values, dtype=np.float64)
    i, j = np.triu_indices(len(values), 1)
    valid = np.isfinite(values[i]) & np.isfinite(values[j])
    left, right = np.where(valid, values[i], np.nan), np.where(valid, values[j], np.nan)
    layer_rho, layer_count = pearson(average_ranks(left), average_ranks(right))
    left = left - finite_mean(left, -1)[..., None]
    right = right - finite_mean(right, -1)[..., None]
    rho, count = pearson(average_ranks(left.reshape(len(i), -1)),
                         average_ranks(right.reshape(len(i), -1)))
    return {"pair_indices": np.stack((i, j), -1), "rho": rho, "valid_count": count,
            "per_layer_rho": layer_rho, "per_layer_valid_count": layer_count}


def reference_bootstrap(values, conditions, datasets, groups, *, replicates=2000, seed=20261112):
    """Bootstrap fixed-reference scores by source; PG19 lengths share Pair draws.

    values[Sample,Feature,Metric] contain per-input scores, never Head or input-pair
    observations. The reference itself is not refitted or resampled.
    """
    values = np.asarray(values, dtype=np.float64)
    conditions, datasets, groups = map(np.asarray, (conditions, datasets, groups))
    labels = sorted(set(conditions.tolist()))
    numeric, points, draws_by_stratum, group_ids, draws = {}, [], {}, {}, []
    valid_counts = []
    for condition in labels:
        indices = np.flatnonzero(conditions == condition)
        stratum = "pg19" if datasets[indices[0]] == "pg19" else condition
        source_ids = sorted(set(groups[indices].tolist()))
        if stratum not in draws_by_stratum:
            group_ids[stratum] = source_ids
            draws_by_stratum[stratum] = bootstrap_draws(seed, f"v2-a:{stratum}", replicates, len(source_ids))
            numeric[f"draws_{stratum}"] = draws_by_stratum[stratum]
            numeric[f"group_ids_{stratum}"] = np.asarray(source_ids)
        elif group_ids[stratum] != source_ids:
            raise ValueError("PG19 length conditions must contain the same source Pairs")
        grouped = np.stack([finite_mean(values[indices[groups[indices] == group]], 0) for group in source_ids])
        points.append(finite_mean(grouped, 0))
        valid_counts.append(np.isfinite(grouped).sum(0))
        draws.append(bootstrap_means(grouped, draws_by_stratum[stratum]))
    samples = np.stack(draws, axis=1)  # Bootstrap,Condition,Feature,Metric
    point = np.stack(points)
    ci, count = percentile_ci(samples)
    equal_samples = finite_mean(samples, 1)
    equal_ci, equal_count = percentile_ci(equal_samples)
    return {**numeric, "condition_names": np.asarray(labels), "condition_mean": point,
            "condition_ci95": ci, "condition_valid_replicates": count,
            "condition_valid_source_count": np.asarray(valid_counts, dtype=np.uint32),
            "condition_bootstrap_means": samples, "equal_condition_mean": finite_mean(point, 0),
            "equal_condition_ci95": equal_ci, "equal_condition_valid_replicates": equal_count,
            "equal_condition_valid_condition_count": np.isfinite(point).sum(0).astype(np.uint32)}


def run_reference_analysis(bundle, reference_bundle, output_dir, *, fixed_inputs,
                           bootstrap_replicates=2000, seed=20261112):
    """Question A: new inputs compared with an immutable Pilot reference."""
    bundle, reference_bundle, output_dir = map(Path, (bundle, reference_bundle, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    source, values, layers, heads = load_profiles(bundle, A_FEATURE_SPECS)
    reference, ref_values, ref_layers, ref_heads = load_profiles(reference_bundle, A_FEATURE_SPECS)
    if not (np.array_equal(layers, ref_layers) and np.array_equal(heads, ref_heads)):
        raise ValueError("reference and target Layer/KV Head coordinates differ")
    for key in ("definitions", "seed", "permutation_replicates"):
        if source.get(key) != reference.get(key):
            raise ValueError(f"reference and target metric protocols differ: {key}")
    for key in ("model", "dtype", "attention_backend"):
        if source.get("sources", [{}])[0].get(key) != reference.get("sources", [{}])[0].get(key):
            raise ValueError(f"reference and target models differ: {key}")
    ids, ref_ids = source["sample_ids"], reference["sample_ids"]
    if set(ids) & set(ref_ids):
        raise ValueError("reference and target sample IDs overlap")
    fixed = {row["sample_id"]: row for row in iter_jsonl(Path(fixed_inputs))}

    def labels(sample_ids):
        records = [fixed[sid] for sid in sample_ids]
        datasets = np.asarray([r["dataset"] for r in records])
        conditions = np.asarray([f"{r['dataset']}_{r['target_prompt_length']}" for r in records])
        groups = np.asarray([f"pg19:{r['pair_id']}" if r["dataset"] == "pg19"
                             else f"{r['dataset']}:{r['sample_id']}" for r in records])
        return datasets, conditions, groups

    datasets, conditions, groups = labels(ids)
    _, _, ref_groups = labels(ref_ids)
    if set(groups) & set(ref_groups):
        raise ValueError("reference and target source groups overlap")
    names = [spec[0] for spec in A_FEATURE_SPECS]
    result = {"schema_version": "headkv-v2-fixed-reference-a-v1", "status": "running",
              "source_bundle": str(bundle.resolve()), "reference_bundle": str(reference_bundle.resolve()),
              "fixed_inputs": str(Path(fixed_inputs).resolve()), "sample_ids": ids,
              "reference_sample_ids": ref_ids, "feature_names": names, "cohorts": [], "features": [],
              "bootstrap_replicates": bootstrap_replicates, "seed": seed,
              "definitions": {
                  "features": "first six: full middle K_post/V rank, E16, mean Norm; last six: same 448 middle rows",
                  "reference": "mean of Pilot input-layer centered profiles; fixed before target evaluation",
                  "skill": "1-SSE_reference/SSE_uniform_Head, negative retained; targets supply their own layer means",
                  "pairwise": "each Layer's Head Spearman plus pooled layer-centered Spearman; pairwise finite support",
                  "decomposition": "existing descriptive within-layer SS decomposition; not population variance or ICC",
                  "bootstrap": "source groups within condition; PG19 lengths share Pair draws; Pilot fixed, not resampled",
                  "aggregate": "condition-equal mean of source-equal per-condition reference scores; percentile 95% CI",
                  "pairwise_inference": "input-pair correlations are descriptive distributions without CI or independent-pair inference",
                  "scope": "relative Head profiles in one fixed model; no absolute prediction, PCA-basis or compression claim"}}
    output_dir.mkdir(parents=True)
    write_json(output_dir / "summary.json", result)
    coordinates = {"feature_names": np.asarray(names), "sample_ids": np.asarray(ids),
                   "reference_sample_ids": np.asarray(ref_ids), "layer_ids": layers, "kv_head_ids": heads}
    write_npz(output_dir / "profiles.npz", {**coordinates, "values": values, "reference_values": ref_values,
              "datasets": datasets, "conditions": conditions, "source_groups": groups})
    comparison, pairs = defaultdict(list), defaultdict(list)
    for name, target, ref in zip(names, values, ref_values):
        fixed_result = fixed_reference_comparison(ref, target)
        pair_result = input_pair_correlations(target)
        for key, value in fixed_result.items():
            comparison[key].append(value)
        for key, value in pair_result.items():
            if key != "pair_indices":
                pairs[key].append(value)
        result["features"].append({"feature": name, "primary": name.startswith("full_"),
            "reference_layer_centered_spearman": describe(fixed_result["rho"]),
            "reference_per_layer_spearman": describe(fixed_result["per_layer_rho"]),
            "reference_skill": describe(fixed_result["skill"])})
        print(f"V2 A feature completed: {name}", flush=True)
    comparison = {key: np.asarray(value) for key, value in comparison.items()}
    pair_arrays = {key: np.asarray(value) for key, value in pairs.items()}
    i, j = pair_result["pair_indices"].T
    pair_groups = {"all_pairs": np.ones(len(i), dtype=bool), "same_condition": conditions[i] == conditions[j],
                   "cross_task": datasets[i] != datasets[j],
                   "same_task_different_length": (datasets[i] == datasets[j]) & (conditions[i] != conditions[j]),
                   "pg19_paired_lengths": (datasets[i] == "pg19") & (groups[i] == groups[j])}
    for feature, summary in enumerate(result["features"]):
        summary["input_pair_groups"] = {name: {"input_pair_count": int(mask.sum()),
            "layer_centered_spearman": describe(pair_arrays["rho"][feature, mask]),
            "per_layer_spearman": describe(pair_arrays["per_layer_rho"][feature, mask])}
            for name, mask in pair_groups.items()}
    write_npz(output_dir / "fixed_reference.npz", {**coordinates, **comparison})
    write_npz(output_dir / "input_pairs.npz", {**coordinates, **pair_arrays, "pair_indices": pair_result["pair_indices"],
              **{f"group_{name}": mask for name, mask in pair_groups.items()}})
    scores = np.stack((finite_mean(comparison["per_layer_rho"], -1), comparison["rho"], comparison["skill"]), -1)
    bootstrap = reference_bootstrap(scores.transpose(1, 0, 2), conditions, datasets, groups,
                                    replicates=bootstrap_replicates, seed=seed)
    score_names = ["mean_layer_spearman", "layer_centered_spearman", "relative_profile_skill"]
    write_npz(output_dir / "bootstrap.npz", {"feature_names": np.asarray(names), "metric_names": np.asarray(score_names), **bootstrap})
    result["fixed_reference_bootstrap"] = _json_value({key: value for key, value in bootstrap.items()
        if key in ("condition_names", "condition_mean", "condition_ci95", "condition_valid_source_count",
                   "condition_valid_replicates", "equal_condition_mean", "equal_condition_ci95",
                   "equal_condition_valid_replicates", "equal_condition_valid_condition_count")})
    result["fixed_reference_bootstrap"]["metric_names"] = score_names
    # Store decomposition effects and SS, reusing values to reconstruct centered/residual arrays.
    cohorts = [("extension", np.arange(len(ids))), *_cohorts(source["samples"])[1:]]
    cohorts.extend((f"condition_{condition}", np.flatnonzero(conditions == condition)) for condition in sorted(set(conditions)))
    for label, indices in cohorts:
        numeric, summary = defaultdict(list), {"name": label, "sample_ids": [ids[i] for i in indices],
                                              "numeric_file": f"{label}.npz", "features": []}
        for name, feature in zip(names, values[:, indices]):
            decomposition = decompose_profiles(feature)
            for key, value in decomposition.items():
                if key in ("centered", "residual"):
                    continue
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        numeric[f"{key}_{subkey}"].append(subvalue)
                else:
                    numeric[key].append(value)
            summary["features"].append({"feature": name, "shares": decomposition["shares"],
                "relative_fixed_share": decomposition["relative_fixed_share"],
                "per_layer_relative_fixed_share": describe(decomposition["per_layer_relative_fixed_share"]),
                "within_to_between_rms_ratio": _ratio(decomposition["within_head_input_rms"], decomposition["between_head_rms"])})
        write_npz(output_dir / summary["numeric_file"], {"feature_names": np.asarray(names), "sample_indices": indices,
                  **{key: np.asarray(value) for key, value in numeric.items()}})
        result["cohorts"].append(_json_value(summary))
    result = _json_value(result)
    result["status"] = "completed"
    write_json(output_dir / "summary.json", result, overwrite=True)
    return result
