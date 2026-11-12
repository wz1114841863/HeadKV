"""Describe fixed Head contrasts and input dependence in existing V2 profiles."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.head_analysis.analysis.kv_representation import _json_value, correlate_heads, describe
from experiments.head_analysis.common.io import read_json, write_json, write_npz
from experiments.head_analysis.common.statistics import correlation_1d, finite_mean


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


def run_analysis(bundle: Path, output_dir: Path):
    """Read the completed V2 numeric package without model, raw K/V or new CKA."""
    bundle, output_dir = Path(bundle), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
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
                                    for _, key, index in FEATURE_SPECS]).astype(np.float64))
    profiles = np.stack(values, axis=1)
    if profiles.shape[2:] != (len(layers), len(heads)):
        raise ValueError("feature arrays differ from their Layer/KV Head coordinates")
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
