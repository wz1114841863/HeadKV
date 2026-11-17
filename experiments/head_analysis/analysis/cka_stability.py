"""Question C: reuse published CKA values to compare within-layer Head pairs."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from experiments.head_analysis.analysis.head_identity import (
    fixed_reference_comparison, input_pair_correlations, load_profiles, reference_bootstrap,
)
from experiments.head_analysis.analysis.kv_representation import _json_value, describe
from experiments.head_analysis.common.io import read_json, write_json, write_npz
from experiments.head_analysis.common.statistics import average_ranks, finite_mean, pearson


REPRESENTATIONS = ("k_pre", "k_post", "v")
KINDS = ("observed", "excess")
FEATURE_SPECS = tuple((f"{rep}.{kind}", f"cka_{rep}_{kind}", None)
                      for rep in REPRESENTATIONS for kind in KINDS)


def top_weights(values, k):
    """Top-k membership; boundary ties share the remaining mass equally.

    A constant or incomplete pair profile has no comparable high-pair set.
    """
    values = np.asarray(values, dtype=np.float64)
    if not 0 < k < values.shape[-1]:
        raise ValueError("top-k must be positive and smaller than the pair count")
    complete = np.isfinite(values).all(-1)
    safe = np.where(np.isfinite(values), values, -np.inf)
    threshold = np.sort(safe, axis=-1)[..., -k, None]
    above, tied = safe > threshold, safe == threshold
    fraction = (k - above.sum(-1)) / tied.sum(-1)
    weights = above.astype(np.float64) + tied * fraction[..., None]
    valid = complete & (safe.max(-1) > safe.min(-1))
    return np.where(valid[..., None], weights, np.nan)


def pair_matrix(values, pairs, heads):
    """Expand unordered pair values symmetrically; diagonal is unmeasured (NaN)."""
    matrix = np.full(values.shape[:-1] + (len(heads), len(heads)), np.nan)
    lookup = {int(head): i for i, head in enumerate(heads)}
    left, right = np.asarray([(lookup[int(i)], lookup[int(j)]) for i, j in pairs]).T
    matrix[..., left, right] = matrix[..., right, left] = values
    return matrix


def _spearman(left, right):
    valid = np.isfinite(left) & np.isfinite(right)
    return pearson(average_ranks(np.where(valid, left, np.nan)),
                   average_ranks(np.where(valid, right, np.nan)))


def run_analysis(bundle, reference_bundle, output_dir, *, sample_index,
                 top_k=7, bootstrap_replicates=2000, seed=20261112):
    bundle, reference_bundle, output_dir = map(Path, (bundle, reference_bundle, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    source, values, layers, pairs = load_profiles(bundle, FEATURE_SPECS, coordinate_key="cka_pairs")
    reference, ref_values, ref_layers, ref_pairs = load_profiles(reference_bundle, FEATURE_SPECS, coordinate_key="cka_pairs")
    if not (np.array_equal(layers, ref_layers) and np.array_equal(pairs, ref_pairs)):
        raise ValueError("Pilot and extension Layer/Head-pair coordinates differ")
    heads = np.unique(pairs)
    if not np.array_equal(pairs, np.asarray(list(combinations(heads, 2)))):
        raise ValueError("CKA requires all unordered pairs in the same physical Head order")
    if source.get("definitions") != reference.get("definitions"):
        raise ValueError("Pilot and extension metric definitions differ")
    ids = reference["sample_ids"] + source["sample_ids"]
    if len(set(ids)) != len(ids):
        raise ValueError("Pilot and extension sample IDs overlap")
    index = {row["sample_id"]: row for row in read_json(sample_index)["samples"]}
    records = [index[sid] for sid in ids]
    for row, sample in zip(records, reference["samples"] + source["samples"]):
        if any(row[key] != sample[key] for key in ("dataset", "pair_id", "prompt_length")):
            raise ValueError("sample index and V2 metadata differ")
    datasets = np.asarray([row["dataset"] for row in records])
    lengths = np.asarray([row["target_prompt_length"] for row in records])
    conditions = np.asarray([f"{dataset}_{length}" for dataset, length in zip(datasets, lengths)])
    groups = np.asarray([f"pg19:{row['pair_id']}" if row["dataset"] == "pg19"
                         else f"{row['dataset']}:{row['sample_id']}" for row in records])
    is_reference = np.arange(len(ids)) < ref_values.shape[1]
    extension = ~is_reference
    if set(groups[is_reference]) & set(groups[extension]):
        raise ValueError("Pilot and extension source groups overlap")
    values = np.concatenate((ref_values, values), axis=1)
    names = np.asarray([spec[0] for spec in FEATURE_SPECS])
    reference_mean = finite_mean(ref_values, 1)
    selected = top_weights(reference_mean, top_k)
    membership = top_weights(values, top_k)
    retention = np.sum(membership[:, extension] * selected[:, None], -1) / top_k
    coordinates = {"feature_names": names, "layer_ids": layers, "cka_pairs": pairs, "kv_head_ids": heads}
    result = {"schema_version": "headkv-v2-cka-stability-c-v1", "status": "running",
        "source_bundle": str(bundle.resolve()), "reference_bundle": str(reference_bundle.resolve()),
        "sample_index": str(Path(sample_index).resolve()), "sample_ids": ids,
        "reference_sample_ids": reference["sample_ids"], "extension_sample_ids": source["sample_ids"],
        "top_k": top_k, "bootstrap_replicates": bootstrap_replicates, "seed": seed,
        "definitions": {
            "input": "published Observed and Excess CKA on the same 448 middle positions; no SVD or CKA recomputation",
            "excess": "published observed minus mean of 50 saved one-sided permutations; negative values retained",
            "sorting": "Spearman across all 28 unordered Head pairs within each Layer, average ranks for ties",
            "reference": "mean CKA over six Pilot inputs for each representation/kind/Layer/pair, fixed before extension evaluation",
            "top_k": "Pilot top-7 per Layer by default; observed/excess selected separately; tied boundary shares membership mass",
            "retention": "sum(Pilot membership * input membership)/k; k/28 random-set reference, not a significance test",
            "missing": "correlations use pairwise finite support; top-k requires complete nonconstant pair profiles",
            "matrices": "each representation has Observed/Excess symmetric Layer x 8 x 8 matrices; diagonal NaN, not measured",
            "matrix_weighting": "source-equal within condition; conditions equal for aggregate matrices; std is weighted ddof=0",
            "bootstrap": "2000 source resamples by condition, seed=20261112; reuse A/B draws, PG19 lengths share Pair draws; Pilot fixed",
            "inference": "per-input Layer means are scored; input pairs, Head pairs and Layers are not independent observations",
            "pairwise": "input-pair distributions are descriptive, without independent-pair confidence intervals",
            "scope": "fixed-model representation topology; does not establish shared Cache, mergeability or compression benefit",
        }, "features": [], "input_pair_groups": {}, "matrices": {}, "comparisons": []}
    output_dir.mkdir(parents=True)
    write_json(output_dir / "summary.json", result)
    write_npz(output_dir / "profiles.npz", {**coordinates, "values": values, "sample_ids": np.asarray(ids),
        "datasets": datasets, "target_lengths": lengths, "conditions": conditions,
        "source_groups": groups, "is_reference": is_reference})

    fixed, input_pairs = [], []
    for f, name in enumerate(names):
        fixed.append(fixed_reference_comparison(ref_values[f], values[f, extension]))
        input_pairs.append(input_pair_correlations(values[f]))
        result["features"].append({"feature": str(name),
            "reference_per_layer_spearman": describe(fixed[-1]["per_layer_rho"]),
            "top_k_retention": describe(retention[f]), "random_set_retention": top_k / len(pairs)})
    write_npz(output_dir / "fixed_reference.npz", {**coordinates,
        "sample_ids": np.asarray(source["sample_ids"]), "reference_mean": reference_mean,
        "reference_top_weights": selected, "top_k_retention": retention,
        "target_top_weights": membership[:, extension],
        **{key: np.stack([row[key] for row in fixed]) for key in fixed[0]}})
    i, j = input_pairs[0]["pair_indices"].T
    both = extension[i] & extension[j]
    pair_masks = {"extension_all": both, "extension_same_condition": both & (conditions[i] == conditions[j]),
        "extension_cross_task": both & (datasets[i] != datasets[j]),
        "extension_same_task_different_length": both & (datasets[i] == datasets[j]) & (conditions[i] != conditions[j]),
        "extension_pg19_paired": both & (datasets[i] == "pg19") & (groups[i] == groups[j]),
        "pilot_all": is_reference[i] & is_reference[j], "all_inputs": np.ones(len(i), bool)}
    for label, mask in pair_masks.items():
        result["input_pair_groups"][label] = {"input_pair_count": int(mask.sum()),
            **{str(name): describe(row["per_layer_rho"][mask]) for name, row in zip(names, input_pairs)}}
    write_npz(output_dir / "input_pairs.npz", {**coordinates, "sample_ids": np.asarray(ids),
        **{key: input_pairs[0][key] if key == "pair_indices" else np.stack([row[key] for row in input_pairs])
           for key in input_pairs[0]}, **{f"group_{key}": mask for key, mask in pair_masks.items()}})

    # Matrix means retain per-condition results so pooled levels cannot hide task shifts.
    cohorts = {"pilot": is_reference, "extension_equal_conditions": extension,
               "all_equal_conditions": np.ones(len(ids), bool)}
    for population, mask in (("extension", extension), ("all", np.ones(len(ids), bool))):
        cohorts.update({f"{population}_{label}": mask & (conditions == label) for label in sorted(set(conditions))})
    means, stds, counts, frequencies = [], [], [], []
    for mask in cohorts.values():
        labels = sorted(set(conditions[mask]))
        mean = finite_mean(np.stack([finite_mean(values[:, mask & (conditions == label)], 1) for label in labels]), 0)
        variance = finite_mean(np.stack([finite_mean((values[:, mask & (conditions == label)] - mean[:, None]) ** 2, 1)
                                        for label in labels]), 0)
        frequency = finite_mean(np.stack([finite_mean(membership[:, mask & (conditions == label)], 1) for label in labels]), 0)
        means.append(mean); stds.append(np.sqrt(variance)); frequencies.append(frequency)
        counts.append(np.isfinite(values[:, mask]).sum(1))
    means, stds, counts, frequencies = map(np.stack, (means, stds, counts, frequencies))
    for ri, rep in enumerate(REPRESENTATIONS):
        select = slice(2 * ri, 2 * ri + 2)
        file = f"matrices_{rep}.npz"
        write_npz(output_dir / file, {"kind_names": np.asarray(KINDS), "cohort_names": np.asarray(list(cohorts)),
            "sample_masks": np.stack(list(cohorts.values())), "layer_ids": layers, "kv_head_ids": heads,
            "mean": pair_matrix(means[:, select], pairs, heads), "std": pair_matrix(stds[:, select], pairs, heads),
            "valid_input_count": np.nan_to_num(pair_matrix(counts[:, select].astype(float), pairs, heads), nan=0).astype(np.uint32),
            "top_k_frequency": pair_matrix(frequencies[:, select], pairs, heads),
            "pilot_top_weights": pair_matrix(selected[select], pairs, heads)})
        result["matrices"][rep] = {"numeric_file": file, "cohort_names": list(cohorts),
            "extension_observed": describe(means[1, 2 * ri]), "extension_excess": describe(means[1, 2 * ri + 1])}

    # Compare representation topologies within the same input, and observed vs excess.
    contrast_names, correlations, supports = [], [], []
    contrasts = [(f"{rep}.observed_vs_excess", 2 * ri, 2 * ri + 1) for ri, rep in enumerate(REPRESENTATIONS)]
    contrasts.extend((f"{left}_vs_{right}.{kind}", 2 * REPRESENTATIONS.index(left) + ki,
                      2 * REPRESENTATIONS.index(right) + ki)
                     for left, right in combinations(REPRESENTATIONS, 2) for ki, kind in enumerate(KINDS))
    for name, left, right in contrasts:
        rho, count = _spearman(values[left], values[right])
        contrast_names.append(name); correlations.append(rho); supports.append(count)
        result["comparisons"].append({"name": name, "extension_per_layer_spearman": describe(rho[extension])})
    correlations = np.stack(correlations)
    write_npz(output_dir / "comparisons.npz", {"comparison_names": np.asarray(contrast_names),
        "sample_ids": np.asarray(ids), "layer_ids": layers, "per_layer_rho": correlations,
        "valid_pair_count": np.stack(supports)})
    score_names, scores = [], []
    for f, name in enumerate(names):
        for metric, score in (("mean_cka", finite_mean(values[f, extension], (1, 2))),
                ("reference_spearman", finite_mean(fixed[f]["per_layer_rho"], -1)),
                ("pilot_top_k_retention", finite_mean(retention[f], -1))):
            score_names.append(f"{name}.{metric}"); scores.append(score)
    for name, rho in zip(contrast_names, correlations):
        score_names.append(name); scores.append(finite_mean(rho[extension], -1))
    scores = np.stack(scores, -1)
    bootstrap = reference_bootstrap(scores[:, None], conditions[extension], datasets[extension], groups[extension],
                                    replicates=bootstrap_replicates, seed=seed)
    write_npz(output_dir / "bootstrap.npz", {"metric_names": np.asarray(score_names),
        "sample_ids": np.asarray(source["sample_ids"]), "input_scores": scores, **bootstrap})
    result["bootstrap"] = {"metric_names": score_names, **{key: value for key, value in bootstrap.items()
        if key.startswith(("condition_", "equal_condition_")) and "bootstrap_means" not in key}}
    result = _json_value(result)
    result["status"] = "completed"
    write_json(output_dir / "summary.json", result, overwrite=True)
    return result
