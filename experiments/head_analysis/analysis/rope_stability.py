"""Cross-input RoPE rank changes from aligned-token V2 numerical packages."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.head_analysis.analysis.head_identity import (
    fixed_reference_comparison, input_pair_correlations, load_profiles, reference_bootstrap,
)
from experiments.head_analysis.analysis.kv_representation import _json_value, describe
from experiments.head_analysis.common.io import read_json, write_json, write_npz
from experiments.head_analysis.common.statistics import finite_mean, percentile_ci


RANK_SPECS = tuple((f"k_{stage}", f"sample_k_{stage}_middle_effective_rank", None)
                   for stage in ("pre", "post"))
FIELD_NAMES = ("rank_pre", "rank_post", "rank_delta", "rank_ratio")
CHANGE_NAMES = FIELD_NAMES[2:]


def rank_changes(pre, post):
    """Use only finite positive ranks on the same sampled middle positions."""
    pre, post = np.asarray(pre, dtype=np.float64), np.asarray(post, dtype=np.float64)
    if pre.shape != post.shape:
        raise ValueError("pre/post ranks must have matching coordinates")
    valid = np.isfinite(pre) & np.isfinite(post) & (pre > 0) & (post > 0)
    pre, post = np.where(valid, pre, np.nan), np.where(valid, post, np.nan)
    return np.stack((pre, post, post - pre, post / pre))


def increase_fraction(delta, axis=0):
    """Strict positive change; invalid ranks do not count as non-increases."""
    return finite_mean(np.where(np.isfinite(delta), delta > 0, np.nan), axis)


def pg19_pairs(datasets, groups, lengths):
    """Return aligned short/long input indices; never pair by row order."""
    indices = np.flatnonzero(np.asarray(datasets) == "pg19")
    pairs, names = [], []
    for group in sorted(set(groups[indices].tolist())):
        rows = indices[groups[indices] == group]
        if len(rows) != 2 or len(set(lengths[rows].tolist())) != 2:
            raise ValueError("PG19 requires exactly two distinct lengths per source Pair")
        pairs.append(rows[np.argsort(lengths[rows])])
        names.append(group)
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2), np.asarray(names)


def _head_statistics(values):
    mean = finite_mean(values, 1)
    valid = np.isfinite(values)
    count = valid.sum(1)
    minimum = np.min(np.where(valid, values, np.inf), axis=1)
    maximum = np.max(np.where(valid, values, -np.inf), axis=1)
    return {"mean": mean, "std": np.sqrt(finite_mean((values - mean[:, None]) ** 2, 1)),
            "min": np.where(count > 0, minimum, np.nan),
            "max": np.where(count > 0, maximum, np.nan), "valid_count": count,
            "increase_fraction": increase_fraction(values[2], 0),
            "decrease_fraction": increase_fraction(-values[2], 0)}


def _cohort_summary(values):
    delta = values[2]
    return {**{name: describe(field) for name, field in zip(FIELD_NAMES, values)},
            "increase_count": int((delta > 0).sum()), "decrease_count": int((delta < 0).sum()),
            "equal_count": int((delta == 0).sum()), "valid_count": int(np.isfinite(delta).sum()),
            "per_input_increase_fraction": describe(increase_fraction(delta, (1, 2)))}


def run_analysis(bundle, reference_bundle, output_dir, *, sample_index,
                 bootstrap_replicates=2000, seed=20261112):
    """Question B: report the 94 validation inputs separately from six Pilot inputs."""
    bundle, reference_bundle, output_dir = map(Path, (bundle, reference_bundle, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    source, ranks, layers, heads = load_profiles(bundle, RANK_SPECS)
    reference, ref_ranks, ref_layers, ref_heads = load_profiles(reference_bundle, RANK_SPECS)
    if not (np.array_equal(layers, ref_layers) and np.array_equal(heads, ref_heads)):
        raise ValueError("reference and extension Layer/KV Head coordinates differ")
    if source.get("definitions") != reference.get("definitions"):
        raise ValueError("reference and extension metric definitions differ")
    ids = reference["sample_ids"] + source["sample_ids"]
    if len(ids) != len(set(ids)):
        raise ValueError("reference and extension sample IDs overlap")
    index = {row["sample_id"]: row for row in read_json(sample_index)["samples"]}
    records = [index[sid] for sid in ids]
    for row, sample in zip(records, reference["samples"] + source["samples"]):
        if any(row[key] != sample[key] for key in ("dataset", "pair_id", "prompt_length")):
            raise ValueError("sample index and V2 sample metadata differ")
    datasets = np.asarray([row["dataset"] for row in records])
    lengths = np.asarray([row["target_prompt_length"] for row in records])
    conditions = np.asarray([f"{dataset}_{length}" for dataset, length in zip(datasets, lengths)])
    groups = np.asarray([f"pg19:{row['pair_id']}" if row["dataset"] == "pg19"
                         else f"{row['dataset']}:{row['sample_id']}" for row in records])
    ref_count = ref_ranks.shape[1]
    is_reference = np.arange(len(ids)) < ref_count
    if set(groups[is_reference]) & set(groups[~is_reference]):
        raise ValueError("reference and extension source groups overlap")
    values = rank_changes(*np.concatenate((ref_ranks, ranks), axis=1))
    paired, pair_names = pg19_pairs(datasets, groups, lengths)
    coordinates = {"sample_ids": np.asarray(ids), "layer_ids": layers, "kv_head_ids": heads}
    result = {"schema_version": "headkv-v2-rope-stability-b-v1", "status": "running",
        "source_bundle": str(bundle.resolve()), "reference_bundle": str(reference_bundle.resolve()),
        "sample_index": str(Path(sample_index).resolve()), "sample_ids": ids,
        "reference_sample_ids": reference["sample_ids"], "extension_sample_ids": source["sample_ids"],
        "seed": seed, "bootstrap_replicates": bootstrap_replicates,
        "definitions": {
            "scope": "K_pre and K_post effective ranks on the same 448 middle Token positions, all layers/physical KV Heads",
            "rank_delta": "rank_post-rank_pre; ranks read losslessly from the published FP32 package into FP64",
            "rank_ratio": "rank_post/rank_pre; finite positive pre/post ranks required",
            "increase": "strict delta>0, decrease delta<0, equal delta==0; invalid coordinates excluded from denominator",
            "reference": "six Pilot mean input-layer-centered delta/ratio profiles; never updated by extension inputs",
            "sorting": "Spearman across Heads within each Layer; pooled layer-centered correlation is supplementary",
            "std": "descriptive across-input standard deviation, ddof=0; per-Head summaries retain valid counts",
            "bootstrap": "reuse A source/condition draws, 2000 by default; PG19 lengths share Pair draws; fixed Pilot not resampled",
            "weighting": "source-equal within condition, six conditions equal; input pairs and Heads are not independent observations",
            "length": "long-short; PG19 paired by source, other tasks unpaired condition comparisons; not a RoPE-only intervention",
        }, "cohorts": [], "fixed_reference": {}, "input_pair_groups": {}, "exceptions": []}
    output_dir.mkdir(parents=True)
    write_json(output_dir / "summary.json", result)
    write_npz(output_dir / "profiles.npz", {**coordinates, "field_names": np.asarray(FIELD_NAMES),
        "values": values, "datasets": datasets, "target_lengths": lengths, "conditions": conditions,
        "source_groups": groups, "is_reference": is_reference})

    cohort_masks = {"pilot": is_reference, "extension": ~is_reference, "all_inputs": np.ones(len(ids), bool)}
    for population, mask in (("extension", ~is_reference), ("all", np.ones(len(ids), bool))):
        cohort_masks.update({f"{population}_{label}": mask & (conditions == label) for label in sorted(set(conditions))})
    statistics = []
    for name, mask in cohort_masks.items():
        stats = _head_statistics(values[:, mask])
        statistics.append(stats)
        result["cohorts"].append({"name": name, "sample_count": int(mask.sum()),
            **_cohort_summary(values[:, mask]),
            "always_increase_coordinates": int(((stats["increase_fraction"] == 1) & (stats["valid_count"][2] == mask.sum())).sum()),
            "always_decrease_coordinates": int(((stats["decrease_fraction"] == 1) & (stats["valid_count"][2] == mask.sum())).sum())})
    write_npz(output_dir / "cohorts.npz", {"cohort_names": np.asarray(list(cohort_masks)),
        "field_names": np.asarray(FIELD_NAMES), "layer_ids": layers, "kv_head_ids": heads,
        "sample_masks": np.stack(list(cohort_masks.values())),
        **{key: np.stack([row[key] for row in statistics]) for key in statistics[0]}})

    fixed, pairs = [], []
    for name, field in zip(CHANGE_NAMES, values[2:]):
        comparison = fixed_reference_comparison(field[is_reference], field[~is_reference])
        fixed.append(comparison)
        pairs.append(input_pair_correlations(field))
        result["fixed_reference"][name] = {
            "mean_layer_spearman": describe(finite_mean(comparison["per_layer_rho"], -1)),
            "per_layer_spearman": describe(comparison["per_layer_rho"]),
            "layer_centered_spearman": describe(comparison["rho"]),
            "relative_profile_skill": describe(comparison["skill"])}
    write_npz(output_dir / "fixed_reference.npz", {"field_names": np.asarray(CHANGE_NAMES),
        "sample_ids": np.asarray(source["sample_ids"]), "reference_sample_ids": np.asarray(reference["sample_ids"]),
        "layer_ids": layers, "kv_head_ids": heads,
        **{key: np.stack([row[key] for row in fixed]) for key in fixed[0]}})
    i, j = pairs[0]["pair_indices"].T
    extended_pair = ~is_reference[i] & ~is_reference[j]
    pair_masks = {"extension_all": extended_pair,
        "extension_same_condition": extended_pair & (conditions[i] == conditions[j]),
        "extension_cross_task": extended_pair & (datasets[i] != datasets[j]),
        "extension_same_task_different_length": extended_pair & (datasets[i] == datasets[j]) & (conditions[i] != conditions[j]),
        "extension_pg19_paired": extended_pair & (datasets[i] == "pg19") & (groups[i] == groups[j]),
        "pilot_all": is_reference[i] & is_reference[j], "all_inputs": np.ones(len(i), bool)}
    for label, mask in pair_masks.items():
        result["input_pair_groups"][label] = {"input_pair_count": int(mask.sum()),
            **{name: {"per_layer_spearman": describe(row["per_layer_rho"][mask]),
                      "layer_centered_spearman": describe(row["rho"][mask])}
               for name, row in zip(CHANGE_NAMES, pairs)}}
    write_npz(output_dir / "input_pairs.npz", {**coordinates, "field_names": np.asarray(CHANGE_NAMES),
        **{key: pairs[0][key] if key == "pair_indices" else np.stack([row[key] for row in pairs]) for key in pairs[0]},
        **{f"group_{key}": mask for key, mask in pair_masks.items()}})

    metric_names = [f"mean_{name}" for name in FIELD_NAMES] + ["increase_fraction"]
    scores = np.concatenate((finite_mean(values[:, ~is_reference], (2, 3)).T,
        increase_fraction(values[2, ~is_reference], (1, 2))[:, None]), axis=1)
    metric_names.extend(f"mean_layer_spearman_{name}" for name in CHANGE_NAMES)
    scores = np.column_stack((scores, *(finite_mean(row["per_layer_rho"], -1) for row in fixed)))
    bootstrap = reference_bootstrap(scores[:, None, :], conditions[~is_reference], datasets[~is_reference],
        groups[~is_reference], replicates=bootstrap_replicates, seed=seed)
    # Conditions use the same resampled PG19 sources, so their difference is paired.
    contrasts, contrast_names, contrast_points = [], [], []
    for dataset in sorted(set(datasets)):
        names = sorted(set(conditions[datasets == dataset]), key=lambda name: int(name.rsplit("_", 1)[1]))
        if len(names) == 2:
            lo, hi = [bootstrap["condition_names"].tolist().index(name) for name in names]
            contrasts.append(bootstrap["condition_bootstrap_means"][:, hi, 0] - bootstrap["condition_bootstrap_means"][:, lo, 0])
            contrast_points.append(bootstrap["condition_mean"][hi, 0] - bootstrap["condition_mean"][lo, 0])
            contrast_names.append(f"{names[1]}_minus_{names[0]}")
    contrast_draws = np.stack(contrasts, axis=1)
    contrast_ci, contrast_valid = percentile_ci(contrast_draws)
    bootstrap.update(length_contrast_names=np.asarray(contrast_names), length_contrast_mean=np.asarray(contrast_points),
        length_contrast_ci95=contrast_ci, length_contrast_valid_replicates=contrast_valid,
        length_contrast_bootstrap_means=contrast_draws)
    write_npz(output_dir / "bootstrap.npz", {"metric_names": np.asarray(metric_names), **bootstrap})
    result["bootstrap"] = {"metric_names": metric_names, **{key: value for key, value in bootstrap.items()
        if key.startswith(("condition_", "equal_condition_", "length_contrast_")) and "bootstrap_means" not in key}}

    short, long = paired.T
    differences = values[:, long] - values[:, short]
    write_npz(output_dir / "pg19_pairs.npz", {"field_names": np.asarray(FIELD_NAMES), "pair_names": pair_names,
        "sample_ids": np.asarray(ids), "pair_indices": paired, "is_reference_pair": is_reference[short],
        "layer_ids": layers, "kv_head_ids": heads, "long_minus_short": differences})
    result["pg19_paired_lengths"] = {label: {"pair_count": int(mask.sum()),
        **{name: describe(finite_mean(field[mask], (1, 2))) for name, field in zip(FIELD_NAMES, differences)}}
        for label, mask in (("extension", ~is_reference[short]), ("pilot", is_reference[short]),
                            ("all", np.ones(len(paired), bool)))}
    for layer, head in ((25, 5), (29, 6)):
        if layer in layers and head in heads:
            li, hi = layers.tolist().index(layer), heads.tolist().index(head)
            result["exceptions"].append({"layer": layer, "kv_head": head,
                "cohorts": {name: _cohort_summary(values[:, mask, li:li + 1, hi:hi + 1])
                            for name, mask in cohort_masks.items()}})
    result = _json_value(result)
    result["status"] = "completed"
    write_json(output_dir / "summary.json", result, overwrite=True)
    return result
