"""V2 representation spectra, aligned-token CKA and descriptive B1 relations."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence
import json

import numpy as np

from experiments.head_analysis.analysis.kv_representation_metrics import (
    cka_statistics, head_cka_means, sampled_statistics, spectrum_metrics,
)
from experiments.head_analysis.common.io import atomic_file, read_json, write_json, write_npz
from experiments.head_analysis.common.statistics import correlation_1d, finite_mean
from experiments.head_analysis.storage.kv_representation import read_record


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    result = {"valid_count": int(valid.size), "total_count": int(values.size)}
    if valid.size:
        result.update(zip(("min", "p10", "median", "p90", "max"),
                          np.quantile(valid, [0, .1, .5, .9, 1], method="linear").tolist()))
        result["mean"] = float(valid.mean())
    else:
        result.update({key: None for key in ("min", "p10", "median", "p90", "max", "mean")})
    return result


def correlate_heads(left, right):
    """Pairwise support, then subtract each Layer's Head mean before ranking."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Head relations require matching [Layer,KVHead] arrays")
    valid = np.isfinite(left) & np.isfinite(right)
    x, y = np.where(valid, left, np.nan), np.where(valid, right, np.nan)
    count, raw, _ = correlation_1d(x.ravel(), y.ravel())
    xc, yc = x - finite_mean(x, -1)[:, None], y - finite_mean(y, -1)[:, None]
    centered_count, centered, _ = correlation_1d(xc.ravel(), yc.ravel())
    per_layer = [correlation_1d(a, b) for a, b in zip(x, y)]
    return {"valid_head_count": count, "spearman": raw,
            "layer_centered_valid_head_count": centered_count,
            "layer_centered_spearman": centered,
            "per_layer_valid_head_count": [row[0] for row in per_layer],
            "per_layer_spearman": [row[1] for row in per_layer]}


def load_b1_profiles(bundle):
    bundle = Path(bundle)
    metadata = read_json(bundle / "analysis_manifest.json")
    if metadata["status"] != "completed":
        raise ValueError("B1 numeric bundle is incomplete")
    required = ("sample_ids", "page_scopes", "rco_k_values", "rco_deltas", "rco_metrics",
                "rco_sample_head_values", "rco_sample_head_valid_pair_count")
    optional = ("token_k_values", "token_deltas", "token_metrics", "token_sample_head_metrics",
                "token_sample_head_valid_pair_count")
    with np.load(bundle / "head_profiles.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in (*required, *optional) if key in archive.files}
    if any(key not in arrays for key in required):
        raise ValueError("B1 bundle is missing Page RCO fields")
    ids = arrays["sample_ids"].tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate B1 sample IDs")
    return arrays, metadata


def _page_rco(profiles, sample_id):
    row = profiles["sample_ids"].tolist().index(sample_id)
    scope = profiles["page_scopes"].tolist().index("prompt_only")
    k = profiles["rco_k_values"].tolist().index(32)
    delta = profiles["rco_deltas"].tolist().index(32)
    metric = profiles["rco_metrics"].tolist().index("rco_raw_mean")
    count = profiles["rco_sample_head_valid_pair_count"][row, :, :, scope, k, delta]
    values = profiles["rco_sample_head_values"][row, :, :, scope, k, delta, metric]
    return np.where(count > 0, values, np.nan), count


def _ruler_token_rco(profiles, sample_id, heads):
    if "token_sample_head_metrics" not in profiles:
        return {}
    row = profiles["sample_ids"].tolist().index(sample_id)
    k = profiles["token_k_values"].tolist().index(32)
    delta = profiles["token_deltas"].tolist().index(1)
    metric = profiles["token_metrics"].tolist().index("rco_raw_mean")
    count = profiles["token_sample_head_valid_pair_count"][row, :, :, k, delta]
    values = profiles["token_sample_head_metrics"][row, :, :, k, delta, metric]
    if values.shape[1] != heads * 4:
        raise ValueError("B1 Token profiles must use the frozen contiguous 4:1 GQA mapping")
    values = np.where(count > 0, values, np.nan)
    return {"b1_token_rco_query_head_values": values,
            "b1_token_rco_query_head_valid_pair_count": count,
            "b1_token_rco_kv_head_mean": finite_mean(values.reshape(values.shape[0], heads, 4), -1)}


def _representation_metrics(statistics):
    metrics = spectrum_metrics(statistics["covariance"])
    energy = np.asarray(statistics["feature_energy"], dtype=np.float64)
    total = energy.sum(-1)
    normalized = np.full_like(energy, np.nan)
    np.divide(energy, total[..., None], out=normalized, where=total[..., None] > 0)
    fraction = np.full(total.shape, np.nan)
    np.divide(np.square(np.asarray(statistics["mean"], dtype=np.float64)).sum(-1), total,
              out=fraction, where=total > 0)
    return {**metrics, "count": np.asarray(statistics["count"], dtype=np.uint32),
            "normalized_feature_energy": normalized,
            "coordinate_energy_top16": np.sort(normalized, axis=-1)[..., -min(16, energy.shape[-1]):].sum(-1),
            "mean_energy_fraction": fraction, "norm_summary": statistics["norm_summary"],
            "element_min": statistics["element_min"], "element_max": statistics["element_max"]}


def _profile_summary(arrays, prefix):
    return {"count": arrays[f"{prefix}_count"].tolist(),
            "effective_rank": describe(arrays[f"{prefix}_effective_rank"]),
            "E16": describe(arrays[f"{prefix}_variance_coverage"][..., 1]),
            "r90": describe(arrays[f"{prefix}_r90"]),
            "mean_norm": describe(arrays[f"{prefix}_norm_summary"][..., 0]),
            "coordinate_energy_top16": describe(arrays[f"{prefix}_coordinate_energy_top16"]),
            "mean_energy_fraction": describe(arrays[f"{prefix}_mean_energy_fraction"])}


def analyze_sample(directory, profiles, permutations):
    directory = Path(directory)
    metadata, manifest = read_json(directory / "sample_metadata.json"), read_json(directory / "metadata.json")
    if metadata["status"] != "completed" or manifest["status"] != "completed":
        raise ValueError(f"incomplete V2 sample: {directory}")
    layers = manifest["num_layers"]
    records = defaultdict(list)
    positions = regions = pairs = None
    for layer in range(layers):
        record = read_record(directory, f"prefill/layer_{layer:03d}.npz")
        if positions is None:
            positions, regions = record["sample_positions"], record["sample_regions"]
        elif not (np.array_equal(positions, record["sample_positions"]) and np.array_equal(regions, record["sample_regions"])):
            raise ValueError("sample positions differ across Layers")
        middle = record["sample_regions"] == 1
        if int(middle.sum()) != permutations.shape[1]:
            raise ValueError("CKA permutations and middle sample count differ")
        for representation in ("k_post", "v"):
            for si, scope in enumerate(("prompt", "middle")):
                statistics = {name: record[f"{representation}_{name}"][si]
                              for name in ("count", "mean", "covariance", "feature_energy", "norm_summary",
                                           "element_min", "element_max")}
                for key, value in _representation_metrics(statistics).items():
                    records[f"full_{representation}_{scope}_{key}"].append(value)
        for representation in ("k_pre", "k_post", "v"):
            vectors = record[representation].float().numpy().astype(np.float64)
            for scope, selected in (("all", vectors), ("middle", vectors[:, middle])):
                for key, value in _representation_metrics(sampled_statistics(selected)).items():
                    records[f"sample_{representation}_{scope}_{key}"].append(value)
            cka = cka_statistics(vectors[:, middle], permutations)
            pairs = cka["pairs"]
            for key, value in cka.items():
                if key != "pairs":
                    records[f"cka_{representation}_{key}"].append(value)
            for key, value in head_cka_means(cka, vectors.shape[0]).items():
                records[f"cka_{representation}_head_{key}"].append(value)
        if (layer + 1) % 8 == 0:
            print(json.dumps({"event": "v2_analysis_layers", "sample_id": metadata["sample_id"],
                              "layers_completed": layer + 1, "layers": layers}), flush=True)
    arrays = {key: np.stack(value) for key, value in records.items()}
    heads = record["k_post"].shape[0]
    arrays.update(layer_ids=np.arange(layers, dtype=np.uint32), kv_head_ids=np.arange(heads, dtype=np.uint32),
                  cka_pairs=pairs, sample_positions=positions, sample_regions=regions)
    summary = {"sample_id": metadata["sample_id"], "dataset": metadata["dataset"],
               "pair_id": metadata.get("pair_id"), "prompt_length": metadata["prompt_length"],
               "source_directory": str(directory.resolve()), "full": {}, "sampled": {}, "cka": {},
               "sampling_representativeness": {}, "rope_comparison": {}}
    for representation in ("k_post", "v"):
        summary["full"][representation] = {}
        summary["sampling_representativeness"][representation] = {}
        for full_scope, sample_scope in (("prompt", "all"), ("middle", "middle")):
            full, sampled = f"full_{representation}_{full_scope}", f"sample_{representation}_{sample_scope}"
            summary["full"][representation][full_scope] = _profile_summary(arrays, full)
            rank_full, rank_sample = arrays[f"{full}_effective_rank"], arrays[f"{sampled}_effective_rank"]
            difference = np.full_like(rank_full, np.nan)
            np.divide(rank_sample - rank_full, rank_full, out=difference, where=rank_full > 0)
            energy_difference = arrays[f"{sampled}_variance_coverage"][..., 1] - arrays[f"{full}_variance_coverage"][..., 1]
            arrays[f"sampling_{representation}_{full_scope}_rank_relative_difference"] = difference
            arrays[f"sampling_{representation}_{full_scope}_E16_difference"] = energy_difference
            summary["sampling_representativeness"][representation][full_scope] = {
                "rank_relative_difference": describe(difference), "E16_difference": describe(energy_difference)}
    for representation in ("k_pre", "k_post", "v"):
        summary["sampled"][representation] = {scope: _profile_summary(arrays, f"sample_{representation}_{scope}")
                                               for scope in ("all", "middle")}
        summary["cka"][representation] = {
            key: describe(arrays[f"cka_{representation}_{key}"])
            for key in ("observed", "permutation_mean", "permutation_p05", "permutation_p95", "excess")}
        valid = arrays[f"cka_{representation}_valid"]
        above = arrays[f"cka_{representation}_observed"] > arrays[f"cka_{representation}_permutation_p95"]
        summary["cka"][representation]["fraction_above_permutation_p95"] = float(above[valid].mean()) if valid.any() else None
    for scope in ("all", "middle"):
        pre, post = f"sample_k_pre_{scope}", f"sample_k_post_{scope}"
        ratio = arrays[f"{post}_effective_rank"] / arrays[f"{pre}_effective_rank"]
        energy_difference = arrays[f"{post}_variance_coverage"][..., 1] - arrays[f"{pre}_variance_coverage"][..., 1]
        arrays[f"rope_{scope}_rank_post_pre_ratio"] = ratio
        arrays[f"rope_{scope}_E16_post_minus_pre"] = energy_difference
        summary["rope_comparison"][scope] = {"rank_post_pre_ratio": describe(ratio),
                                               "E16_post_minus_pre": describe(energy_difference)}
    rco, count = _page_rco(profiles, metadata["sample_id"])
    if rco.shape != (layers, heads):
        raise ValueError("B1/V2 Layer and physical KV Head coordinates differ")
    arrays.update(b1_page_rco_raw_mean=rco, b1_page_rco_valid_pair_count=count)
    eligible = metadata["dataset"] != "ruler"
    summary["page_rco_eligible"] = eligible
    summary["page_rco"] = describe(rco)
    summary["relations"] = []
    feature_names = []
    features = []
    for representation in ("k_post", "v"):
        for name, field in (("effective_rank", "effective_rank"), ("E16", "variance_coverage")):
            value = arrays[f"full_{representation}_middle_{field}"]
            feature_names.append(f"full_{representation}_middle_{name}")
            features.append(value[..., 1] if name == "E16" else value)
    for representation in ("k_pre", "k_post"):
        for name, field in (("effective_rank", "effective_rank"), ("E16", "variance_coverage")):
            value = arrays[f"sample_{representation}_middle_{field}"]
            feature_names.append(f"sample_{representation}_middle_{name}")
            features.append(value[..., 1] if name == "E16" else value)
    for representation in ("k_pre", "k_post", "v"):
        for name in ("observed", "excess"):
            feature_names.append(f"cka_{representation}_head_{name}")
            features.append(arrays[f"cka_{representation}_head_{name}"])
    arrays["relation_feature_names"] = np.asarray(feature_names)
    arrays["relation_feature_values"] = np.stack(features)
    for name, values in zip(feature_names, features):
        relation = correlate_heads(values, rco if eligible else np.full_like(rco, np.nan))
        summary["relations"].append({"feature": name, **relation})
    if not eligible:
        summary["page_rco_exclusion"] = "RULER has six Decode observations; delta=32 is unavailable"
        token = _ruler_token_rco(profiles, metadata["sample_id"], heads)
        arrays.update(token)
        summary["ruler_token_rco"] = (describe(token["b1_token_rco_kv_head_mean"]) if token else None)
    for key, value in arrays.items():
        if value.dtype.kind == "f" and key != "relation_feature_values":
            arrays[key] = value.astype(np.float32)
    return arrays, _json_value(summary)


def run_analysis(run_dirs: Sequence[Path], output_dir: Path, *, b1_bundle: Path,
                 permutation_replicates=50, seed=20261111) -> dict:
    output_dir, b1_bundle = Path(output_dir), Path(b1_bundle)
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    if permutation_replicates <= 0:
        raise ValueError("positive permutation count required")
    profiles, b1_metadata = load_b1_profiles(b1_bundle)
    selected = []
    run_sources = []
    for directory in map(Path, run_dirs):
        metadata = read_json(directory / "run_metadata.json")
        if metadata["status"] != "completed" or metadata["target_sample_ids"] != metadata["completed_sample_ids"]:
            raise ValueError(f"incomplete V2 run: {directory}")
        run_sources.append({"directory": str(directory.resolve()), "source_b1_run_set": metadata["source_b1_run_set"],
                            "fixed_dataset": metadata["fixed_dataset"], "model": metadata["model"],
                            "dtype": metadata["dtype"], "attention_backend": metadata["attention_backend"]})
        for key in ("model", "dtype", "attention_backend", "fixed_dataset", "source_b1_run_set"):
            if run_sources[-1][key] != run_sources[0][key]:
                raise ValueError(f"V2 runs use different collection protocols: {key}")
        selected.extend(directory / "samples" / sample_id for sample_id in metadata["completed_sample_ids"])
    ids = [read_json(directory / "sample_metadata.json")["sample_id"] for directory in selected]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("no V2 samples or duplicate sample IDs across run directories")
    if not set(ids) <= set(profiles["sample_ids"].tolist()):
        raise ValueError("V2 samples missing from B1 profiles")
    rng = np.random.default_rng(seed)
    permutations = np.stack([rng.permutation(448) for _ in range(permutation_replicates)]).astype(np.uint32)
    output_dir.mkdir(parents=True)
    definitions = {"spectrum": "centered population covariance, ddof=0; FP64 eigensolver; FP32 numeric output",
                   "relation_feature_values": "FP64 to reproduce Spearman ranking; other floating numeric arrays are FP32",
                   "variance_coverage_k": [8, 16, 32, 64], "effective_rank": "exp(entropy(eigenvalues / trace))",
                   "r90": "minimum PCA dimensions explaining at least 90% variance",
                   "zero_variance": "invalid (NaN in NPZ, null in JSON)",
                   "full_scopes": {"prompt": "[0,T)", "middle": "[64,T-64)"},
                   "sample_scopes": {"all": "same 512 positions", "middle": "same 448 middle positions"},
                   "norm_statistics": ["mean", "p10", "median", "p90", "max"],
                   "energy": "original coordinate second moments, includes mean energy; not PCA variance",
                   "cka": "feature-centered linear CKA; 448 aligned middle rows; all within-layer unordered KV Head pairs",
                   "permutations": "one side only; same saved permutations for every Head pair, representation and sample",
                   "primary_b1": {"scope": "prompt_only", "k": 32, "delta": 32, "metric": "rco_raw_mean"},
                   "head_relation": "subtract Layer means from both profiles before ranking and Spearman; no independent-Head inference",
                   "ruler_short_b1": {"scope": "all_visible_token", "k": 32, "delta": 1,
                                       "gqa": "equal finite mean of contiguous four Query Heads per KV Head"}}
    result = {"schema_version": "headkv-v2-kv-analysis-v1", "status": "running", "sources": run_sources,
              "b1_bundle": str(b1_bundle.resolve()), "b1_source_run_set_sha256": b1_metadata.get("source_run_set_sha256"),
              "sample_ids": ids, "permutation_replicates": permutation_replicates, "seed": seed,
              "definitions": definitions, "samples": []}
    write_json(output_dir / "summary.json", result)
    write_npz(output_dir / "permutations.npz", {"row_permutations": permutations})
    with atomic_file(output_dir / "sample_summary.jsonl") as handle:
        for directory in selected:
            print(json.dumps({"event": "v2_analysis_sample_started", "sample_id": directory.name}), flush=True)
            arrays, sample_summary = analyze_sample(directory, profiles, permutations)
            relative = f"samples/{directory.name}.npz"
            write_npz(output_dir / relative, arrays)
            sample_summary["numeric_file"] = relative
            result["samples"].append(sample_summary)
            handle.write(json.dumps(sample_summary, ensure_ascii=False, allow_nan=False) + "\n")
            print(json.dumps({"event": "v2_analysis_sample_completed", "sample_id": directory.name}), flush=True)
    result["status"] = "completed"
    write_json(output_dir / "summary.json", result, overwrite=True)
    return result
