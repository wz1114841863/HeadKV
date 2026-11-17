"""Small, Calib-only K_pre pair-sharing study; never changes model inference."""
from itertools import combinations
from pathlib import Path
import struct

import numpy as np
import torch

from experiments.head_analysis.analysis.kv_representation import _json_value
from experiments.head_analysis.analysis.kv_representation_budget import byte_ledger
from experiments.head_analysis.analysis.pca_reconstruction import (
    apply_position_rope, condition_weights, fit_basis, fit_calibration,
    projection_errors, rope_coefficients,
)
from experiments.head_analysis.analysis.pca_subspace import ERROR_ATOL, check_nonnegative
from experiments.head_analysis.analysis.pre_rope_head_budget import gain
from experiments.head_analysis.common.io import atomic_file, read_json, write_json, write_npz
from experiments.head_analysis.common.statistics import correlation_1d
from experiments.head_analysis.runtime.key_codec import decode_key, encode_key
from experiments.head_analysis.storage.representation_inputs import load_codec_cohorts, read_sample


METHODS = ("O64", "I64", "S-index", "S-CKA", "S-error")
# Layout 0: independent; 1: implicit adjacent pairs; 2: explicit uint8 group map.
LAYOUTS = (0, 0, 1, 2, 2)


def _save(path, value):
    write_json(path, _json_value(value))


def perfect_matchings(heads):
    """Canonical, lexicographically ordered pairings (105 for eight Heads)."""
    heads = tuple(heads)
    if not heads:
        yield ()
    else:
        first = heads[0]
        for partner in heads[1:]:
            rest = tuple(h for h in heads[1:] if h != partner)
            for suffix in perfect_matchings(rest):
                yield ((first, partner),) + suffix


def select_matching(weights, pairs, *, maximize=False):
    """Select disjoint pairs, preserving lexicographic tie resolution."""
    heads = np.unique(pairs)
    if len(heads) % 2 or not np.isfinite(weights).all():
        raise ValueError("complete finite pair scores and an even Head count required")
    lookup = {tuple(pair): i for i, pair in enumerate(pairs)}
    choices = np.array([[lookup[p] for p in m] for m in perfect_matchings(heads)])
    scores = np.asarray(weights)[choices].sum(-1)
    return choices[int(np.argmax(scores) if maximize else np.argmin(scores))]


def pair_group_map(selected_pairs, heads):
    if not np.array_equal(np.sort(np.asarray(selected_pairs).ravel()), np.arange(heads)):
        raise ValueError("pairs must cover every Head exactly once")
    groups = np.empty(heads, dtype=np.uint8)
    for group, (left, right) in enumerate(selected_pairs):
        groups[left] = groups[right] = group
    return groups


def fit_layer_pairs(g, scale, rank):
    """Matched independent and normalized pair bases for one Layer."""
    heads, dimension = g.shape[:2]
    pairs = np.array(list(combinations(range(heads), 2)), dtype=np.int64)
    independent = fit_basis(g[None], scale[None])
    shared = fit_basis(g[pairs], scale[pairs], sharing="layer")
    if max(independent["orthogonality_error"].max(), shared["orthogonality_error"].max()) > ERROR_ATOL:
        raise ValueError("fitted basis is not orthonormal within the V3 tolerance")
    head_basis = independent["basis"][0]
    pair_basis = shared["compact_basis"]
    zero = np.zeros((heads, dimension))
    # G is the second moment about the fixed mean. A zero-mean covariance view
    # evaluates the same trace objective without losing the input-mean term.
    head_cost = projection_errors(zero, g, zero, head_basis, [rank])[..., 0] / scale
    pair_cost = projection_errors(zero[pairs], g[pairs], zero[pairs],
                                  pair_basis[:, None], [rank])[..., 0] / scale[pairs]
    member_delta = pair_cost - head_cost[pairs]
    roundoff = check_nonnegative(member_delta, "Calib pair member increase")
    return {"pairs": pairs, "head_basis": head_basis, "pair_basis": pair_basis,
            "head_cost": head_cost, "pair_cost": pair_cost,
            "member_delta": member_delta, "delta": member_delta.mean(-1),
            "roundoff_negative": roundoff}


def read_moments(records, directory, ranks):
    """Full-middle K_pre moments from the already completed codec collection."""
    means, covariances, counts, original = [], [], [], []
    for row in records:
        mu, cov, n, old = [], [], [], []
        for layer in range(row["num_layers"]):
            path = Path(directory) / "samples" / row["sample_id"] / f"prefill/layer_{layer:03d}.npz"
            with np.load(path) as z:
                if not np.array_equal(z["k_pre_count"], [row["prompt_length"], row["prompt_length"]-128]):
                    raise ValueError(f"K_pre scope counts differ: {row['sample_id']}")
                mu.append(z["k_pre_mean"][1]); cov.append(z["k_pre_covariance"][1])
                n.append(z["k_pre_count"][1])
                indices = [list(z["ranks"]).index(int(r)) for r in ranks]
                old.append(z["fixed_ideal_in_domain_mse"][0][..., indices])
        means.append(mu); covariances.append(cov); counts.append(n); original.append(old)
        print(f"K_pre moments: {row['sample_id']}", flush=True)
    result = {"mean": np.asarray(means, dtype=np.float64),
              "covariance": np.asarray(covariances, dtype=np.float64),
              "count": np.asarray(counts), "original_mse": np.asarray(original)}
    if not all(np.isfinite(result[k]).all() for k in ("mean", "covariance", "original_mse")):
        raise ValueError("nonfinite K_pre moments")
    return result


def read_calib_cka(bundle, records, layers, heads):
    """Read only requested Calib files, never aggregate all-input CKA profiles."""
    source = read_json(Path(bundle) / "summary.json")
    if source["status"] != "completed":
        raise ValueError("CKA source is incomplete")
    samples = {r["sample_id"]: r for r in source["samples"]}
    pairs = np.array(list(combinations(range(heads), 2)))
    values = []
    for row in records:
        sample = samples[row["sample_id"]]
        if any(sample[k] != row[k] for k in ("dataset", "pair_id", "prompt_length")):
            raise ValueError("CKA and codec source coordinates differ")
        with np.load(Path(bundle) / sample["numeric_file"]) as z:
            if not (np.array_equal(z["layer_ids"], np.arange(layers))
                    and np.array_equal(z["kv_head_ids"], np.arange(heads))
                    and np.array_equal(z["cka_pairs"], pairs)):
                raise ValueError("CKA Layer/Head-pair axes differ")
            values.append(np.stack((z["cka_k_pre_observed"], z["cka_k_pre_excess"])))
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(records), 2, layers, len(pairs)) or not np.isfinite(values).all():
        raise ValueError("complete finite Calib CKA values required")
    return values


def fit_models(g, scale, original_basis, cka, rank):
    layers, heads, dimension = scale.shape[0], scale.shape[1], g.shape[-1]
    if heads % 2 or heads > 256:
        raise ValueError("pair sharing requires an even Head count fitting uint8 IDs")
    maps = [np.tile(np.arange(heads, dtype=np.uint8), (layers, 1)) for _ in METHODS]
    bases = [original_basis, np.empty_like(original_basis)] + [
        np.empty((layers, heads//2, dimension, dimension)) for _ in range(3)]
    diagnostics = {k: [] for k in ("head_cost", "pair_cost", "member_delta", "delta", "roundoff_negative")}
    for layer in range(layers):
        fitted = fit_layer_pairs(g[layer], scale[layer], rank)
        pairs = fitted["pairs"]
        selections = (np.array([i for i, (a, b) in enumerate(pairs) if a % 2 == 0 and b == a+1]),
                      select_matching(cka[0, layer], pairs, maximize=True),
                      select_matching(fitted["delta"], pairs))
        bases[1][layer] = fitted["head_basis"]
        for method, selected in enumerate(selections, 2):
            maps[method][layer] = pair_group_map(pairs[selected], heads)
            bases[method][layer] = fitted["pair_basis"][selected]
        for key in diagnostics:
            diagnostics[key].append(fitted[key])
    diagnostics = {k: np.stack(v) for k, v in diagnostics.items()}
    diagnostics.update(pairs=pairs, layer_ids=np.arange(layers), kv_head_ids=np.arange(heads),
        cka_kinds=np.array(["observed", "excess"]), cka_mean=cka,
        cka_delta_spearman=np.array([[correlation_1d(cka[k, l], diagnostics["delta"][l])[1]
                                     for l in range(layers)] for k in range(2)]))
    return bases, np.stack(maps), diagnostics


def export_parameters(path, mean, basis, groups, rank, layout):
    """Store a shared basis once per group, not once per member Head."""
    layers, heads, dimension = mean.shape
    with atomic_file(path, binary=True) as handle:
        handle.write(struct.pack("<HHHBBH", layers, heads, dimension, 0, layout, rank))
        if layout == 2:
            handle.write(groups.astype(np.uint8).tobytes())
        handle.write(mean.astype("<f4").tobytes())
        handle.write(basis[..., :rank].astype("<f4").tobytes())


def read_parameters(path):
    """Read the compact candidate payload used by the BF16 reference codec."""
    blob = Path(path).read_bytes()
    layers, heads, dimension, position, layout, rank = struct.unpack("<HHHBBH", blob[:10])
    if position != 0 or layout not in (0, 1, 2):
        raise ValueError("not a supported Pre-RoPE sharing payload")
    offset = 10
    if layout == 2:
        groups = np.frombuffer(blob[offset:offset+layers*heads], np.uint8).reshape(layers, heads).copy()
        offset += layers*heads
    else:
        groups = np.tile(np.arange(heads, dtype=np.uint8) // (2 if layout == 1 else 1), (layers, 1))
    group_count = heads if layout == 0 else heads//2
    expected = offset + 4*layers*heads*dimension + 4*layers*group_count*dimension*rank
    if len(blob) != expected or (groups >= group_count).any():
        raise ValueError("sharing payload size/group map differs from its layout")
    size = 4*layers*heads*dimension
    mean = np.frombuffer(blob[offset:offset+size], "<f4").reshape(layers, heads, dimension).copy()
    basis = np.frombuffer(blob[offset+size:], "<f4").reshape(layers, group_count, dimension, rank).copy()
    return {"mean": mean, "basis": basis, "groups": groups, "rank": rank}


def model_ledger(mean, basis, layout, rank, lengths):
    layers, heads, dimension = mean.shape
    item = byte_ledger(np.full((layers, heads), rank), lengths, dimension=dimension,
                       request_metadata_bytes=8)
    old_total = item["model_total_bytes"]
    item["strategy"] = "per_head" if layout == 0 else "pair"
    item["model_basis_bytes"] = 4 * int(np.prod(basis.shape[:-1])) * rank
    item["model_group_map_bytes"] = layers*heads if layout == 2 else 0
    item["model_total_bytes"] = (item["model_basis_bytes"] + item["model_mean_bytes"]
        + item["model_rank_metadata_bytes"] + item["model_layout_metadata_bytes"] + item["model_group_map_bytes"])
    item["resident_identical_requests_bytes"] += item["model_total_bytes"] - old_total
    value = 2*np.asarray(lengths, dtype=np.int64)*layers*heads*dimension
    item.update(dense_value_bytes=value,
        kv_total_resident_bytes=item["resident_identical_requests_bytes"] + item["request_counts"][:, None]*value[None],
        one_layer_bf16_restore_bytes=2*np.asarray(lengths)*heads*dimension,
        optional_rope_table_bytes=4*np.asarray(lengths)*dimension,
        projection_flops_per_token=2*layers*heads*dimension*rank)
    return item


def full_errors(data, mean, bases, groups, scale, ranks):
    layers = mean.shape[0]
    errors = []
    for basis, mapping in zip(bases, groups):
        expanded = basis[np.arange(layers)[:, None], mapping]
        errors.append(projection_errors(data["mean"], data["covariance"], mean, expanded, ranks))
    raw = np.stack(errors, 1)  # Sample,Method,Layer,Head,Rank
    normalized = raw / scale[None, None, ..., None]
    if np.max(normalized[..., -1]) > ERROR_ATOL or (np.diff(normalized, axis=-1) > ERROR_ATOL).any():
        raise ValueError("full-rank or nested projection check failed")
    np.testing.assert_allclose(normalized[:, 0], data["original_mse"] / scale[None, ..., None],
                               atol=ERROR_ATOL, rtol=0)
    return {"raw_mse": raw, "normalized_mse": normalized, "count": data["count"],
            "energy": np.trace(data["covariance"], axis1=-2, axis2=-1) + np.square(data["mean"]).sum(-1)}


def sampled_errors(records, mean, bases, groups, deployed, scale, ranks, config):
    shape = (len(records), len(METHODS), *scale.shape, len(ranks))
    result = {k: np.empty(shape) for k in ("ideal_pre", "ideal_post", "bf16_post")}
    result["coefficient_bytes"] = np.empty(shape, dtype=np.uint64)
    floor = np.empty((len(records), 2, *scale.shape))
    for sample, row in enumerate(records):
        for layer in range(mean.shape[0]):
            raw = read_sample(row, include_vectors=True, layer_ids=[layer])
            x, target = raw["vectors"][:2, 0].astype(np.float64)
            if layer == 0:
                cos, sin = rope_coefficients(config, raw["positions"], x.shape[-1])
            floor[sample, 0, layer] = np.square(apply_position_rope(x, cos, sin)-target).sum(-1).mean(-1)
            floor[sample, 1, layer] = np.square(apply_position_rope(x, cos, sin, bf16=True)-target).sum(-1).mean(-1)
            sample_mean = x.mean(-2)
            centered = x-sample_mean[:, None]
            covariance = centered.swapaxes(-1, -2) @ centered / x.shape[-2]
            for method, basis in enumerate(bases):
                u = basis[layer, groups[method, layer]]
                analytic = projection_errors(sample_mean, covariance, mean[layer], u, ranks)
                for ri, rank in enumerate(ranks):
                    ideal = ((x-mean[layer, :, None]) @ u[..., :rank]) @ u[..., :rank].swapaxes(-1, -2) + mean[layer, :, None]
                    pre_error = np.square(ideal-x).sum(-1).mean(-1)
                    np.testing.assert_allclose(pre_error/scale[layer], analytic[:, ri]/scale[layer], atol=ERROR_ATOL, rtol=0)
                    if ri == 0:
                        model = deployed[method]
                        actual_mean = model["mean"][layer]
                        actual_basis = model["basis"][layer, model["groups"][layer]]
                    else:  # full-rank runtime diagnostic, not another stored candidate
                        actual_mean, actual_basis = mean[layer].astype(np.float32), u.astype(np.float32)
                    with torch.no_grad():
                        code = encode_key(torch.from_numpy(x), torch.from_numpy(actual_mean), torch.from_numpy(actual_basis), int(rank))
                        actual = decode_key(code, torch.from_numpy(actual_mean), torch.from_numpy(actual_basis)).float().numpy()
                    index = sample, method, layer, slice(None), ri
                    result["ideal_pre"][index] = pre_error
                    result["ideal_post"][index] = np.square(apply_position_rope(ideal, cos, sin)-target).sum(-1).mean(-1)
                    result["bf16_post"][index] = np.square(apply_position_rope(actual, cos, sin, bf16=True)-target).sum(-1).mean(-1)
                    byte_count = code[0].numel()*code.element_size()
                    if byte_count != 2*448*int(rank):
                        raise ValueError("sample coefficient byte count differs")
                    result["coefficient_bytes"][index] = byte_count
        print(f"Sharing BF16 reference: {row['sample_id']}", flush=True)
    for key in ("ideal_pre", "ideal_post", "bf16_post"):
        result[key] /= scale[None, None, ..., None]
        if not np.isfinite(result[key]).all():
            raise ValueError("nonfinite shared codec error")
    if result["ideal_pre"][..., -1].max() > ERROR_ATOL:
        raise ValueError("sample full-rank Pre reconstruction failed")
    result["transform_only_normalized_mse"] = floor/scale[None, None]
    return result


def summarize(values, records):
    """Main-rank scores; all increases use I64 as the matched denominator."""
    weights = condition_weights([r["condition"] for r in records], [r["source_group"] for r in records])
    input_mean = values.mean((2, 3))
    mean = weights @ input_mean
    p90 = weights @ np.quantile(values.reshape(len(records), len(METHODS), -1), .9, axis=-1)
    labels = sorted({r["condition"] for r in records})
    conditions = {}
    for label in labels:
        mask = np.array([r["condition"] == label for r in records])
        conditions[label] = weights[mask] @ input_mean[mask] / weights[mask].sum()
    worst = {}
    for m, method in enumerate(METHODS):
        s, l, h = np.unravel_index(np.argmax(values[:, m]), values[:, m].shape)
        increase = -gain(values[:, 1], values[:, m])
        valid = np.isfinite(increase)
        relative = None
        if valid.any():
            a, b, c = np.unravel_index(np.nanargmax(increase), increase.shape)
            relative = {"sample_id": records[a]["sample_id"], "layer": b, "head": c,
                        "increase": increase[a, b, c], "I64_mse": values[a, 1, b, c],
                        "candidate_mse": values[a, m, b, c]}
        worst[method] = {"sample_id": records[s]["sample_id"], "layer": l, "head": h,
                        "normalized_mse": values[s, m, l, h], "largest_relative_increase": relative,
                        "zero_baseline_positive_count": int(((values[:, 1] == 0) & (values[:, m] > 0)).sum())}
    return {"mean": mean, "input_mean": input_mean, "p90": p90, "condition_mean": conditions,
            "increase_over_I64": -gain(mean[1], mean), "p90_increase_over_I64": -gain(p90[1], p90),
            "I64_over_O64_increase": -gain(mean[0], mean[1]), "worst": worst}


def passes_error_limits(scores):
    def within(values, factor):
        values = np.asarray(values)
        return values <= factor*values[1] if values[1] > 0 else values <= ERROR_ATOL
    return within(scores["mean"], 1.10) & np.all([within(v, 1.25) for v in scores["condition_mean"].values()], axis=0)


def run_analysis(bundle, codec_root, cka_bundle, output, *, rank=64):
    bundle, codec_root, cka_bundle, output = map(Path, (bundle, codec_root, cka_bundle, output))
    calib_records, validation_records = load_codec_cohorts(bundle, codec_root)
    with np.load(bundle / "rope_calibration.npz") as z:
        mean, original_basis = z["mean"][0], z["basis"][0]
    with np.load(bundle / "calibration.npz") as z:
        scale = z["scale"][0]
    if not np.isfinite(scale).all() or not (scale > 0).all() or not 0 < rank < mean.shape[-1]:
        raise ValueError("positive frozen scales and a reduced main rank required")
    ranks = np.array([rank, mean.shape[-1]], dtype=np.int64)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "running", "source_bundle": str(bundle.resolve()),
        "codec_root": str(codec_root.resolve()), "cka_bundle": str(cka_bundle.resolve()),
        "calib": calib_records, "validation": validation_records, "methods": METHODS,
        "ranks": ranks, "error_axes": ["sample", "method", "layer", "head", "rank"],
        "mean_scale_refitted": False, "basis_fit_scope": "Calib6 full middle about original frozen means",
        "full_scope": "own-domain ideal [64,T-64)", "sample_scope": "same 448 rows; final BF16 K_post",
        "full_rank_role": "runtime numerical diagnostic; stored parameter candidates use only main rank",
        "exploratory": True, "bootstrap": False, "inference_path_modified": False}
    _save(output / "metadata.json", metadata)
    cka_values = read_calib_cka(cka_bundle, calib_records, *scale.shape)
    calib = read_moments(calib_records, codec_root / "pilot", ranks)
    fit = fit_calibration(calib["mean"], calib["covariance"], [r["condition"] for r in calib_records],
                          [r["source_group"] for r in calib_records], fixed_mean=mean)
    cka = np.einsum("s,sklp->klp", fit["weights"], cka_values)
    bases, groups, diagnostics = fit_models(fit["G"], scale, original_basis, cka, rank)
    calib_full = full_errors(calib, mean, bases, groups, scale, ranks)
    write_npz(output / "pair_diagnostics.npz", {**diagnostics, "calib_cka": cka_values})
    write_npz(output / "fit.npz", {"mean": mean, "scale": scale, "weights": fit["weights"],
        **{f"{name}_basis": basis[..., :rank] for name, basis in zip(METHODS[1:], bases[1:])}})
    ledger, deployed = {}, []
    records = calib_records + validation_records
    for method, basis, mapping, layout in zip(METHODS, bases, groups, LAYOUTS):
        item = model_ledger(mean, basis, layout, rank, [r["prompt_length"] for r in records])
        path = output / f"{method}_parameters.bin"
        export_parameters(path, mean, basis, mapping, rank, layout)
        model = read_parameters(path)
        np.testing.assert_array_equal(model["mean"], mean.astype(np.float32))
        np.testing.assert_array_equal(model["basis"], basis[..., :rank].astype(np.float32))
        np.testing.assert_array_equal(model["groups"], mapping)
        if path.stat().st_size != item["model_total_bytes"]:
            raise ValueError("stored parameters differ from byte budget")
        ledger[method] = item; deployed.append(model)
    _save(output / "byte_ledger.json", ledger)
    _save(output / "freeze.json", {"status": "frozen_before_validation", "calib_sample_ids": [r["sample_id"] for r in calib_records],
        "methods": METHODS, "group_maps": groups, "layouts": LAYOUTS, "rank": rank,
        "mean_increase_limit": .10, "condition_increase_limit": .25})
    # No Validation moments, vectors or CKA values were read before this point.
    validation = read_moments(validation_records, codec_root / "validation", ranks)
    summaries = {}
    for phase, data, selected in (("calib", calib, calib_records), ("validation", validation, validation_records)):
        full = calib_full if phase == "calib" else full_errors(data, mean, bases, groups, scale, ranks)
        sample = sampled_errors(selected, mean, bases, groups, deployed, scale, ranks, read_json(bundle / "rope_config.json"))
        coordinates = {"sample_ids": np.array([r["sample_id"] for r in selected]), "methods": np.array(METHODS), "ranks": ranks}
        write_npz(output / f"{phase}_full.npz", {**coordinates, **full})
        write_npz(output / f"{phase}_sampled.npz", {**coordinates, **sample})
        summaries[phase] = {"full_ideal": summarize(full["normalized_mse"][..., 0], selected),
            "sampled_ideal_post": summarize(sample["ideal_post"][..., 0], selected),
            "sampled_bf16_post": summarize(sample["bf16_post"][..., 0], selected)}
    geometry = passes_error_limits(summaries["validation"]["full_ideal"])
    actual = passes_error_limits(summaries["validation"]["sampled_bf16_post"])
    decisions = {m: {"full_ideal_limits_met": geometry[i], "sampled_bf16_limits_met": actual[i],
        "descriptive_candidate": geometry[i] and actual[i] and ledger[m]["model_total_bytes"] < ledger["I64"]["model_total_bytes"]}
        for i, m in enumerate(METHODS) if i >= 2}
    high = diagnostics["cka_mean"][0].argmax(-1)
    layers = np.arange(len(high))
    examples = sorted(layers, key=lambda l: -diagnostics["delta"][l, high[l]])[:5]
    result = {"status": "completed", "methods": METHODS, "phases": summaries, "decisions": decisions,
        "cka_delta_spearman": diagnostics["cka_delta_spearman"], "exploratory": True,
        "highest_cka_pair_examples": [{"layer": l, "heads": diagnostics["pairs"][high[l]],
            "observed_cka": diagnostics["cka_mean"][0, l, high[l]],
            "pair_delta": diagnostics["delta"][l, high[l]],
            "member_delta": diagnostics["member_delta"][l, high[l]]} for l in examples],
        "no_confirmatory_ci": True, "full_bf16_or_decode_quality_collected": False,
        "inference_path_modified": False}
    _save(output / "summary.json", result)
    metadata["status"] = "completed"
    write_json(output / "metadata.json", _json_value(metadata), overwrite=True)
    print(f"Pre-RoPE pair-sharing analysis completed: {output}", flush=True)
    return _json_value(result)
