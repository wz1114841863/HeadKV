"""Compose the approved, CPU-only subspace diagnostic from frozen V2/V3 data."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from experiments.head_analysis.analysis.kv_representation import _json_value
from experiments.head_analysis.analysis.pca_reconstruction import _eigh, condition_weights
from experiments.head_analysis.analysis.pca_subspace import (
    RANKS, ANGLE_ATOL_RAD, OVERLAP_ATOL, ERROR_ATOL, check_identity,
    diagnose_moments, diagnose_pairs, principal_geometry, spectral_boundary,
)
from experiments.head_analysis.common.io import read_json, write_json, write_npz
from experiments.head_analysis.common.statistics import (
    average_ranks, bootstrap_draws, finite_mean, pearson, percentile_ci,
)
from experiments.head_analysis.storage.representation_inputs import read_sample


BOOTSTRAP_REPLICATES = 2000
REPRESENTATIONS = ("k_post", "v")
POINT_FIELDS = (
    "main_overlap", "main_angle_median", "main_angle_max",
    "centered_overlap", "mean_shift_overlap", "fixed_mse", "oracle_mse", "gap",
    "fixed_normalized_mse", "oracle_normalized_mse",
    "gap_covariance", "gap_mean", "missed_top_energy", "recovered_tail_energy",
)
PEER_FIELDS = ("M_overlap", "M_angle_median", "M_angle_max",
               "C_overlap", "C_angle_median", "C_angle_max", "forward_gap", "reverse_gap")


def _save(path, value):
    write_json(path, _json_value(value))


def pair_index(records, cohort):
    pairs, descriptions = [], []
    rows = [i for i, row in enumerate(records) if row["split"] == cohort]
    for first, second in combinations(rows, 2):
        a, b = records[first], records[second]
        if (a["condition"], a["sample_id"]) > (b["condition"], b["sample_id"]):
            first, second, a, b = second, first, b, a
        if a["source_group"] == b["source_group"]:
            if a["dataset"] != "pg19" or a["pair_id"] != b["pair_id"]:
                raise ValueError("unplanned repeated source in evaluation pairs")
            kind, group = "pg19_pair", a["pair_id"]
        elif a["condition"] == b["condition"]:
            kind, group = "within", a["condition"]
        else:
            kind, group = "cross", f"{a['condition']}|{b['condition']}"
        pairs.append((first, second))
        descriptions.append({"left": first, "right": second, "left_id": a["sample_id"],
            "right_id": b["sample_id"], "left_condition": a["condition"],
            "right_condition": b["condition"], "kind": kind, "group": group})
    groups = sorted({(row["kind"], row["group"]) for row in descriptions})
    group_index = np.array([groups.index((row["kind"], row["group"])) for row in descriptions])
    return np.asarray(pairs, dtype=np.int64), descriptions, groups, group_index


def source_draw_indices(records, cohort, replicates):
    """Stratify by condition; the two PG19 lengths share the same source draws."""
    rows = [row for row in records if row["split"] == cohort]
    conditions = np.asarray([r["condition"] for r in rows])
    sources = np.asarray([r["source_group"] for r in rows])
    index = np.empty((replicates, len(rows)), dtype=np.int64)
    for condition in sorted(set(conditions)):
        positions = np.flatnonzero(conditions == condition)
        order = sorted(positions, key=lambda i: sources[i])
        if len({sources[i] for i in order}) != len(order):
            raise ValueError("each condition must have one row per frozen source")
        key = "pg19" if condition.startswith("pg19_") else condition
        draw = bootstrap_draws(20261113, f"v3-subspace:{cohort}:{key}", replicates, len(order))
        index[:, positions] = np.asarray(order)[draw]
    return index


def _correlations(x, y, conditions):
    """Input axis last; ordinary input-equal Spearman and raw-value demeaning control."""
    def spearman(a, b):
        valid = np.isfinite(a) & np.isfinite(b)
        return pearson(average_ranks(np.where(valid, a, np.nan)),
                       average_ranks(np.where(valid, b, np.nan)))
    raw, count = spearman(x, y)
    xc, yc = x.copy(), y.copy()
    for condition in np.unique(conditions):
        mask = conditions == condition
        xc[..., mask] -= finite_mean(x[..., mask], -1)[..., None]
        yc[..., mask] -= finite_mean(y[..., mask], -1)[..., None]
    centered, centered_count = spearman(xc, yc)
    return np.stack((raw, centered), -1), np.stack((count, centered_count), -1)


def bootstrap_association(records, point, output, replicates=BOOTSTRAP_REPLICATES):
    """Primary r64 only; no Calib refits, pair-level bootstrap or Head-level tests."""
    numeric, result = {}, {}
    rank_index = list(RANKS).index(64)
    for cohort in ("validation", "test"):
        selected = np.array([i for i, r in enumerate(records) if r["split"] == cohort])
        rows = [records[i] for i in selected]
        conditions = np.array([r["condition"] for r in rows])
        weights = condition_weights(conditions, [r["source_group"] for r in rows])
        indices = source_draw_indices(records, cohort, replicates)
        x = np.moveaxis(1 - point["main_overlap"][selected, ..., rank_index], 0, -1)
        y = np.moveaxis(point["gap"][selected, ..., rank_index], 0, -1)
        correlations, counts = _correlations(x, y, conditions)
        numeric[f"{cohort}_correlations"] = correlations
        numeric[f"{cohort}_correlation_input_count"] = counts
        numeric[f"{cohort}_source_draw_indices"] = indices
        numeric[f"{cohort}_sample_ids"] = np.asarray([r["sample_id"] for r in rows])
        numeric[f"{cohort}_source_groups"] = np.asarray([r["source_group"] for r in rows])
        numeric[f"{cohort}_conditions"] = conditions
        mean_metrics = ("main_overlap", "main_angle_median", "main_angle_max",
                        "gap", "gap_covariance", "gap_mean")
        means = np.stack([finite_mean(point[key][selected, ..., rank_index], (2, 3))
                          for key in mean_metrics], -1)  # Sample,Rep,Metric
        boots, rho_boot, valid_boot = [], [], []
        for start in range(0, replicates, 64):
            draw = indices[start:start + 64]
            boots.append(np.einsum("s,bsrm->brm", weights, means[draw]))
            rho, _ = _correlations(np.take(x, draw, axis=-1).transpose(3, 0, 1, 2, 4),
                                   np.take(y, draw, axis=-1).transpose(3, 0, 1, 2, 4), conditions)
            # Rep,Layer,Head,Batch,Sample -> Batch,Rep,Layer,Head,Mode above.
            rho_boot.append(finite_mean(rho, (2, 3)))
            valid_boot.append(np.isfinite(rho).sum((2, 3)))
            print(f"Subspace bootstrap {cohort}: {min(start+64, replicates)}/{replicates}", flush=True)
        boots, rho_boot, valid_boot = np.concatenate(boots), np.concatenate(rho_boot), np.concatenate(valid_boot)
        ci, valid = percentile_ci(boots)
        rho_ci, rho_valid = percentile_ci(rho_boot)
        numeric.update({f"{cohort}_mean_bootstrap": boots, f"{cohort}_correlation_mean_bootstrap": rho_boot,
                        f"{cohort}_correlation_valid_coordinates": valid_boot})
        result[cohort] = {"mean_metrics": mean_metrics,
            "mean": np.einsum("s,srm->rm", weights, means), "ci95": ci, "valid_replicates": valid,
            "correlation_modes": ["raw", "condition_demeaned"],
            "correlation_mean": finite_mean(correlations, (1, 2)),
            "correlation_mean_ci95": rho_ci, "correlation_valid_replicates": rho_valid,
            "correlation_valid_coordinates": np.isfinite(correlations).sum((1, 2))}
    numeric["mean_metric_names"] = np.asarray(mean_metrics)
    write_npz(Path(output) / "bootstrap.npz", numeric)
    return result, {k: v for k, v in numeric.items() if k.endswith(("_correlations", "_input_count"))}


def _reproduce_a1(data, records, bundle, layer, scale):
    residuals = {}
    for cohort in ("validation", "test"):
        selected = np.array([i for i, row in enumerate(records) if row["split"] == cohort])
        with np.load(Path(bundle) / f"{cohort}_errors.npz", allow_pickle=False) as old:
            positions = {sid: i for i, sid in enumerate(old["sample_ids"].tolist())}
            ids = [positions[records[i]["sample_id"]] for i in selected]
            ri = [list(old["ranks"]).index(int(r)) for r in RANKS]
            for method, field in (("head", "fixed_mse"), ("oracle_same_mean", "oracle_mse")):
                mi = list(old["methods"]).index(method)
                expected = np.take(old["ideal_mse"][mi][ids, :, layer], ri, axis=-1)
                normalized = (data[field][selected] - expected) / scale[None, ..., None]
                check_identity(normalized, f"{cohort} {method} reproduction, layer {layer}")
                residuals[f"{cohort}_{method}"] = np.max(np.abs(normalized), axis=(0, 2))
    return residuals


def _mean_nonfinite_safe(values, weights):
    valid = np.isfinite(values)
    weighted = weights.reshape((len(weights),) + (1,) * (values.ndim - 1))
    total = (valid * weighted).sum(0)
    return np.divide((np.where(valid, values, 0) * weighted).sum(0), total,
                     out=np.full(total.shape, np.nan), where=total > 0)


def run_diagnostic(bundle, output, *, bootstrap_replicates=None):
    bundle, output = Path(bundle), Path(output)
    records = read_json(bundle / "split.json")["samples"]
    with np.load(bundle / "calibration.npz", allow_pickle=False) as archive:
        models = {k: archive[k] for k in ("mean", "scale", "G", "head_basis",
                                         "calibration_weights", "calibration_sample_ids")}
    calib = np.array([i for i, r in enumerate(records) if r["split"] == "calib"])
    if [records[i]["sample_id"] for i in calib] != models["calibration_sample_ids"].tolist():
        raise ValueError("frozen Calib IDs do not match original fit")
    layers = records[0]["num_layers"]
    replicates = BOOTSTRAP_REPLICATES if bootstrap_replicates is None else bootstrap_replicates
    peers = {cohort: pair_index(records, cohort) for cohort in ("validation", "test")}
    output.mkdir(parents=True, exist_ok=False)
    (output / "layers").mkdir()
    _save(output / "metadata.json", {"schema": "headkv-pca-subspace-diagnostic-v1",
        "source_bundle": str(bundle.resolve()), "scope": "full middle [64,T-64)", "ranks": RANKS,
        "primary_rank": 64, "representations": REPRESENTATIONS, "samples": records,
        "pair_indices": {c: info[1] for c, info in peers.items()},
        "pair_groups": {c: info[2] for c, info in peers.items()},
        "forward_gap_target": "left", "reverse_gap_target": "right",
        "boundary_fields": ["lambda_r", "lambda_next", "gap"],
        "angle_order": "ascending", "angle_unit": "radian",
        "point_axes": ["sample", "representation", "head", "rank"],
        "pair_axes": ["pair", "representation", "head", "rank"],
        "angles_axes": ["sample", "representation", "head", "principal_angle"],
        "target_boundary_axes": ["sample", "representation", "head", "rank", "boundary_field"],
        "reference_boundary_axes": ["representation", "head", "rank", "boundary_field"],
        "head_summary_point_axes": ["representation", "layer", "head", "rank"],
        "head_summary_peer_group_axes": ["group", "representation", "layer", "head", "rank"],
        "angle_atol_rad": ANGLE_ATOL_RAD, "overlap_atol": OVERLAP_ATOL,
        "normalized_error_atol": ERROR_ATOL, "bootstrap_replicates": replicates,
        "interpretation": "fixed-basis compromise plus possible out-of-fit loss; post-hoc, no new independent Test"})
    print(f"Subspace diagnostic: {len(records)} inputs, {layers} layers, CPU only; source {bundle}", flush=True)
    point_layers = {key: [] for key in POINT_FIELDS}
    peer_layers = {cohort: {key: [] for key in PEER_FIELDS} for cohort in peers}
    verification = []
    negative_fields = ("gap", "validation_peer_forward_gap", "validation_peer_reverse_gap",
                       "test_peer_forward_gap", "test_peer_reverse_gap")
    negatives = {key: {"roundoff_negative_count": 0, "minimum_negative": None,
                       "minimum_at": None, "values_retained_in_layer_arrays": True}
                 for key in negative_fields}
    for layer in range(layers):
        rows = [read_sample(row, layer_ids=[layer]) for row in records]
        means = np.stack([row["mean"][:, 0] for row in rows]).astype(np.float64)
        covs = np.stack([row["covariance"][:, 0] for row in rows]).astype(np.float64)
        del rows
        fixed_mean, scale = models["mean"][:, layer], models["scale"][:, layer]
        fixed_basis, g = models["head_basis"][:, layer], models["G"][:, layer]
        weights = models["calibration_weights"]
        gc = np.einsum("s,s...ij->...ij", weights, covs[calib])
        delta = means[calib] - fixed_mean
        gmu = np.einsum("s,s...d,s...e->...de", weights, delta, delta)
        check_identity((g - gc - gmu) / np.where(scale > 0, scale, np.nan)[..., None, None], "G=Gc+Gmu")
        fitted_mean = np.einsum("s,s...d->...d", weights, means[calib])
        check_identity((fitted_mean - fixed_mean) / np.sqrt(np.where(scale > 0, scale, np.nan))[..., None],
                       "frozen Calib mean")
        values_gc, centered_basis, _, _ = _eigh(gc)
        centered_basis = np.ascontiguousarray(centered_basis)
        values_g = _eigh(g)[0]
        if np.max(np.abs(fixed_basis.swapaxes(-1, -2) @ fixed_basis - np.eye(fixed_basis.shape[-1]))) > OVERLAP_ATOL:
            raise ValueError("frozen reference basis is not orthonormal")
        data, transient = diagnose_moments(means, covs, fixed_mean, fixed_basis, scale, centered_basis)
        data["boundary_G"], data["boundary_GC"] = spectral_boundary(values_g), spectral_boundary(values_gc)
        data.update({f"reference_mean_shift_{key}": value for key, value in
                     principal_geometry(fixed_basis, centered_basis).items()})
        verification.append(_reproduce_a1(data, records, bundle, layer, scale))
        for key in POINT_FIELDS:
            point_layers[key].append(data[key])
        for cohort, (pairs, _, groups, group_indices) in peers.items():
            pair_data = diagnose_pairs(transient, pairs, scale)
            for key, value in pair_data.items():
                data[f"{cohort}_peer_{key}"] = value
            for key in PEER_FIELDS:
                peer_layers[cohort][key].append(np.stack([
                    finite_mean(pair_data[key][group_indices == group], 0) for group in range(len(groups))]))
        data["layer_id"], data["ranks"] = np.asarray(layer), RANKS
        for key in negative_fields:
            raw = data[key]
            negative = raw[np.isfinite(raw) & (raw < 0)]
            negatives[key]["roundoff_negative_count"] += int(negative.size)
            if negative.size and (negatives[key]["minimum_negative"] is None or
                                  negative.min() < negatives[key]["minimum_negative"]):
                negatives[key]["minimum_negative"] = float(negative.min())
                negatives[key]["minimum_at"] = {"layer": layer,
                    "array_index": list(np.unravel_index(np.nanargmin(raw), raw.shape))}
        write_npz(output / "layers" / f"layer_{layer:03d}.npz", data)
        del transient, means, covs
        print(f"Subspace layer {layer+1}/{layers}: A1, geometry, boundaries and loss checks passed", flush=True)
    point = {key: np.stack(values, axis=2) for key, values in point_layers.items()}
    head, cohorts = {}, {}
    for cohort in ("calib", "validation", "test"):
        ids = [i for i, row in enumerate(records) if row["split"] == cohort]
        rows = [records[i] for i in ids]
        weights = condition_weights([r["condition"] for r in rows], [r["source_group"] for r in rows])
        means = {}
        for key, values in point.items():
            value = _mean_nonfinite_safe(values[ids], weights)
            head[f"{cohort}_{key}"] = value
            means[key] = finite_mean(value, (1, 2))
        alignment = head[f"{cohort}_main_overlap"]
        head[f"{cohort}_alignment_rank_in_layer"] = np.moveaxis(
            alignment.shape[2] - 1 - average_ranks(np.moveaxis(alignment, 2, -1)), -1, 2)
        conditions = sorted({r["condition"] for r in rows})
        condition_means = {condition: {key: finite_mean(point[key][[
            i for i in ids if records[i]["condition"] == condition]], (0, 2, 3))
            for key in POINT_FIELDS} for condition in conditions}
        cohorts[cohort] = {"input_count": len(rows), "source_count": len({r["source_group"] for r in rows}),
            "equal_condition_mean": means, "condition_mean": condition_means,
            "fixed_over_oracle_increase": np.divide(means["fixed_normalized_mse"],
                means["oracle_normalized_mse"], out=np.full_like(means["fixed_normalized_mse"], np.nan),
                where=means["oracle_normalized_mse"] > ERROR_ATOL) - 1}
    for cohort, fields in peer_layers.items():
        groups = peers[cohort][2]
        for key, values in fields.items():
            group_means = np.stack(values, 2)
            head[f"{cohort}_peer_group_{key}"] = group_means
            for kind in ("within", "cross", "pg19_pair"):
                selected = [i for i, group in enumerate(groups) if group[0] == kind]
                value = finite_mean(group_means[selected], 0)
                head[f"{cohort}_peer_{kind}_{key}"] = value
        stability = head[f"{cohort}_peer_within_M_overlap"]
        head[f"{cohort}_within_stability_rank_in_layer"] = np.moveaxis(
            stability.shape[2] - 1 - average_ranks(np.moveaxis(stability, 2, -1)), -1, 2)
    bootstrap, correlation_arrays = bootstrap_association(records, point, output, replicates)
    head.update(correlation_arrays)
    write_npz(output / "head_summary.npz", head)
    _save(output / "summary.json", {"status": "completed", "scope": "post-hoc direction-one diagnostic only",
        "primary_rank": 64, "ranks": RANKS, "cohorts": cohorts, "bootstrap": bootstrap,
        "a1_reproduction": verification, "roundoff_negatives": negatives,
        "main_gap_meaning": "includes in-Calib fixed-basis compromise; not synonymous with generalization failure",
        "source_parameters_refitted": False, "old_results_overwritten": False,
        "layer_files": layers, "head_summary": "head_summary.npz", "bootstrap_file": "bootstrap.npz"})
    return output
