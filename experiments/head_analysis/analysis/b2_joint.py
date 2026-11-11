"""Hash-bound B1/B2 associations on aligned sample/Layer/KV-Head coordinates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from experiments.head_analysis.analysis.b2_consistency import B2RunSetIndex, B2SourceError
from experiments.head_analysis.analysis.b2_relationships import (
    B2RelationshipInputs,
    _pair_rows,
    correlate_profiles,
    load_stage2_relationship_inputs,
)
from experiments.head_analysis.analysis.metric_relationships import PRIMARY_RELATION_DELTAS, QUERY_RELATION_FEATURES
from experiments.head_analysis.common.io import canonical_json_sha256, read_json, sha256_file
from experiments.head_analysis.common.statistics import average_ranks as _ranks
from experiments.head_analysis.common.statistics import bootstrap_draws, bootstrap_means, finite_mean
from experiments.head_analysis.common.statistics import percentile_ci as bootstrap_ci

RCO_DELTAS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _require(condition, message):
    if not condition:
        raise B2SourceError(message)


def _read(path):
    return read_json(path)


@dataclass(frozen=True)
class B2JointInputs:
    stage2: B2RelationshipInputs
    b1: Mapping[str, np.ndarray]
    sources: Mapping[str, object]


def load_b2_joint_inputs(
    index: B2RunSetIndex,
    stage2_dir: Path | str,
    stage3_dir: Path | str,
    *,
    expected_stage2_sha256: str | None = None,
    expected_stage3_sha256: str | None = None,
) -> B2JointInputs:
    """Revalidate accepted sources and join B1 sample rows by explicit ID."""
    try:
        stage2 = load_stage2_relationship_inputs(stage2_dir, index, expected_manifest_sha256=expected_stage2_sha256)
        expected_stage2_sha256 = stage2.stage2_manifest_sha256
        root3 = Path(stage3_dir).resolve()
        manifest3_path = root3 / "stage3_manifest.json"
        verify_hashes = expected_stage3_sha256 is not None
        observed_sha256 = sha256_file(manifest3_path)
        if expected_stage3_sha256 is not None:
            _require(observed_sha256 == expected_stage3_sha256, "stage3 Manifest differs from authority")
        expected_stage3_sha256 = observed_sha256
        m3, p3 = _read(manifest3_path), _read(root3 / "analysis_plan.json")
        _require(
            m3["schema_version"] == "headkv-v1-b2-offline-stage3-manifest-v1"
            and m3["status"] == "partial"
            and m3["completed_stages"] == ["source_validation", "cross_phase_relations"],
            "invalid stage3 schema/state",
        )
        _require(
            m3["source_run_set_sha256"] == p3["source_run_set"]["sha256"] == index.run_set_sha256
            and m3["stage2_source"]["sha256"] == p3["stage2_source"]["sha256"] == expected_stage2_sha256,
            "stage3 source binding differs",
        )
        _require(
            len(m3["outputs"]) == 3
            and {e["relative_path"] for e in m3["outputs"]}
            == {"analysis_plan.json", "stage3_summary.json", "stage3_relationships.npz"},
            "stage3 outputs differ",
        )
        for entry in m3["outputs"]:
            path = root3 / entry["relative_path"]
            _require(path.is_file(), f"missing stage3 output: {path}")
            if verify_hashes:
                _require(
                    path.stat().st_size == entry["file_bytes"] and sha256_file(path) == entry["sha256"],
                    "stage3 output Hash/size differs",
                )
        if verify_hashes:
            _require(sha256_file(root3 / "analysis_plan.json") == m3["plan_sha256"], "stage3 plan binding differs")
            for key in ("parameters", "stage3_parameters"):
                _require(canonical_json_sha256(p3[key]) == p3[key + "_sha256"], "stage3 parameter Hash differs")
        _require(p3["sample_ids"] == stage2.arrays["sample_ids"].tolist(), "stage3 sample coverage differs")
        s3 = _read(root3 / "stage3_summary.json")
        _require(
            s3["schema_version"] == "headkv-v1-b2-offline-stage3-v1"
            and s3["status"] == "partial"
            and s3["plan_sha256"] == m3["plan_sha256"]
            and s3["source_run_set_sha256"] == index.run_set_sha256
            and s3["stage2_source"]["sha256"] == expected_stage2_sha256,
            "stage3 summary binding differs",
        )
        with np.load(root3 / "stage3_relationships.npz", allow_pickle=False) as archive:
            for key in ("sample_ids", "sample_pair_ids", "condition_names", "condition_ids"):
                _require(archive[key].tolist() == stage2.arrays[key].tolist(), "stage3 numeric coordinates differ")
        authority = index.payload["b1_analysis_authority"]
        path1 = index.b1_analysis_path
        if verify_hashes:
            _require(sha256_file(path1) == authority["sha256"], "B1 completed authority differs")
        m1 = _read(path1)
        _require(
            m1["status"] == "completed"
            and m1["schema_version"] == "headkv-v1-b1-offline-completed-v1"
            and m1["source_run_set_sha256"] == index.payload["b1_replay_authority"]["sha256"]
            and m1["sample_count"] == len(index.samples),
            "invalid B1 completed state/source",
        )
        outputs = m1["outputs"]
        _require(len({e["relative_path"] for e in outputs}) == len(outputs), "duplicate B1 output")
        if verify_hashes:
            for entry in outputs:
                path = (path1.parent / entry["relative_path"]).resolve()
                _require(path.is_relative_to(path1.parent.resolve()), "B1 output path escapes bundle")
                _require(
                    path.stat().st_size == entry["file_bytes"] and sha256_file(path) == entry["sha256"],
                    "B1 output Hash/size differs",
                )
        _require(any(e["relative_path"] == "head_profiles.npz" for e in outputs), "missing B1 head_profiles.npz")
        keys = (
            "sample_ids",
            "sample_conditions",
            "condition_names",
            "query_phases",
            "query_feature_names",
            "query_statistics",
            "query_sample_full",
            "query_sample_full_valid_count",
            "page_scopes",
            "rco_k_values",
            "rco_deltas",
            "rco_metrics",
            "rco_sample_head_values",
            "rco_sample_head_valid_pair_count",
            "token_k_values",
            "token_deltas",
            "token_metrics",
            "token_sample_head_metrics",
            "token_sample_head_valid_pair_count",
        )
        with np.load(path1.parent / "head_profiles.npz", allow_pickle=False) as archive:
            b1 = {key: archive[key] for key in keys}
        ids = b1["sample_ids"].tolist()
        expected_ids = stage2.arrays["sample_ids"].tolist()
        _require(
            len(ids) == len(set(ids)) == len(expected_ids) and set(ids) == set(expected_ids), "B1 sample IDs differ"
        )
        lookup = {sid: i for i, sid in enumerate(ids)}
        order = np.asarray([lookup[sid] for sid in expected_ids])
        for key in keys:
            if key.startswith(("sample_", "query_sample_", "rco_sample_", "token_sample_")):
                b1[key] = b1[key][order]
        _require(
            b1["sample_conditions"].tolist() == stage2.arrays["sample_conditions"].tolist()
            and b1["condition_names"].tolist() == stage2.arrays["condition_names"].tolist(),
            "B1/B2 data conditions differ",
        )
        _validate_b1_arrays(b1, index, stage2)
        return B2JointInputs(
            stage2,
            b1,
            {
                "stage2": {"path": str(stage2.stage2_manifest_path), "sha256": stage2.stage2_manifest_sha256},
                "stage3": {"path": str(manifest3_path), "sha256": expected_stage3_sha256},
                "b1": {"path": str(path1), "sha256": authority["sha256"]},
                "b1_profiles_sha256": next(e["sha256"] for e in outputs if e["relative_path"] == "head_profiles.npz"),
                "b1_row_order_in_b2_order": order.tolist(),
            },
        )
    except B2SourceError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise B2SourceError(f"invalid B1/B2 joint source: {exc}") from exc


def _validate_b1_arrays(a, index, stage2):
    dims = index.payload["execution_contract"]["dimensions"]
    n, l, h, q = len(index.samples), dims["L"], dims["Hkv"], dims["Hq"]
    _require(q == 4 * h, "B1/B2 joint analysis requires contiguous 4:1 GQA")
    for key, expected in (
        ("page_scopes", ["all_visible", "prompt_only"]),
        ("rco_k_values", [8, 16, 32]),
        ("token_k_values", [16, 32, 64]),
        ("rco_deltas", list(RCO_DELTAS)),
        ("token_deltas", list(RCO_DELTAS)),
    ):
        _require(a[key].ndim == 1 and a[key].tolist() == expected, f"B1 ordered axis differs: {key}")
    for key, required in (
        ("query_phases", ["decode"]),
        ("query_statistics", ["mean"]),
        ("query_feature_names", list(QUERY_RELATION_FEATURES)),
        ("rco_metrics", ["rco_raw_mean"]),
        ("token_metrics", ["rco_raw_mean"]),
    ):
        values = a[key].tolist()
        _require(
            a[key].ndim == 1 and len(values) == len(set(values)) and set(required) <= set(values),
            f"missing/duplicate B1 axis: {key}",
        )
    shapes = {
        "query_sample_full": (
            n,
            len(a["query_phases"]),
            l,
            q,
            len(a["query_feature_names"]),
            len(a["query_statistics"]),
        ),
        "rco_sample_head_values": (n, l, h, 2, 3, 9, len(a["rco_metrics"])),
        "token_sample_head_metrics": (n, l, q, 3, 9, len(a["token_metrics"])),
    }
    for key, shape in shapes.items():
        _require(
            a[key].shape == shape and a[key].dtype == np.float32 and not np.isinf(a[key]).any(),
            f"invalid B1 shape/dtype: {key}",
        )
    for key, shape in (
        ("query_sample_full_valid_count", shapes["query_sample_full"][:-1]),
        ("rco_sample_head_valid_pair_count", shapes["rco_sample_head_values"][:-1]),
        ("token_sample_head_valid_pair_count", shapes["token_sample_head_metrics"][:-1]),
    ):
        _require(a[key].shape == shape and a[key].dtype.kind == "u", f"invalid B1 count: {key}")
    lengths = stage2.arrays["sample_decode_lengths"]
    decode = a["query_phases"].tolist().index("decode")
    _require(
        np.all(a["query_sample_full_valid_count"][:, decode] <= lengths[:, None, None, None]),
        "B1 Decode query count exceeds C",
    )
    for values_key, count_key, metrics_key in (
        ("rco_sample_head_values", "rco_sample_head_valid_pair_count", "rco_metrics"),
        ("token_sample_head_metrics", "token_sample_head_valid_pair_count", "token_metrics"),
    ):
        counts = a[count_key]
        values = a[values_key][..., a[metrics_key].tolist().index("rco_raw_mean")]
        bound = np.maximum(lengths.astype(np.int64)[:, None] - np.asarray(RCO_DELTAS), 0)
        bound = bound.reshape((n,) + (1,) * (counts.ndim - 2) + (9,))
        _require(np.all(counts <= bound), "B1 pair count exceeds C-delta")
        _require(np.array_equal(np.isfinite(values), counts > 0), "B1 RCO validity/count differs")


def _gqa(values, counts, kv_heads):
    shape = values.shape[:2] + (kv_heads, 4)
    _require(values.shape[2] == 4 * kv_heads and counts.shape == values.shape, "invalid 4:1 GQA arrays")
    _require(np.array_equal(np.isfinite(values), counts > 0), "GQA value/count validity differs")
    v, c = values.reshape(shape), counts.reshape(shape)
    return finite_mean(v, -1), np.isfinite(v).sum(-1), c.min(-1), c.max(-1)


def build_joint_profiles(inputs: B2JointInputs):
    """Return sample-level [N,Feature,L,Hkv], preserving all source axes in metadata."""
    a, b = inputs.stage2.arrays, inputs.b1
    cv = a["sample_prefill_features"][..., 0]
    recall, excess = a["sample_recall_full"][..., 0], a["sample_learned_excess_full"][..., 0]
    x, xm = [], []
    for mi, condition in enumerate(a["condition_ids"]):
        x.append(cv[..., mi])
        xm.append(
            {
                "id": f"cv:{condition}",
                "kind": "cv",
                "condition_id": str(condition),
                "budget": None,
                "scope": None,
                "primary": mi == 0,
            }
        )
        for bi, budget in enumerate(a["budgets"]):
            for ri, scope in enumerate(a["recall_scopes"]):
                for kind, values in (
                    ("learned_recall", recall[..., mi, bi, 0, ri]),
                    ("augmented_recall", recall[..., mi, bi, 1, ri]),
                    ("learned_excess", excess[..., mi, bi, ri]),
                ):
                    x.append(values)
                    xm.append(
                        {
                            "id": f"{kind}:{condition}:b{budget}:{scope}",
                            "kind": kind,
                            "condition_id": str(condition),
                            "budget": int(budget),
                            "scope": str(scope),
                            "primary": mi == 0 and int(budget) == 512 and str(scope) == "prompt_conditional",
                        }
                    )
    y, ym, support, count_min, count_max = [], [], [], [], []

    def append(values, counts, metadata, query_head=False):
        if query_head:
            v, s, lo, hi = _gqa(values, counts, cv.shape[2])
        else:
            _require(np.array_equal(np.isfinite(values), counts > 0), "Page validity/count differs")
            v, s, lo, hi = values, np.isfinite(values).astype(np.uint32), counts, counts
        y.append(v)
        ym.append(metadata)
        support.append(s)
        count_min.append(lo)
        count_max.append(hi)

    decode, mean_stat = b["query_phases"].tolist().index("decode"), b["query_statistics"].tolist().index("mean")
    for feature in QUERY_RELATION_FEATURES:
        fi = b["query_feature_names"].tolist().index(feature)
        append(
            b["query_sample_full"][:, decode, :, :, fi, mean_stat],
            b["query_sample_full_valid_count"][:, decode, :, :, fi],
            {
                "id": f"query:{feature}",
                "kind": "query",
                "feature": feature,
                "scope": "all_visible",
                "k": None,
                "delta": None,
                "primary": True,
            },
            True,
        )
    for kind in ("page_rco", "token_rco"):
        page = kind == "page_rco"
        metrics = b["rco_metrics" if page else "token_metrics"].tolist()
        values = b["rco_sample_head_values" if page else "token_sample_head_metrics"][
            ..., metrics.index("rco_raw_mean")
        ]
        counts = b["rco_sample_head_valid_pair_count" if page else "token_sample_head_valid_pair_count"]
        for si, scope in enumerate(b["page_scopes"] if page else ["all_visible"]):
            for ki, k in enumerate(b["rco_k_values" if page else "token_k_values"]):
                for di, delta in enumerate(RCO_DELTAS):
                    v = values[:, :, :, si, ki, di] if page else values[:, :, :, ki, di]
                    c = counts[:, :, :, si, ki, di] if page else counts[:, :, :, ki, di]
                    append(
                        v,
                        c,
                        {
                            "id": f"{kind}:{scope}:k{k}:d{delta}",
                            "kind": kind,
                            "feature": "rco_raw_mean",
                            "scope": str(scope),
                            "k": int(k),
                            "delta": delta,
                            "primary": int(k) == 32 and delta in PRIMARY_RELATION_DELTAS,
                        },
                        not page,
                    )
    return {
        "x": np.stack(x, 1).astype(np.float64),
        "y": np.stack(y, 1).astype(np.float64),
        "x_metadata": xm,
        "y_metadata": ym,
        "y_valid_source_heads": np.stack(support, 1).astype(np.uint32),
        "y_source_count_min": np.stack(count_min, 1).astype(np.uint64),
        "y_source_count_max": np.stack(count_max, 1).astype(np.uint64),
    }


def _matrix_pearson(x, y):
    varies = (np.ptp(x, axis=-1) > 0)[..., :, None] & (np.ptp(y, axis=-1) > 0)[..., None, :]
    x, y = x - x.mean(-1, keepdims=True), y - y.mean(-1, keepdims=True)
    denominator = np.linalg.norm(x, axis=-1)[..., :, None] * np.linalg.norm(y, axis=-1)[..., None, :]
    result = np.full(denominator.shape, np.nan)
    np.divide(
        x @ np.swapaxes(y, -1, -2), denominator, out=result, where=(denominator > 0) & (x.shape[-1] >= 3) & varies
    )
    return np.clip(result, -1, 1)


def correlate_profile_grid(x, y, *, include_layers=True):
    """Grid [...,X,L,H]×[...,Y,L,H]; fast complete rows, exact missing-data fallback."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    _require(
        x.ndim >= 3 and y.ndim == x.ndim and x.shape[:-3] == y.shape[:-3] and x.shape[-2:] == y.shape[-2:],
        "invalid joint profile grid",
    )
    nx, ny, l, h = x.shape[-3], y.shape[-3], x.shape[-2], x.shape[-1]
    fx, fy = np.isfinite(x), np.isfinite(y)
    complete_x, complete_y = fx.all((-2, -1)), fy.all((-2, -1))
    complete = complete_x[..., :, None] & complete_y[..., None, :]
    sx, sy = np.where(fx, x, 0), np.where(fy, y, 0)
    views = []
    for centered in (False, True):
        xx = sx - sx.mean(-1, keepdims=True) if centered else sx
        yy = sy - sy.mean(-1, keepdims=True) if centered else sy
        xx, yy = xx.reshape(xx.shape[:-2] + (-1,)), yy.reshape(yy.shape[:-2] + (-1,))
        views.append(np.stack((_matrix_pearson(_ranks(xx), _ranks(yy)), _matrix_pearson(xx, yy)), -1))
    values = np.where(complete[..., None, None], np.stack(views, -2), np.nan)
    valid_heads = np.repeat((complete * (l * h))[..., None], 2, -1).astype(np.uint32)
    result = {"values": values, "valid_heads": valid_heads}
    if include_layers:
        xx, yy = np.swapaxes(sx, -3, -2), np.swapaxes(sy, -3, -2)
        lv = np.stack((_matrix_pearson(_ranks(xx), _ranks(yy)), _matrix_pearson(xx, yy)), -1)
        lv = np.moveaxis(lv, -4, -2)
        result["layer_values"] = np.where(complete[..., None, None], lv, np.nan)
        result["layer_valid_heads"] = np.repeat((complete * h)[..., None], l, -1).astype(np.uint32)
    # Partially missing rows need pairwise-complete Layer means and pair-specific ranks.
    partial_x = fx.any((-2, -1)) & ~complete_x
    partial_y = fy.any((-2, -1)) & ~complete_y
    for i in range(nx) if np.any(partial_x) or np.any(partial_y) else ():
        for j in range(ny):
            if np.any(partial_x[..., i] | partial_y[..., j]):
                exact = correlate_profiles(x[..., i, :, :], y[..., j, :, :])
                result["values"][..., i, j, :, :] = exact["values"]
                result["valid_heads"][..., i, j, :] = exact["valid_heads"]
                if include_layers:
                    result["layer_values"][..., i, j, :, :] = exact["layer_values"]
                    result["layer_valid_heads"][..., i, j, :] = exact["layer_valid_heads"]
    return result


def _chunked_ci(values, columns=256):
    shape = values.shape[1:]
    flat = values.reshape(len(values), -1)
    ci, count = np.empty((flat.shape[1], 2), dtype=np.float32), np.empty(flat.shape[1], dtype=np.uint32)
    for start in range(0, flat.shape[1], columns):
        stop = min(start + columns, flat.shape[1])
        ci[start:stop], count[start:stop] = bootstrap_ci(flat[:, start:stop])
    return ci.reshape(shape + (2,)), count.reshape(shape)


@dataclass(frozen=True)
class B2JointSummary:
    arrays: Mapping[str, np.ndarray]
    axes: Mapping[str, list[str]]
    metadata: Mapping[str, object]


def analyze_b2_joint(
    inputs: B2JointInputs,
    *,
    bootstrap_replicates=2000,
    bootstrap_seed=20261110,
    bootstrap_batch_size=64,
    on_progress: Callable[[dict], None] | None = None,
):
    _require(bootstrap_replicates > 0 and bootstrap_batch_size > 0, "positive Bootstrap counts required")
    profiles = build_joint_profiles(inputs)
    a = inputs.stage2.arrays
    x, y = profiles["x"], profiles["y"]
    xm, ym = profiles["x_metadata"], profiles["y_metadata"]
    px = np.asarray([i for i, m in enumerate(xm) if m["primary"]], dtype=np.uint32)
    py = np.asarray([i for i, m in enumerate(ym) if m["primary"]], dtype=np.uint32)
    names = a["condition_names"].tolist()
    pair_ids, pg8, pg32 = _pair_rows(a)
    arrays, axes = {}, {}
    points, intervals, valid_heads, valid_boot, layer_points, layer_heads = [], [], [], [], [], []
    layer_intervals, layer_boot, mxs, mys, support_x, support_y = [], [], [], [], [], []
    units, hashes = {}, {}
    pg_boot, pg_points = {}, {}
    for condition in names:
        rows = (
            pg8
            if condition == "pg19:8k"
            else (
                pg32
                if condition == "pg19:32k"
                else np.asarray(
                    sorted(np.flatnonzero(a["sample_conditions"] == condition), key=lambda i: str(a["sample_ids"][i]))
                )
            )
        )
        _require(len(rows) > 0, "empty joint data condition")
        label = "pg19-pairs" if condition.startswith("pg19:") else condition
        draws = bootstrap_draws(bootstrap_seed, label, bootstrap_replicates, len(rows))
        hashes[condition] = hashlib.sha256(draws.astype("<u4").tobytes()).hexdigest()
        units[condition] = pair_ids if label == "pg19-pairs" else a["sample_ids"][rows].tolist()
        mx, my = finite_mean(x[rows]), finite_mean(y[rows])
        point = correlate_profile_grid(mx, my)
        boot = np.empty((bootstrap_replicates,) + point["values"].shape, dtype=np.float32)
        primary_layers = np.empty((bootstrap_replicates, len(px), len(py), x.shape[2], 2), dtype=np.float32)
        for start in range(0, bootstrap_replicates, bootstrap_batch_size):
            stop = min(start + bootstrap_batch_size, bootstrap_replicates)
            bx, by = bootstrap_means(x[rows], draws[start:stop]), bootstrap_means(y[rows], draws[start:stop])
            boot[start:stop] = correlate_profile_grid(bx, by, include_layers=False)["values"]
            primary_layers[start:stop] = correlate_profile_grid(bx[:, px], by[:, py])["layer_values"]
            if on_progress and (stop == bootstrap_replicates or stop % 256 == 0):
                on_progress({"condition": condition, "completed_replicates": stop, "total": bootstrap_replicates})
        ci, nb = _chunked_ci(boot)
        lc, ln = _chunked_ci(primary_layers)
        points.append(point["values"])
        intervals.append(ci)
        valid_heads.append(point["valid_heads"])
        valid_boot.append(nb)
        layer_points.append(point["layer_values"])
        layer_heads.append(point["layer_valid_heads"])
        layer_intervals.append(lc)
        layer_boot.append(ln)
        mxs.append(mx)
        mys.append(my)
        support_x.append(np.isfinite(x[rows]).sum(0))
        support_y.append(np.isfinite(y[rows]).sum(0))
        if label == "pg19-pairs":
            pg_boot[condition], pg_points[condition] = boot, point["values"]
            if len(pg_boot) == 2:
                difference = pg_boot["pg19:32k"] - pg_boot["pg19:8k"]
                arrays["pg19_correlation_difference"] = (pg_points["pg19:32k"] - pg_points["pg19:8k"]).astype(
                    np.float32
                )
                arrays["pg19_correlation_ci"], arrays["pg19_valid_replicates"] = _chunked_ci(difference)
                pg_boot.clear()
                del difference
        del boot, primary_layers
    values = {
        "correlation": (points, ["DataCondition", "X", "Y", "View", "Method"]),
        "correlation_ci": (intervals, ["DataCondition", "X", "Y", "View", "Method", "CI"]),
        "correlation_valid_heads": (valid_heads, ["DataCondition", "X", "Y", "View"]),
        "correlation_valid_replicates": (valid_boot, ["DataCondition", "X", "Y", "View", "Method"]),
        "per_layer_correlation": (layer_points, ["DataCondition", "X", "Y", "L", "Method"]),
        "per_layer_valid_heads": (layer_heads, ["DataCondition", "X", "Y", "L"]),
        "primary_per_layer_ci": (layer_intervals, ["DataCondition", "PrimaryX", "PrimaryY", "L", "Method", "CI"]),
        "primary_per_layer_valid_replicates": (layer_boot, ["DataCondition", "PrimaryX", "PrimaryY", "L", "Method"]),
        "condition_x_mean": (mxs, ["DataCondition", "X", "L", "Hkv"]),
        "condition_y_mean": (mys, ["DataCondition", "Y", "L", "Hkv"]),
        "condition_x_valid_samples": (support_x, ["DataCondition", "X", "L", "Hkv"]),
        "condition_y_valid_samples": (support_y, ["DataCondition", "Y", "L", "Hkv"]),
    }
    for key, (v, dims) in values.items():
        arrays[key] = np.asarray(v, dtype=np.uint32 if "valid" in key else np.float32)
        axes[key] = dims
    if pair_ids:
        axes.update(
            pg19_correlation_difference=["X", "Y", "View", "Method"],
            pg19_correlation_ci=["X", "Y", "View", "Method", "CI"],
            pg19_valid_replicates=["X", "Y", "View", "Method"],
        )
    for key, v, dims in (
        ("sample_x", x.astype(np.float32), ["N", "X", "L", "Hkv"]),
        ("sample_y", y.astype(np.float32), ["N", "Y", "L", "Hkv"]),
        ("sample_y_valid_source_heads", profiles["y_valid_source_heads"], ["N", "Y", "L", "Hkv"]),
        ("sample_y_source_count_min", profiles["y_source_count_min"], ["N", "Y", "L", "Hkv"]),
        ("sample_y_source_count_max", profiles["y_source_count_max"], ["N", "Y", "L", "Hkv"]),
        ("primary_x_indices", px, ["PrimaryX"]),
        ("primary_y_indices", py, ["PrimaryY"]),
        ("sample_ids", a["sample_ids"], ["N"]),
        ("sample_pair_ids", a["sample_pair_ids"], ["N"]),
        ("sample_conditions", a["sample_conditions"], ["N"]),
        ("condition_names", a["condition_names"], ["DataCondition"]),
        ("pg19_pair_ids", np.asarray(pair_ids, dtype=str), ["Pair"]),
        ("x_ids", np.asarray([m["id"] for m in xm]), ["X"]),
        ("y_ids", np.asarray([m["id"] for m in ym]), ["Y"]),
        ("views", np.asarray(["all_heads", "layer_centered"]), ["View"]),
        ("methods", np.asarray(["spearman", "pearson"]), ["Method"]),
        ("ci_labels", np.asarray(["lower_0.025", "upper_0.975"]), ["CI"]),
    ):
        arrays[key], axes[key] = v, dims
    return B2JointSummary(
        arrays,
        axes,
        {
            "x_profiles": xm,
            "y_profiles": ym,
            "bootstrap_units": units,
            "draws_sha256": hashes,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_batch_size": bootstrap_batch_size,
            "generator": "PCG64",
            "seed_derivation": "SHA256(f'{seed}:{unit_label}') first 8 bytes little-endian",
            "numpy_version": np.__version__,
            "working_dtype": "float64",
            "replicate_buffer_dtype": "float32",
            "resampling_unit": "whole_sample_or_frozen_pg19_pair",
            "ci_method": "linear_percentile",
            "global_bootstrap": "all configurations",
            "per_layer_bootstrap": "primary X and primary Y only",
            "gqa": "sample-level time means first; contiguous q_head//4, finite members equally weighted",
            "primary_view": "layer_centered",
            "primary_method": "spearman",
            "attention_concentration": "separate associations, not multivariate confounder adjustment",
        },
    )
