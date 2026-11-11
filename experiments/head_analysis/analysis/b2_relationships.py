"""B2 stage 3: sample-bootstrap CV/coverage associations and paired contrasts.

Consumes hash-bound stage-2 summaries. It never reads model weights or changes
raw traces. Heads define fixed profile coordinates, not resampling units.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from experiments.head_analysis.analysis.b2_consistency import B2RunSetIndex, B2SourceError
from experiments.head_analysis.common.io import canonical_json_sha256, read_json, sha256_file
from experiments.head_analysis.common.statistics import average_ranks as _ranks
from experiments.head_analysis.common.statistics import bootstrap_draws, bootstrap_means, finite_mean
from experiments.head_analysis.common.statistics import pearson as _pearson
from experiments.head_analysis.common.statistics import percentile_ci as bootstrap_ci

TARGETS = ("learned_recall", "augmented_recall", "learned_excess")
VIEWS = ("all_heads", "layer_centered")
METHODS = ("spearman", "pearson")


def _require(value, message):
    if not value:
        raise B2SourceError(message)


def _json(path):
    value = read_json(path)
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def correlate_profiles(x, y):
    """Correlate [...,L,H] profiles, with pairwise-complete Layer centering."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    _require(x.ndim >= 2 and x.shape[-2] > 0 and x.shape[-1] > 0, "invalid Layer/Head profiles")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = np.where(valid, x, np.nan), np.where(valid, y, np.nan)
    centered_x = x - finite_mean(x, -1)[..., None]
    centered_y = y - finite_mean(y, -1)[..., None]
    global_values, global_counts = [], []
    for left, right in ((x, y), (centered_x, centered_y)):
        left = left.reshape(left.shape[:-2] + (-1,))
        right = right.reshape(right.shape[:-2] + (-1,))
        pearson, count = _pearson(left, right)
        spearman, _ = _pearson(_ranks(left), _ranks(right))
        global_values.append(np.stack((spearman, pearson), axis=-1))
        global_counts.append(count)
    per_layer_pearson, layer_count = _pearson(x, y)
    per_layer_spearman, _ = _pearson(_ranks(x), _ranks(y))
    return {
        "values": np.stack(global_values, axis=-2),
        "valid_heads": np.stack(global_counts, axis=-1),
        "layer_values": np.stack((per_layer_spearman, per_layer_pearson), axis=-1),
        "layer_valid_heads": layer_count,
    }


@dataclass(frozen=True)
class B2RelationshipInputs:
    arrays: Mapping[str, np.ndarray]
    stage2_manifest_path: Path
    stage2_manifest_sha256: str
    source_run_set_sha256: str


def load_stage2_relationship_inputs(
    directory: Path | str,
    index: B2RunSetIndex,
    *,
    expected_manifest_sha256: str | None = None,
) -> B2RelationshipInputs:
    """Validate immutable stage-2 files and join their sample coordinates to B2."""
    try:
        directory = Path(directory).resolve()
        path = directory / "stage2_manifest.json"
        verify_hashes = expected_manifest_sha256 is not None
        observed_sha256 = sha256_file(path)
        if expected_manifest_sha256 is not None:
            _require(observed_sha256 == expected_manifest_sha256, "stage2 Manifest differs from external authority")
        expected_manifest_sha256 = observed_sha256
        manifest = _json(path)
        _require(
            manifest["schema_version"] == "headkv-v1-b2-offline-stage2-manifest-v1" and manifest["status"] == "partial",
            "invalid stage2 Manifest schema/state",
        )
        _require(
            manifest["completed_stages"] == ["source_validation", "distributions_and_recall"], "stage2 is incomplete"
        )
        _require(manifest["source_run_set_sha256"] == index.run_set_sha256, "stage2 used a different B2 Run Set")
        entries = manifest["outputs"]
        _require(
            len(entries) == 3
            and {e["relative_path"] for e in entries}
            == {"analysis_plan.json", "stage2_profiles.npz", "stage2_summary.json"},
            "stage2 output set differs",
        )
        for entry in entries:
            source = directory / entry["relative_path"]
            _require(source.is_file(), f"missing stage2 source: {source}")
            if verify_hashes:
                _require(
                    source.stat().st_size == entry["file_bytes"] and sha256_file(source) == entry["sha256"],
                    f"stage2 source Hash/size differs: {source}",
                )
        plan, summary = _json(directory / "analysis_plan.json"), _json(directory / "stage2_summary.json")
        _require(manifest["plan_sha256"] == summary["plan_sha256"], "stage2 plan binding differs")
        if verify_hashes:
            _require(sha256_file(directory / "analysis_plan.json") == manifest["plan_sha256"], "stage2 plan binding differs")
        _require(
            plan["source_run_set"]["sha256"] == summary["source_run_set_sha256"] == index.run_set_sha256,
            "stage2 source binding differs",
        )
        if verify_hashes:
            _require(
                canonical_json_sha256(plan["stage2_parameters"]) == plan["stage2_parameters_sha256"],
                "stage2 parameter Hash differs",
            )
        required = (
            "sample_ids",
            "sample_conditions",
            "sample_pair_ids",
            "sample_decode_lengths",
            "condition_names",
            "condition_ids",
            "budgets",
            "statistics",
            "set_types",
            "recall_scopes",
            "time_metrics",
            "sample_prefill_features",
            "sample_recall_full",
            "sample_learned_excess_full",
            "sample_recall_quarters",
            "sample_recall_temporal",
        )
        with np.load(directory / "stage2_profiles.npz", allow_pickle=False) as archive:
            _require(set(archive.files) == set(summary["field_axes"]), "stage2 field-axis registry differs")
            arrays = {key: archive[key] for key in required}
        samples = list(index.samples.values())
        _require(
            arrays["sample_ids"].tolist() == [s.sample_id for s in samples] == plan["sample_ids"],
            "stage2 sample order/coverage differs",
        )
        _require(
            arrays["sample_conditions"].tolist() == [s.condition for s in samples]
            and arrays["sample_pair_ids"].tolist() == [s.pair_id or "" for s in samples]
            and arrays["sample_decode_lengths"].tolist() == [s.decode_length for s in samples],
            "stage2 sample metadata differs",
        )
        _require(
            summary["condition_counts"] == index.condition_counts
            and arrays["condition_names"].tolist() == sorted(index.condition_counts),
            "stage2 condition coverage differs",
        )
        _require(
            arrays["condition_ids"].tolist() == index.payload["execution_contract"]["axes"]["conditions"]
            and arrays["budgets"].tolist() == [128, 512, 2048]
            and arrays["set_types"].tolist() == ["learned_middle", "augmented_prompt"]
            and arrays["recall_scopes"].tolist() == ["all_visible", "prompt_conditional"]
            and arrays["statistics"].tolist() == ["mean", "median", "p10", "p90"]
            and arrays["time_metrics"].tolist() == ["slope", "normalized_auc", "last_minus_first_quarter"],
            "stage2 ordered axes differ",
        )
        dims = index.payload["execution_contract"]["dimensions"]
        n, l, h, m, b = len(samples), dims["L"], dims["Hkv"], dims["M"], dims["Nb"]
        shapes = {
            "sample_prefill_features": (n, l, h, m, 5),
            "sample_recall_full": (n, l, h, m, b, 2, 2, 4),
            "sample_learned_excess_full": (n, l, h, m, b, 2, 4),
            "sample_recall_quarters": (n, 4, l, h, m, b, 2, 2),
            "sample_recall_temporal": (n, l, h, m, b, 2, 2, 3),
        }
        for key, shape in shapes.items():
            _require(
                arrays[key].shape == shape and arrays[key].dtype == np.float32 and not np.isinf(arrays[key]).any(),
                f"stage2 numeric shape/dtype differs: {key}",
            )
        return B2RelationshipInputs(arrays, path, expected_manifest_sha256, index.run_set_sha256)
    except B2SourceError:
        raise
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise B2SourceError(f"invalid stage2 relationship source: {exc}") from exc


@dataclass(frozen=True)
class B2RelationshipSummary:
    arrays: Mapping[str, np.ndarray]
    records: tuple[dict, ...]
    metadata: Mapping[str, object]

    @property
    def axes(self):
        axes = {
            "correlation": ["Relation", "View", "Method"],
            "correlation_valid_heads": ["Relation", "View"],
            "per_layer_correlation": ["Relation", "L", "Method"],
            "per_layer_valid_heads": ["Relation", "L"],
            "temporal_condition_mean": ["DataCondition", "M", "B", "U", "R", "TimeMetric"],
            "sensitivity_difference": ["Sensitivity", "View", "Method"],
            "pg19_correlation_difference": ["PairContrast", "View", "Method"],
        }
        for stem, base in (
            ("correlation", "correlation"),
            ("per_layer", "per_layer_correlation"),
            ("temporal_condition", "temporal_condition_mean"),
            ("sensitivity", "sensitivity_difference"),
            ("pg19_correlation", "pg19_correlation_difference"),
        ):
            axes[f"{stem}_ci"] = axes[base] + ["CI"]
            axes[f"{stem}_valid_replicates"] = axes[base]
        for name, shape in (
            ("cv", ["M"]),
            ("coverage", ["M", "B", "Target", "R"]),
            ("temporal", ["M", "B", "U", "R", "TimeMetric"]),
        ):
            if f"pg19_{name}_pair_difference" in self.arrays:
                axes[f"pg19_{name}_pair_difference"] = ["Pair"] + shape
                axes[f"pg19_{name}_mean_difference"] = shape
                axes[f"pg19_{name}_ci"] = shape + ["CI"]
                axes[f"pg19_{name}_valid_replicates"] = shape
        for key, axis in (
            ("relation_ids", "Relation"),
            ("condition_names", "DataCondition"),
            ("condition_ids", "M"),
            ("budgets", "B"),
            ("targets", "Target"),
            ("views", "View"),
            ("methods", "Method"),
            ("sample_ids", "N"),
            ("sample_pair_ids", "N"),
            ("pg19_pair_ids", "Pair"),
            ("recall_scopes", "R"),
            ("set_types", "U"),
            ("time_metrics", "TimeMetric"),
            ("ci_labels", "CI"),
        ):
            axes[key] = [axis]
        _require(set(axes) == set(self.arrays), "stage3 field-axis registry differs")
        return axes


def _pair_rows(a):
    pair_map = {}
    for i, (condition, pair) in enumerate(zip(a["sample_conditions"], a["sample_pair_ids"])):
        if str(condition).startswith("pg19:"):
            _require(bool(pair), "PG19 missing pair_id")
            _require(condition in ("pg19:8k", "pg19:32k"), "unsupported PG19 condition")
            pair_map.setdefault(str(pair), {})
            _require(str(condition) not in pair_map[str(pair)], "duplicate PG19 pair member")
            pair_map[str(pair)][str(condition)] = i
    _require(all(set(v) == {"pg19:8k", "pg19:32k"} for v in pair_map.values()), "incomplete PG19 pairs")
    pairs = sorted(pair_map)
    return (
        pairs,
        np.asarray([pair_map[p]["pg19:8k"] for p in pairs], dtype=int),
        np.asarray([pair_map[p]["pg19:32k"] for p in pairs], dtype=int),
    )


def analyze_b2_relationships(
    inputs: B2RelationshipInputs,
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20261110,
    on_record: Callable[[int, str], None] | None = None,
) -> B2RelationshipSummary:
    a = inputs.arrays
    _require(bootstrap_replicates > 0, "bootstrap_replicates must be positive")
    cv = a["sample_prefill_features"][..., 0].astype(np.float64)
    recall = a["sample_recall_full"][..., 0].astype(np.float64)
    excess = a["sample_learned_excess_full"][..., 0].astype(np.float64)
    y = np.stack((recall[..., 0, :], recall[..., 1, :], excess), axis=-2)
    names, condition_ids = a["condition_names"].tolist(), a["condition_ids"].tolist()
    budgets = a["budgets"].tolist()
    primary = condition_ids.index("paper_token_primary")
    budget512 = budgets.index(512)
    pair_ids, pg8, pg32 = _pair_rows(a)
    records, points, cis, counts, boot_counts, layer_points, layer_cis, layer_counts, layer_boot_counts = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    bootstrap_values = []
    lookup = {}
    units, draw_hashes = {}, {}
    temporal = a["sample_recall_temporal"].astype(np.float64)
    scalar_cv = finite_mean(cv, axis=(1, 2))
    scalar_y = finite_mean(y, axis=(1, 2))
    scalar_time = finite_mean(temporal, axis=(1, 2))
    time_means, time_cis, time_counts = [], [], []

    def append_record(meta, x, target, bx, by):
        point = correlate_profiles(x, target)
        boot = correlate_profiles(bx, by)
        ci, valid = bootstrap_ci(boot["values"])
        layer_ci, layer_valid = bootstrap_ci(boot["layer_values"])
        records.append(meta)
        points.append(point["values"])
        cis.append(ci)
        counts.append(point["valid_heads"])
        boot_counts.append(valid)
        layer_points.append(point["layer_values"])
        layer_cis.append(layer_ci)
        layer_counts.append(point["layer_valid_heads"])
        layer_boot_counts.append(layer_valid)
        bootstrap_values.append(boot["values"].astype(np.float32))
        if on_record:
            on_record(len(records), meta["relation_id"])

    for condition in names:
        if condition == "pg19:8k":
            rows = pg8
        elif condition == "pg19:32k":
            rows = pg32
        else:
            rows = np.asarray(
                sorted(np.flatnonzero(a["sample_conditions"] == condition), key=lambda i: str(a["sample_ids"][i])),
                dtype=int,
            )
        _require(len(rows) > 0, "empty data condition")
        label = "pg19-pairs" if condition.startswith("pg19:") else condition
        draws = bootstrap_draws(bootstrap_seed, label, bootstrap_replicates, len(rows))
        units[condition] = pair_ids if label == "pg19-pairs" else a["sample_ids"][rows].tolist()
        draw_hashes[condition] = hashlib.sha256(draws.astype("<u4").tobytes()).hexdigest()
        mx, my = finite_mean(cv[rows]), finite_mean(y[rows])
        bx, by = bootstrap_means(cv[rows], draws), bootstrap_means(y[rows], draws)
        for mi, metric_condition in enumerate(condition_ids):
            for bi, budget in enumerate(budgets):
                for ti, target in enumerate(TARGETS):
                    for ri, scope in enumerate(a["recall_scopes"]):
                        meta = {
                            "condition": condition,
                            "condition_id": metric_condition,
                            "budget": int(budget),
                            "target": target,
                            "scope": str(scope),
                            "phase": "full",
                            "sample_count": len(rows),
                        }
                        meta["relation_id"] = f"{condition}:{metric_condition}:b{budget}:{target}:{scope}:full"
                        lookup[(condition, mi, bi, ti, ri)] = len(records)
                        append_record(
                            meta, mx[:, :, mi], my[:, :, mi, bi, ti, ri], bx[:, :, :, mi], by[:, :, :, mi, bi, ti, ri]
                        )
        # Pre-registered temporal relations: primary/512/Prompt, both actual sets.
        quarters = a["sample_recall_quarters"][rows, :, :, :, primary, budget512, :, 1].astype(np.float64)
        if np.all(a["sample_decode_lengths"][rows] >= 32):
            mq, bq = finite_mean(quarters), bootstrap_means(quarters, draws)
            for qi in range(4):
                for ti, target in enumerate(TARGETS[:2]):
                    meta = {
                        "condition": condition,
                        "condition_id": "paper_token_primary",
                        "budget": 512,
                        "target": target,
                        "scope": "prompt_conditional",
                        "phase": f"q{qi+1}",
                        "sample_count": len(rows),
                    }
                    meta["relation_id"] = f"{condition}:primary:b512:{target}:prompt:q{qi+1}"
                    append_record(meta, mx[:, :, primary], mq[qi, :, :, ti], bx[:, :, :, primary], bq[:, qi, :, :, ti])
        time_means.append(finite_mean(scalar_time[rows]))
        tc, tn = bootstrap_ci(bootstrap_means(scalar_time[rows], draws))
        time_cis.append(tc)
        time_counts.append(tn)
        del bx, by
    arrays = {
        "correlation": np.asarray(points, dtype=np.float32),
        "correlation_ci": np.asarray(cis, dtype=np.float32),
        "correlation_valid_heads": np.asarray(counts, dtype=np.uint32),
        "correlation_valid_replicates": np.asarray(boot_counts, dtype=np.uint32),
        "per_layer_correlation": np.asarray(layer_points, dtype=np.float32),
        "per_layer_ci": np.asarray(layer_cis, dtype=np.float32),
        "per_layer_valid_heads": np.asarray(layer_counts, dtype=np.uint32),
        "per_layer_valid_replicates": np.asarray(layer_boot_counts, dtype=np.uint32),
        "temporal_condition_mean": np.asarray(time_means, dtype=np.float32),
        "temporal_condition_ci": np.asarray(time_cis, dtype=np.float32),
        "temporal_condition_valid_replicates": np.asarray(time_counts, dtype=np.uint32),
    }
    # Paired parameter differences share the same within-condition Bootstrap draws.
    sensitivity, sensitivity_values, sensitivity_cis, sensitivity_counts = [], [], [], []
    for key, j in lookup.items():
        condition, mi, bi, ti, ri = key
        if mi == primary:
            continue
        base = lookup[(condition, primary, bi, ti, ri)]
        sensitivity.append(
            {
                "condition": condition,
                "condition_id": condition_ids[mi],
                "budget": budgets[bi],
                "target": TARGETS[ti],
                "scope": str(a["recall_scopes"][ri]),
                "direction": "variant_minus_primary",
            }
        )
        sensitivity_values.append(points[j] - points[base])
        ci, nvalid = bootstrap_ci(bootstrap_values[j] - bootstrap_values[base])
        sensitivity_cis.append(ci)
        sensitivity_counts.append(nvalid)
    arrays.update(
        sensitivity_difference=np.asarray(sensitivity_values, dtype=np.float32),
        sensitivity_ci=np.asarray(sensitivity_cis, dtype=np.float32),
        sensitivity_valid_replicates=np.asarray(sensitivity_counts, dtype=np.uint32),
    )
    pair_correlations, pair_corr_values, pair_corr_cis, pair_corr_counts = [], [], [], []
    if pair_ids:
        draws = bootstrap_draws(bootstrap_seed, "pg19-pairs", bootstrap_replicates, len(pair_ids))
        for name, values in (("cv", scalar_cv), ("coverage", scalar_y), ("temporal", scalar_time)):
            difference = values[pg32] - values[pg8]
            arrays[f"pg19_{name}_pair_difference"] = difference.astype(np.float32)
            arrays[f"pg19_{name}_mean_difference"] = finite_mean(difference).astype(np.float32)
            ci, nvalid = bootstrap_ci(bootstrap_means(difference, draws))
            arrays[f"pg19_{name}_ci"] = ci
            arrays[f"pg19_{name}_valid_replicates"] = nvalid
        for key, j in lookup.items():
            condition, mi, bi, ti, ri = key
            if condition != "pg19:32k":
                continue
            base = lookup[("pg19:8k", mi, bi, ti, ri)]
            pair_correlations.append(
                {
                    "condition_id": condition_ids[mi],
                    "budget": budgets[bi],
                    "target": TARGETS[ti],
                    "scope": str(a["recall_scopes"][ri]),
                    "direction": "32k_minus_8k",
                }
            )
            pair_corr_values.append(points[j] - points[base])
            ci, nvalid = bootstrap_ci(bootstrap_values[j] - bootstrap_values[base])
            pair_corr_cis.append(ci)
            pair_corr_counts.append(nvalid)
    arrays.update(
        pg19_correlation_difference=np.asarray(pair_corr_values, dtype=np.float32),
        pg19_correlation_ci=np.asarray(pair_corr_cis, dtype=np.float32),
        pg19_correlation_valid_replicates=np.asarray(pair_corr_counts, dtype=np.uint32),
        relation_ids=np.asarray([r["relation_id"] for r in records]),
        condition_names=np.asarray(names),
        condition_ids=np.asarray(condition_ids),
        budgets=np.asarray(budgets, dtype=np.uint32),
        targets=np.asarray(TARGETS),
        views=np.asarray(VIEWS),
        methods=np.asarray(METHODS),
        sample_ids=a["sample_ids"].copy(),
        sample_pair_ids=a["sample_pair_ids"].copy(),
        pg19_pair_ids=np.asarray(pair_ids, dtype=str),
        recall_scopes=a["recall_scopes"].copy(),
        set_types=np.asarray(["learned_middle", "augmented_prompt"]),
        time_metrics=np.asarray(["slope", "normalized_auc", "last_minus_first_quarter"]),
        ci_labels=np.asarray(["lower_0.025", "upper_0.975"]),
    )
    return B2RelationshipSummary(
        arrays,
        tuple(records),
        {
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "generator": "PCG64",
            "seed_derivation": "SHA256(f'{seed}:{unit_label}') first 8 bytes little-endian",
            "bootstrap_units": units,
            "draws_sha256": draw_hashes,
            "sensitivity_records": sensitivity,
            "pg19_correlation_records": pair_correlations,
            "primary_view": "layer_centered",
            "primary_method": "spearman",
            "minimum_valid_heads": 3,
            "ci_quantiles": [0.025, 0.975],
            "ci_method": "linear_percentile",
            "resampling_unit": "whole_sample_or_frozen_pg19_pair",
            "not_independent_units": ["head", "step", "layer"],
            "temporal_scalar_aggregation": "head_equal_within_sample_then_sample_equal",
        },
    )
