"""Offline statistics and summaries for existing B1 trace artifacts."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.head_analysis.collectors.attention_stats import FEATURE_NAMES
from experiments.head_analysis.common.io import canonical_json_sha256
from experiments.head_analysis.common.statistics import bootstrap_sample_mean_ci, finite_mean_count
from experiments.head_analysis.storage.b1_reader import (
    DEFAULT_DELTAS,
    DEFAULT_PAGE_K,
    FROZEN_DISTANCE_BINS,
    PAGE_RCO_SCHEMA_VERSION,
    RUN_SET_SCHEMA_VERSION,
    B1OfflineError,
    FrozenSample,
    RunBatch,
    RunSetIndex,
    ValidationReport,
    _read_json,
    load_completed_run_set,
    sha256_file,
    validate_analysis_sources,
)
from experiments.head_analysis.storage.trace_writer import PAGE_SCOPES, SCHEMA_VERSION

ANALYSIS_PLAN_SCHEMA_VERSION = "headkv-v1-b1-offline-plan-v1"
ANALYSIS_SCHEMA_VERSION = "headkv-v1-b1-offline-analysis-v1"
DEFAULT_TOKEN_K = (16, 32, 64)
SUMMARY_STATISTICS = ("mean", "median", "p10", "p90")
QUERY_PHASES = ("prefill", "decode")
DECODE_SEGMENTS = ("q1", "q2", "q3", "q4")
PAGE_METRICS = ("scope_mass", "top32_coverage", "scope_valid_rate")
RCO_METRICS = ("overlap_raw_mean", "random_baseline_mean", "rco_raw_mean", "rco_clipped_mean")
INVARIANT_ATOL = 1e-5
INVARIANT_RTOL = 1e-5
DEFAULT_ANALYSIS_PARAMETERS: dict[str, Any] = {
    "page_k": list(DEFAULT_PAGE_K),
    "token_k": list(DEFAULT_TOKEN_K),
    "deltas": list(DEFAULT_DELTAS),
    "bootstrap_replicates": 2000,
    "bootstrap_seed": 20261103,
    "primary_weighting": "sample_equal",
}


@dataclass(frozen=True)
class QueryMetricSummary:
    """Sample and condition Query-Head profiles with explicit array axes."""

    sample_ids: tuple[str, ...]
    sample_conditions: tuple[str, ...]
    condition_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    statistics: tuple[str, ...]
    phases: tuple[str, ...]
    decode_segments: tuple[str, ...]
    sample_full: np.ndarray  # [N,P,L,Hq,F,Stat]
    sample_full_valid_count: np.ndarray  # [N,P,L,Hq,F]
    sample_decode_segments: np.ndarray  # [N,Seg,L,Hq,F,Stat]
    sample_decode_segment_valid_count: np.ndarray  # [N,Seg,L,Hq,F]
    condition_full_mean: np.ndarray  # [Cond,P,L,Hq,F]
    condition_full_valid_samples: np.ndarray  # [Cond,P,L,Hq,F]
    condition_decode_segment_mean: np.ndarray  # [Cond,Seg,L,Hq,F]
    condition_decode_segment_valid_samples: np.ndarray  # [Cond,Seg,L,Hq,F]


@dataclass(frozen=True)
class PageMetricSummary:
    """Sample and condition KV-Head Page profiles with explicit array axes."""

    sample_ids: tuple[str, ...]
    sample_conditions: tuple[str, ...]
    condition_names: tuple[str, ...]
    scopes: tuple[str, ...]
    metrics: tuple[str, ...]
    statistics: tuple[str, ...]
    decode_segments: tuple[str, ...]
    sample_full: np.ndarray  # [N,L,Hkv,S,M,Stat]
    sample_full_valid_count: np.ndarray  # [N,L,Hkv,S,M]
    sample_decode_segments: np.ndarray  # [N,Seg,L,Hkv,S,M,Stat]
    sample_decode_segment_valid_count: np.ndarray  # [N,Seg,L,Hkv,S,M]
    condition_full_mean: np.ndarray  # [Cond,L,Hkv,S,M]
    condition_full_valid_samples: np.ndarray  # [Cond,L,Hkv,S,M]
    condition_decode_segment_mean: np.ndarray  # [Cond,Seg,L,Hkv,S,M]
    condition_decode_segment_valid_samples: np.ndarray  # [Cond,Seg,L,Hkv,S,M]


@dataclass(frozen=True)
class PageRCOSummary:
    """Page RCO sample records, Head profiles, sensitivity, and scalar CIs."""

    sample_ids: tuple[str, ...]
    sample_conditions: tuple[str, ...]
    condition_names: tuple[str, ...]
    scopes: tuple[str, ...]
    k_values: tuple[int, ...]
    deltas: tuple[int, ...]
    metrics: tuple[str, ...]
    sample_head_values: np.ndarray  # [N,L,Hkv,S,K,D,M]
    sample_head_valid_pair_count: np.ndarray  # [N,L,Hkv,S,K,D]
    condition_head_sample_equal_mean: np.ndarray  # [Cond,L,Hkv,S,K,D,M]
    condition_head_valid_samples: np.ndarray  # [Cond,L,Hkv,S,K,D,M]
    condition_head_pair_weighted_mean: np.ndarray  # [Cond,L,Hkv,S,K,D,M]
    condition_head_valid_pair_count: np.ndarray  # [Cond,L,Hkv,S,K,D]
    condition_scalar_sample_equal_mean: np.ndarray  # [Cond,S,K,D,M]
    condition_scalar_valid_samples: np.ndarray  # [Cond,S,K,D,M]
    condition_scalar_pair_weighted_mean: np.ndarray  # [Cond,S,K,D,M]
    condition_scalar_valid_pair_count: np.ndarray  # [Cond,S,K,D]
    condition_scalar_bootstrap_ci: np.ndarray  # [Cond,S,K,D,M,2]
    pg19_pair_ids: tuple[str, ...]
    pg19_pair_head_difference_32k_minus_8k: np.ndarray  # [Pair,L,Hkv,S,K,D,M]
    pg19_pair_head_mean_difference: np.ndarray  # [L,Hkv,S,K,D,M]
    pg19_pair_scalar_difference_32k_minus_8k: np.ndarray  # [Pair,S,K,D,M]
    pg19_pair_scalar_mean_difference: np.ndarray  # [S,K,D,M]
    pg19_pair_scalar_bootstrap_ci: np.ndarray  # [S,K,D,M,2]
    bootstrap_replicates: int
    bootstrap_seed: int


def summarize_sample_equal(
    records: Iterable[Mapping[str, Any]],
    *,
    group_keys: Sequence[str],
    value_key: str,
    sample_key: str = "sample_id",
    secondary_weight_key: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate observations within samples, then average samples equally.

    Null values are retained as invalid observations and never converted to
    zero.  If ``secondary_weight_key`` is supplied, a clearly named
    observation-weighted sensitivity result is emitted alongside the primary
    sample-equal result.
    """

    grouped: dict[tuple[Any, ...], dict[str, list[tuple[float, float | None]]]] = defaultdict(lambda: defaultdict(list))
    seen_samples: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for record in records:
        try:
            group = tuple(record[key] for key in group_keys)
            sample_id = str(record[sample_key])
        except KeyError as exc:
            raise B1OfflineError(f"aggregation record is missing {exc.args[0]!r}") from exc
        if not sample_id:
            raise B1OfflineError("aggregation sample_id must be non-empty")
        seen_samples[group].add(sample_id)
        value = record.get(value_key)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise B1OfflineError(f"{value_key} must be finite or null")
        weight: float | None = None
        if secondary_weight_key is not None:
            raw_weight = record.get(secondary_weight_key)
            if not isinstance(raw_weight, (int, float)) or not math.isfinite(float(raw_weight)) or raw_weight < 0:
                raise B1OfflineError(f"{secondary_weight_key} must be finite and non-negative")
            weight = float(raw_weight)
        grouped[group][sample_id].append((float(value), weight))

    output: list[dict[str, Any]] = []
    for group in sorted(seen_samples, key=lambda key: tuple(str(value) for value in key)):
        sample_values = {
            sample_id: float(np.mean([value for value, _ in observations]))
            for sample_id, observations in grouped[group].items()
            if observations
        }
        values = np.asarray(list(sample_values.values()), dtype=np.float64)
        primary: dict[str, Any] | None
        if values.size:
            primary = {
                "weighting": "sample_equal",
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "p10": float(np.quantile(values, 0.10)),
                "p50": float(np.quantile(values, 0.50)),
                "p90": float(np.quantile(values, 0.90)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        else:
            primary = None
        secondary: dict[str, Any] | None = None
        if secondary_weight_key is not None:
            weighted = [
                (value, weight)
                for observations in grouped[group].values()
                for value, weight in observations
                if weight is not None and weight > 0
            ]
            total_weight = float(sum(weight for _, weight in weighted))
            if total_weight > 0:
                secondary = {
                    "weighting": f"{secondary_weight_key}_weighted",
                    "mean": float(sum(value * weight for value, weight in weighted) / total_weight),
                    "total_weight": total_weight,
                }
        output.append(
            {
                "group": dict(zip(group_keys, group)),
                "value_key": value_key,
                "sample_count_total": len(seen_samples[group]),
                "sample_count_valid": len(sample_values),
                "observation_count_valid": sum(len(values_) for values_ in grouped[group].values()),
                "primary_sample_equal": primary,
                "secondary_observation_weighted": secondary,
            }
        )
    return output


def decode_segment_ids(trace_length: int, *, minimum_length: int = 32) -> np.ndarray:
    """Return frozen normalized-progress segment IDs, or an empty array when ineligible."""

    if trace_length < 0:
        raise B1OfflineError("trace_length must be non-negative")
    if minimum_length <= 0:
        raise B1OfflineError("minimum segment length must be positive")
    if trace_length < minimum_length:
        return np.empty((0,), dtype=np.uint8)
    steps = np.arange(trace_length, dtype=np.int64)
    return np.minimum(3, (4 * steps) // trace_length).astype(np.uint8)


def _time_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the leading time axis while preserving invalid coordinates as NaN."""

    if values.ndim < 1:
        raise B1OfflineError("time-statistic input must have a leading time axis")
    finite = np.isfinite(values)
    counts = finite.sum(axis=0, dtype=np.uint32)
    if finite.all():
        quantiles = np.quantile(values, (0.10, 0.50, 0.90), axis=0)
        statistics = np.stack(
            (
                np.mean(values, axis=0),
                quantiles[1],
                quantiles[0],
                quantiles[2],
            ),
            axis=-1,
        ).astype(np.float32)
        return statistics, counts
    flattened = values.reshape(values.shape[0], -1).astype(np.float64, copy=False)
    flat_counts = counts.reshape(-1)
    output = np.full((flattened.shape[1], len(SUMMARY_STATISTICS)), np.nan, dtype=np.float32)
    for column in np.flatnonzero(flat_counts):
        valid = flattened[:, column][np.isfinite(flattened[:, column])]
        quantiles = np.quantile(valid, (0.10, 0.50, 0.90))
        output[column] = (
            np.mean(valid),
            quantiles[1],
            quantiles[0],
            quantiles[2],
        )
    return output.reshape((*values.shape[1:], len(SUMMARY_STATISTICS))), counts


def _condition_sample_equal(
    sample_means: np.ndarray,
    sample_conditions: Sequence[str],
    condition_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    if sample_means.shape[0] != len(sample_conditions):
        raise B1OfflineError("sample profile count and sample condition count differ")
    result = np.full((len(condition_names), *sample_means.shape[1:]), np.nan, dtype=np.float32)
    counts = np.zeros(result.shape, dtype=np.uint32)
    for condition_index, condition in enumerate(condition_names):
        indices = [index for index, value in enumerate(sample_conditions) if value == condition]
        if not indices:
            continue
        values = sample_means[indices].astype(np.float64, copy=False)
        valid = np.isfinite(values)
        local_counts = valid.sum(axis=0, dtype=np.uint32)
        totals = np.where(valid, values, 0.0).sum(axis=0)
        np.divide(totals, local_counts, out=result[condition_index], where=local_counts > 0)
        counts[condition_index] = local_counts
    return result, counts


def _feature_index(name: str) -> int:
    try:
        return FEATURE_NAMES.index(name)
    except ValueError as exc:  # pragma: no cover - guarded by the frozen schema
        raise B1OfflineError(f"frozen feature {name!r} is missing") from exc


def _validate_scalar_invariants(values: np.ndarray, visible_count: np.ndarray, *, source: Path) -> None:
    if values.ndim != 4 or values.shape[0] != len(visible_count) or values.shape[-1] != len(FEATURE_NAMES):
        raise B1OfflineError(f"{source}: scalar feature shape is incompatible with visibility")
    if not np.isfinite(values).all() or np.any(visible_count <= 0):
        raise B1OfflineError(f"{source}: scalar features must be finite and visibility positive")
    entropy = values[..., _feature_index("entropy")]
    normalized = values[..., _feature_index("entropy_normalized")]
    upper_entropy = np.log(visible_count.astype(np.float64))[:, None, None]
    if np.any(entropy < -INVARIANT_ATOL) or np.any(entropy > upper_entropy + INVARIANT_ATOL):
        raise B1OfflineError(f"{source}: entropy is outside [0, log(N)]")
    expected_normalized = np.divide(
        entropy,
        upper_entropy,
        out=np.zeros_like(entropy, dtype=np.float64),
        where=upper_entropy > 0,
    )
    if (
        np.any(normalized < -INVARIANT_ATOL)
        or np.any(normalized > 1 + INVARIANT_ATOL)
        or not np.allclose(normalized, expected_normalized, atol=INVARIANT_ATOL, rtol=INVARIANT_RTOL)
    ):
        raise B1OfflineError(f"{source}: normalized entropy violates its definition")

    top_mass = values[
        ..., [_feature_index(name) for name in ("top_mass_16", "top_mass_64", "top_mass_256", "top_mass_1024")]
    ]
    if (
        np.any(top_mass < -INVARIANT_ATOL)
        or np.any(top_mass > 1 + INVARIANT_ATOL)
        or np.any(np.diff(top_mass, axis=-1) < -INVARIANT_ATOL)
    ):
        raise B1OfflineError(f"{source}: Top-K mass is out of range or non-monotonic")
    bounded_names = ("sink_mass_4", "recent_mass_64", "prompt_mass", "generated_mass")
    bounded = values[..., [_feature_index(name) for name in bounded_names]]
    if np.any(bounded < -INVARIANT_ATOL) or np.any(bounded > 1 + INVARIANT_ATOL):
        raise B1OfflineError(f"{source}: probability mass feature is outside [0,1]")
    prompt = values[..., _feature_index("prompt_mass")]
    generated = values[..., _feature_index("generated_mass")]
    if not np.allclose(prompt + generated, 1.0, atol=INVARIANT_ATOL, rtol=INVARIANT_RTOL):
        raise B1OfflineError(f"{source}: prompt_mass + generated_mass differs from one")
    distance = values[..., _feature_index("mean_distance")]
    if np.any(distance < -INVARIANT_ATOL) or np.any(distance > (visible_count - 1)[:, None, None] + INVARIANT_ATOL):
        raise B1OfflineError(f"{source}: mean distance is outside the visible range")
    distance_mass = values[..., 11:]
    if (
        np.any(distance_mass < -INVARIANT_ATOL)
        or np.any(distance_mass > 1 + INVARIANT_ATOL)
        or not np.allclose(distance_mass.sum(axis=-1), 1.0, atol=INVARIANT_ATOL, rtol=INVARIANT_RTOL)
    ):
        raise B1OfflineError(f"{source}: distance-mass bins do not form a probability distribution")


def _validate_page_invariants(
    scope_mass: np.ndarray,
    coverage: np.ndarray,
    valid: np.ndarray,
    *,
    source: Path,
) -> None:
    if (
        scope_mass.shape != coverage.shape
        or scope_mass.shape != valid.shape
        or scope_mass.shape[-1] != len(PAGE_SCOPES)
    ):
        raise B1OfflineError(f"{source}: Page metric shapes disagree")
    if (
        not np.isfinite(scope_mass).all()
        or np.any(scope_mass < -INVARIANT_ATOL)
        or np.any(scope_mass > 1 + INVARIANT_ATOL)
    ):
        raise B1OfflineError(f"{source}: Page scope mass is outside [0,1]")
    if not np.allclose(scope_mass[..., 0], 1.0, atol=INVARIANT_ATOL, rtol=INVARIANT_RTOL):
        raise B1OfflineError(f"{source}: all-visible Page scope mass differs from one")
    if not valid[..., 0].all():
        raise B1OfflineError(f"{source}: all-visible Page scope must be valid")
    if (
        not np.isfinite(coverage[valid]).all()
        or not np.isnan(coverage[~valid]).all()
        or np.any(coverage[valid] < -INVARIANT_ATOL)
        or np.any(coverage[valid] > 1 + INVARIANT_ATOL)
    ):
        raise B1OfflineError(f"{source}: Page coverage violates validity or range rules")


def _ordered_samples(index: RunSetIndex) -> tuple[tuple[FrozenSample, RunBatch], ...]:
    ordered: list[tuple[FrozenSample, RunBatch]] = []
    for batch in index.batches:
        ordered.extend((index.samples[sample_id], batch) for sample_id in batch.sample_ids)
    return tuple(ordered)


def _decode_entries(batch: RunBatch, sample_id: str) -> list[tuple[str, Mapping[str, Any]]]:
    prefix = f"samples/{sample_id}/decode_stats_"
    entries = [
        (relative, entry)
        for relative, entry in batch.manifest_files.items()
        if relative.startswith(prefix) and relative.endswith(".npz")
    ]
    return sorted(entries, key=lambda item: int(item[1]["step_start"]))


def summarize_query_and_page_metrics(index: RunSetIndex) -> tuple[QueryMetricSummary, PageMetricSummary]:
    """Read each validated trace once and summarize Query and Page metrics."""

    ordered = _ordered_samples(index)
    if not ordered:
        raise B1OfflineError("run-set index contains no samples")
    dimensions = index.batches[0].schema["dimensions"]
    sample_count = len(ordered)
    layers = int(dimensions["L"])
    query_heads = int(dimensions["Hq"])
    kv_heads = int(dimensions["Hkv"])
    feature_count = len(FEATURE_NAMES)
    statistic_count = len(SUMMARY_STATISTICS)
    segment_count = len(DECODE_SEGMENTS)
    scope_count = len(PAGE_SCOPES)
    metric_count = len(PAGE_METRICS)

    query_full = np.full(
        (sample_count, len(QUERY_PHASES), layers, query_heads, feature_count, statistic_count),
        np.nan,
        dtype=np.float32,
    )
    query_full_count = np.zeros(query_full.shape[:-1], dtype=np.uint32)
    query_segments = np.full(
        (sample_count, segment_count, layers, query_heads, feature_count, statistic_count),
        np.nan,
        dtype=np.float32,
    )
    query_segment_count = np.zeros(query_segments.shape[:-1], dtype=np.uint32)
    page_full = np.full(
        (sample_count, layers, kv_heads, scope_count, metric_count, statistic_count),
        np.nan,
        dtype=np.float32,
    )
    page_full_count = np.zeros(page_full.shape[:-1], dtype=np.uint32)
    page_segments = np.full(
        (sample_count, segment_count, layers, kv_heads, scope_count, metric_count, statistic_count),
        np.nan,
        dtype=np.float32,
    )
    page_segment_count = np.zeros(page_segments.shape[:-1], dtype=np.uint32)

    sample_ids: list[str] = []
    sample_conditions: list[str] = []
    for sample_index, (sample, batch) in enumerate(ordered):
        sample_ids.append(sample.sample_id)
        sample_conditions.append(sample.condition)
        prefill_path = batch.run_dir / f"samples/{sample.sample_id}/prefill_stats.npz"
        with np.load(prefill_path) as archive:
            prefill_values = archive["scalar_features"].copy()
            prefill_visible = archive["visible_token_count"].copy()
        _validate_scalar_invariants(prefill_values, prefill_visible, source=prefill_path)
        query_full[sample_index, 0], query_full_count[sample_index, 0] = _time_statistics(prefill_values)

        decode_scalar: list[np.ndarray] = []
        decode_visible: list[np.ndarray] = []
        decode_scope_mass: list[np.ndarray] = []
        decode_coverage: list[np.ndarray] = []
        decode_valid: list[np.ndarray] = []
        decode_paths: list[Path] = []
        for relative, _ in _decode_entries(batch, sample.sample_id):
            path = batch.run_dir / relative
            decode_paths.append(path)
            with np.load(path) as archive:
                decode_scalar.append(archive["scalar_features"].copy())
                decode_visible.append(archive["visible_token_count"].copy())
                decode_scope_mass.append(archive["page_scope_mass"].copy())
                decode_coverage.append(archive["page_top32_coverage"].copy())
                decode_valid.append(archive["page_scope_valid"].copy())
        if not decode_scalar:
            raise B1OfflineError(f"{sample.sample_id}: completed sample has no Decode trace")
        scalar = np.concatenate(decode_scalar, axis=0)
        visible = np.concatenate(decode_visible, axis=0)
        scope_mass = np.concatenate(decode_scope_mass, axis=0)
        coverage = np.concatenate(decode_coverage, axis=0)
        valid = np.concatenate(decode_valid, axis=0)
        source = decode_paths[0]
        _validate_scalar_invariants(scalar, visible, source=source)
        _validate_page_invariants(scope_mass, coverage, valid, source=source)
        query_full[sample_index, 1], query_full_count[sample_index, 1] = _time_statistics(scalar)
        page_values = np.stack((scope_mass, coverage, valid.astype(np.float32)), axis=-1)
        page_full[sample_index], page_full_count[sample_index] = _time_statistics(page_values)

        segments = decode_segment_ids(len(scalar))
        if segments.size:
            for segment in range(segment_count):
                selected = segments == segment
                query_segments[sample_index, segment], query_segment_count[sample_index, segment] = _time_statistics(
                    scalar[selected]
                )
                page_segments[sample_index, segment], page_segment_count[sample_index, segment] = _time_statistics(
                    page_values[selected]
                )

    condition_names = tuple(sorted(set(sample_conditions)))
    query_condition_full, query_condition_full_count = _condition_sample_equal(
        query_full[..., 0], sample_conditions, condition_names
    )
    query_condition_segments, query_condition_segment_count = _condition_sample_equal(
        query_segments[..., 0], sample_conditions, condition_names
    )
    page_condition_full, page_condition_full_count = _condition_sample_equal(
        page_full[..., 0], sample_conditions, condition_names
    )
    page_condition_segments, page_condition_segment_count = _condition_sample_equal(
        page_segments[..., 0], sample_conditions, condition_names
    )
    query_summary = QueryMetricSummary(
        sample_ids=tuple(sample_ids),
        sample_conditions=tuple(sample_conditions),
        condition_names=condition_names,
        feature_names=tuple(FEATURE_NAMES),
        statistics=SUMMARY_STATISTICS,
        phases=QUERY_PHASES,
        decode_segments=DECODE_SEGMENTS,
        sample_full=query_full,
        sample_full_valid_count=query_full_count,
        sample_decode_segments=query_segments,
        sample_decode_segment_valid_count=query_segment_count,
        condition_full_mean=query_condition_full,
        condition_full_valid_samples=query_condition_full_count,
        condition_decode_segment_mean=query_condition_segments,
        condition_decode_segment_valid_samples=query_condition_segment_count,
    )
    page_summary = PageMetricSummary(
        sample_ids=tuple(sample_ids),
        sample_conditions=tuple(sample_conditions),
        condition_names=condition_names,
        scopes=tuple(PAGE_SCOPES),
        metrics=PAGE_METRICS,
        statistics=SUMMARY_STATISTICS,
        decode_segments=DECODE_SEGMENTS,
        sample_full=page_full,
        sample_full_valid_count=page_full_count,
        sample_decode_segments=page_segments,
        sample_decode_segment_valid_count=page_segment_count,
        condition_full_mean=page_condition_full,
        condition_full_valid_samples=page_condition_full_count,
        condition_decode_segment_mean=page_condition_segments,
        condition_decode_segment_valid_samples=page_condition_segment_count,
    )
    return query_summary, page_summary


def summarize_query_metrics(index: RunSetIndex) -> QueryMetricSummary:
    """Return the Query projection of the shared one-pass summary."""

    return summarize_query_and_page_metrics(index)[0]


def summarize_page_metrics(index: RunSetIndex) -> PageMetricSummary:
    """Return the Page projection of the shared one-pass summary."""

    return summarize_query_and_page_metrics(index)[1]


def _nanmean_axes(values: np.ndarray, axes: int | tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    return finite_mean_count(values, axes, dtype=np.float32, count_dtype=np.uint32)


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Percentile CI over the leading independent-sample axis."""

    if values.ndim < 1 or values.shape[0] == 0:
        raise B1OfflineError("bootstrap input must contain at least one independent sample")
    if replicates <= 0:
        raise B1OfflineError("bootstrap_replicates must be positive")
    return bootstrap_sample_mean_ci(values, replicates, rng)


def _pair_weighted_profiles(
    sample_values: np.ndarray,
    sample_pair_counts: np.ndarray,
    sample_conditions: Sequence[str],
    condition_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    output = np.full((len(condition_names), *sample_values.shape[1:]), np.nan, dtype=np.float32)
    count_output = np.zeros((len(condition_names), *sample_pair_counts.shape[1:]), dtype=np.uint64)
    for condition_index, condition in enumerate(condition_names):
        indices = [index for index, value in enumerate(sample_conditions) if value == condition]
        values = sample_values[indices].astype(np.float64, copy=False)
        weights = sample_pair_counts[indices].astype(np.uint64, copy=False)
        valid = np.isfinite(values)
        expanded_weights = weights[..., None]
        denominators = np.where(valid, expanded_weights, 0).sum(axis=0, dtype=np.uint64)
        numerators = np.where(valid, values * expanded_weights, 0.0).sum(axis=0, dtype=np.float64)
        np.divide(numerators, denominators, out=output[condition_index], where=denominators > 0)
        count_output[condition_index] = weights.sum(axis=0, dtype=np.uint64)
    return output, count_output


def aggregate_page_rco(
    index: RunSetIndex,
    page_rco_reports: Sequence[Path | str],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20261103,
) -> PageRCOSummary:
    """Aggregate validated per-sample Page RCO without Step-pair pseudoreplication."""

    validate_analysis_sources(index, page_rco_reports)
    ordered = _ordered_samples(index)
    sample_ids = tuple(sample.sample_id for sample, _ in ordered)
    sample_conditions = tuple(sample.condition for sample, _ in ordered)
    condition_names = tuple(sorted(set(sample_conditions)))
    sample_index = {sample_id: index_ for index_, sample_id in enumerate(sample_ids)}
    dimensions = index.batches[0].schema["dimensions"]
    layers = int(dimensions["L"])
    kv_heads = int(dimensions["Hkv"])
    shape = (
        len(sample_ids),
        layers,
        kv_heads,
        len(PAGE_SCOPES),
        len(DEFAULT_PAGE_K),
        len(DEFAULT_DELTAS),
    )
    sample_values = np.full((*shape, len(RCO_METRICS)), np.nan, dtype=np.float32)
    sample_pair_counts = np.zeros(shape, dtype=np.uint64)
    scope_index = {value: index_ for index_, value in enumerate(PAGE_SCOPES)}
    k_index = {value: index_ for index_, value in enumerate(DEFAULT_PAGE_K)}
    delta_index = {value: index_ for index_, value in enumerate(DEFAULT_DELTAS)}
    seen: set[tuple[int, int, int, int, int, int]] = set()
    for report_value in page_rco_reports:
        report = _read_json(Path(report_value).resolve())
        for record in report["records"]:
            coordinate = (
                sample_index[str(record["sample_id"])],
                int(record["layer_id"]),
                int(record["kv_head"]),
                scope_index[str(record["scope"])],
                k_index[int(record["k"])],
                delta_index[int(record["delta"])],
            )
            if coordinate in seen:
                raise B1OfflineError(f"duplicate Page RCO coordinate after source validation: {coordinate}")
            seen.add(coordinate)
            sample_pair_counts[coordinate] = int(record["valid_pair_count"])
            if int(record["valid_pair_count"]) > 0:
                sample_values[coordinate] = [float(record[name]) for name in RCO_METRICS]
    if len(seen) != int(np.prod(shape)):
        raise B1OfflineError("Page RCO reports do not cover the complete indexed coordinate grid")

    condition_head_mean, condition_head_valid = _condition_sample_equal(
        sample_values, sample_conditions, condition_names
    )
    condition_head_weighted, condition_head_pair_count = _pair_weighted_profiles(
        sample_values, sample_pair_counts, sample_conditions, condition_names
    )
    sample_scalar, _ = _nanmean_axes(sample_values, axes=(1, 2))
    condition_scalar_mean, condition_scalar_valid = _condition_sample_equal(
        sample_scalar, sample_conditions, condition_names
    )

    scalar_weighted = np.full(
        (len(condition_names), len(PAGE_SCOPES), len(DEFAULT_PAGE_K), len(DEFAULT_DELTAS), len(RCO_METRICS)),
        np.nan,
        dtype=np.float32,
    )
    scalar_pair_count = np.zeros(scalar_weighted.shape[:-1], dtype=np.uint64)
    for condition_index, condition in enumerate(condition_names):
        indices = [index_ for index_, value in enumerate(sample_conditions) if value == condition]
        values = sample_values[indices].astype(np.float64, copy=False)
        weights = sample_pair_counts[indices].astype(np.uint64, copy=False)
        expanded_weights = weights[..., None]
        valid = np.isfinite(values)
        reduce_axes = (0, 1, 2)
        denominators = np.where(valid, expanded_weights, 0).sum(axis=reduce_axes, dtype=np.uint64)
        numerators = np.where(valid, values * expanded_weights, 0.0).sum(axis=reduce_axes, dtype=np.float64)
        np.divide(
            numerators,
            denominators,
            out=scalar_weighted[condition_index],
            where=denominators > 0,
        )
        scalar_pair_count[condition_index] = weights.sum(axis=reduce_axes, dtype=np.uint64)

    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    scalar_ci = np.full((*condition_scalar_mean.shape, 2), np.nan, dtype=np.float32)
    for condition_index, condition in enumerate(condition_names):
        indices = [index_ for index_, value in enumerate(sample_conditions) if value == condition]
        scalar_ci[condition_index] = _bootstrap_mean_ci(
            sample_scalar[indices], replicates=bootstrap_replicates, rng=rng
        )

    pair_members: dict[str, dict[int, int]] = defaultdict(dict)
    for index_, (sample, _) in enumerate(ordered):
        if sample.dataset == "pg19" and sample.pair_id:
            pair_members[sample.pair_id][sample.target_prompt_length] = index_
    pg19_pair_ids = tuple(sorted(pair_members))
    pair_head_difference = np.full((len(pg19_pair_ids), *sample_values.shape[1:]), np.nan, dtype=np.float32)
    pair_scalar_difference = np.full((len(pg19_pair_ids), *sample_scalar.shape[1:]), np.nan, dtype=np.float32)
    for pair_index, pair_id in enumerate(pg19_pair_ids):
        members = pair_members[pair_id]
        if set(members) != {8192, 32768}:
            raise B1OfflineError(f"PG19 pair {pair_id!r} is incomplete during RCO aggregation")
        pair_head_difference[pair_index] = sample_values[members[32768]] - sample_values[members[8192]]
        pair_scalar_difference[pair_index] = sample_scalar[members[32768]] - sample_scalar[members[8192]]
    pair_head_mean, _ = _nanmean_axes(pair_head_difference, axes=0)
    pair_scalar_mean, _ = _nanmean_axes(pair_scalar_difference, axes=0)
    pair_scalar_ci = _bootstrap_mean_ci(
        pair_scalar_difference,
        replicates=bootstrap_replicates,
        rng=np.random.Generator(np.random.PCG64(bootstrap_seed)),
    )
    return PageRCOSummary(
        sample_ids=sample_ids,
        sample_conditions=sample_conditions,
        condition_names=condition_names,
        scopes=tuple(PAGE_SCOPES),
        k_values=DEFAULT_PAGE_K,
        deltas=DEFAULT_DELTAS,
        metrics=RCO_METRICS,
        sample_head_values=sample_values,
        sample_head_valid_pair_count=sample_pair_counts,
        condition_head_sample_equal_mean=condition_head_mean,
        condition_head_valid_samples=condition_head_valid,
        condition_head_pair_weighted_mean=condition_head_weighted,
        condition_head_valid_pair_count=condition_head_pair_count,
        condition_scalar_sample_equal_mean=condition_scalar_mean,
        condition_scalar_valid_samples=condition_scalar_valid,
        condition_scalar_pair_weighted_mean=scalar_weighted,
        condition_scalar_valid_pair_count=scalar_pair_count,
        condition_scalar_bootstrap_ci=scalar_ci,
        pg19_pair_ids=pg19_pair_ids,
        pg19_pair_head_difference_32k_minus_8k=pair_head_difference,
        pg19_pair_head_mean_difference=pair_head_mean,
        pg19_pair_scalar_difference_32k_minus_8k=pair_scalar_difference,
        pg19_pair_scalar_mean_difference=pair_scalar_mean,
        pg19_pair_scalar_bootstrap_ci=pair_scalar_ci,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )


def build_analysis_plan(
    index: RunSetIndex,
    validation: ValidationReport,
    *,
    analysis_id: str,
    git_commit: str | None,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not analysis_id or Path(analysis_id).name != analysis_id:
        raise B1OfflineError("analysis_id must be one non-empty path component")
    if (
        validation.source_run_set_sha256 != index.run_set_sha256
        or validation.sample_count != len(index.samples)
        or validation.batch_count != len(index.batches)
    ):
        raise B1OfflineError("validation report does not belong to this run-set index")
    resolved_parameters = dict(DEFAULT_ANALYSIS_PARAMETERS if parameters is None else parameters)
    return {
        "schema_version": ANALYSIS_PLAN_SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "source_run_set": {"path": str(index.manifest_path), "sha256": index.run_set_sha256},
        "source_page_rco_reports": [dict(item) for item in validation.page_rco_reports],
        "source_schema_version": SCHEMA_VERSION,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "git_commit": git_commit,
        "parameters": resolved_parameters,
        "parameters_sha256": canonical_json_sha256(resolved_parameters),
        "sample_count": len(index.samples),
        "condition_counts": dict(index.condition_counts),
    }


def build_analysis_manifest(
    *,
    index: RunSetIndex,
    plan_path: Path | str,
    output_root: Path | str,
    output_paths: Sequence[Path | str],
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    """Build, but do not write, a completed non-self-hashing manifest."""

    root = Path(output_root).resolve()
    plan = Path(plan_path).resolve()
    if plan != root / "analysis_plan.json":
        raise B1OfflineError("analysis plan must be output_root/analysis_plan.json")
    payload = _read_json(plan)
    if payload.get("schema_version") != ANALYSIS_PLAN_SCHEMA_VERSION:
        raise B1OfflineError("analysis plan has an unsupported schema")
    if payload.get("source_run_set", {}).get("sha256") != index.run_set_sha256:
        raise B1OfflineError("analysis plan and source run set differ")
    outputs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in output_paths:
        path = Path(value).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise B1OfflineError(f"analysis output is outside output_root: {path}") from exc
        if relative in {"analysis_plan.json", "analysis_manifest.json"} or relative in seen:
            raise B1OfflineError(f"invalid or duplicate analysis output {relative!r}")
        if not path.is_file():
            raise B1OfflineError(f"missing analysis output: {path}")
        seen.add(relative)
        outputs.append({"relative_path": relative, "sha256": sha256_file(path), "file_bytes": path.stat().st_size})
    if not outputs:
        raise B1OfflineError("completed analysis manifest requires at least one output")
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_id": payload["analysis_id"],
        "status": "completed",
        "plan_sha256": sha256_file(plan),
        "source_run_set_sha256": index.run_set_sha256,
        "sample_count": len(index.samples),
        "condition_counts": dict(index.condition_counts),
        "outputs": sorted(outputs, key=lambda item: str(item["relative_path"])),
        "warnings": [str(value) for value in warnings],
    }
