"""GQA-aligned relationships between frozen B1 offline Head profiles.

The module consumes the Stage3 and Stage4 numeric artifacts only.  It never
opens model weights or raw Attention traces, and it does not write files.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from experiments.head_analysis.analysis.b1_offline import B1OfflineError

PRIMARY_RELATION_DELTAS = (1, 32, 128, 256)
QUERY_RELATION_FEATURES = (
    "entropy_normalized",
    "top_mass_16",
    "top_mass_64",
    "top_mass_256",
    "top_mass_1024",
    "prompt_mass",
    "mean_distance",
)


@dataclass(frozen=True)
class RelationshipInputs:
    sample_ids: tuple[str, ...]
    sample_conditions: tuple[str, ...]
    condition_names: tuple[str, ...]
    query_feature_names: tuple[str, ...]
    query_statistics: tuple[str, ...]
    query_phases: tuple[str, ...]
    query_sample_full: np.ndarray  # [N,P,L,Hq,F,Stat]
    page_scopes: tuple[str, ...]
    page_metrics: tuple[str, ...]
    page_sample_full: np.ndarray  # [N,L,Hkv,S,M,Stat]
    page_rco_k_values: tuple[int, ...]
    page_rco_deltas: tuple[int, ...]
    page_rco_metrics: tuple[str, ...]
    page_rco_sample_head_values: np.ndarray  # [N,L,Hkv,S,K,D,M]
    token_k_values: tuple[int, ...]
    token_deltas: tuple[int, ...]
    token_metrics: tuple[str, ...]
    token_sample_head_metrics: np.ndarray  # [N,L,Hq,K,D,M]


@dataclass(frozen=True)
class CorrelationEstimate:
    valid_head_count: int
    spearman: float
    spearman_bootstrap_ci: np.ndarray  # [2]
    pearson: float
    pearson_bootstrap_ci: np.ndarray  # [2]
    layer_centered_valid_head_count: int
    layer_centered_spearman: float
    layer_centered_spearman_bootstrap_ci: np.ndarray  # [2]
    layer_centered_pearson: float
    layer_centered_pearson_bootstrap_ci: np.ndarray  # [2]
    per_layer_valid_head_count: np.ndarray  # [L]
    per_layer_spearman: np.ndarray  # [L]
    per_layer_pearson: np.ndarray  # [L]


@dataclass(frozen=True)
class RelationshipRecord:
    relation_id: str
    family: str
    condition_x: str
    condition_y: str
    profile_x: str
    profile_y: str
    bootstrap_mode: str
    estimate: CorrelationEstimate


@dataclass(frozen=True)
class MetricRelationshipSummary:
    sample_ids: tuple[str, ...]
    sample_conditions: tuple[str, ...]
    condition_names: tuple[str, ...]
    primary_deltas: tuple[int, ...]
    bootstrap_replicates: int
    bootstrap_seed: int
    records: tuple[RelationshipRecord, ...]


def _strings(archive: Mapping[str, np.ndarray], name: str) -> tuple[str, ...]:
    if name not in archive:
        raise B1OfflineError(f"relationship input is missing {name!r}")
    array = archive[name]
    if array.ndim != 1 or array.dtype.kind not in "US":
        raise B1OfflineError(f"{name} must be a one-dimensional string array")
    return tuple(str(value) for value in array.tolist())


def _integers(archive: Mapping[str, np.ndarray], name: str) -> tuple[int, ...]:
    if name not in archive:
        raise B1OfflineError(f"relationship input is missing {name!r}")
    array = archive[name]
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise B1OfflineError(f"{name} must be a one-dimensional integer array")
    return tuple(int(value) for value in array.tolist())


def _float_array(archive: Mapping[str, np.ndarray], name: str, dimensions: int) -> np.ndarray:
    if name not in archive:
        raise B1OfflineError(f"relationship input is missing {name!r}")
    array = archive[name]
    if array.ndim != dimensions or array.dtype != np.float32:
        raise B1OfflineError(f"{name} must be float32 with {dimensions} dimensions")
    return array.copy()


def load_relationship_inputs(
    stage3_profiles_path: Path | str,
    stage4_profiles_path: Path | str,
) -> RelationshipInputs:
    """Load and align immutable Stage3/Stage4 NPZ profiles."""

    try:
        with np.load(stage3_profiles_path, allow_pickle=False) as stage3:
            sample_ids = _strings(stage3, "sample_ids")
            sample_conditions = _strings(stage3, "sample_conditions")
            condition_names = _strings(stage3, "condition_names")
            query_feature_names = _strings(stage3, "query_feature_names")
            query_statistics = _strings(stage3, "query_statistics")
            query_phases = _strings(stage3, "query_phases")
            query_sample_full = _float_array(stage3, "query_sample_full", 6)
            page_scopes = _strings(stage3, "page_scopes")
            page_metrics = _strings(stage3, "page_metrics")
            page_sample_full = _float_array(stage3, "page_sample_full", 6)
            page_rco_k_values = _integers(stage3, "rco_k_values")
            page_rco_deltas = _integers(stage3, "rco_deltas")
            page_rco_metrics = _strings(stage3, "rco_metrics")
            page_rco_sample = _float_array(stage3, "rco_sample_head_values", 7)
        with np.load(stage4_profiles_path, allow_pickle=False) as stage4:
            token_sample_ids = _strings(stage4, "sample_ids")
            token_sample_conditions = _strings(stage4, "sample_conditions")
            token_condition_names = _strings(stage4, "condition_names")
            token_k_values = _integers(stage4, "token_k_values")
            token_deltas = _integers(stage4, "token_deltas")
            token_metrics = _strings(stage4, "token_metrics")
            token_sample = _float_array(stage4, "token_sample_head_metrics", 6)
    except (OSError, ValueError) as exc:
        raise B1OfflineError(f"unable to load Stage3/Stage4 relationship inputs: {exc}") from exc

    if (sample_ids, sample_conditions, condition_names) != (
        token_sample_ids,
        token_sample_conditions,
        token_condition_names,
    ):
        raise B1OfflineError("Stage3 and Stage4 sample/condition axes do not match exactly")
    inputs = RelationshipInputs(
        sample_ids=sample_ids,
        sample_conditions=sample_conditions,
        condition_names=condition_names,
        query_feature_names=query_feature_names,
        query_statistics=query_statistics,
        query_phases=query_phases,
        query_sample_full=query_sample_full,
        page_scopes=page_scopes,
        page_metrics=page_metrics,
        page_sample_full=page_sample_full,
        page_rco_k_values=page_rco_k_values,
        page_rco_deltas=page_rco_deltas,
        page_rco_metrics=page_rco_metrics,
        page_rco_sample_head_values=page_rco_sample,
        token_k_values=token_k_values,
        token_deltas=token_deltas,
        token_metrics=token_metrics,
        token_sample_head_metrics=token_sample,
    )
    _validate_inputs(inputs)
    return inputs


def _validate_inputs(inputs: RelationshipInputs) -> None:
    sample_count = len(inputs.sample_ids)
    if sample_count == 0 or len(set(inputs.sample_ids)) != sample_count:
        raise B1OfflineError("relationship inputs require unique non-empty sample IDs")
    if len(inputs.sample_conditions) != sample_count:
        raise B1OfflineError("relationship sample condition count is inconsistent")
    if tuple(sorted(set(inputs.sample_conditions))) != inputs.condition_names:
        raise B1OfflineError("relationship condition_names must be sorted and complete")
    query = inputs.query_sample_full
    page = inputs.page_sample_full
    page_rco = inputs.page_rco_sample_head_values
    token = inputs.token_sample_head_metrics
    if query.shape[0] != sample_count or page.shape[0] != sample_count:
        raise B1OfflineError("Stage3 sample arrays do not match the sample axis")
    if page_rco.shape[0] != sample_count or token.shape[0] != sample_count:
        raise B1OfflineError("RCO sample arrays do not match the sample axis")
    layers, query_heads = query.shape[2:4]
    if page.shape[1] != layers or page_rco.shape[1] != layers or token.shape[1:3] != (layers, query_heads):
        raise B1OfflineError("Layer/Query-Head axes do not align across Stage3 and Stage4")
    kv_heads = page.shape[2]
    if page_rco.shape[2] != kv_heads or query_heads != 4 * kv_heads:
        raise B1OfflineError("frozen B1 relationship analysis requires contiguous 4:1 GQA")
    expected_axes = (
        (query.shape[1], len(inputs.query_phases), "query phase"),
        (query.shape[4], len(inputs.query_feature_names), "query feature"),
        (query.shape[5], len(inputs.query_statistics), "query statistic"),
        (page.shape[3], len(inputs.page_scopes), "page scope"),
        (page.shape[4], len(inputs.page_metrics), "page metric"),
        (page.shape[5], len(inputs.query_statistics), "page statistic"),
        (page_rco.shape[3], len(inputs.page_scopes), "Page RCO scope"),
        (page_rco.shape[4], len(inputs.page_rco_k_values), "Page RCO k"),
        (page_rco.shape[5], len(inputs.page_rco_deltas), "Page RCO delta"),
        (page_rco.shape[6], len(inputs.page_rco_metrics), "Page RCO metric"),
        (token.shape[3], len(inputs.token_k_values), "Token k"),
        (token.shape[4], len(inputs.token_deltas), "Token delta"),
        (token.shape[5], len(inputs.token_metrics), "Token metric"),
    )
    for actual, expected, label in expected_axes:
        if actual != expected:
            raise B1OfflineError(f"{label} axis length {actual} differs from metadata {expected}")
    required = {
        "query phases": (inputs.query_phases, ("decode",)),
        "query statistics": (inputs.query_statistics, ("mean",)),
        "query features": (inputs.query_feature_names, QUERY_RELATION_FEATURES),
        "page scopes": (inputs.page_scopes, ("all_visible", "prompt_only")),
        "page metrics": (inputs.page_metrics, ("top32_coverage",)),
        "Page RCO k": (inputs.page_rco_k_values, (32,)),
        "Page RCO deltas": (inputs.page_rco_deltas, PRIMARY_RELATION_DELTAS),
        "Page RCO metrics": (inputs.page_rco_metrics, ("rco_raw_mean",)),
        "Token k": (inputs.token_k_values, (32,)),
        "Token deltas": (inputs.token_deltas, PRIMARY_RELATION_DELTAS),
        "Token metrics": (inputs.token_metrics, ("rco_raw_mean",)),
    }
    for label, (values, expected) in required.items():
        missing = [value for value in expected if value not in values]
        if missing:
            raise B1OfflineError(f"{label} is missing required values {missing}")


def _finite_mean(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    output = np.full(count.shape, np.nan, dtype=np.float32)
    np.divide(
        np.where(valid, values, 0.0).sum(axis=axis, dtype=np.float64),
        count,
        out=output,
        where=count > 0,
    )
    return output


def _gqa_mean(values: np.ndarray, kv_heads: int) -> np.ndarray:
    if values.ndim != 4 or values.shape[2] != 4 * kv_heads:
        raise B1OfflineError("GQA input must be [N,L,4*Hkv,...]")
    return _finite_mean(values.reshape(values.shape[0], values.shape[1], kv_heads, 4, values.shape[3]), axis=3)


def build_primary_profiles(inputs: RelationshipInputs) -> dict[str, np.ndarray]:
    """Build sample-level ``[N,L,Hkv]`` profiles for frozen primary relations."""

    _validate_inputs(inputs)
    profiles: dict[str, np.ndarray] = {}
    decode = inputs.query_phases.index("decode")
    mean = inputs.query_statistics.index("mean")
    kv_heads = inputs.page_sample_full.shape[2]
    for feature in QUERY_RELATION_FEATURES:
        feature_index = inputs.query_feature_names.index(feature)
        values = inputs.query_sample_full[:, decode, :, :, feature_index, mean]
        profiles[f"query:{feature}"] = _gqa_mean(values[..., None], kv_heads)[..., 0]
    coverage = inputs.page_metrics.index("top32_coverage")
    for scope_index, scope in enumerate(inputs.page_scopes):
        profiles[f"page_coverage:{scope}"] = inputs.page_sample_full[:, :, :, scope_index, coverage, mean].copy()
    page_k = inputs.page_rco_k_values.index(32)
    page_metric = inputs.page_rco_metrics.index("rco_raw_mean")
    token_k = inputs.token_k_values.index(32)
    token_metric = inputs.token_metrics.index("rco_raw_mean")
    for delta in PRIMARY_RELATION_DELTAS:
        page_delta = inputs.page_rco_deltas.index(delta)
        token_delta = inputs.token_deltas.index(delta)
        for scope_index, scope in enumerate(inputs.page_scopes):
            profiles[f"page_rco:{scope}:k32:d{delta}"] = inputs.page_rco_sample_head_values[
                :, :, :, scope_index, page_k, page_delta, page_metric
            ].copy()
        token_values = inputs.token_sample_head_metrics[:, :, :, token_k, token_delta, token_metric]
        profiles[f"token_rco:k32:d{delta}"] = _gqa_mean(token_values[..., None], kv_heads)[..., 0]
    return profiles


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _correlation(x: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid].astype(np.float64, copy=False)
    y = y[valid].astype(np.float64, copy=False)
    count = len(x)
    if count < 3:
        return count, float("nan"), float("nan")

    def pearson(left: np.ndarray, right: np.ndarray) -> float:
        left = left - left.mean()
        right = right - right.mean()
        denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
        return float(np.dot(left, right) / denominator) if denominator > 0 else float("nan")

    return count, pearson(_average_ranks(x), _average_ranks(y)), pearson(x, y)


def _layer_center(values: np.ndarray) -> np.ndarray:
    return values - _finite_mean(values, axis=1)[:, None]


def _seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _bootstrap_mean(values: np.ndarray, draws: np.ndarray) -> np.ndarray:
    return _finite_mean(values[draws], axis=1)


def _ci(values: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    if not values.size:
        return np.full(2, np.nan, dtype=np.float32)
    return np.asarray(np.quantile(values, (0.025, 0.975)), dtype=np.float32)


def _estimate(
    x: np.ndarray,
    y: np.ndarray,
    bootstrap_x: np.ndarray,
    bootstrap_y: np.ndarray,
) -> CorrelationEstimate:
    count, spearman, pearson = _correlation(x.ravel(), y.ravel())
    centered_x = _layer_center(x)
    centered_y = _layer_center(y)
    centered_count, centered_spearman, centered_pearson = _correlation(centered_x.ravel(), centered_y.ravel())
    layer_count = np.zeros(x.shape[0], dtype=np.uint32)
    layer_spearman = np.full(x.shape[0], np.nan, dtype=np.float32)
    layer_pearson = np.full(x.shape[0], np.nan, dtype=np.float32)
    for layer in range(x.shape[0]):
        local_count, local_spearman, local_pearson = _correlation(x[layer], y[layer])
        layer_count[layer] = local_count
        layer_spearman[layer] = local_spearman
        layer_pearson[layer] = local_pearson
    bootstrap_spearman = np.full(len(bootstrap_x), np.nan, dtype=np.float32)
    bootstrap_pearson = np.full(len(bootstrap_x), np.nan, dtype=np.float32)
    bootstrap_centered_spearman = np.full(len(bootstrap_x), np.nan, dtype=np.float32)
    bootstrap_centered_pearson = np.full(len(bootstrap_x), np.nan, dtype=np.float32)
    for replicate, (local_x, local_y) in enumerate(zip(bootstrap_x, bootstrap_y)):
        _, bootstrap_spearman[replicate], bootstrap_pearson[replicate] = _correlation(local_x.ravel(), local_y.ravel())
        _, bootstrap_centered_spearman[replicate], bootstrap_centered_pearson[replicate] = _correlation(
            _layer_center(local_x).ravel(), _layer_center(local_y).ravel()
        )
    return CorrelationEstimate(
        valid_head_count=count,
        spearman=spearman,
        spearman_bootstrap_ci=_ci(bootstrap_spearman),
        pearson=pearson,
        pearson_bootstrap_ci=_ci(bootstrap_pearson),
        layer_centered_valid_head_count=centered_count,
        layer_centered_spearman=centered_spearman,
        layer_centered_spearman_bootstrap_ci=_ci(bootstrap_centered_spearman),
        layer_centered_pearson=centered_pearson,
        layer_centered_pearson_bootstrap_ci=_ci(bootstrap_centered_pearson),
        per_layer_valid_head_count=layer_count,
        per_layer_spearman=layer_spearman,
        per_layer_pearson=layer_pearson,
    )


def _condition_rows(inputs: RelationshipInputs, condition: str) -> np.ndarray:
    return np.flatnonzero(np.asarray(inputs.sample_conditions) == condition)


def _pg19_pair_rows(
    inputs: RelationshipInputs, condition_x: str, condition_y: str
) -> tuple[np.ndarray, np.ndarray] | None:
    if {condition_x, condition_y} != {"pg19:8k", "pg19:32k"}:
        return None

    def members(condition: str) -> dict[str, int]:
        suffix = "_8k" if condition.endswith(":8k") else "_32k"
        output: dict[str, int] = {}
        for index in _condition_rows(inputs, condition):
            sample_id = inputs.sample_ids[int(index)]
            if not sample_id.endswith(suffix):
                raise B1OfflineError(f"PG19 sample {sample_id!r} does not match condition {condition!r}")
            output[sample_id[: -len(suffix)]] = int(index)
        return output

    x_members = members(condition_x)
    y_members = members(condition_y)
    if set(x_members) != set(y_members) or not x_members:
        raise B1OfflineError("PG19 cross-condition relationship requires complete 8K/32K pairs")
    pair_ids = sorted(x_members)
    return np.asarray([x_members[pair_id] for pair_id in pair_ids]), np.asarray(
        [y_members[pair_id] for pair_id in pair_ids]
    )


def _within_record(
    inputs: RelationshipInputs,
    profiles: Mapping[str, np.ndarray],
    *,
    condition: str,
    profile_x: str,
    profile_y: str,
    family: str,
    replicates: int,
    seed: int,
) -> RelationshipRecord:
    relation_id = f"within:{condition}:{profile_x}__{profile_y}"
    rows = _condition_rows(inputs, condition)
    values_x = profiles[profile_x][rows]
    values_y = profiles[profile_y][rows]
    mean_x = _finite_mean(values_x, axis=0)
    mean_y = _finite_mean(values_y, axis=0)
    if np.any(np.isfinite(mean_x) & np.isfinite(mean_y)):
        rng = np.random.Generator(np.random.PCG64(_seed(seed, relation_id)))
        draws = rng.integers(0, len(rows), size=(replicates, len(rows)))
        bootstrap_x = _bootstrap_mean(values_x, draws)
        bootstrap_y = _bootstrap_mean(values_y, draws)
    else:
        bootstrap_x = np.empty((0, *mean_x.shape), dtype=np.float32)
        bootstrap_y = np.empty((0, *mean_y.shape), dtype=np.float32)
    return RelationshipRecord(
        relation_id=relation_id,
        family=family,
        condition_x=condition,
        condition_y=condition,
        profile_x=profile_x,
        profile_y=profile_y,
        bootstrap_mode="paired_samples_within_condition",
        estimate=_estimate(
            mean_x,
            mean_y,
            bootstrap_x,
            bootstrap_y,
        ),
    )


def _cross_condition_record(
    inputs: RelationshipInputs,
    profiles: Mapping[str, np.ndarray],
    *,
    condition_x: str,
    condition_y: str,
    profile: str,
    replicates: int,
    seed: int,
) -> RelationshipRecord:
    relation_id = f"cross:{profile}:{condition_x}__{condition_y}"
    paired_rows = _pg19_pair_rows(inputs, condition_x, condition_y)
    mean_x: np.ndarray
    mean_y: np.ndarray
    if paired_rows is not None:
        rows_x, rows_y = paired_rows
        mode = "paired_pg19_samples"
    else:
        rows_x = _condition_rows(inputs, condition_x)
        rows_y = _condition_rows(inputs, condition_y)
        mode = "independent_condition_samples"
    values_x = profiles[profile][rows_x]
    values_y = profiles[profile][rows_y]
    mean_x = _finite_mean(values_x, axis=0)
    mean_y = _finite_mean(values_y, axis=0)
    if np.any(np.isfinite(mean_x) & np.isfinite(mean_y)):
        if paired_rows is not None:
            rng = np.random.Generator(np.random.PCG64(_seed(seed, relation_id)))
            draws = rng.integers(0, len(rows_x), size=(replicates, len(rows_x)))
            bootstrap_x = _bootstrap_mean(values_x, draws)
            bootstrap_y = _bootstrap_mean(values_y, draws)
        else:
            rng_x = np.random.Generator(np.random.PCG64(_seed(seed, relation_id + ":x")))
            rng_y = np.random.Generator(np.random.PCG64(_seed(seed, relation_id + ":y")))
            draws_x = rng_x.integers(0, len(rows_x), size=(replicates, len(rows_x)))
            draws_y = rng_y.integers(0, len(rows_y), size=(replicates, len(rows_y)))
            bootstrap_x = _bootstrap_mean(values_x, draws_x)
            bootstrap_y = _bootstrap_mean(values_y, draws_y)
    else:
        bootstrap_x = np.empty((0, *mean_x.shape), dtype=np.float32)
        bootstrap_y = np.empty((0, *mean_y.shape), dtype=np.float32)
    return RelationshipRecord(
        relation_id=relation_id,
        family="cross_condition_head_identity",
        condition_x=condition_x,
        condition_y=condition_y,
        profile_x=profile,
        profile_y=profile,
        bootstrap_mode=mode,
        estimate=_estimate(
            mean_x,
            mean_y,
            bootstrap_x,
            bootstrap_y,
        ),
    )


def analyze_metric_relationships(
    inputs: RelationshipInputs,
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20261103,
    progress_callback: Callable[[int, int, str, float], None] | None = None,
) -> MetricRelationshipSummary:
    """Compute the frozen Stage5 within-condition and cross-condition relations."""

    _validate_inputs(inputs)
    if bootstrap_replicates <= 0:
        raise B1OfflineError("bootstrap_replicates must be positive")
    profiles = build_primary_profiles(inputs)
    records: list[RelationshipRecord] = []
    relation_total = len(inputs.condition_names) * 50 + len(tuple(combinations(inputs.condition_names, 2))) * 12

    def append(record: RelationshipRecord, started: float) -> None:
        records.append(record)
        if progress_callback is not None:
            progress_callback(len(records), relation_total, record.relation_id, time.perf_counter() - started)

    for condition in inputs.condition_names:
        for delta in PRIMARY_RELATION_DELTAS:
            all_visible = f"page_rco:all_visible:k32:d{delta}"
            prompt_only = f"page_rco:prompt_only:k32:d{delta}"
            token = f"token_rco:k32:d{delta}"
            pairs = [
                ("concentration_page_stability", "query:entropy_normalized", all_visible),
                ("concentration_page_stability", "query:top_mass_16", all_visible),
                ("concentration_page_stability", "query:top_mass_64", all_visible),
                ("concentration_page_stability", "query:top_mass_256", all_visible),
                ("concentration_page_stability", "query:top_mass_1024", all_visible),
                ("prompt_attention_page_stability", "query:prompt_mass", prompt_only),
                ("distance_page_stability", "query:mean_distance", all_visible),
                ("coverage_page_stability", "page_coverage:all_visible", all_visible),
                ("coverage_page_stability", "page_coverage:prompt_only", prompt_only),
                ("token_page_stability", token, all_visible),
                ("token_page_stability", token, prompt_only),
            ]
            for family, profile_x, profile_y in pairs:
                started = time.perf_counter()
                append(
                    _within_record(
                        inputs,
                        profiles,
                        condition=condition,
                        profile_x=profile_x,
                        profile_y=profile_y,
                        family=family,
                        replicates=bootstrap_replicates,
                        seed=bootstrap_seed,
                    ),
                    started,
                )
        for scope in inputs.page_scopes:
            short = f"page_rco:{scope}:k32:d1"
            for delta in (32, 128, 256):
                started = time.perf_counter()
                append(
                    _within_record(
                        inputs,
                        profiles,
                        condition=condition,
                        profile_x=short,
                        profile_y=f"page_rco:{scope}:k32:d{delta}",
                        family="short_long_page_stability",
                        replicates=bootstrap_replicates,
                        seed=bootstrap_seed,
                    ),
                    started,
                )
    cross_profiles = [
        f"page_rco:{scope}:k32:d{delta}" for scope in inputs.page_scopes for delta in PRIMARY_RELATION_DELTAS
    ] + [f"token_rco:k32:d{delta}" for delta in PRIMARY_RELATION_DELTAS]
    for condition_x, condition_y in combinations(inputs.condition_names, 2):
        for profile in cross_profiles:
            started = time.perf_counter()
            append(
                _cross_condition_record(
                    inputs,
                    profiles,
                    condition_x=condition_x,
                    condition_y=condition_y,
                    profile=profile,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed,
                ),
                started,
            )
    if len(records) != relation_total:
        raise B1OfflineError(f"internal Stage5 relation count {len(records)} differs from expected {relation_total}")
    return MetricRelationshipSummary(
        sample_ids=inputs.sample_ids,
        sample_conditions=inputs.sample_conditions,
        condition_names=inputs.condition_names,
        primary_deltas=PRIMARY_RELATION_DELTAS,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        records=tuple(records),
    )
