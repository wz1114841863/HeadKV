"""Exact offline stability metrics for stored Decode Token Top-K positions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from experiments.head_analysis.analysis.b1_offline import B1OfflineError, RunBatch, RunSetIndex
from experiments.head_analysis.collectors.attention_stats import UINT32_SENTINEL

TOKEN_METRICS = (
    "overlap_raw_mean",
    "rco_raw_mean",
    "rco_clipped_mean",
    "token_entry_rate_mean",
    "token_entry_rate_p90",
    "rank_displacement_mean",
)
AGE_STATISTICS = ("median", "p90")


@dataclass(frozen=True)
class SampleTokenStability:
    sample_id: str
    k_values: tuple[int, ...]
    deltas: tuple[int, ...]
    metrics: tuple[str, ...]
    head_metrics: np.ndarray  # [L,Hq,K,D,M]
    valid_pair_count: np.ndarray  # [L,Hq,K,D]
    rank_valid_pair_count: np.ndarray  # [L,Hq,K,D]
    age_statistics: tuple[str, ...]
    head_age: np.ndarray  # [L,Hq,K,AgeStat]
    age_valid_token_count: np.ndarray  # [L,Hq,K]


@dataclass(frozen=True)
class TokenStabilitySummary:
    sample_ids: tuple[str, ...]
    sample_conditions: tuple[str, ...]
    condition_names: tuple[str, ...]
    k_values: tuple[int, ...]
    deltas: tuple[int, ...]
    metrics: tuple[str, ...]
    age_statistics: tuple[str, ...]
    sample_head_metrics: np.ndarray  # [N,L,Hq,K,D,M]
    sample_head_valid_pair_count: np.ndarray  # [N,L,Hq,K,D]
    sample_head_rank_valid_pair_count: np.ndarray  # [N,L,Hq,K,D]
    sample_head_age: np.ndarray  # [N,L,Hq,K,AgeStat]
    sample_head_age_valid_token_count: np.ndarray  # [N,L,Hq,K]
    condition_head_sample_equal_mean: np.ndarray  # [Cond,L,Hq,K,D,M]
    condition_head_valid_samples: np.ndarray  # [Cond,L,Hq,K,D,M]
    condition_scalar_sample_equal_mean: np.ndarray  # [Cond,K,D,M]
    condition_scalar_valid_samples: np.ndarray  # [Cond,K,D,M]
    condition_scalar_bootstrap_ci: np.ndarray  # [Cond,K,D,M,2]
    condition_age_head_sample_equal_mean: np.ndarray  # [Cond,L,Hq,K,AgeStat]
    pg19_pair_ids: tuple[str, ...]
    pg19_pair_scalar_mean_difference_32k_minus_8k: np.ndarray  # [K,D,M]
    pg19_pair_scalar_bootstrap_ci: np.ndarray  # [K,D,M,2]
    bootstrap_replicates: int
    bootstrap_seed: int


def _step_validity(tokens: np.ndarray, visible_count: np.ndarray, k: int) -> np.ndarray:
    subset = tokens[..., :k]
    no_sentinel = np.all(subset != UINT32_SENTINEL, axis=-1)
    in_range = np.all(subset < visible_count[:, None, None, None], axis=-1)
    ordered = np.sort(subset, axis=-1)
    unique = np.all(np.diff(ordered.astype(np.int64), axis=-1) != 0, axis=-1)
    return no_sentinel & in_range & unique


def _age_summary(
    tokens: np.ndarray,
    query_positions: np.ndarray,
    step_valid: Mapping[int, np.ndarray],
    k_values: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    _, layers, heads, _ = tokens.shape
    result = np.full((layers, heads, len(k_values), len(AGE_STATISTICS)), np.nan, dtype=np.float32)
    counts = np.zeros((layers, heads, len(k_values)), dtype=np.uint32)
    for k_index, k in enumerate(k_values):
        for layer in range(layers):
            positions = tokens[:, layer, :, :k].astype(np.int64)
            ages = query_positions[:, None, None].astype(np.int64) - positions
            valid = step_valid[k][:, layer, :]
            if np.any(ages[valid] < 0):
                raise B1OfflineError("Token Top-K contains a position after its query position")
            values = np.where(valid[:, :, None], ages, -1).transpose(0, 2, 1).reshape(-1, heads)
            if np.all(values >= 0):
                result[layer, :, k_index] = np.quantile(values, (0.50, 0.90), axis=0).T
                counts[layer, :, k_index] = len(values)
                continue
            for head in range(heads):
                head_values = values[:, head]
                head_values = head_values[head_values >= 0]
                counts[layer, head, k_index] = len(head_values)
                if head_values.size:
                    result[layer, head, k_index] = np.quantile(head_values, (0.50, 0.90))
    return result, counts


def compute_token_stability(
    *,
    sample_id: str,
    token_positions: np.ndarray,
    visible_count: np.ndarray,
    query_positions: np.ndarray,
    k_values: Sequence[int] = (16, 32, 64),
    deltas: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256),
    comparison_chunk_steps: int = 16,
) -> SampleTokenStability:
    """Compute exact set/rank stability without retaining pair-level output."""

    if token_positions.ndim != 4 or token_positions.dtype != np.uint32:
        raise B1OfflineError("token_positions must be uint32 [C,L,Hq,Kmax]")
    steps, layers, heads, stored_k = token_positions.shape
    if visible_count.dtype != np.uint32 or visible_count.shape != (steps,):
        raise B1OfflineError("visible_count must be uint32 [C]")
    if query_positions.dtype != np.uint32 or query_positions.shape != (steps,):
        raise B1OfflineError("query_positions must be uint32 [C]")
    if not np.array_equal(query_positions + 1, visible_count):
        raise B1OfflineError("query_positions + 1 must equal visible_count")
    k_values = tuple(int(value) for value in k_values)
    deltas = tuple(int(value) for value in deltas)
    if not k_values or any(value <= 1 or value > stored_k for value in k_values):
        raise B1OfflineError("k_values must be in [2, stored Token Top-K]")
    if tuple(sorted(set(k_values))) != k_values:
        raise B1OfflineError("k_values must be unique and increasing")
    if not deltas or any(value <= 0 for value in deltas):
        raise B1OfflineError("deltas must be positive")
    if comparison_chunk_steps <= 0:
        raise B1OfflineError("comparison_chunk_steps must be positive")

    step_valid = {k: _step_validity(token_positions, visible_count, k) for k in k_values}
    head_age, age_counts = _age_summary(token_positions, query_positions, step_valid, k_values)
    metrics = np.full((layers, heads, len(k_values), len(deltas), len(TOKEN_METRICS)), np.nan, dtype=np.float32)
    valid_counts = np.zeros((layers, heads, len(k_values), len(deltas)), dtype=np.uint32)
    rank_counts = np.zeros_like(valid_counts)
    max_k = max(k_values)
    for delta_index, delta in enumerate(deltas):
        pair_count = max(0, steps - delta)
        if pair_count == 0:
            continue
        sums = np.zeros((layers, heads, len(k_values), 4), dtype=np.float64)
        rank_sums = np.zeros((layers, heads, len(k_values)), dtype=np.float64)
        entry_values = np.full((len(k_values), pair_count, layers, heads), np.nan, dtype=np.float32)
        for start in range(0, pair_count, comparison_chunk_steps):
            end = min(pair_count, start + comparison_chunk_steps)
            earlier = token_positions[start:end, :, :, :max_k]
            later = token_positions[start + delta : end + delta, :, :, :max_k]
            equality = earlier[..., :, None] == later[..., None, :]
            n_earlier = visible_count[start:end]
            n_later = visible_count[start + delta : end + delta]
            for k_index, k in enumerate(k_values):
                valid = (
                    step_valid[k][start:end]
                    & step_valid[k][start + delta : end + delta]
                    & (n_earlier[:, None, None] >= k)
                    & (n_later[:, None, None] > k)
                )
                overlap_count = equality[..., :k, :k].sum(axis=(-1, -2), dtype=np.uint16)
                overlap = overlap_count.astype(np.float32) / k
                baseline = k / n_later.astype(np.float32)
                raw = (overlap - baseline[:, None, None]) / (1.0 - baseline[:, None, None])
                clipped = np.maximum(0.0, raw)
                entry = 1.0 - overlap
                values = (overlap, raw, clipped, entry)
                for value_index, value in enumerate(values):
                    sums[:, :, k_index, value_index] += np.where(valid, value, 0.0).sum(axis=0)
                valid_counts[:, :, k_index, delta_index] += valid.sum(axis=0, dtype=np.uint32)
                entry_values[k_index, start:end] = np.where(valid, entry, np.nan)

                shared = overlap_count > 0
                rank_valid = valid & shared
                earlier_rank = np.arange(k, dtype=np.int16)
                rank_distance = np.abs(earlier_rank[:, None] - earlier_rank[None, :]).astype(np.float32)
                displacement_sum = np.einsum("tlhij,ij->tlh", equality[..., :k, :k], rank_distance, optimize=True)
                normalized_displacement = displacement_sum / np.maximum(overlap_count, 1) / (k - 1)
                rank_sums[:, :, k_index] += np.where(rank_valid, normalized_displacement, 0.0).sum(axis=0)
                rank_counts[:, :, k_index, delta_index] += rank_valid.sum(axis=0, dtype=np.uint32)
        for k_index, _ in enumerate(k_values):
            count = valid_counts[:, :, k_index, delta_index]
            for metric_index in range(4):
                np.divide(
                    sums[:, :, k_index, metric_index],
                    count,
                    out=metrics[:, :, k_index, delta_index, metric_index],
                    where=count > 0,
                )
            entry = entry_values[k_index]
            if np.isfinite(entry).all():
                metrics[:, :, k_index, delta_index, 4] = np.quantile(entry, 0.90, axis=0)
            else:
                for layer in range(layers):
                    for head in range(heads):
                        values = entry[:, layer, head]
                        values = values[np.isfinite(values)]
                        if values.size:
                            metrics[layer, head, k_index, delta_index, 4] = np.quantile(values, 0.90)
            rank_count = rank_counts[:, :, k_index, delta_index]
            np.divide(
                rank_sums[:, :, k_index],
                rank_count,
                out=metrics[:, :, k_index, delta_index, 5],
                where=rank_count > 0,
            )
    return SampleTokenStability(
        sample_id=sample_id,
        k_values=k_values,
        deltas=deltas,
        metrics=TOKEN_METRICS,
        head_metrics=metrics,
        valid_pair_count=valid_counts,
        rank_valid_pair_count=rank_counts,
        age_statistics=AGE_STATISTICS,
        head_age=head_age,
        age_valid_token_count=age_counts,
    )


def _decode_entries(batch: RunBatch, sample_id: str) -> list[tuple[str, Mapping[str, Any]]]:
    prefix = f"samples/{sample_id}/decode_stats_"
    entries = [
        (relative, entry)
        for relative, entry in batch.manifest_files.items()
        if relative.startswith(prefix) and relative.endswith(".npz")
    ]
    return sorted(entries, key=lambda item: int(item[1]["step_start"]))


def _load_token_trace(batch: RunBatch, sample_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens: list[np.ndarray] = []
    visible: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    for relative, _ in _decode_entries(batch, sample_id):
        with np.load(batch.run_dir / relative) as archive:
            tokens.append(archive["token_topk_positions"].copy())
            visible.append(archive["visible_token_count"].copy())
            positions.append(archive["query_positions"].copy())
    if not tokens:
        raise B1OfflineError(f"{sample_id}: no Decode Token Top-K trace")
    return np.concatenate(tokens), np.concatenate(visible), np.concatenate(positions)


def _sample_equal(values: np.ndarray, conditions: Sequence[str], names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    output = np.full((len(names), *values.shape[1:]), np.nan, dtype=np.float32)
    counts = np.zeros(output.shape, dtype=np.uint32)
    for index, name in enumerate(names):
        rows = values[[value == name for value in conditions]].astype(np.float64, copy=False)
        valid = np.isfinite(rows)
        local_count = valid.sum(axis=0, dtype=np.uint32)
        np.divide(
            np.where(valid, rows, 0.0).sum(axis=0),
            local_count,
            out=output[index],
            where=local_count > 0,
        )
        counts[index] = local_count
    return output, counts


def _head_mean(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    counts = valid.sum(axis=(1, 2))
    output = np.full((values.shape[0], *values.shape[3:]), np.nan, dtype=np.float32)
    np.divide(
        np.where(valid, values, 0.0).sum(axis=(1, 2)),
        counts,
        out=output,
        where=counts > 0,
    )
    return output


def _bootstrap(values: np.ndarray, replicates: int, rng: np.random.Generator) -> np.ndarray:
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    flat = values.reshape(len(values), -1).astype(np.float64, copy=False)
    output = np.full((flat.shape[1], 2), np.nan, dtype=np.float32)
    for column in np.flatnonzero(np.isfinite(flat).any(axis=0)):
        selected = flat[draws, column]
        valid = np.isfinite(selected)
        counts = valid.sum(axis=1)
        means = np.full(replicates, np.nan, dtype=np.float64)
        np.divide(np.where(valid, selected, 0.0).sum(axis=1), counts, out=means, where=counts > 0)
        means = means[np.isfinite(means)]
        if means.size:
            output[column] = np.quantile(means, (0.025, 0.975))
    return output.reshape((*values.shape[1:], 2))


def summarize_token_stability(
    index: RunSetIndex,
    *,
    k_values: Sequence[int] = (16, 32, 64),
    deltas: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256),
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20261103,
    comparison_chunk_steps: int = 16,
    progress_callback: Callable[[int, int, str, float], None] | None = None,
) -> TokenStabilitySummary:
    ordered = [(index.samples[sample_id], batch) for batch in index.batches for sample_id in batch.sample_ids]
    sample_results: list[SampleTokenStability] = []
    conditions: list[str] = []
    for sample_index, (sample, batch) in enumerate(ordered):
        started = time.perf_counter()
        tokens, visible, positions = _load_token_trace(batch, sample.sample_id)
        sample_results.append(
            compute_token_stability(
                sample_id=sample.sample_id,
                token_positions=tokens,
                visible_count=visible,
                query_positions=positions,
                k_values=k_values,
                deltas=deltas,
                comparison_chunk_steps=comparison_chunk_steps,
            )
        )
        conditions.append(sample.condition)
        if progress_callback is not None:
            progress_callback(sample_index + 1, len(ordered), sample.sample_id, time.perf_counter() - started)
    sample_metrics = np.stack([result.head_metrics for result in sample_results])
    sample_counts = np.stack([result.valid_pair_count for result in sample_results])
    sample_rank_counts = np.stack([result.rank_valid_pair_count for result in sample_results])
    sample_age = np.stack([result.head_age for result in sample_results])
    sample_age_counts = np.stack([result.age_valid_token_count for result in sample_results])
    condition_names = tuple(sorted(set(conditions)))
    condition_head, condition_head_count = _sample_equal(sample_metrics, conditions, condition_names)
    condition_age, _ = _sample_equal(sample_age, conditions, condition_names)
    sample_scalar = _head_mean(sample_metrics)
    condition_scalar, condition_scalar_count = _sample_equal(sample_scalar, conditions, condition_names)
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    scalar_ci = np.full((*condition_scalar.shape, 2), np.nan, dtype=np.float32)
    for condition_index, condition in enumerate(condition_names):
        scalar_ci[condition_index] = _bootstrap(
            sample_scalar[np.array(conditions) == condition], bootstrap_replicates, rng
        )

    pair_members: dict[str, dict[int, int]] = {}
    for sample_index, (sample, _) in enumerate(ordered):
        if sample.dataset == "pg19" and sample.pair_id:
            pair_members.setdefault(sample.pair_id, {})[sample.target_prompt_length] = sample_index
    pair_ids = tuple(sorted(pair_members))
    pair_difference = np.full((len(pair_ids), *sample_scalar.shape[1:]), np.nan, dtype=np.float32)
    for pair_index, pair_id in enumerate(pair_ids):
        members = pair_members[pair_id]
        if set(members) != {8192, 32768}:
            raise B1OfflineError(f"PG19 pair {pair_id!r} is incomplete during Token aggregation")
        pair_difference[pair_index] = sample_scalar[members[32768]] - sample_scalar[members[8192]]
    valid = np.isfinite(pair_difference)
    pair_counts = valid.sum(axis=0)
    pair_mean_array = np.full(pair_counts.shape, np.nan, dtype=np.float32)
    np.divide(
        np.where(valid, pair_difference, 0.0).sum(axis=0),
        pair_counts,
        out=pair_mean_array,
        where=pair_counts > 0,
    )
    pair_ci = _bootstrap(
        pair_difference,
        bootstrap_replicates,
        np.random.Generator(np.random.PCG64(bootstrap_seed)),
    )
    return TokenStabilitySummary(
        sample_ids=tuple(result.sample_id for result in sample_results),
        sample_conditions=tuple(conditions),
        condition_names=condition_names,
        k_values=tuple(int(value) for value in k_values),
        deltas=tuple(int(value) for value in deltas),
        metrics=TOKEN_METRICS,
        age_statistics=AGE_STATISTICS,
        sample_head_metrics=sample_metrics,
        sample_head_valid_pair_count=sample_counts,
        sample_head_rank_valid_pair_count=sample_rank_counts,
        sample_head_age=sample_age,
        sample_head_age_valid_token_count=sample_age_counts,
        condition_head_sample_equal_mean=condition_head,
        condition_head_valid_samples=condition_head_count,
        condition_scalar_sample_equal_mean=condition_scalar,
        condition_scalar_valid_samples=condition_scalar_count,
        condition_scalar_bootstrap_ci=scalar_ci,
        condition_age_head_sample_equal_mean=condition_age,
        pg19_pair_ids=pair_ids,
        pg19_pair_scalar_mean_difference_32k_minus_8k=pair_mean_array,
        pg19_pair_scalar_bootstrap_ci=pair_ci,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
