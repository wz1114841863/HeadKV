"""Online V1-B2 Prefill consistency and Decode recall collector.

The collector consumes only ephemeral :class:`AttentionProbeContext` values.
It computes compact B2 records, retains only critical-token rankings between
Prefill and Decode, and never mutates or retains model Q/K tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from experiments.head_analysis.collectors.prefill_consistency import (
    B2CollectorConfig,
    UINT32_SENTINEL,
)
from experiments.head_analysis.runtime.attention_probe import AttentionProbeContext
from kvpress.context import KVPhase


SET_TYPES = ("learned_middle", "augmented_prompt")


class B2CollectionError(RuntimeError):
    """Raised when an event violates the B2 collection contract."""


@dataclass(frozen=True)
class B2PrefillLayerRecord:
    sample_id: str
    layer_id: int
    forward_id: int
    condition_ids: tuple[str, ...]
    cv_prefill: np.ndarray  # [Hkv,M]
    frequency_mean: np.ndarray  # [Hkv,M]
    frequency_std: np.ndarray  # [Hkv,M]
    binary_threshold: np.ndarray  # [Hkv,M]
    candidate_token_count: np.ndarray  # [M]
    candidate_position_start: np.ndarray  # [M]
    candidate_position_end_exclusive: np.ndarray  # [M]
    nonzero_frequency_count: np.ndarray  # [Hkv,M]
    frequency_histogram: np.ndarray  # [Hkv,M,W+1]
    critical_token_positions: np.ndarray  # [Hkv,M,Kmax]
    critical_token_frequencies: np.ndarray  # [Hkv,M,Kmax]
    critical_token_valid_count: np.ndarray  # [Hkv,M]
    metric_valid: np.ndarray  # [Hkv,M]


@dataclass(frozen=True)
class B2DecodeLayerRecord:
    sample_id: str
    layer_id: int
    forward_id: int
    decode_step: int
    query_position: int
    prompt_attention_mass: np.ndarray  # [Hkv]
    critical_attention_mass: np.ndarray  # [Hkv,M,Nb,U]


class B2AttentionCollector:
    """Compute B2 Prefill Token CV and exact Decode Attention-Mass recall."""

    def __init__(self, config: B2CollectorConfig | None = None) -> None:
        self.config = config or B2CollectorConfig()
        self._prefill_records: dict[int, B2PrefillLayerRecord] = {}
        self._rankings: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def condition_ids(self) -> tuple[str, ...]:
        return tuple(condition.condition_id for condition in self.config.conditions)

    def clear(self) -> None:
        """Release all per-sample compact state."""

        self._prefill_records.clear()
        self._rankings.clear()

    def collect(self, context: AttentionProbeContext) -> B2PrefillLayerRecord | B2DecodeLayerRecord:
        self._validate_context(context)
        with torch.no_grad():
            if context.phase is KVPhase.PREFILL:
                return self._collect_prefill(context)
            return self._collect_decode(context)

    def _collect_prefill(self, context: AttentionProbeContext) -> B2PrefillLayerRecord:
        if context.layer_id in self._prefill_records:
            raise B2CollectionError(f"duplicate Prefill record for layer {context.layer_id}")

        query = context.query
        key = context.key
        _, query_heads, prompt_length, _ = query.shape
        _, kv_heads, _, _ = key.shape
        group_size = query_heads // kv_heads
        conditions = self.config.conditions
        condition_count = len(conditions)
        max_window = conditions[0].observation_window
        max_frequency = max_window + 1
        kmax = self.config.critical_token_max
        device = query.device
        scaling = float(context.scaling)

        cv = np.full((kv_heads, condition_count), np.nan, dtype=np.float32)
        mean = np.full_like(cv, np.nan)
        std = np.full_like(cv, np.nan)
        threshold = np.full_like(cv, np.nan)
        nonzero = np.zeros((kv_heads, condition_count), dtype=np.uint32)
        histogram = np.zeros((kv_heads, condition_count, max_frequency), dtype=np.uint32)
        ranked_positions = np.full((kv_heads, condition_count, kmax), UINT32_SENTINEL, dtype=np.uint32)
        ranked_frequencies = np.zeros((kv_heads, condition_count, kmax), dtype=np.uint8)
        valid_count = np.zeros((kv_heads, condition_count), dtype=np.uint32)
        valid = np.zeros((kv_heads, condition_count), dtype=bool)
        candidate_count = np.asarray(
            [prompt_length - item.recent_exclusion - item.initial_exclusion for item in conditions],
            dtype=np.uint32,
        )
        candidate_start = np.asarray([item.initial_exclusion for item in conditions], dtype=np.uint32)
        candidate_end = np.asarray([prompt_length - item.recent_exclusion for item in conditions], dtype=np.uint32)

        grouped_conditions: dict[tuple[int, int], list[int]] = {}
        for index, condition in enumerate(conditions):
            if prompt_length <= condition.initial_exclusion + condition.recent_exclusion:
                raise B2CollectionError(
                    f"Prompt length {prompt_length} is too short for condition {condition.condition_id}"
                )
            grouped_conditions.setdefault((condition.initial_exclusion, condition.recent_exclusion), []).append(index)

        selected_query = query[0, :, -max_window:, :]
        for kv_head in range(kv_heads):
            q_group = selected_query[kv_head * group_size : (kv_head + 1) * group_size].to(torch.float32)
            for (start, recent), indices in grouped_conditions.items():
                end = prompt_length - recent
                candidate_key = key[0, kv_head, start:end, :].to(torch.float32)
                logits = torch.einsum("gwd,nd->gwn", q_group, candidate_key) * scaling
                attention = torch.softmax(logits, dim=-1, dtype=torch.float32).mean(dim=0)
                quantiles: dict[float, torch.Tensor] = {}
                for condition_index in indices:
                    condition = conditions[condition_index]
                    quantile = quantiles.get(condition.quantile_k)
                    if quantile is None:
                        quantile = torch.quantile(attention.reshape(-1), condition.quantile_k, interpolation="linear")
                        quantiles[condition.quantile_k] = quantile
                    current_threshold = quantile * condition.threshold_scale_alpha
                    binary = attention >= current_threshold
                    frequencies = binary.sum(dim=0, dtype=torch.int64)
                    frequencies_float = frequencies.to(torch.float32)
                    current_mean = frequencies_float.mean()
                    current_std = frequencies_float.std(correction=0)
                    current_valid = bool(current_mean.item() > 0)
                    current_cv = (
                        current_std / current_mean if current_valid else torch.tensor(float("nan"), device=device)
                    )
                    current_histogram = torch.bincount(frequencies, minlength=max_frequency)[:max_frequency]
                    soft_scores = attention.sum(dim=0, dtype=torch.float32)

                    frequency_cpu = frequencies.to(device="cpu", dtype=torch.uint8).numpy()
                    soft_cpu = soft_scores.cpu().numpy()
                    positions_cpu = np.arange(start, end, dtype=np.uint32)
                    position_row, frequency_row, current_valid_count = _rank_critical_tokens(
                        positions_cpu,
                        frequency_cpu,
                        soft_cpu,
                        limit=kmax,
                    )

                    cv[kv_head, condition_index] = np.float32(current_cv.item())
                    mean[kv_head, condition_index] = np.float32(current_mean.item())
                    std[kv_head, condition_index] = np.float32(current_std.item())
                    threshold[kv_head, condition_index] = np.float32(current_threshold.item())
                    nonzero[kv_head, condition_index] = np.uint32(np.count_nonzero(frequency_cpu))
                    histogram[kv_head, condition_index] = current_histogram.to(device="cpu", dtype=torch.uint32).numpy()
                    ranked_positions[kv_head, condition_index] = position_row
                    ranked_frequencies[kv_head, condition_index] = frequency_row
                    valid_count[kv_head, condition_index] = np.uint32(current_valid_count)
                    valid[kv_head, condition_index] = current_valid

        record = B2PrefillLayerRecord(
            sample_id=context.sample_id,
            layer_id=context.layer_id,
            forward_id=context.forward_id,
            condition_ids=self.condition_ids,
            cv_prefill=cv,
            frequency_mean=mean,
            frequency_std=std,
            binary_threshold=threshold,
            candidate_token_count=candidate_count,
            candidate_position_start=candidate_start,
            candidate_position_end_exclusive=candidate_end,
            nonzero_frequency_count=nonzero,
            frequency_histogram=histogram,
            critical_token_positions=ranked_positions,
            critical_token_frequencies=ranked_frequencies,
            critical_token_valid_count=valid_count,
            metric_valid=valid,
        )
        self._prefill_records[context.layer_id] = record
        ranking_tensor = torch.from_numpy(ranked_positions.astype(np.int64)).to(device=device)
        count_tensor = torch.from_numpy(valid_count.astype(np.int64)).to(device=device)
        self._rankings[context.layer_id] = (ranking_tensor, count_tensor)
        return record

    def _collect_decode(self, context: AttentionProbeContext) -> B2DecodeLayerRecord:
        if context.decode_step is None:
            raise B2CollectionError("Decode context is missing decode_step")
        prefill = self._prefill_records.get(context.layer_id)
        ranking_state = self._rankings.get(context.layer_id)
        if prefill is None or ranking_state is None:
            raise B2CollectionError(f"Decode layer {context.layer_id} has no Prefill state")

        query = context.query
        key = context.key
        _, query_heads, _, _ = query.shape
        _, kv_heads, key_length, _ = key.shape
        group_size = query_heads // kv_heads
        device = query.device
        scaling = float(context.scaling)
        query_vector = query[0, :, 0, :].to(torch.float32)
        kv_for_query = torch.div(
            torch.arange(query_heads, device=device, dtype=torch.long), group_size, rounding_mode="floor"
        )

        log_z = torch.full((query_heads,), -torch.inf, device=device, dtype=torch.float32)
        for start in range(0, key_length, self.config.key_block_size):
            end = min(start + self.config.key_block_size, key_length)
            key_block = key[0, :, start:end, :].index_select(0, kv_for_query).to(torch.float32)
            logits = torch.einsum("hd,hkd->hk", query_vector, key_block) * scaling
            log_z = torch.logaddexp(log_z, torch.logsumexp(logits, dim=-1))
        if not torch.isfinite(log_z).all():
            raise B2CollectionError("Decode attention normalization is non-finite")

        grouped_probabilities = torch.empty((kv_heads, key_length), device=device, dtype=torch.float32)
        for start in range(0, key_length, self.config.key_block_size):
            end = min(start + self.config.key_block_size, key_length)
            key_block = key[0, :, start:end, :].index_select(0, kv_for_query).to(torch.float32)
            logits = torch.einsum("hd,hkd->hk", query_vector, key_block) * scaling
            probabilities = torch.exp(logits - log_z.unsqueeze(-1))
            grouped_probabilities[:, start:end] = probabilities.reshape(kv_heads, group_size, end - start).mean(dim=1)

        prompt_probabilities = grouped_probabilities[:, : context.prompt_length]
        prompt_mass = prompt_probabilities.sum(dim=-1, dtype=torch.float32)
        ranking_positions, ranking_valid_count = ranking_state
        condition_count = len(self.config.conditions)
        budget_count = len(self.config.critical_token_budgets)
        mass = torch.zeros(
            (kv_heads, condition_count, budget_count, len(SET_TYPES)),
            device=device,
            dtype=torch.float32,
        )
        rank_axis = torch.arange(self.config.critical_token_max, device=device).view(1, -1)
        for condition_index, condition in enumerate(self.config.conditions):
            positions = ranking_positions[:, condition_index, :]
            counts = ranking_valid_count[:, condition_index]
            safe_positions = torch.where(positions == UINT32_SENTINEL, torch.zeros_like(positions), positions)
            if torch.any(safe_positions >= context.prompt_length):
                raise B2CollectionError("Prefill ranking contains a non-Prompt position")
            ranked_mass = torch.gather(prompt_probabilities, 1, safe_positions)
            ranked_mass = torch.where(rank_axis < counts.unsqueeze(1), ranked_mass, torch.zeros_like(ranked_mass))
            fixed_mass = prompt_probabilities[:, : condition.initial_exclusion].sum(dim=-1)
            fixed_mass += prompt_probabilities[:, context.prompt_length - condition.recent_exclusion :].sum(dim=-1)
            for budget_index, budget in enumerate(self.config.critical_token_budgets):
                learned = torch.where(
                    rank_axis < torch.minimum(counts, torch.full_like(counts, budget)).unsqueeze(1),
                    ranked_mass,
                    torch.zeros_like(ranked_mass),
                ).sum(dim=-1)
                mass[:, condition_index, budget_index, 0] = learned
                mass[:, condition_index, budget_index, 1] = learned + fixed_mass

        tolerance = 1e-5
        if torch.any(prompt_mass < -tolerance) or torch.any(prompt_mass > 1 + tolerance):
            raise B2CollectionError("Prompt attention mass is outside [0,1]")
        if torch.any(mass < -tolerance) or torch.any(mass > prompt_mass.view(-1, 1, 1, 1) + tolerance):
            raise B2CollectionError("critical attention mass is outside its Prompt-mass bound")
        if budget_count > 1 and torch.any(mass[:, :, 1:, :] + tolerance < mass[:, :, :-1, :]):
            raise B2CollectionError("critical attention mass is not monotonic in budget")
        if torch.any(mass[:, :, :, 1] + tolerance < mass[:, :, :, 0]):
            raise B2CollectionError("augmented Prompt mass is below learned-middle mass")

        return B2DecodeLayerRecord(
            sample_id=context.sample_id,
            layer_id=context.layer_id,
            forward_id=context.forward_id,
            decode_step=context.decode_step,
            query_position=int(context.query_positions[0].item()),
            prompt_attention_mass=prompt_mass.cpu().numpy().astype(np.float32, copy=False),
            critical_attention_mass=mass.cpu().numpy().astype(np.float32, copy=False),
        )

    def _validate_context(self, context: AttentionProbeContext) -> None:
        if context.phase not in (KVPhase.PREFILL, KVPhase.DECODE):
            raise B2CollectionError(f"unsupported phase {context.phase!r}")
        if context.attention_mask is not None:
            raise B2CollectionError("B2 v1 does not implement additive-mask visibility/bias semantics")
        if not context.is_causal:
            raise B2CollectionError("B2 v1 requires standard causal attention")
        if context.scaling is None or not np.isfinite(float(context.scaling)):
            raise B2CollectionError("B2 requires finite attention scaling")
        query, key = context.query, context.key
        if query.ndim != 4 or key.ndim != 4 or query.shape[0] != 1 or key.shape[0] != 1:
            raise B2CollectionError("B2 requires Q/K shaped [1,head,sequence,dim]")
        if query.shape[-1] != key.shape[-1] or query.shape[1] % key.shape[1]:
            raise B2CollectionError("B2 requires an integral contiguous GQA mapping")
        expected_keys = torch.arange(
            key.shape[2], device=context.key_positions.device, dtype=context.key_positions.dtype
        )
        if not torch.equal(context.key_positions, expected_keys):
            raise B2CollectionError("B2 requires continuous logical Key positions")
        if context.phase is KVPhase.PREFILL:
            if query.shape[2] != context.prompt_length or key.shape[2] != context.prompt_length:
                raise B2CollectionError("B2 requires one complete Prompt Prefill")
        elif query.shape[2] != 1 or context.decode_step is None:
            raise B2CollectionError("B2 Decode requires one Query and a decode_step")


def _rank_critical_tokens(
    positions: np.ndarray,
    frequencies: np.ndarray,
    soft_scores: np.ndarray,
    *,
    limit: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    positive = np.flatnonzero(frequencies > 0)
    order = np.lexsort(
        (
            positions[positive].astype(np.uint64),
            -soft_scores[positive].astype(np.float64),
            -frequencies[positive].astype(np.int16),
        )
    )
    ranked = positive[order][:limit]
    count = int(ranked.size)
    ranked_positions = np.full(limit, UINT32_SENTINEL, dtype=np.uint32)
    ranked_frequencies = np.zeros(limit, dtype=np.uint8)
    if count:
        ranked_positions[:count] = positions[ranked]
        ranked_frequencies[:count] = frequencies[ranked]
    return ranked_positions, ranked_frequencies, count


__all__ = [
    "B2AttentionCollector",
    "B2CollectionError",
    "B2DecodeLayerRecord",
    "B2PrefillLayerRecord",
    "SET_TYPES",
]
