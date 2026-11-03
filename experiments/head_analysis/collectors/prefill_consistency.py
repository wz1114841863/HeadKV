"""CPU reference definitions for HeadKV V1-B2 consistency metrics.

This module implements the frozen mathematical definitions without model hooks,
GPU state, persistence, or B1 trace access.  It is intentionally NumPy-only so
that the later online collector can be tested against an independent eager
reference before any real-model collection is attempted.

The reference path may materialize the small observation matrix supplied to it.
The future online path must still obey the B2 resource contract and process one
KV head/condition at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


UINT32_SENTINEL = int(np.iinfo(np.uint32).max)


class PrefillConsistencyError(ValueError):
    """Raised when an input violates the frozen B2 mathematical contract."""


@dataclass(frozen=True)
class B2Condition:
    """One pre-registered paper-formula Token-CV condition."""

    condition_id: str
    observation_window: int
    initial_exclusion: int
    recent_exclusion: int
    quantile_k: float
    threshold_scale_alpha: float

    def __post_init__(self) -> None:
        if not self.condition_id:
            raise ValueError("condition_id must be non-empty")
        if not 1 <= self.observation_window <= np.iinfo(np.uint8).max:
            raise ValueError("observation_window must fit the uint8 frequency schema")
        if self.initial_exclusion < 0:
            raise ValueError("initial_exclusion must be non-negative")
        if self.recent_exclusion < self.observation_window:
            raise ValueError("recent_exclusion must cover the full observation window")
        if not 0.0 < self.quantile_k < 1.0:
            raise ValueError("quantile_k must be in (0, 1)")
        if not 0.0 < self.threshold_scale_alpha <= 1.0:
            raise ValueError("threshold_scale_alpha must be in (0, 1]")


DEFAULT_B2_CONDITIONS = (
    B2Condition("paper_token_primary", 64, 64, 64, 0.984, 0.8),
    B2Condition("paper_token_alpha1", 64, 64, 64, 0.984, 1.0),
    B2Condition("paper_token_k099", 64, 64, 64, 0.990, 0.8),
    B2Condition("paper_token_recent256", 64, 64, 256, 0.984, 0.8),
)


@dataclass(frozen=True)
class B2CollectorConfig:
    """Frozen B2 v1 condition, budget, and temporary-block configuration."""

    conditions: tuple[B2Condition, ...] = DEFAULT_B2_CONDITIONS
    critical_token_budgets: tuple[int, ...] = (128, 512, 2048)
    critical_token_max: int = 2048
    key_block_size: int = 1024

    def __post_init__(self) -> None:
        if not self.conditions:
            raise ValueError("at least one B2 condition is required")
        condition_ids = [condition.condition_id for condition in self.conditions]
        if len(set(condition_ids)) != len(condition_ids):
            raise ValueError("B2 condition IDs must be unique")
        if any(condition.observation_window != 64 for condition in self.conditions):
            raise ValueError("B2 v1 requires observation_window=64 for every condition")
        if any(condition.recent_exclusion < 64 for condition in self.conditions):
            raise ValueError("B2 v1 requires recent_exclusion>=64")
        if not self.critical_token_budgets or any(value <= 0 for value in self.critical_token_budgets):
            raise ValueError("critical_token_budgets must be non-empty and positive")
        if tuple(sorted(set(self.critical_token_budgets))) != self.critical_token_budgets:
            raise ValueError("critical_token_budgets must be strictly increasing")
        if self.critical_token_max != self.critical_token_budgets[-1]:
            raise ValueError("critical_token_max must equal the largest critical-token budget")
        if self.key_block_size <= 0:
            raise ValueError("key_block_size must be positive")


@dataclass(frozen=True)
class PrefillConsistencyReference:
    """Full CPU reference result for one layer/KV-head/condition.

    ``observation_matrix``, ``binary_matrix``, and ``frequency_vector`` are
    diagnostic reference arrays.  The formal B2 writer must not persist these
    full arrays during a production run.
    """

    condition_id: str
    candidate_positions: np.ndarray
    observation_matrix: np.ndarray
    binary_matrix: np.ndarray
    frequency_vector: np.ndarray
    frequency_histogram: np.ndarray
    binary_threshold: np.float32
    frequency_mean: np.float32
    frequency_std: np.float32
    cv_prefill: np.float32
    nonzero_frequency_count: int
    critical_token_positions: np.ndarray
    critical_token_frequencies: np.ndarray
    critical_token_soft_scores: np.ndarray
    critical_token_valid_count: int
    metric_valid: bool


@dataclass(frozen=True)
class PredictionTokenSets:
    """Learned-only and fixed-region-augmented Prompt token sets."""

    learned_middle: np.ndarray
    augmented_prompt: np.ndarray


@dataclass(frozen=True)
class DecodeRecallReference:
    """Full CPU reference for one layer/KV-head Decode observation."""

    key_positions: np.ndarray
    attention_probabilities: np.ndarray
    prompt_attention_mass: np.float32
    critical_attention_mass: Mapping[str, np.float32]
    critical_prompt_recall: Mapping[str, np.float32]


def candidate_token_positions(prompt_length: int, condition: B2Condition) -> np.ndarray:
    """Return the common-history candidate positions ``[I, T0-R)``."""

    if prompt_length <= condition.initial_exclusion + condition.recent_exclusion:
        raise PrefillConsistencyError("Prompt is too short for the configured initial/recent exclusions")
    return np.arange(
        condition.initial_exclusion,
        prompt_length - condition.recent_exclusion,
        dtype=np.uint32,
    )


def compute_prefill_consistency_reference(
    query_heads: np.ndarray,
    candidate_keys: np.ndarray,
    candidate_positions: np.ndarray,
    *,
    scaling: float,
    condition: B2Condition,
    critical_token_max: int,
) -> PrefillConsistencyReference:
    """Compute paper-formula Token CV from Q/K for one GQA KV-head group.

    Args:
        query_heads: RoPE-applied query rows shaped ``[G, W, D]``.
        candidate_keys: RoPE-applied keys shaped ``[N, D]``.
        candidate_positions: Strictly increasing logical IDs shaped ``[N]``.
        scaling: Actual attention scaling used by the model.
        condition: Frozen B2 condition.
        critical_token_max: Padded ranking length.
    """

    query_heads = np.asarray(query_heads, dtype=np.float32)
    candidate_keys = np.asarray(candidate_keys, dtype=np.float32)
    if query_heads.ndim != 3 or candidate_keys.ndim != 2:
        raise PrefillConsistencyError("query_heads and candidate_keys must be [G,W,D] and [N,D]")
    if query_heads.shape[1] != condition.observation_window:
        raise PrefillConsistencyError("query observation length differs from the B2 condition")
    if query_heads.shape[2] != candidate_keys.shape[1]:
        raise PrefillConsistencyError("query/key head dimensions differ")
    if not np.isfinite(scaling):
        raise PrefillConsistencyError("attention scaling must be finite")
    positions = _validated_positions(candidate_positions, expected_length=candidate_keys.shape[0])
    if not np.isfinite(query_heads).all() or not np.isfinite(candidate_keys).all():
        raise PrefillConsistencyError("Q/K reference inputs must be finite")

    logits = np.einsum("gwd,nd->gwn", query_heads, candidate_keys, dtype=np.float32, optimize=True)
    logits = np.asarray(logits * np.float32(scaling), dtype=np.float32)
    return compute_prefill_consistency_from_logits_reference(
        logits,
        positions,
        condition=condition,
        critical_token_max=critical_token_max,
    )


def compute_prefill_consistency_from_logits_reference(
    query_head_logits: np.ndarray,
    candidate_positions: np.ndarray,
    *,
    condition: B2Condition,
    critical_token_max: int,
) -> PrefillConsistencyReference:
    """Apply per-Query-head Softmax, then GQA-mean and Token CV."""

    logits = np.asarray(query_head_logits, dtype=np.float32)
    if logits.ndim != 3 or logits.shape[0] <= 0:
        raise PrefillConsistencyError("query_head_logits must be non-empty [G,W,N]")
    if logits.shape[1] != condition.observation_window:
        raise PrefillConsistencyError("logit observation length differs from the B2 condition")
    positions = _validated_positions(candidate_positions, expected_length=logits.shape[2])
    probabilities = _stable_softmax(logits)
    observation = probabilities.mean(axis=0, dtype=np.float32)
    return summarize_observation_matrix_reference(
        observation,
        positions,
        condition=condition,
        critical_token_max=critical_token_max,
    )


def summarize_observation_matrix_reference(
    observation_matrix: np.ndarray,
    candidate_positions: np.ndarray,
    *,
    condition: B2Condition,
    critical_token_max: int,
) -> PrefillConsistencyReference:
    """Binarize a GQA-mean observation matrix and produce Token CV/ranking."""

    observation = np.asarray(observation_matrix, dtype=np.float32)
    if observation.ndim != 2 or observation.shape[0] != condition.observation_window:
        raise PrefillConsistencyError("observation_matrix must be shaped [condition.W,N]")
    positions = _validated_positions(candidate_positions, expected_length=observation.shape[1])
    if critical_token_max <= 0:
        raise PrefillConsistencyError("critical_token_max must be positive")
    if not np.isfinite(observation).all() or np.any(observation < 0):
        raise PrefillConsistencyError("observation probabilities must be finite and non-negative")
    row_sums = observation.sum(axis=1, dtype=np.float32)
    if not np.allclose(row_sums, np.float32(1.0), rtol=1e-5, atol=1e-6):
        raise PrefillConsistencyError("each observation row must sum to one")

    quantile = np.quantile(observation.reshape(-1), condition.quantile_k, method="linear")
    threshold = np.float32(quantile * condition.threshold_scale_alpha)
    binary = observation >= threshold
    frequency = binary.sum(axis=0, dtype=np.uint16).astype(np.uint8)
    frequency_float = frequency.astype(np.float32)
    mean = np.float32(frequency_float.mean(dtype=np.float32))
    variance = np.mean(np.square(frequency_float - mean), dtype=np.float32)
    std = np.float32(np.sqrt(variance))
    valid = bool(mean > 0)
    cv = np.float32(std / mean) if valid else np.float32(np.nan)
    histogram = np.bincount(frequency, minlength=condition.observation_window + 1).astype(np.uint32)
    soft_scores = observation.sum(axis=0, dtype=np.float32)

    positive_indices = np.flatnonzero(frequency > 0)
    if positive_indices.size:
        order = np.lexsort(
            (
                positions[positive_indices].astype(np.uint64),
                -soft_scores[positive_indices].astype(np.float64),
                -frequency[positive_indices].astype(np.int16),
            )
        )
        ranked_indices = positive_indices[order]
    else:
        ranked_indices = np.empty(0, dtype=np.int64)
    valid_count = min(critical_token_max, int(ranked_indices.size))
    ranked_indices = ranked_indices[:valid_count]

    ranked_positions = np.full(critical_token_max, UINT32_SENTINEL, dtype=np.uint32)
    ranked_frequencies = np.zeros(critical_token_max, dtype=np.uint8)
    ranked_soft_scores = np.full(critical_token_max, np.nan, dtype=np.float32)
    if valid_count:
        ranked_positions[:valid_count] = positions[ranked_indices]
        ranked_frequencies[:valid_count] = frequency[ranked_indices]
        ranked_soft_scores[:valid_count] = soft_scores[ranked_indices]

    return PrefillConsistencyReference(
        condition_id=condition.condition_id,
        candidate_positions=positions,
        observation_matrix=np.ascontiguousarray(observation, dtype=np.float32),
        binary_matrix=np.ascontiguousarray(binary),
        frequency_vector=np.ascontiguousarray(frequency),
        frequency_histogram=np.ascontiguousarray(histogram),
        binary_threshold=threshold,
        frequency_mean=mean,
        frequency_std=std,
        cv_prefill=cv,
        nonzero_frequency_count=int(positive_indices.size),
        critical_token_positions=ranked_positions,
        critical_token_frequencies=ranked_frequencies,
        critical_token_soft_scores=ranked_soft_scores,
        critical_token_valid_count=valid_count,
        metric_valid=valid,
    )


def build_prediction_token_sets(
    result: PrefillConsistencyReference,
    *,
    budget: int,
    prompt_length: int,
    condition: B2Condition,
) -> PredictionTokenSets:
    """Build the learned-middle and fixed-region-augmented Prompt sets."""

    if result.condition_id != condition.condition_id:
        raise PrefillConsistencyError("result and condition IDs differ")
    if budget <= 0 or budget > result.critical_token_positions.size:
        raise PrefillConsistencyError("budget must fit the stored critical-token ranking")
    expected_positions = candidate_token_positions(prompt_length, condition)
    if not np.array_equal(result.candidate_positions, expected_positions):
        raise PrefillConsistencyError("result candidate positions differ from the configured Prompt interval")

    learned_count = min(budget, result.critical_token_valid_count)
    learned = np.asarray(result.critical_token_positions[:learned_count], dtype=np.uint32)
    if np.any(learned == UINT32_SENTINEL) or np.unique(learned).size != learned.size:
        raise PrefillConsistencyError("learned critical-token prefix is invalid")
    initial = np.arange(condition.initial_exclusion, dtype=np.uint32)
    recent = np.arange(prompt_length - condition.recent_exclusion, prompt_length, dtype=np.uint32)
    augmented = np.unique(np.concatenate((initial, learned, recent))).astype(np.uint32, copy=False)
    return PredictionTokenSets(
        learned_middle=np.ascontiguousarray(learned),
        augmented_prompt=np.ascontiguousarray(augmented),
    )


def compute_decode_recall_reference(
    query_head_logits: np.ndarray,
    key_positions: np.ndarray,
    *,
    prompt_length: int,
    prediction_sets: Mapping[str, np.ndarray],
) -> DecodeRecallReference:
    """Compute full-visible and Prompt-conditional mass for named token sets."""

    logits = np.asarray(query_head_logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[0] <= 0:
        raise PrefillConsistencyError("Decode logits must be non-empty [G,N]")
    positions = _validated_positions(key_positions, expected_length=logits.shape[1])
    if prompt_length <= 0:
        raise PrefillConsistencyError("prompt_length must be positive")
    probabilities = _stable_softmax(logits).mean(axis=0, dtype=np.float32)
    prompt_mask = positions < prompt_length
    prompt_mass = np.float32(probabilities[prompt_mask].sum(dtype=np.float32))

    masses: dict[str, np.float32] = {}
    conditional: dict[str, np.float32] = {}
    for name, raw_set in prediction_sets.items():
        if not name:
            raise PrefillConsistencyError("prediction-set names must be non-empty")
        token_set = np.asarray(raw_set)
        if token_set.ndim != 1 or not np.issubdtype(token_set.dtype, np.integer):
            raise PrefillConsistencyError(f"prediction set {name!r} must be a one-dimensional integer array")
        token_set = token_set.astype(np.uint32, copy=False)
        if np.unique(token_set).size != token_set.size:
            raise PrefillConsistencyError(f"prediction set {name!r} contains duplicate positions")
        if token_set.size and not np.isin(token_set, positions).all():
            raise PrefillConsistencyError(f"prediction set {name!r} contains a non-visible position")
        if token_set.size and np.any(token_set >= prompt_length):
            raise PrefillConsistencyError(f"prediction set {name!r} contains a non-Prompt position")
        mass = np.float32(probabilities[np.isin(positions, token_set)].sum(dtype=np.float32))
        masses[name] = mass
        conditional[name] = np.float32(mass / prompt_mass) if prompt_mass > 0 else np.float32(np.nan)

    return DecodeRecallReference(
        key_positions=positions,
        attention_probabilities=np.ascontiguousarray(probabilities, dtype=np.float32),
        prompt_attention_mass=prompt_mass,
        critical_attention_mass=masses,
        critical_prompt_recall=conditional,
    )


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    if not np.isfinite(logits).all() or logits.shape[-1] == 0:
        raise PrefillConsistencyError("Softmax logits must be finite with a non-empty candidate axis")
    shifted = np.asarray(logits - logits.max(axis=-1, keepdims=True), dtype=np.float32)
    exponentials = np.exp(shifted).astype(np.float32, copy=False)
    normalizer = exponentials.sum(axis=-1, keepdims=True, dtype=np.float32)
    if not np.isfinite(normalizer).all() or np.any(normalizer <= 0):
        raise PrefillConsistencyError("Softmax normalization is invalid")
    return np.asarray(exponentials / normalizer, dtype=np.float32)


def _validated_positions(positions: np.ndarray, *, expected_length: int) -> np.ndarray:
    value = np.asarray(positions)
    if value.ndim != 1 or value.size != expected_length or not np.issubdtype(value.dtype, np.integer):
        raise PrefillConsistencyError("logical positions must be a one-dimensional integer array of matching length")
    if value.size == 0:
        raise PrefillConsistencyError("logical positions must be non-empty")
    if np.any(value < 0) or np.any(value > UINT32_SENTINEL - 1):
        raise PrefillConsistencyError("logical positions must fit valid uint32 IDs")
    result = value.astype(np.uint32, copy=False)
    if np.any(result[1:] <= result[:-1]):
        raise PrefillConsistencyError("logical positions must be strictly increasing")
    return np.ascontiguousarray(result)


__all__ = [
    "B2CollectorConfig",
    "B2Condition",
    "DEFAULT_B2_CONDITIONS",
    "DecodeRecallReference",
    "PredictionTokenSets",
    "PrefillConsistencyError",
    "PrefillConsistencyReference",
    "UINT32_SENTINEL",
    "build_prediction_token_sets",
    "candidate_token_positions",
    "compute_decode_recall_reference",
    "compute_prefill_consistency_from_logits_reference",
    "compute_prefill_consistency_reference",
    "summarize_observation_matrix_reference",
]
