import math

import numpy as np
import pytest
import torch

from experiments.head_analysis.collectors.attention_stats import (
    AttentionCollectionError,
    AttentionCollectorConfig,
    AttentionStatsCollector,
    DISTANCE_BIN_LOWER_BOUNDS,
    FEATURE_NAMES,
    UINT32_SENTINEL,
)
from experiments.head_analysis.runtime.attention_probe import AttentionProbeContext
from kvpress.context import KVPhase


def make_context(query, key, *, phase=KVPhase.DECODE, prompt_length=4, decode_step=0, attention_mask=None):
    query_length = query.shape[2]
    key_length = key.shape[2]
    first_query_position = key_length - query_length
    query_positions = torch.arange(first_query_position, key_length)
    return AttentionProbeContext(
        sample_id="unit",
        phase=phase,
        layer_id=1,
        forward_id=3,
        decode_step=decode_step if phase is KVPhase.DECODE else None,
        query=query,
        key=key,
        attention_mask=attention_mask,
        scaling=1 / math.sqrt(query.shape[-1]),
        is_causal=True,
        backend_name="sdpa",
        cache_position=query_positions,
        query_positions=query_positions,
        key_positions=torch.arange(key_length),
        prompt_length=prompt_length,
        query_dtype=query.dtype,
        key_dtype=key.dtype,
        dropout_p=0.0,
    )


def full_reference(query, key, context):
    """Independent unblocked probability reference for one Decode query."""

    hq = query.shape[1]
    hkv = key.shape[1]
    group = hq // hkv
    q_position = int(context.query_positions[0])
    visible = torch.arange(key.shape[2]) <= q_position
    rows = []
    for head in range(hq):
        logits = (query[0, head, 0].float() @ key[0, head // group].float().T) * context.scaling
        probabilities = torch.softmax(logits.masked_fill(~visible, -torch.inf), dim=0)
        positions = torch.arange(key.shape[2])
        entropy = -(probabilities[visible] * probabilities[visible].log()).sum()
        normalized = entropy / math.log(int(visible.sum()))
        top_mass = [
            torch.topk(probabilities[visible], min(k, int(visible.sum()))).values.sum() for k in (16, 64, 256, 1024)
        ]
        distances = q_position - positions
        distance_values = []
        for index, lower in enumerate(DISTANCE_BIN_LOWER_BOUNDS):
            upper = DISTANCE_BIN_LOWER_BOUNDS[index + 1] if index + 1 < len(DISTANCE_BIN_LOWER_BOUNDS) else None
            mask = distances >= lower if upper is None else (distances >= lower) & (distances < upper)
            distance_values.append(probabilities[mask].sum())
        rows.append(
            torch.stack(
                (
                    entropy,
                    normalized,
                    *top_mass,
                    probabilities[positions < 4].sum(),
                    probabilities[positions >= q_position - 63].sum(),
                    probabilities[positions < context.prompt_length].sum(),
                    probabilities[positions >= context.prompt_length].sum(),
                    (probabilities * distances).sum(),
                    *distance_values,
                )
            )
        )
    return torch.stack(rows)


def test_decode_statistics_match_independent_full_attention_reference():
    torch.manual_seed(12)
    query = torch.randn(1, 4, 1, 3)
    key = torch.randn(1, 2, 5, 3)
    context = make_context(query, key)
    collector = AttentionStatsCollector(
        AttentionCollectorConfig(key_block_size=2, token_topk=5, page_size=2, page_topk=3)
    )

    record = collector.collect(context)

    assert record.scalar_features.shape == (1, 4, len(FEATURE_NAMES))
    np.testing.assert_allclose(
        record.scalar_features[0], full_reference(query, key, context).numpy(), rtol=1e-6, atol=1e-6
    )
    assert record.query_positions.tolist() == [4]
    assert record.visible_token_count.tolist() == [5]
    assert record.token_topk_positions.shape == (1, 4, 5)
    assert record.page_topk_ids_all_visible.shape == (1, 2, 3)
    assert record.page_candidate_count_all_visible.tolist() == [3]
    assert record.page_candidate_count_prompt_only.tolist() == [2]
    np.testing.assert_allclose(record.page_scope_mass[:, :, 0], 1.0, rtol=1e-6, atol=1e-6)
    assert record.page_scope_valid.all()


def test_key_block_size_does_not_change_statistics_or_rankings():
    torch.manual_seed(2)
    query = torch.randn(1, 4, 1, 4)
    key = torch.randn(1, 2, 9, 4)
    context = make_context(query, key, prompt_length=6)
    small = AttentionStatsCollector(
        AttentionCollectorConfig(key_block_size=1, token_topk=6, page_size=2, page_topk=4)
    ).collect(context)
    large = AttentionStatsCollector(
        AttentionCollectorConfig(key_block_size=32, token_topk=6, page_size=2, page_topk=4)
    ).collect(context)

    np.testing.assert_allclose(small.scalar_features, large.scalar_features, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(small.token_topk_positions, large.token_topk_positions)
    np.testing.assert_array_equal(small.page_topk_ids_all_visible, large.page_topk_ids_all_visible)
    np.testing.assert_array_equal(small.page_topk_ids_prompt_only, large.page_topk_ids_prompt_only)
    np.testing.assert_allclose(small.page_scope_mass, large.page_scope_mass, rtol=1e-6, atol=1e-6)


def test_token_and_page_ties_use_smaller_logical_ids_first_and_pad_with_sentinel():
    query = torch.zeros(1, 4, 1, 2)
    key = torch.zeros(1, 2, 5, 2)
    record = AttentionStatsCollector(
        AttentionCollectorConfig(key_block_size=2, token_topk=7, page_size=2, page_topk=5)
    ).collect(make_context(query, key))

    assert record.token_topk_positions[0, 0].tolist() == [0, 1, 2, 3, 4, UINT32_SENTINEL, UINT32_SENTINEL]
    assert record.page_topk_ids_all_visible[0, 0].tolist() == [0, 1, 2, UINT32_SENTINEL, UINT32_SENTINEL]
    assert record.page_topk_ids_prompt_only[0, 0].tolist() == [0, 1, UINT32_SENTINEL, UINT32_SENTINEL, UINT32_SENTINEL]


def test_prefill_only_selects_terminal_window_and_has_no_decode_page_or_token_trace():
    torch.manual_seed(5)
    query = torch.randn(1, 4, 4, 2)
    key = torch.randn(1, 2, 4, 2)
    context = make_context(query, key, phase=KVPhase.PREFILL, prompt_length=4)
    record = AttentionStatsCollector(AttentionCollectorConfig(prefill_window=2, key_block_size=2)).collect(context)

    assert record.query_positions.tolist() == [2, 3]
    assert record.visible_token_count.tolist() == [3, 4]
    assert record.scalar_features.shape == (2, 4, 28)
    assert record.token_topk_positions is None
    assert record.page_topk_ids_all_visible is None


def test_nonstandard_mask_is_rejected_instead_of_assuming_full_visibility():
    query = torch.randn(1, 4, 1, 2)
    key = torch.randn(1, 2, 5, 2)
    context = make_context(query, key, attention_mask=torch.zeros(1, 1, 1, 5))

    with pytest.raises(AttentionCollectionError, match="additive-mask"):
        AttentionStatsCollector().collect(context)
