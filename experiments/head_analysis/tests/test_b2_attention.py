import math

import numpy as np
import pytest
import torch

from experiments.head_analysis.collectors.b2_attention import B2AttentionCollector, B2CollectionError
from experiments.head_analysis.collectors.prefill_consistency import (
    B2CollectorConfig,
    B2Condition,
    build_prediction_token_sets,
    compute_decode_recall_reference,
    compute_prefill_consistency_reference,
)
from experiments.head_analysis.runtime.attention_probe import AttentionProbeContext
from kvpress.context import KVPhase


def make_context(query, key, *, phase, prompt_length, layer_id=0, decode_step=None, attention_mask=None):
    query_length = query.shape[2]
    key_length = key.shape[2]
    query_positions = torch.arange(key_length - query_length, key_length)
    return AttentionProbeContext(
        sample_id="b2-unit",
        phase=phase,
        layer_id=layer_id,
        forward_id=0 if phase is KVPhase.PREFILL else int(decode_step) + 1,
        decode_step=decode_step,
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


def make_config():
    condition = B2Condition("toy64", 64, 2, 64, 0.75, 0.8)
    return B2CollectorConfig(
        conditions=(condition,),
        critical_token_budgets=(1, 2, 4),
        critical_token_max=4,
        key_block_size=3,
    )


def test_online_prefill_matches_cpu_reference_and_does_not_mutate_qk():
    torch.manual_seed(21)
    prompt_length = 70
    query = torch.randn(1, 4, prompt_length, 3)
    key = torch.randn(1, 2, prompt_length, 3)
    query_before = query.clone()
    key_before = key.clone()
    config = make_config()
    condition = config.conditions[0]
    collector = B2AttentionCollector(config)

    record = collector.collect(make_context(query, key, phase=KVPhase.PREFILL, prompt_length=prompt_length))

    assert record.cv_prefill.shape == (2, 1)
    assert record.critical_token_positions.shape == (2, 1, 4)
    for kv_head in range(2):
        reference = compute_prefill_consistency_reference(
            query[0, kv_head * 2 : (kv_head + 1) * 2, -64:, :].numpy(),
            key[0, kv_head, 2:6, :].numpy(),
            np.arange(2, 6),
            scaling=1 / math.sqrt(3),
            condition=condition,
            critical_token_max=4,
        )
        np.testing.assert_allclose(record.cv_prefill[kv_head, 0], reference.cv_prefill, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(
            record.binary_threshold[kv_head, 0], reference.binary_threshold, rtol=2e-5, atol=2e-5
        )
        np.testing.assert_array_equal(record.frequency_histogram[kv_head, 0], reference.frequency_histogram)
        np.testing.assert_array_equal(record.critical_token_positions[kv_head, 0], reference.critical_token_positions)
    assert torch.equal(query, query_before)
    assert torch.equal(key, key_before)


def test_online_decode_matches_full_cpu_reference_for_each_kv_head():
    torch.manual_seed(22)
    prompt_length = 70
    prefill_query = torch.randn(1, 4, prompt_length, 3)
    prefill_key = torch.randn(1, 2, prompt_length, 3)
    config = make_config()
    condition = config.conditions[0]
    collector = B2AttentionCollector(config)
    collector.collect(make_context(prefill_query, prefill_key, phase=KVPhase.PREFILL, prompt_length=prompt_length))
    decode_query = torch.randn(1, 4, 1, 3)
    decode_key = torch.cat((prefill_key, torch.randn(1, 2, 1, 3)), dim=2)

    decode = collector.collect(
        make_context(
            decode_query,
            decode_key,
            phase=KVPhase.DECODE,
            prompt_length=prompt_length,
            decode_step=0,
        )
    )

    assert decode.prompt_attention_mass.shape == (2,)
    assert decode.critical_attention_mass.shape == (2, 1, 3, 2)
    for kv_head in range(2):
        logits = (
            decode_query[0, kv_head * 2 : (kv_head + 1) * 2, 0].float() @ decode_key[0, kv_head].float().T
        ).numpy() * (1 / math.sqrt(3))
        # Reconstruct the compact Prefill result needed by the set helper from
        # the independent reference, not from collector-private GPU state.
        reference_prefill = compute_prefill_consistency_reference(
            prefill_query[0, kv_head * 2 : (kv_head + 1) * 2, -64:, :].numpy(),
            prefill_key[0, kv_head, 2:6, :].numpy(),
            np.arange(2, 6),
            scaling=1 / math.sqrt(3),
            condition=condition,
            critical_token_max=4,
        )
        for budget_index, budget in enumerate(config.critical_token_budgets):
            sets = build_prediction_token_sets(
                reference_prefill,
                budget=budget,
                prompt_length=prompt_length,
                condition=condition,
            )
            reference = compute_decode_recall_reference(
                logits,
                np.arange(prompt_length + 1),
                prompt_length=prompt_length,
                prediction_sets={
                    "learned": sets.learned_middle,
                    "augmented": sets.augmented_prompt,
                },
            )
            assert decode.prompt_attention_mass[kv_head] == pytest.approx(
                reference.prompt_attention_mass, rel=2e-5, abs=2e-5
            )
            assert decode.critical_attention_mass[kv_head, 0, budget_index, 0] == pytest.approx(
                reference.critical_attention_mass["learned"], rel=2e-5, abs=2e-5
            )
            assert decode.critical_attention_mass[kv_head, 0, budget_index, 1] == pytest.approx(
                reference.critical_attention_mass["augmented"], rel=2e-5, abs=2e-5
            )


def test_decode_requires_prefill_state_and_clear_removes_it():
    query = torch.randn(1, 4, 1, 3)
    key = torch.randn(1, 2, 71, 3)
    collector = B2AttentionCollector(make_config())
    context = make_context(query, key, phase=KVPhase.DECODE, prompt_length=70, decode_step=0)

    with pytest.raises(B2CollectionError, match="no Prefill state"):
        collector.collect(context)

    prefill_query = torch.randn(1, 4, 70, 3)
    prefill_key = key[:, :, :70]
    collector.collect(make_context(prefill_query, prefill_key, phase=KVPhase.PREFILL, prompt_length=70))
    collector.clear()
    with pytest.raises(B2CollectionError, match="no Prefill state"):
        collector.collect(context)


def test_collector_rejects_additive_mask_instead_of_guessing_visibility():
    query = torch.randn(1, 4, 70, 3)
    key = torch.randn(1, 2, 70, 3)
    context = make_context(
        query,
        key,
        phase=KVPhase.PREFILL,
        prompt_length=70,
        attention_mask=torch.zeros(1, 1, 70, 70),
    )

    with pytest.raises(B2CollectionError, match="additive-mask"):
        B2AttentionCollector(make_config()).collect(context)
