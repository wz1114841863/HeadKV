import math

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from experiments.head_analysis.runtime.attention_probe import (
    AttentionProbe,
    AttentionProbeError,
    KVPhase,
    UnsupportedAttentionBackendError,
)


def make_model():
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        eos_token_id=None,
        pad_token_id=None,
    )
    return LlamaForCausalLM(config).eval()


def test_probe_captures_post_rope_post_cache_qk_and_lifecycle_metadata():
    model = make_model()
    snapshots = []

    def capture(context):
        snapshots.append(
            {
                "phase": context.phase,
                "layer": context.layer_id,
                "forward": context.forward_id,
                "decode_step": context.decode_step,
                "query_shape": tuple(context.query.shape),
                "key_shape": tuple(context.key.shape),
                "query_positions": context.query_positions.tolist(),
                "key_positions": context.key_positions.tolist(),
                "mask_is_none": context.attention_mask is None,
                "scaling": context.scaling,
                "dropout": context.dropout_p,
            }
        )

    probe = AttentionProbe(sample_id="tiny", prompt_length=4, on_attention=capture)
    with probe(model):
        generated = model.generate(torch.tensor([[1, 2, 3, 4]]), max_new_tokens=3, do_sample=False)

    assert generated.shape == (1, 7)
    assert len(snapshots) == 6
    assert [(entry["forward"], entry["layer"]) for entry in snapshots] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    ]

    for entry in snapshots[:2]:
        assert entry["phase"] is KVPhase.PREFILL
        assert entry["decode_step"] is None
        assert entry["query_shape"] == (1, 4, 4, 8)
        assert entry["key_shape"] == (1, 2, 4, 8)
        assert entry["query_positions"] == [0, 1, 2, 3]
        assert entry["key_positions"] == [0, 1, 2, 3]

    for expected_step, start in ((0, 2), (1, 4)):
        for entry in snapshots[start : start + 2]:
            assert entry["phase"] is KVPhase.DECODE
            assert entry["decode_step"] == expected_step
            assert entry["query_shape"] == (1, 4, 1, 8)
            assert entry["key_shape"] == (1, 2, 5 + expected_step, 8)
            assert entry["query_positions"] == [4 + expected_step]
            assert entry["key_positions"] == list(range(5 + expected_step))

    assert all(entry["mask_is_none"] for entry in snapshots)
    assert all(entry["dropout"] == 0.0 for entry in snapshots)
    assert all(math.isclose(entry["scaling"], 1 / math.sqrt(8)) for entry in snapshots)
    assert len(probe.events) == len(snapshots)


def test_probe_preserves_generation_restores_registry_and_calls_previous_function_once(monkeypatch):
    torch.manual_seed(7)
    model = make_model()
    reference = make_model()
    reference.load_state_dict(model.state_dict())
    backend = model.config._attn_implementation
    registry_before = ALL_ATTENTION_FUNCTIONS[backend]
    previous_calls = []

    def counted_previous(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        previous_calls.append((int(module.layer_idx), tuple(query.shape), tuple(key.shape)))
        return registry_before(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, backend, counted_previous)
    probe = AttentionProbe(sample_id="tiny", prompt_length=4)
    with probe(model):
        assert ALL_ATTENTION_FUNCTIONS[backend] is not counted_previous
        observed = model.generate(torch.tensor([[1, 2, 3, 4]]), max_new_tokens=3, do_sample=False)

    assert ALL_ATTENTION_FUNCTIONS[backend] is counted_previous
    assert len(previous_calls) == len(probe.events) == 6

    baseline = reference.generate(torch.tensor([[1, 2, 3, 4]]), max_new_tokens=3, do_sample=False)
    assert torch.equal(observed, baseline)


def test_probe_restores_registry_after_callback_error():
    model = make_model()
    backend = model.config._attn_implementation
    registry_before = ALL_ATTENTION_FUNCTIONS[backend]

    def fail_callback(context):
        raise RuntimeError("deliberate callback failure")

    probe = AttentionProbe(sample_id="tiny", prompt_length=4, on_attention=fail_callback)
    with pytest.raises(RuntimeError, match="deliberate callback failure"):
        with probe(model):
            model(torch.tensor([[1, 2, 3, 4]]), use_cache=True)

    assert ALL_ATTENTION_FUNCTIONS[backend] is registry_before
    assert probe.backend_name is None


def test_probe_rejects_unvalidated_backend_without_changing_registry():
    model = make_model()
    model.config._attn_implementation = "eager"
    registry_before = ALL_ATTENTION_FUNCTIONS["sdpa"]
    probe = AttentionProbe(sample_id="tiny", prompt_length=4)

    with pytest.raises(UnsupportedAttentionBackendError, match="only sdpa"):
        with probe(model):
            pass

    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is registry_before
    assert probe.backend_name is None


def test_probe_rejects_incomplete_prefill():
    model = make_model()
    probe = AttentionProbe(sample_id="tiny", prompt_length=4)

    with pytest.raises(AttentionProbeError, match="complete Prefill"):
        with probe(model):
            model(torch.tensor([[1, 2, 3]]), use_cache=True)
