import math
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama import modeling_llama

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
    probe = AttentionProbe(sample_id="tiny", prompt_length=4, on_key_projection=lambda context: None)
    with probe(model):
        assert ALL_ATTENTION_FUNCTIONS[backend] is not counted_previous
        observed = model.generate(torch.tensor([[1, 2, 3, 4]]), max_new_tokens=3, do_sample=False)

    assert ALL_ATTENTION_FUNCTIONS[backend] is counted_previous
    assert len(previous_calls) == len(probe.events) == 6

    baseline = reference.generate(torch.tensor([[1, 2, 3, 4]]), max_new_tokens=3, do_sample=False)
    assert torch.equal(observed, baseline)


@pytest.mark.parametrize("callback", ["on_attention", "on_key_projection"])
def test_probe_restores_registry_after_callback_error(callback):
    model = make_model()
    backend = model.config._attn_implementation
    registry_before = ALL_ATTENTION_FUNCTIONS[backend]

    def fail_callback(context):
        raise RuntimeError("deliberate callback failure")

    probe = AttentionProbe(sample_id="tiny", prompt_length=4, **{callback: fail_callback})
    with pytest.raises(RuntimeError, match="deliberate callback failure"):
        with probe(model):
            model(torch.tensor([[1, 2, 3, 4]]), use_cache=True)

    assert ALL_ATTENTION_FUNCTIONS[backend] is registry_before
    assert probe.backend_name is None
    assert not probe._handles and not probe._active_calls
    assert all(not layer.self_attn.k_proj._forward_hooks for layer in model.model.layers)
    # An early projection-callback failure must also leave ordinary execution usable.
    with torch.inference_mode():
        assert torch.isfinite(model(torch.tensor([[1, 2, 3, 4]]), use_cache=True).logits).all()


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


def run_fixed_forwards(model):
    """One complete Prefill followed by three forced single-token Decode forwards."""
    cache = None
    logits, snapshots = [], []
    with torch.inference_mode():
        for tokens, positions in (([1, 2, 3, 4], [0, 1, 2, 3]), ([5], [4]), ([6], [5]), ([7], [6])):
            result = model(input_ids=torch.tensor([tokens]), cache_position=torch.tensor(positions),
                           past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            logits.append(result.logits.detach().cpu().clone())
            snapshots.append([(layer.keys.detach().cpu().clone(), layer.values.detach().cpu().clone())
                              for layer in cache.layers])
    return logits, snapshots


@pytest.mark.parametrize("dtype,norm_rtol", [(torch.float32, 1e-6), (torch.bfloat16, 1e-2)])
def test_pre_rope_and_attention_kv_match_projection_rope_cache_and_baseline(monkeypatch, dtype, norm_rtol):
    torch.manual_seed(23)
    model = make_model().to(dtype=dtype)
    # Match the real target's Llama 3.1 RoPE type, without loading any model weights.
    model.config.rope_parameters = {
        "rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0,
        "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 64,
    }
    model.model.rotary_emb = modeling_llama.LlamaRotaryEmbedding(model.config)
    assert model.model.rotary_emb.attention_scaling == 1.0
    baseline_logits, baseline_cache = run_fixed_forwards(model)
    projected_inputs, pre, post, rope_inputs = {}, {}, {}, []
    projection_calls = []

    def projection_spy(layer_id, module, args, output):
        forward = sum(item == layer_id for item in projection_calls)
        projection_calls.append(layer_id)
        projected_inputs[forward, layer_id] = args[0].detach().cpu().clone()

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(partial(projection_spy, i)))
    original_rope = modeling_llama.apply_rotary_pos_emb

    def counted_rope(q, k, cos, sin, **kwargs):
        rope_inputs.append((cos.detach().cpu().clone(), sin.detach().cpu().clone()))
        return original_rope(q, k, cos, sin, **kwargs)

    monkeypatch.setattr(modeling_llama, "apply_rotary_pos_emb", counted_rope)

    def capture_pre(context):
        pre[context.forward_id, context.layer_id] = {
            "k": context.key_pre.detach().cpu().clone(), "positions": context.token_positions.tolist(),
            "phase": context.phase, "step": context.decode_step, "sample": context.sample_id,
        }

    def capture_post(context):
        post[context.forward_id, context.layer_id] = {
            "k": context.key.detach().cpu().clone(), "v": context.value.detach().cpu().clone(),
            "positions": context.query_positions.tolist(), "phase": context.phase, "step": context.decode_step,
        }

    probe = AttentionProbe(sample_id="kv_tiny", prompt_length=4,
                           on_key_projection=capture_pre, on_attention=capture_post)
    try:
        with probe(model):
            observed_logits, observed_cache = run_fixed_forwards(model)
    finally:
        for handle in handles:
            handle.remove()
    assert len(projection_calls) == len(rope_inputs) == len(pre) == len(post) == 8
    assert projection_calls == [0, 1] * 4
    errors = []
    for forward in range(4):
        torch.testing.assert_close(observed_logits[forward], baseline_logits[forward], rtol=0, atol=0)
        for layer_id, layer in enumerate(model.model.layers):
            before, after = pre[forward, layer_id], post[forward, layer_id]
            positions = [0, 1, 2, 3] if forward == 0 else [3 + forward]
            assert before["positions"] == after["positions"] == positions
            assert before["phase"] == after["phase"] == (KVPhase.PREFILL if forward == 0 else KVPhase.DECODE)
            assert before["step"] == after["step"] == (None if forward == 0 else forward - 1)
            assert before["sample"] == "kv_tiny"
            with torch.no_grad():
                linear = F.linear(projected_inputs[forward, layer_id], layer.self_attn.k_proj.weight,
                                  layer.self_attn.k_proj.bias)
                value_linear = F.linear(projected_inputs[forward, layer_id], layer.self_attn.v_proj.weight,
                                        layer.self_attn.v_proj.bias)
            expected_pre = linear.view(1, len(positions), 2, 8).transpose(1, 2)
            torch.testing.assert_close(before["k"], expected_pre, rtol=0, atol=0)
            cos, sin = rope_inputs[forward * 2 + layer_id]
            rotated_half = torch.cat((-before["k"][..., 4:], before["k"][..., :4]), dim=-1)
            expected_post = before["k"] * cos.unsqueeze(1) + rotated_half * sin.unsqueeze(1)
            new_post = after["k"][:, :, positions]
            torch.testing.assert_close(new_post, expected_post, rtol=0, atol=0)
            expected_v = value_linear.view(1, len(positions), 2, 8).transpose(1, 2)
            torch.testing.assert_close(after["v"][:, :, positions], expected_v, rtol=0, atol=0)
            cache_k, cache_v = observed_cache[forward][layer_id]
            torch.testing.assert_close(after["k"], cache_k, rtol=0, atol=0)
            torch.testing.assert_close(after["v"], cache_v, rtol=0, atol=0)
            for value, reference in zip((cache_k, cache_v), baseline_cache[forward][layer_id]):
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
            if forward > 0:
                for value, reference in zip((cache_k, cache_v), observed_cache[0][layer_id]):
                    torch.testing.assert_close(value[:, :, :4], reference, rtol=0, atol=0)
            norm_pre = before["k"].float().norm(dim=-1)
            norm_post = new_post.float().norm(dim=-1)
            errors.append(float(((norm_post - norm_pre).abs() / norm_pre.clamp_min(1e-12)).max()))
    assert max(errors) <= norm_rtol
    print(f"{dtype}: maximum sampled K norm relative difference = {max(errors):.8g}")
