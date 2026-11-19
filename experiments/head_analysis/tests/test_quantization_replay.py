"""V3-B numerical encoding, Dense intervention, and isolated-checkpoint checks."""

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from experiments.head_analysis.runners.b2_runner import teacher_forced_replay
from experiments.head_analysis.runners.b2_runner import B1ReplaySample
from experiments.head_analysis.runners import quantization_runner as runner
from experiments.head_analysis.collectors.quantization_sensitivity import QuantizationSensitivityCollector
from experiments.head_analysis.common.io import read_json, write_json
from experiments.head_analysis.storage.quantization_results import QuantizationResultWriter
from kvpress.context import KVPhase
from experiments.head_analysis.runners.quantization_runner import (
    prediction_errors,
    replay_quantization_checkpoints,
    run_checkpoint_branches,
)
from experiments.head_analysis.runtime.head_intervention import HeadIntervention, single_head_intervention
from experiments.head_analysis.runners.quantization_diagnostics import (
    D1_SEQUENCE,
    replay_d1_checkpoints,
    run_d1_checkpoint,
)
from experiments.head_analysis.runners.quantization_runner import isolated_checkpoint_forward
from experiments.head_analysis.runtime.kv_quantization import (
    pack_codes,
    quantization_byte_ledger,
    quantize_head,
    quantized_prompt_range,
    unpack_codes,
)


def make_model(dtype=torch.float32):
    torch.manual_seed(83)
    config = LlamaConfig(
        vocab_size=48,
        hidden_size=256,
        intermediate_size=288,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        eos_token_id=None,
        pad_token_id=None,
    )
    config._attn_implementation = "sdpa"
    return LlamaForCausalLM(config).eval().to(dtype=dtype)


def prompt_ids():
    # A 13-token middle residual must stay BF16 in addition to the 64+64 guards.
    return [(i % 47) + 1 for i in range(205)]


def snapshot(cache):
    return [(layer, layer.keys, layer.values) for layer in cache.layers]


def restore(history):
    for layer, keys, values in history:
        layer.keys, layer.values = keys, values


@pytest.mark.parametrize("side", ["key", "value"])
@pytest.mark.parametrize("bits", [4, 8])
def test_quantizer_matches_independent_group_reference_and_real_packing(side, bits):
    torch.manual_seed(4)
    x = torch.randn(64, 128).bfloat16()
    encoded = quantize_head(x, side, bits, block_rows=32)
    whole = quantize_head(x, side, bits, block_rows=1024)
    for field in ("codes", "minimum", "scale"):
        assert torch.equal(getattr(encoded, field), getattr(whole, field))
    expected = torch.empty_like(x, dtype=torch.float32)
    expected_codes = torch.empty_like(x, dtype=torch.uint8)
    # A slow direct grouping reference, independent of the codec's reshape order.
    for a in range(0, x.shape[0], 32 if side == "key" else 1):
        for b in range(0, x.shape[1], 1 if side == "key" else 32):
            index = (slice(a, a + 32), b) if side == "key" else (a, slice(b, b + 32))
            group = x[index].float()
            low, high = group.min(), group.max()
            scale = (high - low) / (2**bits - 1)
            codes = torch.tensor([round(float((v - low) / scale)) for v in group], dtype=torch.uint8)
            expected_codes[index] = codes
            expected[index] = low + scale * codes.float()
    assert torch.equal(unpack_codes(encoded.codes, bits, x.numel()).reshape_as(x), expected_codes)
    torch.testing.assert_close(encoded.decode(torch.float32), expected, rtol=0, atol=0)
    assert torch.equal(encoded.decode(), expected.bfloat16())
    assert encoded.minimum.dtype == encoded.scale.dtype == torch.float32
    physical = b"".join(t.cpu().numpy().tobytes() for t in (encoded.codes, encoded.minimum, encoded.scale))
    assert len(physical) == encoded.payload_nbytes == x.numel() * bits // 8 + x.numel() // 32 * 8


@pytest.mark.parametrize("side", ["key", "value"])
def test_constant_groups_and_round_to_even(side):
    constant = torch.full((32, 32), 1.25, dtype=torch.bfloat16)
    encoded = quantize_head(constant, side, 4)
    assert not encoded.codes.any() and not encoded.scale.any()
    assert torch.equal(encoded.decode(), constant)
    values = torch.tensor([0.0, 15.0, 0.5, 1.5, 2.5, 3.5] + [0.0] * 26)
    x = values[:, None].repeat(1, 32) if side == "key" else values[None].repeat(32, 1)
    q = unpack_codes(quantize_head(x, side, 4).codes, 4, x.numel()).reshape_as(x)
    actual = q[:6, 0] if side == "key" else q[0, :6]
    assert actual.tolist() == [0, 15, 0, 2, 2, 4]


def test_nibbles_odd_tail_and_unsupported_quantization_do_not_silently_fallback():
    q = torch.tensor([0, 1, 15, 14, 2], dtype=torch.uint8)
    packed = pack_codes(q, 4)
    assert packed.tolist() == [0x10, 0xEF, 0x02]
    assert torch.equal(unpack_codes(packed, 4, 5), q)
    with pytest.raises(ValueError, match="4/8"):
        quantize_head(torch.ones(32, 32), "key", 2)
    with pytest.raises(ValueError, match="multiples of 32"):
        quantize_head(torch.ones(33, 32), "key", 4)
    with pytest.raises(ValueError, match=r"\[0, 15\]"):
        pack_codes(torch.tensor([16], dtype=torch.uint8), 4)


@pytest.mark.parametrize("side", ["key", "value"])
@pytest.mark.parametrize("bits", [4, 8])
def test_byte_ledger_counts_protection_residual_new_rows_metadata_and_padding(side, bits):
    start, end = quantized_prompt_range(205)
    assert (start, end) == (64, 128)
    encoded = quantize_head(torch.randn(end - start, 128).bfloat16(), side, bits)
    ledger = quantization_byte_ledger(205, 3, 32, 8, 128, bits)
    assert ledger["code_bytes"] + ledger["statistic_bytes"] == encoded.payload_nbytes
    assert ledger["unquantized_target_side_bytes"] == (64 + 64 + 13 + 3) * 128 * 2
    other_data = ledger["dense_kv_data_bytes"] - (205 + 3) * 128 * 2
    assert ledger["request_bytes"] == other_data + encoded.payload_nbytes + ledger["unquantized_target_side_bytes"] + 64
    assert ledger["total_bytes"] == ledger["request_bytes"] + 24 + 40
    bypass = quantization_byte_ledger(205, 3, 32, 8, 128, None)
    assert bypass["total_bytes"] - ledger["total_bytes"] == ledger["net_saved_bytes"]
    assert bypass["model_bytes"] == 0 and bypass["net_saved_bytes"] == 0


@pytest.mark.parametrize("side", ["key", "value"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_intervention_changes_only_target_middle_keeps_dense_domain_and_gqa(side, dtype, monkeypatch):
    model = make_model(dtype)
    prompt = prompt_ids()
    backend = ALL_ATTENTION_FUNCTIONS["sdpa"]
    observed = []

    def capture(module, q, k, v, mask, **kwargs):
        result = backend(module, q, k, v, mask, **kwargs)
        if module.layer_idx == 1:
            observed.append((q.clone(), k.clone(), v.clone(), result[0].clone()))
        return result

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, "sdpa", capture)
    with torch.inference_mode():
        cache = model(torch.tensor([prompt]), use_cache=True, logits_to_keep=1).past_key_values
        history = snapshot(cache)
        original = [(k.clone(), v.clone()) for _, k, v in history]
        observed.clear()
        model(torch.tensor([[7]]), past_key_values=cache, cache_position=torch.tensor([205]), logits_to_keep=1)
        restore(history)
        configs = [layer.self_attn.config for layer in model.model.layers]
        with single_head_intervention(model, HeadIntervention(1, 1, side, 4), 205, block_rows=32) as state:
            assert model.model.layers[0].self_attn.config is configs[0]
            assert model.model.layers[1].self_attn.config is not configs[1]
            model(torch.tensor([[7]]), past_key_values=cache, cache_position=torch.tensor([205]), logits_to_keep=1)
        before, after = observed
        start, end = quantized_prompt_range(205)
        side_index = 1 if side == "key" else 2
        for i in (0, 1, 2):
            if i != side_index:
                assert torch.equal(before[i], after[i])
        reference, changed = before[side_index], after[side_index]
        assert reference.shape == changed.shape == (1, 2, 206, 32)
        assert torch.equal(reference[:, 0], changed[:, 0])
        assert torch.equal(reference[:, 1, :start], changed[:, 1, :start])
        assert torch.equal(reference[:, 1, end:], changed[:, 1, end:])
        assert not torch.equal(reference[:, 1, start:end], changed[:, 1, start:end])
        assert torch.equal(before[3][:, :, :4], after[3][:, :, :4])
        assert not torch.equal(before[3][:, :, 4:], after[3][:, :, 4:])
        q, k, v, output = after
        # Full-domain manual QK/softmax/AV verifies all 206 positions, not a Top-K subset.
        for head in range(2):
            group = q[:, head * 4 : (head + 1) * 4].float()
            probabilities = (group @ k[:, head : head + 1].float().transpose(-1, -2) / math.sqrt(32)).softmax(-1)
            expected = probabilities @ v[:, head : head + 1].float()
            torch.testing.assert_close(
                output[:, :, head * 4 : (head + 1) * 4].float(),
                expected.transpose(1, 2),
                rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
                atol=1e-3 if dtype == torch.bfloat16 else 1e-7,
            )
        for layer, (old_key, old_value) in zip(cache.layers, original):
            assert torch.equal(layer.keys[:, :, :205], old_key)
            assert torch.equal(layer.values[:, :, :205], old_value)
        assert all(layer.self_attn.config is config for layer, config in zip(model.model.layers, configs))
        assert ALL_ATTENTION_FUNCTIONS["sdpa"] is capture
        assert "headkv_single_head" not in ALL_ATTENTION_FUNCTIONS
        assert state.measurement.cache_length == 206 and state.measurement.quantized_tokens == 64
        target_original = reference[0, 1, start:end].double()
        direct_error = (changed[0, 1, start:end].double() - target_original).square().sum().item()
        assert state.measurement.squared_error == pytest.approx(direct_error, rel=1e-12, abs=1e-12)
        assert state.measurement.nmse > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_replay_keeps_main_logits_and_reuses_one_prefill(dtype):
    model = make_model(dtype)
    prompt = prompt_ids()
    with torch.inference_mode():
        ids = model.generate(torch.tensor([prompt]), max_new_tokens=5, do_sample=False)[0, len(prompt) :].tolist()
    baseline = teacher_forced_replay(
        model, device="cpu", sample_id="tiny", prompt_ids=prompt, replay_generated_ids=ids, capture_logits=True
    )
    lengths = []
    hook = model.register_forward_pre_hook(
        lambda m, args, kwargs: lengths.append(kwargs["input_ids"].shape[1]), with_kwargs=True
    )
    targets = [
        HeadIntervention(0, 0, "key", None),
        HeadIntervention(0, 0, "key", 4),
        HeadIntervention(1, 1, "key", 8),
        HeadIntervention(1, 1, "value", 4),
        HeadIntervention(0, 1, "value", 8),
    ]
    try:
        result = replay_quantization_checkpoints(
            model,
            device="cpu",
            sample_id="tiny",
            prompt_ids=prompt,
            replay_generated_ids=ids,
            checkpoints={0: targets, 2: targets},
            capture_logits=True,
            block_rows=32,
        )
    finally:
        hook.remove()
    assert lengths.count(len(prompt)) == 1
    assert len(lengths) == 1 + 4 + 2 * (1 + len(targets))
    for actual, expected in zip(result.baseline.logits, baseline.logits):
        assert torch.equal(actual, expected)
    assert result.baseline.all_predicted_token_ids == tuple(ids)
    for checkpoint in result.checkpoints:
        s = checkpoint.decode_step
        assert checkpoint.input_token_id == ids[s] and checkpoint.expected_token_id == ids[s + 1]
        for branch in checkpoint.branches:
            assert branch.measurement.cache_length == len(prompt) + s + 1
            assert branch.logits.device.type == "cpu" and branch.prediction_kl >= -1e-6
            if branch.target.bits is None:
                assert torch.equal(branch.logits, baseline.logits[s + 1])
                assert branch.prediction_kl == branch.delta_nll == 0 and branch.top1_match
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


def test_branch_order_failure_cleanup_and_future_baseline_are_independent(monkeypatch):
    model = make_model()
    prompt = prompt_ids()
    with torch.inference_mode():
        output = model(torch.tensor([prompt]), use_cache=True, logits_to_keep=1)
    cache = output.past_key_values
    history = snapshot(cache)
    copies = [(k.clone(), v.clone()) for _, k, v in history]
    targets = [HeadIntervention(0, 0, "key", 4), HeadIntervention(1, 1, "value", 4)]
    args = dict(
        cache=cache,
        device="cpu",
        prompt_length=len(prompt),
        decode_step=0,
        input_token=int(output.logits[0, -1].argmax()),
        expected_token=7,
        capture_logits=True,
    )
    first = run_checkpoint_branches(model, targets=targets, **args)
    reversed_result = run_checkpoint_branches(model, targets=targets[::-1], **args)
    for a, b in zip(first.branches, reversed_result.branches[::-1]):
        assert a.target == b.target and torch.equal(a.logits, b.logits)
        assert a.prediction_kl == b.prediction_kl

    # Fail after both layers append, not before model execution.
    projection = model.model.layers[1].self_attn.o_proj
    original_forward = projection.forward

    def fail_only_during_intervention(*args, **kwargs):
        if "headkv_single_head" in ALL_ATTENTION_FUNCTIONS:
            raise RuntimeError("deliberate branch failure")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(projection, "forward", fail_only_during_intervention)
    with pytest.raises(RuntimeError, match="deliberate branch failure"):
        run_checkpoint_branches(model, targets=targets, **args)
    for (layer, key, value), (key_copy, value_copy) in zip(history, copies):
        assert layer.keys is key and layer.values is value
        assert torch.equal(layer.keys, key_copy) and torch.equal(layer.values, value_copy)
    assert "headkv_single_head" not in ALL_ATTENTION_FUNCTIONS
    assert all(layer.self_attn.config is model.config for layer in model.model.layers)
    monkeypatch.setattr(projection, "forward", original_forward)
    recovered = run_checkpoint_branches(model, targets=targets[:1], **args)
    assert torch.equal(recovered.branches[0].logits, first.branches[0].logits)
    with pytest.raises(ValueError, match="Cache length"):
        run_checkpoint_branches(model, targets=targets, **{**args, "decode_step": 1})


def test_prediction_metrics_use_full_vocabulary_and_signed_nll_change():
    reference = torch.tensor([2.0, 1.0, -2.0])
    branch = torch.tensor([2.0, 1.0, 0.0])  # Same Top-1 and same two largest logits; tail changes.
    result = prediction_errors(reference, branch, 2)
    p, q = reference.double().softmax(0), branch.double().softmax(0)
    assert result["prediction_kl"] == pytest.approx(float((p * (p.log() - q.log())).sum()), abs=1e-7)
    assert result["top1_match"] and result["prediction_kl"] > 0
    assert result["delta_nll"] < 0


def attention_context(dtype=torch.bfloat16):
    torch.manual_seed(9)
    return SimpleNamespace(
        sample_id="tiny",
        layer_id=0,
        phase=KVPhase.DECODE,
        decode_step=2,
        prompt_length=205,
        scaling=32**-0.5,
        attention_mask=None,
        query=torch.randn(1, 8, 1, 32).to(dtype),
        key=torch.randn(1, 2, 208, 32).to(dtype),
        value=torch.randn(1, 2, 208, 32).to(dtype),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_head_collector_matches_fp64_qk_softmax_av_and_keeps_query_groups(dtype):
    context = attention_context(dtype)
    originals = [t.clone() for t in (context.query, context.key, context.value)]
    record = QuantizationSensitivityCollector(205, lambda record: None, block_rows=32).collect(context)
    arrays = record.arrays
    assert arrays["output_nmse"].shape == (2, 2, 2)
    assert arrays["per_query_output_nmse"].shape == (2, 2, 2, 4)
    assert arrays["formula_sdpa_squared_error"].shape == (2, 4)
    assert not arrays["per_query_attention_kl"][1].any()
    for head in range(2):
        q = context.query[0, head * 4 : (head + 1) * 4, 0].double()
        key, value = context.key[0, head].double(), context.value[0, head].double()
        p = (q @ key.T * context.scaling).softmax(-1)
        output = p @ value
        for si, side in enumerate(("key", "value")):
            for bi, bits in enumerate((4, 8)):
                modified = (context.key if side == "key" else context.value)[0, head].clone()
                modified[64:128] = quantize_head(modified[64:128], side, bits).decode(dtype)
                if side == "key":
                    probability = (q @ modified.double().T * context.scaling).softmax(-1)
                    actual = probability @ value
                    kl = (p * (p.log() - probability.log())).sum(-1).mean().item()
                else:
                    actual, kl = p @ modified.double(), 0.0
                error = (actual - output).square().sum(-1)
                energy = output.square().sum(-1)
                index = si, bi, head
                assert arrays["output_nmse"][index] == pytest.approx(
                    (error.sum() / energy.sum()).item(), rel=1e-3, abs=2e-8
                )
                np.testing.assert_allclose(
                    arrays["per_query_output_nmse"][index], (error / energy).numpy(), rtol=1e-3, atol=2e-8
                )
                assert arrays["attention_kl"][index] == pytest.approx(kl, rel=1e-3, abs=3e-7)
                assert arrays["representation_valid"][index] and arrays["output_valid"][index]
    for current, original in zip((context.query, context.key, context.value), originals):
        assert torch.equal(current, original)


def test_zero_energy_is_invalid_not_epsilon_and_partial_domain_is_rejected():
    context = attention_context()
    context.value.zero_()
    arrays = QuantizationSensitivityCollector(205, lambda record: None).collect(context).arrays
    assert np.isnan(arrays["output_nmse"]).all()
    assert not arrays["output_valid"].any() and not arrays["output_squared_error"].any()
    assert not arrays["representation_valid"][1].any()
    context.key = context.key[:, :, :-1]
    with pytest.raises(ValueError, match="complete continuous"):
        QuantizationSensitivityCollector(205, lambda record: None).collect(context)


def tiny_sample(model, sample_id="tiny", dataset="govreport", length=205, offset=0, generated_length=4):
    prompt = [((i + offset) % 47) + 1 for i in range(length)]
    with torch.inference_mode():
        ids = model.generate(torch.tensor([prompt]), max_new_tokens=generated_length, do_sample=False)[
            0, length:
        ].tolist()
    record = {"sample_id": sample_id, "dataset": dataset, "input_ids": prompt, "actual_prompt_length": length}
    return B1ReplaySample(record, "tiny_batch", "tiny_run", tuple(ids), "unused", "unused")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_collector_branch_writer_end_to_end_has_complete_axes_and_matching_encoding(tmp_path, dtype):
    model = make_model(dtype)
    sample = tiny_sample(model)
    directory = tmp_path / "collected"
    root_lengths = []
    hook = model.register_forward_pre_hook(
        lambda m, args, kwargs: root_lengths.append(kwargs["input_ids"].shape[1]), with_kwargs=True
    )
    try:
        summary = runner.collect_quantization_sample(
            model,
            device="cpu",
            sample=sample,
            directory=directory,
            model_layers=(0, 1),
            local_steps=(0, 2),
            block_rows=32,
        )
    finally:
        hook.remove()
    metadata = read_json(directory / "metadata.json")
    assert metadata["status"] == "completed" and metadata["baseline_top1_match"]
    assert metadata["completed_checkpoint_steps"] == [0, 2]
    assert root_lengths.count(205) == 1
    assert len(root_lengths) == 1 + 3 + 2 + 17
    assert summary["resources"]["forward_counts"] == {
        "prefill": 1,
        "main_decode": 3,
        "checkpoint_reference": 2,
        "quantized_branches": 16,
        "bypass_branches": 1,
    }
    assert summary["resources"]["local_quantization_cases"] == 32
    replay = read_json(directory / "replay.json")
    assert tuple([replay["prefill_predicted_token_id"], *replay["decode_predicted_token_ids"]]) == sample.generated_ids
    with np.load(directory / "local_step_000.npz") as local, np.load(directory / "model_branches.npz") as branches:
        assert local["output_nmse"].shape == (2, 2, 2, 2)
        assert local["per_query_output_nmse"].shape == (2, 2, 2, 2, 4)
        assert local["formula_sdpa_squared_error"].shape == (2, 2, 4)
        assert branches["bits"].shape == (17,)
        for row, bits in enumerate(branches["bits"]):
            if bits == 16:
                assert branches["prediction_kl"][row] == 0 and branches["top1_match"][row]
                continue
            index = (
                int(branches["side_id"][row]),
                (4, 8).index(int(bits)),
                int(branches["layer_id"][row]),
                int(branches["head_id"][row]),
            )
            assert branches["squared_error"][row] == pytest.approx(
                local["representation_squared_error"][index], abs=1e-12
            )
            assert branches["original_squared_sum"][row] == pytest.approx(
                local["representation_energy"][index], abs=1e-12
            )
            assert branches["code_bytes"][row] == local["code_bytes"][index]
            assert branches["statistic_bytes"][row] == local["statistic_bytes"][index]
        assert all(a.size <= 64 for a in local.values())
        assert "logits" not in branches
    with np.load(directory / "local_step_002.npz") as local:
        expected = local["output_nmse"][0, 0]
        assert summary["local"]["2"]["key"]["4"]["output_nmse"]["mean"] == pytest.approx(float(expected.mean()))
    with pytest.raises(FileExistsError):
        runner.collect_quantization_sample(
            model, device="cpu", sample=sample, directory=directory, model_layers=(0, 1), local_steps=(0, 2)
        )


def test_full_256_step_protocol_keeps_prompt_encoding_and_late_checkpoint_local_only(tmp_path):
    model = make_model(torch.bfloat16)
    sample = tiny_sample(model, generated_length=257)
    directory = tmp_path / "full"
    summary = runner.collect_quantization_sample(
        model, device="cpu", sample=sample, directory=directory, model_layers=(0, 1)
    )
    assert summary["resources"]["forward_counts"] == {
        "prefill": 1,
        "main_decode": 256,
        "checkpoint_reference": 2,
        "quantized_branches": 16,
        "bypass_branches": 1,
    }
    metadata = read_json(directory / "metadata.json")
    assert metadata["local_steps"] == [31, 255] and metadata["primary_step"] == 31
    replay = read_json(directory / "replay.json")
    for step in (31, 255):
        assert replay["checkpoints"][str(step)]["input_token_id"] == sample.generated_ids[step]
        assert replay["checkpoints"][str(step)]["expected_token_id"] == sample.generated_ids[step + 1]
    with np.load(directory / "local_step_031.npz") as early, np.load(directory / "local_step_255.npz") as late:
        np.testing.assert_array_equal(early["representation_squared_error"], late["representation_squared_error"])
        np.testing.assert_array_equal(early["code_bytes"], late["code_bytes"])
    with np.load(directory / "model_branches.npz") as branches:
        assert len(branches["bits"]) == 17 and np.all(branches["cache_length"] == 205 + 32)
    ledger = read_json(directory / "byte_ledger.json")["by_step"]
    assert ledger["255"]["4"]["request_bytes"] - ledger["31"]["4"]["request_bytes"] == 224 * 4 * 2 * 2 * 32


def test_failed_sample_and_incomplete_writer_never_publish_completed(tmp_path):
    model = make_model()
    sample = tiny_sample(model)
    bad = replace(sample, generated_ids=(*sample.generated_ids[:-1], (sample.generated_ids[-1] + 1) % 48))
    with pytest.raises(ValueError, match="checkpoint reference differs"):
        runner.collect_quantization_sample(
            model, device="cpu", sample=bad, directory=tmp_path / "failed", model_layers=(0, 1), local_steps=(0, 2)
        )
    assert read_json(tmp_path / "failed/metadata.json")["status"] == "failed"
    assert not (tmp_path / "failed/summary.json").exists()
    assert "headkv_single_head" not in ALL_ATTENTION_FUNCTIONS
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    writer = QuantizationResultWriter(
        tmp_path / "incomplete",
        sample_id="tiny",
        prompt_length=205,
        num_layers=2,
        query_heads=8,
        kv_heads=2,
        head_dim=32,
        local_steps=(2,),
        targets=(),
        source={},
        execution_dtype="float32",
    )
    record = QuantizationSensitivityCollector(205, lambda record: None).collect(attention_context())
    writer.write_local(record)
    with pytest.raises(ValueError, match="duplicate"):
        writer.write_local(record)
    malformed = replace(
        record, layer_id=1, arrays={**record.arrays, "output_nmse": record.arrays["output_nmse"][..., :1]}
    )
    with pytest.raises(ValueError, match="shape"):
        writer.write_local(malformed)
    with pytest.raises(ValueError, match="missing"):
        writer.finalize((), None, {})
    assert read_json(tmp_path / "incomplete/metadata.json")["status"] == "collecting"


def test_cli_calib_then_validation_freezes_profiles_and_checks_phase_before_model_load(tmp_path, monkeypatch):
    from experiments.head_analysis.scripts.run_quantization_sensitivity import main

    model = make_model(torch.bfloat16)
    monkeypatch.setattr(runner, "MODEL_LAYERS", (0, 1))
    monkeypatch.setattr(runner, "LOCAL_STEPS", {"ruler": (0,), "govreport": (0, 2), "pg19": (0, 2)})
    records, samples = [], {}
    for i, sid in enumerate((*runner.PILOT_IDS, *runner.VALIDATION_IDS)):
        parts = sid.split("_")
        dataset = parts[0]
        samples[sid] = tiny_sample(model, sid, dataset, length=205 + 32 * (i % 2), offset=i)
        condition = "pg19_" + parts[-1] if dataset == "pg19" else "_".join(parts[:2])
        source = "_".join(parts[:2]) if dataset == "pg19" else sid
        records.append(
            {
                "sample_id": sid,
                "condition": condition,
                "source_id": source,
                "split": "calib" if sid in runner.PILOT_IDS else "validation",
            }
        )
    bundle = tmp_path / "bundle"
    write_json(bundle / "split.json", {"samples": records})
    index = SimpleNamespace(
        source_root=tmp_path,
        payload={
            "execution_contract": {
                "attention_backend": "sdpa",
                "model": {"path": "tiny", "tokenizer_path": "tiny", "dtype": "bfloat16"},
            }
        },
        batches=(SimpleNamespace(batch_id="tiny", run_metadata={"tokenizer": "tiny"}),),
    )
    monkeypatch.setattr(runner, "load_b1_replay_index", lambda *a, **k: index)
    monkeypatch.setattr(runner, "load_replay_samples", lambda index, ids: [samples[sid] for sid in ids])
    loads = []

    def load_model(*args):
        loads.append(args)
        return model, torch.device("cpu")

    monkeypatch.setattr(runner, "load_model", load_model)
    base = [
        "--bundle",
        str(bundle),
        "--run-set",
        str(tmp_path / "tiny_b1.json"),
        "--model",
        "tiny",
        "--tokenizer",
        "tiny",
        "--device",
        "cpu",
        "--block-rows",
        "32",
    ]
    calib, validation = tmp_path / "calib", tmp_path / "validation"
    assert main([*base, "--output-dir", str(calib)]) == 0
    assert read_json(calib / "run_metadata.json")["status"] == "completed"
    frozen_config = (calib / "config.json").read_bytes()
    frozen_profile = (calib / "calib_profile.npz").read_bytes()
    local = []
    for sid in runner.PILOT_IDS:
        with np.load(calib / "samples" / sid / "local_step_000.npz") as archive:
            local.append(archive["output_nmse"])
    with np.load(calib / "calib_profile.npz") as archive:
        np.testing.assert_allclose(archive["local_output_nmse"], np.mean(local, 0), rtol=0, atol=0)
        assert np.all(archive["local_output_nmse_valid_condition_count"] == 6)
        assert np.all(archive["prediction_kl_valid_condition_count"] == 6)
    assert main([*base, "--cohort", "validation", "--calib-dir", str(calib), "--output-dir", str(validation)]) == 0
    assert read_json(validation / "run_metadata.json")["completed_sample_ids"] == list(runner.VALIDATION_IDS)
    assert (calib / "config.json").read_bytes() == frozen_config
    assert (calib / "calib_profile.npz").read_bytes() == frozen_profile
    assert len(loads) == 2
    monkeypatch.setattr(runner, "MODEL_LAYERS", (1,))
    with pytest.raises(ValueError, match="differs from frozen Calib"):
        main([*base, "--cohort", "validation", "--calib-dir", str(calib), "--output-dir", str(tmp_path / "changed")])
    assert len(loads) == 2 and not (tmp_path / "changed").exists()
    with pytest.raises(FileExistsError):
        main([*base, "--output-dir", str(calib)])
    assert sum(p.stat().st_size for p in calib.rglob("*") if p.is_file()) < 2**20


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("side", ["key", "value"])
def test_d1_sequence_replay_isolation_fixed_reuse_and_observation_toggle(dtype, side, monkeypatch):
    from experiments.head_analysis.runtime import head_intervention as intervention
    from dataclasses import asdict
    import json

    model, prompt = make_model(dtype), prompt_ids()
    with torch.inference_mode():
        ids = model.generate(torch.tensor([prompt]), max_new_tokens=5, do_sample=False)[0, len(prompt) :].tolist()
    baseline = teacher_forced_replay(
        model, device="cpu", sample_id="tiny", prompt_ids=prompt, replay_generated_ids=ids, capture_logits=True
    )
    encoding_calls, lengths, observed = [], [], []
    encode = intervention.quantize_and_restore

    def count_encoding(*args):
        encoding_calls.append((args[1], args[2]))
        return encode(*args)

    monkeypatch.setattr(intervention, "quantize_and_restore", count_encoding)
    handle = model.register_forward_pre_hook(
        lambda m, args, kw: lengths.append(kw["input_ids"].shape[1]), with_kwargs=True
    )
    observer = lambda c: observed.append((c.decode_step, c.layer_id))
    targets = {0: [HeadIntervention(1, 1, side, None)], 2: [HeadIntervention(0, 0, side, None)]}
    try:
        result = replay_d1_checkpoints(
            model,
            device="cpu",
            sample_id="tiny",
            prompt_ids=prompt,
            replay_generated_ids=ids,
            checkpoints=targets,
            capture_logits=True,
            block_rows=32,
            on_attention=observer,
        )
    finally:
        handle.remove()
    assert lengths.count(len(prompt)) == 1 and len(lengths) == 1 + 4 + 2 * 12
    assert len(encoding_calls) == 8  # First/repeated Q4/Q8, never D4/D8 or controls.
    assert observed == [(s, l) for s in (0, 2) for _ in range(3) for l in range(2)]
    for expected, actual in zip(baseline.logits, result.baseline.logits):
        assert torch.equal(expected, actual)
    for checkpoint in result.checkpoints:
        assert tuple(r["label"] for r in checkpoint.records) == D1_SEQUENCE
        assert all(
            c["logits_equal"] and c["max_absolute_difference"] == c["prediction_kl"] == 0
            for c in checkpoint.comparisons.values()
        )
        records = {r["label"]: r for r in checkpoint.records}
        for label, record in records.items():
            assert all(record["input_checks"].values())
            if record["isolation_checks"] is not None:
                assert all(record["isolation_checks"].values())
            assert record["encoding_performed"] == label.startswith("Q")
            assert record["layout"]["actual"]["cache_position"] == [len(prompt) + checkpoint.decode_step]
        for bits in (4, 8):
            assert records[f"D{bits}"]["mode"] == "fixed"
            assert records[f"D{bits}"]["measurement"] == records[f"Q{bits}"]["measurement"]
            assert records[f"Q{bits}"]["measurement"]["squared_error"] > 0
        for label in ("B0", "B1", "B2", "N", "C0", "C1"):
            z = checkpoint.logits[D1_SEQUENCE.index(label)]
            assert torch.equal(z, baseline.logits[checkpoint.decode_step + 1])
        json.dumps({"records": checkpoint.records, "comparisons": checkpoint.comparisons})
    # Same sequence and original observer, with the new diagnostic hooks disabled.
    observed.clear()
    bare = replay_d1_checkpoints(
        model,
        device="cpu",
        sample_id="tiny",
        prompt_ids=prompt,
        replay_generated_ids=ids,
        checkpoints={0: targets[0]},
        capture_logits=True,
        check_inputs=False,
        block_rows=32,
        on_attention=observer,
    )
    assert observed == [(0, l) for _ in range(3) for l in range(2)]
    for a, b in zip(result.checkpoints[0].logits, bare.checkpoints[0].logits):
        assert torch.equal(a, b)
    assert set(asdict(HeadIntervention(0, 0, side, 4))) == {"layer_id", "head_id", "side", "bits"}
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    assert all(layer.self_attn.config is model.config for layer in model.model.layers)
    assert "headkv_single_head" not in ALL_ATTENTION_FUNCTIONS


@pytest.mark.parametrize("side", ["key", "value"])
def test_fixed_reconstruction_binds_input_layout_and_skips_encoding(side, monkeypatch):
    from experiments.head_analysis.runtime import head_intervention as intervention

    model, prompt = make_model(torch.bfloat16), prompt_ids()
    with torch.inference_mode():
        output = model(torch.tensor([prompt]), use_cache=True, logits_to_keep=1)
    cache, target = output.past_key_values, HeadIntervention(1, 1, side, 4)
    history = snapshot(cache)
    original = [(k.clone(), v.clone()) for _, k, v in history]
    with isolated_checkpoint_forward(
        model, cache=cache, device="cpu", prompt_length=len(prompt), decode_step=0, input_token=7, block_rows=32
    ) as forward:
        q_logits, q_state = forward(target, capture_reconstruction=True, check_isolation=True)
        frozen = q_state.reconstruction

        def forbidden(*args):
            raise AssertionError("fixed injection must not quantize again")

        monkeypatch.setattr(intervention, "quantize_and_restore", forbidden)
        logits, state = forward(target, fixed_reconstruction=frozen, check_isolation=True)
        assert torch.equal(q_logits, logits) and not state.encoding_performed
        assert state.measurement is frozen.measurement
        for invalid in (
            replace(frozen, restored=frozen.restored.double()),
            replace(frozen, original=frozen.original + 1),
            replace(frozen, measurement=replace(frozen.measurement, cache_length=207)),
        ):
            with pytest.raises(ValueError, match="checkpoint input/layout"):
                forward(target, fixed_reconstruction=invalid, check_isolation=True)
        with pytest.raises(ValueError, match="target/Prompt"):
            forward(replace(target, head_id=0), fixed_reconstruction=frozen)
        with pytest.raises(ValueError, match="clone-only"):
            forward(target, clone_only=True)
    for (layer, k, v), (kc, vc) in zip(history, original):
        assert layer.keys is k and layer.values is v
        assert torch.equal(k, kc) and torch.equal(v, vc)
    assert "headkv_single_head" not in ALL_ATTENTION_FUNCTIONS
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


@pytest.mark.parametrize("failure", ["partial_forward", "source_mutation", "upstream_input", "reconstruction_drift"])
def test_d1_detects_violations_and_restores_cache_backend_and_hooks(failure, monkeypatch):
    from experiments.head_analysis.runtime import head_intervention as intervention

    model, prompt = make_model(), prompt_ids()
    with torch.inference_mode():
        output = model(torch.tensor([prompt]), use_cache=True, logits_to_keep=1)
    cache = output.past_key_values
    history = snapshot(cache)
    original = [(k.clone(), v.clone()) for _, k, v in history]
    args = dict(
        cache=cache,
        device="cpu",
        prompt_length=len(prompt),
        decode_step=0,
        input_token=7,
        expected_token=8,
        target=HeadIntervention(1, 1, "key", None),
        block_rows=32,
    )
    with monkeypatch.context() as patch:
        if failure in ("source_mutation", "reconstruction_drift"):
            encode = intervention.quantize_and_restore
            four_bit_calls = []

            def corrupt(x, *args):
                if failure == "source_mutation":
                    x[0, 0] += 1
                result = encode(x, *args)
                if args[1] == 4:
                    four_bit_calls.append(1)
                    if failure == "reconstruction_drift" and len(four_bit_calls) == 2:
                        result[0][0, 0] += 0.25
                return result

            patch.setattr(intervention, "quantize_and_restore", corrupt)
            message = "source_middle_unchanged" if failure == "source_mutation" else "reconstruction_matches_first_Q"
        else:
            projection = model.model.layers[1 if failure == "partial_forward" else 0].self_attn.o_proj
            normal = projection.forward

            def corrupt(*args, **kwargs):
                result = normal(*args, **kwargs)
                if "headkv_single_head" in ALL_ATTENTION_FUNCTIONS:
                    if failure == "partial_forward":
                        raise RuntimeError("deliberate partial Forward")
                    return result + 0.25
                return result

            patch.setattr(projection, "forward", corrupt)
            message = "deliberate partial Forward" if failure == "partial_forward" else "checkpoint inputs differ"
        with pytest.raises(RuntimeError, match=message):
            run_d1_checkpoint(model, **args)
    for (layer, k, v), (kc, vc) in zip(history, original):
        assert layer.keys is k and layer.values is v
        assert torch.equal(k, kc) and torch.equal(v, vc)
    assert "headkv_single_head" not in ALL_ATTENTION_FUNCTIONS
    assert all(layer.self_attn.config is model.config for layer in model.model.layers)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    recovered = run_d1_checkpoint(model, **args)
    assert all(c["logits_equal"] for c in recovered.comparisons.values())


@pytest.mark.parametrize("trigger_original_order", [False, True])
def test_d1_sample_output_and_conditional_original_order(tmp_path, trigger_original_order):
    from experiments.head_analysis.runners.quantization_diagnostics import collect_d1_sample
    from experiments.head_analysis.common.io import write_npz

    model = make_model(torch.bfloat16)
    sample = tiny_sample(model)
    source = tmp_path / "original"
    runner.collect_quantization_sample(
        model, device="cpu", sample=sample, directory=source, model_layers=(0, 1), local_steps=(0,), block_rows=32
    )
    if trigger_original_order:
        with np.load(source / "model_branches.npz") as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        mask = (arrays["layer_id"] == 1) & (arrays["head_id"] == 1) & (arrays["side_id"] == 0) & (arrays["bits"] == 8)
        arrays["prediction_kl"][mask] += 0.001  # Test fixture simulates an old value that does not reproduce.
        write_npz(source / "model_branches.npz", arrays, overwrite=True)
    original_files = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / "d1"
    result = collect_d1_sample(
        model,
        device="cpu",
        sample=sample,
        source_directory=source,
        directory=output,
        targets=(HeadIntervention(1, 1, "key", None), HeadIntervention(0, 0, "value", None)),
        conditional_old_order=True,
        block_rows=32,
    )
    assert result["status"] == "completed" and len(result["completed_cases"]) == 3
    assert (result["old_order_trigger"] is not None) == trigger_original_order
    assert result["resources"]["forward_timing"]["completed_forward_count"] == 38 + (
        18 if trigger_original_order else 0
    )
    assert read_json(output / "replay.json")["generated_ids"] == read_json(output / "replay.json")["predicted_ids"]
    assert all(c["logits_equal"] for c in read_json(output / "observation_comparison.json").values())
    for name in result["completed_cases"]:
        case = read_json(output / (name + ".json"))
        assert case["status"] == "completed" and len(case["records"]) == 12
        with np.load(output / (name + ".npz")) as a:
            assert a["logits"].shape == (12, 48) and a["logits"].dtype == np.float32
            assert tuple(a["labels"]) == D1_SEQUENCE
            assert a["native_dtype"].tolist() == ["torch.bfloat16"] * 12
            for j, row in enumerate(case["records"]):
                errors = prediction_errors(
                    torch.from_numpy(a["logits"][0]), torch.from_numpy(a["logits"][j]), sample.generated_ids[1]
                )
                assert errors["prediction_kl"] == row["prediction_kl"]
    if trigger_original_order:
        assert len(read_json(output / "original_order.json")["records"]) == 17
        with np.load(output / "original_order.npz") as a:
            assert a["logits"].shape == (5, 48)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original_files


def test_d1_sample_retains_partial_outputs_on_isolation_failure(tmp_path, monkeypatch):
    from experiments.head_analysis.runners.quantization_diagnostics import collect_d1_sample
    from experiments.head_analysis.runtime import head_intervention as intervention

    model, source, output = make_model(), tmp_path / "original", tmp_path / "partial"
    sample = tiny_sample(model)
    runner.collect_quantization_sample(
        model, device="cpu", sample=sample, directory=source, model_layers=(0, 1), local_steps=(0,), block_rows=32
    )
    encode = intervention.quantize_and_restore

    def corrupt(x, *args):
        x[0, 0] += 1
        return encode(x, *args)

    with monkeypatch.context() as patch:
        patch.setattr(intervention, "quantize_and_restore", corrupt)
        with pytest.raises(RuntimeError, match="source_middle_unchanged"):
            collect_d1_sample(
                model,
                device="cpu",
                sample=sample,
                source_directory=source,
                directory=output,
                targets=(HeadIntervention(1, 1, "key", None), HeadIntervention(0, 0, "value", None)),
                block_rows=32,
            )
    assert read_json(output / "metadata.json")["status"] == "failed"
    partial = read_json(output / "L01_H01_key_checked.json")
    assert partial["status"] == "failed" and [r["label"] for r in partial["records"]] == ["B0", "N", "C0"]
    with np.load(output / "L01_H01_key_checked.npz") as a:
        assert a["logits"].shape == (3, 48)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    assert "headkv_single_head" not in ALL_ATTENTION_FUNCTIONS


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("side", ["key", "value"])
def test_d2_real_outputs_formulas_propagation_and_d1_equivalence(dtype, side):
    from experiments.head_analysis.collectors.quantization_diagnostics import PROJECTION_LABELS

    model, prompt = make_model(dtype), prompt_ids()
    with torch.inference_mode():
        cache = model(torch.tensor([prompt]), use_cache=True, logits_to_keep=1).past_key_values
    original = [(layer, k, v, k.clone(), v.clone()) for layer, k, v in snapshot(cache)]
    args = dict(
        cache=cache,
        device="cpu",
        prompt_length=len(prompt),
        decode_step=0,
        input_token=7,
        expected_token=8,
        target=HeadIntervention(0, 1, side, None),
        capture_logits=True,
        block_rows=32,
    )
    bare = run_d1_checkpoint(model, **args)
    result = run_d1_checkpoint(model, **args, capture_propagation=True)
    assert bare.propagation is None
    for a, b in zip(bare.logits, result.logits):
        assert torch.equal(a, b)
    arrays = result.propagation.arrays
    assert tuple(arrays["labels"]) == PROJECTION_LABELS
    assert arrays["final_hidden"].shape == (3, 256)
    assert arrays["fp32_projected_logits"].shape == (3, 48)
    assert arrays["actual_attention_output"].shape == (3, 8, 32)
    assert arrays["formula_fp32_output"].shape == arrays["formula_cpu_fp64_output"].shape == (3, 4, 32)
    records = {row["label"]: row["propagation"] for row in result.records}
    expected_stages = (
        "attention_output",
        "o_proj_output",
        "layer_00_output",
        "layer_01_output",
        "final_norm",
        "lm_head_input",
        "native_logits",
    )
    for label in D1_SEQUENCE:
        trace = records[label]
        assert tuple(trace["stages"]) == expected_stages
        assert all(metric["dtype"] == str(dtype) for metric in trace["stages"].values())
        assert trace["attention"]["target_query_head_ids"] == [4, 5, 6, 7]
        assert trace["attention"]["other_query_heads_vs_baseline"]["squared_error"] == 0
    for label in ("B0", "N", "C0", "B1", "C1", "B2"):
        assert records[label]["first_different_stage"] is None
        assert all(v["equal"] and v["squared_error"] == 0 for v in records[label]["stages"].values())
    for bits in (4, 8):
        for label in (f"Q{bits}_repeat", f"D{bits}"):
            assert records[label]["stages"] == records[f"Q{bits}"]["stages"]
    # Independent hooks and a NumPy dense formula check the capture point/head layout.
    raw, manual, local = {}, {}, []
    current_label = None

    def inputs(context):
        if hasattr(context, "layer_id") and context.layer_id != 0:
            return
        raw[current_label] = (
            context.query[0, 4:8, 0].detach().double().numpy().copy(),
            context.key[0, 1].detach().double().numpy().copy(),
            context.value[0, 1].detach().double().numpy().copy(),
            context.scaling,
        )
        if hasattr(context, "layer_id"):
            local.append(QuantizationSensitivityCollector(len(prompt), lambda r: None, 32).collect(context))

    def actual_attention(module, args):
        manual[(current_label, "attention")] = args[0].detach().float().numpy().reshape(8, 32).copy()

    def final_hidden(module, args):
        manual[(current_label, "hidden")] = args[0].detach().float().numpy().flatten().copy()

    handles = [
        model.model.layers[0].self_attn.o_proj.register_forward_pre_hook(actual_attention),
        model.lm_head.register_forward_pre_hook(final_hidden),
    ]
    try:
        with isolated_checkpoint_forward(
            model, cache=cache, device="cpu", prompt_length=len(prompt), decode_step=0, input_token=7, block_rows=32
        ) as forward:
            for current_label in PROJECTION_LABELS:
                if current_label == "B0":
                    forward(on_attention=inputs)
                else:
                    forward(
                        HeadIntervention(0, 1, side, int(current_label[1])),
                        on_inputs=lambda before, after: inputs(after),
                    )
    finally:
        for handle in handles:
            handle.remove()
    for idx, label in enumerate(PROJECTION_LABELS):
        np.testing.assert_array_equal(arrays["actual_attention_output"][idx], manual[(label, "attention")])
        np.testing.assert_array_equal(arrays["final_hidden"][idx], manual[(label, "hidden")])
        q, k, v, scale = raw[label]
        scores = q @ k.T * scale
        probability = np.exp(scores - scores.max(-1, keepdims=True))
        probability /= probability.sum(-1, keepdims=True)
        np.testing.assert_allclose(arrays["formula_cpu_fp64_output"][idx], probability @ v, rtol=1e-12, atol=1e-12)
        if idx:
            metric = records[label]["attention"]["formulas"]["fp32_vs_baseline"]
            position = (("key", "value").index(side), idx - 1, 1)
            assert metric["squared_error"] == pytest.approx(
                local[0].arrays["output_squared_error"][position], rel=1e-12
            )
            assert metric["reference_energy"] == pytest.approx(local[0].arrays["output_energy"][position], rel=1e-12)
    for layer, k, v, kc, vc in original:
        assert layer.keys is k and layer.values is v and torch.equal(k, kc) and torch.equal(v, vc)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_d2_fp32_projection_resolves_native_tie_and_restores_arithmetic(monkeypatch):
    from experiments.head_analysis.collectors import quantization_diagnostics as diagnostics

    head = torch.nn.Linear(2, 2, bias=False).bfloat16()
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 1 / 1024], [1.0, 1 / 512]]))
    hidden = torch.tensor([[1.0, 1.0], [1.0, 0.5], [0.5, 1.0]], dtype=torch.bfloat16)
    original_precision = torch.get_float32_matmul_precision()
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_float32_matmul_precision("high")
        before = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            native = head(hidden).detach().float()
            projected, settings = diagnostics.fp32_final_projection(head, hidden, block_rows=1)
        expected = hidden.float() @ head.weight.detach().float().T
        torch.testing.assert_close(projected, expected, rtol=0, atol=0)
        assert not torch.equal(native, projected)
        assert diagnostics.prediction_boundary(native[0])["top1_tie_count"] == 2
        assert diagnostics.prediction_boundary(native[0])["predicted_token_id"] == 0
        boundary = diagnostics.prediction_boundary(projected[0])
        assert boundary["top1_tie_count"] == 1 and boundary["predicted_token_id"] == 1
        assert boundary["margin"] == 1 / 1024
        assert settings["before"] == settings["after"]
        assert (
            not settings["auxiliary"]["allow_tf32"] and settings["auxiliary"]["float32_matmul_precision"] == "highest"
        )
        assert before == (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)
        normal, calls = diagnostics.F.linear, []

        def fail_second_block(*args):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("deliberate projection failure")
            return normal(*args)

        with monkeypatch.context() as patch:
            patch.setattr(diagnostics.F, "linear", fail_second_block)
            with pytest.raises(RuntimeError, match="deliberate projection failure"):
                diagnostics.fp32_final_projection(head, hidden, block_rows=1)
        assert before == (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)
    finally:
        torch.set_float32_matmul_precision(original_precision)
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
    reference, branch = torch.zeros(2), torch.tensor([math.log(0.25), math.log(0.75)], dtype=torch.float64)
    metric = diagnostics.prediction_errors_fp64(reference, branch, 0)
    assert metric["prediction_kl"] == pytest.approx(0.5 * math.log(4 / 3), abs=1e-15)
    assert metric["delta_nll"] == pytest.approx(math.log(2), abs=1e-15)
    assert not metric["top1_match"]
    assert diagnostics.tensor_error(torch.zeros(2), torch.ones(2))["nmse"] is None


def test_d2_trace_locates_synthetic_post_attention_change(monkeypatch):
    model, prompt = make_model(), prompt_ids()
    with torch.inference_mode():
        cache = model(torch.tensor([prompt]), use_cache=True, logits_to_keep=1).past_key_values
    projection = model.model.layers[0].self_attn.o_proj
    normal = projection.forward

    def changed(*args, **kwargs):
        output = normal(*args, **kwargs)
        return output + 0.25 if "headkv_single_head" in ALL_ATTENTION_FUNCTIONS else output

    monkeypatch.setattr(projection, "forward", changed)
    result = run_d1_checkpoint(
        model,
        cache=cache,
        device="cpu",
        prompt_length=len(prompt),
        decode_step=0,
        input_token=7,
        expected_token=8,
        target=HeadIntervention(0, 1, "key", None),
        capture_propagation=True,
    )
    control = result.records[D1_SEQUENCE.index("N")]["propagation"]
    assert control["stages"]["attention_output"]["equal"]
    assert control["first_different_stage"] == "o_proj_output"
    assert control["stages"]["layer_01_output"]["squared_error"] > 0


def test_d2_failure_removes_only_own_hooks_and_restores_cache(monkeypatch):
    model, prompt = make_model(), prompt_ids()
    with torch.inference_mode():
        cache = model(torch.tensor([prompt]), use_cache=True, logits_to_keep=1).past_key_values
    original = [(layer, k, v, k.clone(), v.clone()) for layer, k, v in snapshot(cache)]
    existing = model.model.norm.register_forward_hook(lambda m, args, out: None)
    normal = model.model.layers[1].forward

    def failed(*args, **kwargs):
        if "headkv_single_head" in ALL_ATTENTION_FUNCTIONS:
            raise RuntimeError("deliberate D2 Forward failure")
        return normal(*args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(model.model.layers[1], "forward", failed)
            with pytest.raises(RuntimeError, match="deliberate D2 Forward failure"):
                run_d1_checkpoint(
                    model,
                    cache=cache,
                    device="cpu",
                    prompt_length=len(prompt),
                    decode_step=0,
                    input_token=7,
                    expected_token=8,
                    target=HeadIntervention(0, 1, "key", None),
                    capture_propagation=True,
                )
        assert list(model.model.norm._forward_hooks) == [existing.id]
        for layer, k, v, kc, vc in original:
            assert layer.keys is k and layer.values is v and torch.equal(k, kc) and torch.equal(v, vc)
        assert "headkv_single_head" not in ALL_ATTENTION_FUNCTIONS
        assert all(layer.self_attn.config is model.config for layer in model.model.layers)
    finally:
        existing.remove()
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_d2_writer_reuses_one_replay_and_preserves_original_results(tmp_path):
    from experiments.head_analysis.runners.quantization_diagnostics import collect_d1_sample

    model = make_model(torch.bfloat16)
    sample = tiny_sample(model)
    source, output = tmp_path / "original", tmp_path / "d2"
    runner.collect_quantization_sample(
        model, device="cpu", sample=sample, directory=source, model_layers=(0, 1), local_steps=(0,), block_rows=32
    )
    original_files = {p.name: p.read_bytes() for p in source.iterdir()}
    state = collect_d1_sample(
        model,
        device="cpu",
        sample=sample,
        source_directory=source,
        directory=output,
        targets=(HeadIntervention(1, 1, "key", None), HeadIntervention(0, 0, "value", None)),
        block_rows=32,
        capture_propagation=True,
    )
    assert state["status"] == "completed" and state["capture_propagation"]
    assert state["resources"]["forward_timing"]["completed_forward_count"] == 38
    for case in state["completed_cases"]:
        data = read_json(output / (case + ".json"))
        if not data["checked"]:
            assert "projection" not in data and all("propagation" not in row for row in data["records"])
            continue
        assert all("propagation" in row for row in data["records"])
        assert len(data["projection"]["records"]) == 3
        assert data["projection"]["settings"]["before"] == data["projection"]["settings"]["after"]
        with np.load(output / (case + "_propagation.npz")) as a:
            assert a["fp32_projected_logits"].shape == (3, 48)
            assert a["hidden_native_dtype"].tolist() == ["torch.bfloat16"] * 3
            expected = torch.from_numpy(a["final_hidden"]) @ model.lm_head.weight.detach().float().T
            np.testing.assert_allclose(a["fp32_projected_logits"], expected.numpy(), rtol=1e-6, atol=1e-7)
    assert len(list(output.glob("*_propagation.npz"))) == 2
    assert all(v["logits_equal"] for v in read_json(output / "observation_comparison.json").values())
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original_files
