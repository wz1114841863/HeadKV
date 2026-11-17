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
