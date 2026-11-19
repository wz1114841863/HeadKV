"""Exact packed budgets and whole-model combination replay, independent of real weights."""

import itertools

import numpy as np
import pytest
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from experiments.head_analysis.analysis.quantization_budget import (
    POLICIES,
    fit_budget_plans,
    write_budget_plans,
    read_budget_plans,
    summarize_budget_samples,
)
from experiments.head_analysis.common.io import read_json, write_json, write_npz
from experiments.head_analysis.runners.head_budget_runner import collect_budget_sample, run_collection
from experiments.head_analysis.runners.quantization_runner import isolated_checkpoint_forward
from experiments.head_analysis.runtime.head_intervention import multiple_head_intervention
from experiments.head_analysis.runtime.kv_quantization import (
    HeadBitPlan,
    cache_request_header,
    head_plan_byte_ledger,
    pack_codes,
    unpack_codes,
    quantize_head,
    quantized_prompt_range,
)
from experiments.head_analysis.tests.test_quantization_replay import make_model, tiny_sample


def plans_for_tiny():
    return {
        f"{side}_{method}": HeadBitPlan(
            side,
            {"uniform6": ((6, 6), (6, 6)), "head_aware": ((4, 8), (8, 4)), "shifted": ((8, 4), (4, 8))}[method],
            32,
        )
        for side, method in itertools.product(("key", "value"), ("uniform6", "head_aware", "shifted"))
    }


@pytest.mark.parametrize("side", ["key", "value"])
def test_six_bit_packing_and_quantization_match_independent_references(side):
    values = [0, 1, 63, 32, 7, 19, 55]
    for count in range(1, len(values) + 1):
        codes = torch.tensor(values[:count], dtype=torch.uint8)
        expected = sum(value << (6 * i) for i, value in enumerate(values[:count])).to_bytes(
            (count * 6 + 7) // 8, "little"
        )
        packed = pack_codes(codes, 6)
        assert bytes(packed.tolist()) == expected
        assert torch.equal(unpack_codes(packed, 6, count), codes)
    torch.manual_seed(8)
    x = torch.randn(64, 32).bfloat16()
    encoded = quantize_head(x, side, 6, block_rows=32)
    reference = torch.empty_like(x, dtype=torch.float32)
    for row in range(0, 64, 32 if side == "key" else 1):
        for col in range(0, 32, 1 if side == "key" else 32):
            index = (slice(row, row + 32), col) if side == "key" else (row, slice(col, col + 32))
            group = x[index].float()
            low, high = float(group.min()), float(group.max())
            scale = torch.tensor((high - low) / 63, dtype=torch.float32)
            code = torch.tensor([round(float((v - low) / scale)) for v in group], dtype=torch.float32)
            reference[index] = low + scale * code
    assert torch.equal(encoded.decode(), reference.bfloat16())
    assert torch.equal(encoded.codes, quantize_head(x, side, 6, block_rows=64).codes)
    assert encoded.payload_nbytes == x.numel() * 6 // 8 + x.numel() // 32 * 8


@pytest.mark.parametrize("side", ["key", "value"])
def test_layout_counts_serialized_metadata_stats_guards_and_dense_other_side(side):
    plans = plans_for_tiny()
    source, other = torch.randn(2, 2, 212, 32).bfloat16(), torch.randn(2, 2, 212, 32).bfloat16()
    start, end = quantized_prompt_range(205)
    ledgers = []
    raw = lambda tensor: tensor.contiguous().view(torch.uint8).numpy().tobytes()
    for method in ("uniform6", "head_aware"):
        plan = plans[f"{side}_{method}"]
        descriptor = plan.to_bytes()
        assert HeadBitPlan.from_bytes(descriptor) == plan and len(descriptor) == 64
        streams = [cache_request_header(205, 7)]
        for layer, head in itertools.product(range(2), range(2)):
            x = source[layer, head]
            encoded = quantize_head(x[start:end], side, plan.bits[layer][head])
            streams += [
                raw(other[layer, head]),
                raw(x[:start]),
                raw(encoded.codes),
                raw(encoded.minimum),
                raw(encoded.scale),
                raw(x[end:]),
            ]
        assert all(len(stream) % 64 == 0 for stream in streams)
        ledger = head_plan_byte_ledger(plan, 205, 7)
        assert len(b"".join(streams)) == ledger["request_bytes"]
        assert len(b"".join(streams)) + len(descriptor) == ledger["total_bytes"]
        assert ledger["model_metadata_bytes"] == (24 if method == "uniform6" else 26)
        ledgers.append(ledger)
    assert ledgers[0]["total_bytes"] == ledgers[1]["total_bytes"]
    real = HeadBitPlan(side, tuple((4, 8) * 4 for _ in range(32)), 128)
    assert real.metadata_bytes == 56 and len(real.to_bytes()) == 64


def write_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    samples = [{"sample_id": f"sample_{i}", "condition": f"condition_{i}"} for i in range(6)]
    cfg = {
        "calib_samples": samples,
        "validation_samples": [{"sample_id": "unopened_validation"}],
        "bit_widths": [4, 8],
        "sides": ["key", "value"],
        "group_size": 32,
        "protected_prompt_tokens_each_end": 64,
        "key_representation": "post_rope",
        "dtype": "bfloat16",
    }
    write_json(source / "config.json", cfg)
    write_json(
        source / "run_metadata.json",
        {
            "status": "completed",
            "cohort": "calib",
            "completed_sample_ids": [row["sample_id"] for row in samples],
            "geometry": {"num_layers": 2, "kv_heads": 4, "head_dim": 32, "query_heads": 8},
        },
    )
    costs = np.ones((2, 2, 2, 4))
    costs[0, 0] = [4, 3, 2, 1]
    costs[0, 1] = [3.9, 0, 1.9, 0]
    costs[1, 0] = 2  # All V gains tied: stable ascending Head IDs.
    write_npz(
        source / "calib_profile.npz",
        {
            "local_output_nmse": costs,
            "local_output_nmse_valid_condition_count": np.full(costs.shape, 6),
            "prediction_kl": np.full(costs.shape, np.nan),
        },
    )
    return source, costs


def test_fit_uses_only_calib_freezes_ties_and_serialized_plans(tmp_path):
    source, costs = write_source(tmp_path)
    config, plans = fit_budget_plans(source)
    assert plans["key_head_aware"].bits == ((4, 8, 4, 8),) * 2
    assert plans["value_head_aware"].bits == ((8, 8, 4, 4),) * 2
    assert plans["key_shifted"].bits == ((8, 4, 8, 4),) * 2
    # Exhaust all six feasible assignments: selection minimizes the additive fit objective.
    for chosen in itertools.combinations(range(4), 2):
        expected = sum(costs[0, int(h in chosen), 0, h] for h in range(4))
        actual = sum(costs[0, int(b == 8), 0, h] for h, b in enumerate(plans["key_head_aware"].bits[0]))
        assert actual <= expected
    directory = tmp_path / "allocation"
    write_budget_plans(directory, config, plans)
    loaded_config, loaded = read_budget_plans(directory)
    assert loaded_config == config and loaded == plans
    assert not (source / "validation").exists()
    with pytest.raises(FileExistsError):
        write_budget_plans(directory, config, plans)
    write_npz(
        source / "calib_profile.npz",
        {"local_output_nmse": costs, "local_output_nmse_valid_condition_count": np.full(costs.shape, 5)},
        overwrite=True,
    )
    with pytest.raises(ValueError, match="cover"):
        fit_budget_plans(source)


@pytest.mark.parametrize("side", ["key", "value"])
@pytest.mark.parametrize("method", ["uniform6", "head_aware"])
def test_multi_head_forward_matches_independent_decode_and_restores_cache(side, method, monkeypatch):
    model = make_model(torch.bfloat16)
    prompt = [(i % 47) + 1 for i in range(205)]
    with torch.inference_mode():
        cache = model(torch.tensor([prompt]), use_cache=True, logits_to_keep=1).past_key_values
    original = [(layer, layer.keys, layer.values, layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    plan = plans_for_tiny()[f"{side}_{method}"]
    standard = ALL_ATTENTION_FUNCTIONS["sdpa"]
    start, end = quantized_prompt_range(len(prompt))

    def reference(module, q, k, v, mask, **kwargs):
        source = k if side == "key" else v
        changed = source.clone()
        for head, bits in enumerate(plan.bits[module.layer_idx]):
            changed[0, head, start:end] = quantize_head(source[0, head, start:end], side, bits).decode()
        return standard(module, q, changed if side == "key" else k, changed if side == "value" else v, mask, **kwargs)

    with isolated_checkpoint_forward(
        model, cache=cache, device="cpu", prompt_length=205, decode_step=0, input_token=7
    ) as forward:
        with multiple_head_intervention(model, plan, 205) as state:
            actual, _ = forward()
        with monkeypatch.context() as patch:
            patch.setitem(ALL_ATTENTION_FUNCTIONS, "sdpa", reference)
            expected, _ = forward()
    assert torch.equal(actual, expected)
    assert len(state.measurements) == 4
    ledger = head_plan_byte_ledger(plan, 205, 1)
    assert state.code_bytes == ledger["code_bytes"] and state.statistic_bytes == ledger["statistic_bytes"]
    for layer, k, v, kc, vc in original:
        assert layer.keys is k and layer.values is v and torch.equal(k, kc) and torch.equal(v, vc)
    assert "headkv_head_budget" not in ALL_ATTENTION_FUNCTIONS
    assert all(layer.self_attn.config is model.config for layer in model.model.layers)


def test_combination_writer_uses_one_replay_and_both_precisions(tmp_path):
    model = make_model(torch.bfloat16)
    sample = tiny_sample(model, generated_length=5)
    plans = plans_for_tiny()
    directory = tmp_path / "run"
    root_lengths = []
    handle = model.register_forward_pre_hook(
        lambda m, args, kw: root_lengths.append(kw["input_ids"].shape[1]), with_kwargs=True
    )
    try:
        summary = collect_budget_sample(
            model,
            device="cpu",
            sample=sample,
            directory=directory,
            plans=plans,
            steps=(0, 2),
            condition="tiny",
            block_rows=32,
        )
    finally:
        handle.remove()
    assert root_lengths.count(205) == 1 and len(root_lengths) == 1 + 3 + 14
    assert read_json(directory / "metadata.json")["status"] == "completed"
    replay = read_json(directory / "replay.json")
    assert replay["generated_ids"] == replay["predicted_ids"] == list(sample.generated_ids[:4])
    for step in (0, 2):
        point = read_json(directory / f"step_{step:03d}.json")
        with np.load(directory / f"step_{step:03d}.npz") as a:
            assert a["native_logits"].shape == a["fp32_projected_logits"].shape == (7, 48)
            expected = torch.from_numpy(a["final_hidden"]) @ model.lm_head.weight.detach().float().T
            np.testing.assert_allclose(a["fp32_projected_logits"], expected.numpy(), rtol=1e-6, atol=1e-7)
            assert a["representation_nmse"].shape == (6, 2, 2)
        assert point["projection_settings"]["before"] == point["projection_settings"]["after"]
        for side in ("key", "value"):
            uniform, aware = (point["rows"][f"{side}_{m}"] for m in ("uniform6", "head_aware"))
            assert uniform["bytes"]["total_bytes"] == aware["bytes"]["total_bytes"]
    assert summarize_budget_samples([summary])["input_count"] == 1
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_combination_failure_restores_cache_configs_and_records_failure(tmp_path, monkeypatch):
    model = make_model(torch.bfloat16)
    sample = tiny_sample(model)
    with torch.inference_mode():
        cache = model(
            torch.tensor([sample.frozen_record["input_ids"]]), use_cache=True, logits_to_keep=1
        ).past_key_values
    history = [(layer, layer.keys, layer.values, layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    normal = model.model.layers[1].forward

    def failed(*args, **kwargs):
        if "headkv_head_budget" in ALL_ATTENTION_FUNCTIONS:
            raise RuntimeError("deliberate combined Forward failure")
        return normal(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(model.model.layers[1], "forward", failed)
        with pytest.raises(RuntimeError, match="combined Forward failure"):
            with isolated_checkpoint_forward(
                model, cache=cache, device="cpu", prompt_length=205, decode_step=0, input_token=7
            ) as forward:
                with multiple_head_intervention(model, plans_for_tiny()["key_head_aware"], 205):
                    forward()
        for layer, k, v, kc, vc in history:
            assert layer.keys is k and layer.values is v and torch.equal(k, kc) and torch.equal(v, vc)
        with pytest.raises(RuntimeError, match="combined Forward failure"):
            collect_budget_sample(
                model,
                device="cpu",
                sample=sample,
                directory=tmp_path / "failed",
                plans=plans_for_tiny(),
                steps=(0,),
                condition="tiny",
            )
    assert read_json(tmp_path / "failed/metadata.json")["status"] == "failed"
    assert "headkv_head_budget" not in ALL_ATTENTION_FUNCTIONS
    assert all(layer.self_attn.config is model.config for layer in model.model.layers)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_validation_gate_precedes_model_loading(tmp_path):
    source, _ = write_source(tmp_path)
    config, plans = fit_budget_plans(source)
    write_budget_plans(tmp_path / "allocation", config, plans)
    from types import SimpleNamespace

    args = SimpleNamespace(
        allocation_dir=tmp_path / "allocation",
        output_dir=tmp_path / "new",
        cohort="validation",
        calib_dir=None,
        device="cpu",
    )
    with pytest.raises(ValueError, match="completed combination Calib"):
        run_collection(args)
    assert not args.output_dir.exists()
    args.calib_dir = tmp_path / "incompatible_calib"
    args.calib_dir.mkdir()
    write_json(
        args.calib_dir / "run_metadata.json",
        {"status": "completed", "cohort": "calib", "completed_sample_ids": config["calib_sample_ids"]},
    )
    write_json(args.calib_dir / "config.json", {"plans": "different allocation"})
    with pytest.raises(ValueError, match="reuse the completed Calib"):
        run_collection(args)
    assert not args.output_dir.exists()


def test_summary_weights_inputs_before_conditions_not_decode_count():
    def point(value):
        return {
            policy: {
                precision: {
                    "prediction_kl": value * (0.5 if policy.endswith("head_aware") else 1),
                    "delta_nll": value,
                    "top1_match": True,
                }
                for precision in ("native_fp64", "projection_fp32")
            }
            for policy in POLICIES
        }

    samples = [
        {"sample_id": "a", "condition": "A", "checkpoints": [point(1)]},
        {"sample_id": "b", "condition": "A", "checkpoints": [point(3)] * 3},
        {"sample_id": "c", "condition": "B", "checkpoints": [point(10)]},
    ]
    summary = summarize_budget_samples(samples)
    for side, precision in itertools.product(("key", "value"), ("native_fp64", "projection_fp32")):
        values = summary["condition_equal_means"][side][precision]
        assert values["uniform6_mean_kl"] == 6
        assert values["head_aware_mean_kl"] == values["uniform_minus_aware_kl"] == 3
        assert values["shifted_minus_aware_mean_kl"] == 3


def write_passed_pair_gate(directory, *, source_config=None):
    from experiments.head_analysis.analysis.quantization_budget import LOCAL_GATE_THRESHOLDS

    directory.mkdir()
    source_config = source_config or {
        "calib_samples": [{"sample_id": f"s{i}", "condition": f"c{i}"} for i in range(6)],
        "validation_samples": [{"sample_id": "never_read_validation"}],
    }
    ids = [s["sample_id"] for s in source_config["calib_samples"]]
    plans = {}
    for side in ("key", "value"):
        plans[side + "_uniform6"] = HeadBitPlan(side, ((6, 6), (6, 6)), 32)
        plans[side + "_fixed"] = HeadBitPlan(side, ((4, 8), (6, 6)), 32)
    allocation = {
        "schema_version": "headkv-local-pair-allocation-v1",
        "source_config": source_config,
        "geometry": {"num_layers": 2, "query_heads": 8, "kv_heads": 2, "head_dim": 32},
        "calib_sample_ids": ids,
        "eligible_sides": ["key", "value"],
        "thresholds": LOCAL_GATE_THRESHOLDS,
        "rule": "frozen one-pair local gate",
        "plans": {
            n: {"side": p.side, "bits": [list(r) for r in p.bits], "head_dim": p.head_dim} for n, p in plans.items()
        },
    }
    write_budget_plans(directory / "allocation", allocation, plans)
    write_json(
        directory / "run_metadata.json",
        {
            "status": "completed",
            "cohort": "calib",
            "target_sample_ids": ids,
            "completed_sample_ids": ids,
        },
    )
    write_json(directory / "config.json", {"source_config": source_config})
    write_json(
        directory / "summary.json",
        {"sides": {s: {"gate_decision": "eligible_for_combination_review"} for s in ("key", "value")}},
    )
    return allocation, plans


def test_pair_loader_reads_frozen_descriptors_derives_only_plus_one_and_never_refits(tmp_path, monkeypatch):
    from experiments.head_analysis.analysis import quantization_budget as analysis

    gate = tmp_path / "gate"
    _, original = write_passed_pair_gate(gate)
    before = {p.name: p.read_bytes() for p in (gate / "allocation").iterdir()}
    monkeypatch.setattr(analysis, "choose_local_pairs", lambda *a: pytest.fail("unexpected refit"))
    monkeypatch.setattr(analysis, "fit_budget_plans", lambda *a: pytest.fail("unexpected old-profile fit"))
    config, plans = analysis.read_pair_replay_plans(gate)
    assert config["shift"] == 1
    for side in ("key", "value"):
        assert plans[side + "_fixed"] == original[side + "_fixed"]
        np.testing.assert_array_equal(plans[side + "_shifted"].bits, np.roll(original[side + "_fixed"].bits, 1, axis=1))
    assert before == {p.name: p.read_bytes() for p in (gate / "allocation").iterdir()}
    state = read_json(gate / "run_metadata.json")
    state["status"] = "failed"
    write_json(gate / "run_metadata.json", state, overwrite=True)
    with pytest.raises(ValueError, match="passed frozen"):
        analysis.read_pair_replay_plans(gate)


def test_pair_replay_has_eight_isolated_branches_and_preserves_both_precision_vectors(tmp_path):
    from experiments.head_analysis.analysis.quantization_budget import PAIR_POLICIES, read_pair_replay_plans

    model = make_model(torch.bfloat16)
    sample = tiny_sample(model, generated_length=5)
    gate = tmp_path / "gate"
    write_passed_pair_gate(gate)
    _, plans = read_pair_replay_plans(gate)
    lengths = []
    h = model.register_forward_pre_hook(lambda m, a, kw: lengths.append(kw["input_ids"].shape[1]), with_kwargs=True)
    try:
        result = collect_budget_sample(
            model,
            device="cpu",
            sample=sample,
            directory=tmp_path / "out",
            plans=plans,
            steps=(0, 2),
            condition="tiny",
            block_rows=32,
            policies=PAIR_POLICIES,
            dense_after=True,
        )
    finally:
        h.remove()
    assert lengths.count(205) == 1 and len(lengths) == 1 + 3 + 16
    assert result["steps"] == [0, 2]
    for step in (0, 2):
        point = read_json(tmp_path / "out" / f"step_{step:03d}.json")
        with np.load(tmp_path / "out" / f"step_{step:03d}.npz") as a:
            assert a["labels"].tolist() == ["dense", *PAIR_POLICIES, "dense_after"]
            assert a["native_logits"].shape == a["fp32_projected_logits"].shape == (8, 48)
            assert a["head_bits"].shape == (6, 2, 2)
            for field in ("native_logits", "final_hidden"):
                np.testing.assert_array_equal(a[field][0], a[field][-1])
            x = a["native_logits"].astype(np.float64)
            lp = x - x.max(1, keepdims=True)
            lp -= np.log(np.exp(lp).sum(1, keepdims=True))
            kl = np.sum(np.exp(lp[0]) * (lp[0] - lp), axis=1)
            for i, name in enumerate(a["labels"]):
                assert point["rows"][name]["native_fp64"]["prediction_kl"] == pytest.approx(kl[i], abs=1e-12)
            projected = a["fp32_projected_logits"].astype(np.float64)
            projected_logp = projected - projected.max(1, keepdims=True)
            projected_logp -= np.log(np.exp(projected_logp).sum(1, keepdims=True))
            projected_kl = np.sum(np.exp(projected_logp[0]) * (projected_logp[0] - projected_logp), axis=1)
            # A batched FP32 GEMM can round equal hidden rows differently.
            # The saved after-Dense KL measures that background rather than forcing zero.
            for i, name in enumerate(a["labels"]):
                assert point["rows"][name]["projection_fp32"]["prediction_kl"] == pytest.approx(
                    projected_kl[i], abs=1e-12
                )
            np.testing.assert_allclose(
                a["fp32_projected_logits"],
                (torch.from_numpy(a["final_hidden"]) @ model.lm_head.weight.detach().float().T).numpy(),
                rtol=1e-6,
                atol=1e-7,
            )
        assert point["projection_settings"]["before"] == point["projection_settings"]["after"]
        for side in ("key", "value"):
            assert (
                len({point["rows"][side + "_" + m]["bytes"]["total_bytes"] for m in ("uniform6", "fixed", "shifted")})
                == 1
            )
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def synthetic_pair_samples():
    from experiments.head_analysis.analysis.quantization_budget import PAIR_POLICIES

    samples = []
    for i in range(6):
        steps = [5] if i < 2 else [31, 255]
        points = []
        for j, step in enumerate(steps):
            scale = 1 if j == 0 else 3
            point = {}
            for name in ("dense", *PAIR_POLICIES, "dense_after"):
                v = (
                    0
                    if name.startswith("dense")
                    else scale * {"uniform6": 1e-3, "fixed": 5e-4, "shifted": 8e-4}[name.split("_", 1)[1]]
                )
                point[name] = {
                    p: {"prediction_kl": v, "delta_nll": v / 2, "top1_match": True}
                    for p in ("native_fp64", "projection_fp32")
                }
            points.append(point)
        samples.append({"sample_id": f"s{i}", "condition": f"c{i}", "steps": steps, "checkpoints": points})
    return samples


@pytest.mark.parametrize("case", ["pass", "background", "top1", "native", "tail", "wins", "zero"])
def test_combination_gate_applies_frozen_rules_and_keeps_primary_late_scopes(case):
    from experiments.head_analysis.analysis.quantization_budget import summarize_pair_budget_samples

    samples = synthetic_pair_samples()
    for i, sample in enumerate(samples):
        for point in sample["checkpoints"]:
            if case == "background":
                for p in ("native_fp64", "projection_fp32"):
                    point["dense_after"][p]["prediction_kl"] = 1e-3
            if case == "zero":
                for row in point.values():
                    for metric in row.values():
                        metric["prediction_kl"] = 0
            if case == "wins" and i < 3:
                for p in ("native_fp64", "projection_fp32"):
                    point["key_fixed"][p]["prediction_kl"] = point["key_uniform6"][p]["prediction_kl"]
    first = samples[0]["checkpoints"][0]
    if case == "top1":
        first["key_fixed"]["projection_fp32"]["top1_match"] = False
    if case == "native":
        for sample in samples:
            for point in sample["checkpoints"]:
                point["key_fixed"]["native_fp64"]["prediction_kl"] = (
                    2 * point["key_uniform6"]["native_fp64"]["prediction_kl"]
                )
    if case == "tail":
        for p in ("native_fp64", "projection_fp32"):
            first["key_fixed"][p]["prediction_kl"] = 1.2 * first["key_uniform6"][p]["prediction_kl"]
    report = summarize_pair_budget_samples(samples)
    decision = report["gate_decisions"]["key"]
    assert (decision["gate_decision"] == "eligible_for_validation_review") == (case == "pass")
    if case in ("top1", "native", "tail", "wins"):
        assert report["gate_decisions"]["value"]["gate_decision"] == "eligible_for_validation_review"
    if case == "pass":
        assert decision["improved_input_count"] == 6
        assert report["condition_equal_means"]["key"]["projection_fp32"]["uniform6_mean_kl"] == pytest.approx(5e-3 / 3)
        assert report["primary_only"]["condition_equal_means"]["key"]["projection_fp32"][
            "uniform6_mean_kl"
        ] == pytest.approx(1e-3)
        assert report["late_only"]["condition_equal_means"]["key"]["projection_fp32"][
            "uniform6_mean_kl"
        ] == pytest.approx(3e-3)
    if case == "zero":
        assert decision["fixed_improvement_vs_uniform"] is None
    if case == "background":
        assert report["dense_background"]["projection_fp32"]["absolute_threshold"] == 5e-3
    assert report["validation_started"] is False


def test_pair_cli_reads_only_calib_keeps_source_allocation_and_never_chains_validation(tmp_path, monkeypatch):
    from dataclasses import replace
    from experiments.head_analysis.runners import head_budget_runner as runner

    model = make_model(torch.bfloat16)
    sample = tiny_sample(model, generated_length=5)
    ids = [f"s{i}" for i in range(6)]
    samples = [replace(sample, frozen_record={**sample.frozen_record, "sample_id": sid}) for sid in ids]
    cfg = {
        "calib_samples": [{"sample_id": sid, "condition": f"c{i}"} for i, sid in enumerate(ids)],
        "validation_samples": [{"sample_id": "never_read_validation"}],
        "bundle": "unused",
        "run_set": "unused",
        "source_root": ".",
        "model": "tiny",
        "dtype": "bfloat16",
        "local_steps": {"govreport": [0, 2]},
        "block_rows": 32,
    }
    gate = tmp_path / "gate"
    write_passed_pair_gate(gate, source_config=cfg)
    original = {p.name: p.read_bytes() for p in (gate / "allocation").iterdir()}
    monkeypatch.setattr(runner, "_frozen_cohorts", lambda *a: None)
    monkeypatch.setattr(runner, "load_b1_replay_index", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_validate_execution_contract", lambda *a: None)
    monkeypatch.setattr(
        runner,
        "load_replay_samples",
        lambda index, chosen: samples if chosen == ids else pytest.fail("Validation read"),
    )
    monkeypatch.setattr(runner, "load_model", lambda *a: (model, torch.device("cpu")))
    monkeypatch.setattr(runner, "fit_budget_plans", lambda *a: pytest.fail("unexpected refit"))
    output = tmp_path / "out"
    assert (
        runner.main(["replay-pairs", "--local-gate-dir", str(gate), "--output-dir", str(output), "--device", "cpu"])
        == 0
    )
    assert read_json(output / "run_metadata.json")["completed_sample_ids"] == ids
    assert not read_json(output / "run_metadata.json")["validation_started"]
    assert not (output / "validation").exists()
    assert read_json(output / "summary.json")["input_count"] == 6
    assert original == {p.name: p.read_bytes() for p in (gate / "allocation").iterdir()}
    for side in ("key", "value"):
        assert (output / "allocation" / f"{side}_fixed.bin").read_bytes() == original[f"{side}_fixed.bin"]
    state = read_json(gate / "run_metadata.json")
    state["status"] = "failed"
    write_json(gate / "run_metadata.json", state, overwrite=True)
    monkeypatch.setattr(runner, "load_model", lambda *a: pytest.fail("loaded before gate check"))
    with pytest.raises(ValueError, match="passed frozen"):
        runner.main(
            ["replay-pairs", "--local-gate-dir", str(gate), "--output-dir", str(tmp_path / "bad"), "--device", "cpu"]
        )
    assert not (tmp_path / "bad").exists()
