"""First-gate numerical opportunity, real descriptor budgets and read-only replay."""

from dataclasses import replace
import itertools
import struct

import numpy as np
import pytest
import torch

from experiments.head_analysis.analysis.quantization_budget import (
    analyze_local_pairs,
    choose_local_pairs,
    fit_budget_plans,
    local_plan_cost,
    write_budget_plans,
)
from experiments.head_analysis.collectors.quantization_sensitivity import QuantizationSensitivityCollector
from experiments.head_analysis.common.io import read_json, write_json, write_npz
from experiments.head_analysis.runners import head_budget_runner as runner
from experiments.head_analysis.runners.quantization_runner import replay_quantization_checkpoints
from experiments.head_analysis.runtime.kv_quantization import (
    HeadBitPlan,
    head_plan_byte_ledger,
    quantize_head,
    cache_request_header,
)
from experiments.head_analysis.tests.test_quantization_replay import make_model, tiny_sample


def old_plans(layers=1, heads=8):
    return {
        f"{side}_head_aware": HeadBitPlan(side, tuple((4, 8) * (heads // 2) for _ in range(layers)), 32)
        for side in ("key", "value")
    }


def test_pair_search_matches_exhaustive_objective_and_preserves_ties_and_nonmonotonicity():
    rng = np.random.default_rng(12)
    cost = rng.uniform(0.01, 2, (3, 2, 8))
    bits = choose_local_pairs(cost)
    for layer in range(2):
        candidates = [np.full((1, 8), 6, np.uint8)]
        for d, u in itertools.permutations(range(8), 2):
            b = candidates[0].copy()
            b[0, d] = 4
            b[0, u] = 8
            candidates.append(b)
        actual = local_plan_cost(cost[:, layer : layer + 1], bits[layer : layer + 1])
        assert actual == min(local_plan_cost(cost[:, layer : layer + 1], b) for b in candidates)
    assert np.all(choose_local_pairs(np.ones((3, 1, 8))) == 6)
    tied = np.ones((3, 1, 8))
    tied[2] = 0
    assert choose_local_pairs(tied).tolist() == [[4, 8, 6, 6, 6, 6, 6, 6]]
    c = np.ones((3, 1, 8))
    c[0] = 4
    c[0, 0, 0] = 0.5
    c[2] = 2
    assert np.all(choose_local_pairs(c) == 6)


@pytest.mark.parametrize("side", ["key", "value"])
def test_paired_descriptors_cover_every_pair_keep_legacy_bytes_and_count_actual_buffers(side):
    for d, u in itertools.permutations(range(8), 2):
        b = [6] * 8
        b[d] = 4
        b[u] = 8
        p = HeadBitPlan(side, (tuple(b), (6,) * 8), 32)
        assert len(p.to_bytes()) == 64 and p.metadata_bytes == 26
        assert HeadBitPlan.from_bytes(p.to_bytes()) == p
    legacy = HeadBitPlan(side, ((4, 8) * 4,) * 32, 128)
    header = struct.pack("<4s5I", b"HQB1", 32, 8, 128, 32, int(side == "value"))
    assert legacy.to_bytes() == header + bytes([0b10101010]) * 32 + bytes(8)
    plan = HeadBitPlan(side, ((4, 8, 6, 6, 6, 6, 6, 6),) * 32, 128)
    uniform = HeadBitPlan(side, ((6,) * 8,) * 32, 128)
    assert len(plan.to_bytes()) == 64 and plan.metadata_bytes == 56
    assert (
        head_plan_byte_ledger(plan, 8192, 32)["total_bytes"] == head_plan_byte_ledger(uniform, 8192, 32)["total_bytes"]
    )
    small = HeadBitPlan(side, ((6, 6, 4, 6, 6, 8, 6, 6),), 32)
    x = torch.randn(8, 205, 32).bfloat16()
    start, end = 64, 128
    raw = lambda x: x.contiguous().view(torch.uint8).numpy().tobytes()
    streams = [cache_request_header(205, 0)]
    for h, b in enumerate(small.bits[0]):
        q = quantize_head(x[h, start:end], side, b, 32)
        streams.extend((raw(x[h]), raw(x[h, :start]), raw(q.codes), raw(q.minimum), raw(q.scale), raw(x[h, end:])))
    ledger = head_plan_byte_ledger(small, 205, 0)
    assert len(b"".join(streams)) + len(small.to_bytes()) == ledger["total_bytes"]
    with pytest.raises(ValueError, match="one 4-bit"):
        HeadBitPlan(side, ((4, 4, 8, 8, 6, 6, 6, 6),), 32)


@pytest.mark.parametrize(
    "scenario,decision",
    [
        ("none", "stop_insufficient_local_opportunity"),
        ("fixed", "eligible_for_combination_review"),
        ("rotating", "stop_fixed_profile_not_stable"),
        ("zero", "stop_undefined_uniform_reference"),
    ],
)
def test_local_gate_distinguishes_no_space_fixed_success_and_input_specific_space(scenario, decision):
    c = np.ones((6, 2, 3, 1, 8))
    c[:, :, 0] = 20
    c[:, :, 2] = 0.9
    for i in range(6):
        if scenario in ("fixed", "rotating"):
            d = 0 if scenario == "fixed" else i
            c[i, :, 0, 0, d] = 1.01
            c[i, :, 2, 0, (d + 1) % 8] = 0.1
    if scenario == "zero":
        c[:] = 0
    result, arrays, plans = analyze_local_pairs(
        c, [f"s{i}" for i in range(6)], [f"c{i}" for i in range(6)], old_plans()
    )
    assert all(v["gate_decision"] == decision for v in result["sides"].values())
    assert not result["next_stage_started"]
    for i in range(6):
        for si in range(2):
            assert arrays["J_oracle"][i, si] <= arrays["J_fixed"][i, si] + 1e-12
            assert arrays["J_oracle"][i, si] <= arrays["J_uniform6"][i, si] + 1e-12
    if scenario == "fixed":
        assert result["sides"]["key"]["improved_input_count"] == 6
        assert result["sides"]["key"]["fixed_improvement"] == pytest.approx(0.89 / 8)
    for p in plans.values():
        assert HeadBitPlan.from_bytes(p.to_bytes()) == p


def source_for_sample(model, sample, directory):
    directory.mkdir(parents=True)
    rows = []
    collector = QuantizationSensitivityCollector(len(sample.frozen_record["input_ids"]), rows.append, 32)
    replay_quantization_checkpoints(
        model,
        device="cpu",
        sample_id=sample.frozen_record["sample_id"],
        prompt_ids=sample.frozen_record["input_ids"],
        replay_generated_ids=sample.generated_ids[:2],
        checkpoints={0: ()},
        on_attention=collector.on_attention,
        block_rows=32,
    )
    rows.sort(key=lambda r: r.layer_id)
    arrays = {
        k: np.stack([r.arrays[k] for r in rows], axis=0 if k.startswith("formula_sdpa_") else 2) for k in rows[0].arrays
    }
    write_npz(directory / "local_step_000.npz", arrays)
    write_json(
        directory / "metadata.json",
        {
            "status": "completed",
            "sample_id": sample.frozen_record["sample_id"],
            "primary_step": 0,
            "prompt_length": len(sample.frozen_record["input_ids"]),
            "local_files": {"0": "local_step_000.npz"},
        },
    )
    write_json(directory / "replay.json", {"generated_ids": list(sample.generated_ids[:2])})
    return arrays


def test_local6_replay_joins_real_old_statistics_and_rejects_reference_drift(tmp_path):
    model = make_model(torch.bfloat16)
    sample = tiny_sample(model)
    source = tmp_path / "source"
    old = source_for_sample(model, sample, source)
    original = (source / "local_step_000.npz").read_bytes()
    lengths = []
    h = model.register_forward_pre_hook(lambda m, a, kw: lengths.append(kw["input_ids"].shape[1]), with_kwargs=True)
    try:
        costs = runner.collect_local6_sample(
            model,
            device="cpu",
            sample=sample,
            source_directory=source,
            directory=tmp_path / "local6",
            primary_step=0,
            block_rows=32,
        )
    finally:
        h.remove()
    assert lengths == [205, 1, 1]
    assert costs.shape == (2, 3, 2, 2)
    np.testing.assert_array_equal(costs[:, 0], old["output_nmse"][:, 0])
    np.testing.assert_array_equal(costs[:, 2], old["output_nmse"][:, 1])
    assert (source / "local_step_000.npz").read_bytes() == original
    assert read_json(tmp_path / "local6/metadata.json")["status"] == "completed"
    assert read_json(tmp_path / "local6/summary.json")["all_H0_anchors_equal"]
    with np.load(tmp_path / "local6/local6.npz") as z:
        assert z["output_nmse"].shape == (2, 1, 2, 2)
        np.testing.assert_array_equal(z["output_nmse"], z["output_squared_error"] / z["output_energy"])
        assert np.all(z["attention_kl"][1] == 0)
    old["output_energy"] *= 2
    write_npz(source / "local_step_000.npz", old, overwrite=True)
    with pytest.raises(AssertionError, match="reference output_energy"):
        runner.collect_local6_sample(
            model,
            device="cpu",
            sample=sample,
            source_directory=source,
            directory=tmp_path / "failed",
            primary_step=0,
            block_rows=32,
        )
    assert read_json(tmp_path / "failed/metadata.json")["status"] == "failed"
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_local_diagnostic_cli_finishes_analysis_without_validation_or_model_branches(tmp_path, monkeypatch):
    model = make_model(torch.bfloat16)
    sample = tiny_sample(model)
    ids = [f"sample_{i}" for i in range(6)]
    samples = [replace(sample, frozen_record={**sample.frozen_record, "sample_id": sid}) for sid in ids]
    source = tmp_path / "source"
    arrays = source_for_sample(model, samples[0], source / "samples" / ids[0])
    for s in samples[1:]:
        p = source / "samples" / s.frozen_record["sample_id"]
        p.mkdir()
        write_npz(p / "local_step_000.npz", arrays)
        meta = read_json(source / "samples" / ids[0] / "metadata.json")
        meta["sample_id"] = s.frozen_record["sample_id"]
        write_json(p / "metadata.json", meta)
        write_json(p / "replay.json", {"generated_ids": list(sample.generated_ids[:2])})
    cfg = {
        "calib_samples": [{"sample_id": sid, "condition": f"c{i}"} for i, sid in enumerate(ids)],
        "validation_samples": [],
        "bit_widths": [4, 8],
        "sides": ["key", "value"],
        "group_size": 32,
        "protected_prompt_tokens_each_end": 64,
        "key_representation": "post_rope",
        "dtype": "bfloat16",
        "bundle": "unused",
        "run_set": "unused",
        "source_root": ".",
        "model": "tiny",
        "local_steps": {"govreport": [0, 2]},
        "block_rows": 32,
    }
    write_json(source / "config.json", cfg)
    write_json(
        source / "run_metadata.json",
        {
            "status": "completed",
            "cohort": "calib",
            "completed_sample_ids": ids,
            "geometry": {"num_layers": 2, "query_heads": 8, "kv_heads": 2, "head_dim": 32},
        },
    )
    write_npz(
        source / "calib_profile.npz",
        {
            "local_output_nmse": arrays["output_nmse"],
            "local_output_nmse_valid_condition_count": np.full((2, 2, 2, 2), 6, np.uint8),
        },
    )
    allocation, plans = fit_budget_plans(source)
    write_budget_plans(tmp_path / "allocation", allocation, plans)
    monkeypatch.setattr(runner, "_frozen_cohorts", lambda *a: None)
    monkeypatch.setattr(runner, "load_b1_replay_index", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_validate_execution_contract", lambda *a: None)
    monkeypatch.setattr(
        runner,
        "load_replay_samples",
        lambda index, chosen: samples if chosen == ids else pytest.fail("non-Calib reads"),
    )
    monkeypatch.setattr(runner, "load_model", lambda *a: (model, torch.device("cpu")))
    output = tmp_path / "out"
    assert (
        runner.main(
            [
                "diagnose-local",
                "--allocation-dir",
                str(tmp_path / "allocation"),
                "--source-calib-dir",
                str(source),
                "--output-dir",
                str(output),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    meta = read_json(output / "run_metadata.json")
    assert meta["status"] == "completed" and meta["completed_sample_ids"] == ids and not meta["next_stage_started"]
    assert not (output / "validation").exists() and not list(output.rglob("model_branches.npz"))
    with np.load(output / "analysis.npz") as z:
        assert z["output_nmse"].shape == (6, 2, 3, 2, 2)
    frozen = read_json(output / "allocation/allocation.json")
    assert frozen["calib_sample_ids"] == ids
    for name, p in frozen["plans"].items():
        descriptor = HeadBitPlan.from_bytes((output / "allocation" / f"{name}.bin").read_bytes())
        assert np.array_equal(descriptor.bits, p["bits"])
