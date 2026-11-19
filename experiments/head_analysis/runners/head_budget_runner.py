"""Same-byte uniform6 versus Calib-fitted 4/8-bit Head combination replay."""

import argparse
from contextlib import contextmanager
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from experiments.head_analysis.analysis.quantization_budget import (
    POLICIES,
    fit_budget_plans,
    write_budget_plans,
    read_budget_plans,
    summarize_budget_samples,
    analyze_local_pairs,
    LOCAL_GATE_THRESHOLDS,
    PAIR_POLICIES,
    COMBINATION_GATE_THRESHOLDS,
    read_pair_replay_plans,
    summarize_pair_budget_samples,
)
from experiments.head_analysis.collectors.quantization_sensitivity import QuantizationSensitivityCollector
from experiments.head_analysis.collectors.quantization_diagnostics import (
    fp32_final_projection,
    prediction_errors_fp64,
    prediction_boundary,
)
from experiments.head_analysis.common.io import read_json, write_json, write_npz
from experiments.head_analysis.runners.b2_runner import (
    teacher_forced_replay,
    load_b1_replay_index,
    load_replay_samples,
    _validate_execution_contract,
    emit_progress,
)
from experiments.head_analysis.runners.quantization_runner import (
    isolated_checkpoint_forward,
    prediction_errors,
    _frozen_cohorts,
    replay_quantization_checkpoints,
)
from experiments.head_analysis.runners.resource_metrics import GenerationPhaseTimer
from experiments.head_analysis.runtime.head_intervention import multiple_head_intervention
from experiments.head_analysis.runtime.kv_quantization import head_plan_byte_ledger, quantization_byte_ledger
from experiments.head_analysis.runtime.model import load_model, model_dimensions


@contextmanager
def _final_hidden(model):
    captured = []

    def observe(module, args):
        if args[0].shape[:2] != (1, 1) or captured:
            raise ValueError("budget branches require exactly one single-token lm_head call")
        captured.append(args[0].detach().to(device="cpu", dtype=torch.float32, copy=True).flatten())

    handle = model.lm_head.register_forward_pre_hook(observe)
    try:
        yield captured
    finally:
        handle.remove()


def run_budget_checkpoint(
    model,
    *,
    cache,
    device,
    prompt_length,
    decode_step,
    input_token,
    expected_token,
    plans,
    block_rows=1024,
    policies=POLICIES,
    dense_after=False,
):
    ledgers = {name: head_plan_byte_ledger(plans[name], prompt_length, decode_step + 1) for name in policies}
    for side in ("key", "value"):
        a = ledgers[f"{side}_uniform6"]
        for name in (n for n in policies if n.startswith(side + "_") and n != f"{side}_uniform6"):
            b = ledgers[name]
            for field in ("code_bytes", "statistic_bytes", "request_bytes", "model_bytes", "total_bytes"):
                if a[field] != b[field]:
                    raise ValueError(f"unequal encoded budget before Forward: {side}/{field}")
    names = ("dense", *policies, *(("dense_after",) if dense_after else ()))
    rows, native, hidden, reconstruction = {}, [], [], []
    with isolated_checkpoint_forward(
        model,
        cache=cache,
        device=device,
        prompt_length=prompt_length,
        decode_step=decode_step,
        input_token=input_token,
        block_rows=block_rows,
    ) as forward:
        for name in names:
            with _final_hidden(model) as h:
                if name in ("dense", "dense_after"):
                    logits, _ = forward()
                    if name == "dense":
                        reference = logits
                        if int(reference.argmax()) != expected_token:
                            raise ValueError("Dense checkpoint differs from the frozen B1 prediction")
                    state = None
                else:
                    with multiple_head_intervention(model, plans[name], prompt_length, block_rows) as state:
                        logits, _ = forward()
            if len(h) != 1:
                raise RuntimeError("missing final hidden vector")
            hidden.append(h[0])
            native.append(logits.detach().cpu().float().clone())
            rows[name] = {
                "native_fp32": prediction_errors(reference, logits, expected_token),
                "native_fp64": prediction_errors_fp64(native[0], native[-1], expected_token),
                "native_boundary": prediction_boundary(native[-1]),
            }
            if name in ("dense", "dense_after"):
                plan = plans[policies[0]]
                rows[name]["bytes"] = quantization_byte_ledger(
                    prompt_length, decode_step + 1, len(plan.bits), len(plan.bits[0]), plan.head_dim, None
                )
            if state is not None:
                for field in (
                    "code_bytes",
                    "statistic_bytes",
                    "protected_residual_decode_bytes",
                    "dense_other_side_bytes",
                ):
                    if getattr(state, field) != ledgers[name][field]:
                        raise ValueError(f"actual encoding bytes differ from the frozen layout: {name}/{field}")
                rows[name]["bytes"] = ledgers[name]
                shape = (len(plans[name].bits), len(plans[name].bits[0]))
                error, energy = np.empty(shape), np.empty(shape)
                for (layer, head), measurement in state.measurements.items():
                    error[layer, head], energy[layer, head] = (
                        measurement.squared_error,
                        measurement.original_squared_sum,
                    )
                reconstruction.append((error, energy))
    hidden = torch.stack(hidden)
    projected, settings = fp32_final_projection(model.lm_head, hidden)
    for i, name in enumerate(names):
        rows[name]["projection_fp32"] = prediction_errors_fp64(projected[0], projected[i], expected_token)
        rows[name]["projection_boundary"] = prediction_boundary(projected[i])
        rows[name]["native_top1_matches_projection"] = int(native[i].argmax()) == int(projected[i].argmax())
    errors = np.stack([r[0] for r in reconstruction])
    energies = np.stack([r[1] for r in reconstruction])
    nmse = np.divide(errors, energies, out=np.full_like(errors, np.nan), where=energies > 0)
    arrays = {
        "labels": np.asarray(names),
        "native_logits": torch.stack(native).numpy(),
        "final_hidden": hidden.numpy(),
        "fp32_projected_logits": projected.numpy(),
        "head_bits": np.asarray([plans[name].bits for name in policies], dtype=np.uint8),
        "representation_squared_error": errors,
        "representation_energy": energies,
        "representation_nmse": nmse,
        "representation_valid": energies > 0,
    }
    return {"rows": rows, "projection_settings": settings, "decode_step": decode_step}, arrays


def collect_budget_sample(
    model,
    *,
    device,
    sample,
    directory,
    plans,
    steps,
    condition,
    block_rows=1024,
    policies=POLICIES,
    dense_after=False,
):
    directory = Path(directory)
    if not steps or list(steps) != sorted(set(steps)) or len(sample.generated_ids) < steps[-1] + 2:
        raise ValueError("budget checkpoints require ordered steps and consumed/next B1 IDs")
    directory.mkdir(parents=True, exist_ok=False)
    sid = sample.frozen_record["sample_id"]
    length = len(sample.frozen_record["input_ids"])
    state = {
        "status": "running",
        "sample_id": sid,
        "condition": condition,
        "steps": list(steps),
        "completed_steps": [],
        "prompt_length": length,
    }
    write_json(directory / "metadata.json", state)
    points = []

    def before_decode(step, cache):
        if step not in steps:
            return
        emit_progress("head_budget_checkpoint_started", sample_id=sid, decode_step=step)
        point, arrays = run_budget_checkpoint(
            model,
            cache=cache,
            device=device,
            prompt_length=length,
            decode_step=step,
            input_token=sample.generated_ids[step],
            expected_token=sample.generated_ids[step + 1],
            plans=plans,
            block_rows=block_rows,
            policies=policies,
            dense_after=dense_after,
        )
        write_npz(directory / f"step_{step:03d}.npz", arrays)
        write_json(directory / f"step_{step:03d}.json", point)
        points.append(point["rows"])
        state["completed_steps"].append(step)
        write_json(directory / "metadata.json", state, overwrite=True)
        emit_progress("head_budget_checkpoint_completed", sample_id=sid, decode_step=step)

    started = time.perf_counter()
    device = torch.device(device)
    timer = GenerationPhaseTimer(model, device)
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        with timer:
            replay = teacher_forced_replay(
                model,
                device=device,
                sample_id=sid,
                prompt_ids=sample.frozen_record["input_ids"],
                replay_generated_ids=sample.generated_ids[: steps[-1] + 2],
                before_decode=before_decode,
            )
        timing = timer.summary()
        if state["completed_steps"] != list(steps) or timing["completed_forward_count"] != 1 + steps[-1] + 1 + (
            1 + len(policies) + int(dense_after)
        ) * len(steps):
            raise ValueError("Head budget replay is incomplete")
        resources = {
            "wall_seconds": time.perf_counter() - started,
            "forward_timing": timing,
            "scope": "Dense shadow replay with actual coding and auxiliary FP32 projection; not compressed GPU Cache/latency",
        }
        if device.type == "cuda":
            resources.update(
                peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        write_json(
            directory / "replay.json",
            {
                "generated_ids": list(sample.generated_ids[: steps[-1] + 2]),
                "predicted_ids": list(replay.all_predicted_token_ids),
            },
        )
        summary = {"sample_id": sid, "condition": condition, "checkpoints": points, "resources": resources}
        if dense_after:
            summary["steps"] = list(steps)
        write_json(directory / "summary.json", summary)
        state.update(status="completed", baseline_top1_match=True)
        write_json(directory / "metadata.json", state, overwrite=True)
        return summary
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(directory / "metadata.json", state, overwrite=True)
        raise


def run_pair_collection(args):
    """Calib only: reuse frozen gate-one F, derive +1 once, and never refit."""
    allocation, plans = read_pair_replay_plans(args.local_gate_dir)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    cfg, ids = allocation["source_config"], allocation["calib_sample_ids"]
    _frozen_cohorts(cfg["bundle"])
    index = load_b1_replay_index(
        cfg["run_set"], source_root=cfg["source_root"], expected_run_set_sha256=None, expected_data_sha256=None
    )
    _validate_execution_contract(index, SimpleNamespace(**cfg))
    samples = load_replay_samples(index, ids)
    if [s.frozen_record["sample_id"] for s in samples] != ids:
        raise ValueError("pair replay order differs from frozen Calib")
    # Check every actual point's full layout before loading the model.
    for sample in samples:
        length = len(sample.frozen_record["input_ids"])
        for step in cfg["local_steps"][sample.frozen_record["dataset"]]:
            for side in ("key", "value"):
                ledgers = [
                    head_plan_byte_ledger(plans[f"{side}_{m}"], length, step + 1)
                    for m in ("uniform6", "fixed", "shifted")
                ]
                for field in ("code_bytes", "statistic_bytes", "request_bytes", "model_bytes", "total_bytes"):
                    if len({row[field] for row in ledgers}) != 1:
                        raise ValueError(f"pair replay requires equal actual budgets: {side}/{field}")
    model, device = load_model(cfg["model"], args.device, cfg["dtype"])
    layers, queries, heads = model_dimensions(model)
    geometry = {
        "num_layers": layers,
        "query_heads": queries,
        "kv_heads": heads,
        "head_dim": model.model.layers[0].self_attn.head_dim,
    }
    if geometry != allocation["geometry"]:
        raise ValueError("pair replay model differs from the frozen allocation")
    output.mkdir(parents=True)
    write_budget_plans(output / "allocation", allocation, plans)
    write_json(
        output / "config.json",
        {
            "schema_version": "headkv-pair-combination-v1",
            "source_config": cfg,
            "geometry": geometry,
            "local_gate_directory": str(args.local_gate_dir),
            "plans": allocation["plans"],
            "thresholds": COMBINATION_GATE_THRESHOLDS,
            "policies": list(PAIR_POLICIES),
            "branch_order": ["dense", *PAIR_POLICIES, "dense_after"],
            "shift": 1,
            "scope": "frozen Calib combination checkpoints only; no continuous compressed Cache/Validation/Test",
        },
    )
    state = {
        "status": "running",
        "cohort": "calib",
        "target_sample_ids": ids,
        "completed_sample_ids": [],
        "validation_started": False,
    }
    write_json(output / "run_metadata.json", state)
    summaries = []
    conditions = {row["sample_id"]: row["condition"] for row in cfg["calib_samples"]}
    try:
        for sample in samples:
            sid = sample.frozen_record["sample_id"]
            summaries.append(
                collect_budget_sample(
                    model,
                    device=device,
                    sample=sample,
                    directory=output / "samples" / sid,
                    plans=plans,
                    steps=cfg["local_steps"][sample.frozen_record["dataset"]],
                    condition=conditions[sid],
                    block_rows=cfg["block_rows"],
                    policies=PAIR_POLICIES,
                    dense_after=True,
                )
            )
            state["completed_sample_ids"].append(sid)
            write_json(output / "run_metadata.json", state, overwrite=True)
        report = summarize_pair_budget_samples(summaries)
        write_json(output / "summary.json", report)
        state.update(
            status="completed", gate_decisions={s: row["gate_decision"] for s, row in report["gate_decisions"].items()}
        )
        write_json(output / "run_metadata.json", state, overwrite=True)
        emit_progress("head_budget_pair_calib_completed", samples=len(ids), validation_started=False)
        return 0
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "run_metadata.json", state, overwrite=True)
        raise


def run_collection(args):
    allocation, plans = read_budget_plans(args.allocation_dir)
    cfg = allocation["source_config"]
    ids = allocation[f"{args.cohort}_sample_ids"]
    configuration = {
        "source_config": cfg,
        "plans": allocation["plans"],
        "geometry": allocation["geometry"],
        "allocation_rule": allocation["rule"],
        "policies": list(POLICIES),
    }
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if args.cohort == "validation":
        if not args.calib_dir:
            raise ValueError("Validation requires completed combination Calib results")
        old = read_json(Path(args.calib_dir) / "run_metadata.json")
        if (
            old["status"] != "completed"
            or old["cohort"] != "calib"
            or old["completed_sample_ids"] != allocation["calib_sample_ids"]
            or read_json(Path(args.calib_dir) / "config.json") != configuration
        ):
            raise ValueError("Validation must reuse the completed Calib allocation/configuration")
    _frozen_cohorts(cfg["bundle"])
    index = load_b1_replay_index(
        cfg["run_set"], source_root=cfg["source_root"], expected_run_set_sha256=None, expected_data_sha256=None
    )
    _validate_execution_contract(index, SimpleNamespace(**cfg))
    samples = load_replay_samples(index, ids)
    if [s.frozen_record["sample_id"] for s in samples] != ids:
        raise ValueError("budget replay sample order differs from its frozen cohort")
    model, device = load_model(cfg["model"], args.device, cfg["dtype"])
    layers, queries, heads = model_dimensions(model)
    geometry = {
        "num_layers": layers,
        "query_heads": queries,
        "kv_heads": heads,
        "head_dim": model.model.layers[0].self_attn.head_dim,
    }
    if geometry != allocation["geometry"]:
        raise ValueError("budget replay geometry differs from fitted Calib")
    output.mkdir(parents=True)
    write_json(output / "config.json", configuration)
    state = {"status": "running", "cohort": args.cohort, "target_sample_ids": ids, "completed_sample_ids": []}
    write_json(output / "run_metadata.json", state)
    summaries = []
    conditions = {row["sample_id"]: row["condition"] for row in cfg[f"{args.cohort}_samples"]}
    try:
        for sample in samples:
            sid = sample.frozen_record["sample_id"]
            summary = collect_budget_sample(
                model,
                device=device,
                sample=sample,
                directory=output / "samples" / sid,
                plans=plans,
                steps=cfg["local_steps"][sample.frozen_record["dataset"]],
                condition=conditions[sid],
                block_rows=cfg["block_rows"],
            )
            summaries.append(summary)
            state["completed_sample_ids"].append(sid)
            write_json(output / "run_metadata.json", state, overwrite=True)
        write_json(output / "summary.json", summarize_budget_samples(summaries))
        state["status"] = "completed"
        write_json(output / "run_metadata.json", state, overwrite=True)
        emit_progress("head_budget_run_completed", cohort=args.cohort, samples=len(samples))
        return 0
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "run_metadata.json", state, overwrite=True)
        raise


def collect_local6_sample(model, *, device, sample, source_directory, directory, primary_step, block_rows=1024):
    """One Prefill and one observed primary branch; reuse the old 4/8-bit tables."""
    source, directory = Path(source_directory), Path(directory)
    old_meta = read_json(source / "metadata.json")
    sid, length = sample.frozen_record["sample_id"], len(sample.frozen_record["input_ids"])
    if (
        old_meta["status"] != "completed"
        or old_meta["sample_id"] != sid
        or old_meta["primary_step"] != primary_step
        or old_meta["prompt_length"] != length
    ):
        raise ValueError("local6 source differs from the completed frozen primary sample")
    replay_ids = tuple(sample.generated_ids[: primary_step + 2])
    if len(replay_ids) != primary_step + 2 or read_json(source / "replay.json")["generated_ids"][
        : len(replay_ids)
    ] != list(replay_ids):
        raise ValueError("local6 replay IDs differ from the old sensitivity input")
    with np.load(source / old_meta["local_files"][str(primary_step)]) as archive:
        old_arrays = {name: archive[name].copy() for name in archive.files}
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "sample_id": sid,
        "status": "running",
        "primary_step": primary_step,
        "prompt_length": length,
        "bit_widths": [6],
        "anchor_bit_widths": [4, 8],
        "anchor_heads": [0],
        "axes": ["side", "bits", "layer", "kv_head"],
        "source_directory": str(source),
    }
    write_json(directory / "metadata.json", metadata)
    records, anchors = {}, {}
    collector = QuantizationSensitivityCollector(length, None, block_rows, bit_widths=(6,))
    anchor = QuantizationSensitivityCollector(length, None, block_rows, heads=(0,))

    def observe(context):
        if context.decode_step != primary_step or context.layer_id in records:
            raise ValueError("local6 must observe each layer at the primary point exactly once")
        records[context.layer_id] = collector.collect(context)
        anchors[context.layer_id] = anchor.collect(context)

    started = time.perf_counter()
    device = torch.device(device)
    timer = GenerationPhaseTimer(model, device)
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        with timer:
            result = replay_quantization_checkpoints(
                model,
                device=device,
                sample_id=sid,
                prompt_ids=sample.frozen_record["input_ids"],
                replay_generated_ids=replay_ids,
                checkpoints={primary_step: ()},
                block_rows=block_rows,
                on_attention=observe,
            )
        layers, _, heads = model_dimensions(model)
        if set(records) != set(range(layers)) or timer.summary()["completed_forward_count"] != primary_step + 3:
            raise ValueError("incomplete local6 primary replay")

        def stack(rows):
            return {
                name: np.stack(
                    [rows[layer].arrays[name] for layer in range(layers)],
                    axis=0 if name.startswith("formula_sdpa_") else 2,
                )
                for name in rows[0].arrays
            }

        arrays, anchor_arrays = stack(records), stack(anchors)
        for name in ("representation_energy", "output_energy"):
            np.testing.assert_array_equal(
                arrays[name][:, 0], old_arrays[name][:, 0], err_msg=f"{sid}: reference {name}"
            )
        for name, value in anchor_arrays.items():
            expected = old_arrays[name][:, :1] if name.startswith("formula_sdpa_") else old_arrays[name][:, :, :, :1]
            np.testing.assert_array_equal(value, expected, err_msg=f"{sid}: H0 4/8 anchor {name}")
        dim = model.model.layers[0].self_attn.head_dim
        ledger = quantization_byte_ledger(length, primary_step + 1, layers, heads, dim, 6)
        for name in ("code_bytes", "statistic_bytes"):
            if not np.all(arrays[name] == ledger[name]):
                raise ValueError("local6 actual encoding differs from the existing byte layout")
        costs = np.stack(
            (old_arrays["output_nmse"][:, 0], arrays["output_nmse"][:, 0], old_arrays["output_nmse"][:, 1]), axis=1
        )
        if not np.isfinite(costs).all():
            raise ValueError("local gate cannot fit incomplete or undefined Head Output NMSE")
        write_npz(directory / "local6.npz", arrays)
        write_npz(directory / "anchor48.npz", anchor_arrays)
        write_json(
            directory / "replay.json",
            {"generated_ids": list(replay_ids), "predicted_ids": list(result.baseline.all_predicted_token_ids)},
        )
        resources = {"wall_seconds": time.perf_counter() - started, "forward_timing": timer.summary()}
        if device.type == "cuda":
            resources.update(
                peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        write_json(
            directory / "summary.json",
            {"reference_energy_equal": True, "all_H0_anchors_equal": True, "resources": resources},
        )
        metadata.update(status="completed", baseline_top1_match=True, num_layers=layers, kv_heads=heads)
        write_json(directory / "metadata.json", metadata, overwrite=True)
        return costs
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(directory / "metadata.json", metadata, overwrite=True)
        raise


def run_local_diagnostic(args):
    """Collect only Calib primary-point local6, fit once, and stop after gate one."""
    allocation, old_plans = read_budget_plans(args.allocation_dir)
    source = Path(args.source_calib_dir)
    old_fit, fitted = fit_budget_plans(source)
    cfg = allocation["source_config"]
    if fitted != old_plans or old_fit["source_config"] != cfg or old_fit["scores"] != allocation["scores"]:
        raise ValueError("local diagnostic must reuse the frozen original Calib allocation/profile")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    _frozen_cohorts(cfg["bundle"])
    index = load_b1_replay_index(
        cfg["run_set"], source_root=cfg["source_root"], expected_run_set_sha256=None, expected_data_sha256=None
    )
    _validate_execution_contract(index, SimpleNamespace(**cfg))
    ids = allocation["calib_sample_ids"]
    samples = load_replay_samples(index, ids)
    if [s.frozen_record["sample_id"] for s in samples] != ids:
        raise ValueError("local6 sample order differs from frozen Calib")
    model, device = load_model(cfg["model"], args.device, cfg["dtype"])
    layers, queries, heads = model_dimensions(model)
    geometry = {
        "num_layers": layers,
        "query_heads": queries,
        "kv_heads": heads,
        "head_dim": model.model.layers[0].self_attn.head_dim,
    }
    if geometry != allocation["geometry"]:
        raise ValueError("local6 model geometry differs from frozen Calib")
    output.mkdir(parents=True)
    write_json(
        output / "config.json",
        {
            "schema_version": "headkv-local-budget-gate-v1",
            "source_config": cfg,
            "source_calib_directory": str(source),
            "old_allocation_directory": str(args.allocation_dir),
            "geometry": geometry,
            "thresholds": LOCAL_GATE_THRESHOLDS,
            "scope": "Calib primary local6 plus fixed/Oracle analysis only; no combination/Validation/Test",
        },
    )
    state = {"status": "running", "cohort": "calib", "target_sample_ids": ids, "completed_sample_ids": []}
    write_json(output / "run_metadata.json", state)
    costs = []
    try:
        for sample in samples:
            sid = sample.frozen_record["sample_id"]
            step = cfg["local_steps"][sample.frozen_record["dataset"]][0]
            emit_progress("head_budget_local6_started", sample_id=sid, decode_step=step)
            costs.append(
                collect_local6_sample(
                    model,
                    device=device,
                    sample=sample,
                    source_directory=source / "samples" / sid,
                    directory=output / "samples" / sid,
                    primary_step=step,
                    block_rows=cfg["block_rows"],
                )
            )
            state["completed_sample_ids"].append(sid)
            write_json(output / "run_metadata.json", state, overwrite=True)
            emit_progress("head_budget_local6_completed", sample_id=sid)
        conditions = [row["condition"] for row in cfg["calib_samples"]]
        report, arrays, plans = analyze_local_pairs(np.stack(costs), ids, conditions, old_plans)
        # Verify the full serialized budget for every actual primary point.
        for sample in samples:
            length = len(sample.frozen_record["input_ids"])
            consumed = cfg["local_steps"][sample.frozen_record["dataset"]][0] + 1
            for side in ("key", "value"):
                a, b = (
                    head_plan_byte_ledger(plans[f"{side}_{method}"], length, consumed)
                    for method in ("uniform6", "fixed")
                )
                if (
                    a["total_bytes"] != b["total_bytes"]
                    or a["request_bytes"] != b["request_bytes"]
                    or a["model_bytes"] != b["model_bytes"]
                ):
                    raise ValueError("paired allocation does not have the exact uniform encoded budget")
        frozen = {
            "schema_version": "headkv-local-pair-allocation-v1",
            "geometry": geometry,
            "calib_sample_ids": ids,
            "source_config": cfg,
            "thresholds": LOCAL_GATE_THRESHOLDS,
            "eligible_sides": [
                s for s in report["sides"] if report["sides"][s]["gate_decision"] == "eligible_for_combination_review"
            ],
            "rule": "per layer all6 or one 4/8 pair; maximize mean Calib (6-to-8 benefit minus 6-to-4 penalty); positive gain only; ties donor/receiver ascending",
            "plans": {
                name: {"side": p.side, "bits": [list(row) for row in p.bits], "head_dim": p.head_dim}
                for name, p in plans.items()
            },
        }
        write_budget_plans(output / "allocation", frozen, plans)
        write_npz(output / "analysis.npz", arrays)
        write_json(output / "summary.json", report)
        state.update(
            status="completed",
            gate_decisions={s: row["gate_decision"] for s, row in report["sides"].items()},
            next_stage_started=False,
        )
        write_json(output / "run_metadata.json", state, overwrite=True)
        emit_progress("head_budget_local_gate_completed", samples=len(ids), decisions=state["gate_decisions"])
        return 0
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "run_metadata.json", state, overwrite=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="freeze plans from existing Calib numerical packages; no model")
    fit.add_argument("--source-calib-dir", required=True)
    fit.add_argument("--output-dir", required=True)
    run = commands.add_parser("run", help="replay one frozen cohort; never auto-start Validation")
    run.add_argument("--allocation-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--cohort", choices=("calib", "validation"), default="calib")
    run.add_argument("--calib-dir")
    run.add_argument("--device", default="cuda:4")
    local = commands.add_parser(
        "diagnose-local", help="Calib primary local6, fixed-pair and local Oracle; stop at gate one"
    )
    local.add_argument("--allocation-dir", required=True)
    local.add_argument("--source-calib-dir", required=True)
    local.add_argument("--output-dir", required=True)
    local.add_argument("--device", default="cuda:4")
    pairs = commands.add_parser(
        "replay-pairs", help="Calib only: frozen F, uniform6, +1 control and two Dense references"
    )
    pairs.add_argument("--local-gate-dir", required=True)
    pairs.add_argument("--output-dir", required=True)
    pairs.add_argument("--device", default="cuda:4")
    args = parser.parse_args(argv)
    if args.command == "replay-pairs":
        return run_pair_collection(args)
    if args.command == "diagnose-local":
        return run_local_diagnostic(args)
    if args.command == "fit":
        config, plans = fit_budget_plans(args.source_calib_dir)
        write_budget_plans(args.output_dir, config, plans)
        return 0
    return run_collection(args)
