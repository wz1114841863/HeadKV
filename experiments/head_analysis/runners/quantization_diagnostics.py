"""D1 controls and opt-in D2 propagation on the existing fixed-replay path."""

from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import argparse
import time
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments.head_analysis.common.io import read_json, write_json, write_npz
from experiments.head_analysis.collectors.quantization_sensitivity import QuantizationSensitivityCollector
from experiments.head_analysis.collectors.quantization_diagnostics import (
    PropagationResult,
    QuantizationPropagationCollector,
)
from experiments.head_analysis.runners.b2_runner import (
    TeacherForcedReplayResult,
    teacher_forced_replay,
    emit_progress,
    load_b1_replay_index,
    load_replay_samples,
    _validate_execution_contract,
)
from experiments.head_analysis.runners.quantization_runner import (
    isolated_checkpoint_forward,
    prediction_errors,
    _frozen_cohorts,
)
from experiments.head_analysis.runners.resource_metrics import GenerationPhaseTimer
from experiments.head_analysis.runtime.head_intervention import HeadAttentionInputs, HeadIntervention
from experiments.head_analysis.runtime.kv_quantization import quantized_prompt_range
from experiments.head_analysis.runtime.model import load_model


D1_SEQUENCE = ("B0", "N", "C0", "Q4", "Q8", "B1", "Q8_repeat", "Q4_repeat", "D4", "D8", "C1", "B2")
D1_CASES = {
    "govreport_16k_003": (HeadIntervention(3, 7, "key", None), HeadIntervention(3, 6, "key", None)),
    "pg19_001_32k": (HeadIntervention(11, 4, "value", None), HeadIntervention(3, 2, "value", None)),
}


@dataclass(frozen=True)
class D1CheckpointResult:
    decode_step: int
    target: HeadIntervention
    baseline_predicted_token_id: int
    records: tuple[dict, ...]
    comparisons: dict
    logits: tuple[torch.Tensor, ...] | None = None  # Opt-in CPU copies.
    propagation: PropagationResult | None = None


@dataclass(frozen=True)
class D1ReplayResult:
    baseline: TeacherForcedReplayResult
    checkpoints: tuple[D1CheckpointResult, ...]


def _snapshot(inputs):
    # Only the selected layer, never a complete model Cache. Probe tensors stay read-only.
    return replace(
        inputs,
        **{
            name: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for name, value in vars(inputs).items()
        },
    )


def _layout(inputs):
    return {
        name: (
            None
            if value is None
            else {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "contiguous": value.is_contiguous(),
                "storage_offset": value.storage_offset(),
            }
        )
        for name in ("query", "key", "value", "attention_mask")
        for value in (getattr(inputs, name),)
    } | {
        "cache_position": inputs.cache_position.tolist(),
        "scaling": inputs.scaling,
        "is_causal_requested": inputs.is_causal,
        "dropout_p": inputs.dropout_p,
    }


def _match_inputs(reference, current):
    checks = {}
    for name in vars(reference):
        a, b = getattr(reference, name), getattr(current, name)
        checks[name] = (
            isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.device == b.device and torch.equal(a, b)
            if isinstance(a, torch.Tensor)
            else a == b
        )
    if not all(checks.values()):
        raise RuntimeError(f"D1 checkpoint inputs differ: {[k for k, v in checks.items() if not v]}")
    return checks


def run_d1_checkpoint(
    model,
    *,
    cache,
    device,
    prompt_length: int,
    decode_step: int,
    input_token: int,
    expected_token: int,
    target: HeadIntervention,
    sample_id="checkpoint",
    block_rows=1024,
    capture_logits=False,
    check_inputs=True,
    on_attention=None,
    on_record=None,
    capture_propagation=False,
) -> D1CheckpointResult:
    """Run B/N/C/Q/D controls at one coordinate, always relative to B0.

    Nonzero output differences are evidence, not failures. Input/isolation violations
    raise before interpretation; all Cache references and scoped backends are restored.
    ``check_inputs=False`` disables new diagnostic hooks/copies for matched controls;
    any original ``on_attention`` observer remains active on B forwards.
    ``capture_propagation=True`` adds D2 to the same forwards; FP32 projection
    uses only captured B0/Q4/Q8 hidden vectors after the real sequence finishes.
    """
    if target.bits is not None:
        raise ValueError("D1 target specifies a coordinate/side with bits=None")
    if capture_propagation and not check_inputs:
        raise ValueError("D2 propagation requires the checked input callbacks")
    propagation = (
        QuantizationPropagationCollector(target, model.config.num_attention_heads, model.config.num_key_value_heads)
        if capture_propagation
        else None
    )
    history = [(layer, layer.keys, layer.values) for layer in cache.layers]
    reference_inputs = None
    current_checks, current_layout = {}, {}

    def observe(context):
        nonlocal reference_inputs, current_checks, current_layout
        if context.layer_id == target.layer_id and check_inputs:
            inputs = HeadAttentionInputs(
                context.query,
                context.key,
                context.value,
                context.attention_mask,
                context.scaling,
                context.is_causal,
                context.dropout_p,
                context.cache_position,
            )
            if reference_inputs is None:
                reference_inputs = _snapshot(inputs)
            current_checks = _match_inputs(reference_inputs, inputs)
            current_layout = {"incoming": _layout(inputs), "actual": _layout(inputs)}
            if propagation is not None:
                propagation.capture_inputs(inputs)
        if on_attention is not None:
            on_attention(context)

    def observe_intervention(before, after):
        nonlocal current_checks, current_layout
        current_checks = _match_inputs(reference_inputs, before)
        if label.startswith(("Q", "D")) and label not in ("Q4", "Q8"):
            start, end = quantized_prompt_range(prompt_length)
            source = after.key if target.side == "key" else after.value
            same = torch.equal(source[0, target.head_id, start:end], reconstructions[int(label[1])].restored)
            current_checks["reconstruction_matches_first_Q"] = same
            if not same:
                raise RuntimeError(f"D1 reconstruction_matches_first_Q failed at {label}")
        current_layout = {"incoming": _layout(before), "actual": _layout(after)}
        if propagation is not None:
            propagation.capture_inputs(after)

    saved, reconstructions, records, captured = {}, {}, [], [] if capture_logits else None
    with isolated_checkpoint_forward(
        model,
        cache=cache,
        device=device,
        prompt_length=prompt_length,
        decode_step=decode_step,
        input_token=input_token,
        block_rows=block_rows,
        sample_id=sample_id,
    ) as forward:
        for label in D1_SEQUENCE:
            current_checks, current_layout = {}, {}
            with propagation.capture(model, label) if propagation is not None else nullcontext():
                if label.startswith("B"):
                    logits, state = forward(on_attention=observe if check_inputs or on_attention is not None else None)
                else:
                    bits = int(label[1]) if label.startswith(("Q", "D")) else None
                    options = {
                        "clone_only": label.startswith("C"),
                        "capture_reconstruction": label in ("Q4", "Q8"),
                        "check_isolation": check_inputs,
                        "on_inputs": observe_intervention if check_inputs else None,
                    }
                    if label.startswith("D"):
                        options["fixed_reconstruction"] = reconstructions[bits]
                    logits, state = forward(replace(target, bits=bits), **options)
                    if label in ("Q4", "Q8"):
                        reconstructions[bits] = state.reconstruction
            if label == "B0":
                saved["B0"] = logits
            checks = dict(current_checks)
            checks["all_cache_references_restored"] = all(
                layer.keys is k and layer.values is v and layer.get_seq_length() == prompt_length + decode_step
                for layer, k, v in history
            )
            if check_inputs:
                layer = cache.layers[target.layer_id]
                checks["target_layer_history_values_restored"] = torch.equal(
                    layer.keys, reference_inputs.key[:, :, :-1]
                ) and torch.equal(layer.values, reference_inputs.value[:, :, :-1])
            if not all(checks.values()):
                raise RuntimeError(f"D1 Cache isolation failed at {label}: {checks}")
            records.append(
                {
                    "label": label,
                    "mode": "baseline" if state is None else state.mode,
                    **prediction_errors(saved["B0"], logits, expected_token),
                    "encoding_performed": state is not None and state.encoding_performed,
                    "measurement": None if state is None else asdict(state.measurement),
                    "input_checks": checks,
                    "isolation_checks": None if state is None else state.isolation_checks,
                    "layout": current_layout,
                }
            )
            if propagation is not None:
                records[-1]["propagation"] = propagation.record(logits, expected_token)
            saved[label] = logits
            if on_record is not None:
                on_record(records[-1], logits)  # Read-only; copy any retained tensor.
            if captured is not None:
                captured.append(logits.detach().cpu().clone())
        comparisons = {}
        pairs = [(k, "B0") for k in ("N", "C0", "B1", "C1", "B2")]
        pairs += [("Q4_repeat", "Q4"), ("Q8_repeat", "Q8"), ("D4", "Q4"), ("D8", "Q8")]
        for current, reference in pairs:
            a, b = saved[reference], saved[current]
            comparisons[f"{current}_vs_{reference}"] = {
                "logits_equal": torch.equal(a, b),
                "max_absolute_difference": (a.double() - b.double()).abs().max().item(),
                "prediction_kl": prediction_errors(a, b, expected_token)["prediction_kl"],
            }
    return D1CheckpointResult(
        decode_step,
        target,
        int(saved["B0"].argmax().item()),
        tuple(records),
        comparisons,
        tuple(captured) if captured is not None else None,
        propagation.finish(model.lm_head, expected_token) if propagation is not None else None,
    )


def replay_d1_checkpoints(
    model,
    *,
    device,
    sample_id: str,
    prompt_ids: Sequence[int],
    replay_generated_ids: Sequence[int],
    checkpoints: Mapping[int, Sequence[HeadIntervention]],
    require_top1_match=True,
    capture_logits=False,
    check_inputs=True,
    block_rows=1024,
    on_attention=None,
    capture_propagation=False,
) -> D1ReplayResult:
    """One original Prefill/replay; every coordinate branches from the original history."""
    if any(s < 0 or s >= len(replay_generated_ids) - 1 for s in checkpoints):
        raise ValueError("D1 checkpoints require consumed and next replay token IDs")
    results = []

    def before_decode(step, cache):
        for target in checkpoints.get(step, ()):
            result = run_d1_checkpoint(
                model,
                cache=cache,
                device=device,
                prompt_length=len(prompt_ids),
                decode_step=step,
                input_token=int(replay_generated_ids[step]),
                expected_token=int(replay_generated_ids[step + 1]),
                target=target,
                sample_id=sample_id,
                capture_logits=capture_logits,
                check_inputs=check_inputs,
                block_rows=block_rows,
                on_attention=on_attention,
                capture_propagation=capture_propagation,
            )
            if require_top1_match and result.baseline_predicted_token_id != replay_generated_ids[step + 1]:
                raise ValueError(f"{sample_id}: D1 baseline differs from B1 at step {step}")
            results.append(result)

    baseline = teacher_forced_replay(
        model,
        device=device,
        sample_id=sample_id,
        prompt_ids=prompt_ids,
        replay_generated_ids=replay_generated_ids,
        require_top1_match=require_top1_match,
        capture_logits=capture_logits,
        before_decode=before_decode,
    )
    return D1ReplayResult(baseline, tuple(results))


def collect_d1_sample(
    model,
    *,
    device,
    sample,
    source_directory,
    directory,
    targets,
    conditional_old_order=False,
    block_rows=1024,
    capture_propagation=False,
):
    """Persist D1 controls while keeping one Prefill and the original B1 prefix."""
    directory, source = Path(directory), Path(source_directory)
    metadata = read_json(source / "metadata.json")
    sid, step = sample.frozen_record["sample_id"], metadata["primary_step"]
    prompt_length = len(sample.frozen_record["input_ids"])
    if metadata["status"] != "completed" or metadata["sample_id"] != sid or metadata["prompt_length"] != prompt_length:
        raise ValueError("D1 source does not match the completed original sample")
    directory.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    old_targets = [HeadIntervention(**row) for row in metadata["expected_model_targets"]]
    with np.load(source / "model_branches.npz") as archive:
        old = {
            (int(l), int(h), int(s), int(b)): (float(k), bool(t))
            for l, h, s, b, k, t in zip(
                archive["layer_id"],
                archive["head_id"],
                archive["side_id"],
                archive["bits"],
                archive["prediction_kl"],
                archive["top1_match"],
            )
        }
    observer = QuantizationSensitivityCollector(prompt_length, lambda record: None, block_rows).on_attention
    state = {
        "status": "running",
        "sample_id": sid,
        "source_directory": str(source.resolve()),
        "decode_step": step,
        "targets": [asdict(t) for t in targets],
        "completed_cases": [],
        "old_order_trigger": None,
        "original_observer": "QuantizationSensitivityCollector on B forwards",
        "capture_propagation": capture_propagation,
    }
    write_json(directory / "metadata.json", state)
    started = time.perf_counter()

    def save_case(cache, target, checked):
        name = f"L{target.layer_id:02d}_H{target.head_id:02d}_{target.side}_" + ("checked" if checked else "no_checks")
        state["active_case"] = name
        rows, vectors, native_dtype = [], [], []
        emit_progress("d1_case_started", sample_id=sid, case=name)

        def record(row, logits):
            rows.append(row)
            vectors.append(logits.detach().cpu().float().numpy().copy())
            native_dtype.append(str(logits.dtype))
            emit_progress("d1_branch_completed", sample_id=sid, case=name, label=row["label"])

        payload = {"target": asdict(target), "checked": checked, "status": "running", "records": rows}
        try:
            result = run_d1_checkpoint(
                model,
                cache=cache,
                device=device,
                prompt_length=prompt_length,
                decode_step=step,
                input_token=sample.generated_ids[step],
                expected_token=sample.generated_ids[step + 1],
                target=target,
                sample_id=sid,
                block_rows=block_rows,
                check_inputs=checked,
                on_attention=observer,
                on_record=record,
                capture_propagation=capture_propagation and checked,
            )
            payload.update(status="completed", comparisons=result.comparisons)
            if result.baseline_predicted_token_id != sample.generated_ids[step + 1]:
                raise ValueError("D1 checkpoint baseline differs from original B1")
            if result.propagation is not None:
                payload["projection"] = result.propagation.projection
                write_npz(directory / (name + "_propagation.npz"), result.propagation.arrays)
            return payload, vectors
        except Exception as exc:
            payload.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write_json(directory / (name + ".json"), payload)
            if vectors:
                write_npz(
                    directory / (name + ".npz"),
                    {
                        "logits": np.stack(vectors),
                        "labels": np.asarray([r["label"] for r in rows]),
                        "native_dtype": np.asarray(native_dtype),
                    },
                )
            if payload["status"] == "completed":
                state["completed_cases"].append(name)
            write_json(directory / "metadata.json", state, overwrite=True)

    def before_decode(current, cache):
        if current != step:
            return
        primary, primary_vectors = save_case(cache, targets[0], True)
        save_case(cache, targets[1], True)
        bare, bare_vectors = save_case(cache, targets[0], False)
        write_json(
            directory / "observation_comparison.json",
            {
                label: {
                    "logits_equal": bool(np.array_equal(a, b)),
                    "max_absolute_difference": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))),
                }
                for label, a, b in zip(D1_SEQUENCE, primary_vectors, bare_vectors)
            },
        )
        first = {r["label"]: r for r in primary["records"]}
        side_id = ("key", "value").index(targets[0].side)
        reference = {str(b): old[(targets[0].layer_id, targets[0].head_id, side_id, b)] for b in (4, 8)}
        observed = {str(b): (first[f"Q{b}"]["prediction_kl"], first[f"Q{b}"]["top1_match"]) for b in (4, 8)}
        state["original_primary_comparison"] = {"old": reference, "isolated": observed}
        # A conservative trigger, not a scientific gate: do not choose a tolerance after seeing the new result.
        if conditional_old_order and observed != reference:
            state["old_order_trigger"] = "original Q4/Q8 KL or Top-1 not reproduced exactly; not a quality verdict"
            emit_progress("d1_original_order_started", sample_id=sid, candidates=len(old_targets))
            rows, vectors, labels = [], [], []
            payload = {"status": "running", "records": rows, "trigger": state["old_order_trigger"]}
            try:
                with isolated_checkpoint_forward(
                    model,
                    cache=cache,
                    device=device,
                    prompt_length=prompt_length,
                    decode_step=step,
                    input_token=sample.generated_ids[step],
                    block_rows=block_rows,
                    sample_id=sid,
                ) as forward:
                    baseline, _ = forward(on_attention=observer)
                    payload["native_dtype"] = str(baseline.dtype)
                    vectors.append(baseline.detach().cpu().float().numpy().copy())
                    labels.append("baseline")
                    for target in old_targets:
                        logits, intervention = forward(target)
                        rows.append(
                            {
                                "target": asdict(target),
                                **prediction_errors(baseline, logits, sample.generated_ids[step + 1]),
                            }
                        )
                        if any(
                            (target.layer_id, target.head_id, target.side) == (t.layer_id, t.head_id, t.side)
                            for t in targets
                        ) and target.bits in (4, 8):
                            vectors.append(logits.detach().cpu().float().numpy().copy())
                            labels.append(f"L{target.layer_id}_H{target.head_id}_{target.side}_{target.bits}")
                payload["status"] = "completed"
            except Exception as exc:
                payload.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                write_json(directory / "original_order.json", payload)
                if vectors:
                    write_npz(
                        directory / "original_order.npz", {"logits": np.stack(vectors), "labels": np.asarray(labels)}
                    )
        write_json(directory / "metadata.json", state, overwrite=True)

    timer = GenerationPhaseTimer(model, device)
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        with timer:
            baseline = teacher_forced_replay(
                model,
                device=device,
                sample_id=sid,
                prompt_ids=sample.frozen_record["input_ids"],
                replay_generated_ids=sample.generated_ids[: step + 2],
                before_decode=before_decode,
            )
        expected = 1 + (step + 1) + 3 * len(D1_SEQUENCE)
        if state["old_order_trigger"] is not None:
            expected += 1 + len(old_targets)
        timing = timer.summary()
        if timing["completed_forward_count"] != expected or timing["incomplete_forward_count"]:
            raise ValueError("D1 root Forward count differs from the fixed matrix")
        resources = {
            "wall_seconds": time.perf_counter() - started,
            "forward_timing": timing,
            "scope": (
                "D1/D2 controls, original Observer, arithmetic references and writing; not deployment performance"
                if capture_propagation
                else "D1 research controls, original Observer and writing; not deployment performance"
            ),
        }
        if device.type == "cuda":
            resources.update(
                peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        write_json(
            directory / "replay.json",
            {
                "generated_ids": list(sample.generated_ids[: step + 2]),
                "predicted_ids": list(baseline.all_predicted_token_ids),
            },
        )
        state.update(status="completed", resources=resources)
        write_json(directory / "metadata.json", state, overwrite=True)
        return state
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(directory / "metadata.json", state, overwrite=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run fixed D1 controls with optional D2 propagation references.")
    for name in ("calib-dir", "validation-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--capture-propagation", action="store_true", help="Capture D2 on the four checked coordinates")
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    calib, validation = Path(args.calib_dir), Path(args.validation_dir)
    cfg = read_json(calib / "config.json")
    if cfg != read_json(validation / "config.json"):
        raise ValueError("Calib and Validation frozen configurations differ")
    for root, role in ((calib, "calib"), (validation, "validation")):
        state = read_json(root / "run_metadata.json")
        if state["status"] != "completed" or state["cohort"] != role:
            raise ValueError("D1 requires completed original Calib and Validation sources")
    _frozen_cohorts(cfg["bundle"])
    index = load_b1_replay_index(
        cfg["run_set"], source_root=cfg["source_root"], expected_run_set_sha256=None, expected_data_sha256=None
    )
    _validate_execution_contract(index, SimpleNamespace(**cfg))
    samples = load_replay_samples(index, list(D1_CASES))
    if [s.frozen_record["sample_id"] for s in samples] != list(D1_CASES):
        raise ValueError("D1 replay selection differs from the fixed two samples")
    sources = {sid: (validation if sid == "govreport_16k_003" else calib) / "samples" / sid for sid in D1_CASES}
    for sid, source in sources.items():
        if read_json(source / "metadata.json")["primary_step"] != 31:
            raise ValueError("D1 requires the frozen s=31 source checkpoint")
    model, device = load_model(cfg["model"], args.device, cfg["dtype"])
    output.mkdir(parents=True)
    write_json(
        output / "config.json",
        {
            "source_config": cfg,
            "cases": {s: [asdict(t) for t in v] for s, v in D1_CASES.items()},
            "sequence": list(D1_SEQUENCE),
            "calib_dir": str(calib.resolve()),
            "validation_dir": str(validation.resolve()),
            "d2_enabled": args.capture_propagation,
        },
    )
    state = {"status": "running", "target_sample_ids": list(D1_CASES), "completed_sample_ids": []}
    write_json(output / "run_metadata.json", state)
    try:
        for sample in samples:
            sid = sample.frozen_record["sample_id"]
            state["active_sample_id"] = sid
            write_json(output / "run_metadata.json", state, overwrite=True)
            emit_progress("d1_sample_started", sample_id=sid)
            collect_d1_sample(
                model,
                device=device,
                sample=sample,
                source_directory=sources[sid],
                directory=output / "samples" / sid,
                targets=D1_CASES[sid],
                block_rows=cfg["block_rows"],
                conditional_old_order=sid == "govreport_16k_003",
                capture_propagation=args.capture_propagation,
            )
            state["completed_sample_ids"].append(sid)
            write_json(output / "run_metadata.json", state, overwrite=True)
        state["status"] = "completed"
        write_json(output / "run_metadata.json", state, overwrite=True)
        emit_progress("d1_run_completed", samples=len(samples))
        return 0
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "run_metadata.json", state, overwrite=True)
        emit_progress("d1_run_failed", error=state["error"])
        return 1
