"""Isolated single-Forward quantization branches on the existing fixed replay."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
import argparse
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from experiments.head_analysis.runners.b2_runner import (
    TeacherForcedReplayResult,
    _last_token_logits,
    teacher_forced_replay,
)
from experiments.head_analysis.runtime.head_intervention import (
    HeadIntervention,
    InterventionMeasurement,
    single_head_intervention,
)
from experiments.head_analysis.runtime.attention_probe import AttentionProbe, AttentionProbeContext
from experiments.head_analysis.collectors.quantization_sensitivity import (
    BIT_WIDTHS,
    SIDES,
    KL_NEGATIVE_TOLERANCE,
    QuantizationSensitivityCollector,
)
from experiments.head_analysis.common.io import read_json, write_json, write_npz
from experiments.head_analysis.common.statistics import finite_mean_count
from experiments.head_analysis.runners.b2_runner import (
    _validate_execution_contract,
    emit_progress,
    load_b1_replay_index,
    load_replay_samples,
)
from experiments.head_analysis.runners.input_selection import frozen_length
from experiments.head_analysis.runners.key_encoding_runner import PILOT_IDS, VALIDATION_IDS
from experiments.head_analysis.runners.resource_metrics import GenerationPhaseTimer
from experiments.head_analysis.runtime.kv_quantization import GROUP_SIZE, PROTECTED_TOKENS, quantized_prompt_range
from experiments.head_analysis.runtime.model import load_model, model_dimensions
from experiments.head_analysis.storage.quantization_results import QuantizationResultWriter, read_sample_profiles


MODEL_LAYERS = (3, 11, 19, 27)
LOCAL_STEPS = {"ruler": (5,), "govreport": (31, 255), "pg19": (31, 255)}


@dataclass(frozen=True)
class BranchResult:
    target: HeadIntervention
    predicted_token_id: int
    prediction_kl: float
    kl_is_roundoff: bool
    delta_nll: float
    top1_match: bool
    measurement: InterventionMeasurement
    logits: torch.Tensor | None = None  # Opt-in CPU copy for tiny-model tests only.


@dataclass(frozen=True)
class CheckpointResult:
    decode_step: int
    input_token_id: int
    expected_token_id: int
    baseline_predicted_token_id: int
    branches: tuple[BranchResult, ...]


@dataclass(frozen=True)
class QuantizationReplayResult:
    baseline: TeacherForcedReplayResult
    checkpoints: tuple[CheckpointResult, ...]


def prediction_errors(reference: torch.Tensor, branch: torch.Tensor, expected_token: int) -> dict:
    """Full-vocabulary KL(ref||branch), NLL delta, and Top-1; no clipping."""
    if reference.ndim != 1 or reference.shape != branch.shape:
        raise ValueError("prediction comparison requires matching vocabulary vectors")
    if not torch.isfinite(reference).all() or not torch.isfinite(branch).all():
        raise ValueError("prediction logits must be finite")
    log_ref = reference.float().log_softmax(-1)
    log_branch = branch.float().log_softmax(-1)
    kl = (log_ref.exp().double() * (log_ref.double() - log_branch.double())).sum().item()
    if kl < -KL_NEGATIVE_TOLERANCE:
        raise FloatingPointError(f"prediction KL below the FP32 log-softmax tolerance: {kl}")
    predicted = int(branch.argmax().item())
    return {
        "predicted_token_id": predicted,
        "prediction_kl": kl,
        "kl_is_roundoff": kl < 0,
        "delta_nll": (log_ref[expected_token] - log_branch[expected_token]).item(),
        "top1_match": predicted == int(reference.argmax().item()),
    }


@contextmanager
def isolated_checkpoint_forward(
    model,
    *,
    cache,
    device,
    prompt_length: int,
    decode_step: int,
    input_token: int,
    block_rows: int = 1024,
    sample_id: str = "checkpoint",
):
    """Yield a one-token Forward that restores all DynamicLayer references.

    Transformers 5.2 DynamicLayer appends with torch.cat, so its old K/V tensors
    stay immutable. Saving those references avoids copying a complete 32K Cache
    and also restores all layers if a Forward fails halfway through.
    """
    if not isinstance(cache, DynamicCache) or cache.offloading or len(cache.layers) != len(model.model.layers):
        raise ValueError("checkpoint branches require a complete, non-offloaded DynamicCache")
    history = []
    for layer in cache.layers:
        if type(layer) is not DynamicLayer or not layer.is_initialized:
            raise ValueError("checkpoint branches support initialized full DynamicLayer histories only")
        if layer.get_seq_length() != prompt_length + decode_step:
            raise ValueError("checkpoint Cache length does not match the requested Decode step")
        history.append((layer, layer.keys, layer.values))
    device = torch.device(device)
    inputs = torch.tensor([[input_token]], dtype=torch.long, device=device)
    position = torch.tensor([prompt_length + decode_step], dtype=torch.long, device=device)

    def forward(target=None, *, on_attention=None, **intervention_options):
        try:
            if target is None:
                observation = (
                    nullcontext()
                    if on_attention is None
                    else AttentionProbe(
                        sample_id=sample_id,
                        prompt_length=prompt_length,
                        decode_step_offset=decode_step,
                        on_attention=on_attention,
                    )(model)
                )
                with observation:
                    output = model(
                        input_ids=inputs,
                        past_key_values=cache,
                        use_cache=True,
                        cache_position=position,
                        logits_to_keep=1,
                    )
                return _last_token_logits(output), None
            with single_head_intervention(model, target, prompt_length, block_rows, **intervention_options) as state:
                output = model(
                    input_ids=inputs, past_key_values=cache, use_cache=True, cache_position=position, logits_to_keep=1
                )
            return _last_token_logits(output), state
        finally:
            for layer, keys, values in history:
                layer.keys, layer.values = keys, values

    try:
        with torch.inference_mode():
            yield forward
    finally:
        for layer, keys, values in history:
            layer.keys, layer.values = keys, values


def run_checkpoint_branches(
    model,
    *,
    cache,
    device,
    prompt_length: int,
    decode_step: int,
    input_token: int,
    expected_token: int,
    targets: Sequence[HeadIntervention],
    capture_logits: bool = False,
    block_rows: int = 1024,
    sample_id: str = "checkpoint",
    on_attention: Callable[[AttentionProbeContext], None] | None = None,
) -> CheckpointResult:
    """Normal V3-B branches, sharing the checkpoint lifecycle with D1 controls."""
    results = []
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
        reference, _ = forward(on_attention=on_attention)
        for target in targets:
            logits, state = forward(target)
            errors = prediction_errors(reference, logits, expected_token)
            results.append(
                BranchResult(
                    target=target,
                    **errors,
                    measurement=state.measurement,
                    logits=logits.detach().cpu() if capture_logits else None,
                )
            )
    return CheckpointResult(decode_step, input_token, expected_token, int(reference.argmax().item()), tuple(results))


def replay_quantization_checkpoints(
    model,
    *,
    device,
    sample_id: str,
    prompt_ids: Sequence[int],
    replay_generated_ids: Sequence[int],
    checkpoints: Mapping[int, Sequence[HeadIntervention]],
    require_top1_match: bool = True,
    capture_logits: bool = False,
    block_rows: int = 1024,
    on_attention: Callable[[AttentionProbeContext], None] | None = None,
    on_checkpoint: Callable[[CheckpointResult], None] | None = None,
) -> QuantizationReplayResult:
    """One Prefill, one unmodified main replay, and isolated selected-step branches."""
    if any(step < 0 or step >= len(replay_generated_ids) - 1 for step in checkpoints):
        raise ValueError("each checkpoint needs both its consumed and next B1 token")
    results = []

    def before_decode(step, cache):
        if step in checkpoints:
            checkpoint = run_checkpoint_branches(
                model,
                cache=cache,
                device=device,
                prompt_length=len(prompt_ids),
                decode_step=step,
                input_token=int(replay_generated_ids[step]),
                expected_token=int(replay_generated_ids[step + 1]),
                targets=checkpoints[step],
                capture_logits=capture_logits,
                block_rows=block_rows,
                sample_id=sample_id,
                on_attention=on_attention,
            )
            if require_top1_match and checkpoint.baseline_predicted_token_id != checkpoint.expected_token_id:
                raise ValueError(f"{sample_id}: checkpoint reference differs from B1 at step {step}")
            if on_checkpoint is not None:
                on_checkpoint(checkpoint)
            results.append(checkpoint)

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
    return QuantizationReplayResult(baseline, tuple(results))


def collect_quantization_sample(
    model, *, device, sample, directory, source=None, model_layers=None, local_steps=None, block_rows=1024
):
    """Compose the Collector, isolated replay and Writer; CPU arrays only survive callbacks."""
    record = sample.frozen_record
    length = frozen_length(record)
    model_layers = tuple(MODEL_LAYERS if model_layers is None else model_layers)
    steps = tuple(LOCAL_STEPS[record["dataset"]] if local_steps is None else local_steps)
    layers, query_heads, heads = model_dimensions(model)
    dim = model.model.layers[0].self_attn.head_dim
    if (
        not model_layers
        or len(set(model_layers)) != len(model_layers)
        or any(l < 0 or l >= layers for l in model_layers)
    ):
        raise ValueError("model branch layers must be distinct coordinates inside this model")
    if not steps or list(steps) != sorted(set(steps)) or len(sample.generated_ids) < steps[-1] + 2:
        raise ValueError("checkpoints need distinct ordered steps and the consumed/next B1 IDs")
    quantized_prompt_range(length)
    replay_ids = sample.generated_ids[: steps[-1] + 2]
    targets = (HeadIntervention(model_layers[0], 0, "key", None),) + tuple(
        HeadIntervention(layer, head, side, bits)
        for layer in model_layers
        for head in range(heads)
        for side in SIDES
        for bits in BIT_WIDTHS
    )
    writer = QuantizationResultWriter(
        directory,
        sample_id=record["sample_id"],
        prompt_length=length,
        num_layers=layers,
        query_heads=query_heads,
        kv_heads=heads,
        head_dim=dim,
        local_steps=steps,
        targets=targets,
        source=source or {"batch_id": sample.batch_id, "run_id": sample.run_id},
        execution_dtype=str(next(model.parameters()).dtype).removeprefix("torch."),
    )

    def write_local(local):
        writer.write_local(local)
        if local.layer_id == layers - 1:
            emit_progress(
                "v3b_local_checkpoint_completed", sample_id=record["sample_id"], decode_step=local.decode_step
            )

    def write_checkpoint(checkpoint):
        writer.write_checkpoint(checkpoint)
        emit_progress(
            "v3b_checkpoint_completed",
            sample_id=record["sample_id"],
            decode_step=checkpoint.decode_step,
            quantized_model_branches=sum(b.target.bits is not None for b in checkpoint.branches),
        )

    collector = QuantizationSensitivityCollector(length, write_local, block_rows)
    device = torch.device(device)
    timer = GenerationPhaseTimer(model, device)
    started = time.perf_counter()
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        with timer:
            result = replay_quantization_checkpoints(
                model,
                device=device,
                sample_id=record["sample_id"],
                prompt_ids=record["input_ids"],
                replay_generated_ids=replay_ids,
                checkpoints={step: targets if step == steps[0] else () for step in steps},
                block_rows=block_rows,
                on_attention=collector.on_attention,
                on_checkpoint=write_checkpoint,
            )
        timing = timer.summary()
        counts = {
            "prefill": 1,
            "main_decode": len(replay_ids) - 1,
            "checkpoint_reference": len(steps),
            "quantized_branches": len(targets) - 1,
            "bypass_branches": 1,
        }
        if (
            timing["prefill"]["count"] != 1
            or timing["decode"]["count"] != sum(counts.values()) - 1
            or timing["incomplete_forward_count"] != 0
        ):
            raise ValueError("root Forward counts differ from the main/checkpoint/branch plan")
        resources = {
            "device": str(device),
            "wall_seconds": time.perf_counter() - started,
            "forward_counts": counts,
            "forward_timing": timing,
            "scope": "shadow replay, local statistics, branches and synchronous writing; not deployed compressed Cache",
            "local_quantization_cases": len(steps) * layers * heads * len(SIDES) * len(BIT_WIDTHS),
        }
        if device.type == "cuda":
            resources.update(
                peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        return writer.finalize(replay_ids, result, resources)
    except Exception as exc:
        writer.fail(exc)
        raise


def _frozen_cohorts(bundle):
    by_id = {row["sample_id"]: row for row in read_json(Path(bundle) / "split.json")["samples"]}
    for role, ids in (("calib", PILOT_IDS), ("validation", VALIDATION_IDS)):
        if any(by_id[sid]["split"] != role for sid in ids):
            raise ValueError("V3-B sample IDs differ from the frozen V3 split")
        pg = [by_id[sid]["source_id"] for sid in ids if sid.startswith("pg19_")]
        if len(pg) != 2 or pg[0] != pg[1]:
            raise ValueError("PG19 lengths must remain grouped by source")
    if {by_id[s]["source_id"] for s in PILOT_IDS} & {by_id[s]["source_id"] for s in VALIDATION_IDS}:
        raise ValueError("Calib and Validation sources overlap")
    return by_id


def run_collection(args):
    directory = Path(args.output_dir)
    if directory.exists():
        raise FileExistsError(f"output already exists: {directory}")
    by_id = _frozen_cohorts(args.bundle)
    ids = PILOT_IDS if args.cohort == "calib" else VALIDATION_IDS
    calib = None
    if args.cohort == "validation":
        if args.calib_dir is None:
            raise ValueError("Validation requires --calib-dir with completed frozen Calib outputs")
        calib = read_json(Path(args.calib_dir) / "run_metadata.json")
        if (
            calib["status"] != "completed"
            or calib["cohort"] != "calib"
            or calib["completed_sample_ids"] != list(PILOT_IDS)
        ):
            raise ValueError("Calib is incomplete or uses different samples")
        with np.load(Path(args.calib_dir) / "calib_profile.npz") as archive:
            frozen_shapes = {name: archive[name].shape for name in ("local_output_nmse", "prediction_kl")}
    index = load_b1_replay_index(
        args.run_set, source_root=args.source_root, expected_run_set_sha256=None, expected_data_sha256=None
    )
    _validate_execution_contract(index, args)
    samples = load_replay_samples(index, ids)
    if [s.frozen_record["sample_id"] for s in samples] != list(ids):
        raise ValueError("replay selection differs from the planned sample order")
    for sample in samples:
        quantized_prompt_range(frozen_length(sample.frozen_record))
        steps = LOCAL_STEPS[sample.frozen_record["dataset"]]
        if len(sample.generated_ids) < steps[-1] + 2:
            raise ValueError("B1 history is shorter than the frozen V3-B checkpoints")
    config = {
        "schema_version": "headkv-v3b-quantization-v1",
        "model": args.model,
        "tokenizer": args.tokenizer,
        "dtype": args.dtype,
        "attention_backend": "sdpa",
        "bundle": str(Path(args.bundle).resolve()),
        "run_set": str(Path(args.run_set).resolve()),
        "source_root": str(Path(args.source_root).resolve()),
        "model_layers": list(MODEL_LAYERS),
        "local_steps": {k: list(v) for k, v in LOCAL_STEPS.items()},
        "sides": list(SIDES),
        "bit_widths": list(BIT_WIDTHS),
        "group_size": GROUP_SIZE,
        "protected_prompt_tokens_each_end": PROTECTED_TOKENS,
        "middle_residual": "BF16",
        "decode_kv": "BF16",
        "group_statistics_dtype": "float32",
        "key_representation": "post_rope",
        "group_axes": {"key": "tokens", "value": "features"},
        "rounding": "round_to_even",
        "code_order": "token-major; low nibble first for 4-bit codes",
        "negative_kl_tolerance": KL_NEGATIVE_TOLERANCE,
        "block_rows": args.block_rows,
        "calib_samples": [by_id[s] for s in PILOT_IDS],
        "validation_samples": [by_id[s] for s in VALIDATION_IDS],
    }
    if calib is not None and config != read_json(Path(args.calib_dir) / "config.json"):
        raise ValueError("Validation configuration differs from frozen Calib")
    emit_progress("v3b_model_loading", model=args.model, device=args.device, cohort=args.cohort)
    model, device = load_model(args.model, args.device, args.dtype)
    layers, queries, heads = model_dimensions(model)
    dim = model.model.layers[0].self_attn.head_dim
    if max(MODEL_LAYERS) >= layers or dim % 32:
        raise ValueError("model geometry cannot execute the fixed layers/group size")
    geometry = {"num_layers": layers, "query_heads": queries, "kv_heads": heads, "head_dim": dim}
    if calib is not None and (
        calib["geometry"] != geometry or any(shape != (2, 2, layers, heads) for shape in frozen_shapes.values())
    ):
        raise ValueError("model/profile geometry differs from frozen Calib")
    directory.mkdir(parents=True)
    write_json(directory / "config.json", config)
    state = {
        "status": "running",
        "cohort": args.cohort,
        "target_sample_ids": list(ids),
        "completed_sample_ids": [],
        "geometry": geometry,
        "device": str(device),
        "calib_directory": str(Path(args.calib_dir).resolve()) if calib is not None else None,
        "torch_version": torch.__version__,
    }
    write_json(directory / "run_metadata.json", state)
    started = time.perf_counter()
    try:
        for sample in samples:
            sid = sample.frozen_record["sample_id"]
            emit_progress(
                "v3b_sample_started",
                sample_id=sid,
                completed_samples=len(state["completed_sample_ids"]),
                target_samples=len(ids),
            )
            collect_quantization_sample(
                model,
                device=device,
                sample=sample,
                directory=directory / "samples" / sid,
                source={**by_id[sid], "batch_id": sample.batch_id, "run_id": sample.run_id},
                block_rows=args.block_rows,
            )
            state["completed_sample_ids"].append(sid)
            write_json(directory / "run_metadata.json", state, overwrite=True)
            emit_progress("v3b_sample_completed", sample_id=sid, elapsed_seconds=time.perf_counter() - started)
        if args.cohort == "calib":
            profiles = [read_sample_profiles(directory / "samples" / sid) for sid in ids]
            # Exactly one fixed sample per condition: this is six-condition equal weighting.
            if len({by_id[s]["condition"] for s in ids}) != len(ids):
                raise ValueError("Calib profile expects one sample per frozen condition")
            frozen_profile = {}
            for name in profiles[0]:
                mean, count = finite_mean_count(np.stack([p[name] for p in profiles]), 0, count_dtype=np.uint8)
                frozen_profile[name] = mean
                frozen_profile[f"{name}_valid_condition_count"] = count
            write_npz(directory / "calib_profile.npz", frozen_profile)
        write_json(
            directory / "summary.json",
            {
                "cohort": args.cohort,
                "sample_ids": list(ids),
                "sample_summaries": {sid: f"samples/{sid}/summary.json" for sid in ids},
                "wall_seconds": time.perf_counter() - started,
                "wall_scope": "sample loop/profile/writing, excluding model loading",
                "scope": "fixed-input single-Head sensitivity; no combined compression, task-quality or hardware claim",
            },
        )
        state["status"] = "completed"
        write_json(directory / "run_metadata.json", state, overwrite=True)
        emit_progress("v3b_run_completed", cohort=args.cohort, completed_samples=len(ids))
        return 0
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(directory / "run_metadata.json", state, overwrite=True)
        emit_progress("v3b_run_failed", error=state["error"])
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("bundle", "run-set", "model", "tokenizer", "output-dir"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--cohort", choices=("calib", "validation"), default="calib")
    parser.add_argument("--calib-dir")
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--block-rows", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.block_rows <= 0 or args.block_rows % 32:
        parser.error("--block-rows must be a positive multiple of 32")
    return run_collection(args)
