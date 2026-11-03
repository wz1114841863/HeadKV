"""Replay frozen B1 token histories and collect compact V1-B2 traces.

The runner never retokenizes a Prompt and never chooses a new Decode branch.
It validates the completed B1 run set, performs one complete Prefill followed
by teacher-forced single-token Decode forwards, and fails the sample closed if
any Top-1 token differs from the authoritative B1 trajectory.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import transformers
from transformers import AutoTokenizer

from experiments.head_analysis.analysis.b1_offline import (
    B1OfflineError,
    RunBatch,
    RunSetIndex,
    load_completed_run_set,
    sha256_file,
)
from experiments.head_analysis.collectors.b2_attention import (
    B2AttentionCollector,
    B2DecodeLayerRecord,
    B2PrefillLayerRecord,
)
from experiments.head_analysis.collectors.prefill_consistency import B2CollectorConfig
from experiments.head_analysis.runners.b1_runner import (
    frozen_length,
    generated_ids_sha256,
    git_state,
    load_frozen_selection,
    load_model,
    model_dimensions,
)
from experiments.head_analysis.runners.resource_metrics import (
    GenerationPhaseTimer,
    measurement_contract,
    validate_standard_greedy_timing,
)
from experiments.head_analysis.runtime.attention_probe import AttentionProbe
from experiments.head_analysis.storage.b2_trace_writer import (
    B2SampleWriteRequest,
    B2TraceSchema,
    B2TraceWriter,
    collector_config_sha256,
)


FROZEN_B1_RUN_SET_SHA256 = "039afecab0135169ccac0e96050a414b3e313d37d6108d7d648cae80bbcbd85d"
FROZEN_DATA_SHA256 = "30ea0f8a85e203c8cf63a87e83805e851345c42a2a2fd9ab7f4dbdc1c63492ee"


class B2RunnerError(RuntimeError):
    """Raised when replay provenance or execution differs from B2 v1."""


def emit_progress(event: str, **fields: Any) -> None:
    """Emit one machine-readable progress line for tmux/log observation."""

    print(json.dumps({"event": event, **fields}, ensure_ascii=False, sort_keys=True), flush=True)


@dataclass(frozen=True)
class B1ReplaySample:
    """One frozen Prompt plus its hash-bound B1 generated-token trajectory."""

    frozen_record: Mapping[str, Any]
    batch_id: str
    run_id: str
    generated_ids: tuple[int, ...]
    generated_ids_sha256: str
    generated_ids_file_sha256: str


@dataclass(frozen=True)
class TeacherForcedReplayResult:
    """Compact replay outputs; optional logits exist only for small-model tests."""

    prefill_predicted_token_id: int
    decode_predicted_token_ids: tuple[int, ...]
    prefill_records: tuple[B2PrefillLayerRecord, ...]
    decode_records: tuple[B2DecodeLayerRecord, ...]
    logits: tuple[torch.Tensor, ...] | None = None

    @property
    def all_predicted_token_ids(self) -> tuple[int, ...]:
        return (self.prefill_predicted_token_id, *self.decode_predicted_token_ids)


def _last_token_logits(outputs) -> torch.Tensor:
    logits = getattr(outputs, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1:
        raise B2RunnerError("causal-LM forward did not return batch-one token logits")
    return logits[0, -1]


def teacher_forced_replay(
    model,
    *,
    device: torch.device | str,
    sample_id: str,
    prompt_ids: Sequence[int],
    replay_generated_ids: Sequence[int],
    collector: B2AttentionCollector | None = None,
    require_top1_match: bool = True,
    capture_logits: bool = False,
) -> TeacherForcedReplayResult:
    """Execute Prompt Prefill plus ``G_out - 1`` forced Decode forwards."""

    device = torch.device(device)
    prompt = tuple(int(value) for value in prompt_ids)
    replay = tuple(int(value) for value in replay_generated_ids)
    if not sample_id or not prompt:
        raise B2RunnerError("sample_id and Prompt IDs must be non-empty")
    if not replay:
        raise B2RunnerError("B2 requires at least one B1 generated token")
    if any(value < 0 for value in (*prompt, *replay)):
        raise B2RunnerError("Prompt and replay token IDs must be non-negative")

    prefill_records: list[B2PrefillLayerRecord] = []
    decode_records: list[B2DecodeLayerRecord] = []
    captured_logits: list[torch.Tensor] | None = [] if capture_logits else None

    def collect_callback(context) -> None:
        if collector is None:
            raise B2RunnerError("Attention callback was installed without a B2 collector")
        record = collector.collect(context)
        if isinstance(record, B2PrefillLayerRecord):
            prefill_records.append(record)
        else:
            decode_records.append(record)

    probe = None
    if collector is not None:
        collector.clear()
        probe = AttentionProbe(
            sample_id=sample_id,
            prompt_length=len(prompt),
            on_attention=collect_callback,
        )

    def execute() -> tuple[int, tuple[int, ...]]:
        prompt_tensor = torch.tensor([prompt], dtype=torch.long, device=device)
        prompt_positions = torch.arange(len(prompt), dtype=torch.long, device=device)
        with torch.inference_mode():
            outputs = model(
                input_ids=prompt_tensor,
                use_cache=True,
                cache_position=prompt_positions,
                logits_to_keep=1,
            )
            logits = _last_token_logits(outputs)
            if captured_logits is not None:
                captured_logits.append(logits.detach().cpu())
            prefill_prediction = int(torch.argmax(logits).item())
            if require_top1_match and prefill_prediction != replay[0]:
                raise B2RunnerError(
                    f"{sample_id}: Prefill Top-1 {prefill_prediction} differs from B1 token {replay[0]}"
                )
            cache = getattr(outputs, "past_key_values", None)
            if cache is None:
                raise B2RunnerError("teacher-forced replay requires a returned KV cache")

            predictions: list[int] = []
            for step, input_token in enumerate(replay[:-1]):
                position = len(prompt) + step
                outputs = model(
                    input_ids=torch.tensor([[input_token]], dtype=torch.long, device=device),
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=torch.tensor([position], dtype=torch.long, device=device),
                    logits_to_keep=1,
                )
                logits = _last_token_logits(outputs)
                if captured_logits is not None:
                    captured_logits.append(logits.detach().cpu())
                prediction = int(torch.argmax(logits).item())
                expected = replay[step + 1]
                if require_top1_match and prediction != expected:
                    raise B2RunnerError(
                        f"{sample_id}: Decode step {step} Top-1 {prediction} differs from B1 token {expected}"
                    )
                predictions.append(prediction)
                cache = getattr(outputs, "past_key_values", None)
                if cache is None:
                    raise B2RunnerError(f"{sample_id}: Decode step {step} did not return a KV cache")
        return prefill_prediction, tuple(predictions)

    try:
        if probe is None:
            prefill_prediction, predictions = execute()
        else:
            with probe(model):
                prefill_prediction, predictions = execute()
    finally:
        if collector is not None:
            collector.clear()

    return TeacherForcedReplayResult(
        prefill_predicted_token_id=prefill_prediction,
        decode_predicted_token_ids=predictions,
        prefill_records=tuple(prefill_records),
        decode_records=tuple(decode_records),
        logits=None if captured_logits is None else tuple(captured_logits),
    )


def load_b1_replay_index(
    run_set_path: Path | str,
    *,
    source_root: Path | str,
    expected_run_set_sha256: str | None = FROZEN_B1_RUN_SET_SHA256,
    expected_data_sha256: str | None = FROZEN_DATA_SHA256,
) -> RunSetIndex:
    """Load the full completed B1 index without rereading 12 GiB of B1 arrays."""

    actual_hash = sha256_file(run_set_path)
    if expected_run_set_sha256 is not None and actual_hash != expected_run_set_sha256:
        raise B2RunnerError("B1 run-set SHA-256 differs from the frozen B2 authority")
    try:
        index = load_completed_run_set(
            run_set_path,
            source_root=source_root,
            verify_file_hashes=False,
            verify_trace_arrays=False,
        )
    except B1OfflineError as exc:
        raise B2RunnerError(f"B1 run-set validation failed: {exc}") from exc
    data_hash = index.payload.get("fixed_dataset", {}).get("data_sha256")
    if expected_data_sha256 is not None and data_hash != expected_data_sha256:
        raise B2RunnerError("B1 run set uses a different frozen dataset SHA-256")
    return index


def _resolve_from_root(source_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise B2RunnerError(f"{label} must be a non-empty path")
    path = Path(value)
    return (path if path.is_absolute() else source_root / path).resolve()


def _sample_batches(index: RunSetIndex) -> dict[str, RunBatch]:
    result: dict[str, RunBatch] = {}
    for batch in index.batches:
        for sample_id in batch.sample_ids:
            if sample_id in result:
                raise B2RunnerError(f"B1 sample {sample_id} appears in multiple batches")
            result[sample_id] = batch
    return result


def load_replay_samples(index: RunSetIndex, sample_ids: Sequence[str]) -> list[B1ReplaySample]:
    """Load selected Prompt IDs and independently verify each B1 ID artifact."""

    fixed = index.payload.get("fixed_dataset")
    if not isinstance(fixed, dict):
        raise B2RunnerError("B1 run set is missing fixed_dataset")
    prompts_path = _resolve_from_root(index.source_root, fixed.get("prompts_path"), "prompts_path")
    manifest_path = _resolve_from_root(index.source_root, fixed.get("manifest_path"), "manifest_path")
    selected, _ = load_frozen_selection(prompts_path, manifest_path, sample_ids)
    batches = _sample_batches(index)
    result: list[B1ReplaySample] = []
    for record in selected:
        sample_id = record["sample_id"]
        batch = batches.get(sample_id)
        if batch is None:
            raise B2RunnerError(f"B1 run set has no batch for {sample_id}")
        relative_path = f"samples/{sample_id}/generated_ids.json"
        entry = batch.manifest_files.get(relative_path)
        if not isinstance(entry, Mapping):
            raise B2RunnerError(f"B1 file manifest is missing {relative_path}")
        path = batch.run_dir / relative_path
        actual_file_hash = sha256_file(path)
        if actual_file_hash != entry.get("sha256"):
            raise B2RunnerError(f"{sample_id}: B1 generated-ID file SHA-256 drifted")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise B2RunnerError(f"{sample_id}: unable to read B1 generated IDs: {exc}") from exc
        ids = payload.get("ids") if isinstance(payload, dict) else None
        if (
            not isinstance(ids, list)
            or not ids
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ids)
        ):
            raise B2RunnerError(f"{sample_id}: B1 generated IDs are not a non-empty integer list")
        ids_hash = generated_ids_sha256(ids)
        if payload.get("dtype") != "uint32-little-endian" or payload.get("sha256") != ids_hash:
            raise B2RunnerError(f"{sample_id}: B1 generated-ID payload hash/schema is invalid")
        sample_metadata_path = batch.run_dir / "samples" / sample_id / "sample_metadata.json"
        sample_metadata = json.loads(sample_metadata_path.read_text(encoding="utf-8"))
        if (
            sample_metadata.get("status") != "completed"
            or sample_metadata.get("input_sha256") != record["input_sha256"]
            or sample_metadata.get("actual_generated_length") != len(ids)
        ):
            raise B2RunnerError(f"{sample_id}: B1 sample metadata differs from Prompt/ID artifacts")
        result.append(
            B1ReplaySample(
                frozen_record=record,
                batch_id=batch.batch_id,
                run_id=str(batch.run_metadata.get("run_id", batch.run_dir.name)),
                generated_ids=tuple(ids),
                generated_ids_sha256=ids_hash,
                generated_ids_file_sha256=actual_file_hash,
            )
        )
    return result


def _validate_execution_contract(index: RunSetIndex, args: argparse.Namespace) -> None:
    execution = index.payload.get("execution_contract")
    if not isinstance(execution, dict):
        raise B2RunnerError("B1 run set is missing execution_contract")
    model_contract = execution.get("model")
    if isinstance(model_contract, dict):
        observed_model = model_contract.get("path")
        observed_dtype = model_contract.get("dtype")
        observed_tokenizer = model_contract.get("tokenizer_path")
    else:
        observed_model = execution.get("model_snapshot")
        observed_dtype = execution.get("dtype")
        observed_tokenizer = None
    expected = {
        "model": (str(args.model), observed_model),
        "dtype": (args.dtype, observed_dtype),
        "attention_backend": ("sdpa", execution.get("attention_backend")),
    }
    for name, (requested, observed) in expected.items():
        if requested != observed:
            raise B2RunnerError(f"B2 {name}={requested!r} differs from B1 {observed!r}")
    if observed_tokenizer is not None and str(args.tokenizer) != observed_tokenizer:
        raise B2RunnerError("B2 tokenizer differs from the B1 execution contract")
    for batch in index.batches:
        if batch.run_metadata.get("tokenizer") != str(args.tokenizer):
            raise B2RunnerError(f"B2 tokenizer differs from B1 batch {batch.batch_id}")


def _sample_metadata(sample: B1ReplaySample, index: RunSetIndex) -> dict[str, Any]:
    record = sample.frozen_record
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return {
        "sample_id": record["sample_id"],
        "dataset": record.get("dataset"),
        "task": record.get("task"),
        "pair_id": record.get("pair_id", metadata.get("pair_id")),
        "input_sha256": record["input_sha256"],
        "prompt_length": frozen_length(record),
        "max_new_tokens": record["max_new_tokens"],
        "attention_backend": "sdpa",
        "b1_run_set_sha256": index.run_set_sha256,
        "b1_batch_id": sample.batch_id,
        "b1_run_id": sample.run_id,
        "b1_generated_ids_sha256": sample.generated_ids_sha256,
        "b1_generated_ids_file_sha256": sample.generated_ids_file_sha256,
    }


def _environment_metadata(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        result["gpu_name"] = torch.cuda.get_device_name(device)
        result["gpu_compute_capability"] = list(torch.cuda.get_device_capability(device))
    return result


def _canonical_file_hash(path: Path) -> str:
    return sha256_file(path)


def run_collection(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root).resolve()
    emit_progress("b2_preflight_started", target_samples=len(args.sample_id))
    index = load_b1_replay_index(
        args.run_set,
        source_root=source_root,
        expected_run_set_sha256=args.expected_run_set_sha256,
        expected_data_sha256=args.expected_data_sha256,
    )
    _validate_execution_contract(index, args)
    samples = load_replay_samples(index, args.sample_id)
    emit_progress("b2_provenance_validated", target_samples=len(samples))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    emit_progress("b2_model_loading", model=str(args.model), device=args.device, dtype=args.dtype)
    model, device = load_model(args.model, args.device, args.dtype)
    num_layers, num_query_heads, num_kv_heads = model_dimensions(model)
    config = B2CollectorConfig(key_block_size=args.key_block_size)
    schema = B2TraceSchema(
        num_layers=num_layers,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        collector_config=config,
    )
    writer = B2TraceWriter(args.run_dir, schema, decode_chunk_steps=args.decode_chunk_steps)
    fixed = index.payload["fixed_dataset"]
    fixed_manifest_path = _resolve_from_root(index.source_root, fixed["manifest_path"], "manifest_path")
    writer.initialize(
        {
            "run_id": Path(args.run_dir).name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "git": git_state(),
            "b1_replay_authority": {
                "path": str(Path(args.run_set)),
                "sha256": index.run_set_sha256,
            },
            "fixed_dataset": {
                "data_sha256": fixed["data_sha256"],
                "manifest_path": str(fixed_manifest_path),
                "manifest_sha256": _canonical_file_hash(fixed_manifest_path),
            },
            "model": str(args.model),
            "model_class": type(model).__name__,
            "tokenizer": str(args.tokenizer),
            "tokenizer_class": type(tokenizer).__name__,
            "device": str(device),
            "dtype": args.dtype,
            "attention_backend": "sdpa",
            "environment": _environment_metadata(device),
            "resource_measurement": measurement_contract(),
            "teacher_forced_replay": True,
            "verify_no_probe": bool(args.verify_no_probe),
            "target_sample_ids": list(args.sample_id),
        },
        config_snapshot={
            "selected_sample_ids": list(args.sample_id),
            "decode_chunk_steps": args.decode_chunk_steps,
            "verify_no_probe": bool(args.verify_no_probe),
            "seed": args.seed,
        },
    )

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    resources: list[dict[str, Any]] = []
    successes = 0
    failures = 0
    run_started = time.perf_counter()
    for sample in samples:
        record = sample.frozen_record
        metadata = _sample_metadata(sample, index)
        baseline_resource: dict[str, Any] | None = None
        sample_started = time.perf_counter()
        emit_progress(
            "b2_sample_started",
            sample_id=record["sample_id"],
            prompt_length=frozen_length(record),
            replay_length=len(sample.generated_ids),
            verify_no_probe=bool(args.verify_no_probe),
        )
        try:
            if args.verify_no_probe:
                baseline_timer = GenerationPhaseTimer(model, device)
                baseline_started = time.perf_counter()
                with baseline_timer:
                    baseline = teacher_forced_replay(
                        model,
                        device=device,
                        sample_id=record["sample_id"],
                        prompt_ids=record["input_ids"],
                        replay_generated_ids=sample.generated_ids,
                        require_top1_match=True,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                baseline_timing = baseline_timer.summary()
                validate_standard_greedy_timing(baseline_timing, len(sample.generated_ids))
                baseline_resource = {
                    "wall_seconds": time.perf_counter() - baseline_started,
                    "forward_timing": baseline_timing,
                    "predicted_ids_sha256": generated_ids_sha256(baseline.all_predicted_token_ids),
                }
                emit_progress(
                    "b2_no_probe_completed",
                    sample_id=record["sample_id"],
                    wall_seconds=round(float(baseline_resource["wall_seconds"]), 6),
                )

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            collector = B2AttentionCollector(config)
            timer = GenerationPhaseTimer(model, device)
            replay_started = time.perf_counter()
            with timer:
                result = teacher_forced_replay(
                    model,
                    device=device,
                    sample_id=record["sample_id"],
                    prompt_ids=record["input_ids"],
                    replay_generated_ids=sample.generated_ids,
                    collector=collector,
                    require_top1_match=True,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            replay_wall = time.perf_counter() - replay_started
            timing = timer.summary()
            validate_standard_greedy_timing(timing, len(sample.generated_ids))
            if args.verify_no_probe and baseline.all_predicted_token_ids != result.all_predicted_token_ids:
                raise B2RunnerError(f"{record['sample_id']}: Probe changed teacher-forced Top-1 outputs")
            write_started = time.perf_counter()
            write_result = writer.write_sample(
                B2SampleWriteRequest(
                    sample_metadata=metadata,
                    replay_generated_ids=sample.generated_ids,
                    prefill_predicted_token_id=result.prefill_predicted_token_id,
                    prefill_records=result.prefill_records,
                    decode_records=result.decode_records,
                    decode_predicted_token_ids=result.decode_predicted_token_ids,
                )
            )
            write_wall = time.perf_counter() - write_started
            resource = {
                "sample_id": record["sample_id"],
                "status": "completed",
                "teacher_forced_replay_wall_seconds": replay_wall,
                "forward_timing": timing,
                "no_probe_replay": baseline_resource,
                "trace_write_wall_seconds": write_wall,
                "trace_storage": write_result.to_dict(),
                "sample_end_to_end_seconds": time.perf_counter() - sample_started,
            }
            if device.type == "cuda":
                resource["peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
                resource["peak_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
            resources.append(resource)
            successes += 1
            emit_progress(
                "b2_sample_completed",
                sample_id=record["sample_id"],
                replay_wall_seconds=round(replay_wall, 6),
                write_wall_seconds=round(write_wall, 6),
                artifact_file_bytes=write_result.artifact_file_bytes,
                peak_memory_reserved_bytes=resource.get("peak_memory_reserved_bytes"),
            )
        except Exception as exc:
            failures += 1
            metadata["error"] = f"{type(exc).__name__}: {exc}"
            metadata["completed_stage"] = "replay_or_write_failed"
            try:
                writer.write_sample(
                    B2SampleWriteRequest(
                        sample_metadata=metadata,
                        replay_generated_ids=sample.generated_ids,
                        prefill_predicted_token_id=None,
                        prefill_records=(),
                        decode_records=(),
                        decode_predicted_token_ids=(),
                        status="failed",
                    )
                )
            except Exception as write_exc:
                metadata["failure_metadata_write_error"] = f"{type(write_exc).__name__}: {write_exc}"
            resources.append(
                {
                    "sample_id": record["sample_id"],
                    "status": "failed",
                    "error": metadata["error"],
                    "sample_end_to_end_seconds": time.perf_counter() - sample_started,
                    "no_probe_replay": baseline_resource,
                }
            )
            emit_progress(
                "b2_sample_failed",
                sample_id=record["sample_id"],
                error=metadata["error"],
            )
            if args.stop_on_failure:
                break

    if failures == 0 and successes == len(samples):
        final_status = "completed"
    elif successes:
        final_status = "partial"
    else:
        final_status = "failed"
    completed_storage = [entry["trace_storage"] for entry in resources if entry.get("status") == "completed"]
    writer.finalize(
        status=final_status,
        resource_summary={
            "status": final_status,
            "samples_targeted": len(samples),
            "samples_completed": successes,
            "samples_failed": failures,
            "run_wall_seconds": time.perf_counter() - run_started,
            "per_sample": resources,
            "completed_trace_storage_totals": {
                "logical_array_bytes": sum(int(entry["logical_array_bytes"]) for entry in completed_storage),
                "artifact_file_bytes": sum(int(entry["artifact_file_bytes"]) for entry in completed_storage),
            },
            "collector_config_sha256": collector_config_sha256(config),
        },
    )
    emit_progress(
        "b2_run_completed",
        status=final_status,
        samples_completed=successes,
        samples_failed=failures,
        run_wall_seconds=round(time.perf_counter() - run_started, 6),
        run_dir=str(args.run_dir),
    )
    return 0 if final_status == "completed" else 1


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-set", required=True, help="completed B1 run-set manifest")
    parser.add_argument("--source-root", default=".", help="root for relative paths in the B1 run set")
    parser.add_argument("--model", required=True, help="same local model snapshot used by B1")
    parser.add_argument("--tokenizer", required=True, help="same local tokenizer snapshot used by B1")
    parser.add_argument("--run-dir", required=True, help="new empty B2 run directory")
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--device", required=True, help="explicit device, for example cuda:4")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--key-block-size", type=int, default=1024)
    parser.add_argument("--decode-chunk-steps", type=int, default=128)
    parser.add_argument("--verify-no-probe", action="store_true", help="run an independent replay before B2")
    parser.add_argument("--continue-on-failure", action="store_false", dest="stop_on_failure")
    parser.set_defaults(stop_on_failure=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-run-set-sha256", default=FROZEN_B1_RUN_SET_SHA256)
    parser.add_argument("--expected-data-sha256", default=FROZEN_DATA_SHA256)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    if args.key_block_size <= 0 or not 1 <= args.decode_chunk_steps <= 128:
        raise SystemExit("--key-block-size must be positive and --decode-chunk-steps must be in [1,128]")
    try:
        raise SystemExit(run_collection(args))
    except (B2RunnerError, B1OfflineError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()


__all__ = [
    "B1ReplaySample",
    "B2RunnerError",
    "FROZEN_B1_RUN_SET_SHA256",
    "FROZEN_DATA_SHA256",
    "TeacherForcedReplayResult",
    "load_b1_replay_index",
    "load_replay_samples",
    "run_collection",
    "teacher_forced_replay",
]
