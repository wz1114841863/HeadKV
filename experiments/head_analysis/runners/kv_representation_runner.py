"""Collect V2 representations while replaying frozen B1 inputs and tokens."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Sequence

import torch

from experiments.head_analysis.collectors.kv_representation import (
    KVRepresentationCollector, KVRepresentationConfig, sample_positions,
)
from experiments.head_analysis.common.io import write_json
from experiments.head_analysis.runners.b2_runner import (
    B1ReplaySample, _environment_metadata, _validate_execution_contract, emit_progress,
    load_b1_replay_index, load_replay_samples, teacher_forced_replay,
)
from experiments.head_analysis.runners.input_selection import frozen_length
from experiments.head_analysis.runners.resource_metrics import (
    GenerationPhaseTimer, validate_standard_greedy_timing,
)
from experiments.head_analysis.runtime.attention_probe import AttentionProbe
from experiments.head_analysis.runtime.model import load_model, model_dimensions
from experiments.head_analysis.storage.kv_representation import KVRepresentationWriter


def replay_config(sample: B1ReplaySample, block_size: int = 1024) -> KVRepresentationConfig:
    """Use the approved six-forward RULER / 256-forward long-text protocol."""
    dataset = sample.frozen_record["dataset"]
    if dataset == "ruler":
        steps = (5,)
    elif dataset in ("govreport", "pg19"):
        steps = (31, 255)
    else:
        raise ValueError(f"unsupported V2 dataset: {dataset!r}")
    # Forward s consumes ids[s] and predicts ids[s+1]; one extra ID is required.
    required = steps[-1] + 2
    if len(sample.generated_ids) < required:
        raise ValueError(f"{sample.frozen_record['sample_id']}: replay requires {required} B1 generated IDs; "
                         f"got {len(sample.generated_ids)}")
    sample_positions(frozen_length(sample.frozen_record))
    return KVRepresentationConfig(block_size=block_size, decode_steps=steps)


def collect_sample(model, *, device, sample: B1ReplaySample, directory: Path,
                   block_size: int = 1024) -> dict:
    """Compose the existing Probe, Collector, replay loop and layer-wise Writer."""
    device, directory = torch.device(device), Path(directory)
    record = sample.frozen_record
    config = replay_config(sample, block_size)
    prompt_length = frozen_length(record)
    replay_ids = sample.generated_ids[:config.decode_steps[-1] + 2]
    num_layers, query_heads, kv_heads = model_dimensions(model)
    writer = KVRepresentationWriter(directory, sample_id=record["sample_id"],
                                    prompt_length=prompt_length, num_layers=num_layers, config=config)
    collector = KVRepresentationCollector(prompt_length, writer.write_record, config)
    timer = GenerationPhaseTimer(model, device)
    metadata = {
        "sample_id": record["sample_id"], "status": "running", "dataset": record["dataset"],
        "task": record.get("task"),
        "pair_id": record.get("pair_id", record.get("metadata", {}).get("pair_id")),
        "input_sha256": record.get("input_sha256"), "prompt_length": prompt_length,
        "num_layers": num_layers, "num_query_heads": query_heads, "num_kv_heads": kv_heads,
        "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        "checkpoints": list(config.decode_steps), "decode_forward_count": len(replay_ids) - 1,
        "source_b1": {"batch_id": sample.batch_id, "run_id": sample.run_id,
                      "generated_length": len(sample.generated_ids)},
        "timing_scope": "replay and root forwards include synchronous collection and NPZ writing",
    }
    write_json(directory / "sample_metadata.json", metadata)
    # Retain the actual forced inputs even when a subsequent Top-1 check fails.
    write_json(directory / "replay.json", {"generated_ids": list(replay_ids)})
    started = time.perf_counter()
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        probe = AttentionProbe(sample_id=record["sample_id"], prompt_length=prompt_length,
                               on_key_projection=collector.on_key_projection,
                               on_attention=collector.on_attention)
        with timer, probe(model):
            result = teacher_forced_replay(
                model, device=device, sample_id=record["sample_id"],
                prompt_ids=record["input_ids"], replay_generated_ids=replay_ids,
                require_top1_match=True,
            )
        timing = timer.summary()
        validate_standard_greedy_timing(timing, len(replay_ids))
        write_json(directory / "replay.json", {
            "generated_ids": list(replay_ids),
            "prefill_predicted_token_id": result.prefill_predicted_token_id,
            "decode_predicted_token_ids": list(result.decode_predicted_token_ids),
        }, overwrite=True)
        metadata.update(top1_match=True, forward_timing=timing,
                        replay_wall_seconds=time.perf_counter() - started)
        if device.type == "cuda":
            metadata.update(peak_memory_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                            peak_memory_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
        # A full checkpoint file set alone does not prove the final prediction matched.
        writer.finalize()
        metadata["status"] = "completed"
        write_json(directory / "sample_metadata.json", metadata, overwrite=True)
        return metadata
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(directory / "sample_metadata.json", metadata, overwrite=True)
        raise
    finally:
        collector.clear()


def run_collection(args: argparse.Namespace) -> int:
    directory = Path(args.run_dir)
    if directory.exists():
        raise FileExistsError(f"output already exists: {directory}")
    emit_progress("v2_preflight_started", target_samples=len(args.sample_id))
    index = load_b1_replay_index(args.run_set, source_root=args.source_root,
                                expected_run_set_sha256=None, expected_data_sha256=None)
    _validate_execution_contract(index, args)
    samples = load_replay_samples(index, args.sample_id)
    for sample in samples:
        replay_config(sample, args.block_size)
    emit_progress("v2_model_loading", model=str(args.model), device=args.device, dtype=args.dtype)
    model, device = load_model(args.model, args.device, args.dtype)
    directory.mkdir(parents=True)
    metadata = {
        "schema_version": "headkv-v2-kv-run-v1", "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_b1_run_set": str(Path(args.run_set).resolve()),
        "source_root": str(index.source_root), "fixed_dataset": dict(index.payload["fixed_dataset"]),
        "model": str(args.model), "tokenizer": str(args.tokenizer), "dtype": args.dtype,
        "model_class": type(model).__name__, "environment": _environment_metadata(device),
        "device": str(device), "attention_backend": "sdpa", "block_size": args.block_size,
        "teacher_forced_replay": True, "target_sample_ids": list(args.sample_id),
        "completed_sample_ids": [],
    }
    write_json(directory / "run_metadata.json", metadata)
    for sample in samples:
        sample_id = sample.frozen_record["sample_id"]
        emit_progress("v2_sample_started", sample_id=sample_id,
                      prompt_length=frozen_length(sample.frozen_record),
                      b1_generated_length=len(sample.generated_ids))
        try:
            result = collect_sample(model, device=device, sample=sample,
                                    directory=directory / "samples" / sample_id, block_size=args.block_size)
        except Exception as exc:
            metadata.update(status="failed", failed_sample_id=sample_id, error=f"{type(exc).__name__}: {exc}")
            write_json(directory / "run_metadata.json", metadata, overwrite=True)
            emit_progress("v2_sample_failed", sample_id=sample_id, error=metadata["error"])
            return 1
        metadata["completed_sample_ids"].append(sample_id)
        write_json(directory / "run_metadata.json", metadata, overwrite=True)
        emit_progress("v2_sample_completed", sample_id=sample_id,
                      decode_forward_count=result["decode_forward_count"],
                      replay_wall_seconds=round(result["replay_wall_seconds"], 6))
    metadata["status"] = "completed"
    write_json(directory / "run_metadata.json", metadata, overwrite=True)
    emit_progress("v2_run_completed", completed_samples=len(metadata["completed_sample_ids"]))
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-set", required=True, help="completed B1 run-set manifest")
    parser.add_argument("--source-root", default=".", help="root of B1 paths and frozen inputs")
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--model", required=True, help="same local model snapshot as B1")
    parser.add_argument("--tokenizer", required=True, help="B1 tokenizer path; inputs are not retokenized")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--run-dir", required=True, help="new output directory; existing data is preserved")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run_collection(make_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
