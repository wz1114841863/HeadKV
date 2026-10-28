"""Generate frozen B1 inputs without a Probe and persist a comparison artifact.

This runner deliberately shares frozen-input and model-loading contracts with
``b1_runner`` but does not import or attach ``AttentionProbe``.  Its output is
an immutable, self-checking artifact for a later collected run's exact greedy
ID comparison; it is not an attention trace.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoTokenizer

from experiments.head_analysis.runners.b1_runner import (
    B1RunnerError,
    eos_token_ids,
    frozen_length,
    generated_ids_sha256,
    generation_stop_reason,
    git_state,
    load_frozen_selection,
    load_model,
    requested_generation_budget,
    sha256_bytes,
)
from experiments.head_analysis.runners.resource_metrics import (
    GenerationPhaseTimer,
    measurement_contract,
    validate_standard_greedy_timing,
)


BASELINE_SCHEMA_VERSION = "headkv-baseline-decode-v1"


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    """Atomically create a baseline artifact and refuse to overwrite one."""

    if path.exists():
        raise B1RunnerError(f"baseline output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # link(2) is atomic and, unlike replace(2), cannot overwrite an
            # existing completed baseline artifact.
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise B1RunnerError(f"baseline output already exists: {path}") from exc
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def run_baseline(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    if output_path.exists():
        raise B1RunnerError(f"baseline output already exists: {output_path}")
    selected, manifest = load_frozen_selection(input_path, manifest_path, args.sample_id)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    model, device = load_model(args.model, args.device, args.dtype)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    eos_ids = eos_token_ids(model, tokenizer)
    sample_entries: list[dict[str, Any]] = []
    for record in selected:
        budget = requested_generation_budget(record, args.decode_cap)
        input_tensor = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        phase_timer = GenerationPhaseTimer(model, device)
        with phase_timer, torch.inference_mode():
            output = model.generate(
                input_ids=input_tensor,
                attention_mask=torch.ones_like(input_tensor),
                do_sample=False,
                num_beams=1,
                max_new_tokens=budget,
                use_cache=True,
                pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        prompt_length = frozen_length(record)
        if output.ndim != 2 or output.shape[0] != 1 or not torch.equal(output[0, :prompt_length], input_tensor[0]):
            raise B1RunnerError("generation output does not preserve the frozen input prefix")
        generated_ids = output[0, prompt_length:].tolist()
        timing = phase_timer.summary()
        validate_standard_greedy_timing(timing, len(generated_ids))
        entry: dict[str, Any] = {
            "sample_id": record["sample_id"],
            "input_sha256": record["input_sha256"],
            "prompt_length": prompt_length,
            "generation_max_new_tokens_requested": budget,
            "generated_ids": generated_ids,
            "generated_ids_sha256": generated_ids_sha256(generated_ids),
            "generation_stop_reason": generation_stop_reason(generated_ids, budget, eos_ids),
            "elapsed_seconds": elapsed,
            "generation_end_to_end_seconds": elapsed,
            "forward_timing": timing,
        }
        if device.type == "cuda":
            entry["peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            entry["peak_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        sample_entries.append(entry)

    payload = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "fixed_dataset": {
            "path": str(input_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
            "data_sha256": manifest["data_sha256"],
        },
        "model": str(args.model),
        "tokenizer": str(args.tokenizer),
        "attention_backend": "sdpa",
        "device": str(device),
        "dtype": args.dtype,
        "generation": {"do_sample": False, "num_beams": 1, "use_cache": True, "decode_cap": args.decode_cap},
        "resource_measurement": measurement_contract(),
        "seed": args.seed,
        "samples": sample_entries,
    }
    write_json_new(output_path, payload)
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="frozen prompts.jsonl")
    parser.add_argument("--manifest", required=True, help="matching frozen manifest.json")
    parser.add_argument("--tokenizer", required=True, help="local tokenizer snapshot")
    parser.add_argument("--model", required=True, help="local model snapshot")
    parser.add_argument("--output", required=True, help="new baseline JSON artifact")
    parser.add_argument(
        "--sample-id", action="append", required=True, help="one frozen sample ID; repeat for a small run"
    )
    parser.add_argument("--device", required=True, help="explicit device, for example cuda:4")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--decode-cap", type=int, help="explicit temporary cap; never implicit")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    if args.decode_cap is not None and args.decode_cap <= 0:
        raise SystemExit("--decode-cap must be positive")
    try:
        raise SystemExit(run_baseline(args))
    except (B1RunnerError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
