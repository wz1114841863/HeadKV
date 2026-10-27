"""Run B1 collection on explicit frozen ``input_ids`` without retokenizing.

This runner is deliberately conservative.  It accepts an explicit sample list,
checks the frozen JSONL against its manifest, uses greedy batch-one generation,
and composes the downstream Probe, Collector, and Writer.  It does not apply a
Press, alter cache state, or silently substitute an attention backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.head_analysis.collectors.attention_stats import AttentionCollectorConfig, AttentionStatsCollector
from experiments.head_analysis.runtime.attention_probe import AttentionProbe
from experiments.head_analysis.storage.trace_writer import (
    AttentionTraceSchema,
    AttentionTraceWriter,
    SampleWriteRequest,
)


class B1RunnerError(RuntimeError):
    """Raised when the fixed-input B1 execution contract is violated."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def input_ids_sha256(ids: Sequence[int]) -> str:
    return sha256_bytes(json.dumps(list(ids), separators=(",", ":")).encode("utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise B1RunnerError(f"JSONL parse error {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise B1RunnerError(f"JSONL record {path}:{line_number} is not an object")
            yield record


def frozen_length(record: dict[str, Any]) -> int:
    actual = record.get("actual_prompt_length")
    legacy = record.get("input_length")
    if actual is not None and legacy is not None and actual != legacy:
        raise B1RunnerError(f"{record.get('sample_id')}: actual_prompt_length and input_length disagree")
    length = actual if actual is not None else legacy
    if not isinstance(length, int) or length <= 0:
        raise B1RunnerError(f"{record.get('sample_id')}: missing positive frozen prompt length")
    return length


def validate_frozen_record(record: dict[str, Any]) -> None:
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise B1RunnerError("fixed record has no non-empty sample_id")
    input_ids = record.get("input_ids")
    if not isinstance(input_ids, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in input_ids
    ):
        raise B1RunnerError(f"{sample_id}: input_ids is not a list of integer IDs")
    if frozen_length(record) != len(input_ids):
        raise B1RunnerError(f"{sample_id}: frozen prompt length differs from input_ids length")
    if record.get("input_sha256") != input_ids_sha256(input_ids):
        raise B1RunnerError(f"{sample_id}: input_sha256 does not match frozen input_ids")
    if not isinstance(record.get("max_new_tokens"), int) or record["max_new_tokens"] <= 0:
        raise B1RunnerError(f"{sample_id}: max_new_tokens must be positive")


def load_frozen_selection(
    input_path: Path, manifest_path: Path, sample_ids: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("data_sha256"), str):
        raise B1RunnerError("manifest is missing data_sha256")
    if sha256_bytes(input_path.read_bytes()) != manifest["data_sha256"]:
        raise B1RunnerError("input checksum differs from manifest data_sha256; refuse unfrozen prompts")
    requested = list(sample_ids)
    if not requested or any(not sample_id for sample_id in requested):
        raise B1RunnerError("at least one non-empty --sample-id is required")
    if len(set(requested)) != len(requested):
        raise B1RunnerError("duplicate --sample-id values are not allowed")
    by_id = {}
    for record in iter_jsonl(input_path):
        validate_frozen_record(record)
        sample_id = record["sample_id"]
        if sample_id in by_id:
            raise B1RunnerError(f"fixed input contains duplicate sample_id {sample_id}")
        by_id[sample_id] = record
    missing = [sample_id for sample_id in requested if sample_id not in by_id]
    if missing:
        raise B1RunnerError(f"requested sample_id is absent from frozen input: {missing}")
    return [by_id[sample_id] for sample_id in requested], manifest


def resolve_dtype(name: str) -> torch.dtype | None:
    if name == "auto":
        return None
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_model(model_path: str, device_name: str, dtype_name: str):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise B1RunnerError(f"CUDA device requested ({device_name}) but torch.cuda.is_available() is false")
    kwargs: dict[str, Any] = {"local_files_only": True, "attn_implementation": "sdpa"}
    dtype = resolve_dtype(dtype_name)
    if dtype is not None:
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.to(device)
    model.eval()
    return model, device


def git_state() -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"commit": command("git", "rev-parse", "HEAD"), "dirty": bool(command("git", "status", "--porcelain"))}


def model_dimensions(model) -> tuple[int, int, int]:
    config = model.config
    try:
        return int(config.num_hidden_layers), int(config.num_attention_heads), int(config.num_key_value_heads)
    except (AttributeError, TypeError, ValueError) as exc:
        raise B1RunnerError("model config lacks valid Llama layer/query/KV-head dimensions") from exc


def eos_token_ids(model, tokenizer) -> set[int]:
    values = getattr(model.generation_config, "eos_token_id", None)
    if values is None:
        values = getattr(tokenizer, "eos_token_id", None)
    if isinstance(values, int):
        return {values}
    if isinstance(values, (tuple, list)):
        return {value for value in values if isinstance(value, int)}
    return set()


def requested_generation_budget(record: dict[str, Any], decode_cap: int | None) -> int:
    budget = int(record["max_new_tokens"])
    if decode_cap is not None:
        budget = min(budget, decode_cap)
    if budget <= 0:
        raise B1RunnerError("requested generation budget must be positive")
    return budget


def sample_metadata(record: dict[str, Any], *, attention_backend: str, requested_budget: int) -> dict[str, Any]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return {
        "sample_id": record["sample_id"],
        "dataset": record.get("dataset"),
        "task": record.get("task"),
        "pair_id": record.get("pair_id", metadata.get("pair_id")),
        "input_sha256": record["input_sha256"],
        "prompt_length": frozen_length(record),
        "max_new_tokens": record["max_new_tokens"],
        "generation_max_new_tokens_requested": requested_budget,
        "attention_backend": attention_backend,
    }


def generation_stop_reason(generated_ids: Sequence[int], requested_budget: int, eos_ids: set[int]) -> str:
    if generated_ids and generated_ids[-1] in eos_ids:
        return "eos"
    if len(generated_ids) >= requested_budget:
        return "max_new_tokens"
    return "stopped_without_eos"


def run_collection(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    manifest_path = Path(args.manifest)
    selected, manifest = load_frozen_selection(input_path, manifest_path, args.sample_id)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    model, device = load_model(args.model, args.device, args.dtype)
    num_layers, num_query_heads, num_kv_heads = model_dimensions(model)
    if num_query_heads % num_kv_heads:
        raise B1RunnerError("model has non-integral GQA grouping")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    collector_config = AttentionCollectorConfig(
        key_block_size=args.key_block_size,
        prefill_window=args.prefill_window,
        token_topk=args.token_topk,
        save_token_topk=args.save_token_topk,
        page_size=args.page_size,
        page_topk=args.page_topk,
    )
    schema = AttentionTraceSchema(
        num_layers=num_layers,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        prefill_window=args.prefill_window,
        page_size=args.page_size,
        token_topk=args.token_topk,
        page_topk=args.page_topk,
        save_token_topk=args.save_token_topk,
    )
    writer = AttentionTraceWriter(args.run_dir, schema, decode_chunk_steps=args.decode_chunk_steps)
    started_at = datetime.now(timezone.utc).isoformat()
    writer.initialize(
        {
            "run_id": Path(args.run_dir).name,
            "created_at_utc": started_at,
            "status": "running",
            "git": git_state(),
            "fixed_dataset": {
                "path": str(input_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
                "data_sha256": manifest["data_sha256"],
            },
            "model": str(args.model),
            "model_class": type(model).__name__,
            "tokenizer": str(args.tokenizer),
            "tokenizer_class": type(tokenizer).__name__,
            "attention_backend": "sdpa",
            "device": str(device),
            "dtype": args.dtype,
            "generation": {"do_sample": False, "num_beams": 1, "use_cache": True},
        },
        config_snapshot={
            "schema_version": schema.schema_version,
            "collection": {
                "key_block_size": args.key_block_size,
                "prefill_window": args.prefill_window,
                "token_topk": args.token_topk,
                "save_token_topk": args.save_token_topk,
                "page_size": args.page_size,
                "page_topk": args.page_topk,
                "decode_chunk_steps": args.decode_chunk_steps,
            },
            "selected_sample_ids": list(args.sample_id),
            "decode_cap": args.decode_cap,
            "seed": args.seed,
        },
    )

    successes = 0
    failures = 0
    resource_records: list[dict[str, Any]] = []
    eos_ids = eos_token_ids(model, tokenizer)
    for record in selected:
        budget = requested_generation_budget(record, args.decode_cap)
        collector = AttentionStatsCollector(collector_config)
        collected_records = []
        prompt_length = frozen_length(record)
        input_tensor = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        try:

            def collect_callback(context) -> None:
                collected_records.append(collector.collect(context))

            with AttentionProbe(
                sample_id=record["sample_id"], prompt_length=prompt_length, on_attention=collect_callback
            )(model):
                output = model.generate(
                    input_ids=input_tensor,
                    attention_mask=torch.ones_like(input_tensor),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=budget,
                    use_cache=True,
                    pad_token_id=(
                        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                    ),
                )
        except Exception as exc:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            failure_metadata = sample_metadata(record, attention_backend="sdpa", requested_budget=budget)
            failure_metadata["error"] = f"{type(exc).__name__}: {exc}"
            writer.write_sample(
                SampleWriteRequest(
                    sample_metadata=failure_metadata,
                    generated_ids=[],
                    prefill_records=[],
                    decode_records=[],
                    eos_reached=False,
                    generation_stop_reason="error",
                    status="failed",
                )
            )
            failures += 1
            resource_records.append({"sample_id": record["sample_id"], "status": "failed", "elapsed_seconds": elapsed})
            continue

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if output.ndim != 2 or output.shape[0] != 1 or not torch.equal(output[0, :prompt_length], input_tensor[0]):
            raise B1RunnerError("generation output does not preserve the frozen input prefix")
        generated_ids = output[0, prompt_length:].tolist()
        prefill_records = [item for item in collected_records if item.phase.value == "prefill"]
        decode_records = [item for item in collected_records if item.phase.value == "decode"]
        writer.write_sample(
            SampleWriteRequest(
                sample_metadata=sample_metadata(record, attention_backend="sdpa", requested_budget=budget),
                generated_ids=generated_ids,
                prefill_records=prefill_records,
                decode_records=decode_records,
                eos_reached=bool(generated_ids and generated_ids[-1] in eos_ids),
                generation_stop_reason=generation_stop_reason(generated_ids, budget, eos_ids),
            )
        )
        successes += 1
        resource = {"sample_id": record["sample_id"], "status": "completed", "elapsed_seconds": elapsed}
        if device.type == "cuda":
            resource["peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            resource["peak_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        resource_records.append(resource)

    final_status = "completed" if failures == 0 else "failed" if successes == 0 else "partial"
    writer.finalize(
        status=final_status,
        resource_summary={"samples_completed": successes, "samples_failed": failures, "per_sample": resource_records},
    )
    return 0 if failures == 0 else 1


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="frozen prompts.jsonl")
    parser.add_argument("--manifest", required=True, help="matching frozen manifest.json")
    parser.add_argument("--tokenizer", required=True, help="local tokenizer snapshot")
    parser.add_argument("--model", required=True, help="local model snapshot")
    parser.add_argument("--run-dir", required=True, help="new empty B1 run directory")
    parser.add_argument(
        "--sample-id", action="append", required=True, help="one frozen sample ID; repeat for a small run"
    )
    parser.add_argument("--device", required=True, help="explicit device, for example cuda:4")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--key-block-size", type=int, default=1024)
    parser.add_argument("--prefill-window", type=int, default=64)
    parser.add_argument("--token-topk", type=int, default=64)
    parser.add_argument("--no-save-token-topk", action="store_false", dest="save_token_topk")
    parser.set_defaults(save_token_topk=True)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--page-topk", type=int, default=32)
    parser.add_argument("--decode-chunk-steps", type=int, default=128)
    parser.add_argument("--decode-cap", type=int, help="explicit temporary cap recorded in metadata; never implicit")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    if args.decode_cap is not None and args.decode_cap <= 0:
        raise SystemExit("--decode-cap must be positive")
    try:
        raise SystemExit(run_collection(args))
    except (B1RunnerError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
