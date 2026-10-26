"""Validate fixed prompts and measure ordinary Decode behavior without KV hooks.

The fixed dataset stores the exact rendered chat prompt and the corresponding
``input_ids``.  These experiments intentionally call ``model.generate`` with
those saved IDs directly: they do not re-template inputs, install an Observer,
or apply a Press.

experiment-a verifies the saved IDs, tokenizer/template round trips, and a
short ordinary generation probe for a stratified 12-sample subset.  experiment-b
runs the same subset using each record's Decode budget and records the actual
length, EOS behavior, repetition indicators, and task-specific diagnostics.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SMOKE_BUCKETS = (
    ("ruler", 8192),
    ("ruler", 32768),
    ("govreport", 8192),
    ("govreport", 16384),
    ("pg19", 8192),
    ("pg19", 32768),
)


def fail(message: str) -> None:
    raise ValueError(message)


def digest(data: str | bytes) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def ids_digest(ids: list[int]) -> str:
    return digest(json.dumps(ids, separators=(",", ":")))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"JSONL parse error {path}:{line_no}: {exc}")
            if not isinstance(record, dict):
                fail(f"JSONL record {path}:{line_no} is not an object")
            yield record


def write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(f"Refusing to overwrite {path}; use --overwrite for an existing report")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def saved_input_length(record: dict[str, Any]) -> int | None:
    actual = record.get("actual_prompt_length")
    legacy = record.get("input_length")
    if actual is not None and legacy is not None and actual != legacy:
        return None
    length = actual if actual is not None else legacy
    return length if isinstance(length, int) else None


def select_smoke_records(records: list[dict[str, Any]], per_bucket: int) -> list[dict[str, Any]]:
    if per_bucket <= 0:
        fail("--samples-per-bucket must be positive")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (record.get("dataset"), record.get("target_prompt_length"))
        if key in SMOKE_BUCKETS:
            grouped[key].append(record)

    selected = []
    for bucket in SMOKE_BUCKETS:
        candidates = sorted(grouped[bucket], key=lambda record: str(record.get("sample_id", "")))
        if len(candidates) < per_bucket:
            fail(f"{bucket[0]} target {bucket[1]}: need {per_bucket} samples, found {len(candidates)}")
        selected.extend(candidates[:per_bucket])

    sample_ids = [record.get("sample_id") for record in selected]
    if len(sample_ids) != len(set(sample_ids)):
        fail("selected smoke records contain duplicate sample_id values")
    return selected


def render_saved_template(tokenizer, record: dict[str, Any]) -> str:
    content = record.get("user_content")
    assistant_prefix = record.get("assistant_prefix", "")
    if not isinstance(content, str) or not isinstance(assistant_prefix, str):
        fail("missing user_content or assistant_prefix needed for template check")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        fail("chat_template did not return text")
    return rendered + assistant_prefix


def validate_saved_record(tokenizer, record: dict[str, Any]) -> dict[str, Any]:
    sample_id = record.get("sample_id", "<missing>")
    errors = []
    token_ids = record.get("input_ids")
    prompt = record.get("prompt")
    input_length = saved_input_length(record)
    if not isinstance(token_ids, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in token_ids
    ):
        errors.append("input_ids is not a list of integer token IDs")
        token_ids = []
    if not isinstance(prompt, str):
        errors.append("prompt is not text")
        prompt = ""
    if input_length is None:
        errors.append("actual_prompt_length/input_length is missing or inconsistent")
    elif input_length != len(token_ids):
        errors.append(f"saved length {input_length} differs from input_ids length {len(token_ids)}")
    vocab_size = len(tokenizer)
    if any(token < 0 or token >= vocab_size for token in token_ids):
        errors.append(f"input_ids contains a token outside tokenizer vocabulary size {vocab_size}")
    if record.get("input_sha256") != ids_digest(token_ids):
        errors.append("input_sha256 does not match saved input_ids")
    try:
        reencoded = tokenizer.encode(prompt, add_special_tokens=False)
        if reencoded != token_ids:
            errors.append("tokenizer.encode(prompt) does not match saved input_ids")
    except Exception as exc:  # tokenizer-specific errors are part of the report
        errors.append(f"tokenizer.encode(prompt) failed: {type(exc).__name__}: {exc}")
    try:
        expected_prompt = render_saved_template(tokenizer, record)
        if expected_prompt != prompt:
            errors.append("saved prompt differs from current chat-template rendering")
    except Exception as exc:  # template failures should be recorded rather than hidden
        errors.append(f"chat-template rendering failed: {type(exc).__name__}: {exc}")
    return {
        "sample_id": sample_id,
        "dataset": record.get("dataset"),
        "task": record.get("task"),
        "target_prompt_length": record.get("target_prompt_length"),
        "input_length": input_length,
        "max_new_tokens": record.get("max_new_tokens"),
        "input_sha256": record.get("input_sha256"),
        "input_integrity_passed": not errors,
        "input_integrity_errors": errors,
    }


def resolve_dtype(name: str):
    if name == "auto":
        return None
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model(model_path: str, device_name: str, dtype_name: str, attn_implementation: str):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        fail(f"CUDA device requested ({device_name}) but torch.cuda.is_available() is false")
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "attn_implementation": attn_implementation,
    }
    dtype = resolve_dtype(dtype_name)
    if dtype is not None:
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.to(device)
    model.eval()
    return model, device


def eos_ids(model, tokenizer) -> set[int]:
    values = getattr(model.generation_config, "eos_token_id", None)
    if values is None:
        values = getattr(tokenizer, "eos_token_id", None)
    if isinstance(values, int):
        return {values}
    if isinstance(values, (list, tuple)):
        return {value for value in values if isinstance(value, int)}
    return set()


def max_repeated_run(ids: list[int]) -> int:
    if not ids:
        return 0
    longest = current = 1
    for previous, current_id in zip(ids, ids[1:]):
        if previous == current_id:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def repeated_ngram_fraction(ids: list[int], n: int = 4) -> float:
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[index : index + n]) for index in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def generation_metrics(
    record: dict[str, Any],
    generated_ids: list[int],
    generated_text: str,
    requested_new_tokens: int,
    eos_token_ids: set[int],
    pg19_min_new_tokens: int,
    repeat_ngram_threshold: float,
    unique_token_threshold: float,
    task_diagnostics: bool,
) -> dict[str, Any]:
    actual = len(generated_ids)
    ends_with_eos = bool(generated_ids and generated_ids[-1] in eos_token_ids)
    unique_ratio = len(set(generated_ids)) / actual if actual else 0.0
    repeated_fraction = repeated_ngram_fraction(generated_ids)
    anomalies = []
    if actual == 0:
        anomalies.append("empty_generation")
    if actual < requested_new_tokens and ends_with_eos:
        anomalies.append("early_eos")
    if actual >= 32 and (repeated_fraction >= repeat_ngram_threshold or unique_ratio <= unique_token_threshold):
        anomalies.append("high_token_repetition")
    if task_diagnostics and record.get("dataset") == "pg19" and actual < pg19_min_new_tokens:
        anomalies.append("pg19_shorter_than_min_decode")
    ruler_answer_match = None
    if task_diagnostics and record.get("dataset") == "ruler":
        answers = record.get("reference_answer")
        if isinstance(answers, list) and len(answers) == 1 and isinstance(answers[0], str):
            ruler_answer_match = normalize_text(answers[0]) in normalize_text(generated_text)
            if not ruler_answer_match:
                anomalies.append("ruler_answer_not_found")
    return {
        "requested_new_tokens": requested_new_tokens,
        "actual_new_tokens": actual,
        "ended_with_eos": ends_with_eos,
        "ended_before_limit": actual < requested_new_tokens,
        "unique_token_ratio": unique_ratio,
        "max_repeated_token_run": max_repeated_run(generated_ids),
        "repeated_4gram_fraction": repeated_fraction,
        "ruler_answer_match": ruler_answer_match,
        "generated_text": generated_text,
        "generated_text_sha256": digest(generated_text),
        "anomalies": anomalies,
    }


@torch.inference_mode()
def generate_record(
    model,
    tokenizer,
    device: torch.device,
    record: dict[str, Any],
    requested_new_tokens: int,
    pg19_min_new_tokens: int,
    repeat_ngram_threshold: float,
    unique_token_threshold: float,
    task_diagnostics: bool,
) -> dict[str, Any]:
    token_ids = record["input_ids"]
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = model.generate(
        input_ids=input_tensor,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=requested_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    generated_ids = output[0, input_tensor.shape[1] :].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    metrics = generation_metrics(
        record,
        generated_ids,
        generated_text,
        requested_new_tokens,
        eos_ids(model, tokenizer),
        pg19_min_new_tokens,
        repeat_ngram_threshold,
        unique_token_threshold,
        task_diagnostics,
    )
    metrics["generation_seconds"] = elapsed_seconds
    metrics["decode_tokens_per_second"] = len(generated_ids) / elapsed_seconds if elapsed_seconds else None
    return metrics


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_metadata(args, tokenizer, model=None) -> dict[str, Any]:
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "input": str(Path(args.input)),
        "tokenizer": str(args.tokenizer),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "model": str(args.model),
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "generation": {"do_sample": False, "use_cache": True},
    }
    if model is not None:
        metadata["model_class"] = type(model).__name__
        metadata["model_context_limit"] = getattr(model.config, "max_position_embeddings", None)
    return metadata


def run_experiment(args) -> int:
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        fail(f"Refusing to overwrite {output}; use --overwrite for an existing report")
    input_path = Path(args.input)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_bytes = input_path.read_bytes()
    if manifest.get("data_sha256") != digest(input_bytes):
        fail("input checksum differs from manifest data_sha256; refuse to run an unfrozen prompt set")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    selected = select_smoke_records(list(iter_jsonl(input_path)), args.samples_per_bucket)
    input_checks = [validate_saved_record(tokenizer, record) for record in selected]
    report: dict[str, Any] = {
        "experiment": args.command,
        "metadata": run_metadata(args, tokenizer),
        "selection": {
            "samples_per_bucket": args.samples_per_bucket,
            "buckets": [{"dataset": dataset, "target_prompt_length": target} for dataset, target in SMOKE_BUCKETS],
            "sample_ids": [record["sample_id"] for record in selected],
        },
        "dataset_manifest_sha256": digest(manifest_path.read_bytes()),
        "dataset_data_sha256": manifest["data_sha256"],
        "input_checks": input_checks,
        "input_checks_passed": all(check["input_integrity_passed"] for check in input_checks),
        "results": [],
    }
    if not report["input_checks_passed"]:
        report["passed"] = False
        report["error"] = "saved prompt/input_ids consistency checks failed; model was not loaded"
        write_json(output, report, args.overwrite)
        return 1
    if args.check_only:
        report["model_generation_skipped"] = True
        report["passed"] = True
        write_json(output, report, args.overwrite)
        return 0

    model, device = load_model(args.model, args.device, args.dtype, args.attn_implementation)
    report["metadata"] = run_metadata(args, tokenizer, model)
    for record, check in zip(selected, input_checks):
        requested = args.probe_max_new_tokens if args.command == "experiment-a" else record["max_new_tokens"]
        if args.decode_cap is not None and args.command == "experiment-b":
            requested = min(requested, args.decode_cap)
        result = dict(check)
        result["model_generation_passed"] = False
        if not isinstance(requested, int) or requested <= 0:
            result["generation_error"] = f"invalid requested_new_tokens: {requested!r}"
            report["results"].append(result)
            continue
        try:
            result.update(
                generate_record(
                    model,
                    tokenizer,
                    device,
                    record,
                    requested,
                    args.pg19_min_new_tokens,
                    args.repeat_ngram_threshold,
                    args.unique_token_threshold,
                    args.command == "experiment-b",
                )
            )
            result["model_generation_passed"] = True
        except Exception as exc:  # retain evidence for all successful prior samples
            result["generation_error"] = f"{type(exc).__name__}: {exc}"
        report["results"].append(result)
    report["summary"] = summarize_results(report["results"])
    report["passed"] = report["input_checks_passed"] and all(
        result["model_generation_passed"] for result in report["results"]
    )
    write_json(output, report, args.overwrite)
    return 0 if report["passed"] else 1


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_dataset[str(result.get("dataset"))].append(result)
    summary = {}
    for dataset, rows in sorted(by_dataset.items()):
        completed = [row for row in rows if row.get("model_generation_passed")]
        summary[dataset] = {
            "samples": len(rows),
            "generation_successes": len(completed),
            "actual_new_tokens": [row.get("actual_new_tokens") for row in completed],
            "anomaly_counts": dict(Counter(anomaly for row in completed for anomaly in row.get("anomalies", []))),
        }
        if dataset == "ruler":
            matches = [row.get("ruler_answer_match") for row in completed]
            summary[dataset]["ruler_answer_matches"] = sum(match is True for match in matches)
            summary[dataset]["ruler_answer_mismatches"] = sum(match is False for match in matches)
    return summary


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="fixed_v2 prompts.jsonl")
    parser.add_argument("--manifest", required=True, help="matching fixed_v2 manifest.json used to freeze --input")
    parser.add_argument("--tokenizer", required=True, help="local tokenizer snapshot used by the fixed dataset")
    parser.add_argument("--model", required=True, help="local model snapshot; no Hub download is attempted")
    parser.add_argument("--device", required=True, help="explicit device, for example cuda:0 or cpu")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa", help="Transformers attention implementation")
    parser.add_argument("--output", required=True, help="new JSON report path outside fixed_v2")
    parser.add_argument("--samples-per-bucket", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--check-only", action="store_true", help="validate frozen prompt/ID consistency without loading model"
    )
    parser.add_argument("--pg19-min-new-tokens", type=int, default=512)
    parser.add_argument("--repeat-ngram-threshold", type=float, default=0.6)
    parser.add_argument("--unique-token-threshold", type=float, default=0.2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    experiment_a = commands.add_parser("experiment-a", help="saved prompt/ID checks plus short normal generation")
    add_common_arguments(experiment_a)
    experiment_a.add_argument("--probe-max-new-tokens", type=int, default=8)
    experiment_a.set_defaults(decode_cap=None)
    experiment_b = commands.add_parser("experiment-b", help="actual Decode-length smoke test for the same 12 records")
    add_common_arguments(experiment_b)
    experiment_b.add_argument("--decode-cap", type=int, help="optional cap; default uses each record's max_new_tokens")
    experiment_b.set_defaults(probe_max_new_tokens=None)
    args = parser.parse_args(argv)
    if args.command == "experiment-a" and args.probe_max_new_tokens <= 0:
        parser.error("--probe-max-new-tokens must be positive")
    if args.command == "experiment-b" and args.check_only:
        parser.error("--check-only is only meaningful for experiment-a")
    if args.decode_cap is not None and args.decode_cap <= 0:
        parser.error("--decode-cap must be positive")
    if args.command == "experiment-b" and args.decode_cap is not None and args.decode_cap < args.pg19_min_new_tokens:
        parser.error("--decode-cap cannot be below --pg19-min-new-tokens for experiment-b")
    try:
        sys.exit(run_experiment(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
