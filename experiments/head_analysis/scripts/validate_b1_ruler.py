"""Validate RULER retrieval answers from a completed B1 run without rerunning a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from experiments.head_analysis.analysis.page_rco import PageRCOError, write_json_new
from experiments.head_analysis.analysis.task_checks import TaskCheckError, evaluate_ruler_answer
from experiments.head_analysis.runners.b1_runner import generated_ids_sha256, iter_jsonl, validate_frozen_record


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_run(input_path: Path, manifest_path: Path, tokenizer_path: str, run_dir: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("data_sha256") != sha256_bytes(input_path.read_bytes()):
        raise TaskCheckError("input checksum differs from manifest data_sha256")
    records: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl(input_path):
        validate_frozen_record(record)
        records[record["sample_id"]] = record
    run_metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if run_metadata.get("status") != "completed":
        raise TaskCheckError("B1 run is not completed")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
    checks: list[dict[str, Any]] = []
    for sample_dir in sorted((run_dir / "samples").iterdir()):
        metadata = json.loads((sample_dir / "sample_metadata.json").read_text(encoding="utf-8"))
        sample_id = metadata.get("sample_id")
        record = records.get(sample_id)
        if record is None or record.get("dataset") != "ruler":
            continue
        if metadata.get("status") != "completed" or metadata.get("input_sha256") != record["input_sha256"]:
            raise TaskCheckError(f"{sample_id}: completed-status or frozen-input provenance mismatch")
        generated = json.loads((sample_dir / "generated_ids.json").read_text(encoding="utf-8"))
        ids = generated.get("ids")
        if not isinstance(ids, list) or generated.get("sha256") != generated_ids_sha256(ids):
            raise TaskCheckError(f"{sample_id}: generated ID artifact checksum mismatch")
        result = evaluate_ruler_answer(record, ids, tokenizer)
        result.update(
            {
                "generated_length": len(ids),
                "generated_ids_sha256": generated["sha256"],
                "decode_trace_length": metadata.get("decode_trace_length"),
            }
        )
        checks.append(result)
    if not checks:
        raise TaskCheckError("run contains no completed RULER samples")
    return {
        "schema_version": "headkv-v1-b1-ruler-check-v1",
        "source_run": str(run_dir),
        "source_run_metadata_sha256": sha256_bytes((run_dir / "run_metadata.json").read_bytes()),
        "checks": checks,
        "all_answer_matches": all(bool(check["answer_match"]) for check in checks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True, help="new JSON report; never overwrites")
    args = parser.parse_args()
    try:
        write_json_new(
            Path(args.output),
            validate_run(Path(args.input), Path(args.manifest), args.tokenizer, Path(args.run_dir)),
        )
    except (TaskCheckError, PageRCOError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
