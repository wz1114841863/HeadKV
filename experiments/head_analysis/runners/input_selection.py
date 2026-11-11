"""Selection and validation of fixed input records shared by experiment runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence


class B1RunnerError(RuntimeError):
    """Raised when the fixed-input B1 execution contract is violated."""


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
    if not isinstance(record.get("max_new_tokens"), int) or record["max_new_tokens"] <= 0:
        raise B1RunnerError(f"{sample_id}: max_new_tokens must be positive")


def load_frozen_selection(
    input_path: Path, manifest_path: Path, sample_ids: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise B1RunnerError("input manifest must be a JSON object")
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
