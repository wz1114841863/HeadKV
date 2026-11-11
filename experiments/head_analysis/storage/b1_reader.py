"""Read and validate existing B1 run sets and Page RCO reports."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.head_analysis.collectors.attention_stats import FEATURE_NAMES
from experiments.head_analysis.common.io import canonical_json_sha256, read_json
from experiments.head_analysis.common.io import sha256_file as file_sha256
from experiments.head_analysis.storage.trace_writer import PAGE_SCOPES, SCHEMA_VERSION

RUN_SET_SCHEMA_VERSION = "headkv-v1-b1-run-set-v1"
PAGE_RCO_SCHEMA_VERSION = "headkv-v1-b1-page-rco-v1"
DEFAULT_PAGE_K = (8, 16, 32)
DEFAULT_DELTAS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
FROZEN_DISTANCE_BINS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


class B1OfflineError(RuntimeError):
    """Raised when an input cannot enter the frozen B1 offline analysis."""


@dataclass(frozen=True)
class FrozenSample:
    sample_id: str
    dataset: str
    task: str
    pair_id: str | None
    target_prompt_length: int
    actual_prompt_length: int
    input_sha256: str
    condition: str


@dataclass(frozen=True)
class RunBatch:
    batch_id: str
    run_dir: Path
    sample_ids: tuple[str, ...]
    run_metadata: Mapping[str, Any]
    schema: Mapping[str, Any]
    manifest_files: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class RunSetIndex:
    manifest_path: Path
    source_root: Path
    run_set_sha256: str
    payload: Mapping[str, Any]
    samples: Mapping[str, FrozenSample]
    batches: tuple[RunBatch, ...]

    @property
    def condition_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for sample in self.samples.values():
            counts[sample.condition] += 1
        return dict(sorted(counts.items()))


@dataclass(frozen=True)
class ValidationReport:
    source_run_set_sha256: str
    sample_count: int
    batch_count: int
    condition_counts: Mapping[str, int]
    page_rco_reports: tuple[Mapping[str, Any], ...]


def sha256_file(path: Path | str) -> str:
    try:
        return file_sha256(path)
    except OSError as exc:
        raise B1OfflineError(f"unable to hash {path}: {exc}") from exc


def _read_json(path: Path) -> Any:
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise B1OfflineError(f"unable to read JSON {path}: {exc}") from exc


def _resolve(source_root: Path, value: str, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise B1OfflineError(f"{label} must be a non-empty path string")
    path = Path(value)
    return (path if path.is_absolute() else source_root / path).resolve()


def _length_label(target: int) -> str:
    if target <= 0:
        raise B1OfflineError("target_prompt_length must be positive")
    return f"{target // 1024}k" if target % 1024 == 0 else str(target)


def _load_frozen_samples(prompts_path: Path) -> dict[str, FrozenSample]:
    samples: dict[str, FrozenSample] = {}
    try:
        lines = prompts_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise B1OfflineError(f"unable to read frozen prompts {prompts_path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise B1OfflineError(f"{prompts_path}:{line_number}: invalid JSON: {exc}") from exc
        required = (
            "sample_id",
            "dataset",
            "task",
            "target_prompt_length",
            "actual_prompt_length",
            "input_sha256",
        )
        missing = [name for name in required if name not in record]
        if missing:
            raise B1OfflineError(f"{prompts_path}:{line_number}: missing fields {missing}")
        sample_id = str(record["sample_id"])
        if not sample_id or sample_id in samples:
            raise B1OfflineError(f"{prompts_path}:{line_number}: duplicate or empty sample_id {sample_id!r}")
        dataset = str(record["dataset"])
        target = int(record["target_prompt_length"])
        actual = int(record["actual_prompt_length"])
        if actual <= 0:
            raise B1OfflineError(f"{sample_id}: actual_prompt_length must be positive")
        pair_value = record.get("pair_id")
        pair_id = None if pair_value is None else str(pair_value)
        samples[sample_id] = FrozenSample(
            sample_id=sample_id,
            dataset=dataset,
            task=str(record["task"]),
            pair_id=pair_id,
            target_prompt_length=target,
            actual_prompt_length=actual,
            input_sha256=str(record["input_sha256"]),
            condition=f"{dataset}:{_length_label(target)}",
        )
    if not samples:
        raise B1OfflineError(f"frozen prompts are empty: {prompts_path}")
    return samples


def _validate_pg19_pairs(samples: Mapping[str, FrozenSample]) -> None:
    pairs: dict[str, list[FrozenSample]] = defaultdict(list)
    for sample in samples.values():
        if sample.dataset != "pg19":
            continue
        if not sample.pair_id:
            raise B1OfflineError(f"{sample.sample_id}: PG19 sample has no pair_id")
        pairs[sample.pair_id].append(sample)
    for pair_id, members in pairs.items():
        if len(members) != 2:
            raise B1OfflineError(f"PG19 pair {pair_id!r} has {len(members)} members, expected 2")
        if {member.target_prompt_length for member in members} != {8192, 32768}:
            raise B1OfflineError(f"PG19 pair {pair_id!r} is not an 8K/32K pair")


def _manifest_entries(run_dir: Path, *, verify_file_hashes: bool) -> dict[str, Mapping[str, Any]]:
    manifest_path = run_dir / "file_manifest.json"
    payload = _read_json(manifest_path)
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        raise B1OfflineError(f"{manifest_path}: files must be a list")
    entries: dict[str, Mapping[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise B1OfflineError(f"{manifest_path}: non-object file entry")
        relative = entry.get("relative_path")
        if not isinstance(relative, str) or not relative or relative in entries:
            raise B1OfflineError(f"{manifest_path}: invalid or duplicate relative_path {relative!r}")
        target = (run_dir / relative).resolve()
        try:
            target.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise B1OfflineError(f"{manifest_path}: path escapes run directory: {relative}") from exc
        if not target.is_file():
            raise B1OfflineError(f"{manifest_path}: missing registered file {relative}")
        if verify_file_hashes and int(entry.get("file_bytes", -1)) != target.stat().st_size:
            raise B1OfflineError(f"{target}: file size differs from manifest")
        if verify_file_hashes and sha256_file(target) != entry.get("sha256"):
            raise B1OfflineError(f"{target}: SHA-256 differs from manifest")
        entries[relative] = entry
    return entries


def _require_array(archive: Any, name: str, *, dtype: np.dtype[Any], shape: tuple[Any, ...]) -> np.ndarray:
    if name not in archive:
        raise B1OfflineError(f"trace archive is missing {name}")
    value = archive[name]
    if value.dtype != dtype or value.ndim != len(shape):
        raise B1OfflineError(f"{name}: expected dtype {dtype} and {len(shape)} dimensions")
    for observed, expected in zip(value.shape, shape):
        if expected is not None and observed != expected:
            raise B1OfflineError(f"{name}: expected shape {shape}, got {value.shape}")
    return value


def _validate_prefill(path: Path, schema: Mapping[str, Any]) -> None:
    dimensions = schema["dimensions"]
    with np.load(path) as archive:
        positions = _require_array(archive, "query_positions", dtype=np.dtype("uint32"), shape=(None,))
        visible = _require_array(archive, "visible_token_count", dtype=np.dtype("uint32"), shape=(len(positions),))
        features = _require_array(
            archive,
            "scalar_features",
            dtype=np.dtype("float32"),
            shape=(len(positions), dimensions["L"], dimensions["Hq"], dimensions["F"]),
        )
        if len(positions) > int(schema["prefill_window"]) or not np.isfinite(features).all():
            raise B1OfflineError(f"{path}: invalid Prefill window or non-finite scalar feature")
        if positions.size and (np.any(positions + 1 != visible) or np.any(np.diff(positions.astype(np.int64)) <= 0)):
            raise B1OfflineError(f"{path}: invalid Prefill query positions/visibility")


def _validate_decode_chunk(
    path: Path,
    schema: Mapping[str, Any],
    *,
    expected_start: int,
) -> tuple[int, int]:
    dimensions = schema["dimensions"]
    scopes = len(schema["page_scopes"])
    with np.load(path) as archive:
        steps = _require_array(archive, "decode_steps", dtype=np.dtype("uint32"), shape=(None,))
        count = len(steps)
        if count == 0 or steps.tolist() != list(range(expected_start, expected_start + count)):
            raise B1OfflineError(f"{path}: Decode steps are not continuous from {expected_start}")
        positions = _require_array(archive, "query_positions", dtype=np.dtype("uint32"), shape=(count,))
        visible = _require_array(archive, "visible_token_count", dtype=np.dtype("uint32"), shape=(count,))
        features = _require_array(
            archive,
            "scalar_features",
            dtype=np.dtype("float32"),
            shape=(count, dimensions["L"], dimensions["Hq"], dimensions["F"]),
        )
        if not np.isfinite(features).all() or np.any(positions + 1 != visible):
            raise B1OfflineError(f"{path}: invalid Decode scalar features or visibility")
        if schema["optional_arrays"].get("save_token_topk"):
            _require_array(
                archive,
                "token_topk_positions",
                dtype=np.dtype("uint32"),
                shape=(count, dimensions["L"], dimensions["Hq"], schema["token_topk"]),
            )
        for name in ("page_topk_ids_all_visible", "page_topk_ids_prompt_only"):
            _require_array(
                archive,
                name,
                dtype=np.dtype("uint32"),
                shape=(count, dimensions["L"], dimensions["Hkv"], schema["page_topk"]),
            )
        for name in ("page_candidate_count_all_visible", "page_candidate_count_prompt_only"):
            _require_array(archive, name, dtype=np.dtype("uint32"), shape=(count,))
        scope_shape = (count, dimensions["L"], dimensions["Hkv"], scopes)
        scope_mass = _require_array(archive, "page_scope_mass", dtype=np.dtype("float32"), shape=scope_shape)
        coverage = _require_array(archive, "page_top32_coverage", dtype=np.dtype("float32"), shape=scope_shape)
        valid = _require_array(archive, "page_scope_valid", dtype=np.dtype("bool"), shape=scope_shape)
        if (
            not np.isfinite(scope_mass).all()
            or not np.isfinite(coverage[valid]).all()
            or not np.isnan(coverage[~valid]).all()
        ):
            raise B1OfflineError(f"{path}: invalid Page scope mass/coverage values")
    return int(steps[0]), int(steps[-1])


def _validate_schema(schema: Any, *, path: Path) -> Mapping[str, Any]:
    if not isinstance(schema, dict) or schema.get("schema_version") != SCHEMA_VERSION:
        raise B1OfflineError(f"{path}: unsupported B1 trace schema")
    dimensions = schema.get("dimensions")
    if not isinstance(dimensions, dict) or any(int(dimensions.get(name, 0)) <= 0 for name in ("L", "Hq", "Hkv", "F")):
        raise B1OfflineError(f"{path}: invalid dimensions")
    if dimensions["F"] != len(FEATURE_NAMES) or schema.get("feature_names") != list(FEATURE_NAMES):
        raise B1OfflineError(f"{path}: feature order differs from the frozen B1 schema")
    if dimensions["Hq"] % dimensions["Hkv"] or dimensions["Hq"] // dimensions["Hkv"] != 4:
        raise B1OfflineError(f"{path}: expected the frozen four-to-one GQA mapping")
    frozen_values = {
        "prefill_window": 64,
        "page_size": 16,
        "token_topk": 64,
        "page_topk": 32,
    }
    if schema.get("page_scopes") != list(PAGE_SCOPES) or schema.get("distance_bins") != list(FROZEN_DISTANCE_BINS):
        raise B1OfflineError(f"{path}: Page scope or distance-bin contract differs from frozen B1")
    if any(schema.get(name) != value for name, value in frozen_values.items()):
        raise B1OfflineError(f"{path}: window/Page/Top-K contract differs from frozen B1")
    if not isinstance(schema.get("optional_arrays"), dict) or not schema["optional_arrays"].get("save_token_topk"):
        raise B1OfflineError(f"{path}: frozen B1 requires token Top-K positions")
    return schema


def _validate_sample(
    *,
    run_dir: Path,
    sample: FrozenSample,
    entries: Mapping[str, Mapping[str, Any]],
    schema: Mapping[str, Any],
    verify_trace_arrays: bool,
) -> None:
    prefix = f"samples/{sample.sample_id}/"
    metadata_rel = prefix + "sample_metadata.json"
    generated_rel = prefix + "generated_ids.json"
    prefill_rel = prefix + "prefill_stats.npz"
    for relative in (metadata_rel, generated_rel, prefill_rel):
        if relative not in entries:
            raise B1OfflineError(f"{run_dir}: missing {relative} from file manifest")
    metadata = _read_json(run_dir / metadata_rel)
    expected = {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "task": sample.task,
        "pair_id": sample.pair_id,
        "prompt_length": sample.actual_prompt_length,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise B1OfflineError(f"{sample.sample_id}: sample metadata {key} differs from frozen input")
    if metadata.get("status") != "completed":
        raise B1OfflineError(f"{sample.sample_id}: sample status is not completed")
    generated = _read_json(run_dir / generated_rel)
    ids = generated.get("ids") if isinstance(generated, dict) else None
    if not isinstance(ids, list) or any(not isinstance(value, int) or value < 0 for value in ids):
        raise B1OfflineError(f"{sample.sample_id}: invalid generated_ids.json")
    trace_length = int(metadata.get("decode_trace_length", -1))
    if int(metadata.get("actual_generated_length", -1)) != len(ids) or trace_length != max(0, len(ids) - 1):
        raise B1OfflineError(f"{sample.sample_id}: generation and Decode trace lengths disagree")
    chunks = sorted(
        relative for relative in entries if relative.startswith(prefix + "decode_stats_") and relative.endswith(".npz")
    )
    expected_start = 0
    for relative in chunks:
        entry = entries[relative]
        if int(entry.get("step_start", -1)) != expected_start:
            raise B1OfflineError(f"{sample.sample_id}: Decode chunk manifest is not continuous")
        end = int(entry.get("step_end", -1))
        if end < expected_start:
            raise B1OfflineError(f"{sample.sample_id}: invalid Decode chunk interval")
        if verify_trace_arrays:
            observed_start, observed_end = _validate_decode_chunk(
                run_dir / relative, schema, expected_start=expected_start
            )
            if (observed_start, observed_end) != (expected_start, end):
                raise B1OfflineError(f"{sample.sample_id}: NPZ steps differ from file manifest")
        expected_start = end + 1
    if expected_start != trace_length:
        raise B1OfflineError(f"{sample.sample_id}: Decode chunks cover {expected_start} steps, expected {trace_length}")
    if verify_trace_arrays:
        _validate_prefill(run_dir / prefill_rel, schema)


def load_completed_run_set(
    path: Path | str,
    *,
    source_root: Path | str | None = None,
    expected_batch_count: int | None = None,
    expected_sample_count: int | None = None,
    verify_file_hashes: bool = False,
    verify_trace_arrays: bool = True,
) -> RunSetIndex:
    """Load and fail-closed validate one completed B1 run set."""

    manifest_path = Path(path).resolve()
    root = Path.cwd().resolve() if source_root is None else Path(source_root).resolve()
    payload = _read_json(manifest_path)
    if not isinstance(payload, dict) or payload.get("schema_version") != RUN_SET_SCHEMA_VERSION:
        raise B1OfflineError(f"{manifest_path}: unsupported run-set schema")
    if payload.get("status") != "completed":
        raise B1OfflineError(f"{manifest_path}: run set is not completed")
    batches_payload = payload.get("batches")
    if not isinstance(batches_payload, list) or not batches_payload:
        raise B1OfflineError(f"{manifest_path}: batches must be a non-empty list")
    if expected_batch_count is not None and len(batches_payload) != expected_batch_count:
        raise B1OfflineError(f"{manifest_path}: expected {expected_batch_count} batches, got {len(batches_payload)}")

    fixed = payload.get("fixed_dataset")
    if not isinstance(fixed, dict):
        raise B1OfflineError(f"{manifest_path}: missing fixed_dataset")
    prompts_path = _resolve(root, fixed.get("prompts_path"), label="fixed_dataset.prompts_path")
    fixed_manifest_path = _resolve(root, fixed.get("manifest_path"), label="fixed_dataset.manifest_path")
    fixed_manifest = _read_json(fixed_manifest_path)
    if not isinstance(fixed_manifest, dict) or fixed_manifest.get("data_sha256") != fixed.get("data_sha256"):
        raise B1OfflineError("run set and frozen dataset manifest disagree on data_sha256")
    all_frozen_samples = _load_frozen_samples(prompts_path)

    batch_ids: set[str] = set()
    selected_ids: list[str] = []
    batches: list[RunBatch] = []
    reference_schema_hash: str | None = None
    reference_generation_hash: str | None = None
    reference_collection_hash: str | None = None
    execution = payload.get("execution_contract") if isinstance(payload.get("execution_contract"), dict) else {}
    for batch_payload in batches_payload:
        if not isinstance(batch_payload, dict) or batch_payload.get("status") != "completed":
            raise B1OfflineError("every run-set batch must be a completed object")
        batch_id = str(batch_payload.get("batch_id", ""))
        if not batch_id or batch_id in batch_ids:
            raise B1OfflineError(f"invalid or duplicate batch_id {batch_id!r}")
        batch_ids.add(batch_id)
        sample_ids_value = batch_payload.get("sample_ids")
        if not isinstance(sample_ids_value, list) or not sample_ids_value:
            raise B1OfflineError(f"{batch_id}: sample_ids must be a non-empty list")
        sample_ids = tuple(str(value) for value in sample_ids_value)
        if len(set(sample_ids)) != len(sample_ids):
            raise B1OfflineError(f"{batch_id}: duplicate sample_ids")
        selected_ids.extend(sample_ids)
        run_dir = _resolve(root, batch_payload.get("run_dir"), label=f"{batch_id}.run_dir")
        metadata_path = run_dir / "run_metadata.json"
        file_manifest_path = run_dir / "file_manifest.json"
        if verify_file_hashes and sha256_file(metadata_path) != batch_payload.get("run_metadata_sha256"):
            raise B1OfflineError(f"{batch_id}: run_metadata SHA-256 differs from run set")
        if verify_file_hashes and sha256_file(file_manifest_path) != batch_payload.get("file_manifest_sha256"):
            raise B1OfflineError(f"{batch_id}: file_manifest SHA-256 differs from run set")
        run_metadata = _read_json(metadata_path)
        if not isinstance(run_metadata, dict) or run_metadata.get("status") != "completed":
            raise B1OfflineError(f"{batch_id}: run metadata is not completed")
        if run_metadata.get("fixed_dataset", {}).get("data_sha256") != fixed.get("data_sha256"):
            raise B1OfflineError(f"{batch_id}: run used a different frozen dataset")
        checks = {
            "git.commit": (run_metadata.get("git", {}).get("commit"), execution.get("git_commit")),
            "model": (run_metadata.get("model"), execution.get("model_snapshot")),
            "dtype": (run_metadata.get("dtype"), execution.get("dtype")),
            "attention_backend": (run_metadata.get("attention_backend"), execution.get("attention_backend")),
        }
        for label, (observed, expected) in checks.items():
            if expected is not None and observed != expected:
                raise B1OfflineError(f"{batch_id}: {label} differs from execution contract")
        entries = _manifest_entries(run_dir, verify_file_hashes=verify_file_hashes)
        schema_path = run_dir / "schema.json"
        schema = _validate_schema(_read_json(schema_path), path=schema_path)
        schema_hash = canonical_json_sha256(schema)
        if reference_schema_hash is None:
            reference_schema_hash = schema_hash
        elif schema_hash != reference_schema_hash:
            raise B1OfflineError(f"{batch_id}: trace schema differs from other batches")
        generation_hash = canonical_json_sha256(run_metadata.get("generation"))
        config_snapshot = _read_json(run_dir / "config_snapshot.yaml")
        if not isinstance(config_snapshot, dict) or not isinstance(config_snapshot.get("collection"), dict):
            raise B1OfflineError(f"{batch_id}: missing collection configuration snapshot")
        collection_hash = canonical_json_sha256(config_snapshot["collection"])
        if reference_generation_hash is None:
            reference_generation_hash = generation_hash
            reference_collection_hash = collection_hash
        elif generation_hash != reference_generation_hash or collection_hash != reference_collection_hash:
            raise B1OfflineError(f"{batch_id}: generation or collection configuration differs from other batches")
        for sample_id in sample_ids:
            if sample_id not in all_frozen_samples:
                raise B1OfflineError(f"{batch_id}: unknown frozen sample {sample_id!r}")
            _validate_sample(
                run_dir=run_dir,
                sample=all_frozen_samples[sample_id],
                entries=entries,
                schema=schema,
                verify_trace_arrays=verify_trace_arrays,
            )
        actual_sample_dirs = {item.name for item in (run_dir / "samples").iterdir() if item.is_dir()}
        if actual_sample_dirs != set(sample_ids):
            raise B1OfflineError(f"{batch_id}: sample directories differ from batch sample_ids")
        batches.append(RunBatch(batch_id, run_dir, sample_ids, run_metadata, schema, entries))

    if len(set(selected_ids)) != len(selected_ids):
        raise B1OfflineError("sample_id appears in more than one run-set batch")
    if expected_sample_count is not None and len(selected_ids) != expected_sample_count:
        raise B1OfflineError(f"expected {expected_sample_count} samples, got {len(selected_ids)}")
    if set(selected_ids) != set(all_frozen_samples):
        missing = sorted(set(all_frozen_samples) - set(selected_ids))
        extra = sorted(set(selected_ids) - set(all_frozen_samples))
        raise B1OfflineError(f"run set does not exactly cover the frozen dataset; missing={missing}, extra={extra}")
    selected_samples = {sample_id: all_frozen_samples[sample_id] for sample_id in selected_ids}
    _validate_pg19_pairs(selected_samples)
    return RunSetIndex(
        manifest_path=manifest_path,
        source_root=root,
        run_set_sha256=sha256_file(manifest_path),
        payload=payload,
        samples=selected_samples,
        batches=tuple(batches),
    )


def validate_analysis_sources(
    index: RunSetIndex,
    page_rco_reports: Sequence[Path | str],
    *,
    expected_k_values: Sequence[int] = DEFAULT_PAGE_K,
    expected_deltas: Sequence[int] = DEFAULT_DELTAS,
) -> ValidationReport:
    """Validate one exact Page-RCO report for every indexed batch."""

    expected_k = tuple(int(value) for value in expected_k_values)
    expected_delta = tuple(int(value) for value in expected_deltas)
    by_metadata_hash = {sha256_file(batch.run_dir / "run_metadata.json"): batch for batch in index.batches}
    seen_batches: set[str] = set()
    sources: list[Mapping[str, Any]] = []
    for report_value in page_rco_reports:
        report_path = Path(report_value).resolve()
        report = _read_json(report_path)
        if not isinstance(report, dict) or report.get("schema_version") != PAGE_RCO_SCHEMA_VERSION:
            raise B1OfflineError(f"{report_path}: unsupported Page RCO schema")
        if tuple(report.get("k_values", ())) != expected_k or tuple(report.get("deltas", ())) != expected_delta:
            raise B1OfflineError(f"{report_path}: Page RCO k/delta grid differs from frozen analysis")
        metadata_hash = report.get("source_run_metadata_sha256")
        batch = by_metadata_hash.get(metadata_hash)
        if batch is None or batch.batch_id in seen_batches:
            raise B1OfflineError(f"{report_path}: unknown or duplicate source run")
        seen_batches.add(batch.batch_id)
        source_run = _resolve(index.source_root, report.get("source_run"), label="source_run")
        # Content hashes are authoritative after artifacts are relocated.  If
        # the report's recorded path still exists locally, it must resolve to
        # the same indexed run; a stale remote absolute path is retained only
        # as provenance and is not treated as the local identity.
        if source_run.exists() and source_run != batch.run_dir:
            raise B1OfflineError(f"{report_path}: source_run does not match indexed batch")
        records = report.get("records")
        if not isinstance(records, list):
            raise B1OfflineError(f"{report_path}: records must be a list")
        dimensions = batch.schema["dimensions"]
        expected_count = (
            len(batch.sample_ids)
            * len(PAGE_SCOPES)
            * len(expected_k)
            * len(expected_delta)
            * dimensions["L"]
            * dimensions["Hkv"]
        )
        if len(records) != expected_count:
            raise B1OfflineError(f"{report_path}: expected {expected_count} RCO records, got {len(records)}")
        coordinates: set[tuple[Any, ...]] = set()
        for record in records:
            if not isinstance(record, dict):
                raise B1OfflineError(f"{report_path}: non-object RCO record")
            coordinate = (
                record.get("sample_id"),
                record.get("scope"),
                record.get("k"),
                record.get("delta"),
                record.get("layer_id"),
                record.get("kv_head"),
            )
            if coordinate in coordinates:
                raise B1OfflineError(f"{report_path}: duplicate RCO coordinate {coordinate}")
            coordinates.add(coordinate)
            sample_id, scope, k, delta, layer, head = coordinate
            if (
                sample_id not in batch.sample_ids
                or scope not in PAGE_SCOPES
                or k not in expected_k
                or delta not in expected_delta
                or not isinstance(layer, int)
                or not 0 <= layer < dimensions["L"]
                or not isinstance(head, int)
                or not 0 <= head < dimensions["Hkv"]
            ):
                raise B1OfflineError(f"{report_path}: invalid RCO coordinate {coordinate}")
            valid_count = record.get("valid_pair_count")
            if not isinstance(valid_count, int) or valid_count < 0:
                raise B1OfflineError(f"{report_path}: invalid valid_pair_count")
            metrics = ("overlap_raw_mean", "random_baseline_mean", "rco_raw_mean", "rco_clipped_mean")
            values = [record.get(name) for name in metrics]
            if valid_count == 0 and any(value is not None for value in values):
                raise B1OfflineError(f"{report_path}: zero-pair RCO record must contain null metrics")
            if valid_count > 0 and any(
                not isinstance(value, (int, float)) or not math.isfinite(value) for value in values
            ):
                raise B1OfflineError(f"{report_path}: valid RCO record contains a non-finite metric")
        sources.append(
            {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
                "batch_id": batch.batch_id,
                "record_count": len(records),
            }
        )
    if seen_batches != {batch.batch_id for batch in index.batches}:
        missing = sorted({batch.batch_id for batch in index.batches} - seen_batches)
        raise B1OfflineError(f"missing Page RCO reports for batches {missing}")
    return ValidationReport(
        source_run_set_sha256=index.run_set_sha256,
        sample_count=len(index.samples),
        batch_count=len(index.batches),
        condition_counts=index.condition_counts,
        page_rco_reports=tuple(sorted(sources, key=lambda item: str(item["batch_id"]))),
    )
