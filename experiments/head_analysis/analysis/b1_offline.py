"""Validated source index and common statistics for the frozen B1 analysis.

The module is deliberately read-only: it validates completed run artifacts and
returns Python/NumPy objects.  File creation belongs to the thin CLI layer.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.head_analysis.collectors.attention_stats import FEATURE_NAMES
from experiments.head_analysis.storage.trace_writer import PAGE_SCOPES, SCHEMA_VERSION

RUN_SET_SCHEMA_VERSION = "headkv-v1-b1-run-set-v1"
PAGE_RCO_SCHEMA_VERSION = "headkv-v1-b1-page-rco-v1"
ANALYSIS_PLAN_SCHEMA_VERSION = "headkv-v1-b1-offline-plan-v1"
ANALYSIS_SCHEMA_VERSION = "headkv-v1-b1-offline-analysis-v1"
DEFAULT_PAGE_K = (8, 16, 32)
DEFAULT_TOKEN_K = (16, 32, 64)
DEFAULT_DELTAS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
FROZEN_DISTANCE_BINS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
SUMMARY_STATISTICS = ("mean", "median", "p10", "p90")
QUERY_PHASES = ("prefill", "decode")
DECODE_SEGMENTS = ("q1", "q2", "q3", "q4")
PAGE_METRICS = ("scope_mass", "top32_coverage", "scope_valid_rate")
RCO_METRICS = ("overlap_raw_mean", "random_baseline_mean", "rco_raw_mean", "rco_clipped_mean")
INVARIANT_ATOL = 1e-5
INVARIANT_RTOL = 1e-5
DEFAULT_ANALYSIS_PARAMETERS: dict[str, Any] = {
    "page_k": list(DEFAULT_PAGE_K),
    "token_k": list(DEFAULT_TOKEN_K),
    "deltas": list(DEFAULT_DELTAS),
    "bootstrap_replicates": 2000,
    "bootstrap_seed": 20261103,
    "primary_weighting": "sample_equal",
}


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


@dataclass(frozen=True)
class QueryMetricSummary:
    """Sample and condition Query-Head profiles with explicit array axes."""

    sample_ids: tuple[str, ...]
    sample_conditions: tuple[str, ...]
    condition_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    statistics: tuple[str, ...]
    phases: tuple[str, ...]
    decode_segments: tuple[str, ...]
    sample_full: np.ndarray  # [N,P,L,Hq,F,Stat]
    sample_full_valid_count: np.ndarray  # [N,P,L,Hq,F]
    sample_decode_segments: np.ndarray  # [N,Seg,L,Hq,F,Stat]
    sample_decode_segment_valid_count: np.ndarray  # [N,Seg,L,Hq,F]
    condition_full_mean: np.ndarray  # [Cond,P,L,Hq,F]
    condition_full_valid_samples: np.ndarray  # [Cond,P,L,Hq,F]
    condition_decode_segment_mean: np.ndarray  # [Cond,Seg,L,Hq,F]
    condition_decode_segment_valid_samples: np.ndarray  # [Cond,Seg,L,Hq,F]


@dataclass(frozen=True)
class PageMetricSummary:
    """Sample and condition KV-Head Page profiles with explicit array axes."""

    sample_ids: tuple[str, ...]
    sample_conditions: tuple[str, ...]
    condition_names: tuple[str, ...]
    scopes: tuple[str, ...]
    metrics: tuple[str, ...]
    statistics: tuple[str, ...]
    decode_segments: tuple[str, ...]
    sample_full: np.ndarray  # [N,L,Hkv,S,M,Stat]
    sample_full_valid_count: np.ndarray  # [N,L,Hkv,S,M]
    sample_decode_segments: np.ndarray  # [N,Seg,L,Hkv,S,M,Stat]
    sample_decode_segment_valid_count: np.ndarray  # [N,Seg,L,Hkv,S,M]
    condition_full_mean: np.ndarray  # [Cond,L,Hkv,S,M]
    condition_full_valid_samples: np.ndarray  # [Cond,L,Hkv,S,M]
    condition_decode_segment_mean: np.ndarray  # [Cond,Seg,L,Hkv,S,M]
    condition_decode_segment_valid_samples: np.ndarray  # [Cond,Seg,L,Hkv,S,M]


@dataclass(frozen=True)
class PageRCOSummary:
    """Page RCO sample records, Head profiles, sensitivity, and scalar CIs."""

    sample_ids: tuple[str, ...]
    sample_conditions: tuple[str, ...]
    condition_names: tuple[str, ...]
    scopes: tuple[str, ...]
    k_values: tuple[int, ...]
    deltas: tuple[int, ...]
    metrics: tuple[str, ...]
    sample_head_values: np.ndarray  # [N,L,Hkv,S,K,D,M]
    sample_head_valid_pair_count: np.ndarray  # [N,L,Hkv,S,K,D]
    condition_head_sample_equal_mean: np.ndarray  # [Cond,L,Hkv,S,K,D,M]
    condition_head_valid_samples: np.ndarray  # [Cond,L,Hkv,S,K,D,M]
    condition_head_pair_weighted_mean: np.ndarray  # [Cond,L,Hkv,S,K,D,M]
    condition_head_valid_pair_count: np.ndarray  # [Cond,L,Hkv,S,K,D]
    condition_scalar_sample_equal_mean: np.ndarray  # [Cond,S,K,D,M]
    condition_scalar_valid_samples: np.ndarray  # [Cond,S,K,D,M]
    condition_scalar_pair_weighted_mean: np.ndarray  # [Cond,S,K,D,M]
    condition_scalar_valid_pair_count: np.ndarray  # [Cond,S,K,D]
    condition_scalar_bootstrap_ci: np.ndarray  # [Cond,S,K,D,M,2]
    pg19_pair_ids: tuple[str, ...]
    pg19_pair_head_difference_32k_minus_8k: np.ndarray  # [Pair,L,Hkv,S,K,D,M]
    pg19_pair_head_mean_difference: np.ndarray  # [L,Hkv,S,K,D,M]
    pg19_pair_scalar_difference_32k_minus_8k: np.ndarray  # [Pair,S,K,D,M]
    pg19_pair_scalar_mean_difference: np.ndarray  # [S,K,D,M]
    pg19_pair_scalar_bootstrap_ci: np.ndarray  # [S,K,D,M,2]
    bootstrap_replicates: int
    bootstrap_seed: int


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise B1OfflineError(f"unable to hash {path}: {exc}") from exc
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
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
        if int(entry.get("file_bytes", -1)) != target.stat().st_size:
            raise B1OfflineError(f"{target}: file size differs from manifest")
        if verify_file_hashes and sha256_file(target) != entry.get("sha256"):
            raise B1OfflineError(f"{target}: SHA-256 differs from manifest")
        entries[relative] = entry
    actual = {
        path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*") if path.is_file() and path != manifest_path
    }
    if actual != set(entries):
        missing = sorted(set(entries) - actual)
        unregistered = sorted(actual - set(entries))
        raise B1OfflineError(f"{run_dir}: manifest/file-set mismatch; missing={missing}, unregistered={unregistered}")
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
        "input_sha256": sample.input_sha256,
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
    generated_bytes = np.asarray(ids, dtype="<u4").tobytes()
    if hashlib.sha256(generated_bytes).hexdigest() != generated.get("sha256"):
        raise B1OfflineError(f"{sample.sample_id}: generated ID content hash mismatch")
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
    verify_file_hashes: bool = True,
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
        if sha256_file(metadata_path) != batch_payload.get("run_metadata_sha256"):
            raise B1OfflineError(f"{batch_id}: run_metadata SHA-256 differs from run set")
        if sha256_file(file_manifest_path) != batch_payload.get("file_manifest_sha256"):
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


def summarize_sample_equal(
    records: Iterable[Mapping[str, Any]],
    *,
    group_keys: Sequence[str],
    value_key: str,
    sample_key: str = "sample_id",
    secondary_weight_key: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate observations within samples, then average samples equally.

    Null values are retained as invalid observations and never converted to
    zero.  If ``secondary_weight_key`` is supplied, a clearly named
    observation-weighted sensitivity result is emitted alongside the primary
    sample-equal result.
    """

    grouped: dict[tuple[Any, ...], dict[str, list[tuple[float, float | None]]]] = defaultdict(lambda: defaultdict(list))
    seen_samples: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for record in records:
        try:
            group = tuple(record[key] for key in group_keys)
            sample_id = str(record[sample_key])
        except KeyError as exc:
            raise B1OfflineError(f"aggregation record is missing {exc.args[0]!r}") from exc
        if not sample_id:
            raise B1OfflineError("aggregation sample_id must be non-empty")
        seen_samples[group].add(sample_id)
        value = record.get(value_key)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise B1OfflineError(f"{value_key} must be finite or null")
        weight: float | None = None
        if secondary_weight_key is not None:
            raw_weight = record.get(secondary_weight_key)
            if not isinstance(raw_weight, (int, float)) or not math.isfinite(float(raw_weight)) or raw_weight < 0:
                raise B1OfflineError(f"{secondary_weight_key} must be finite and non-negative")
            weight = float(raw_weight)
        grouped[group][sample_id].append((float(value), weight))

    output: list[dict[str, Any]] = []
    for group in sorted(seen_samples, key=lambda key: tuple(str(value) for value in key)):
        sample_values = {
            sample_id: float(np.mean([value for value, _ in observations]))
            for sample_id, observations in grouped[group].items()
            if observations
        }
        values = np.asarray(list(sample_values.values()), dtype=np.float64)
        primary: dict[str, Any] | None
        if values.size:
            primary = {
                "weighting": "sample_equal",
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "p10": float(np.quantile(values, 0.10)),
                "p50": float(np.quantile(values, 0.50)),
                "p90": float(np.quantile(values, 0.90)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        else:
            primary = None
        secondary: dict[str, Any] | None = None
        if secondary_weight_key is not None:
            weighted = [
                (value, weight)
                for observations in grouped[group].values()
                for value, weight in observations
                if weight is not None and weight > 0
            ]
            total_weight = float(sum(weight for _, weight in weighted))
            if total_weight > 0:
                secondary = {
                    "weighting": f"{secondary_weight_key}_weighted",
                    "mean": float(sum(value * weight for value, weight in weighted) / total_weight),
                    "total_weight": total_weight,
                }
        output.append(
            {
                "group": dict(zip(group_keys, group)),
                "value_key": value_key,
                "sample_count_total": len(seen_samples[group]),
                "sample_count_valid": len(sample_values),
                "observation_count_valid": sum(len(values_) for values_ in grouped[group].values()),
                "primary_sample_equal": primary,
                "secondary_observation_weighted": secondary,
            }
        )
    return output


def decode_segment_ids(trace_length: int, *, minimum_length: int = 32) -> np.ndarray:
    """Return frozen normalized-progress segment IDs, or an empty array when ineligible."""

    if trace_length < 0:
        raise B1OfflineError("trace_length must be non-negative")
    if minimum_length <= 0:
        raise B1OfflineError("minimum segment length must be positive")
    if trace_length < minimum_length:
        return np.empty((0,), dtype=np.uint8)
    steps = np.arange(trace_length, dtype=np.int64)
    return np.minimum(3, (4 * steps) // trace_length).astype(np.uint8)


def _time_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the leading time axis while preserving invalid coordinates as NaN."""

    if values.ndim < 1:
        raise B1OfflineError("time-statistic input must have a leading time axis")
    finite = np.isfinite(values)
    counts = finite.sum(axis=0, dtype=np.uint32)
    if finite.all():
        quantiles = np.quantile(values, (0.10, 0.50, 0.90), axis=0)
        statistics = np.stack(
            (
                np.mean(values, axis=0),
                quantiles[1],
                quantiles[0],
                quantiles[2],
            ),
            axis=-1,
        ).astype(np.float32)
        return statistics, counts
    flattened = values.reshape(values.shape[0], -1).astype(np.float64, copy=False)
    flat_counts = counts.reshape(-1)
    output = np.full((flattened.shape[1], len(SUMMARY_STATISTICS)), np.nan, dtype=np.float32)
    for column in np.flatnonzero(flat_counts):
        valid = flattened[:, column][np.isfinite(flattened[:, column])]
        quantiles = np.quantile(valid, (0.10, 0.50, 0.90))
        output[column] = (
            np.mean(valid),
            quantiles[1],
            quantiles[0],
            quantiles[2],
        )
    return output.reshape((*values.shape[1:], len(SUMMARY_STATISTICS))), counts


def _condition_sample_equal(
    sample_means: np.ndarray,
    sample_conditions: Sequence[str],
    condition_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    if sample_means.shape[0] != len(sample_conditions):
        raise B1OfflineError("sample profile count and sample condition count differ")
    result = np.full((len(condition_names), *sample_means.shape[1:]), np.nan, dtype=np.float32)
    counts = np.zeros(result.shape, dtype=np.uint32)
    for condition_index, condition in enumerate(condition_names):
        indices = [index for index, value in enumerate(sample_conditions) if value == condition]
        if not indices:
            continue
        values = sample_means[indices].astype(np.float64, copy=False)
        valid = np.isfinite(values)
        local_counts = valid.sum(axis=0, dtype=np.uint32)
        totals = np.where(valid, values, 0.0).sum(axis=0)
        np.divide(totals, local_counts, out=result[condition_index], where=local_counts > 0)
        counts[condition_index] = local_counts
    return result, counts


def _feature_index(name: str) -> int:
    try:
        return FEATURE_NAMES.index(name)
    except ValueError as exc:  # pragma: no cover - guarded by the frozen schema
        raise B1OfflineError(f"frozen feature {name!r} is missing") from exc


def _validate_scalar_invariants(values: np.ndarray, visible_count: np.ndarray, *, source: Path) -> None:
    if values.ndim != 4 or values.shape[0] != len(visible_count) or values.shape[-1] != len(FEATURE_NAMES):
        raise B1OfflineError(f"{source}: scalar feature shape is incompatible with visibility")
    if not np.isfinite(values).all() or np.any(visible_count <= 0):
        raise B1OfflineError(f"{source}: scalar features must be finite and visibility positive")
    entropy = values[..., _feature_index("entropy")]
    normalized = values[..., _feature_index("entropy_normalized")]
    upper_entropy = np.log(visible_count.astype(np.float64))[:, None, None]
    if np.any(entropy < -INVARIANT_ATOL) or np.any(entropy > upper_entropy + INVARIANT_ATOL):
        raise B1OfflineError(f"{source}: entropy is outside [0, log(N)]")
    expected_normalized = np.divide(
        entropy,
        upper_entropy,
        out=np.zeros_like(entropy, dtype=np.float64),
        where=upper_entropy > 0,
    )
    if (
        np.any(normalized < -INVARIANT_ATOL)
        or np.any(normalized > 1 + INVARIANT_ATOL)
        or not np.allclose(normalized, expected_normalized, atol=INVARIANT_ATOL, rtol=INVARIANT_RTOL)
    ):
        raise B1OfflineError(f"{source}: normalized entropy violates its definition")

    top_mass = values[
        ..., [_feature_index(name) for name in ("top_mass_16", "top_mass_64", "top_mass_256", "top_mass_1024")]
    ]
    if (
        np.any(top_mass < -INVARIANT_ATOL)
        or np.any(top_mass > 1 + INVARIANT_ATOL)
        or np.any(np.diff(top_mass, axis=-1) < -INVARIANT_ATOL)
    ):
        raise B1OfflineError(f"{source}: Top-K mass is out of range or non-monotonic")
    bounded_names = ("sink_mass_4", "recent_mass_64", "prompt_mass", "generated_mass")
    bounded = values[..., [_feature_index(name) for name in bounded_names]]
    if np.any(bounded < -INVARIANT_ATOL) or np.any(bounded > 1 + INVARIANT_ATOL):
        raise B1OfflineError(f"{source}: probability mass feature is outside [0,1]")
    prompt = values[..., _feature_index("prompt_mass")]
    generated = values[..., _feature_index("generated_mass")]
    if not np.allclose(prompt + generated, 1.0, atol=INVARIANT_ATOL, rtol=INVARIANT_RTOL):
        raise B1OfflineError(f"{source}: prompt_mass + generated_mass differs from one")
    distance = values[..., _feature_index("mean_distance")]
    if np.any(distance < -INVARIANT_ATOL) or np.any(distance > (visible_count - 1)[:, None, None] + INVARIANT_ATOL):
        raise B1OfflineError(f"{source}: mean distance is outside the visible range")
    distance_mass = values[..., 11:]
    if (
        np.any(distance_mass < -INVARIANT_ATOL)
        or np.any(distance_mass > 1 + INVARIANT_ATOL)
        or not np.allclose(distance_mass.sum(axis=-1), 1.0, atol=INVARIANT_ATOL, rtol=INVARIANT_RTOL)
    ):
        raise B1OfflineError(f"{source}: distance-mass bins do not form a probability distribution")


def _validate_page_invariants(
    scope_mass: np.ndarray,
    coverage: np.ndarray,
    valid: np.ndarray,
    *,
    source: Path,
) -> None:
    if (
        scope_mass.shape != coverage.shape
        or scope_mass.shape != valid.shape
        or scope_mass.shape[-1] != len(PAGE_SCOPES)
    ):
        raise B1OfflineError(f"{source}: Page metric shapes disagree")
    if (
        not np.isfinite(scope_mass).all()
        or np.any(scope_mass < -INVARIANT_ATOL)
        or np.any(scope_mass > 1 + INVARIANT_ATOL)
    ):
        raise B1OfflineError(f"{source}: Page scope mass is outside [0,1]")
    if not np.allclose(scope_mass[..., 0], 1.0, atol=INVARIANT_ATOL, rtol=INVARIANT_RTOL):
        raise B1OfflineError(f"{source}: all-visible Page scope mass differs from one")
    if not valid[..., 0].all():
        raise B1OfflineError(f"{source}: all-visible Page scope must be valid")
    if (
        not np.isfinite(coverage[valid]).all()
        or not np.isnan(coverage[~valid]).all()
        or np.any(coverage[valid] < -INVARIANT_ATOL)
        or np.any(coverage[valid] > 1 + INVARIANT_ATOL)
    ):
        raise B1OfflineError(f"{source}: Page coverage violates validity or range rules")


def _ordered_samples(index: RunSetIndex) -> tuple[tuple[FrozenSample, RunBatch], ...]:
    ordered: list[tuple[FrozenSample, RunBatch]] = []
    for batch in index.batches:
        ordered.extend((index.samples[sample_id], batch) for sample_id in batch.sample_ids)
    return tuple(ordered)


def _decode_entries(batch: RunBatch, sample_id: str) -> list[tuple[str, Mapping[str, Any]]]:
    prefix = f"samples/{sample_id}/decode_stats_"
    entries = [
        (relative, entry)
        for relative, entry in batch.manifest_files.items()
        if relative.startswith(prefix) and relative.endswith(".npz")
    ]
    return sorted(entries, key=lambda item: int(item[1]["step_start"]))


def summarize_query_and_page_metrics(index: RunSetIndex) -> tuple[QueryMetricSummary, PageMetricSummary]:
    """Read each validated trace once and summarize Query and Page metrics."""

    ordered = _ordered_samples(index)
    if not ordered:
        raise B1OfflineError("run-set index contains no samples")
    dimensions = index.batches[0].schema["dimensions"]
    sample_count = len(ordered)
    layers = int(dimensions["L"])
    query_heads = int(dimensions["Hq"])
    kv_heads = int(dimensions["Hkv"])
    feature_count = len(FEATURE_NAMES)
    statistic_count = len(SUMMARY_STATISTICS)
    segment_count = len(DECODE_SEGMENTS)
    scope_count = len(PAGE_SCOPES)
    metric_count = len(PAGE_METRICS)

    query_full = np.full(
        (sample_count, len(QUERY_PHASES), layers, query_heads, feature_count, statistic_count),
        np.nan,
        dtype=np.float32,
    )
    query_full_count = np.zeros(query_full.shape[:-1], dtype=np.uint32)
    query_segments = np.full(
        (sample_count, segment_count, layers, query_heads, feature_count, statistic_count),
        np.nan,
        dtype=np.float32,
    )
    query_segment_count = np.zeros(query_segments.shape[:-1], dtype=np.uint32)
    page_full = np.full(
        (sample_count, layers, kv_heads, scope_count, metric_count, statistic_count),
        np.nan,
        dtype=np.float32,
    )
    page_full_count = np.zeros(page_full.shape[:-1], dtype=np.uint32)
    page_segments = np.full(
        (sample_count, segment_count, layers, kv_heads, scope_count, metric_count, statistic_count),
        np.nan,
        dtype=np.float32,
    )
    page_segment_count = np.zeros(page_segments.shape[:-1], dtype=np.uint32)

    sample_ids: list[str] = []
    sample_conditions: list[str] = []
    for sample_index, (sample, batch) in enumerate(ordered):
        sample_ids.append(sample.sample_id)
        sample_conditions.append(sample.condition)
        prefill_path = batch.run_dir / f"samples/{sample.sample_id}/prefill_stats.npz"
        with np.load(prefill_path) as archive:
            prefill_values = archive["scalar_features"].copy()
            prefill_visible = archive["visible_token_count"].copy()
        _validate_scalar_invariants(prefill_values, prefill_visible, source=prefill_path)
        query_full[sample_index, 0], query_full_count[sample_index, 0] = _time_statistics(prefill_values)

        decode_scalar: list[np.ndarray] = []
        decode_visible: list[np.ndarray] = []
        decode_scope_mass: list[np.ndarray] = []
        decode_coverage: list[np.ndarray] = []
        decode_valid: list[np.ndarray] = []
        decode_paths: list[Path] = []
        for relative, _ in _decode_entries(batch, sample.sample_id):
            path = batch.run_dir / relative
            decode_paths.append(path)
            with np.load(path) as archive:
                decode_scalar.append(archive["scalar_features"].copy())
                decode_visible.append(archive["visible_token_count"].copy())
                decode_scope_mass.append(archive["page_scope_mass"].copy())
                decode_coverage.append(archive["page_top32_coverage"].copy())
                decode_valid.append(archive["page_scope_valid"].copy())
        if not decode_scalar:
            raise B1OfflineError(f"{sample.sample_id}: completed sample has no Decode trace")
        scalar = np.concatenate(decode_scalar, axis=0)
        visible = np.concatenate(decode_visible, axis=0)
        scope_mass = np.concatenate(decode_scope_mass, axis=0)
        coverage = np.concatenate(decode_coverage, axis=0)
        valid = np.concatenate(decode_valid, axis=0)
        source = decode_paths[0]
        _validate_scalar_invariants(scalar, visible, source=source)
        _validate_page_invariants(scope_mass, coverage, valid, source=source)
        query_full[sample_index, 1], query_full_count[sample_index, 1] = _time_statistics(scalar)
        page_values = np.stack((scope_mass, coverage, valid.astype(np.float32)), axis=-1)
        page_full[sample_index], page_full_count[sample_index] = _time_statistics(page_values)

        segments = decode_segment_ids(len(scalar))
        if segments.size:
            for segment in range(segment_count):
                selected = segments == segment
                query_segments[sample_index, segment], query_segment_count[sample_index, segment] = _time_statistics(
                    scalar[selected]
                )
                page_segments[sample_index, segment], page_segment_count[sample_index, segment] = _time_statistics(
                    page_values[selected]
                )

    condition_names = tuple(sorted(set(sample_conditions)))
    query_condition_full, query_condition_full_count = _condition_sample_equal(
        query_full[..., 0], sample_conditions, condition_names
    )
    query_condition_segments, query_condition_segment_count = _condition_sample_equal(
        query_segments[..., 0], sample_conditions, condition_names
    )
    page_condition_full, page_condition_full_count = _condition_sample_equal(
        page_full[..., 0], sample_conditions, condition_names
    )
    page_condition_segments, page_condition_segment_count = _condition_sample_equal(
        page_segments[..., 0], sample_conditions, condition_names
    )
    query_summary = QueryMetricSummary(
        sample_ids=tuple(sample_ids),
        sample_conditions=tuple(sample_conditions),
        condition_names=condition_names,
        feature_names=tuple(FEATURE_NAMES),
        statistics=SUMMARY_STATISTICS,
        phases=QUERY_PHASES,
        decode_segments=DECODE_SEGMENTS,
        sample_full=query_full,
        sample_full_valid_count=query_full_count,
        sample_decode_segments=query_segments,
        sample_decode_segment_valid_count=query_segment_count,
        condition_full_mean=query_condition_full,
        condition_full_valid_samples=query_condition_full_count,
        condition_decode_segment_mean=query_condition_segments,
        condition_decode_segment_valid_samples=query_condition_segment_count,
    )
    page_summary = PageMetricSummary(
        sample_ids=tuple(sample_ids),
        sample_conditions=tuple(sample_conditions),
        condition_names=condition_names,
        scopes=tuple(PAGE_SCOPES),
        metrics=PAGE_METRICS,
        statistics=SUMMARY_STATISTICS,
        decode_segments=DECODE_SEGMENTS,
        sample_full=page_full,
        sample_full_valid_count=page_full_count,
        sample_decode_segments=page_segments,
        sample_decode_segment_valid_count=page_segment_count,
        condition_full_mean=page_condition_full,
        condition_full_valid_samples=page_condition_full_count,
        condition_decode_segment_mean=page_condition_segments,
        condition_decode_segment_valid_samples=page_condition_segment_count,
    )
    return query_summary, page_summary


def summarize_query_metrics(index: RunSetIndex) -> QueryMetricSummary:
    """Return the Query projection of the shared one-pass summary."""

    return summarize_query_and_page_metrics(index)[0]


def summarize_page_metrics(index: RunSetIndex) -> PageMetricSummary:
    """Return the Page projection of the shared one-pass summary."""

    return summarize_query_and_page_metrics(index)[1]


def _nanmean_axes(values: np.ndarray, axes: int | tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    counts = valid.sum(axis=axes, dtype=np.uint32)
    totals = np.where(valid, values, 0.0).sum(axis=axes, dtype=np.float64)
    result = np.full(counts.shape, np.nan, dtype=np.float32)
    np.divide(totals, counts, out=result, where=counts > 0)
    return result, counts


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Percentile CI over the leading independent-sample axis."""

    if values.ndim < 1 or values.shape[0] == 0:
        raise B1OfflineError("bootstrap input must contain at least one independent sample")
    if replicates <= 0:
        raise B1OfflineError("bootstrap_replicates must be positive")
    sample_count = values.shape[0]
    draws = rng.integers(0, sample_count, size=(replicates, sample_count))
    flat = values.reshape(sample_count, -1).astype(np.float64, copy=False)
    output = np.full((flat.shape[1], 2), np.nan, dtype=np.float32)
    for column in np.flatnonzero(np.isfinite(flat).any(axis=0)):
        selected = flat[draws, column]
        valid = np.isfinite(selected)
        counts = valid.sum(axis=1)
        means = np.full(replicates, np.nan, dtype=np.float64)
        np.divide(
            np.where(valid, selected, 0.0).sum(axis=1),
            counts,
            out=means,
            where=counts > 0,
        )
        finite_means = means[np.isfinite(means)]
        if finite_means.size:
            output[column] = np.quantile(finite_means, (0.025, 0.975))
    return output.reshape((*values.shape[1:], 2))


def _pair_weighted_profiles(
    sample_values: np.ndarray,
    sample_pair_counts: np.ndarray,
    sample_conditions: Sequence[str],
    condition_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    output = np.full((len(condition_names), *sample_values.shape[1:]), np.nan, dtype=np.float32)
    count_output = np.zeros((len(condition_names), *sample_pair_counts.shape[1:]), dtype=np.uint64)
    for condition_index, condition in enumerate(condition_names):
        indices = [index for index, value in enumerate(sample_conditions) if value == condition]
        values = sample_values[indices].astype(np.float64, copy=False)
        weights = sample_pair_counts[indices].astype(np.uint64, copy=False)
        valid = np.isfinite(values)
        expanded_weights = weights[..., None]
        denominators = np.where(valid, expanded_weights, 0).sum(axis=0, dtype=np.uint64)
        numerators = np.where(valid, values * expanded_weights, 0.0).sum(axis=0, dtype=np.float64)
        np.divide(numerators, denominators, out=output[condition_index], where=denominators > 0)
        count_output[condition_index] = weights.sum(axis=0, dtype=np.uint64)
    return output, count_output


def aggregate_page_rco(
    index: RunSetIndex,
    page_rco_reports: Sequence[Path | str],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20261103,
) -> PageRCOSummary:
    """Aggregate validated per-sample Page RCO without Step-pair pseudoreplication."""

    validate_analysis_sources(index, page_rco_reports)
    ordered = _ordered_samples(index)
    sample_ids = tuple(sample.sample_id for sample, _ in ordered)
    sample_conditions = tuple(sample.condition for sample, _ in ordered)
    condition_names = tuple(sorted(set(sample_conditions)))
    sample_index = {sample_id: index_ for index_, sample_id in enumerate(sample_ids)}
    dimensions = index.batches[0].schema["dimensions"]
    layers = int(dimensions["L"])
    kv_heads = int(dimensions["Hkv"])
    shape = (
        len(sample_ids),
        layers,
        kv_heads,
        len(PAGE_SCOPES),
        len(DEFAULT_PAGE_K),
        len(DEFAULT_DELTAS),
    )
    sample_values = np.full((*shape, len(RCO_METRICS)), np.nan, dtype=np.float32)
    sample_pair_counts = np.zeros(shape, dtype=np.uint64)
    scope_index = {value: index_ for index_, value in enumerate(PAGE_SCOPES)}
    k_index = {value: index_ for index_, value in enumerate(DEFAULT_PAGE_K)}
    delta_index = {value: index_ for index_, value in enumerate(DEFAULT_DELTAS)}
    seen: set[tuple[int, int, int, int, int, int]] = set()
    for report_value in page_rco_reports:
        report = _read_json(Path(report_value).resolve())
        for record in report["records"]:
            coordinate = (
                sample_index[str(record["sample_id"])],
                int(record["layer_id"]),
                int(record["kv_head"]),
                scope_index[str(record["scope"])],
                k_index[int(record["k"])],
                delta_index[int(record["delta"])],
            )
            if coordinate in seen:
                raise B1OfflineError(f"duplicate Page RCO coordinate after source validation: {coordinate}")
            seen.add(coordinate)
            sample_pair_counts[coordinate] = int(record["valid_pair_count"])
            if int(record["valid_pair_count"]) > 0:
                sample_values[coordinate] = [float(record[name]) for name in RCO_METRICS]
    if len(seen) != int(np.prod(shape)):
        raise B1OfflineError("Page RCO reports do not cover the complete indexed coordinate grid")

    condition_head_mean, condition_head_valid = _condition_sample_equal(
        sample_values, sample_conditions, condition_names
    )
    condition_head_weighted, condition_head_pair_count = _pair_weighted_profiles(
        sample_values, sample_pair_counts, sample_conditions, condition_names
    )
    sample_scalar, _ = _nanmean_axes(sample_values, axes=(1, 2))
    condition_scalar_mean, condition_scalar_valid = _condition_sample_equal(
        sample_scalar, sample_conditions, condition_names
    )

    scalar_weighted = np.full(
        (len(condition_names), len(PAGE_SCOPES), len(DEFAULT_PAGE_K), len(DEFAULT_DELTAS), len(RCO_METRICS)),
        np.nan,
        dtype=np.float32,
    )
    scalar_pair_count = np.zeros(scalar_weighted.shape[:-1], dtype=np.uint64)
    for condition_index, condition in enumerate(condition_names):
        indices = [index_ for index_, value in enumerate(sample_conditions) if value == condition]
        values = sample_values[indices].astype(np.float64, copy=False)
        weights = sample_pair_counts[indices].astype(np.uint64, copy=False)
        expanded_weights = weights[..., None]
        valid = np.isfinite(values)
        reduce_axes = (0, 1, 2)
        denominators = np.where(valid, expanded_weights, 0).sum(axis=reduce_axes, dtype=np.uint64)
        numerators = np.where(valid, values * expanded_weights, 0.0).sum(axis=reduce_axes, dtype=np.float64)
        np.divide(
            numerators,
            denominators,
            out=scalar_weighted[condition_index],
            where=denominators > 0,
        )
        scalar_pair_count[condition_index] = weights.sum(axis=reduce_axes, dtype=np.uint64)

    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    scalar_ci = np.full((*condition_scalar_mean.shape, 2), np.nan, dtype=np.float32)
    for condition_index, condition in enumerate(condition_names):
        indices = [index_ for index_, value in enumerate(sample_conditions) if value == condition]
        scalar_ci[condition_index] = _bootstrap_mean_ci(
            sample_scalar[indices], replicates=bootstrap_replicates, rng=rng
        )

    pair_members: dict[str, dict[int, int]] = defaultdict(dict)
    for index_, (sample, _) in enumerate(ordered):
        if sample.dataset == "pg19" and sample.pair_id:
            pair_members[sample.pair_id][sample.target_prompt_length] = index_
    pg19_pair_ids = tuple(sorted(pair_members))
    pair_head_difference = np.full((len(pg19_pair_ids), *sample_values.shape[1:]), np.nan, dtype=np.float32)
    pair_scalar_difference = np.full((len(pg19_pair_ids), *sample_scalar.shape[1:]), np.nan, dtype=np.float32)
    for pair_index, pair_id in enumerate(pg19_pair_ids):
        members = pair_members[pair_id]
        if set(members) != {8192, 32768}:
            raise B1OfflineError(f"PG19 pair {pair_id!r} is incomplete during RCO aggregation")
        pair_head_difference[pair_index] = sample_values[members[32768]] - sample_values[members[8192]]
        pair_scalar_difference[pair_index] = sample_scalar[members[32768]] - sample_scalar[members[8192]]
    pair_head_mean, _ = _nanmean_axes(pair_head_difference, axes=0)
    pair_scalar_mean, _ = _nanmean_axes(pair_scalar_difference, axes=0)
    pair_scalar_ci = _bootstrap_mean_ci(
        pair_scalar_difference,
        replicates=bootstrap_replicates,
        rng=np.random.Generator(np.random.PCG64(bootstrap_seed)),
    )
    return PageRCOSummary(
        sample_ids=sample_ids,
        sample_conditions=sample_conditions,
        condition_names=condition_names,
        scopes=tuple(PAGE_SCOPES),
        k_values=DEFAULT_PAGE_K,
        deltas=DEFAULT_DELTAS,
        metrics=RCO_METRICS,
        sample_head_values=sample_values,
        sample_head_valid_pair_count=sample_pair_counts,
        condition_head_sample_equal_mean=condition_head_mean,
        condition_head_valid_samples=condition_head_valid,
        condition_head_pair_weighted_mean=condition_head_weighted,
        condition_head_valid_pair_count=condition_head_pair_count,
        condition_scalar_sample_equal_mean=condition_scalar_mean,
        condition_scalar_valid_samples=condition_scalar_valid,
        condition_scalar_pair_weighted_mean=scalar_weighted,
        condition_scalar_valid_pair_count=scalar_pair_count,
        condition_scalar_bootstrap_ci=scalar_ci,
        pg19_pair_ids=pg19_pair_ids,
        pg19_pair_head_difference_32k_minus_8k=pair_head_difference,
        pg19_pair_head_mean_difference=pair_head_mean,
        pg19_pair_scalar_difference_32k_minus_8k=pair_scalar_difference,
        pg19_pair_scalar_mean_difference=pair_scalar_mean,
        pg19_pair_scalar_bootstrap_ci=pair_scalar_ci,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )


def build_analysis_plan(
    index: RunSetIndex,
    validation: ValidationReport,
    *,
    analysis_id: str,
    git_commit: str,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not analysis_id or Path(analysis_id).name != analysis_id:
        raise B1OfflineError("analysis_id must be one non-empty path component")
    if not git_commit:
        raise B1OfflineError("analysis git_commit must be explicit")
    if (
        validation.source_run_set_sha256 != index.run_set_sha256
        or validation.sample_count != len(index.samples)
        or validation.batch_count != len(index.batches)
    ):
        raise B1OfflineError("validation report does not belong to this run-set index")
    resolved_parameters = dict(DEFAULT_ANALYSIS_PARAMETERS if parameters is None else parameters)
    return {
        "schema_version": ANALYSIS_PLAN_SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "source_run_set": {"path": str(index.manifest_path), "sha256": index.run_set_sha256},
        "source_page_rco_reports": [dict(item) for item in validation.page_rco_reports],
        "source_schema_version": SCHEMA_VERSION,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "git_commit": git_commit,
        "parameters": resolved_parameters,
        "parameters_sha256": canonical_json_sha256(resolved_parameters),
        "sample_count": len(index.samples),
        "condition_counts": dict(index.condition_counts),
    }


def build_analysis_manifest(
    *,
    index: RunSetIndex,
    plan_path: Path | str,
    output_root: Path | str,
    output_paths: Sequence[Path | str],
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    """Build, but do not write, a completed non-self-hashing manifest."""

    root = Path(output_root).resolve()
    plan = Path(plan_path).resolve()
    if plan != root / "analysis_plan.json":
        raise B1OfflineError("analysis plan must be output_root/analysis_plan.json")
    payload = _read_json(plan)
    if payload.get("schema_version") != ANALYSIS_PLAN_SCHEMA_VERSION:
        raise B1OfflineError("analysis plan has an unsupported schema")
    if payload.get("source_run_set", {}).get("sha256") != index.run_set_sha256:
        raise B1OfflineError("analysis plan and source run set differ")
    outputs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in output_paths:
        path = Path(value).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise B1OfflineError(f"analysis output is outside output_root: {path}") from exc
        if relative in {"analysis_plan.json", "analysis_manifest.json"} or relative in seen:
            raise B1OfflineError(f"invalid or duplicate analysis output {relative!r}")
        if not path.is_file():
            raise B1OfflineError(f"missing analysis output: {path}")
        seen.add(relative)
        outputs.append({"relative_path": relative, "sha256": sha256_file(path), "file_bytes": path.stat().st_size})
    if not outputs:
        raise B1OfflineError("completed analysis manifest requires at least one output")
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_id": payload["analysis_id"],
        "status": "completed",
        "plan_sha256": sha256_file(plan),
        "source_run_set_sha256": index.run_set_sha256,
        "sample_count": len(index.samples),
        "condition_counts": dict(index.condition_counts),
        "outputs": sorted(outputs, key=lambda item: str(item["relative_path"])),
        "warnings": [str(value) for value in warnings],
    }
