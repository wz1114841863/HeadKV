"""Atomic storage and completeness checks for V1-B2 compact traces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.head_analysis.collectors.b2_attention import SET_TYPES, B2DecodeLayerRecord, B2PrefillLayerRecord
from experiments.head_analysis.collectors.prefill_consistency import UINT32_SENTINEL, B2CollectorConfig
from experiments.head_analysis.common.io import file_entry, read_json, write_json, write_npz

B2_SCHEMA_VERSION = "headkv-v1-b2-prefill-consistency-v1"


class B2TraceWriterError(RuntimeError):
    """Raised for B2 schema, alignment, or atomic-write violations."""


def collector_config_payload(config: B2CollectorConfig) -> dict[str, Any]:
    """Return the stable, JSON-serializable B2 collection configuration."""

    return {
        "conditions": [asdict(condition) for condition in config.conditions],
        "critical_token_budgets": list(config.critical_token_budgets),
        "critical_token_max": config.critical_token_max,
        "key_block_size": config.key_block_size,
    }


def collector_config_sha256(config: B2CollectorConfig) -> str:
    """Hash the frozen B2 parameters independently of run metadata."""

    content = json.dumps(
        collector_config_payload(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class B2TraceSchema:
    num_layers: int
    num_query_heads: int
    num_kv_heads: int
    collector_config: B2CollectorConfig = B2CollectorConfig()
    schema_version: str = B2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if min(self.num_layers, self.num_query_heads, self.num_kv_heads) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")

    def to_dict(self) -> dict[str, Any]:
        config = self.collector_config
        return {
            "schema_version": self.schema_version,
            "collector_config_sha256": collector_config_sha256(config),
            "dimensions": {
                "L": self.num_layers,
                "Hq": self.num_query_heads,
                "Hkv": self.num_kv_heads,
                "G": self.num_query_heads // self.num_kv_heads,
                "M": len(config.conditions),
                "Nb": len(config.critical_token_budgets),
                "U": len(SET_TYPES),
                "Kmax": config.critical_token_max,
                "W": config.conditions[0].observation_window,
            },
            "axes": {
                "conditions": [condition.condition_id for condition in config.conditions],
                "budgets": list(config.critical_token_budgets),
                "set_types": list(SET_TYPES),
                "prefill": "[L,Hkv,M,(Kmax|W+1)]",
                "decode_prompt_mass": "[C,L,Hkv]",
                "decode_critical_mass": "[C,L,Hkv,M,Nb,U]",
            },
            "conditions": [asdict(condition) for condition in config.conditions],
            "quantile_interpolation": "linear",
            "frequency_standard_deviation": "population_ddof_0",
            "invalid_values": {
                "critical_token_position": "uint32_max",
                "critical_token_frequency": 0,
                "derived_prompt_recall_when_prompt_mass_zero": "NaN",
            },
            "probe_contract_version": "llama-sdpa-continuous-causal-v1",
        }


@dataclass(frozen=True)
class B2SampleWriteRequest:
    sample_metadata: Mapping[str, Any]
    replay_generated_ids: Sequence[int]
    prefill_predicted_token_id: int | None
    prefill_records: Sequence[B2PrefillLayerRecord]
    decode_records: Sequence[B2DecodeLayerRecord]
    decode_predicted_token_ids: Sequence[int]
    status: str = "completed"


@dataclass(frozen=True)
class B2SampleWriteResult:
    sample_id: str
    artifact_file_count: int
    logical_array_bytes: int
    artifact_file_bytes: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class B2RunValidationResult:
    status: str
    sample_count: int
    completed_sample_count: int
    failed_sample_count: int
    artifact_file_count: int
    artifact_file_bytes: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


class B2TraceWriter:
    """Write one B2 run with sample isolation and a hash manifest."""

    def __init__(
        self,
        run_dir: Path | str,
        schema: B2TraceSchema,
        *,
        decode_chunk_steps: int = 128,
    ) -> None:
        if not 1 <= decode_chunk_steps <= 128:
            raise ValueError("decode_chunk_steps must be in [1,128]")
        self.run_dir = Path(run_dir)
        self.schema = schema
        self.decode_chunk_steps = decode_chunk_steps
        self._manifest_entries: dict[str, dict[str, Any]] = {}
        self._initialized = False

    def initialize(
        self,
        run_metadata: Mapping[str, Any],
        *,
        config_snapshot: Mapping[str, Any],
    ) -> None:
        if self._initialized:
            raise B2TraceWriterError("writer is already initialized")
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise B2TraceWriterError(f"run directory is not empty: {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "samples").mkdir(exist_ok=False)
        metadata = dict(run_metadata)
        metadata.setdefault("schema_version", self.schema.schema_version)
        metadata.setdefault("status", "running")
        expected_config_hash = collector_config_sha256(self.schema.collector_config)
        snapshot = dict(config_snapshot)
        supplied_hash = snapshot.get("collector_config_sha256")
        if supplied_hash is not None and supplied_hash != expected_config_hash:
            raise B2TraceWriterError("config snapshot hash differs from the writer Schema")
        snapshot["collector_config"] = collector_config_payload(self.schema.collector_config)
        snapshot["collector_config_sha256"] = expected_config_hash
        self._write_json_and_register("schema.json", self.schema.to_dict())
        self._write_json_and_register("config_snapshot.yaml", snapshot)
        self._write_json_and_register("run_metadata.json", metadata)
        self._write_manifest()
        self._initialized = True

    def finalize(self, *, status: str, resource_summary: Mapping[str, Any]) -> None:
        self._require_initialized()
        if status not in {"completed", "partial", "failed"}:
            raise ValueError("final status must be completed, partial, or failed")
        self._write_json_and_register("resource_metrics.json", dict(resource_summary))
        metadata = self._read_json(self.run_dir / "run_metadata.json")
        metadata["status"] = status
        metadata["resource_metrics_path"] = "resource_metrics.json"
        self._write_json_and_register("run_metadata.json", metadata)
        self._write_manifest()

    def write_sample(self, request: B2SampleWriteRequest) -> B2SampleWriteResult:
        self._require_initialized()
        if request.status not in {"completed", "failed"}:
            raise B2TraceWriterError("B2 v1 sample status must be completed or failed")
        metadata = self._validate_sample_metadata(request.sample_metadata)
        sample_id = metadata["sample_id"]
        sample_dir = self.run_dir / "samples" / sample_id
        if sample_dir.exists():
            raise B2TraceWriterError(f"sample directory already exists: {sample_id}")
        replay_ids = self._uint32_ids(request.replay_generated_ids, "replay_generated_ids")
        predicted_ids = self._uint32_ids(request.decode_predicted_token_ids, "decode_predicted_token_ids")

        prefill_arrays: dict[str, np.ndarray] | None = None
        decode_chunks: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
        if request.status == "completed":
            if replay_ids.size == 0 or request.prefill_predicted_token_id is None:
                raise B2TraceWriterError("completed B2 sample requires a non-empty replay and Prefill prediction")
            if int(request.prefill_predicted_token_id) != int(replay_ids[0]):
                raise B2TraceWriterError("Prefill Top-1 differs from replay_generated_ids[0]")
            expected_decode = max(0, replay_ids.size - 1)
            if predicted_ids.size != expected_decode:
                raise B2TraceWriterError("Decode predicted-ID count differs from replay length minus one")
            if not np.array_equal(predicted_ids, replay_ids[1:]):
                raise B2TraceWriterError("teacher-forced Decode Top-1 IDs differ from the B1 replay")
            prefill_arrays = self._assemble_prefill(request.prefill_records, sample_id)
            decode_chunks = self._assemble_decode(
                request.decode_records,
                sample_id,
                replay_ids,
                predicted_ids,
                prompt_length=int(metadata["prompt_length"]),
            )
        elif request.prefill_records or request.decode_records or predicted_ids.size:
            raise B2TraceWriterError("failed samples must not masquerade partial arrays as completed trace data")

        sample_dir.mkdir(parents=False)
        local_entries: list[dict[str, Any]] = []
        replay_payload = {
            "ids": replay_ids.tolist(),
            "dtype": "uint32-little-endian",
            "sha256": hashlib.sha256(replay_ids.astype("<u4", copy=False).tobytes()).hexdigest(),
        }
        local_entries.append(self._write_json(sample_dir / "replay_generated_ids.json", replay_payload, sample_id))
        if prefill_arrays is not None:
            local_entries.append(self._write_npz(sample_dir / "prefill_consistency.npz", prefill_arrays, sample_id))
        for chunk_index, (steps, arrays) in enumerate(decode_chunks):
            local_entries.append(
                self._write_npz(
                    sample_dir / f"decode_recall_{chunk_index:03d}.npz",
                    arrays,
                    sample_id,
                    step_start=int(steps[0]),
                    step_end=int(steps[-1]),
                )
            )
        sample_output = dict(metadata)
        sample_output.update(
            {
                "status": request.status,
                "actual_generated_length": int(replay_ids.size),
                "decode_trace_length": max(0, int(replay_ids.size) - 1) if request.status == "completed" else 0,
                "prefill_predicted_token_id": request.prefill_predicted_token_id,
                "prefill_top1_match": bool(
                    request.prefill_predicted_token_id is not None
                    and replay_ids.size
                    and int(request.prefill_predicted_token_id) == int(replay_ids[0])
                ),
                "teacher_forced_all_top1_match": bool(
                    request.status == "completed" and np.array_equal(predicted_ids, replay_ids[1:])
                ),
            }
        )
        local_entries.append(self._write_json(sample_dir / "sample_metadata.json", sample_output, sample_id))
        for entry in local_entries:
            self._manifest_entries[entry["relative_path"]] = entry
        self._write_manifest()
        return B2SampleWriteResult(
            sample_id=sample_id,
            artifact_file_count=len(local_entries),
            logical_array_bytes=sum(int(entry["logical_array_bytes"]) for entry in local_entries),
            artifact_file_bytes=sum(int(entry["file_bytes"]) for entry in local_entries),
        )

    def _assemble_prefill(self, records: Sequence[B2PrefillLayerRecord], sample_id: str) -> dict[str, np.ndarray]:
        layers = self._records_by_layer(records, sample_id, decode_step=None)
        config = self.schema.collector_config
        first = layers[0]
        shared_names = (
            "candidate_token_count",
            "candidate_position_start",
            "candidate_position_end_exclusive",
        )
        for record in layers:
            self._validate_prefill_record(record)
            if record.condition_ids != first.condition_ids:
                raise B2TraceWriterError("Prefill Layers disagree on Condition order")
            for name in shared_names:
                if not np.array_equal(getattr(record, name), getattr(first, name)):
                    raise B2TraceWriterError(f"Prefill Layers disagree on {name}")
        result = {name: np.asarray(getattr(first, name)) for name in shared_names}
        for name in (
            "cv_prefill",
            "frequency_mean",
            "frequency_std",
            "binary_threshold",
            "nonzero_frequency_count",
            "frequency_histogram",
            "critical_token_positions",
            "critical_token_frequencies",
            "critical_token_valid_count",
            "metric_valid",
        ):
            result[name] = np.stack([getattr(record, name) for record in layers], axis=0)
        expected_conditions = tuple(condition.condition_id for condition in config.conditions)
        if first.condition_ids != expected_conditions:
            raise B2TraceWriterError("Prefill Condition IDs differ from the Schema")
        return result

    def _assemble_decode(
        self,
        records: Sequence[B2DecodeLayerRecord],
        sample_id: str,
        replay_ids: np.ndarray,
        predicted_ids: np.ndarray,
        *,
        prompt_length: int,
    ) -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
        by_step: dict[int, list[B2DecodeLayerRecord]] = {}
        for record in records:
            if record.sample_id != sample_id:
                raise B2TraceWriterError("Decode record belongs to another sample")
            by_step.setdefault(record.decode_step, []).append(record)
        expected_steps = list(range(max(0, replay_ids.size - 1)))
        if sorted(by_step) != expected_steps:
            raise B2TraceWriterError("Decode records do not cover replay steps exactly")

        assembled: list[tuple[int, dict[str, np.ndarray]]] = []
        for step in expected_steps:
            layers = self._records_by_layer(by_step[step], sample_id, decode_step=step)
            query_positions = {record.query_position for record in layers}
            if len(query_positions) != 1:
                raise B2TraceWriterError("Decode Layers disagree on query_position")
            query_position = query_positions.pop()
            current_prompt = query_position - step
            if current_prompt != prompt_length:
                raise B2TraceWriterError("Decode query positions violate Prompt+step mapping")
            for record in layers:
                self._validate_decode_record(record)
            assembled.append(
                (
                    step,
                    {
                        "query_positions": np.asarray(query_position, dtype=np.uint32),
                        "input_token_ids": np.asarray(replay_ids[step], dtype=np.uint32),
                        "predicted_token_ids": np.asarray(predicted_ids[step], dtype=np.uint32),
                        "teacher_forced_top1_match": np.asarray(predicted_ids[step] == replay_ids[step + 1]),
                        "prompt_attention_mass": np.stack([record.prompt_attention_mass for record in layers], axis=0),
                        "critical_attention_mass": np.stack(
                            [record.critical_attention_mass for record in layers], axis=0
                        ),
                    },
                )
            )

        chunks: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
        for start in range(0, len(assembled), self.decode_chunk_steps):
            chunk = assembled[start : start + self.decode_chunk_steps]
            steps = np.asarray([step for step, _ in chunk], dtype=np.uint32)
            arrays = {name: np.stack([entry[name] for _, entry in chunk]) for name in chunk[0][1]}
            arrays["decode_steps"] = steps
            chunks.append((steps, arrays))
        return chunks

    def _records_by_layer(self, records, sample_id: str, *, decode_step: int | None):
        if len(records) != self.schema.num_layers:
            raise B2TraceWriterError(f"expected {self.schema.num_layers} Layer records, got {len(records)}")
        by_layer = {}
        for record in records:
            if record.sample_id != sample_id:
                raise B2TraceWriterError("Layer record belongs to another sample")
            if decode_step is not None and record.decode_step != decode_step:
                raise B2TraceWriterError("Layer record has an inconsistent Decode step")
            if record.layer_id in by_layer:
                raise B2TraceWriterError(f"duplicate Layer {record.layer_id}")
            by_layer[record.layer_id] = record
        if sorted(by_layer) != list(range(self.schema.num_layers)):
            raise B2TraceWriterError("Layer IDs must be complete and zero-based")
        return [by_layer[index] for index in range(self.schema.num_layers)]

    def _validate_prefill_record(self, record: B2PrefillLayerRecord) -> None:
        config = self.schema.collector_config
        hkv = self.schema.num_kv_heads
        m = len(config.conditions)
        kmax = config.critical_token_max
        w = config.conditions[0].observation_window
        expected = {
            "candidate_token_count": (np.uint32, (m,)),
            "candidate_position_start": (np.uint32, (m,)),
            "candidate_position_end_exclusive": (np.uint32, (m,)),
            "cv_prefill": (np.float32, (hkv, m)),
            "frequency_mean": (np.float32, (hkv, m)),
            "frequency_std": (np.float32, (hkv, m)),
            "binary_threshold": (np.float32, (hkv, m)),
            "nonzero_frequency_count": (np.uint32, (hkv, m)),
            "frequency_histogram": (np.uint32, (hkv, m, w + 1)),
            "critical_token_positions": (np.uint32, (hkv, m, kmax)),
            "critical_token_frequencies": (np.uint8, (hkv, m, kmax)),
            "critical_token_valid_count": (np.uint32, (hkv, m)),
            "metric_valid": (np.dtype(bool), (hkv, m)),
        }
        for name, (dtype, shape) in expected.items():
            value = getattr(record, name)
            if value.dtype != dtype or value.shape != shape:
                raise B2TraceWriterError(f"{name} must be {np.dtype(dtype)} {shape}")
        if not record.metric_valid.all() or not np.isfinite(record.cv_prefill).all():
            raise B2TraceWriterError("completed Prefill metrics must all be valid and finite")
        for name in ("frequency_mean", "frequency_std", "binary_threshold"):
            if not np.isfinite(getattr(record, name)).all():
                raise B2TraceWriterError(f"completed {name} must be finite")
        if not np.all(record.frequency_histogram.sum(axis=-1) == record.candidate_token_count):
            raise B2TraceWriterError("frequency Histogram totals differ from candidate_token_count")
        if np.any(record.critical_token_valid_count > kmax):
            raise B2TraceWriterError("critical_token_valid_count exceeds Kmax")
        rank_axis = np.arange(kmax).reshape(1, 1, -1)
        invalid = rank_axis >= record.critical_token_valid_count[..., None]
        if not np.all(record.critical_token_positions[invalid] == UINT32_SENTINEL):
            raise B2TraceWriterError("invalid critical-token positions must use uint32 Sentinel")
        if not np.all(record.critical_token_frequencies[invalid] == 0):
            raise B2TraceWriterError("invalid critical-token frequencies must be zero")
        for kv_head in range(hkv):
            for condition_index in range(m):
                count = int(record.critical_token_valid_count[kv_head, condition_index])
                positions = record.critical_token_positions[kv_head, condition_index, :count]
                start = record.candidate_position_start[condition_index]
                end = record.candidate_position_end_exclusive[condition_index]
                if np.unique(positions).size != count or np.any(positions < start) or np.any(positions >= end):
                    raise B2TraceWriterError("valid critical-token positions violate candidate bounds/uniqueness")

    def _validate_decode_record(self, record: B2DecodeLayerRecord) -> None:
        config = self.schema.collector_config
        expected_mass = (
            self.schema.num_kv_heads,
            len(config.conditions),
            len(config.critical_token_budgets),
            len(SET_TYPES),
        )
        if record.prompt_attention_mass.dtype != np.float32 or record.prompt_attention_mass.shape != (
            self.schema.num_kv_heads,
        ):
            raise B2TraceWriterError("prompt_attention_mass has the wrong dtype/shape")
        if record.critical_attention_mass.dtype != np.float32 or record.critical_attention_mass.shape != expected_mass:
            raise B2TraceWriterError("critical_attention_mass has the wrong dtype/shape")
        tolerance = 1e-5
        if not np.isfinite(record.prompt_attention_mass).all() or not np.isfinite(record.critical_attention_mass).all():
            raise B2TraceWriterError("Decode masses must be finite")
        if np.any(record.prompt_attention_mass < -tolerance) or np.any(record.prompt_attention_mass > 1 + tolerance):
            raise B2TraceWriterError("Prompt attention mass is outside [0,1]")
        if np.any(record.critical_attention_mass < -tolerance) or np.any(
            record.critical_attention_mass > record.prompt_attention_mass[:, None, None, None] + tolerance
        ):
            raise B2TraceWriterError("critical mass violates Prompt-mass bounds")
        if np.any(
            record.critical_attention_mass[:, :, 1:, :] + tolerance < record.critical_attention_mass[:, :, :-1, :]
        ):
            raise B2TraceWriterError("critical mass is not monotonic in budget")
        if np.any(record.critical_attention_mass[:, :, :, 1] + tolerance < record.critical_attention_mass[:, :, :, 0]):
            raise B2TraceWriterError("augmented mass is below learned mass")

    @staticmethod
    def _validate_sample_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
        required = (
            "sample_id",
            "dataset",
            "task",
            "pair_id",
            "input_sha256",
            "prompt_length",
            "max_new_tokens",
            "attention_backend",
            "b1_run_set_sha256",
            "b1_generated_ids_sha256",
        )
        missing = [name for name in required if name not in metadata]
        if missing:
            raise B2TraceWriterError(f"sample metadata is missing required fields: {missing}")
        result = dict(metadata)
        sample_id = str(result["sample_id"])
        if not sample_id or Path(sample_id).name != sample_id:
            raise B2TraceWriterError("sample_id must be one non-empty path component")
        result["sample_id"] = sample_id
        if int(result["prompt_length"]) <= 0:
            raise B2TraceWriterError("prompt_length must be positive")
        return result

    @staticmethod
    def _uint32_ids(values: Sequence[int], name: str) -> np.ndarray:
        result = list(values)
        if any(not isinstance(value, (int, np.integer)) or value < 0 or value > UINT32_SENTINEL for value in result):
            raise B2TraceWriterError(f"{name} must contain uint32-compatible integers")
        return np.asarray(result, dtype=np.uint32)

    def _write_json_and_register(self, relative_path: str, data: Mapping[str, Any]) -> None:
        entry = self._write_json(self.run_dir / relative_path, data)
        self._manifest_entries[entry["relative_path"]] = entry

    def _write_json(self, path: Path, data: Mapping[str, Any], sample_id: str | None = None) -> dict[str, Any]:
        write_json(path, data, overwrite=True, allow_nan=True)
        return file_entry(path, self.run_dir, sample_id=sample_id)

    def _write_npz(
        self,
        path: Path,
        arrays: Mapping[str, np.ndarray],
        sample_id: str,
        *,
        step_start: int | None = None,
        step_end: int | None = None,
    ) -> dict[str, Any]:
        write_npz(path, arrays, overwrite=True)
        return file_entry(
            path, self.run_dir, arrays=arrays, sample_id=sample_id, step_start=step_start, step_end=step_end
        )

    def _write_manifest(self) -> None:
        payload = {"files": [self._manifest_entries[key] for key in sorted(self._manifest_entries)]}
        write_json(self.run_dir / "file_manifest.json", payload, overwrite=True)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        value = read_json(path)
        if not isinstance(value, dict):
            raise B2TraceWriterError(f"expected JSON object in {path}")
        return value

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise B2TraceWriterError("call initialize() before writing B2 traces")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise B2TraceWriterError(f"unable to read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise B2TraceWriterError(f"expected a JSON object in {path}")
    return payload


def _validate_npz_entry(run_dir: Path, entry: Mapping[str, Any]) -> dict[str, np.ndarray]:
    relative_path = str(entry["relative_path"])
    try:
        with np.load(run_dir / relative_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise B2TraceWriterError(f"unable to read NPZ {relative_path}: {exc}") from exc
    descriptors = entry.get("arrays")
    if not isinstance(descriptors, list):
        raise B2TraceWriterError(f"NPZ manifest entry lacks arrays: {relative_path}")
    expected = {
        str(item.get("name")): (list(item.get("shape", [])), str(item.get("dtype")))
        for item in descriptors
        if isinstance(item, dict)
    }
    observed = {name: (list(array.shape), str(array.dtype)) for name, array in arrays.items()}
    if observed != expected:
        raise B2TraceWriterError(f"NPZ arrays differ from the manifest: {relative_path}")
    logical_bytes = sum(array.nbytes for array in arrays.values())
    if logical_bytes != entry.get("logical_array_bytes"):
        raise B2TraceWriterError(f"NPZ logical byte count differs from the manifest: {relative_path}")
    return arrays


def validate_b2_run(
    run_dir: Path | str, *, require_completed: bool = True, verify_hashes: bool = True
) -> B2RunValidationResult:
    """Independently validate hashes, schema, alignment, and B2 invariants."""

    run_dir = Path(run_dir)
    schema = _load_json_object(run_dir / "schema.json")
    metadata = _load_json_object(run_dir / "run_metadata.json")
    resources = _load_json_object(run_dir / "resource_metrics.json")
    snapshot = _load_json_object(run_dir / "config_snapshot.yaml")
    manifest = _load_json_object(run_dir / "file_manifest.json")
    if schema.get("schema_version") != B2_SCHEMA_VERSION:
        raise B2TraceWriterError("unsupported B2 schema version")
    status = str(metadata.get("status", ""))
    if status not in {"completed", "partial", "failed"}:
        raise B2TraceWriterError("run metadata has an invalid final status")
    if require_completed and status != "completed":
        raise B2TraceWriterError("B2 run is not completed")
    if resources.get("status") != status or metadata.get("resource_metrics_path") != "resource_metrics.json":
        raise B2TraceWriterError("run/resource status provenance is inconsistent")

    config_payload = snapshot.get("collector_config")
    config_hash = snapshot.get("collector_config_sha256")
    if not isinstance(config_payload, dict) or (verify_hashes and not isinstance(config_hash, str)):
        raise B2TraceWriterError("config snapshot lacks collector configuration/hash")
    if verify_hashes:
        actual_config_hash = hashlib.sha256(
            json.dumps(config_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if actual_config_hash != config_hash or schema.get("collector_config_sha256") != config_hash:
            raise B2TraceWriterError("collector configuration hash is inconsistent")

    entries_value = manifest.get("files")
    if not isinstance(entries_value, list):
        raise B2TraceWriterError("file manifest lacks a files list")
    entries: dict[str, Mapping[str, Any]] = {}
    total_bytes = 0
    for entry in entries_value:
        if not isinstance(entry, dict) or not isinstance(entry.get("relative_path"), str):
            raise B2TraceWriterError("file manifest contains a malformed entry")
        relative_path = entry["relative_path"]
        if relative_path in entries or Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise B2TraceWriterError(f"invalid or duplicate manifest path {relative_path!r}")
        path = run_dir / relative_path
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise B2TraceWriterError(f"missing manifest artifact {relative_path}: {exc}") from exc
        if verify_hashes and (size != entry.get("file_bytes") or hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256")):
            raise B2TraceWriterError(f"artifact hash/size differs from manifest: {relative_path}")
        entries[relative_path] = entry
        total_bytes += size
    if verify_hashes:
        actual_files = {
            path.relative_to(run_dir).as_posix()
            for path in run_dir.rglob("*")
            if path.is_file() and path.name != "file_manifest.json"
        }
        if actual_files != set(entries):
            raise B2TraceWriterError("run files differ from the file manifest")

    dimensions = schema.get("dimensions")
    axes = schema.get("axes")
    if not isinstance(dimensions, dict) or not isinstance(axes, dict):
        raise B2TraceWriterError("schema lacks dimensions/axes")
    try:
        layers = int(dimensions["L"])
        kv_heads = int(dimensions["Hkv"])
        conditions = int(dimensions["M"])
        budgets = int(dimensions["Nb"])
        set_types = int(dimensions["U"])
        kmax = int(dimensions["Kmax"])
        window = int(dimensions["W"])
    except (KeyError, TypeError, ValueError) as exc:
        raise B2TraceWriterError("schema dimensions are invalid") from exc
    if len(axes.get("conditions", [])) != conditions or len(axes.get("budgets", [])) != budgets:
        raise B2TraceWriterError("schema axes disagree with dimensions")
    if tuple(axes.get("set_types", [])) != SET_TYPES or set_types != len(SET_TYPES):
        raise B2TraceWriterError("schema set-type axis is invalid")

    target_ids = metadata.get("target_sample_ids")
    if not isinstance(target_ids, list) or not target_ids or len(set(target_ids)) != len(target_ids):
        raise B2TraceWriterError("run metadata target_sample_ids is invalid")
    completed = 0
    failed = 0
    for raw_sample_id in target_ids:
        sample_id = str(raw_sample_id)
        prefix = f"samples/{sample_id}/"
        sample_entries = {key: value for key, value in entries.items() if key.startswith(prefix)}
        sample_metadata_path = f"{prefix}sample_metadata.json"
        replay_path = f"{prefix}replay_generated_ids.json"
        if sample_metadata_path not in sample_entries or replay_path not in sample_entries:
            raise B2TraceWriterError(f"{sample_id}: missing sample metadata or replay IDs")
        sample_metadata = _load_json_object(run_dir / sample_metadata_path)
        replay_payload = _load_json_object(run_dir / replay_path)
        ids = replay_payload.get("ids")
        if not isinstance(ids, list) or not ids:
            raise B2TraceWriterError(f"{sample_id}: replay IDs must be non-empty")
        if replay_payload.get("dtype") != "uint32-little-endian":
            raise B2TraceWriterError(f"{sample_id}: replay payload hash/schema is invalid")
        replay_array = np.asarray(ids, dtype=np.uint32)
        if verify_hashes:
            replay_hash = hashlib.sha256(replay_array.astype("<u4", copy=False).tobytes()).hexdigest()
            if replay_payload.get("sha256") != replay_hash:
                raise B2TraceWriterError(f"{sample_id}: replay payload hash/schema is invalid")
        sample_status = sample_metadata.get("status")
        if sample_status == "failed":
            failed += 1
            if any(path.endswith(".npz") for path in sample_entries):
                raise B2TraceWriterError(f"{sample_id}: failed sample contains formal NPZ traces")
            continue
        if sample_status != "completed":
            raise B2TraceWriterError(f"{sample_id}: invalid sample status")
        completed += 1
        if (
            sample_metadata.get("actual_generated_length") != len(ids)
            or sample_metadata.get("decode_trace_length") != len(ids) - 1
            or sample_metadata.get("prefill_predicted_token_id") != ids[0]
            or sample_metadata.get("prefill_top1_match") is not True
            or sample_metadata.get("teacher_forced_all_top1_match") is not True
        ):
            raise B2TraceWriterError(f"{sample_id}: replay/sample completion fields are inconsistent")

        prefill_path = f"{prefix}prefill_consistency.npz"
        if prefill_path not in sample_entries:
            raise B2TraceWriterError(f"{sample_id}: missing Prefill trace")
        prefill = _validate_npz_entry(run_dir, sample_entries[prefill_path])
        expected_prefill = {
            "candidate_token_count": (np.dtype("uint32"), (conditions,)),
            "candidate_position_start": (np.dtype("uint32"), (conditions,)),
            "candidate_position_end_exclusive": (np.dtype("uint32"), (conditions,)),
            "cv_prefill": (np.dtype("float32"), (layers, kv_heads, conditions)),
            "frequency_mean": (np.dtype("float32"), (layers, kv_heads, conditions)),
            "frequency_std": (np.dtype("float32"), (layers, kv_heads, conditions)),
            "binary_threshold": (np.dtype("float32"), (layers, kv_heads, conditions)),
            "nonzero_frequency_count": (np.dtype("uint32"), (layers, kv_heads, conditions)),
            "frequency_histogram": (np.dtype("uint32"), (layers, kv_heads, conditions, window + 1)),
            "critical_token_positions": (np.dtype("uint32"), (layers, kv_heads, conditions, kmax)),
            "critical_token_frequencies": (np.dtype("uint8"), (layers, kv_heads, conditions, kmax)),
            "critical_token_valid_count": (np.dtype("uint32"), (layers, kv_heads, conditions)),
            "metric_valid": (np.dtype("bool"), (layers, kv_heads, conditions)),
        }
        if set(prefill) != set(expected_prefill):
            raise B2TraceWriterError(f"{sample_id}: Prefill field set is invalid")
        for name, (dtype, shape) in expected_prefill.items():
            if prefill[name].dtype != dtype or prefill[name].shape != shape:
                raise B2TraceWriterError(f"{sample_id}: Prefill {name} has an invalid dtype/shape")
        if not prefill["metric_valid"].all() or not np.isfinite(prefill["cv_prefill"]).all():
            raise B2TraceWriterError(f"{sample_id}: Prefill metrics are not all valid")
        if not np.all(prefill["frequency_histogram"].sum(axis=-1) == prefill["candidate_token_count"]):
            raise B2TraceWriterError(f"{sample_id}: Prefill Histogram totals are invalid")

        decode_entries = sorted(
            (value for key, value in sample_entries.items() if key.startswith(f"{prefix}decode_recall_")),
            key=lambda value: int(value.get("step_start", -1)),
        )
        expected_step = 0
        for entry in decode_entries:
            arrays = _validate_npz_entry(run_dir, entry)
            chunk_steps = arrays.get("decode_steps")
            if chunk_steps is None or chunk_steps.dtype != np.uint32 or chunk_steps.ndim != 1:
                raise B2TraceWriterError(f"{sample_id}: invalid Decode step array")
            count = len(chunk_steps)
            expected_steps = np.arange(expected_step, expected_step + count, dtype=np.uint32)
            if not np.array_equal(chunk_steps, expected_steps):
                raise B2TraceWriterError(f"{sample_id}: Decode steps are not continuous")
            expected_shapes = {
                "query_positions": (np.uint32, (count,)),
                "input_token_ids": (np.uint32, (count,)),
                "predicted_token_ids": (np.uint32, (count,)),
                "teacher_forced_top1_match": (np.dtype(bool), (count,)),
                "prompt_attention_mass": (np.float32, (count, layers, kv_heads)),
                "critical_attention_mass": (
                    np.float32,
                    (count, layers, kv_heads, conditions, budgets, set_types),
                ),
            }
            if set(arrays) != {"decode_steps", *expected_shapes}:
                raise B2TraceWriterError(f"{sample_id}: Decode field set is invalid")
            for name, (dtype, shape) in expected_shapes.items():
                if arrays[name].dtype != np.dtype(dtype) or arrays[name].shape != shape:
                    raise B2TraceWriterError(f"{sample_id}: Decode {name} has an invalid dtype/shape")
            prompt_length = int(sample_metadata["prompt_length"])
            if (
                not np.array_equal(arrays["query_positions"], prompt_length + expected_steps)
                or not np.array_equal(arrays["input_token_ids"], replay_array[expected_steps])
                or not np.array_equal(arrays["predicted_token_ids"], replay_array[expected_steps + 1])
                or not arrays["teacher_forced_top1_match"].all()
            ):
                raise B2TraceWriterError(f"{sample_id}: Decode Token/position alignment is invalid")
            prompt_mass = arrays["prompt_attention_mass"]
            critical_mass = arrays["critical_attention_mass"]
            tolerance = 1e-5
            if (
                not np.isfinite(prompt_mass).all()
                or not np.isfinite(critical_mass).all()
                or np.any(prompt_mass < -tolerance)
                or np.any(prompt_mass > 1 + tolerance)
                or np.any(critical_mass < -tolerance)
                or np.any(critical_mass > prompt_mass[..., None, None, None] + tolerance)
                or np.any(critical_mass[..., 1:, :] + tolerance < critical_mass[..., :-1, :])
                or np.any(critical_mass[..., 1] + tolerance < critical_mass[..., 0])
            ):
                raise B2TraceWriterError(f"{sample_id}: Decode mass invariants are invalid")
            expected_step += count
        if expected_step != len(ids) - 1:
            raise B2TraceWriterError(f"{sample_id}: Decode chunks do not cover the replay")

    if status == "completed" and (completed != len(target_ids) or failed):
        raise B2TraceWriterError("completed run does not contain only completed target samples")
    if resources.get("samples_completed") != completed or resources.get("samples_failed") != failed:
        raise B2TraceWriterError("resource sample counts differ from artifact statuses")
    return B2RunValidationResult(
        status=status,
        sample_count=len(target_ids),
        completed_sample_count=completed,
        failed_sample_count=failed,
        artifact_file_count=len(entries),
        artifact_file_bytes=total_bytes,
    )


__all__ = [
    "B2_SCHEMA_VERSION",
    "B2SampleWriteRequest",
    "B2SampleWriteResult",
    "B2RunValidationResult",
    "B2TraceSchema",
    "B2TraceWriter",
    "B2TraceWriterError",
    "collector_config_payload",
    "collector_config_sha256",
    "validate_b2_run",
]
