"""Atomic NPZ/JSON storage for compact B1 attention traces.

The writer is intentionally independent of model execution.  It accepts only
CPU ``CollectedAttention`` records, validates their Layer/Step alignment, and
serializes the fixed B1 schema.  It never receives Q/K tensors or model hooks.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.head_analysis.collectors.attention_stats import (
    FEATURE_NAMES,
    UINT32_SENTINEL,
    CollectedAttention,
)
from kvpress.context import KVPhase


SCHEMA_VERSION = "headkv-v1-b1-attention-v1"
PAGE_SCOPES = ("all_visible", "prompt_only")


class TraceWriterError(RuntimeError):
    """Raised for B1 schema, ordering, or atomic-write contract violations."""


@dataclass(frozen=True)
class AttentionTraceSchema:
    """Dimensions and optional-array policy frozen for one B1 run."""

    num_layers: int
    num_query_heads: int
    num_kv_heads: int
    prefill_window: int = 64
    page_size: int = 16
    token_topk: int = 64
    page_topk: int = 32
    save_token_topk: bool = True
    save_token_topk_weights: bool = False
    save_page_topk_scores: bool = False
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "num_layers",
            "num_query_heads",
            "num_kv_heads",
            "prefill_window",
            "page_size",
            "token_topk",
            "page_topk",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("num_query_heads must be divisible by num_kv_heads for GQA")
        if self.save_token_topk_weights and not self.save_token_topk:
            raise ValueError("token Top-K weights require token Top-K positions")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_names": list(FEATURE_NAMES),
            "feature_dtypes": {"scalar_features": "float32", "token_topk_positions": "uint32"},
            "dimensions": {
                "L": self.num_layers,
                "Hq": self.num_query_heads,
                "Hkv": self.num_kv_heads,
                "F": len(FEATURE_NAMES),
            },
            "axis_conventions": {
                "prefill_scalar_features": "[W_eff,L,Hq,F]",
                "decode_scalar_features": "[C,L,Hq,F]",
                "decode_page_scope": "[C,L,Hkv,S]",
                "scope_axis": list(PAGE_SCOPES),
            },
            "prefill_window": self.prefill_window,
            "page_size": self.page_size,
            "token_topk": self.token_topk,
            "page_topk": self.page_topk,
            "page_scopes": list(PAGE_SCOPES),
            "distance_bins": [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
            "invalid_value_rules": {
                "token_or_page_id": "uint32_max",
                "invalid_page_coverage": "NaN with page_scope_valid=false",
                "absent_optional_array": "not_created",
            },
            "probe_contract_version": "llama-sdpa-continuous-causal-v1",
            "optional_arrays": {
                "save_token_topk": self.save_token_topk,
                "save_token_topk_weights": self.save_token_topk_weights,
                "save_page_topk_scores": self.save_page_topk_scores,
            },
        }


@dataclass(frozen=True)
class SampleWriteRequest:
    """All compact records necessary to write one isolated B1 sample."""

    sample_metadata: Mapping[str, Any]
    generated_ids: Sequence[int]
    prefill_records: Sequence[CollectedAttention]
    decode_records: Sequence[CollectedAttention]
    eos_reached: bool
    generation_stop_reason: str
    status: str = "completed"


@dataclass(frozen=True)
class SampleWriteResult:
    """Byte accounting for one successfully materialized sample artifact."""

    sample_id: str
    artifact_file_count: int
    logical_array_bytes: int
    artifact_file_bytes: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "sample_id": self.sample_id,
            "artifact_file_count": self.artifact_file_count,
            "logical_array_bytes": self.logical_array_bytes,
            "artifact_file_bytes": self.artifact_file_bytes,
        }


class AttentionTraceWriter:
    """Write one run's B1 traces with sample isolation and manifest hashes."""

    def __init__(
        self,
        run_dir: Path | str,
        schema: AttentionTraceSchema,
        *,
        decode_chunk_steps: int = 128,
    ) -> None:
        if not 1 <= decode_chunk_steps <= 128:
            raise ValueError("decode_chunk_steps must be in [1, 128]")
        self.run_dir = Path(run_dir)
        self.schema = schema
        self.decode_chunk_steps = decode_chunk_steps
        self._manifest_entries: dict[str, dict[str, Any]] = {}
        self._initialized = False

    def initialize(
        self,
        run_metadata: Mapping[str, Any],
        *,
        config_snapshot: Mapping[str, Any] | None = None,
    ) -> None:
        """Create immutable-schema run files before the first sample write."""

        if self._initialized:
            raise TraceWriterError("writer is already initialized")
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise TraceWriterError(f"run directory is not empty: {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "samples").mkdir(exist_ok=False)

        metadata = dict(run_metadata)
        metadata.setdefault("schema_version", self.schema.schema_version)
        metadata.setdefault("status", "running")
        self._write_json_and_register("schema.json", self.schema.to_dict())
        # JSON is a valid YAML subset, so this remains portable without adding
        # a YAML dependency solely for a configuration snapshot.
        self._write_json_and_register("config_snapshot.yaml", dict(config_snapshot or {}))
        self._write_json_and_register("run_metadata.json", metadata)
        self._write_manifest()
        self._initialized = True

    def finalize(self, *, status: str, resource_summary: Mapping[str, Any] | None = None) -> None:
        """Update run status and preserve the corresponding metadata hash."""

        self._require_initialized()
        if status not in {"completed", "partial", "failed"}:
            raise ValueError("final status must be completed, partial, or failed")
        metadata_path = self.run_dir / "run_metadata.json"
        metadata = self._read_json(metadata_path)
        metadata["status"] = status
        if resource_summary is not None:
            metadata["resource_summary"] = dict(resource_summary)
        self._write_json_and_register("run_metadata.json", metadata)
        self._write_manifest()

    def write_sample(self, request: SampleWriteRequest) -> SampleWriteResult:
        """Validate and atomically write one sample without rewriting the prompt."""

        self._require_initialized()
        if request.status not in {"completed", "partial", "failed"}:
            raise TraceWriterError("sample status must be completed, partial, or failed")
        metadata = self._validate_sample_metadata(request.sample_metadata)
        sample_id = str(metadata["sample_id"])
        sample_dir = self.run_dir / "samples" / sample_id
        if sample_dir.exists():
            raise TraceWriterError(f"sample directory already exists: {sample_id}")

        generated_ids = self._normalized_generated_ids(request.generated_ids)
        if generated_ids.size > int(metadata["max_new_tokens"]):
            raise TraceWriterError("generated_ids exceeds sample max_new_tokens")
        prefill_arrays = self._assemble_prefill(
            request.prefill_records, sample_id, required=request.status == "completed"
        )
        decode_chunks, captured_steps = self._assemble_decode(request.decode_records, sample_id)
        expected_trace_length = max(0, int(generated_ids.size) - 1)
        if request.status == "completed" and captured_steps != list(range(expected_trace_length)):
            raise TraceWriterError(
                "completed sample Decode records must cover exactly steps "
                f"0..{max(-1, expected_trace_length - 1)}, got {captured_steps}"
            )

        sample_dir.mkdir(parents=False)
        local_entries: list[dict[str, Any]] = []
        try:
            generated_payload = {
                "ids": generated_ids.tolist(),
                "dtype": "uint32-little-endian",
                "sha256": hashlib.sha256(generated_ids.astype("<u4", copy=False).tobytes()).hexdigest(),
            }
            local_entries.append(
                self._write_json(sample_dir / "generated_ids.json", generated_payload, sample_id=sample_id)
            )
            if prefill_arrays is not None:
                local_entries.append(
                    self._write_npz(sample_dir / "prefill_stats.npz", prefill_arrays, sample_id=sample_id)
                )
            for chunk_index, (steps, arrays) in enumerate(decode_chunks):
                local_entries.append(
                    self._write_npz(
                        sample_dir / f"decode_stats_{chunk_index:03d}.npz",
                        arrays,
                        sample_id=sample_id,
                        step_start=int(steps[0]),
                        step_end=int(steps[-1]),
                    )
                )

            actual_trace_length = len(captured_steps)
            sample_output = dict(metadata)
            sample_output.update(
                {
                    "actual_generated_length": int(generated_ids.size),
                    "decode_trace_length": actual_trace_length,
                    "eos_reached": bool(request.eos_reached),
                    "generation_stop_reason": request.generation_stop_reason,
                    "status": request.status,
                    "prefill_query_positions": (
                        [] if prefill_arrays is None else prefill_arrays["query_positions"].tolist()
                    ),
                }
            )
            local_entries.append(
                self._write_json(sample_dir / "sample_metadata.json", sample_output, sample_id=sample_id)
            )
        except Exception:
            # Individual files are atomic.  Keep any successfully written
            # diagnostics out of the manifest and let the caller mark/retry the
            # sample explicitly rather than falsely declaring it complete.
            raise

        for entry in local_entries:
            self._manifest_entries[entry["relative_path"]] = entry
        self._write_manifest()
        return SampleWriteResult(
            sample_id=sample_id,
            artifact_file_count=len(local_entries),
            logical_array_bytes=sum(int(entry["logical_array_bytes"]) for entry in local_entries),
            artifact_file_bytes=sum(int(entry["file_bytes"]) for entry in local_entries),
        )

    def _assemble_prefill(
        self, records: Sequence[CollectedAttention], sample_id: str, *, required: bool
    ) -> dict[str, np.ndarray] | None:
        if not records:
            if required:
                raise TraceWriterError("completed sample is missing Prefill records")
            return None
        layers = self._records_by_layer(records, sample_id, phase=KVPhase.PREFILL, decode_step=None)
        first = layers[0]
        query_positions = self._uint32_vector(first.query_positions, "prefill query_positions")
        visible_count = self._uint32_vector(first.visible_token_count, "prefill visible_token_count")
        if query_positions.size > self.schema.prefill_window:
            raise TraceWriterError("Prefill record exceeds schema prefill_window")
        scalar_by_layer = []
        for record in layers:
            self._validate_common_record(record, sample_id, phase=KVPhase.PREFILL)
            if not np.array_equal(record.query_positions, query_positions) or not np.array_equal(
                record.visible_token_count, visible_count
            ):
                raise TraceWriterError("Prefill Layer records disagree on query positions or visibility")
            scalar_by_layer.append(self._scalar_array(record, query_positions.size))
        return {
            "query_positions": query_positions,
            "visible_token_count": visible_count,
            "scalar_features": np.stack(scalar_by_layer, axis=1).astype(np.float32, copy=False),
        }

    def _assemble_decode(
        self, records: Sequence[CollectedAttention], sample_id: str
    ) -> tuple[list[tuple[np.ndarray, dict[str, np.ndarray]]], list[int]]:
        by_step: dict[int, list[CollectedAttention]] = {}
        for record in records:
            if record.sample_id != sample_id or record.phase is not KVPhase.DECODE or record.decode_step is None:
                raise TraceWriterError("Decode records must match the sample and declare one decode_step")
            by_step.setdefault(int(record.decode_step), []).append(record)
        steps = sorted(by_step)
        if steps and steps != list(range(len(steps))):
            raise TraceWriterError(f"Decode steps must be continuous from zero, got {steps}")

        assembled: list[tuple[int, dict[str, np.ndarray]]] = []
        for step in steps:
            layers = self._records_by_layer(by_step[step], sample_id, phase=KVPhase.DECODE, decode_step=step)
            assembled.append((step, self._assemble_decode_step(layers, sample_id, step)))

        chunks: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
        for start in range(0, len(assembled), self.decode_chunk_steps):
            chunk = assembled[start : start + self.decode_chunk_steps]
            chunk_steps = np.asarray([step for step, _ in chunk], dtype=np.uint32)
            keys = tuple(chunk[0][1])
            arrays = {key: np.stack([entry[key] for _, entry in chunk], axis=0) for key in keys}
            arrays["decode_steps"] = chunk_steps
            chunks.append((chunk_steps, arrays))
        return chunks, steps

    def _assemble_decode_step(
        self, layers: Sequence[CollectedAttention], sample_id: str, step: int
    ) -> dict[str, np.ndarray]:
        first = layers[0]
        self._validate_common_record(first, sample_id, phase=KVPhase.DECODE)
        if first.query_positions.shape != (1,) or first.visible_token_count.shape != (1,):
            raise TraceWriterError("B1 Decode record must contain exactly one query row")
        query_position = int(first.query_positions[0])
        visible_count = int(first.visible_token_count[0])
        if query_position + 1 != visible_count:
            raise TraceWriterError("B1 Decode visible_token_count must equal query_position + 1")
        if first.decode_step != step:
            raise TraceWriterError("Decode record has inconsistent step")

        scalar_by_layer = []
        token_by_layer = []
        all_visible_by_layer = []
        prompt_only_by_layer = []
        scope_mass_by_layer = []
        coverage_by_layer = []
        valid_by_layer = []
        all_candidate_counts: list[int] = []
        prompt_candidate_counts: list[int] = []
        for record in layers:
            self._validate_common_record(record, sample_id, phase=KVPhase.DECODE)
            if (
                record.decode_step != step
                or record.query_positions.tolist() != [query_position]
                or record.visible_token_count.tolist() != [visible_count]
            ):
                raise TraceWriterError("Decode Layer records disagree on step, query position, or visibility")
            scalar_by_layer.append(self._scalar_array(record, 1)[0])
            if self.schema.save_token_topk:
                if record.token_topk_positions is None:
                    raise TraceWriterError("schema requires token_topk_positions")
                token_by_layer.append(self._token_topk_array(record)[0])
            elif record.token_topk_positions is not None:
                raise TraceWriterError("schema disables token_topk_positions")
            page_arrays = self._page_arrays(record)
            all_visible_by_layer.append(page_arrays["page_topk_ids_all_visible"][0])
            prompt_only_by_layer.append(page_arrays["page_topk_ids_prompt_only"][0])
            scope_mass_by_layer.append(page_arrays["page_scope_mass"][0])
            coverage_by_layer.append(page_arrays["page_top32_coverage"][0])
            valid_by_layer.append(page_arrays["page_scope_valid"][0])
            all_candidate_counts.append(int(page_arrays["page_candidate_count_all_visible"][0]))
            prompt_candidate_counts.append(int(page_arrays["page_candidate_count_prompt_only"][0]))

        if len(set(all_candidate_counts)) != 1 or len(set(prompt_candidate_counts)) != 1:
            raise TraceWriterError("B1 requires shared Page candidate counts across all Decode Layers")
        expected_all = (visible_count + self.schema.page_size - 1) // self.schema.page_size
        prompt_length = int(first.query_positions[0]) - step
        expected_prompt = (prompt_length + self.schema.page_size - 1) // self.schema.page_size
        if all_candidate_counts[0] != expected_all or prompt_candidate_counts[0] != expected_prompt:
            raise TraceWriterError("Page candidate counts violate continuous causal B1 mapping")

        arrays = {
            "query_positions": np.asarray(query_position, dtype=np.uint32),
            "visible_token_count": np.asarray(visible_count, dtype=np.uint32),
            "scalar_features": np.stack(scalar_by_layer, axis=0).astype(np.float32, copy=False),
            "page_topk_ids_all_visible": np.stack(all_visible_by_layer, axis=0).astype(np.uint32, copy=False),
            "page_topk_ids_prompt_only": np.stack(prompt_only_by_layer, axis=0).astype(np.uint32, copy=False),
            "page_candidate_count_all_visible": np.asarray(all_candidate_counts[0], dtype=np.uint32),
            "page_candidate_count_prompt_only": np.asarray(prompt_candidate_counts[0], dtype=np.uint32),
            "page_scope_mass": np.stack(scope_mass_by_layer, axis=0).astype(np.float32, copy=False),
            "page_top32_coverage": np.stack(coverage_by_layer, axis=0).astype(np.float32, copy=False),
            "page_scope_valid": np.stack(valid_by_layer, axis=0).astype(bool, copy=False),
        }
        if self.schema.save_token_topk:
            arrays["token_topk_positions"] = np.stack(token_by_layer, axis=0).astype(np.uint32, copy=False)
        return arrays

    def _records_by_layer(
        self,
        records: Sequence[CollectedAttention],
        sample_id: str,
        *,
        phase: KVPhase,
        decode_step: int | None,
    ) -> list[CollectedAttention]:
        if len(records) != self.schema.num_layers:
            raise TraceWriterError(f"expected {self.schema.num_layers} {phase.value} Layer records, got {len(records)}")
        by_layer: dict[int, CollectedAttention] = {}
        for record in records:
            if record.sample_id != sample_id or record.phase is not phase or record.decode_step != decode_step:
                raise TraceWriterError("record phase/sample/decode_step does not match its write group")
            if record.layer_id in by_layer:
                raise TraceWriterError(f"duplicate Layer {record.layer_id} in one attention event")
            by_layer[record.layer_id] = record
        if sorted(by_layer) != list(range(self.schema.num_layers)):
            raise TraceWriterError("Layer IDs must be complete and zero-based")
        return [by_layer[layer_id] for layer_id in range(self.schema.num_layers)]

    def _validate_common_record(self, record: CollectedAttention, sample_id: str, *, phase: KVPhase) -> None:
        if record.sample_id != sample_id or record.phase is not phase:
            raise TraceWriterError("record does not match target sample/phase")
        if record.query_positions.dtype != np.uint32 or record.visible_token_count.dtype != np.uint32:
            raise TraceWriterError("position arrays must be uint32")
        if record.scalar_features.dtype != np.float32:
            raise TraceWriterError("scalar_features must be float32")

    def _scalar_array(self, record: CollectedAttention, query_count: int) -> np.ndarray:
        expected = (query_count, self.schema.num_query_heads, len(FEATURE_NAMES))
        if record.scalar_features.shape != expected:
            raise TraceWriterError(f"scalar_features shape must be {expected}, got {record.scalar_features.shape}")
        if not np.isfinite(record.scalar_features).all():
            raise TraceWriterError("scalar_features must be finite")
        return record.scalar_features

    def _token_topk_array(self, record: CollectedAttention) -> np.ndarray:
        assert record.token_topk_positions is not None
        expected = (1, self.schema.num_query_heads, self.schema.token_topk)
        if record.token_topk_positions.dtype != np.uint32 or record.token_topk_positions.shape != expected:
            raise TraceWriterError(f"token_topk_positions shape/dtype must be uint32 {expected}")
        return record.token_topk_positions

    def _page_arrays(self, record: CollectedAttention) -> dict[str, np.ndarray]:
        names = (
            "page_topk_ids_all_visible",
            "page_topk_ids_prompt_only",
            "page_candidate_count_all_visible",
            "page_candidate_count_prompt_only",
            "page_scope_mass",
            "page_top32_coverage",
            "page_scope_valid",
        )
        arrays = {name: getattr(record, name) for name in names}
        if any(value is None for value in arrays.values()):
            raise TraceWriterError("Decode Page arrays are mandatory")
        result = {name: value for name, value in arrays.items() if value is not None}
        expected_topk = (1, self.schema.num_kv_heads, self.schema.page_topk)
        for name in ("page_topk_ids_all_visible", "page_topk_ids_prompt_only"):
            if result[name].dtype != np.uint32 or result[name].shape != expected_topk:
                raise TraceWriterError(f"{name} shape/dtype must be uint32 {expected_topk}")
        for name in ("page_candidate_count_all_visible", "page_candidate_count_prompt_only"):
            if result[name].dtype != np.uint32 or result[name].shape != (1,):
                raise TraceWriterError(f"{name} shape/dtype must be uint32 (1,)")
        expected_scope = (1, self.schema.num_kv_heads, len(PAGE_SCOPES))
        if result["page_scope_mass"].dtype != np.float32 or result["page_scope_mass"].shape != expected_scope:
            raise TraceWriterError(f"page_scope_mass shape/dtype must be float32 {expected_scope}")
        if result["page_top32_coverage"].dtype != np.float32 or result["page_top32_coverage"].shape != expected_scope:
            raise TraceWriterError(f"page_top32_coverage shape/dtype must be float32 {expected_scope}")
        if result["page_scope_valid"].dtype != bool or result["page_scope_valid"].shape != expected_scope:
            raise TraceWriterError(f"page_scope_valid shape/dtype must be bool {expected_scope}")
        valid = result["page_scope_valid"]
        coverage = result["page_top32_coverage"]
        if not np.isnan(coverage[~valid]).all() or not np.isfinite(coverage[valid]).all():
            raise TraceWriterError("invalid Page coverage must be NaN and valid coverage must be finite")
        return result

    @staticmethod
    def _uint32_vector(array: np.ndarray, name: str) -> np.ndarray:
        if array.dtype != np.uint32 or array.ndim != 1:
            raise TraceWriterError(f"{name} must be a one-dimensional uint32 array")
        return array

    def _validate_sample_metadata(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        required = (
            "sample_id",
            "dataset",
            "task",
            "pair_id",
            "input_sha256",
            "prompt_length",
            "max_new_tokens",
            "attention_backend",
        )
        missing = [name for name in required if name not in metadata]
        if missing:
            raise TraceWriterError(f"sample metadata is missing required fields: {missing}")
        result = dict(metadata)
        sample_id = str(result["sample_id"])
        if not sample_id or Path(sample_id).name != sample_id:
            raise TraceWriterError("sample_id must be a non-empty single path component")
        if int(result["prompt_length"]) <= 0 or int(result["max_new_tokens"]) < 0:
            raise TraceWriterError("prompt_length must be positive and max_new_tokens non-negative")
        return result

    @staticmethod
    def _normalized_generated_ids(generated_ids: Sequence[int]) -> np.ndarray:
        values = list(generated_ids)
        if any(not isinstance(value, (int, np.integer)) or value < 0 or value > UINT32_SENTINEL for value in values):
            raise TraceWriterError("generated_ids must be uint32-compatible non-negative integers")
        return np.asarray(values, dtype=np.uint32)

    def _write_json_and_register(self, relative_path: str, data: Mapping[str, Any]) -> None:
        entry = self._write_json(self.run_dir / relative_path, data)
        self._manifest_entries[entry["relative_path"]] = entry

    def _write_json(self, path: Path, data: Mapping[str, Any], *, sample_id: str | None = None) -> dict[str, Any]:
        self._atomic_write_bytes(
            path, json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
        )
        return self._file_entry(path, sample_id=sample_id)

    def _write_npz(
        self,
        path: Path,
        arrays: Mapping[str, np.ndarray],
        *,
        sample_id: str,
        step_start: int | None = None,
        step_end: int | None = None,
    ) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            try:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
                os.replace(temporary_path, path)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
        return self._file_entry(path, sample_id=sample_id, arrays=arrays, step_start=step_start, step_end=step_end)

    def _write_manifest(self) -> None:
        payload = {"files": [self._manifest_entries[key] for key in sorted(self._manifest_entries)]}
        self._atomic_write_bytes(
            self.run_dir / "file_manifest.json", json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )

    def _file_entry(
        self,
        path: Path,
        *,
        sample_id: str | None = None,
        arrays: Mapping[str, np.ndarray] | None = None,
        step_start: int | None = None,
        step_end: int | None = None,
    ) -> dict[str, Any]:
        content = path.read_bytes()
        entry: dict[str, Any] = {
            "relative_path": path.relative_to(self.run_dir).as_posix(),
            "sha256": hashlib.sha256(content).hexdigest(),
            "file_bytes": len(content),
            "logical_array_bytes": 0 if arrays is None else int(sum(array.nbytes for array in arrays.values())),
        }
        if arrays is not None:
            entry["arrays"] = [
                {"name": name, "shape": list(array.shape), "dtype": str(array.dtype)} for name, array in arrays.items()
            ]
        if sample_id is not None:
            entry["sample_id"] = sample_id
        if step_start is not None and step_end is not None:
            entry["step_start"] = step_start
            entry["step_end"] = step_end
        return entry

    @staticmethod
    def _atomic_write_bytes(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            try:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                os.replace(temporary_path, path)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise TraceWriterError(f"expected JSON object in {path}")
        return value

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise TraceWriterError("call initialize() before writing traces")
