"""Read-only, fail-closed source index for the frozen B2 offline analysis.

Stage 1 validates sources and returns tensor-free coordinates. It computes no
research summaries, loads no model, and writes no files.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from zipfile import BadZipFile

import numpy as np

from experiments.head_analysis.common.io import canonical_json_sha256, read_json, sha256_file
from experiments.head_analysis.storage.b2_trace_writer import B2TraceSchema, validate_b2_run

B2_RUN_SET_SCHEMA = "headkv-v1-b2-run-set-v1"
B2_PLAN_SCHEMA = "headkv-v1-b2-offline-plan-v1"
DEFAULT_ANALYSIS_PARAMETERS = {
    "primary_condition": "paper_token_primary",
    "primary_budget": 512,
    "primary_set_type": "learned_middle",
    "primary_recall": "recall_prompt",
    "primary_correlation": "layer_centered_spearman",
    "weighting": "sample_equal",
    "bootstrap_replicates": 2000,
    "bootstrap_seed": 20261110,
    "long_decode_min_steps": 32,
    "decode_segments": 4,
    "progress_bins": 20,
}


class B2SourceError(RuntimeError):
    """An artifact cannot enter the frozen B2 analysis."""


@dataclass(frozen=True)
class B2SampleIndex:
    sample_id: str
    dataset: str
    task: str
    pair_id: str | None
    target_prompt_length: int
    prompt_length: int
    input_sha256: str
    generated_ids_sha256: str
    generated_length: int
    decode_length: int
    batch_id: str
    sample_dir: Path
    decode_chunks: tuple[Path, ...]

    @property
    def condition(self) -> str:
        return f"{self.dataset}:{self.target_prompt_length // 1024}k"


@dataclass(frozen=True)
class B2BatchIndex:
    batch_id: str
    run_dir: Path
    sample_ids: tuple[str, ...]
    file_manifest_sha256: str
    artifact_file_count: int
    artifact_file_bytes: int


@dataclass(frozen=True)
class B2RunSetIndex:
    manifest_path: Path
    source_root: Path
    run_set_sha256: str
    payload: Mapping[str, Any]
    samples: Mapping[str, B2SampleIndex]
    batches: tuple[B2BatchIndex, ...]
    validation_report_path: Path
    b1_replay_path: Path
    b1_analysis_path: Path
    verified_b1_analysis_outputs: int | None

    @property
    def condition_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(sample.condition for sample in self.samples.values()).items()))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B2SourceError(message)


def _json(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    _require(isinstance(payload, dict), f"{path}: expected JSON object")
    return payload


def _path(root: Path, value: str) -> Path:
    _require(isinstance(value, str) and bool(value), "missing source path")
    path = Path(value)
    _require(not path.is_absolute() and ".." not in path.parts, f"unsafe source path: {value}")
    resolved = (root / path).resolve()
    _require(resolved.is_relative_to(root), f"source path escapes project root: {value}")
    return resolved


def _reference(root: Path, entry: Mapping[str, Any]) -> Path:
    path = _path(root, entry["path"])
    _require(path.is_file(), f"{path}: missing source file")
    return path


def _batch_file(run: Path, name: str) -> Path:
    path = _path(run, name)
    _require(path.is_file(), f"{path}: missing source file")
    return path


def _ids(payload: Mapping[str, Any]) -> list[int]:
    ids = payload["ids"]
    _require(
        isinstance(ids, list) and bool(ids) and all(type(value) is int and 0 <= value < 2**32 for value in ids),
        "generated IDs must be non-empty uint32 integers",
    )
    _require(payload["dtype"] == "uint32-little-endian", "unsupported generated-ID encoding")
    return ids


def _unique_entries(entries: list[dict], key: str, label: str) -> dict[str, dict]:
    result = {}
    _require(isinstance(entries, list), f"{label}: expected list")
    for entry in entries:
        _require(isinstance(entry, dict), f"{label}: expected object")
        value = entry[key]
        _require(isinstance(value, str) and value and value not in result, f"{label}: duplicate/invalid {key}")
        result[value] = entry
    return result


def _validate_prefill(path: Path, prompt_length: int, schema: Mapping[str, Any]) -> None:
    """Check persisted sufficient statistics, without reconstructing Attention."""

    with np.load(path, allow_pickle=False) as arrays:
        starts = np.asarray([c["initial_exclusion"] for c in schema["conditions"]], dtype=np.int64)
        ends = prompt_length - np.asarray([c["recent_exclusion"] for c in schema["conditions"]])
        candidates = ends - starts
        for name, expected in (
            ("candidate_position_start", starts),
            ("candidate_position_end_exclusive", ends),
            ("candidate_token_count", candidates),
        ):
            _require(np.array_equal(arrays[name], expected) and (candidates > 0).all(), f"{path}: {name} differs")
        histogram = arrays["frequency_histogram"].astype(np.float64)
        axis = np.arange(schema["dimensions"]["W"] + 1, dtype=np.float64)
        mean = (histogram * axis).sum(-1) / candidates
        std = np.sqrt(np.maximum(0, (histogram * axis**2).sum(-1) / candidates - mean**2))
        _require((mean > 0).all(), f"{path}: zero mean for completed CV")
        for name, expected in (("frequency_mean", mean), ("frequency_std", std), ("cv_prefill", std / mean)):
            _require(
                np.isfinite(arrays[name]).all() and np.allclose(arrays[name], expected, rtol=2e-5, atol=2e-6),
                f"{path}: inconsistent {name}",
            )
        _require(
            np.isfinite(arrays["binary_threshold"]).all() and (arrays["binary_threshold"] >= 0).all(),
            f"{path}: invalid binary threshold",
        )
        nonzero = candidates - histogram[..., 0]
        counts = arrays["critical_token_valid_count"]
        kmax = schema["dimensions"]["Kmax"]
        _require(
            np.array_equal(arrays["nonzero_frequency_count"], nonzero)
            and np.array_equal(counts, np.minimum(nonzero, kmax)),
            f"{path}: critical count differs",
        )
        positions, frequencies = arrays["critical_token_positions"], arrays["critical_token_frequencies"]
        valid = np.arange(kmax) < counts[..., None]
        sentinel = np.iinfo(np.uint32).max
        _require(
            (positions[~valid] == sentinel).all() and (frequencies[~valid] == 0).all(),
            f"{path}: invalid ranking sentinel",
        )
        _require(
            ((positions >= starts[None, None, :, None]) & (positions < ends[None, None, :, None]))[valid].all(),
            f"{path}: critical positions outside candidate interval",
        )
        _require(
            ((frequencies[valid] > 0) & (frequencies[valid] <= axis[-1])).all()
            and (frequencies[..., 1:] <= frequencies[..., :-1]).all(),
            f"{path}: invalid frequency ranking",
        )
        ordered = np.sort(positions, axis=-1)
        _require(
            not ((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] != sentinel)).any(),
            f"{path}: duplicate critical positions",
        )




def load_completed_b2_run_set(
    path: Path | str,
    *,
    source_root: Path | str,
    expected_run_set_sha256: str | None = None,
    expected_batch_count: int = 10,
    expected_sample_count: int = 100,
    model_dimensions: tuple[int, int, int] = (32, 32, 8),
) -> B2RunSetIndex:
    """Read a completed set; an old external checksum is only checked when supplied."""

    try:
        return _load(
            path,
            Path(source_root).resolve(),
            expected_run_set_sha256,
            expected_batch_count,
            expected_sample_count,
            model_dimensions,
        )
    except B2SourceError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, BadZipFile) as exc:
        raise B2SourceError(f"B2 source validation failed: {type(exc).__name__}: {exc}") from exc


def _load(path, root, pinned_hash, batch_count, sample_count, dimensions) -> B2RunSetIndex:
    manifest_path = Path(path).resolve()
    manifest_hash = sha256_file(manifest_path)
    if pinned_hash is not None:
        _require(manifest_hash == pinned_hash, "B2 Run Set SHA-256 differs from external authority")
    payload = _json(manifest_path)
    _require(
        payload["schema_version"] == B2_RUN_SET_SCHEMA and payload["status"] == "completed",
        "B2 Run Set is not a completed supported schema",
    )
    _require(payload["path_base"] == "project_root", "unsupported path base")
    batches = _unique_entries(payload["batches"], "batch_id", "B2 batches")
    _require(len(batches) == payload["batch_count"] == batch_count, "B2 batch count differs")
    selected = [sid for batch in batches.values() for sid in batch["sample_ids"]]
    _require(
        all(
            isinstance(sid, str) and bool(sid) and Path(sid).name == sid and sid not in (".", "..") for sid in selected
        ),
        "invalid sample path component",
    )
    _require(
        len(set(selected)) == len(selected) == payload["sample_count"] == sample_count,
        "duplicate sample IDs or sample count differs",
    )
    report_path = _path(root, payload["validation_report"]["path"])
    b1path = _reference(root, payload["b1_replay_authority"])
    analysis_path = _path(root, payload["b1_analysis_authority"]["path"])
    b1 = _json(b1path)
    _require(
        b1["schema_version"] == "headkv-v1-b1-run-set-v1" and b1["status"] == "completed",
        "B1 replay authority is not completed",
    )
    fixed = payload["fixed_dataset"]
    _require(
        all(fixed[k] == b1["fixed_dataset"][k] for k in ("prompts_path", "manifest_path")),
        "B1 and B2 fixed datasets differ",
    )
    _json(_batch_file(root, fixed["manifest_path"]))
    prompts_path = _batch_file(root, fixed["prompts_path"])
    records = [json.loads(line) for line in prompts_path.read_text().splitlines() if line.strip()]
    prompts = _unique_entries(records, "sample_id", "fixed prompts")
    _require(set(prompts) == set(selected), "Run Set does not exactly cover frozen prompts")
    b1batches = _unique_entries(b1["batches"], "batch_id", "B1 batches")
    _require(set(b1batches) == set(batches), "B1/B2 batch IDs differ")
    contract = payload["execution_contract"]
    schema = B2TraceSchema(*dimensions).to_dict()
    for name in ("dimensions", "axes", "conditions"):
        _require(contract[name] == schema[name], f"frozen B2 {name} differs")
    _require(
        contract["teacher_forced_replay"] is True
        and contract["attention_backend"] == "sdpa",
        "unsupported execution contract",
    )
    for name, observed in (
        ("model", b1["execution_contract"]["model"]["path"]),
        ("tokenizer", b1["execution_contract"]["model"]["tokenizer_path"]),
        ("dtype", b1["execution_contract"]["model"]["dtype"]),
    ):
        _require(contract[name] == observed, f"B1/B2 {name} differs")
    sample_index = {}
    batch_index = []
    totals = Counter()
    for bid, batch in batches.items():
        run = _path(root, batch["run_dir"])
        _require(batch["status"] == "completed" and batch["b1_batch_id"] == bid, f"{bid}: invalid batch state")
        upstream = b1batches[bid]
        _require(
            upstream["status"] == "completed" and batch["sample_ids"] == upstream["sample_ids"],
            f"{bid}: B1 sample scope differs",
        )
        for name in ("run_metadata.json", "file_manifest.json", "schema.json", "config_snapshot.yaml", "resource_metrics.json"):
            _batch_file(run, name)
        entries = _unique_entries(_json(run / "file_manifest.json")["files"], "relative_path", "B2 files")
        for relative in entries:
            _path(run, relative)
        result = validate_b2_run(run, verify_hashes=False)
        _require(result.completed_sample_count == len(batch["sample_ids"]), f"{bid}: sample count differs")
        rm, rs = _json(run / "run_metadata.json"), _json(run / "resource_metrics.json")
        stored_schema = _json(run / "schema.json")
        _require(
            all(stored_schema.get(key) == value for key, value in schema.items() if key != "collector_config_sha256"),
            f"{bid}: schema differs from frozen schema",
        )
        _require(
            rm["run_id"] == batch["run_id"] and rm["target_sample_ids"] == batch["sample_ids"],
            f"{bid}: run identity differs",
        )
        for name in (
            "model",
            "tokenizer",
            "dtype",
            "attention_backend",
            "teacher_forced_replay",
        ):
            _require(rm[name] == contract[name], f"{bid}: {name} differs")
        _require(
            {p.name for p in (run / "samples").iterdir()} == set(batch["sample_ids"]),
            f"{bid}: sample directories differ",
        )
        resources = _unique_entries(rs["per_sample"], "sample_id", "resource samples")
        _require(set(resources) == set(batch["sample_ids"]), f"{bid}: resource coverage differs")
        b1dir = _path(root, upstream["run_dir"])
        _batch_file(b1dir, "run_metadata.json")
        _batch_file(b1dir, "file_manifest.json")
        b1files = _unique_entries(_json(b1dir / "file_manifest.json")["files"], "relative_path", "B1 files")
        for sid in batch["sample_ids"]:
            item = _validate_sample(
                run, sid, bid, prompts[sid], resources[sid], schema, b1dir, b1files
            )
            sample_index[sid] = item
            totals.update(
                generated_tokens=item.generated_length,
                decode_steps=item.decode_length,
                decode_chunks=len(item.decode_chunks),
            )
        batch_index.append(
            B2BatchIndex(
                bid,
                run,
                tuple(batch["sample_ids"]),
                batch["file_manifest_sha256"],
                result.artifact_file_count,
                result.artifact_file_bytes,
            )
        )
        totals.update(artifact_files=result.artifact_file_count, artifact_bytes=result.artifact_file_bytes)
    for name in ("generated_tokens", "decode_steps", "decode_chunks"):
        _require(payload["summary"][name] == totals[name], f"Run Set summary {name} differs")
    raw_counts = dict(Counter(f"{p['dataset']}:{p['target_prompt_length']}" for p in prompts.values()))
    _require(raw_counts == payload["summary"]["condition_counts"], "condition counts differ")
    pairs = defaultdict(list)
    for item in sample_index.values():
        if item.dataset == "pg19":
            _require(bool(item.pair_id), "PG19 missing pair_id")
            pairs[item.pair_id].append(item.target_prompt_length)
    _require(all(sorted(lengths) == [8192, 32768] for lengths in pairs.values()), "invalid PG19 pair coverage")
    return B2RunSetIndex(
        manifest_path,
        root,
        manifest_hash,
        payload,
        sample_index,
        tuple(batch_index),
        report_path,
        b1path,
        analysis_path,
        None,
    )


def _validate_sample(run, sid, bid, prompt, resource, schema, b1dir, b1files):
    _require(Path(sid).name == sid and sid not in (".", ".."), "invalid sample path component")
    sample_dir = run / "samples" / sid
    sm = _json(sample_dir / "sample_metadata.json")
    ids = prompt["input_ids"]
    _require(
        isinstance(ids, list) and all(type(value) is int and 0 <= value < 2**32 for value in ids),
        f"{sid}: invalid frozen input IDs",
    )
    input_hash = prompt["input_sha256"]
    _require(len(ids) == prompt["actual_prompt_length"] == sm["prompt_length"], f"{sid}: prompt length differs")
    for name in ("dataset", "task", "pair_id"):
        _require(sm[name] == prompt.get(name), f"{sid}: {name} differs")
    _require(sm["sample_id"] == sid and sm["b1_batch_id"] == bid, f"{sid}: upstream coordinate differs")
    upstream_ids = f"samples/{sid}/generated_ids.json"
    upstream_meta = f"samples/{sid}/sample_metadata.json"
    for name in (upstream_ids, upstream_meta):
        _require(name in b1files, f"{sid}: B1 file is missing from manifest")
        _batch_file(b1dir, name)
    original = _json(b1dir / upstream_ids)
    replay = _json(sample_dir / "replay_generated_ids.json")
    generated = _ids(replay)
    _require(generated == original.get("ids"), f"{sid}: replay differs from B1 generated IDs")
    b1sm = _json(b1dir / upstream_meta)
    _require(
        b1sm["status"] == "completed"
        and b1sm["actual_generated_length"] == len(generated),
        f"{sid}: B1 sample differs",
    )
    _require(
        sm["max_new_tokens"] == prompt["max_new_tokens"] and len(generated) <= prompt["max_new_tokens"],
        f"{sid}: generation budget differs",
    )
    _validate_prefill(sample_dir / "prefill_consistency.npz", len(ids), schema)
    entries = _unique_entries(_json(run / "file_manifest.json")["files"], "relative_path", "B2 files")
    chunks = tuple(sorted(sample_dir.glob("decode_recall_*.npz")))
    cursor = 0
    for i, path in enumerate(chunks):
        _require(path.name == f"decode_recall_{i:03d}.npz", f"{sid}: chunk filenames differ")
        with np.load(path, allow_pickle=False) as arrays:
            count = len(arrays["decode_steps"])
        entry = entries[path.relative_to(run).as_posix()]
        _require(
            1 <= count <= 128 and entry["step_start"] == cursor and entry["step_end"] == cursor + count - 1,
            f"{sid}: chunk interval differs",
        )
        cursor += count
    timing = resource["forward_timing"]
    _require(
        resource["status"] == "completed"
        and timing["incomplete_forward_count"] == 0
        and timing["prefill"]["count"] == 1
        and timing["decode"]["count"] == cursor,
        f"{sid}: Forward resource counts differ",
    )
    return B2SampleIndex(
        sid,
        prompt["dataset"],
        prompt["task"],
        prompt.get("pair_id"),
        prompt["target_prompt_length"],
        len(ids),
        input_hash,
        replay["sha256"],
        len(generated),
        cursor,
        bid,
        sample_dir,
        chunks,
    )


def build_b2_analysis_plan(index: B2RunSetIndex, *, analysis_id: str, analysis_code: Mapping[str, Any]) -> dict:
    """Describe validated phase-1 inputs; never claim completed analysis."""

    _require(
        bool(analysis_id) and Path(analysis_id).name == analysis_id and analysis_id not in (".", ".."),
        "analysis_id must be a non-empty path component",
    )
    parameters = dict(DEFAULT_ANALYSIS_PARAMETERS)
    return {
        "schema_version": B2_PLAN_SCHEMA,
        "analysis_id": analysis_id,
        "status": "prepared",
        "completed_stages": ["source_validation"],
        "pending_stages": ["distributions_and_recall", "cross_phase_relations", "b1_b2_relations", "finalize"],
        "source_run_set": {"path": str(index.manifest_path), "sha256": index.run_set_sha256},
        "source_root": str(index.source_root),
        "validation_report": dict(index.payload["validation_report"]),
        "b1_replay_authority": dict(index.payload["b1_replay_authority"]),
        "b1_analysis_authority": dict(index.payload["b1_analysis_authority"]),
        "analysis_code": dict(analysis_code),
        "parameters": parameters,
        "parameters_sha256": canonical_json_sha256(parameters),
        "sample_count": len(index.samples),
        "batch_count": len(index.batches),
        "condition_counts": index.condition_counts,
        "sample_ids": list(index.samples),
        "pg19_pair_ids": sorted({s.pair_id for s in index.samples.values() if s.dataset == "pg19"}),
        "source_collection_summary": dict(index.payload["summary"]),
    }
