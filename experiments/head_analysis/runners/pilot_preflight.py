"""Create a checked, immutable launch plan for the frozen B1 pilot batch.

This module performs only host and provenance checks.  It does not load a
model, allocate CUDA memory, or start a baseline/B1 collection.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from experiments.head_analysis.runners.b1_runner import git_state, load_frozen_selection


PILOT_SCHEMA_VERSION = "headkv-v1-b1-pilot-plan-v1"
PILOT_SAMPLE_IDS = (
    "ruler_8k_001", "ruler_32k_001", "govreport_8k_001", "govreport_8k_002",
    "govreport_16k_001", "govreport_16k_002", "pg19_001_8k", "pg19_003_8k",
    "pg19_001_32k", "pg19_003_32k",
)
EXPECTED_DATA_SHA256 = "30ea0f8a85e203c8cf63a87e83805e851345c42a2a2fd9ab7f4dbdc1c63492ee"
PILOT_COLLECTION = {
    "key_block_size": 1024, "prefill_window": 64, "token_topk": 64,
    "save_token_topk": True, "page_size": 16, "page_topk": 32, "decode_chunk_steps": 128,
}
PILOT_BUDGET = {
    "max_batch_generation_seconds": 5 * 60 * 60,
    "max_batch_artifact_file_bytes": 5 * 1024**3,
    "max_sample_peak_reserved_bytes": 32 * 1024**3,
    "max_sample_logical_array_bytes": 1024**3,
    "max_sample_artifact_file_bytes": 512 * 1024**2,
    "max_sample_trace_write_seconds": 90,
}


class PilotPreflightError(RuntimeError):
    """Raised when a frozen B1 pilot must not be launched."""


@dataclass(frozen=True)
class GPUInspection:
    device: str
    index: int
    name: str
    free_memory_bytes: int
    compute_pids: tuple[int, ...]


def _run_text(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PilotPreflightError(f"unable to inspect GPU with {' '.join(command)}: {exc}") from exc
    return result.stdout.strip()


def inspect_cuda_gpu(device: str, *, run_text: Callable[[Sequence[str]], str] = _run_text) -> GPUInspection:
    """Return free memory and compute PIDs for one explicit ``cuda:N`` device."""

    prefix, separator, index_text = device.partition(":")
    if prefix != "cuda" or not separator or not index_text.isdigit():
        raise PilotPreflightError("pilot preflight requires an explicit CUDA device in the form cuda:N")
    index = int(index_text)
    rows = [
        row.strip()
        for row in run_text((
            "nvidia-smi", f"--id={index}", "--query-gpu=index,name,memory.free",
            "--format=csv,noheader,nounits",
        )).splitlines()
        if row.strip()
    ]
    if len(rows) != 1:
        raise PilotPreflightError(f"{device}: expected one nvidia-smi GPU row, got {rows!r}")
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 3 or not parts[0].isdigit() or int(parts[0]) != index or not parts[2].isdigit():
        raise PilotPreflightError(f"{device}: malformed nvidia-smi GPU row: {rows[0]!r}")
    pids: list[int] = []
    for row in run_text((
        "nvidia-smi", f"--id={index}", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    )).splitlines():
        value = row.strip()
        if not value or value.lower() == "no running processes found":
            continue
        if not value.isdigit():
            raise PilotPreflightError(f"{device}: malformed nvidia-smi process row: {value!r}")
        pids.append(int(value))
    return GPUInspection(device, index, parts[1], int(parts[2]) * 1024**2, tuple(pids))


def current_git_status() -> tuple[str, ...]:
    try:
        result = subprocess.run(("git", "status", "--porcelain"), check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PilotPreflightError(f"unable to inspect Git status: {exc}") from exc
    return tuple(line for line in result.stdout.splitlines() if line)


def _validate_git_state(expected_commit: str, observed_git: Mapping[str, Any], status_lines: Sequence[str]) -> None:
    if observed_git.get("commit") != expected_commit:
        raise PilotPreflightError(
            f"HEAD commit {observed_git.get('commit')!r} differs from expected {expected_commit!r}"
        )
    unexpected = [line for line in status_lines if line != "?? data/"]
    if unexpected:
        raise PilotPreflightError(f"tracked or unexpected working-tree changes: {unexpected}")


def _ensure_new_artifact_paths(paths: Sequence[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise PilotPreflightError(f"pilot artifacts must be new and non-overwriting: {existing}")


def _command_arguments(
    *, python_executable: str, input_path: Path, manifest_path: Path, tokenizer_path: Path,
    model_path: Path, device: str, baseline_output: Path, run_dir: Path,
) -> dict[str, list[str]]:
    common = [
        "--input", str(input_path), "--manifest", str(manifest_path), "--tokenizer", str(tokenizer_path),
        "--model", str(model_path), "--device", device, "--dtype", "bfloat16", "--seed", "0",
    ]
    samples = [item for sample_id in PILOT_SAMPLE_IDS for item in ("--sample-id", sample_id)]
    baseline = [python_executable, "-m", "experiments.head_analysis.runners.baseline_decode", *common]
    baseline.extend(("--output", str(baseline_output), *samples))
    collection = [python_executable, "-m", "experiments.head_analysis.runners.b1_runner", *common]
    collection.extend((
        "--run-dir", str(run_dir), *samples,
        "--key-block-size", str(PILOT_COLLECTION["key_block_size"]),
        "--prefill-window", str(PILOT_COLLECTION["prefill_window"]),
        "--token-topk", str(PILOT_COLLECTION["token_topk"]),
        "--page-size", str(PILOT_COLLECTION["page_size"]),
        "--page-topk", str(PILOT_COLLECTION["page_topk"]),
        "--decode-chunk-steps", str(PILOT_COLLECTION["decode_chunk_steps"]),
        "--verify-baseline", str(baseline_output),
        "--max-batch-generation-seconds", str(PILOT_BUDGET["max_batch_generation_seconds"]),
        "--max-batch-artifact-file-bytes", str(PILOT_BUDGET["max_batch_artifact_file_bytes"]),
        "--max-sample-peak-reserved-bytes", str(PILOT_BUDGET["max_sample_peak_reserved_bytes"]),
        "--max-sample-logical-array-bytes", str(PILOT_BUDGET["max_sample_logical_array_bytes"]),
        "--max-sample-artifact-file-bytes", str(PILOT_BUDGET["max_sample_artifact_file_bytes"]),
        "--max-sample-trace-write-seconds", str(PILOT_BUDGET["max_sample_trace_write_seconds"]),
        "--stop-on-failure",
    ))
    return {"baseline": baseline, "b1_collection": collection}


def build_pilot_plan(
    *, input_path: Path, manifest_path: Path, tokenizer_path: Path, model_path: Path, device: str,
    run_dir: Path, baseline_output: Path, expected_commit: str, python_executable: str,
    minimum_free_memory_bytes: int, minimum_free_disk_bytes: int,
    expected_data_sha256: str = EXPECTED_DATA_SHA256, gpu: GPUInspection | None = None,
    disk_free_bytes: int | None = None, observed_git: Mapping[str, Any] | None = None,
    status_lines: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate the frozen 10-sample pilot and return an immutable launch plan."""

    if not expected_commit:
        raise PilotPreflightError("expected_commit must be explicit")
    if not model_path.is_dir() or not tokenizer_path.is_dir():
        raise PilotPreflightError("model and tokenizer must both name existing local snapshot directories")
    _ensure_new_artifact_paths((run_dir, baseline_output))
    selected, manifest = load_frozen_selection(input_path, manifest_path, PILOT_SAMPLE_IDS)
    if manifest.get("data_sha256") != expected_data_sha256:
        raise PilotPreflightError("fixed dataset data_sha256 differs from the frozen B1 pilot value")
    observed_git = git_state() if observed_git is None else observed_git
    status_lines = current_git_status() if status_lines is None else status_lines
    _validate_git_state(expected_commit, observed_git, status_lines)
    gpu = inspect_cuda_gpu(device) if gpu is None else gpu
    if gpu.free_memory_bytes < minimum_free_memory_bytes:
        raise PilotPreflightError(f"{device}: free GPU memory is below the required threshold")
    if gpu.compute_pids:
        raise PilotPreflightError(f"{device}: GPU has active compute PIDs {list(gpu.compute_pids)}")
    disk_free_bytes = shutil.disk_usage(run_dir.parent).free if disk_free_bytes is None else disk_free_bytes
    if disk_free_bytes < minimum_free_disk_bytes:
        raise PilotPreflightError("run filesystem free bytes are below the required threshold")
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": "ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"commit": observed_git.get("commit"), "allowed_status_lines": list(status_lines)},
        "fixed_dataset": {
            "path": str(input_path), "manifest_path": str(manifest_path), "data_sha256": manifest["data_sha256"],
            "selected_sample_ids": list(PILOT_SAMPLE_IDS),
            "selected_samples": [
                {"sample_id": r["sample_id"], "dataset": r.get("dataset"),
                 "prompt_length": len(r["input_ids"]), "max_new_tokens": r["max_new_tokens"]}
                for r in selected
            ],
        },
        "model": {"path": str(model_path), "tokenizer_path": str(tokenizer_path), "dtype": "bfloat16"},
        "generation": {"do_sample": False, "num_beams": 1, "use_cache": True, "seed": 0},
        "attention_backend": "sdpa",
        "collection": dict(PILOT_COLLECTION),
        "resource_budget": dict(PILOT_BUDGET),
        "host_preflight": {
            "gpu": {**asdict(gpu), "compute_pids": list(gpu.compute_pids)},
            "minimum_free_memory_bytes": minimum_free_memory_bytes,
            "disk_free_bytes": disk_free_bytes, "minimum_free_disk_bytes": minimum_free_disk_bytes,
        },
        "launch_arguments": _command_arguments(
            python_executable=python_executable, input_path=input_path, manifest_path=manifest_path,
            tokenizer_path=tokenizer_path, model_path=model_path, device=device,
            baseline_output=baseline_output, run_dir=run_dir,
        ),
    }


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create a preflight plan without overwriting prior evidence."""

    if path.exists():
        raise PilotPreflightError(f"preflight output already exists: {path}")
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
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise PilotPreflightError(f"preflight output already exists: {path}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
