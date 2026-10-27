"""Offline exact Page RCO calculation from completed B1 trace chunks.

The implementation consumes only the stored ordered Page Top-K IDs, candidate
counts, and scope-valid flags.  It never reconstructs attention weights or
changes the original B1 artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PAGE_SCOPES = ("all_visible", "prompt_only")
SENTINEL = np.iinfo(np.uint32).max


class PageRCOError(RuntimeError):
    """Raised when a completed B1 artifact violates the offline RCO contract."""


def compute_page_rco(
    *,
    sample_id: str,
    scope: str,
    page_ids: np.ndarray,
    candidate_counts: np.ndarray,
    scope_valid: np.ndarray,
    k_values: Sequence[int] = (8, 16, 32),
    deltas: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256),
) -> list[dict[str, Any]]:
    """Return per-(k, delta, layer, KV-head) exact RCO summaries for one scope.

    Each record averages only valid step pairs.  A record with zero valid pairs
    is retained with null metrics, rather than silently reporting zero RCO.
    """

    if scope not in PAGE_SCOPES:
        raise PageRCOError(f"unsupported Page scope {scope!r}")
    if page_ids.ndim != 4:
        raise PageRCOError("page_ids must have shape [C,L,Hkv,Kmax]")
    if candidate_counts.shape != (page_ids.shape[0],):
        raise PageRCOError("candidate_counts must have shape [C]")
    if scope_valid.shape != page_ids.shape[:3]:
        raise PageRCOError("scope_valid must have shape [C,L,Hkv]")
    if page_ids.dtype != np.uint32 or candidate_counts.dtype != np.uint32 or scope_valid.dtype != bool:
        raise PageRCOError("Page RCO arrays must retain B1 uint32/bool dtypes")
    k_values = tuple(int(value) for value in k_values)
    deltas = tuple(int(value) for value in deltas)
    if not k_values or any(value <= 0 or value > page_ids.shape[-1] for value in k_values):
        raise PageRCOError("k_values must be non-empty and within stored Page Top-K width")
    if not deltas or any(value <= 0 for value in deltas):
        raise PageRCOError("deltas must be non-empty positive integers")

    steps, layers, kv_heads, _ = page_ids.shape
    records: list[dict[str, Any]] = []
    for k in k_values:
        for delta in deltas:
            for layer_id in range(layers):
                for kv_head in range(kv_heads):
                    overlaps: list[float] = []
                    random_baselines: list[float] = []
                    raw_values: list[float] = []
                    clipped_values: list[float] = []
                    for step in range(steps - delta):
                        later = step + delta
                        n_earlier = int(candidate_counts[step])
                        n_later = int(candidate_counts[later])
                        if n_earlier < k or n_later <= k:
                            continue
                        if not (scope_valid[step, layer_id, kv_head] and scope_valid[later, layer_id, kv_head]):
                            continue
                        earlier_ids = page_ids[step, layer_id, kv_head, :k]
                        later_ids = page_ids[later, layer_id, kv_head, :k]
                        if (
                            np.any(earlier_ids == SENTINEL)
                            or np.any(later_ids == SENTINEL)
                            or len(set(earlier_ids.tolist())) != k
                            or len(set(later_ids.tolist())) != k
                        ):
                            continue
                        overlap = len(set(earlier_ids.tolist()) & set(later_ids.tolist())) / k
                        random_baseline = k / n_later
                        raw = (overlap - random_baseline) / (1.0 - random_baseline)
                        overlaps.append(overlap)
                        random_baselines.append(random_baseline)
                        raw_values.append(raw)
                        clipped_values.append(max(0.0, raw))
                    valid_pair_count = len(overlaps)
                    records.append(
                        {
                            "sample_id": sample_id,
                            "scope": scope,
                            "k": k,
                            "delta": delta,
                            "layer_id": layer_id,
                            "kv_head": kv_head,
                            "valid_pair_count": valid_pair_count,
                            "overlap_raw_mean": None if not valid_pair_count else float(np.mean(overlaps)),
                            "random_baseline_mean": None if not valid_pair_count else float(np.mean(random_baselines)),
                            "rco_raw_mean": None if not valid_pair_count else float(np.mean(raw_values)),
                            "rco_clipped_mean": None if not valid_pair_count else float(np.mean(clipped_values)),
                        }
                    )
    return records


def _load_sample_arrays(sample_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunks = sorted(sample_dir.glob("decode_stats_*.npz"))
    if not chunks:
        raise PageRCOError(f"{sample_dir}: no Decode chunks")
    required = (
        "decode_steps",
        "page_topk_ids_all_visible",
        "page_topk_ids_prompt_only",
        "page_candidate_count_all_visible",
        "page_candidate_count_prompt_only",
        "page_scope_valid",
    )
    loaded: list[dict[str, np.ndarray]] = []
    for path in chunks:
        with np.load(path) as archive:
            missing = [name for name in required if name not in archive]
            if missing:
                raise PageRCOError(f"{path}: missing required arrays {missing}")
            loaded.append({name: archive[name].copy() for name in required})
    steps = np.concatenate([chunk["decode_steps"] for chunk in loaded])
    if steps.dtype != np.uint32 or steps.tolist() != list(range(len(steps))):
        raise PageRCOError(f"{sample_dir}: Decode steps are not continuous from zero")
    all_ids = np.concatenate([chunk["page_topk_ids_all_visible"] for chunk in loaded])
    prompt_ids = np.concatenate([chunk["page_topk_ids_prompt_only"] for chunk in loaded])
    page_ids = np.stack((all_ids, prompt_ids), axis=3)
    all_counts = np.concatenate([chunk["page_candidate_count_all_visible"] for chunk in loaded])
    prompt_counts = np.concatenate([chunk["page_candidate_count_prompt_only"] for chunk in loaded])
    if not np.array_equal(all_counts, prompt_counts):
        # These counts differ by scope in normal B1 traces.  The caller selects
        # the matching one below; retain both in a two-column array here.
        candidate_counts = np.stack((all_counts, prompt_counts), axis=1)
    else:
        candidate_counts = np.stack((all_counts, prompt_counts), axis=1)
    scope_valid = np.concatenate([chunk["page_scope_valid"] for chunk in loaded])
    return page_ids, candidate_counts, scope_valid


def analyze_completed_run(
    run_dir: Path | str,
    *,
    k_values: Sequence[int] = (8, 16, 32),
    deltas: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256),
) -> dict[str, Any]:
    """Analyze every completed sample in a B1 run without writing into it."""

    root = Path(run_dir)
    run_metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    if run_metadata.get("status") != "completed":
        raise PageRCOError(f"{root}: run status is not completed")
    records: list[dict[str, Any]] = []
    for sample_dir in sorted((root / "samples").iterdir()):
        metadata = json.loads((sample_dir / "sample_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("status") != "completed":
            raise PageRCOError(f"{sample_dir}: sample status is not completed")
        page_ids, candidate_counts_by_scope, scope_valid = _load_sample_arrays(sample_dir)
        for scope_index, scope in enumerate(PAGE_SCOPES):
            scope_records = compute_page_rco(
                sample_id=str(metadata["sample_id"]),
                scope=scope,
                page_ids=page_ids[:, :, :, scope_index, :],
                candidate_counts=candidate_counts_by_scope[:, scope_index],
                scope_valid=scope_valid[:, :, :, scope_index],
                k_values=k_values,
                deltas=deltas,
            )
            records.extend(scope_records)
    return {
        "schema_version": "headkv-v1-b1-page-rco-v1",
        "source_run": str(root),
        "source_run_metadata_sha256": hashlib.sha256((root / "run_metadata.json").read_bytes()).hexdigest(),
        "k_values": list(k_values),
        "deltas": list(deltas),
        "records": records,
    }


def write_json_new(path: Path | str, payload: dict[str, Any]) -> None:
    """Atomically create a new analysis artifact without overwriting an old one."""

    destination = Path(path)
    if destination.exists():
        raise PageRCOError(f"analysis output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise PageRCOError(f"analysis output already exists: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)
