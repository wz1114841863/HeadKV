"""File operations extracted from the existing writers and analysis scripts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_jsonl(path, *, require_objects=False):
    """Read nonempty JSONL lines with a useful path and line on invalid input."""
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL parse error {path}:{line_no}: {exc}") from exc
            if require_objects and not isinstance(record, dict):
                raise ValueError(f"JSONL record {path}:{line_no} is not an object")
            yield record


@contextmanager
def atomic_file(path, *, binary=False, overwrite=False):
    """Publish a complete file, preserving an existing destination by default."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        options = {} if binary else {"encoding": "utf-8"}
        with os.fdopen(descriptor, "wb" if binary else "w", **options) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, payload, *, overwrite=False, allow_nan=False):
    with atomic_file(path, overwrite=overwrite) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=allow_nan)
        handle.write("\n")


def write_npz(path, arrays, *, overwrite=False):
    with atomic_file(path, binary=True, overwrite=overwrite) as handle:
        np.savez_compressed(handle, **arrays)


def sha256_file(path):
    """Encode identity fields required by existing Trace/result formats."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload):
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def file_entry(path, root, *, arrays=None, sample_id=None, step_start=None, step_end=None):
    """Describe a saved file in the existing B1/B2 Trace format."""
    path = Path(path)
    entry = {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "file_bytes": path.stat().st_size,
        "logical_array_bytes": 0 if arrays is None else int(sum(array.nbytes for array in arrays.values())),
    }
    if arrays is not None:
        entry["arrays"] = [
            {"name": name, "shape": list(array.shape), "dtype": str(array.dtype)} for name, array in arrays.items()
        ]
    if sample_id is not None:
        entry["sample_id"] = sample_id
    if step_start is not None and step_end is not None:
        entry.update(step_start=step_start, step_end=step_end)
    return entry
