"""Read existing V2 representations and freeze the approved V3 source split."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.head_analysis.common.io import iter_jsonl, read_json, write_json
from experiments.head_analysis.runners.input_selection import frozen_length


REPRESENTATIONS = ("k_post", "v")
VECTOR_REPRESENTATIONS = ("k_pre", "k_post", "v")
CONDITIONS = (("ruler", 8192), ("ruler", 32768), ("govreport", 8192), ("govreport", 16384))


def load_codec_cohorts(bundle, codec_root):
    """Reuse the completed six-input codec cohorts within the original V3 split."""
    rows = {r["sample_id"]: r for r in read_json(Path(bundle) / "split.json")["samples"]}
    cohorts = []
    for phase, split in (("pilot", "calib"), ("validation", "validation")):
        meta = read_json(Path(codec_root) / phase / "run_metadata.json")
        if meta["status"] != "completed" or meta["completed_sample_ids"] != meta["target_sample_ids"]:
            raise ValueError("source codec run is incomplete")
        selected = [rows[sid] for sid in meta["target_sample_ids"]]
        if len(selected) != 6 or any(r["split"] != split for r in selected):
            raise ValueError("expected the frozen six Calib/six Validation inputs")
        cohorts.append(selected)
    return cohorts


def discover_samples(run_dirs, fixed_inputs):
    """Join completed collection directories to the unchanged frozen input records."""
    fixed_rows = list(iter_jsonl(Path(fixed_inputs), require_objects=True))
    fixed = {row["sample_id"]: row for row in fixed_rows}
    if len(fixed) != len(fixed_rows):
        raise ValueError("duplicate frozen sample IDs")
    found, model = {}, None
    for run_dir in map(Path, run_dirs):
        run = read_json(run_dir / "run_metadata.json")
        if run["status"] != "completed" or run["completed_sample_ids"] != run["target_sample_ids"]:
            raise ValueError(f"incomplete V2 collection: {run_dir}")
        if model is None:
            model = run["model"]
        elif model != run["model"]:
            raise ValueError("V2 collection model paths differ")
        for sid in run["completed_sample_ids"]:
            if sid in found or sid not in fixed:
                raise ValueError(f"duplicate or unexpected V2 sample: {sid}")
            directory = run_dir / "samples" / sid
            sample = read_json(directory / "sample_metadata.json")
            metadata = read_json(directory / "metadata.json")
            row = fixed[sid]
            length = frozen_length(row)
            if "input_ids" in row and len(row["input_ids"]) != length:
                raise ValueError(f"frozen input length differs: {sid}")
            if sample["status"] != "completed" or metadata["status"] != "completed":
                raise ValueError(f"incomplete V2 sample: {sid}")
            if (sample["sample_id"] != sid or metadata["sample_id"] != sid
                    or sample["dataset"] != row["dataset"] or sample.get("pair_id") != row.get("pair_id")
                    or sample["prompt_length"] != length or metadata["prompt_length"] != length
                    or sample["num_layers"] != metadata["num_layers"]):
                raise ValueError(f"V2 and frozen input metadata differ: {sid}")
            layers, heads = int(sample["num_layers"]), int(sample["num_kv_heads"])
            shape = metadata["files"]["prefill/layer_000.npz"]["fields"]["k_post"]["shape"]
            if len(shape) != 3 or shape[:2] != [heads, 512] or shape[2] <= 0 or layers <= 0 or heads <= 0 or length < 576:
                raise ValueError(f"invalid sampled vector geometry: {sid}")
            dataset, target = row["dataset"], int(row["target_prompt_length"])
            found[sid] = {"sample_id": sid, "dataset": dataset, "pair_id": row.get("pair_id"),
                "source_id": row.get("source_id"), "source_group": f"pg19:{row['pair_id']}" if dataset == "pg19" else f"{dataset}:{sid}",
                "prompt_length": length, "target_prompt_length": target, "condition": f"{dataset}_{target}",
                "run_dir": str(run_dir.resolve()), "sample_dir": str(directory.resolve()), "model": model,
                "num_layers": layers, "num_kv_heads": heads, "head_dim": int(shape[2])}
    if set(found) != set(fixed):
        raise ValueError("V2 collections do not cover the frozen input IDs exactly")
    records = [found[row["sample_id"]] for row in fixed_rows]
    gov = [row["source_id"] for row in records if row["dataset"] == "govreport"]
    if any(source is None for source in gov) or len(gov) != len(set(gov)):
        raise ValueError("GovReport sources are missing or repeated")
    pg_groups = set(row["pair_id"] for row in records if row["dataset"] == "pg19")
    if None in pg_groups:
        raise ValueError("PG19 Pair ID is missing")
    pg_sources = []
    for pair in pg_groups:
        rows = [row for row in records if row["dataset"] == "pg19" and row["pair_id"] == pair]
        if (not pair or len(rows) != 2 or {row["target_prompt_length"] for row in rows} != {8192, 32768}
                or len({row["source_id"] for row in rows}) != 1 or rows[0]["source_id"] is None):
            raise ValueError("PG19 Pair source or length coordinates differ")
        pg_sources.append(rows[0]["source_id"])
    if len(pg_sources) != len(set(pg_sources)):
        raise ValueError("PG19 book sources are repeated across Pairs")
    return records


def freeze_split(records, output):
    """Create the predeclared 60/20/20 split before computing any V3 errors."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"split already exists: {output}")
    ids = [row["sample_id"] for row in records]
    if len(records) != 100 or len(set(ids)) != 100:
        raise ValueError("V3 split requires exactly 100 unique frozen sample IDs")
    rng = np.random.Generator(np.random.PCG64(20261113))
    assigned = {}
    for dataset, length in CONDITIONS:
        rows = [r for r in records if r["dataset"] == dataset and r["target_prompt_length"] == length]
        pilot = [r["sample_id"] for r in rows if r["sample_id"].endswith("_001")]
        if len(rows) != 20 or len(pilot) != 1:
            raise ValueError("each RULER/GovReport condition requires 20 rows and its 001 Pilot")
        ordered = sorted(r["sample_id"] for r in rows if r["sample_id"] not in pilot)
        shuffled = rng.permutation(ordered).tolist()
        assigned.update({sid: "calib" for sid in pilot + shuffled[8:]})
        assigned.update({sid: "validation" for sid in shuffled[:4]})
        assigned.update({sid: "test" for sid in shuffled[4:8]})
    pg = [row for row in records if row["dataset"] == "pg19"]
    if any(not row.get("pair_id") for row in pg):
        raise ValueError("PG19 Pair ID is missing")
    pair_ids = sorted(set(row["pair_id"] for row in pg))
    if len(pg) != 20 or len(pair_ids) != 10 or "pg19_001" not in pair_ids:
        raise ValueError("V3 requires the original ten PG19 Pairs including pg19_001")
    for pair in pair_ids:
        rows = [r for r in pg if r["pair_id"] == pair]
        if len(rows) != 2 or {r["target_prompt_length"] for r in rows} != {8192, 32768}:
            raise ValueError("PG19 Pair must contain both length conditions")
    shuffled = rng.permutation([pair for pair in pair_ids if pair != "pg19_001"]).tolist()
    for row in pg:
        pair = row["pair_id"]
        assigned[row["sample_id"]] = "validation" if pair in shuffled[:2] else "test" if pair in shuffled[2:4] else "calib"
    if set(assigned) != set(ids):
        raise ValueError("unassigned sample IDs in V3 split")
    samples = [{**row, "split": assigned[row["sample_id"]]} for row in records]
    splits = {name: [row["sample_id"] for row in samples if row["split"] == name]
              for name in ("calib", "validation", "test")}
    counts = {name: len(rows) for name, rows in splits.items()}
    source_counts = {name: len({row["source_group"] for row in samples if row["split"] == name}) for name in splits}
    if counts != {"calib": 60, "validation": 20, "test": 20} or source_counts != {"calib": 54, "validation": 18, "test": 18}:
        raise ValueError("V3 split counts differ from the approved protocol")
    result = {"schema_version": "headkv-v3-source-split-v1", "status": "frozen", "seed": 20261113,
              "samples": samples, "splits": splits, "counts": counts, "source_counts": source_counts}
    write_json(output, result)
    return result


def _float_from_bf16_bits(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _read_sample(record, *, include_vectors, validate_vectors, layer_ids=None):
    directory = Path(record["sample_dir"])
    metadata = read_json(directory / "metadata.json")
    if metadata["status"] != "completed" or metadata["covariance_ddof"] != 0 or metadata["scope_order"] != ["prompt", "middle"]:
        raise ValueError(f"V2 statistics protocol differs: {record['sample_id']}")
    layers, heads, dimension = (record[k] for k in ("num_layers", "num_kv_heads", "head_dim"))
    length = record["prompt_length"]
    expected = np.concatenate((np.arange(32), 64 + ((2*np.arange(448)+1)*(length-128))//896, np.arange(length-32, length)))
    regions = np.repeat(np.array([0, 1, 2], dtype=np.uint8), [32, 448, 32])
    paths = {f"prefill/layer_{layer:03d}.npz" for layer in range(layers)}
    if {key for key in metadata["files"] if key.startswith("prefill/")} != paths:
        raise ValueError(f"Prefill Layer coverage differs: {record['sample_id']}")
    means, covariances, counts, vectors = [], [], [], []
    selected = list(range(layers)) if layer_ids is None else list(layer_ids)
    if not selected or len(set(selected)) != len(selected) or any(l < 0 or l >= layers for l in selected):
        raise ValueError("selected Layer IDs must be unique and within the collection")
    for layer in selected:
        relative = f"prefill/layer_{layer:03d}.npz"
        entry = metadata["files"][relative]
        if entry["layer_id"] != layer or entry["decode_step"] is not None:
            raise ValueError("Prefill file Layer/phase metadata differs")
        with np.load(directory / relative, allow_pickle=False) as a:
            if not (np.array_equal(a["sample_positions"], expected) and np.array_equal(a["sample_regions"], regions)):
                raise ValueError("V2 sampled position/region coordinates differ")
            layer_means, layer_covariances, layer_counts, layer_vectors = [], [], [], []
            for rep in REPRESENTATIONS:
                n, mean, covariance = (a[f"{rep}_{key}"] for key in ("count", "mean", "covariance"))
                if (n.shape != (2,) or not np.issubdtype(n.dtype, np.integer)
                        or not np.array_equal(n, [length, length-128])
                        or mean.shape != (2, heads, dimension) or mean.dtype != np.float32
                        or covariance.shape != (2, heads, dimension, dimension) or covariance.dtype != np.float32
                        or not np.isfinite(mean).all() or not np.isfinite(covariance).all()):
                    raise ValueError(f"invalid full Prompt/middle {rep} moments")
                layer_means.append(mean[1]); layer_covariances.append(covariance[1]); layer_counts.append(n[1])
            if validate_vectors or include_vectors:
                for rep in VECTOR_REPRESENTATIONS:
                    bits = a[rep]
                    descriptor = entry["fields"][rep]
                    if (bits.shape != (heads, 512, dimension) or bits.dtype != np.uint16
                            or descriptor.get("tensor_dtype") != "bfloat16"
                            or descriptor["stored_dtype"] != "uint16" or descriptor["shape"] != list(bits.shape)):
                        raise ValueError(f"invalid BF16 raw vector encoding: {rep}")
                    value = _float_from_bf16_bits(bits)
                    if not np.isfinite(value).all():
                        raise ValueError(f"nonfinite saved raw vectors: {rep}")
                    if include_vectors:
                        layer_vectors.append(value[:, 32:480])
            means.append(np.stack(layer_means)); covariances.append(np.stack(layer_covariances)); counts.append(np.asarray(layer_counts))
            if include_vectors:
                vectors.append(np.stack(layer_vectors))
    result = {"mean": np.stack(means, 1), "covariance": np.stack(covariances, 1),
              "count": np.stack(counts, 1), "positions": expected[32:480].astype(np.uint32)}
    if include_vectors:
        result["vectors"] = np.stack(vectors, 1)
    return result


def read_sample(record, include_vectors=False, *, layer_ids=None):
    """Read middle moments [K_post/V,Layer,Head,...], optionally three raw samples."""
    return _read_sample(record, include_vectors=include_vectors, validate_vectors=False, layer_ids=layer_ids)


def preflight(records, progress=None):
    """Read all full/middle moments and raw BF16 vectors, without reconstruction."""
    layers = 0
    for i, record in enumerate(records, 1):
        _read_sample(record, include_vectors=False, validate_vectors=True)
        layers += record["num_layers"]
        message = {"event": "v3_preflight_sample", "sample_id": record["sample_id"], "completed": i, "total": len(records)}
        if progress is None:
            print(json.dumps(message), flush=True)
        else:
            progress(message)
    return {"status": "passed", "sample_count": len(records), "prefill_layer_files": layers,
            "moment_representations": list(REPRESENTATIONS), "moment_scopes": ["prompt", "middle"],
            "raw_representations": list(VECTOR_REPRESENTATIONS), "raw_positions": 512, "middle_positions": 448,
            "reconstruction_errors_computed": False}
