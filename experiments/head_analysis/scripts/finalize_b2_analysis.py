"""Collect B2 stage results into a portable numeric and text bundle."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from experiments.head_analysis.analysis.b2_consistency import B2SourceError
from experiments.head_analysis.common.io import read_json, sha256_file, write_json

NUMERIC = {2: "stage2_profiles.npz", 3: "stage3_relationships.npz", 4: "stage4_joint_profiles.npz"}


def _require(value, message):
    if not value:
        raise B2SourceError(message)


def _read_stage(number, directory):
    directory = Path(directory).resolve()
    name = f"stage{number}"
    manifest_path = directory / f"{name}_manifest.json"
    manifest = read_json(manifest_path)
    plan = read_json(directory / "analysis_plan.json")
    summary = read_json(directory / f"{name}_summary.json")
    _require(manifest.get("schema_version") == f"headkv-v1-b2-offline-{name}-manifest-v1", f"invalid {name} manifest")
    _require(summary.get("schema_version") == f"headkv-v1-b2-offline-{name}-v1", f"invalid {name} summary")
    _require(manifest.get("status") == summary.get("status") == "partial", f"invalid {name} status")
    source = manifest.get("source_run_set_sha256")
    _require(isinstance(source, str) and bool(source), f"missing {name} source")
    _require(source == summary.get("source_run_set_sha256") == plan.get("source_run_set", {}).get("sha256"), f"{name} source differs")
    _require(manifest.get("plan_sha256") == summary.get("plan_sha256"), f"{name} plan references differ")
    required = {"analysis_plan.json", f"{name}_summary.json", NUMERIC[number]}
    entries = manifest.get("outputs", [])
    names = [entry.get("relative_path") for entry in entries]
    _require(len(names) == len(required) and set(names) == required, f"{name} output list differs")
    for filename in required:
        _require((directory / filename).is_file(), f"missing {name} file: {filename}")
    axes = summary.get("field_axes" if number == 2 else "axes")
    _require(isinstance(axes, dict), f"{name} has no array axes")
    catalog = {}
    with np.load(directory / NUMERIC[number], allow_pickle=False) as arrays:
        _require(set(arrays.files) == set(axes), f"{name} array fields differ from axes")
        for key in arrays.files:
            value = arrays[key]
            _require(value.dtype.kind != "O" and value.ndim == len(axes[key]), f"{name} invalid array axes: {key}")
            catalog[key] = {"shape": list(value.shape), "dtype": str(value.dtype), "axes": axes[key]}
        coordinates = {k: arrays[k].tolist() for k in ("sample_ids", "sample_pair_ids", "condition_names")}
        sample_conditions = arrays["sample_conditions"].tolist() if number in (2, 4) else None
    return {"directory": directory, "manifest": manifest, "manifest_path": manifest_path,
            "plan": plan, "summary": summary, "source": source, "catalog": catalog, "coordinates": coordinates,
            "sample_conditions": sample_conditions}


def finalize(stage2_dir, stage3_dir, stage4_dir, output_dir, analysis_id, *, report=None):
    destination = Path(output_dir)
    _require(not destination.exists(), f"output directory already exists: {destination}")
    _require(bool(analysis_id) and Path(analysis_id).name == analysis_id and analysis_id not in (".", ".."), "invalid analysis ID")
    stages = {n: _read_stage(n, directory) for n, directory in ((2, stage2_dir), (3, stage3_dir), (4, stage4_dir))}
    baseline = stages[2]
    ids = baseline["coordinates"]["sample_ids"]
    _require(isinstance(ids, list) and len(ids) == len(set(ids)), "duplicate or invalid sample IDs")
    _require(len(baseline["coordinates"]["sample_pair_ids"]) == len(ids), "sample pair coverage differs")
    _require(len(baseline["sample_conditions"]) == len(ids), "sample condition coverage differs")
    for number, stage in stages.items():
        _require(stage["source"] == baseline["source"], f"stage{number} source differs")
        _require(stage["coordinates"] == baseline["coordinates"], f"stage{number} coordinates differ")
        _require(stage["plan"].get("sample_ids") == ids, f"stage{number} sample plan differs")
        _require(stage["plan"].get("condition_counts") == baseline["plan"].get("condition_counts"), f"stage{number} condition counts differ")
    _require(stages[4]["sample_conditions"] == baseline["sample_conditions"], "stage4 sample conditions differ")
    stage2_hash = sha256_file(stages[2]["manifest_path"])
    stage3_hash = sha256_file(stages[3]["manifest_path"])
    _require(stages[3]["manifest"].get("stage2_source", {}).get("sha256") == stage2_hash, "stage3 references another stage2")
    joint = stages[4]["manifest"].get("joint_sources", {})
    _require(joint.get("stage2", {}).get("sha256") == stage2_hash and joint.get("stage3", {}).get("sha256") == stage3_hash, "stage4 references different stages")
    if report is not None:
        _require(Path(report).is_file(), f"missing report: {report}")
    destination.mkdir(parents=True)
    catalog, sources = {}, {}
    for number, stage in stages.items():
        name = f"stage{number}"
        target = destination / "stages" / name
        target.mkdir(parents=True)
        for filename in ("analysis_plan.json", f"{name}_summary.json", f"{name}_manifest.json", NUMERIC[number]):
            shutil.copyfile(stage["directory"] / filename, target / filename)
        catalog[name] = {"numeric_file": f"stages/{name}/{NUMERIC[number]}", "fields": stage["catalog"]}
        sources[name] = {"path": str(stage["directory"]), "manifest_sha256": sha256_file(stage["manifest_path"])}
    write_json(destination / "array_catalog.json", catalog)
    if report is not None:
        shutil.copyfile(report, destination / "REPORT.md")
    else:
        (destination / "REPORT.md").write_text(
            "# B2 分析结果索引\n\n本包保留阶段数值和摘要，没有重新计算指标。\n\n"
            "- stage2：分布与覆盖；stage3：CV 与覆盖关系；stage4：B1/B2 联合关系。\n"
            "- 各阶段 analysis_plan.json 记录参数，summary.json 记录结果与有效数，array_catalog.json 记录数组轴。\n"
            "- 此索引不预设研究结论；请结合对应阶段研究报告解释结果。\n", encoding="utf-8")
    write_json(destination / "analysis_manifest.json", {
        "schema_version": "headkv-v1-b2-numeric-bundle-v2", "status": "completed",
        "analysis_id": analysis_id, "bundle_kind": "numeric_and_text_only", "source_run_set_sha256": baseline["source"],
        "sample_count": len(ids), "condition_counts": baseline["plan"].get("condition_counts"),
        "stages": sources, "arrays": "array_catalog.json", "report": "REPORT.md",
        "validation": "stage source links, sample coordinates, array fields and dimensions; no metric recalculation",
    })
    return destination / "analysis_manifest.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in (2, 3, 4):
        parser.add_argument(f"--stage{stage}-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--report", help="optional existing text report to include")
    args = parser.parse_args(argv)
    try:
        path = finalize(args.stage2_dir, args.stage3_dir, args.stage4_dir, args.output_dir, args.analysis_id, report=args.report)
    except (B2SourceError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps({"status": "completed", "manifest": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
