"""Validate B2 sources; summarize coverage or Bootstrap CV/coverage relationships."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.head_analysis.analysis import b2_coverage, b2_joint, b2_relationships
from experiments.head_analysis.analysis.b2_consistency import (
    B2SourceError,
    build_b2_analysis_plan,
    load_completed_b2_run_set,
)
from experiments.head_analysis.analysis.b2_coverage import summarize_b2_coverage
from experiments.head_analysis.analysis.b2_joint import analyze_b2_joint, load_b2_joint_inputs
from experiments.head_analysis.analysis.b2_relationships import (
    analyze_b2_relationships,
    load_stage2_relationship_inputs,
)
from experiments.head_analysis.common.io import canonical_json_sha256, sha256_file
from experiments.head_analysis.common.io import write_json as _write_json_new
from experiments.head_analysis.common.io import write_npz


def write_analysis_plan_new(output_dir: Path | str, plan: dict) -> Path:
    destination = Path(output_dir)
    if destination.exists():
        raise B2SourceError(f"analysis output directory already exists: {destination}")
    destination.mkdir(parents=True)
    path = destination / "analysis_plan.json"
    _write_json_new(path, plan)
    return path


def prepare_analysis(args: argparse.Namespace) -> Path:
    if sum((args.run_stage2, args.run_stage3, args.run_stage4)) > 1:
        raise B2SourceError("run stages 2, 3 and 4 in separate output directories")
    if (args.run_stage3 or args.run_stage4) and not args.stage2_dir:
        raise B2SourceError("stages 3/4 require --stage2-dir")
    if args.run_stage4 and not args.stage3_dir:
        raise B2SourceError("--run-stage4 requires --stage3-dir")
    if args.bootstrap_replicates <= 0 or args.bootstrap_batch_size <= 0:
        raise B2SourceError("positive Bootstrap counts required")
    output = Path(args.output_dir)
    if output.exists():
        raise B2SourceError(f"analysis output directory already exists: {output}")
    code = {"git_commit": args.analysis_git_commit}
    started = time.perf_counter()
    index = load_completed_b2_run_set(
        args.run_set,
        source_root=args.source_root,
        expected_run_set_sha256=args.expected_run_set_sha256,
        expected_batch_count=args.expected_batches,
        expected_sample_count=args.expected_samples,
    )
    plan = build_b2_analysis_plan(index, analysis_id=args.analysis_id, analysis_code=code)
    relationship_inputs = None
    joint_inputs = None
    if args.run_stage4:
        joint_inputs = load_b2_joint_inputs(
            index,
            args.stage2_dir,
            args.stage3_dir,
            expected_stage2_sha256=args.expected_stage2_manifest_sha256,
            expected_stage3_sha256=args.expected_stage3_manifest_sha256,
        )
        plan["joint_sources"] = joint_inputs.sources
        plan["parameters"]["bootstrap_replicates"] = args.bootstrap_replicates
        plan["parameters"]["bootstrap_seed"] = args.bootstrap_seed
        plan["parameters_sha256"] = canonical_json_sha256(plan["parameters"])
        plan["stage4_parameters"] = {
            "query_features": list(b2_joint.QUERY_RELATION_FEATURES),
            "page_scopes": ["all_visible", "prompt_only"],
            "page_k": [8, 16, 32],
            "token_k": [16, 32, 64],
            "deltas": list(b2_joint.RCO_DELTAS),
            "primary_k": 32,
            "primary_deltas": list(b2_joint.PRIMARY_RELATION_DELTAS),
            "rco_metric": "rco_raw_mean",
            "bootstrap_batch_size": args.bootstrap_batch_size,
            "global_bootstrap": "all configurations",
            "per_layer_bootstrap": "primary X and primary Y only",
            "gqa_ratio": 4,
            "missing_rco": "NaN/count zero",
            "attention_concentration": "separate associations; no partial regression",
        }
        plan["stage4_parameters_sha256"] = canonical_json_sha256(plan["stage4_parameters"])
    if args.run_stage3:
        relationship_inputs = load_stage2_relationship_inputs(
            args.stage2_dir, index, expected_manifest_sha256=args.expected_stage2_manifest_sha256
        )
        plan["stage2_source"] = {
            "path": str(relationship_inputs.stage2_manifest_path),
            "sha256": relationship_inputs.stage2_manifest_sha256,
        }
        plan["parameters"]["bootstrap_replicates"] = args.bootstrap_replicates
        plan["parameters"]["bootstrap_seed"] = args.bootstrap_seed
        plan["parameters_sha256"] = canonical_json_sha256(plan["parameters"])
        plan["stage3_parameters"] = {
            "targets": list(b2_relationships.TARGETS),
            "views": list(b2_relationships.VIEWS),
            "methods": list(b2_relationships.METHODS),
            "minimum_valid_heads": 3,
            "seed_derivation": "SHA256(f'{seed}:{unit_label}') first 8 bytes little-endian",
            "paired_parameter_sensitivity": True,
            "paired_pg19_lengths": True,
            "temporal_relations": "primary/512/Prompt; learned and augmented q1-q4 for C>=32",
        }
        plan["stage3_parameters_sha256"] = canonical_json_sha256(plan["stage3_parameters"])
    if args.run_stage2:
        plan["stage2_parameters"] = {
            "statistics": list(b2_coverage.STATISTICS),
            "recall_scopes": list(b2_coverage.RECALL_SCOPES),
            "time_metrics": list(b2_coverage.TIME_METRICS),
            "quantile_method": "linear",
            "working_dtype": "float64",
            "summary_dtype": "float32",
            "fixed_mass_tolerance": b2_coverage.TOLERANCE,
            "auc_missing_policy": "require_all_steps_finite",
            "curve_progress_bins": 20,
            "minimum_temporal_length": 32,
            "memory_map_column_block": 128,
        }
        plan["stage2_parameters_sha256"] = canonical_json_sha256(plan["stage2_parameters"])
    plan["source_validation_wall_seconds"] = time.perf_counter() - started
    plan["process_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    plan["peak_rss_measurement"] = "Linux process high-water RSS including imports; ru_maxrss KiB converted to bytes"
    path = write_analysis_plan_new(output, plan)
    if args.run_stage2:
        stage_started = time.perf_counter()

        def progress(sample_id, completed, total):
            print(
                json.dumps(
                    {
                        "event": "b2_coverage_sample_completed",
                        "sample_id": sample_id,
                        "completed": completed,
                        "total": total,
                    }
                ),
                flush=True,
            )

        coverage = summarize_b2_coverage(index, scratch_dir=args.scratch_dir, on_sample=progress)
        write_stage2_outputs(output, path, index, coverage, time.perf_counter() - stage_started)
    if args.run_stage3:
        stage_started = time.perf_counter()

        def relationship_progress(completed, relation_id):
            if completed % 12 == 0:
                print(
                    json.dumps(
                        {"event": "b2_relationship_progress", "completed": completed, "relation_id": relation_id}
                    ),
                    flush=True,
                )

        relations = analyze_b2_relationships(
            relationship_inputs,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            on_record=relationship_progress,
        )
        write_stage3_outputs(output, path, relationship_inputs, relations, time.perf_counter() - stage_started)
    if args.run_stage4:
        stage_started = time.perf_counter()

        def joint_progress(event):
            print(json.dumps({"event": "b2_joint_progress", **event}), flush=True)

        joint = analyze_b2_joint(
            joint_inputs,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_batch_size=args.bootstrap_batch_size,
            on_progress=joint_progress,
        )
        write_stage4_outputs(output, path, joint_inputs, joint, time.perf_counter() - stage_started)
    print(
        json.dumps(
            {
                "status": "partial" if args.run_stage2 or args.run_stage3 or args.run_stage4 else "prepared",
                "analysis_plan": str(path),
                "sample_count": len(index.samples),
                "batch_count": len(index.batches),
                "condition_counts": index.condition_counts,
                "completed_analysis_manifest_created": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return path


def _json_values(values):
    values = np.asarray(values).astype(object)
    values[~np.isfinite(np.asarray(values, dtype=np.float64))] = None
    return values.tolist()


def _head_mean(values):
    finite = np.isfinite(values)
    count = finite.sum(axis=(1, 2))
    result = np.full(count.shape, np.nan, dtype=np.float64)
    np.divide(np.where(finite, values, 0).sum(axis=(1, 2), dtype=np.float64), count, out=result, where=count > 0)
    return result


def write_stage2_outputs(output, plan_path, index, coverage, wall_seconds):
    write_started = time.perf_counter()
    profile_path = output / "stage2_profiles.npz"
    write_npz(profile_path, coverage.arrays)
    arrays = coverage.arrays
    summary_path = output / "stage2_summary.json"
    summary = {
        "schema_version": "headkv-v1-b2-offline-stage2-v1",
        "status": "partial",
        "source_run_set_sha256": index.run_set_sha256,
        "plan_sha256": sha256_file(plan_path),
        "sample_count": len(index.samples),
        "condition_counts": index.condition_counts,
        "field_axes": dict(coverage.axes),
        "diagnostics": dict(coverage.diagnostics),
        "accumulation_dtype": "float64",
        "persisted_summary_dtype": "float32",
        "quantile_method": "linear",
        "weighting": "sample_equal",
        "auc_missing_policy": "require_all_steps_finite",
        "condition_names": arrays["condition_names"].tolist(),
        "condition_ids": arrays["condition_ids"].tolist(),
        "budgets": arrays["budgets"].tolist(),
        "set_types": arrays["set_types"].tolist(),
        "recall_scopes": arrays["recall_scopes"].tolist(),
        "statistics": arrays["statistics"].tolist(),
        "prefill_feature_names": arrays["prefill_feature_names"].tolist(),
        "distribution_statistics": arrays["distribution_statistics"].tolist(),
        "time_metrics": arrays["time_metrics"].tolist(),
        "control_names": arrays["control_names"].tolist(),
        "budget_pairs": arrays["budget_pairs"].tolist(),
        "condition_head_equal_mean_cv": _json_values(_head_mean(arrays["condition_prefill_features_mean"][..., 0])),
        "condition_head_equal_recall_mean": _json_values(_head_mean(arrays["condition_recall_full_mean"][..., 0])),
        "condition_head_equal_learned_excess_mean": _json_values(
            _head_mean(arrays["condition_learned_excess_full_mean"][..., 0])
        ),
        "coverage_wall_seconds_excluding_output_write": wall_seconds,
        "profile_write_wall_seconds": time.perf_counter() - write_started,
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "pending_stages": ["cross_phase_relations", "b1_b2_relations", "finalize"],
    }
    _write_json_new(summary_path, summary)
    manifest = {
        "schema_version": "headkv-v1-b2-offline-stage2-manifest-v1",
        "status": "partial",
        "source_run_set_sha256": index.run_set_sha256,
        "plan_sha256": sha256_file(plan_path),
        "completed_stages": ["source_validation", "distributions_and_recall"],
        "pending_stages": summary["pending_stages"],
        "outputs": [
            {"relative_path": path.name, "sha256": sha256_file(path), "file_bytes": path.stat().st_size}
            for path in (plan_path, profile_path, summary_path)
        ],
        "logical_array_bytes": int(sum(value.nbytes for value in arrays.values())),
    }
    _write_json_new(output / "stage2_manifest.json", manifest)


def write_stage3_outputs(output, plan_path, inputs, relations, wall_seconds):
    profile_path = output / "stage3_relationships.npz"
    write_started = time.perf_counter()
    write_npz(profile_path, relations.arrays)
    npz_write_seconds = time.perf_counter() - write_started
    records = []
    a = relations.arrays
    for i, metadata in enumerate(relations.records):
        records.append(
            {
                **metadata,
                "coefficients": _json_values(a["correlation"][i]),
                "bootstrap_ci": _json_values(a["correlation_ci"][i]),
                "valid_heads": a["correlation_valid_heads"][i].tolist(),
                "valid_replicates": a["correlation_valid_replicates"][i].tolist(),
            }
        )
    summary_path = output / "stage3_summary.json"
    summary = {
        "schema_version": "headkv-v1-b2-offline-stage3-v1",
        "status": "partial",
        "source_run_set_sha256": inputs.source_run_set_sha256,
        "stage2_source": {"path": str(inputs.stage2_manifest_path), "sha256": inputs.stage2_manifest_sha256},
        "plan_sha256": sha256_file(plan_path),
        "records": records,
        "statistics": dict(relations.metadata),
        "array_descriptors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in a.items()},
        "axes": relations.axes,
        "wall_seconds_excluding_output_write": wall_seconds,
        "npz_write_wall_seconds": npz_write_seconds,
        "logical_array_bytes": sum(v.nbytes for v in a.values()),
        "npz_file_bytes": profile_path.stat().st_size,
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "pending_stages": ["b1_b2_relations", "finalize"],
        "notes": [
            "No Head/Step resampling and no inferential p values.",
            "Pointwise percentile intervals are not simultaneous multiple-comparison confidence bands.",
            "Constant or fewer than 3 valid Heads yields null coefficients; "
            "undefined CI replicates are excluded and counted.",
        ],
    }
    _write_json_new(summary_path, summary)
    _write_json_new(
        output / "stage3_manifest.json",
        {
            "schema_version": "headkv-v1-b2-offline-stage3-manifest-v1",
            "status": "partial",
            "source_run_set_sha256": inputs.source_run_set_sha256,
            "stage2_source": summary["stage2_source"],
            "plan_sha256": summary["plan_sha256"],
            "completed_stages": ["source_validation", "cross_phase_relations"],
            "pending_stages": summary["pending_stages"],
            "outputs": [
                {"relative_path": p.name, "file_bytes": p.stat().st_size, "sha256": sha256_file(p)}
                for p in (plan_path, profile_path, summary_path)
            ],
        },
    )


def write_stage4_outputs(output, plan_path, inputs, joint, wall_seconds):
    profile_path = output / "stage4_joint_profiles.npz"
    started = time.perf_counter()
    write_npz(profile_path, joint.arrays)
    write_seconds = time.perf_counter() - started
    a = joint.arrays
    primary_records = []
    for ci, condition in enumerate(a["condition_names"]):
        for xi in a["primary_x_indices"]:
            for yi in a["primary_y_indices"]:
                primary_records.append(
                    {
                        "condition": str(condition),
                        "x_id": str(a["x_ids"][xi]),
                        "y_id": str(a["y_ids"][yi]),
                        "coefficients": _json_values(a["correlation"][ci, xi, yi]),
                        "bootstrap_ci": _json_values(a["correlation_ci"][ci, xi, yi]),
                        "valid_heads": a["correlation_valid_heads"][ci, xi, yi].tolist(),
                        "valid_replicates": a["correlation_valid_replicates"][ci, xi, yi].tolist(),
                    }
                )
    summary = {
        "schema_version": "headkv-v1-b2-offline-stage4-v1",
        "status": "partial",
        "source_run_set_sha256": inputs.stage2.source_run_set_sha256,
        "joint_sources": inputs.sources,
        "plan_sha256": sha256_file(plan_path),
        "statistics": dict(joint.metadata),
        "relation_count": int(np.prod(a["correlation"].shape[:3])),
        "primary_records": primary_records,
        "array_descriptors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in a.items()},
        "axes": dict(joint.axes),
        "wall_seconds_excluding_output_write": wall_seconds,
        "npz_write_wall_seconds": write_seconds,
        "npz_file_bytes": profile_path.stat().st_size,
        "logical_array_bytes": sum(v.nbytes for v in a.values()),
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "pending_stages": ["finalize"],
        "notes": [
            "Sample/pair Bootstrap; Head/Layer/Step are fixed coordinates, not independent resampling units.",
            "All global configuration CIs retained; per-Layer CIs only on pre-registered primary axes.",
            "B1 query/token metrics map to KV Heads after within-sample time aggregation; equal GQA weights.",
            "Missing RULER long-delta RCO stays NaN/count zero; JSON uses null.",
            "Attention concentration associations are not multivariate partial-correlation controls.",
            "Pointwise percentile CIs; no p values, simultaneous bands, causal or system-efficiency claims.",
        ],
    }
    summary_path = output / "stage4_summary.json"
    _write_json_new(summary_path, summary)
    _write_json_new(
        output / "stage4_manifest.json",
        {
            "schema_version": "headkv-v1-b2-offline-stage4-manifest-v1",
            "status": "partial",
            "source_run_set_sha256": summary["source_run_set_sha256"],
            "joint_sources": inputs.sources,
            "plan_sha256": summary["plan_sha256"],
            "completed_stages": ["source_validation", "b1_b2_relations"],
            "pending_stages": ["finalize"],
            "outputs": [
                {"relative_path": p.name, "file_bytes": p.stat().st_size, "sha256": sha256_file(p)}
                for p in (plan_path, profile_path, summary_path)
            ],
        },
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-set", required=True)
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--expected-run-set-sha256", default=None)
    parser.add_argument("--expected-batches", type=int, default=10)
    parser.add_argument("--expected-samples", type=int, default=100)
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--analysis-git-commit", help="optional source revision metadata")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run-stage2", action="store_true", help="summarize validated masses; publish partial outputs only"
    )
    parser.add_argument("--scratch-dir", help="parent directory for automatically removed per-sample memory maps")
    parser.add_argument("--run-stage3", action="store_true")
    parser.add_argument("--run-stage4", action="store_true")
    parser.add_argument("--stage2-dir", help="independently accepted stage2 directory")
    parser.add_argument("--stage3-dir", help="independently accepted stage3 directory")
    parser.add_argument("--expected-stage2-manifest-sha256", default=None)
    parser.add_argument("--expected-stage3-manifest-sha256", default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261110)
    parser.add_argument("--bootstrap-batch-size", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    if args.expected_batches <= 0 or args.expected_samples <= 0:
        raise SystemExit("expected batch/sample counts must be positive")
    if args.bootstrap_replicates <= 0 or args.bootstrap_batch_size <= 0:
        raise SystemExit("Bootstrap replicate and batch counts must be positive")
    try:
        prepare_analysis(args)
    except (B2SourceError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
