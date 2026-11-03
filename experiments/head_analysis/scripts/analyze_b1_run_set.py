"""Prepare a new, validated B1 offline-analysis directory.

The current command implements phase 1 of the frozen offline-analysis plan. It
does not compute metrics and therefore never creates a completed analysis
manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.head_analysis.analysis.b1_offline import (
    DEFAULT_ANALYSIS_PARAMETERS,
    B1OfflineError,
    PageMetricSummary,
    PageRCOSummary,
    QueryMetricSummary,
    aggregate_page_rco,
    build_analysis_plan,
    load_completed_run_set,
    sha256_file,
    summarize_query_and_page_metrics,
    validate_analysis_sources,
)


def validate_analysis_git_state(expected_commit: str) -> None:
    """Require the plan commit to be the current clean analysis checkout."""

    try:
        commit = subprocess.run(("git", "rev-parse", "HEAD"), check=True, text=True, capture_output=True).stdout.strip()
        status = subprocess.run(
            ("git", "status", "--porcelain", "--untracked-files=all"),
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise B1OfflineError(f"unable to inspect analysis Git state: {exc}") from exc
    if commit != expected_commit:
        raise B1OfflineError(f"analysis HEAD {commit!r} differs from expected {expected_commit!r}")
    if status:
        raise B1OfflineError(f"analysis working tree is not clean: {status}")


def write_json_new(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Atomically create one strict JSON file without replacing an old one."""

    destination = Path(path)
    if destination.exists():
        raise B1OfflineError(f"analysis output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise B1OfflineError(f"analysis output already exists: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def prepare_analysis_directory_new(output_dir: Path | str, plan: Mapping[str, Any]) -> Path:
    """Create a new analysis directory containing only its immutable plan."""

    destination = Path(output_dir)
    if destination.exists():
        raise B1OfflineError(f"analysis directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    plan_path = destination / "analysis_plan.json"
    write_json_new(plan_path, plan)
    return plan_path


def write_npz_new(path: Path | str, arrays: Mapping[str, np.ndarray]) -> None:
    destination = Path(path)
    if destination.exists():
        raise B1OfflineError(f"analysis output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise B1OfflineError(f"analysis output already exists: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _json_array(array: np.ndarray) -> Any:
    if np.issubdtype(array.dtype, np.floating):
        values = array.astype(object)
        values[~np.isfinite(array)] = None
        return values.tolist()
    return array.tolist()


def _profile_mean(values: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    valid = np.isfinite(values)
    counts = valid.sum(axis=axes)
    output = np.full(counts.shape, np.nan, dtype=np.float32)
    np.divide(
        np.where(valid, values, 0.0).sum(axis=axes, dtype=np.float64),
        counts,
        out=output,
        where=counts > 0,
    )
    return output


def stage3_profile_arrays(
    query: QueryMetricSummary,
    page: PageMetricSummary,
    rco: PageRCOSummary,
) -> dict[str, np.ndarray]:
    return {
        "sample_ids": np.asarray(query.sample_ids),
        "sample_conditions": np.asarray(query.sample_conditions),
        "condition_names": np.asarray(query.condition_names),
        "query_feature_names": np.asarray(query.feature_names),
        "query_statistics": np.asarray(query.statistics),
        "query_phases": np.asarray(query.phases),
        "decode_segments": np.asarray(query.decode_segments),
        "query_sample_full": query.sample_full,
        "query_sample_full_valid_count": query.sample_full_valid_count,
        "query_sample_decode_segments": query.sample_decode_segments,
        "query_sample_decode_segment_valid_count": query.sample_decode_segment_valid_count,
        "query_condition_full_mean": query.condition_full_mean,
        "query_condition_full_valid_samples": query.condition_full_valid_samples,
        "query_condition_decode_segment_mean": query.condition_decode_segment_mean,
        "query_condition_decode_segment_valid_samples": query.condition_decode_segment_valid_samples,
        "page_scopes": np.asarray(page.scopes),
        "page_metrics": np.asarray(page.metrics),
        "page_sample_full": page.sample_full,
        "page_sample_full_valid_count": page.sample_full_valid_count,
        "page_sample_decode_segments": page.sample_decode_segments,
        "page_sample_decode_segment_valid_count": page.sample_decode_segment_valid_count,
        "page_condition_full_mean": page.condition_full_mean,
        "page_condition_full_valid_samples": page.condition_full_valid_samples,
        "page_condition_decode_segment_mean": page.condition_decode_segment_mean,
        "page_condition_decode_segment_valid_samples": page.condition_decode_segment_valid_samples,
        "rco_k_values": np.asarray(rco.k_values, dtype=np.uint32),
        "rco_deltas": np.asarray(rco.deltas, dtype=np.uint32),
        "rco_metrics": np.asarray(rco.metrics),
        "rco_sample_head_values": rco.sample_head_values,
        "rco_sample_head_valid_pair_count": rco.sample_head_valid_pair_count,
        "rco_condition_head_sample_equal_mean": rco.condition_head_sample_equal_mean,
        "rco_condition_head_valid_samples": rco.condition_head_valid_samples,
        "rco_condition_head_pair_weighted_mean": rco.condition_head_pair_weighted_mean,
        "rco_condition_head_valid_pair_count": rco.condition_head_valid_pair_count,
        "rco_pg19_pair_ids": np.asarray(rco.pg19_pair_ids),
        "rco_pg19_pair_head_difference_32k_minus_8k": rco.pg19_pair_head_difference_32k_minus_8k,
        "rco_pg19_pair_head_mean_difference": rco.pg19_pair_head_mean_difference,
    }


def stage3_summary_payload(
    *,
    analysis_id: str,
    query: QueryMetricSummary,
    page: PageMetricSummary,
    rco: PageRCOSummary,
) -> dict[str, Any]:
    return {
        "schema_version": "headkv-v1-b1-offline-stage3-v1",
        "analysis_id": analysis_id,
        "status": "partial",
        "pending_stages": ["token_stability", "metric_relationships", "completed_manifest"],
        "axes": {
            "conditions": list(query.condition_names),
            "query_phases": list(query.phases),
            "query_features": list(query.feature_names),
            "page_scopes": list(page.scopes),
            "page_metrics": list(page.metrics),
            "page_rco_k": list(rco.k_values),
            "deltas": list(rco.deltas),
            "page_rco_metrics": list(rco.metrics),
            "ci_bounds": [0.025, 0.975],
        },
        "query_condition_head_profile_mean": _json_array(_profile_mean(query.condition_full_mean, axes=(2, 3))),
        "page_condition_head_profile_mean": _json_array(_profile_mean(page.condition_full_mean, axes=(1, 2))),
        "page_rco": {
            "condition_scalar_sample_equal_mean": _json_array(rco.condition_scalar_sample_equal_mean),
            "condition_scalar_valid_samples": _json_array(rco.condition_scalar_valid_samples),
            "condition_scalar_pair_weighted_mean": _json_array(rco.condition_scalar_pair_weighted_mean),
            "condition_scalar_valid_pair_count": _json_array(rco.condition_scalar_valid_pair_count),
            "condition_scalar_bootstrap_ci": _json_array(rco.condition_scalar_bootstrap_ci),
            "pg19_pair_ids": list(rco.pg19_pair_ids),
            "pg19_difference_direction": "32k_minus_8k",
            "pg19_pair_scalar_mean_difference": _json_array(rco.pg19_pair_scalar_mean_difference),
            "pg19_pair_scalar_bootstrap_ci": _json_array(rco.pg19_pair_scalar_bootstrap_ci),
            "bootstrap_replicates": rco.bootstrap_replicates,
            "bootstrap_seed": rco.bootstrap_seed,
        },
    }


def write_stage3_bundle_new(
    output_dir: Path | str,
    *,
    plan_path: Path,
    query: QueryMetricSummary,
    page: PageMetricSummary,
    rco: PageRCOSummary,
) -> Path:
    root = Path(output_dir)
    profiles_path = root / "stage3_profiles.npz"
    summary_path = root / "stage3_summary.json"
    partial_manifest_path = root / "stage3_manifest.json"
    write_npz_new(profiles_path, stage3_profile_arrays(query, page, rco))
    write_json_new(
        summary_path,
        stage3_summary_payload(
            analysis_id=json.loads(plan_path.read_text(encoding="utf-8"))["analysis_id"],
            query=query,
            page=page,
            rco=rco,
        ),
    )
    outputs = [
        {"relative_path": path.name, "sha256": sha256_file(path), "file_bytes": path.stat().st_size}
        for path in (profiles_path, summary_path)
    ]
    write_json_new(
        partial_manifest_path,
        {
            "schema_version": "headkv-v1-b1-offline-stage3-manifest-v1",
            "status": "partial",
            "analysis_id": json.loads(plan_path.read_text(encoding="utf-8"))["analysis_id"],
            "plan_sha256": sha256_file(plan_path),
            "outputs": outputs,
            "pending_stages": ["token_stability", "metric_relationships", "completed_manifest"],
        },
    )
    return partial_manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate frozen B1 sources and create a non-overwriting offline-analysis plan."
    )
    parser.add_argument("--run-set", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--page-rco-report", type=Path, action="append", required=True)
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--analysis-git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-batches", type=int, default=10)
    parser.add_argument("--expected-samples", type=int, default=100)
    parser.add_argument("--run-stage3", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261103)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_analysis_git_state(args.analysis_git_commit)
        index = load_completed_run_set(
            args.run_set,
            source_root=args.source_root,
            expected_batch_count=args.expected_batches,
            expected_sample_count=args.expected_samples,
            verify_file_hashes=True,
            verify_trace_arrays=True,
        )
        validation = validate_analysis_sources(index, args.page_rco_report)
        parameters = dict(DEFAULT_ANALYSIS_PARAMETERS)
        parameters["bootstrap_replicates"] = args.bootstrap_replicates
        parameters["bootstrap_seed"] = args.bootstrap_seed
        plan = build_analysis_plan(
            index,
            validation,
            analysis_id=args.analysis_id,
            git_commit=args.analysis_git_commit,
            parameters=parameters,
        )
        plan_path = prepare_analysis_directory_new(args.output_dir, plan)
        partial_manifest = None
        if args.run_stage3:
            query, page = summarize_query_and_page_metrics(index)
            rco = aggregate_page_rco(
                index,
                args.page_rco_report,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed,
            )
            partial_manifest = write_stage3_bundle_new(
                args.output_dir,
                plan_path=plan_path,
                query=query,
                page=page,
                rco=rco,
            )
    except B1OfflineError as exc:
        raise SystemExit(f"B1 offline analysis preparation failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "prepared",
                "analysis_id": plan["analysis_id"],
                "sample_count": plan["sample_count"],
                "condition_counts": plan["condition_counts"],
                "analysis_plan": str(plan_path),
                "stage3_manifest": None if partial_manifest is None else str(partial_manifest),
                "completed_manifest_created": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
