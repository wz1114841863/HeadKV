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
    canonical_json_sha256,
    load_completed_run_set,
    sha256_file,
    summarize_query_and_page_metrics,
    validate_analysis_sources,
)
from experiments.head_analysis.analysis.metric_relationships import (
    PRIMARY_RELATION_DELTAS,
    MetricRelationshipSummary,
    analyze_metric_relationships,
    load_relationship_inputs,
)
from experiments.head_analysis.analysis.token_stability import TokenStabilitySummary, summarize_token_stability


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


def stage4_profile_arrays(token: TokenStabilitySummary) -> dict[str, np.ndarray]:
    return {
        "sample_ids": np.asarray(token.sample_ids),
        "sample_conditions": np.asarray(token.sample_conditions),
        "condition_names": np.asarray(token.condition_names),
        "token_k_values": np.asarray(token.k_values, dtype=np.uint32),
        "token_deltas": np.asarray(token.deltas, dtype=np.uint32),
        "token_metrics": np.asarray(token.metrics),
        "token_age_statistics": np.asarray(token.age_statistics),
        "token_sample_head_metrics": token.sample_head_metrics,
        "token_sample_head_valid_pair_count": token.sample_head_valid_pair_count,
        "token_sample_head_rank_valid_pair_count": token.sample_head_rank_valid_pair_count,
        "token_sample_head_age": token.sample_head_age,
        "token_sample_head_age_valid_token_count": token.sample_head_age_valid_token_count,
        "token_condition_head_sample_equal_mean": token.condition_head_sample_equal_mean,
        "token_condition_head_valid_samples": token.condition_head_valid_samples,
        "token_condition_age_head_sample_equal_mean": token.condition_age_head_sample_equal_mean,
        "token_pg19_pair_ids": np.asarray(token.pg19_pair_ids),
    }


def stage4_summary_payload(*, analysis_id: str, token: TokenStabilitySummary) -> dict[str, Any]:
    return {
        "schema_version": "headkv-v1-b1-offline-stage4-v1",
        "analysis_id": analysis_id,
        "status": "partial",
        "pending_stages": ["metric_relationships", "completed_manifest"],
        "axes": {
            "conditions": list(token.condition_names),
            "token_k": list(token.k_values),
            "deltas": list(token.deltas),
            "token_metrics": list(token.metrics),
            "age_statistics": list(token.age_statistics),
            "ci_bounds": [0.025, 0.975],
        },
        "condition_scalar_sample_equal_mean": _json_array(token.condition_scalar_sample_equal_mean),
        "condition_scalar_valid_samples": _json_array(token.condition_scalar_valid_samples),
        "condition_scalar_bootstrap_ci": _json_array(token.condition_scalar_bootstrap_ci),
        "pg19_pair_ids": list(token.pg19_pair_ids),
        "pg19_difference_direction": "32k_minus_8k",
        "pg19_pair_scalar_mean_difference": _json_array(token.pg19_pair_scalar_mean_difference_32k_minus_8k),
        "pg19_pair_scalar_bootstrap_ci": _json_array(token.pg19_pair_scalar_bootstrap_ci),
        "bootstrap_replicates": token.bootstrap_replicates,
        "bootstrap_seed": token.bootstrap_seed,
    }


def write_stage4_bundle_new(
    output_dir: Path | str,
    *,
    plan_path: Path,
    token: TokenStabilitySummary,
) -> Path:
    root = Path(output_dir)
    profiles_path = root / "stage4_token_profiles.npz"
    summary_path = root / "stage4_token_summary.json"
    partial_manifest_path = root / "stage4_manifest.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    write_npz_new(profiles_path, stage4_profile_arrays(token))
    write_json_new(summary_path, stage4_summary_payload(analysis_id=plan["analysis_id"], token=token))
    outputs = [
        {"relative_path": path.name, "sha256": sha256_file(path), "file_bytes": path.stat().st_size}
        for path in (profiles_path, summary_path)
    ]
    write_json_new(
        partial_manifest_path,
        {
            "schema_version": "headkv-v1-b1-offline-stage4-manifest-v1",
            "status": "partial",
            "analysis_id": plan["analysis_id"],
            "plan_sha256": sha256_file(plan_path),
            "outputs": outputs,
            "pending_stages": ["metric_relationships", "completed_manifest"],
        },
    )
    return partial_manifest_path


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise B1OfflineError(f"unable to read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise B1OfflineError(f"{label} {path} must contain a JSON object")
    return payload


def _validate_partial_stage_bundle(root: Path, stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = root / "analysis_plan.json"
    manifest_path = root / f"{stage}_manifest.json"
    plan = _read_json_object(plan_path, label=f"{stage} plan")
    manifest = _read_json_object(manifest_path, label=f"{stage} manifest")
    expected_schema = f"headkv-v1-b1-offline-{stage}-manifest-v1"
    if manifest.get("schema_version") != expected_schema or manifest.get("status") != "partial":
        raise B1OfflineError(f"{manifest_path} is not a supported partial {stage} manifest")
    if manifest.get("analysis_id") != plan.get("analysis_id"):
        raise B1OfflineError(f"{stage} plan and manifest analysis_id differ")
    if manifest.get("plan_sha256") != sha256_file(plan_path):
        raise B1OfflineError(f"{stage} plan hash differs from its manifest")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise B1OfflineError(f"{stage} manifest has no outputs")
    for entry in outputs:
        if not isinstance(entry, dict) or not isinstance(entry.get("relative_path"), str):
            raise B1OfflineError(f"{stage} manifest contains an invalid output entry")
        path = root / entry["relative_path"]
        if (
            not path.is_file()
            or path.stat().st_size != entry.get("file_bytes")
            or sha256_file(path) != entry.get("sha256")
        ):
            raise B1OfflineError(f"{stage} output does not match its manifest: {path}")
    return plan, manifest


def validate_relationship_sources(stage3_dir: Path | str, stage4_dir: Path | str) -> dict[str, Any]:
    stage3_root = Path(stage3_dir).resolve()
    stage4_root = Path(stage4_dir).resolve()
    stage3_plan, stage3_manifest = _validate_partial_stage_bundle(stage3_root, "stage3")
    stage4_plan, stage4_manifest = _validate_partial_stage_bundle(stage4_root, "stage4")
    for key in ("source_schema_version", "sample_count", "condition_counts"):
        if stage3_plan.get(key) != stage4_plan.get(key):
            raise B1OfflineError(f"Stage3 and Stage4 plans disagree on {key}")
    if stage3_plan.get("source_run_set", {}).get("sha256") != stage4_plan.get("source_run_set", {}).get("sha256"):
        raise B1OfflineError("Stage3 and Stage4 plans use different frozen Run Sets")
    stage3_reports = sorted(item.get("sha256") for item in stage3_plan.get("source_page_rco_reports", []))
    stage4_reports = sorted(item.get("sha256") for item in stage4_plan.get("source_page_rco_reports", []))
    if stage3_reports != stage4_reports:
        raise B1OfflineError("Stage3 and Stage4 plans use different Page RCO reports")
    parameter_keys = (
        "page_k",
        "token_k",
        "deltas",
        "bootstrap_replicates",
        "bootstrap_seed",
        "primary_weighting",
    )
    for key in parameter_keys:
        if stage3_plan.get("parameters", {}).get(key) != stage4_plan.get("parameters", {}).get(key):
            raise B1OfflineError(f"Stage3 and Stage4 plans disagree on parameter {key!r}")
    return {
        "stage3": {
            "root": str(stage3_root),
            "analysis_id": stage3_plan["analysis_id"],
            "plan_sha256": sha256_file(stage3_root / "analysis_plan.json"),
            "manifest_sha256": sha256_file(stage3_root / "stage3_manifest.json"),
            "profiles_sha256": sha256_file(stage3_root / "stage3_profiles.npz"),
        },
        "stage4": {
            "root": str(stage4_root),
            "analysis_id": stage4_plan["analysis_id"],
            "plan_sha256": sha256_file(stage4_root / "analysis_plan.json"),
            "manifest_sha256": sha256_file(stage4_root / "stage4_manifest.json"),
            "profiles_sha256": sha256_file(stage4_root / "stage4_token_profiles.npz"),
        },
        "source_run_set_sha256": stage3_plan["source_run_set"]["sha256"],
        "source_page_rco_sha256": stage3_reports,
        "bootstrap_replicates": stage3_plan["parameters"]["bootstrap_replicates"],
        "bootstrap_seed": stage3_plan["parameters"]["bootstrap_seed"],
    }


def _estimate_payload(record: Any) -> dict[str, Any]:
    estimate = record.estimate
    return {
        "relation_id": record.relation_id,
        "family": record.family,
        "condition_x": record.condition_x,
        "condition_y": record.condition_y,
        "profile_x": record.profile_x,
        "profile_y": record.profile_y,
        "bootstrap_mode": record.bootstrap_mode,
        "valid_head_count": estimate.valid_head_count,
        "spearman": _json_array(np.asarray(estimate.spearman, dtype=np.float32)),
        "spearman_bootstrap_ci": _json_array(estimate.spearman_bootstrap_ci),
        "pearson": _json_array(np.asarray(estimate.pearson, dtype=np.float32)),
        "pearson_bootstrap_ci": _json_array(estimate.pearson_bootstrap_ci),
        "layer_centered_valid_head_count": estimate.layer_centered_valid_head_count,
        "layer_centered_spearman": _json_array(np.asarray(estimate.layer_centered_spearman, dtype=np.float32)),
        "layer_centered_spearman_bootstrap_ci": _json_array(estimate.layer_centered_spearman_bootstrap_ci),
        "layer_centered_pearson": _json_array(np.asarray(estimate.layer_centered_pearson, dtype=np.float32)),
        "layer_centered_pearson_bootstrap_ci": _json_array(estimate.layer_centered_pearson_bootstrap_ci),
        "per_layer_valid_head_count": _json_array(estimate.per_layer_valid_head_count),
        "per_layer_spearman": _json_array(estimate.per_layer_spearman),
        "per_layer_pearson": _json_array(estimate.per_layer_pearson),
    }


def stage5_profile_arrays(summary: MetricRelationshipSummary) -> dict[str, np.ndarray]:
    records = summary.records
    return {
        "relation_ids": np.asarray([record.relation_id for record in records]),
        "relation_families": np.asarray([record.family for record in records]),
        "condition_x": np.asarray([record.condition_x for record in records]),
        "condition_y": np.asarray([record.condition_y for record in records]),
        "profile_x": np.asarray([record.profile_x for record in records]),
        "profile_y": np.asarray([record.profile_y for record in records]),
        "bootstrap_modes": np.asarray([record.bootstrap_mode for record in records]),
        "valid_head_count": np.asarray([record.estimate.valid_head_count for record in records], dtype=np.uint32),
        "spearman": np.asarray([record.estimate.spearman for record in records], dtype=np.float32),
        "spearman_bootstrap_ci": np.stack([record.estimate.spearman_bootstrap_ci for record in records]),
        "pearson": np.asarray([record.estimate.pearson for record in records], dtype=np.float32),
        "pearson_bootstrap_ci": np.stack([record.estimate.pearson_bootstrap_ci for record in records]),
        "layer_centered_valid_head_count": np.asarray(
            [record.estimate.layer_centered_valid_head_count for record in records], dtype=np.uint32
        ),
        "layer_centered_spearman": np.asarray(
            [record.estimate.layer_centered_spearman for record in records], dtype=np.float32
        ),
        "layer_centered_spearman_bootstrap_ci": np.stack(
            [record.estimate.layer_centered_spearman_bootstrap_ci for record in records]
        ),
        "layer_centered_pearson": np.asarray(
            [record.estimate.layer_centered_pearson for record in records], dtype=np.float32
        ),
        "layer_centered_pearson_bootstrap_ci": np.stack(
            [record.estimate.layer_centered_pearson_bootstrap_ci for record in records]
        ),
        "per_layer_valid_head_count": np.stack([record.estimate.per_layer_valid_head_count for record in records]),
        "per_layer_spearman": np.stack([record.estimate.per_layer_spearman for record in records]),
        "per_layer_pearson": np.stack([record.estimate.per_layer_pearson for record in records]),
    }


def write_stage5_bundle_new(
    output_dir: Path | str,
    *,
    plan_path: Path,
    source_artifacts: Mapping[str, Any],
    summary: MetricRelationshipSummary,
) -> Path:
    root = Path(output_dir)
    profiles_path = root / "stage5_relationship_profiles.npz"
    summary_path = root / "stage5_relationships.json"
    manifest_path = root / "stage5_manifest.json"
    plan = _read_json_object(plan_path, label="Stage5 plan")
    write_npz_new(profiles_path, stage5_profile_arrays(summary))
    write_json_new(
        summary_path,
        {
            "schema_version": "headkv-v1-b1-offline-stage5-v1",
            "analysis_id": plan["analysis_id"],
            "status": "partial",
            "pending_stages": ["completed_manifest"],
            "bootstrap_replicates": summary.bootstrap_replicates,
            "bootstrap_seed": summary.bootstrap_seed,
            "primary_deltas": list(summary.primary_deltas),
            "record_count": len(summary.records),
            "records": [_estimate_payload(record) for record in summary.records],
        },
    )
    outputs = [
        {"relative_path": path.name, "sha256": sha256_file(path), "file_bytes": path.stat().st_size}
        for path in (profiles_path, summary_path)
    ]
    write_json_new(
        manifest_path,
        {
            "schema_version": "headkv-v1-b1-offline-stage5-manifest-v1",
            "status": "partial",
            "analysis_id": plan["analysis_id"],
            "plan_sha256": sha256_file(plan_path),
            "relationship_sources": dict(source_artifacts),
            "outputs": outputs,
            "pending_stages": ["completed_manifest"],
        },
    )
    return manifest_path


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
    parser.add_argument("--run-stage4", action="store_true")
    parser.add_argument("--run-stage5", action="store_true")
    parser.add_argument("--stage3-dir", type=Path)
    parser.add_argument("--stage4-dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261103)
    parser.add_argument("--token-comparison-chunk-steps", type=int, default=16)
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
        relationship_sources = None
        if args.run_stage5:
            if args.stage3_dir is None or args.stage4_dir is None:
                raise B1OfflineError("--run-stage5 requires --stage3-dir and --stage4-dir")
            relationship_sources = validate_relationship_sources(args.stage3_dir, args.stage4_dir)
            if relationship_sources["source_run_set_sha256"] != validation.source_run_set_sha256:
                raise B1OfflineError("Stage5 source artifacts do not belong to the requested frozen Run Set")
            requested_reports = sorted(item["sha256"] for item in validation.page_rco_reports)
            if relationship_sources["source_page_rco_sha256"] != requested_reports:
                raise B1OfflineError("Stage5 source artifacts do not belong to the requested Page RCO reports")
            if relationship_sources["bootstrap_replicates"] != args.bootstrap_replicates:
                raise B1OfflineError("Stage5 bootstrap_replicates differs from its Stage3/Stage4 sources")
            if relationship_sources["bootstrap_seed"] != args.bootstrap_seed:
                raise B1OfflineError("Stage5 bootstrap_seed differs from its Stage3/Stage4 sources")
        parameters = dict(DEFAULT_ANALYSIS_PARAMETERS)
        parameters["bootstrap_replicates"] = args.bootstrap_replicates
        parameters["bootstrap_seed"] = args.bootstrap_seed
        parameters["token_comparison_chunk_steps"] = args.token_comparison_chunk_steps
        if args.run_stage5:
            parameters["relationship_primary_deltas"] = list(PRIMARY_RELATION_DELTAS)
            parameters["relationship_head_granularity"] = "layer_kv_head"
            parameters["relationship_primary_correlation"] = "spearman"
        plan = build_analysis_plan(
            index,
            validation,
            analysis_id=args.analysis_id,
            git_commit=args.analysis_git_commit,
            parameters=parameters,
        )
        if relationship_sources is not None:
            plan["relationship_sources"] = relationship_sources
            plan["parameters_sha256"] = canonical_json_sha256(plan["parameters"])
        plan_path = prepare_analysis_directory_new(args.output_dir, plan)
        partial_manifest = None
        stage4_manifest = None
        stage5_manifest = None
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
        if args.run_stage4:

            def report_token_progress(done: int, total: int, sample_id: str, seconds: float) -> None:
                print(
                    json.dumps(
                        {
                            "event": "token_sample_completed",
                            "completed": done,
                            "total": total,
                            "sample_id": sample_id,
                            "sample_seconds": round(seconds, 3),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            token = summarize_token_stability(
                index,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed,
                comparison_chunk_steps=args.token_comparison_chunk_steps,
                progress_callback=report_token_progress,
            )
            stage4_manifest = write_stage4_bundle_new(
                args.output_dir,
                plan_path=plan_path,
                token=token,
            )
        if args.run_stage5:
            assert args.stage3_dir is not None and args.stage4_dir is not None and relationship_sources is not None
            relationship_inputs = load_relationship_inputs(
                args.stage3_dir / "stage3_profiles.npz",
                args.stage4_dir / "stage4_token_profiles.npz",
            )

            def report_relationship_progress(done: int, total: int, relation_id: str, seconds: float) -> None:
                print(
                    json.dumps(
                        {
                            "event": "relationship_record_completed",
                            "completed": done,
                            "total": total,
                            "relation_id": relation_id,
                            "record_seconds": round(seconds, 3),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            relationships = analyze_metric_relationships(
                relationship_inputs,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed,
                progress_callback=report_relationship_progress,
            )
            stage5_manifest = write_stage5_bundle_new(
                args.output_dir,
                plan_path=plan_path,
                source_artifacts=relationship_sources,
                summary=relationships,
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
                "stage4_manifest": None if stage4_manifest is None else str(stage4_manifest),
                "stage5_manifest": None if stage5_manifest is None else str(stage5_manifest),
                "completed_manifest_created": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
