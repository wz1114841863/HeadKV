"""Assemble validated Stage3/4/5 artifacts into one completed B1 bundle."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from experiments.head_analysis.storage.b1_reader import B1OfflineError
from experiments.head_analysis.common.io import atomic_file, canonical_json_sha256, read_json, sha256_file
from experiments.head_analysis.common.io import write_json as write_json_new
from experiments.head_analysis.common.io import write_npz as write_npz_new

FINAL_SCHEMA_VERSION = "headkv-v1-b1-offline-completed-v1"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise B1OfflineError(f"unable to read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise B1OfflineError(f"{label} {path} must contain a JSON object")
    return payload


def _validate_stage(root: Path, stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = root / "analysis_plan.json"
    manifest_path = root / f"{stage}_manifest.json"
    plan = _read_json(plan_path, f"{stage} plan")
    manifest = _read_json(manifest_path, f"{stage} manifest")
    if manifest.get("schema_version") != f"headkv-v1-b1-offline-{stage}-manifest-v1":
        raise B1OfflineError(f"unsupported {stage} manifest schema")
    if manifest.get("status") != "partial" or manifest.get("analysis_id") != plan.get("analysis_id"):
        raise B1OfflineError(f"{stage} plan/manifest status or analysis_id is inconsistent")
    if sha256_file(plan_path) != manifest.get("plan_sha256"):
        raise B1OfflineError(f"{stage} plan hash differs from its manifest")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise B1OfflineError(f"{stage} manifest has no registered outputs")
    for entry in outputs:
        if not isinstance(entry, dict) or not isinstance(entry.get("relative_path"), str):
            raise B1OfflineError(f"{stage} manifest contains an invalid output entry")
        path = root / entry["relative_path"]
        if (
            not path.is_file()
            or path.stat().st_size != entry.get("file_bytes")
            or sha256_file(path) != entry.get("sha256")
        ):
            raise B1OfflineError(f"{stage} output differs from its manifest: {path}")
    return plan, manifest


def validate_final_sources(
    stage3_dir: Path | str,
    stage4_dir: Path | str,
    stage5_dir: Path | str,
) -> dict[str, Any]:
    roots = {
        "stage3": Path(stage3_dir).resolve(),
        "stage4": Path(stage4_dir).resolve(),
        "stage5": Path(stage5_dir).resolve(),
    }
    plans: dict[str, dict[str, Any]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for stage, root in roots.items():
        plans[stage], manifests[stage] = _validate_stage(root, stage)
    run_hashes = {plan.get("source_run_set", {}).get("sha256") for plan in plans.values()}
    report_hashes = {
        tuple(sorted(item.get("sha256") for item in plan.get("source_page_rco_reports", []))) for plan in plans.values()
    }
    if len(run_hashes) != 1 or None in run_hashes or len(report_hashes) != 1:
        raise B1OfflineError("Stage3/4/5 do not share exactly one frozen source set")
    for key in ("sample_count", "condition_counts", "source_schema_version"):
        if len({json.dumps(plan.get(key), sort_keys=True) for plan in plans.values()}) != 1:
            raise B1OfflineError(f"Stage3/4/5 disagree on {key}")
    relationship_sources = manifests["stage5"].get("relationship_sources", {})
    for stage, profile_name in (("stage3", "stage3_profiles.npz"), ("stage4", "stage4_token_profiles.npz")):
        expected = relationship_sources.get(stage, {})
        if expected.get("plan_sha256") != sha256_file(roots[stage] / "analysis_plan.json"):
            raise B1OfflineError(f"Stage5 does not bind the accepted {stage} plan")
        if expected.get("manifest_sha256") != sha256_file(roots[stage] / f"{stage}_manifest.json"):
            raise B1OfflineError(f"Stage5 does not bind the accepted {stage} manifest")
        if expected.get("profiles_sha256") != sha256_file(roots[stage] / profile_name):
            raise B1OfflineError(f"Stage5 does not bind the accepted {stage} profiles")
    return {
        "roots": {stage: str(root) for stage, root in roots.items()},
        "source_run_set_sha256": next(iter(run_hashes)),
        "source_page_rco_sha256": list(next(iter(report_hashes))),
        "sample_count": plans["stage5"]["sample_count"],
        "condition_counts": plans["stage5"]["condition_counts"],
        "parameters": plans["stage5"]["parameters"],
        "stages": {
            stage: {
                "analysis_id": plans[stage]["analysis_id"],
                "git_commit": plans[stage]["git_commit"],
                "plan_sha256": sha256_file(roots[stage] / "analysis_plan.json"),
                "manifest_sha256": sha256_file(roots[stage] / f"{stage}_manifest.json"),
            }
            for stage in roots
        },
    }


def _write_text_new(path: Path, writer: Callable[[Any], None]) -> None:
    try:
        with atomic_file(path) as handle:
            writer(handle)
    except FileExistsError as exc:
        raise B1OfflineError(f"final output already exists: {path}") from exc


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    def write(handle: Any) -> None:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _write_text_new(path, write)


def _json_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _merge_head_profiles(stage3_path: Path, stage4_path: Path, destination: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(stage3_path, allow_pickle=False) as stage3:
            arrays.update({name: stage3[name].copy() for name in stage3.files})
        with np.load(stage4_path, allow_pickle=False) as stage4:
            for name in stage4.files:
                value = stage4[name]
                if name in arrays:
                    if np.issubdtype(arrays[name].dtype, np.floating):
                        equal = np.array_equal(arrays[name], value, equal_nan=True)
                    else:
                        equal = np.array_equal(arrays[name], value)
                    if not equal:
                        raise B1OfflineError(f"Stage3/Stage4 duplicate axis {name!r} differs")
                    continue
                arrays[name] = value.copy()
    except (OSError, ValueError) as exc:
        raise B1OfflineError(f"unable to merge Stage3/Stage4 profiles: {exc}") from exc
    write_npz_new(destination, arrays)


def _write_numeric_bundle(root: Path, sources: Mapping[str, Any]) -> None:
    roots = {stage: Path(value) for stage, value in sources["roots"].items()}
    stage3 = _read_json(roots["stage3"] / "stage3_summary.json", "Stage3 summary")
    stage4 = _read_json(roots["stage4"] / "stage4_token_summary.json", "Stage4 summary")
    stage5 = _read_json(roots["stage5"] / "stage5_relationships.json", "Stage5 summary")
    write_json_new(
        root / "condition_summary.json",
        {
            "schema_version": "headkv-v1-b1-condition-summary-v1",
            "analysis_id": root.name,
            "source_analysis_id": stage3["analysis_id"],
            "condition_counts": sources["condition_counts"],
            "axes": {
                key: stage3["axes"][key]
                for key in ("conditions", "query_phases", "query_features", "page_scopes", "page_metrics")
            },
            "query_condition_head_profile_mean": stage3["query_condition_head_profile_mean"],
            "page_condition_head_profile_mean": stage3["page_condition_head_profile_mean"],
        },
    )
    write_json_new(
        root / "page_rco_summary.json",
        {
            "schema_version": "headkv-v1-b1-page-rco-summary-v1",
            "analysis_id": root.name,
            "source_analysis_id": stage3["analysis_id"],
            "axes": {
                "conditions": stage3["axes"]["conditions"],
                "page_scopes": stage3["axes"]["page_scopes"],
                "page_rco_k": stage3["axes"]["page_rco_k"],
                "deltas": stage3["axes"]["deltas"],
                "page_rco_metrics": stage3["axes"]["page_rco_metrics"],
                "ci_bounds": stage3["axes"]["ci_bounds"],
            },
            **stage3["page_rco"],
        },
    )
    write_json_new(
        root / "token_stability_summary.json",
        {
            **stage4,
            "analysis_id": root.name,
            "source_analysis_id": stage4["analysis_id"],
            "status": "completed_numeric",
            "pending_stages": [],
        },
    )
    write_json_new(
        root / "correlations.json",
        {
            **stage5,
            "analysis_id": root.name,
            "source_analysis_id": stage5["analysis_id"],
            "status": "completed_numeric",
            "pending_stages": [],
        },
    )
    _merge_head_profiles(
        roots["stage3"] / "stage3_profiles.npz",
        roots["stage4"] / "stage4_token_profiles.npz",
        root / "head_profiles.npz",
    )
    shutil.copyfile(roots["stage5"] / "stage5_relationship_profiles.npz", root / "correlation_profiles.npz")


def _write_tables(root: Path) -> None:
    page = _read_json(root / "page_rco_summary.json", "Page RCO summary")
    axes = page["axes"]
    means = _json_array(page["condition_scalar_sample_equal_mean"])
    valid = np.asarray(page["condition_scalar_valid_samples"], dtype=np.uint32)
    ci = _json_array(page["condition_scalar_bootstrap_ci"])
    rows: list[dict[str, Any]] = []
    for c, condition in enumerate(axes["conditions"]):
        for s, scope in enumerate(axes["page_scopes"]):
            for k, topk in enumerate(axes["page_rco_k"]):
                for d, delta in enumerate(axes["deltas"]):
                    for m, metric in enumerate(axes["page_rco_metrics"]):
                        rows.append(
                            {
                                "condition": condition,
                                "scope": scope,
                                "k": topk,
                                "delta": delta,
                                "metric": metric,
                                "sample_equal_mean": means[c, s, k, d, m],
                                "valid_samples": valid[c, s, k, d, m],
                                "ci_2_5": ci[c, s, k, d, m, 0],
                                "ci_97_5": ci[c, s, k, d, m, 1],
                            }
                        )
    _write_csv(root / "tables/page_rco.csv", tuple(rows[0]), rows)

    token = _read_json(root / "token_stability_summary.json", "Token summary")
    axes = token["axes"]
    means = _json_array(token["condition_scalar_sample_equal_mean"])
    valid = np.asarray(token["condition_scalar_valid_samples"], dtype=np.uint32)
    ci = _json_array(token["condition_scalar_bootstrap_ci"])
    rows = []
    for c, condition in enumerate(axes["conditions"]):
        for k, topk in enumerate(axes["token_k"]):
            for d, delta in enumerate(axes["deltas"]):
                for m, metric in enumerate(axes["token_metrics"]):
                    rows.append(
                        {
                            "condition": condition,
                            "k": topk,
                            "delta": delta,
                            "metric": metric,
                            "sample_equal_mean": means[c, k, d, m],
                            "valid_samples": valid[c, k, d, m],
                            "ci_2_5": ci[c, k, d, m, 0],
                            "ci_97_5": ci[c, k, d, m, 1],
                        }
                    )
    _write_csv(root / "tables/token_stability.csv", tuple(rows[0]), rows)

    correlations = _read_json(root / "correlations.json", "Correlations")
    rows = []
    for record in correlations["records"]:
        rows.append(
            {
                key: record[key]
                for key in (
                    "relation_id",
                    "family",
                    "condition_x",
                    "condition_y",
                    "profile_x",
                    "profile_y",
                    "bootstrap_mode",
                    "valid_head_count",
                    "spearman",
                    "pearson",
                    "layer_centered_spearman",
                    "layer_centered_pearson",
                )
            }
            | {
                "spearman_ci_2_5": record["spearman_bootstrap_ci"][0],
                "spearman_ci_97_5": record["spearman_bootstrap_ci"][1],
                "layer_centered_spearman_ci_2_5": record["layer_centered_spearman_bootstrap_ci"][0],
                "layer_centered_spearman_ci_97_5": record["layer_centered_spearman_bootstrap_ci"][1],
            }
        )
    _write_csv(root / "tables/relationships.csv", tuple(rows[0]), rows)

    condition = _read_json(root / "condition_summary.json", "Condition summary")
    rows = []
    query = _json_array(condition["query_condition_head_profile_mean"])
    for c, name in enumerate(condition["axes"]["conditions"]):
        for p, phase in enumerate(condition["axes"]["query_phases"]):
            for f, feature in enumerate(condition["axes"]["query_features"]):
                rows.append({"condition": name, "phase": phase, "feature": feature, "head_mean": query[c, p, f]})
    _write_csv(root / "tables/query_condition_metrics.csv", tuple(rows[0]), rows)


def _condition_label(value: str) -> str:
    return value.replace("govreport", "GovReport").replace("pg19", "PG19").replace("ruler", "RULER")


def _save_figure(path: Path, figure: Any) -> None:
    if path.exists():
        raise B1OfflineError(f"analysis figure already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")


def _set_decode_axis(panel: Any, deltas: list[int] | tuple[int, ...]) -> None:
    positive = [int(delta) for delta in deltas if int(delta) > 0]
    if not positive:
        raise B1OfflineError("figure decode axis has no positive delta")
    panel.set_xscale("log", base=2)
    if len(positive) == 1:
        panel.set_xlim(positive[0] / 2, positive[0] * 2)
    else:
        panel.set_xlim(min(positive), max(positive))


def _write_figures(root: Path) -> list[dict[str, Any]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise B1OfflineError("Matplotlib is required for the explicit analysis figure stage") from exc
    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.25})
    figure_specs: list[dict[str, Any]] = []

    page = _read_json(root / "page_rco_summary.json", "Page RCO summary")
    axes = page["axes"]
    values = _json_array(page["condition_scalar_sample_equal_mean"])
    metric = axes["page_rco_metrics"].index("rco_raw_mean")
    topk = axes["page_rco_k"].index(32)
    fig, panels = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for scope_index, scope in enumerate(axes["page_scopes"]):
        for condition_index, condition in enumerate(axes["conditions"]):
            panels[scope_index].plot(
                axes["deltas"],
                values[condition_index, scope_index, topk, :, metric],
                marker="o",
                ms=3,
                label=_condition_label(condition),
            )
        _set_decode_axis(panels[scope_index], axes["deltas"])
        panels[scope_index].set_title(f"Page RCO k=32: {scope}")
        panels[scope_index].set_xlabel("Decode delta")
        panels[scope_index].set_ylim(-0.02, 1.0)
    panels[0].set_ylabel("Chance-corrected RCO")
    panels[1].legend(fontsize=7, ncol=2)
    path = root / "figures/page_rco_decay_k32.png"
    _save_figure(path, fig)
    plt.close(fig)
    figure_specs.append({"relative_path": path.relative_to(root).as_posix(), "source": "page_rco_summary.json"})

    token = _read_json(root / "token_stability_summary.json", "Token summary")
    axes = token["axes"]
    values = _json_array(token["condition_scalar_sample_equal_mean"])
    metric = axes["token_metrics"].index("rco_raw_mean")
    topk = axes["token_k"].index(32)
    fig, panel = plt.subplots(figsize=(6.5, 4))
    for condition_index, condition in enumerate(axes["conditions"]):
        panel.plot(
            axes["deltas"],
            values[condition_index, topk, :, metric],
            marker="o",
            ms=3,
            label=_condition_label(condition),
        )
    _set_decode_axis(panel, axes["deltas"])
    panel.set_ylim(-0.02, 1.0)
    panel.set_xlabel("Decode delta")
    panel.set_ylabel("Chance-corrected Token RCO")
    panel.set_title("Token Top-K stability, k=32")
    panel.legend(fontsize=7, ncol=2)
    path = root / "figures/token_rco_decay_k32.png"
    _save_figure(path, fig)
    plt.close(fig)
    figure_specs.append({"relative_path": path.relative_to(root).as_posix(), "source": "token_stability_summary.json"})

    correlations = _read_json(root / "correlations.json", "Correlations")
    records = {record["relation_id"]: record for record in correlations["records"]}
    conditions = page["axes"]["conditions"]
    fig, panels = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    profiles = (
        ("page_rco:all_visible:k32:d1", "Page all-visible, d=1"),
        ("page_rco:all_visible:k32:d256", "Page all-visible, d=256"),
        ("token_rco:k32:d1", "Token, d=1"),
        ("token_rco:k32:d256", "Token, d=256"),
    )
    for panel, (profile, title) in zip(panels.ravel(), profiles):
        matrix = np.full((len(conditions), len(conditions)), np.nan)
        for i, left in enumerate(conditions):
            for j in range(i + 1, len(conditions)):
                right = conditions[j]
                relation_id = f"cross:{profile}:{left}__{right}"
                if relation_id in records and records[relation_id]["valid_head_count"] > 0:
                    matrix[i, j] = matrix[j, i] = records[relation_id]["spearman"]
        for index in range(len(conditions)):
            if np.isfinite(matrix[index]).any():
                matrix[index, index] = 1.0
        image = panel.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        panel.set_title(title)
        panel.set_xticks(range(len(conditions)), [_condition_label(x) for x in conditions], rotation=45, ha="right")
        panel.set_yticks(range(len(conditions)), [_condition_label(x) for x in conditions])
        for i in range(len(conditions)):
            for j in range(len(conditions)):
                if np.isfinite(matrix[i, j]):
                    panel.text(
                        j,
                        i,
                        f"{matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if matrix[i, j] < 0.65 else "black",
                    )
    for panel in panels[0]:
        panel.tick_params(labelbottom=False)
    for panel in panels[:, 1]:
        panel.tick_params(labelleft=False)
    fig.colorbar(image, ax=panels.ravel().tolist(), shrink=0.75, label="Spearman")
    path = root / "figures/cross_condition_head_identity.png"
    _save_figure(path, fig)
    plt.close(fig)
    figure_specs.append({"relative_path": path.relative_to(root).as_posix(), "source": "correlations.json"})

    fig, panels = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for condition in conditions:
        token_page = []
        short_long = []
        for delta in (1, 32, 128, 256):
            relation_id = f"within:{condition}:token_rco:k32:d{delta}__page_rco:all_visible:k32:d{delta}"
            token_page.append(records[relation_id]["spearman"])
        for delta in (32, 128, 256):
            relation_id = f"within:{condition}:page_rco:all_visible:k32:d1__page_rco:all_visible:k32:d{delta}"
            short_long.append(records[relation_id]["spearman"])
        panels[0].plot((1, 32, 128, 256), token_page, marker="o", label=_condition_label(condition))
        if np.isfinite(np.asarray(short_long, dtype=np.float64)).any():
            panels[1].plot((32, 128, 256), short_long, marker="o", label=_condition_label(condition))
    for panel, deltas in zip(panels, ((1, 32, 128, 256), (32, 128, 256))):
        _set_decode_axis(panel, deltas)
        panel.set_ylim(0, 1)
        panel.set_xlabel("Decode delta")
    panels[0].set_title("Token RCO vs Page RCO")
    panels[0].set_ylabel("Head-profile Spearman")
    panels[1].set_title("Page d=1 vs longer horizon")
    handles, labels = panels[1].get_legend_handles_labels()
    if handles:
        panels[1].legend(handles, labels, fontsize=7, ncol=2)
    path = root / "figures/stability_relationships.png"
    _save_figure(path, fig)
    plt.close(fig)
    figure_specs.append({"relative_path": path.relative_to(root).as_posix(), "source": "correlations.json"})

    write_json_new(
        root / "figure_manifest.json",
        {
            "schema_version": "headkv-v1-b1-figure-manifest-v1",
            "analysis_id": root.name,
            "figures": [
                {
                    **spec,
                    "sha256": sha256_file(root / spec["relative_path"]),
                    "file_bytes": (root / spec["relative_path"]).stat().st_size,
                }
                for spec in figure_specs
            ],
        },
    )
    return figure_specs


def _output_entries(root: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "analysis_manifest.json"):
        entries.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "file_bytes": path.stat().st_size,
            }
        )
    return entries


def finalize_bundle_new(
    *,
    stage3_dir: Path | str,
    stage4_dir: Path | str,
    stage5_dir: Path | str,
    output_dir: Path | str,
    analysis_id: str,
    git_commit: str,
) -> Path:
    sources = validate_final_sources(stage3_dir, stage4_dir, stage5_dir)
    root = Path(output_dir)
    if root.exists():
        raise B1OfflineError(f"final analysis directory already exists: {root}")
    if root.name != analysis_id:
        raise B1OfflineError("final output directory name must equal analysis_id")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    plan = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "git_commit": git_commit,
        "source_run_set_sha256": sources["source_run_set_sha256"],
        "source_page_rco_sha256": sources["source_page_rco_sha256"],
        "sample_count": sources["sample_count"],
        "condition_counts": sources["condition_counts"],
        "parameters": sources["parameters"],
        "parameters_sha256": canonical_json_sha256(sources["parameters"]),
        "source_stages": sources["stages"],
        "output_contract": {
            "numeric": [
                "condition_summary.json",
                "head_profiles.npz",
                "page_rco_summary.json",
                "token_stability_summary.json",
                "correlations.json",
                "correlation_profiles.npz",
            ],
            "tables": "tables/*.csv",
            "figures": "figures/*.png",
        },
    }
    write_json_new(root / "analysis_plan.json", plan)
    _write_numeric_bundle(root, sources)
    _write_tables(root)
    _write_figures(root)
    manifest_path = root / "analysis_manifest.json"
    write_json_new(
        manifest_path,
        {
            "schema_version": FINAL_SCHEMA_VERSION,
            "analysis_id": analysis_id,
            "status": "completed",
            "git_commit": git_commit,
            "plan_sha256": sha256_file(root / "analysis_plan.json"),
            "source_run_set_sha256": sources["source_run_set_sha256"],
            "sample_count": sources["sample_count"],
            "condition_counts": sources["condition_counts"],
            "outputs": _output_entries(root),
            "warnings": [
                "RULER has only six Decode observations; primary relations beyond delta=1 are not applicable.",
                "GovReport 8K/16K samples are not content-paired; their difference is descriptive, not causal.",
                "B1 does not contain CateKV Prefill CV or Prefill-to-Decode Recall and proves no system benefit.",
            ],
        },
    )
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize validated B1 Stage3/4/5 numeric artifacts and figures.")
    parser.add_argument("--stage3-dir", type=Path, required=True)
    parser.add_argument("--stage4-dir", type=Path, required=True)
    parser.add_argument("--stage5-dir", type=Path, required=True)
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--analysis-git-commit", help="optional source revision metadata")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = finalize_bundle_new(
            stage3_dir=args.stage3_dir,
            stage4_dir=args.stage4_dir,
            stage5_dir=args.stage5_dir,
            output_dir=args.output_dir,
            analysis_id=args.analysis_id,
            git_commit=args.analysis_git_commit,
        )
    except B1OfflineError as exc:
        raise SystemExit(f"B1 finalization failed: {exc}") from exc
    print(json.dumps({"status": "completed", "analysis_id": args.analysis_id, "manifest": str(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
