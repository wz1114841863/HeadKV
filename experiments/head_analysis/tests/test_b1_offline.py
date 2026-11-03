import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.head_analysis.analysis.b1_offline import (
    B1OfflineError,
    aggregate_page_rco,
    build_analysis_manifest,
    build_analysis_plan,
    decode_segment_ids,
    load_completed_run_set,
    summarize_query_and_page_metrics,
    summarize_sample_equal,
    validate_analysis_sources,
)
from experiments.head_analysis.scripts import analyze_b1_run_set
from experiments.head_analysis.storage.trace_writer import AttentionTraceSchema


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar_features(visible: np.ndarray, normalized_entropy: np.ndarray) -> np.ndarray:
    values = np.zeros((len(visible), 1, 4, 28), dtype=np.float32)
    normalized = np.asarray(normalized_entropy, dtype=np.float32)
    values[..., 0] = (normalized * np.log(visible.astype(np.float32)))[:, None, None]
    values[..., 1] = normalized[:, None, None]
    values[..., 2:6] = 1.0
    values[..., 6] = 1.0
    values[..., 7] = 1.0
    values[..., 8] = 1.0
    values[..., 9] = 0.0
    values[..., 10] = 0.0
    values[..., 11] = 1.0
    return values


def write_trace(
    run_dir: Path,
    records: list[dict],
    *,
    data_sha256: str,
    commit: str,
    decode_length: int = 2,
) -> None:
    schema = AttentionTraceSchema(
        num_layers=1,
        num_query_heads=4,
        num_kv_heads=1,
        prefill_window=64,
        token_topk=64,
        page_topk=32,
    ).to_dict()
    write_json(run_dir / "schema.json", schema)
    write_json(run_dir / "config_snapshot.yaml", {"collection": {"page_topk": 32}})
    write_json(
        run_dir / "run_metadata.json",
        {
            "status": "completed",
            "schema_version": schema["schema_version"],
            "git": {"commit": commit},
            "fixed_dataset": {"data_sha256": data_sha256},
            "dtype": "bfloat16",
            "attention_backend": "sdpa",
            "generation": {"do_sample": False, "num_beams": 1, "use_cache": True},
        },
    )
    for record in records:
        sample_id = record["sample_id"]
        sample_dir = run_dir / "samples" / sample_id
        sample_dir.mkdir(parents=True)
        base = 0.2 if record["target_prompt_length"] == 8192 else 0.6
        generated = np.arange(101, 102 + decode_length, dtype="<u4")
        write_json(
            sample_dir / "generated_ids.json",
            {"ids": generated.tolist(), "sha256": hashlib.sha256(generated.tobytes()).hexdigest()},
        )
        write_json(
            sample_dir / "sample_metadata.json",
            {
                "sample_id": sample_id,
                "dataset": record["dataset"],
                "task": record["task"],
                "pair_id": record["pair_id"],
                "input_sha256": record["input_sha256"],
                "prompt_length": record["actual_prompt_length"],
                "status": "completed",
                "actual_generated_length": decode_length + 1,
                "decode_trace_length": decode_length,
            },
        )
        prefill_visible = np.asarray([record["actual_prompt_length"]], dtype=np.uint32)
        np.savez_compressed(
            sample_dir / "prefill_stats.npz",
            query_positions=np.asarray([record["actual_prompt_length"] - 1], dtype=np.uint32),
            visible_token_count=prefill_visible,
            scalar_features=scalar_features(prefill_visible, np.asarray([base])),
        )
        decode_steps = np.arange(decode_length, dtype=np.uint32)
        decode_visible = record["actual_prompt_length"] + 1 + decode_steps
        decode_normalized = (
            np.asarray([base, base + 0.2]) if decode_length == 2 else base + 0.01 * np.arange(decode_length)
        )
        coverage_value = 0.25 if record["target_prompt_length"] == 8192 else 0.75
        page_scope_mass = np.ones((decode_length, 1, 1, 2), dtype=np.float32)
        page_scope_mass[..., 1] = 0.6
        np.savez_compressed(
            sample_dir / "decode_stats_000.npz",
            decode_steps=decode_steps,
            query_positions=decode_visible - 1,
            visible_token_count=decode_visible,
            scalar_features=scalar_features(decode_visible, decode_normalized),
            token_topk_positions=np.tile(np.arange(64, dtype=np.uint32), (decode_length, 1, 4, 1)),
            page_topk_ids_all_visible=np.zeros((decode_length, 1, 1, 32), dtype=np.uint32),
            page_topk_ids_prompt_only=np.zeros((decode_length, 1, 1, 32), dtype=np.uint32),
            page_candidate_count_all_visible=np.full(decode_length, 520, dtype=np.uint32),
            page_candidate_count_prompt_only=np.full(decode_length, 512, dtype=np.uint32),
            page_scope_mass=page_scope_mass,
            page_top32_coverage=np.full((decode_length, 1, 1, 2), coverage_value, dtype=np.float32),
            page_scope_valid=np.ones((decode_length, 1, 1, 2), dtype=bool),
        )

    entries = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(run_dir).as_posix()
        entry = {
            "relative_path": relative,
            "sha256": sha256(path),
            "file_bytes": path.stat().st_size,
            "logical_array_bytes": 0,
        }
        if relative.endswith("decode_stats_000.npz"):
            entry.update({"step_start": 0, "step_end": decode_length - 1})
        entries.append(entry)
    write_json(run_dir / "file_manifest.json", {"files": entries})


def make_fixture(tmp_path: Path, *, decode_length: int = 2) -> tuple[Path, Path, Path]:
    data_sha256 = "frozen-data"
    commit = "a" * 40
    records = [
        {
            "sample_id": "pg19_001_8k",
            "dataset": "pg19",
            "task": "continuation",
            "pair_id": "pg19_001",
            "target_prompt_length": 8192,
            "actual_prompt_length": 8190,
            "input_sha256": "input-8k",
        },
        {
            "sample_id": "pg19_001_32k",
            "dataset": "pg19",
            "task": "continuation",
            "pair_id": "pg19_001",
            "target_prompt_length": 32768,
            "actual_prompt_length": 32766,
            "input_sha256": "input-32k",
        },
    ]
    prompts = tmp_path / "data" / "fixed" / "prompts.jsonl"
    prompts.parent.mkdir(parents=True)
    prompts.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    fixed_manifest = prompts.parent / "manifest.json"
    write_json(fixed_manifest, {"data_sha256": data_sha256})
    run_dir = tmp_path / "data" / "runs" / "batch-1"
    write_trace(run_dir, records, data_sha256=data_sha256, commit=commit, decode_length=decode_length)
    run_set = tmp_path / "data" / "runs" / "set.manifest.json"
    write_json(
        run_set,
        {
            "schema_version": "headkv-v1-b1-run-set-v1",
            "status": "completed",
            "fixed_dataset": {
                "prompts_path": str(prompts.relative_to(tmp_path)),
                "manifest_path": str(fixed_manifest.relative_to(tmp_path)),
                "data_sha256": data_sha256,
            },
            "execution_contract": {
                "git_commit": commit,
                "dtype": "bfloat16",
                "attention_backend": "sdpa",
            },
            "batches": [
                {
                    "batch_id": "pg19_b01",
                    "sample_ids": [record["sample_id"] for record in records],
                    "run_dir": str(run_dir.relative_to(tmp_path)),
                    "status": "completed",
                    "run_metadata_sha256": sha256(run_dir / "run_metadata.json"),
                    "file_manifest_sha256": sha256(run_dir / "file_manifest.json"),
                    "validation_reports": [],
                }
            ],
        },
    )
    report = tmp_path / "data" / "runs" / "batch-1.page-rco.json"
    rco_records = []
    for record in records:
        rco_value = 0.2 if record["target_prompt_length"] == 8192 else 0.6
        for scope in ("all_visible", "prompt_only"):
            for k in (8, 16, 32):
                for delta in (1, 2, 4, 8, 16, 32, 64, 128, 256):
                    valid_pair_count = max(0, decode_length - delta)
                    rco_records.append(
                        {
                            "sample_id": record["sample_id"],
                            "scope": scope,
                            "k": k,
                            "delta": delta,
                            "layer_id": 0,
                            "kv_head": 0,
                            "valid_pair_count": valid_pair_count,
                            "overlap_raw_mean": None if not valid_pair_count else 0.5,
                            "random_baseline_mean": None if not valid_pair_count else 0.1,
                            "rco_raw_mean": None if not valid_pair_count else rco_value,
                            "rco_clipped_mean": None if not valid_pair_count else rco_value,
                        }
                    )
    write_json(
        report,
        {
            "schema_version": "headkv-v1-b1-page-rco-v1",
            "source_run": str(run_dir),
            "source_run_metadata_sha256": sha256(run_dir / "run_metadata.json"),
            "k_values": [8, 16, 32],
            "deltas": [1, 2, 4, 8, 16, 32, 64, 128, 256],
            "records": rco_records,
        },
    )
    return run_set, report, run_dir


def test_completed_run_set_and_page_rco_sources_build_a_frozen_plan(tmp_path):
    run_set, report, _ = make_fixture(tmp_path)

    index = load_completed_run_set(run_set, source_root=tmp_path, expected_batch_count=1, expected_sample_count=2)
    validation = validate_analysis_sources(index, [report])
    plan = build_analysis_plan(index, validation, analysis_id="analysis-v1", git_commit="b" * 40)

    assert index.condition_counts == {"pg19:32k": 1, "pg19:8k": 1}
    assert validation.sample_count == 2
    assert plan["parameters"]["primary_weighting"] == "sample_equal"
    assert len(plan["parameters_sha256"]) == 64
    assert plan["source_page_rco_reports"][0]["record_count"] == 108


def test_source_validation_fails_closed_on_registered_file_corruption(tmp_path):
    run_set, _, run_dir = make_fixture(tmp_path)
    with (run_dir / "samples" / "pg19_001_8k" / "generated_ids.json").open("a", encoding="utf-8") as handle:
        handle.write("corruption")

    with pytest.raises(B1OfflineError, match="file size differs from manifest"):
        load_completed_run_set(run_set, source_root=tmp_path)


def test_source_validation_rejects_incomplete_run_set_sample_membership(tmp_path):
    run_set, _, _ = make_fixture(tmp_path)
    payload = json.loads(run_set.read_text())
    payload["batches"][0]["sample_ids"] = ["pg19_001_8k"]
    write_json(run_set, payload)

    with pytest.raises(B1OfflineError, match="sample directories differ|exactly cover the frozen dataset"):
        load_completed_run_set(run_set, source_root=tmp_path)


def test_sample_equal_primary_is_not_replaced_by_step_weighted_secondary():
    records = [
        {"condition": "x", "sample_id": "short", "value": 0.0, "pairs": 1},
        *[{"condition": "x", "sample_id": "long", "value": 1.0, "pairs": 1} for _ in range(9)],
        {"condition": "x", "sample_id": "missing", "value": None, "pairs": 0},
    ]

    result = summarize_sample_equal(
        records,
        group_keys=("condition",),
        value_key="value",
        secondary_weight_key="pairs",
    )[0]

    assert result["sample_count_total"] == 3
    assert result["sample_count_valid"] == 2
    assert result["primary_sample_equal"]["mean"] == 0.5
    assert result["secondary_observation_weighted"]["mean"] == 0.9


def test_query_and_page_summaries_keep_sample_equal_conditions_and_short_trace_empty(tmp_path):
    run_set, _, _ = make_fixture(tmp_path)
    index = load_completed_run_set(run_set, source_root=tmp_path)

    query, page = summarize_query_and_page_metrics(index)

    entropy_normalized = query.feature_names.index("entropy_normalized")
    decode_phase = query.phases.index("decode")
    mean_stat = query.statistics.index("mean")
    condition_8k = query.condition_names.index("pg19:8k")
    condition_32k = query.condition_names.index("pg19:32k")
    assert query.sample_full.shape == (2, 2, 1, 4, 28, 4)
    np.testing.assert_allclose(query.condition_full_mean[condition_8k, decode_phase, 0, :, entropy_normalized], 0.3)
    np.testing.assert_allclose(query.condition_full_mean[condition_32k, decode_phase, 0, :, entropy_normalized], 0.7)
    assert query.sample_full_valid_count[:, decode_phase, 0, 0, entropy_normalized].tolist() == [2, 2]
    assert np.isnan(query.sample_decode_segments).all()
    assert not query.sample_decode_segment_valid_count.any()

    coverage = page.metrics.index("top32_coverage")
    prompt_only = page.scopes.index("prompt_only")
    valid_rate = page.metrics.index("scope_valid_rate")
    np.testing.assert_allclose(page.condition_full_mean[condition_8k, 0, 0, prompt_only, coverage], 0.25)
    np.testing.assert_allclose(page.condition_full_mean[condition_32k, 0, 0, prompt_only, coverage], 0.75)
    np.testing.assert_allclose(page.sample_full[:, 0, 0, prompt_only, valid_rate, mean_stat], 1.0)
    assert np.isnan(page.sample_decode_segments).all()


def test_decode_segments_require_32_steps_and_follow_frozen_formula():
    assert decode_segment_ids(6).size == 0
    assert np.bincount(decode_segment_ids(32), minlength=4).tolist() == [8, 8, 8, 8]
    assert decode_segment_ids(33).tolist() == [min(3, (4 * step) // 33) for step in range(33)]


def test_long_decode_summary_uses_four_sample_local_segments(tmp_path):
    run_set, _, _ = make_fixture(tmp_path, decode_length=32)
    index = load_completed_run_set(run_set, source_root=tmp_path)

    query, page = summarize_query_and_page_metrics(index)

    feature = query.feature_names.index("entropy_normalized")
    mean_stat = query.statistics.index("mean")
    np.testing.assert_allclose(
        query.sample_decode_segments[0, :, 0, 0, feature, mean_stat], [0.235, 0.315, 0.395, 0.475]
    )
    assert query.sample_decode_segment_valid_count[0, :, 0, 0, feature].tolist() == [8, 8, 8, 8]
    coverage = page.metrics.index("top32_coverage")
    np.testing.assert_allclose(page.sample_decode_segments[0, :, 0, 0, 0, coverage, mean_stat], 0.25)
    assert page.sample_decode_segment_valid_count[0, :, 0, 0, 0, coverage].tolist() == [8, 8, 8, 8]


def test_metric_summary_rejects_feature_invariant_failure(tmp_path):
    run_set, _, run_dir = make_fixture(tmp_path)
    index = load_completed_run_set(run_set, source_root=tmp_path)
    path = run_dir / "samples" / "pg19_001_8k" / "prefill_stats.npz"
    with np.load(path) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["scalar_features"][..., 9] = 0.5
    np.savez_compressed(path, **arrays)

    with pytest.raises(B1OfflineError, match=r"prompt_mass \+ generated_mass"):
        summarize_query_and_page_metrics(index)


def test_page_rco_aggregation_is_sample_equal_bootstrapped_and_pg19_paired(tmp_path):
    run_set, report, _ = make_fixture(tmp_path)
    index = load_completed_run_set(run_set, source_root=tmp_path)

    summary = aggregate_page_rco(index, [report], bootstrap_replicates=100, bootstrap_seed=17)
    repeated = aggregate_page_rco(index, [report], bootstrap_replicates=100, bootstrap_seed=17)

    clipped = summary.metrics.index("rco_clipped_mean")
    delta_1 = summary.deltas.index(1)
    delta_2 = summary.deltas.index(2)
    k_32 = summary.k_values.index(32)
    all_visible = summary.scopes.index("all_visible")
    condition_8k = summary.condition_names.index("pg19:8k")
    condition_32k = summary.condition_names.index("pg19:32k")
    coordinate_8k = (condition_8k, all_visible, k_32, delta_1, clipped)
    coordinate_32k = (condition_32k, all_visible, k_32, delta_1, clipped)
    assert summary.condition_scalar_sample_equal_mean[coordinate_8k] == pytest.approx(0.2)
    assert summary.condition_scalar_pair_weighted_mean[coordinate_8k] == pytest.approx(0.2)
    assert summary.condition_scalar_sample_equal_mean[coordinate_32k] == pytest.approx(0.6)
    np.testing.assert_allclose(summary.condition_scalar_bootstrap_ci[coordinate_8k], [0.2, 0.2])
    assert np.isnan(summary.condition_scalar_sample_equal_mean[condition_8k, all_visible, k_32, delta_2, clipped])
    assert summary.condition_scalar_valid_pair_count[condition_8k, all_visible, k_32, delta_2] == 0
    assert summary.pg19_pair_ids == ("pg19_001",)
    assert summary.pg19_pair_scalar_mean_difference[all_visible, k_32, delta_1, clipped] == pytest.approx(0.4)
    np.testing.assert_array_equal(
        summary.condition_scalar_bootstrap_ci,
        repeated.condition_scalar_bootstrap_ci,
    )


def test_completed_manifest_hashes_outputs_but_not_itself(tmp_path):
    run_set, report, _ = make_fixture(tmp_path)
    index = load_completed_run_set(run_set, source_root=tmp_path)
    validation = validate_analysis_sources(index, [report])
    plan = build_analysis_plan(index, validation, analysis_id="analysis-v1", git_commit="b" * 40)
    output_root = tmp_path / "analysis-v1"
    output_root.mkdir()
    plan_path = output_root / "analysis_plan.json"
    write_json(plan_path, plan)
    output = output_root / "condition_summary.json"
    write_json(output, {"ok": True})

    manifest = build_analysis_manifest(
        index=index,
        plan_path=plan_path,
        output_root=output_root,
        output_paths=[output],
    )

    assert manifest["status"] == "completed"
    assert manifest["outputs"] == [
        {
            "relative_path": "condition_summary.json",
            "sha256": sha256(output),
            "file_bytes": output.stat().st_size,
        }
    ]


def test_prepare_cli_is_end_to_end_and_non_overwriting(tmp_path, monkeypatch):
    run_set, report, _ = make_fixture(tmp_path)
    output = tmp_path / "analysis-output"
    arguments = [
        "--run-set",
        str(run_set),
        "--source-root",
        str(tmp_path),
        "--page-rco-report",
        str(report),
        "--analysis-id",
        "analysis-v1",
        "--analysis-git-commit",
        "b" * 40,
        "--output-dir",
        str(output),
        "--expected-batches",
        "1",
        "--expected-samples",
        "2",
    ]

    monkeypatch.setattr(analyze_b1_run_set, "validate_analysis_git_state", lambda _: None)
    assert analyze_b1_run_set.main(arguments) == 0
    assert (output / "analysis_plan.json").is_file()
    assert not (output / "analysis_manifest.json").exists()
    with pytest.raises(SystemExit, match="already exists"):
        analyze_b1_run_set.main(arguments)


def test_stage3_cli_writes_partial_bundle_without_completed_manifest(tmp_path, monkeypatch):
    run_set, report, _ = make_fixture(tmp_path)
    output = tmp_path / "analysis-stage3"
    monkeypatch.setattr(analyze_b1_run_set, "validate_analysis_git_state", lambda _: None)

    assert (
        analyze_b1_run_set.main(
            [
                "--run-set",
                str(run_set),
                "--source-root",
                str(tmp_path),
                "--page-rco-report",
                str(report),
                "--analysis-id",
                "analysis-stage3",
                "--analysis-git-commit",
                "b" * 40,
                "--output-dir",
                str(output),
                "--expected-batches",
                "1",
                "--expected-samples",
                "2",
                "--run-stage3",
                "--bootstrap-replicates",
                "20",
                "--bootstrap-seed",
                "17",
            ]
        )
        == 0
    )

    partial = json.loads((output / "stage3_manifest.json").read_text())
    summary = json.loads((output / "stage3_summary.json").read_text())
    assert partial["status"] == "partial"
    assert summary["page_rco"]["bootstrap_replicates"] == 20
    assert (output / "stage3_profiles.npz").is_file()
    assert not (output / "analysis_manifest.json").exists()


def test_stage4_cli_writes_token_partial_bundle_without_completed_manifest(tmp_path, monkeypatch):
    run_set, report, _ = make_fixture(tmp_path)
    output = tmp_path / "analysis-stage4"
    monkeypatch.setattr(analyze_b1_run_set, "validate_analysis_git_state", lambda _: None)

    assert (
        analyze_b1_run_set.main(
            [
                "--run-set",
                str(run_set),
                "--source-root",
                str(tmp_path),
                "--page-rco-report",
                str(report),
                "--analysis-id",
                "analysis-stage4",
                "--analysis-git-commit",
                "b" * 40,
                "--output-dir",
                str(output),
                "--expected-batches",
                "1",
                "--expected-samples",
                "2",
                "--run-stage4",
                "--bootstrap-replicates",
                "20",
                "--bootstrap-seed",
                "17",
                "--token-comparison-chunk-steps",
                "1",
            ]
        )
        == 0
    )

    partial = json.loads((output / "stage4_manifest.json").read_text())
    summary = json.loads((output / "stage4_token_summary.json").read_text())
    assert partial["status"] == "partial"
    assert summary["axes"]["token_k"] == [16, 32, 64]
    assert (output / "stage4_token_profiles.npz").is_file()
    assert not (output / "analysis_manifest.json").exists()


def test_stage5_cli_validates_partial_sources_and_writes_relationship_bundle(tmp_path, monkeypatch):
    run_set, report, _ = make_fixture(tmp_path)
    stage3 = tmp_path / "analysis-stage3"
    stage4 = tmp_path / "analysis-stage4"
    stage5 = tmp_path / "analysis-stage5"
    monkeypatch.setattr(analyze_b1_run_set, "validate_analysis_git_state", lambda _: None)
    common = [
        "--run-set",
        str(run_set),
        "--source-root",
        str(tmp_path),
        "--page-rco-report",
        str(report),
        "--analysis-git-commit",
        "b" * 40,
        "--expected-batches",
        "1",
        "--expected-samples",
        "2",
        "--bootstrap-replicates",
        "10",
        "--bootstrap-seed",
        "17",
    ]
    assert (
        analyze_b1_run_set.main(
            common
            + [
                "--analysis-id",
                "analysis-stage3",
                "--output-dir",
                str(stage3),
                "--run-stage3",
            ]
        )
        == 0
    )
    assert (
        analyze_b1_run_set.main(
            common
            + [
                "--analysis-id",
                "analysis-stage4",
                "--output-dir",
                str(stage4),
                "--run-stage4",
                "--token-comparison-chunk-steps",
                "1",
            ]
        )
        == 0
    )
    assert (
        analyze_b1_run_set.main(
            common
            + [
                "--analysis-id",
                "analysis-stage5",
                "--output-dir",
                str(stage5),
                "--run-stage5",
                "--stage3-dir",
                str(stage3),
                "--stage4-dir",
                str(stage4),
            ]
        )
        == 0
    )

    manifest = json.loads((stage5 / "stage5_manifest.json").read_text())
    summary = json.loads((stage5 / "stage5_relationships.json").read_text())
    assert manifest["status"] == "partial"
    assert manifest["pending_stages"] == ["completed_manifest"]
    assert summary["record_count"] == 112
    assert (stage5 / "stage5_relationship_profiles.npz").is_file()
    assert not (stage5 / "analysis_manifest.json").exists()


def test_analysis_git_state_requires_exact_clean_head(monkeypatch):
    class Result:
        def __init__(self, stdout):
            self.stdout = stdout

    responses = iter((Result("a" * 40 + "\n"), Result("")))
    monkeypatch.setattr(analyze_b1_run_set.subprocess, "run", lambda *args, **kwargs: next(responses))
    analyze_b1_run_set.validate_analysis_git_state("a" * 40)

    responses = iter((Result("a" * 40 + "\n"), Result(" M changed.py\n")))
    monkeypatch.setattr(analyze_b1_run_set.subprocess, "run", lambda *args, **kwargs: next(responses))
    with pytest.raises(B1OfflineError, match="not clean"):
        analyze_b1_run_set.validate_analysis_git_state("a" * 40)
