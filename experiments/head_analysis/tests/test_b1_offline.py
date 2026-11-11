import json

import numpy as np
import pytest

from experiments.head_analysis.analysis.b1_offline import (
    B1OfflineError,
    aggregate_page_rco,
    build_analysis_plan,
    decode_segment_ids,
    load_completed_run_set,
    summarize_query_and_page_metrics,
    summarize_sample_equal,
    validate_analysis_sources,
)
from experiments.head_analysis.scripts import analyze_b1_run_set
from experiments.head_analysis.tests.fixtures import make_fixture, write_json
from experiments.head_analysis.storage import b1_reader


def test_b1_reader_preserves_existing_imports():
    from experiments.head_analysis.analysis import b1_offline

    assert b1_offline.load_completed_run_set is b1_reader.load_completed_run_set
    assert b1_offline.B1OfflineError is b1_reader.B1OfflineError
    assert b1_offline.RunSetIndex is b1_reader.RunSetIndex


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

    with pytest.raises(B1OfflineError, match="unable to read JSON"):
        load_completed_run_set(run_set, source_root=tmp_path)


def test_normal_read_does_not_require_current_file_hash(tmp_path):
    run_set, _, run_dir = make_fixture(tmp_path)
    with (run_dir / "samples" / "pg19_001_8k" / "generated_ids.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    load_completed_run_set(run_set, source_root=tmp_path)
    with pytest.raises(B1OfflineError, match="file size differs from manifest"):
        load_completed_run_set(run_set, source_root=tmp_path, verify_file_hashes=True)


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

    assert analyze_b1_run_set.main(arguments) == 0
    assert (output / "analysis_plan.json").is_file()
    assert not (output / "analysis_manifest.json").exists()
    with pytest.raises(SystemExit, match="already exists"):
        analyze_b1_run_set.main(arguments)


def test_stage3_cli_writes_partial_bundle_without_completed_manifest(tmp_path, monkeypatch):
    run_set, report, _ = make_fixture(tmp_path)
    output = tmp_path / "analysis-stage3"

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
