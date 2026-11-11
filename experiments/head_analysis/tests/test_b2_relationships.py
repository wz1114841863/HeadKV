import hashlib
import json

import numpy as np
import pytest

from experiments.head_analysis.analysis.b1_offline import sha256_file
from experiments.head_analysis.analysis.b2_consistency import B2SourceError
from experiments.head_analysis.analysis.b2_relationships import (
    analyze_b2_relationships,
    bootstrap_ci,
    bootstrap_draws,
    bootstrap_means,
    correlate_profiles,
    load_stage2_relationship_inputs,
)
from experiments.head_analysis.scripts import analyze_b2_run_set as cli
from experiments.head_analysis.tests.b2_fixtures import oracle_correlation, relationship_fixture, stage2_fixture


def test_correlations_match_independent_oracle_with_ties_and_missing_heads():
    rng = np.random.default_rng(17)
    x, y = rng.integers(0, 5, size=(2, 9, 3, 8)).astype(float)
    x[1, 2] = np.nan
    y[4, 0, 0] = np.nan
    result = correlate_profiles(x, y)
    for item in range(len(x)):
        for view in range(2):
            for method in range(2):
                expected = oracle_correlation(x[item], y[item], centered=bool(view), ranked=method == 0)
                np.testing.assert_allclose(result["values"][item, view, method], expected, atol=1e-14)
        for layer in range(3):
            expected = oracle_correlation(x[item, layer : layer + 1], y[item, layer : layer + 1], ranked=True)
            np.testing.assert_allclose(result["layer_values"][item, layer, 0], expected, equal_nan=True)
    assert result["valid_heads"][1, 0] == 16


def test_layer_confounding_removed_before_ranking():
    x = np.asarray([[0, 1, 2, 3], [100, 101, 102, 103]])
    y = np.asarray([[3, 2, 1, 0], [103, 102, 101, 100]])
    result = correlate_profiles(x, y)
    assert (result["values"][0] > 0).all()
    np.testing.assert_allclose(result["values"][1], -1)
    np.testing.assert_allclose(result["layer_values"], -1)


@pytest.mark.parametrize("kind", ["constant", "all_missing", "two_valid"])
def test_undefined_correlations_keep_nan_and_explicit_counts(kind):
    x = np.ones((1, 4))
    y = np.arange(4, dtype=float)[None]
    if kind == "all_missing":
        y[:] = np.nan
    elif kind == "two_valid":
        x = y.copy()
        y[0, 2:] = np.nan
    result = correlate_profiles(x, y)
    assert np.isnan(result["values"]).all()
    assert result["valid_heads"].tolist() == [dict(constant=4, all_missing=0, two_valid=2)[kind]] * 2
    ci, n = bootstrap_ci(np.stack([result["values"]] * 10))
    assert np.isnan(ci).all() and (n == 0).all()


def test_cluster_bootstrap_matches_manual_sample_reconstruction():
    values = np.asarray([[[1, 4], [9, np.nan]], [[3, 2], [7, 5]], [[8, 6], [1, np.nan]]])
    draws = bootstrap_draws(20261110, "example", 17, 3)
    expected = []
    for draw in draws:
        profile = np.empty((2, 2))
        for layer in range(2):
            for h in range(2):
                observed = [values[s, layer, h] for s in draw if np.isfinite(values[s, layer, h])]
                profile[layer, h] = sum(observed) / len(observed) if observed else np.nan
        expected.append(profile)
    np.testing.assert_allclose(bootstrap_means(values, draws), expected, equal_nan=True)
    derived = int.from_bytes(hashlib.sha256(b"20261110:example").digest()[:8], "little")
    manual = np.random.Generator(np.random.PCG64(derived)).integers(0, 3, size=(17, 3))
    np.testing.assert_array_equal(draws, manual)
    np.testing.assert_array_equal(draws, bootstrap_draws(20261110, "example", 17, 3))
    assert not np.array_equal(draws, bootstrap_draws(20261110, "different", 17, 3))


def test_percentile_ci_excludes_only_undefined_replicates():
    values = np.asarray([[1, np.nan, 0], [2, np.nan, 4], [3, np.nan, np.nan], [4, np.nan, 8]])
    ci, counts = bootstrap_ci(values)
    np.testing.assert_allclose(ci[0], [1.075, 3.925])
    np.testing.assert_allclose(ci[2], [0.2, 7.8])
    assert np.isnan(ci[1]).all() and counts.tolist() == [4, 0, 3]


def test_end_to_end_relations_pairing_sensitivity_and_percentile_intervals(tmp_path):
    inputs = relationship_fixture(tmp_path)
    result = analyze_b2_relationships(inputs, bootstrap_replicates=13)
    a = result.arrays
    assert set(result.axes) == set(a)
    assert all(len(result.axes[k]) == v.ndim for k, v in a.items())
    assert len(result.records) == 2 * 4 * 3 * 3 * 2 + 2 * 4 * 2
    for i, record in enumerate(result.records):
        expected = -1 if record["target"] == "learned_excess" else 1
        np.testing.assert_allclose(a["correlation"][i], expected)
        np.testing.assert_allclose(a["correlation_ci"][i], expected)
        np.testing.assert_allclose(a["per_layer_correlation"][i], expected)
        np.testing.assert_allclose(a["per_layer_ci"][i], expected)
    assert (a["correlation_valid_replicates"] == 13).all()
    assert a["pg19_pair_ids"].tolist() == ["a", "b", "c"]
    np.testing.assert_allclose(a["pg19_cv_pair_difference"], 2)
    np.testing.assert_allclose(a["pg19_cv_ci"], 2)
    np.testing.assert_allclose(a["pg19_coverage_pair_difference"][..., :2, :], 2 / 32)
    np.testing.assert_allclose(a["pg19_coverage_pair_difference"][..., 2, :], -2 / 32)
    np.testing.assert_allclose(a["sensitivity_difference"], 0)
    np.testing.assert_allclose(a["sensitivity_ci"], 0)
    np.testing.assert_allclose(a["pg19_correlation_difference"], 0)
    np.testing.assert_allclose(a["pg19_correlation_ci"], 0)
    assert result.metadata["draws_sha256"]["pg19:8k"] == result.metadata["draws_sha256"]["pg19:32k"]


def test_bootstrap_correlations_rebuild_profiles_and_do_not_resample_heads(tmp_path):
    inputs = relationship_fixture(tmp_path)
    a = inputs.arrays
    rng = np.random.default_rng(917)
    a["sample_prefill_features"][..., 0] = rng.uniform(1, 9, size=(6, 2, 4, 4))
    a["sample_recall_full"][..., 0] = rng.random((6, 2, 4, 4, 3, 2, 2))
    result = analyze_b2_relationships(inputs, bootstrap_replicates=19)
    i = next(i for i, r in enumerate(result.records) if r["phase"] == "full" and r["target"] == "learned_recall")
    rows = [1, 0, 2]  # pg32, sorted by opaque pair a,b,c
    x = a["sample_prefill_features"][rows, :, :, 0, 0].astype(float)
    y = a["sample_recall_full"][rows, :, :, 0, 0, 0, 0, 0].astype(float)
    expected = []
    for draw in bootstrap_draws(20261110, "pg19-pairs", 19, 3):
        expected.append(oracle_correlation(x[draw].mean(0), y[draw].mean(0), centered=True, ranked=True))
    np.testing.assert_allclose(result.arrays["correlation_ci"][i, 1, 0], np.quantile(expected, [0.025, 0.975]))
    np.testing.assert_allclose(
        result.arrays["correlation"][i, 1, 0], oracle_correlation(x.mean(0), y.mean(0), centered=True, ranked=True)
    )


def test_short_samples_have_no_quarter_relations_or_temporal_confidence(tmp_path):
    inputs = relationship_fixture(tmp_path)
    inputs.arrays["sample_decode_lengths"][:] = 6
    inputs.arrays["sample_recall_quarters"][:] = np.nan
    inputs.arrays["sample_recall_temporal"][:] = np.nan
    result = analyze_b2_relationships(inputs, bootstrap_replicates=5)
    assert all(r["phase"] == "full" for r in result.records)
    assert np.isnan(result.arrays["temporal_condition_ci"]).all()
    assert (result.arrays["temporal_condition_valid_replicates"] == 0).all()
    assert np.isnan(result.arrays["pg19_temporal_ci"]).all()


def test_shared_draws_produce_zero_paired_contrast_for_variable_identical_profiles(tmp_path):
    inputs = relationship_fixture(tmp_path)
    a = inputs.arrays
    rng = np.random.default_rng(71)
    x = rng.uniform(1, 5, (3, 2, 4))
    y = rng.random((3, 2, 4))
    for i, pair in enumerate(a["sample_pair_ids"]):
        pair_row = {"a": 0, "b": 1, "c": 2}[str(pair)]
        a["sample_prefill_features"][i, ..., 0] = x[pair_row, ..., None]
        a["sample_recall_full"][i, ..., 0] = y[pair_row, ..., None, None, None, None]
        a["sample_learned_excess_full"][i, ..., 0] = y[pair_row, ..., None, None, None]
    result = analyze_b2_relationships(inputs, bootstrap_replicates=23)
    # Profile correlations vary between replicates, so unrelated draws would not give zero intervals.
    assert np.ptp(result.arrays["correlation_ci"][0, 1, 0]) > 0.1
    np.testing.assert_allclose(result.arrays["sensitivity_ci"], 0)
    np.testing.assert_allclose(result.arrays["pg19_correlation_ci"], 0)


def test_nonpositive_bootstrap_count_rejected_before_analysis(tmp_path):
    with pytest.raises(B2SourceError, match="positive"):
        analyze_b2_relationships(relationship_fixture(tmp_path), bootstrap_replicates=0)


@pytest.mark.parametrize("kind", ["missing_pair", "duplicate_pair", "no_pair_id"])
def test_invalid_pg19_pairs_fail(tmp_path, kind):
    inputs = relationship_fixture(tmp_path)
    if kind == "missing_pair":
        inputs.arrays["sample_pair_ids"][0] = "z"
    elif kind == "duplicate_pair":
        inputs.arrays["sample_pair_ids"][0] = "a"
    else:
        inputs.arrays["sample_pair_ids"][0] = ""
    with pytest.raises(B2SourceError, match="pair"):
        analyze_b2_relationships(inputs, bootstrap_replicates=2)


@pytest.mark.parametrize("kind", ["manifest", "source", "sample_order", "axis"])
def test_stage2_loader_detects_authority_artifact_and_coordinate_drift(tmp_path, monkeypatch, kind):
    index, output, _ = stage2_fixture(tmp_path, monkeypatch)
    digest = sha256_file(output / "stage2_manifest.json")
    valid = load_stage2_relationship_inputs(output, index, expected_manifest_sha256=digest)
    assert len(valid.arrays["sample_ids"]) == 2
    if kind == "manifest":
        (output / "stage2_manifest.json").write_text("{}")
    elif kind == "source":
        with (output / "stage2_profiles.npz").open("ab") as f:
            f.write(b"drift")
    else:
        # Re-sign a synthetic upstream manifest to prove semantic checks are independent of Hash checks.
        with np.load(output / "stage2_profiles.npz") as archive:
            arrays = {k: archive[k] for k in archive.files}
        if kind == "sample_order":
            arrays["sample_ids"] = arrays["sample_ids"][::-1]
        else:
            arrays["budgets"] = np.asarray([128, 2048, 512], dtype=np.uint32)
        np.savez_compressed(output / "stage2_profiles.npz", **arrays)
        path = output / "stage2_manifest.json"
        payload = json.loads(path.read_text())
        for entry in payload["outputs"]:
            p = output / entry["relative_path"]
            entry.update(sha256=sha256_file(p), file_bytes=p.stat().st_size)
        path.write_text(json.dumps(payload))
        digest = sha256_file(path)
    with pytest.raises(B2SourceError):
        load_stage2_relationship_inputs(output, index, expected_manifest_sha256=digest)


def test_stage2_normal_read_does_not_require_file_hash(tmp_path, monkeypatch):
    index, output, _ = stage2_fixture(tmp_path, monkeypatch)
    with (output / "stage2_profiles.npz").open("ab") as handle:
        handle.write(b"extra")
    assert len(load_stage2_relationship_inputs(output, index).arrays["sample_ids"]) == 2


def test_stage3_cli_hash_bindings_nulls_partial_state_and_no_overwrite(tmp_path, monkeypatch):
    index, stage2, common = stage2_fixture(tmp_path, monkeypatch)
    output = tmp_path / "analysis/stage3"
    args = cli.make_parser().parse_args(
        common
        + [
            "--analysis-id",
            "s3",
            "--output-dir",
            str(output),
            "--run-stage3",
            "--stage2-dir",
            str(stage2),
            "--expected-stage2-manifest-sha256",
            sha256_file(stage2 / "stage2_manifest.json"),
            "--bootstrap-replicates",
            "3",
        ]
    )
    cli.prepare_analysis(args)
    assert {p.name for p in output.iterdir()} == {
        "analysis_plan.json",
        "stage3_relationships.npz",
        "stage3_summary.json",
        "stage3_manifest.json",
    }
    manifest = json.loads((output / "stage3_manifest.json").read_text())
    assert manifest["status"] == "partial" and manifest["source_run_set_sha256"] == index.run_set_sha256
    assert manifest["stage2_source"]["sha256"] == sha256_file(stage2 / "stage2_manifest.json")
    for entry in manifest["outputs"]:
        path = output / entry["relative_path"]
        assert entry["sha256"] == sha256_file(path) and entry["file_bytes"] == path.stat().st_size
    summary = json.loads((output / "stage3_summary.json").read_text())
    assert summary["records"][0]["coefficients"] == [[None, None], [None, None]]  # only 2 Heads in fixture
    assert summary["statistics"]["bootstrap_replicates"] == 3
    assert not (output / "analysis_manifest.json").exists()
    with pytest.raises(B2SourceError, match="already exists"):
        cli.prepare_analysis(args)
    args.output_dir = str(tmp_path / "bad_input")
    args.expected_stage2_manifest_sha256 = "0" * 64
    with pytest.raises(B2SourceError):
        cli.prepare_analysis(args)
    assert not (tmp_path / "bad_input").exists()
