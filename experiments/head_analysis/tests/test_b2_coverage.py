import hashlib
import json

import numpy as np
import pytest

from experiments.head_analysis.analysis.b2_consistency import B2BatchIndex, B2RunSetIndex, B2SampleIndex, B2SourceError
from experiments.head_analysis.analysis.b2_coverage import (
    derive_coverage_metrics,
    summarize_b2_coverage,
    summarize_trajectory,
)
from experiments.head_analysis.scripts import analyze_b2_run_set as cli
from experiments.head_analysis.storage.b2_trace_writer import B2TraceSchema
from experiments.head_analysis.tests.b2_fixtures import fixture, load
from experiments.head_analysis.tests.fixtures import make_prefill


def example():
    p = np.asarray([0.5, 0.25]).reshape(2, 1, 1)
    learned = np.asarray([[0.2, 0.3, 0.35], [0.02, 0.10, 0.15]]).reshape(2, 1, 1, 1, 3)
    fixed = np.asarray([0.1, 0.05]).reshape(2, 1, 1, 1, 1)
    a = np.stack((learned, learned + fixed), axis=-1)
    return p, a, np.full((1, 1, 1), 600, dtype=np.uint32), np.asarray([1000], dtype=np.uint32)


def test_derivations_match_independent_arithmetic_without_mutating_inputs():
    p, a, counts, candidates = example()
    originals = [value.copy() for value in (p, a, counts, candidates)]
    metrics, sizes, diagnostics = derive_coverage_metrics(p, a, counts, candidates, (128, 512, 2048), np.asarray([128]))
    expected = np.empty(a.shape + (2,))
    for step in range(2):
        for budget_index in range(3):
            for set_type in range(2):
                value = a[step, 0, 0, 0, budget_index, set_type]
                expected[step, 0, 0, 0, budget_index, set_type] = [value, value / p[step, 0, 0]]
    np.testing.assert_allclose(metrics["recall"], expected)
    np.testing.assert_allclose(
        metrics["uniform_expected_recall"][0, 0, 0, 0, :, 0, 0], 0.4 * np.asarray([128, 512, 600]) / 1000
    )
    np.testing.assert_allclose(metrics["learned_excess"][0, 0, 0, 0, :, 0], [0.2 - 0.0512, 0.3 - 0.2048, 0.35 - 0.24])
    np.testing.assert_allclose(metrics["budget_gain"][0, 0, 0, 0, :, 0, 0], [0.1, 0.05])
    np.testing.assert_allclose(metrics["gain_per_added_token"][0, 0, 0, 0, :, 0, 0], [0.1 / 384, 0.05 / 88])
    assert sizes[0, 0, 0].tolist() == [[128, 256], [512, 640], [600, 728]]
    assert diagnostics["middle_mass_clipped_count"] == 0
    reduced = summarize_trajectory(metrics["recall"])
    np.testing.assert_allclose(reduced["full"][0, 0, 0, 0, 0, 1, 0], (0.2 / 0.5 + 0.02 / 0.25) / 2)
    assert not np.isclose((0.2 + 0.02) / (0.5 + 0.25), reduced["full"][0, 0, 0, 0, 0, 1, 0])
    for value, original in zip((p, a, counts, candidates), originals):
        np.testing.assert_array_equal(value, original)


def test_zero_prompt_mass_is_nan_but_small_positive_mass_remains_valid():
    p = np.asarray([0.0, 1e-30]).reshape(2, 1, 1)
    a = np.broadcast_to(p[..., None, None, None] * 0.2, (2, 1, 1, 1, 3, 2)).copy()
    metrics, _, _ = derive_coverage_metrics(
        p, a, np.asarray([[[1]]]), np.asarray([1000]), (128, 512, 2048), np.asarray([128])
    )
    assert np.isnan(metrics["recall"][0, ..., 1]).all()
    np.testing.assert_allclose(metrics["recall"][1, ..., 1], 0.2)
    reduced = summarize_trajectory(metrics["recall"])
    assert (reduced["valid_count"][..., 0] == 2).all()
    assert (reduced["valid_count"][..., 1] == 1).all()
    assert np.isnan(metrics["gain_per_added_token"]).all()


@pytest.mark.parametrize("kind", ["fixed_budget", "nonmonotonic", "nan", "negative", "count", "shape"])
def test_rejects_invalid_derivation_inputs(kind):
    p, a, counts, candidates = example()
    if kind == "fixed_budget":
        a[..., 0, 1] += 0.02
    elif kind == "nonmonotonic":
        a[..., 1, :] = 0
    elif kind == "nan":
        p[0] = np.nan
    elif kind == "negative":
        p[0] = -0.1
    elif kind == "count":
        counts[:] = 1001
    else:
        a = a[..., :1]
    with pytest.raises(B2SourceError):
        derive_coverage_metrics(p, a, counts, candidates, (128, 512, 2048), np.asarray([128]))


def test_tiny_negative_middle_mass_is_clipped_and_recorded_only_within_tolerance():
    p = np.asarray([[[0.1]]])
    a = np.zeros((1, 1, 1, 1, 3, 2))
    a[..., 1] = 0.1 + 5e-6
    metrics, _, diagnostics = derive_coverage_metrics(
        p, a, np.asarray([[[0]]]), np.asarray([1000]), (128, 512, 2048), np.asarray([128])
    )
    assert diagnostics["middle_mass_clipped_count"] == 1
    np.testing.assert_allclose(diagnostics["middle_mass_max_negative_magnitude"], 5e-6)
    assert (metrics["fixed_middle_mass"][..., 1, 0] == 0).all()
    a[..., 1] = 0.1 + 2e-5
    with pytest.raises(B2SourceError):
        derive_coverage_metrics(p, a, np.asarray([[[0]]]), np.asarray([1000]), (128, 512, 2048), np.asarray([128]))


def test_exact_temporal_statistics_linear_trajectory_and_missing_auc_policy():
    u = np.arange(40) / 39
    values = np.stack((1 + 2 * u, 1 + 2 * u), axis=-1)
    values[7, 1] = np.nan
    result = summarize_trajectory(values, column_block=1)
    np.testing.assert_allclose(
        result["full"][0], [2, 2, np.quantile(values[:, 0], 0.1), np.quantile(values[:, 0], 0.9)]
    )
    np.testing.assert_allclose(result["temporal"][0], [2, 2, 60 / 39], rtol=1e-6)
    np.testing.assert_allclose(result["temporal"][1, 0], 2)
    assert np.isnan(result["temporal"][1, 1]) and result["temporal_count"][1, 1] == 0
    assert result["valid_count"].tolist() == [40, 39]
    assert result["quarter_count"][:, 0].tolist() == [10] * 4
    assert result["bin_count"][:, 0].tolist() == [2] * 20
    np.testing.assert_allclose(result["bins"][0, 0], values[:2, 0].mean())
    for key, value in summarize_trajectory(values, column_block=7).items():
        np.testing.assert_allclose(value, result[key], equal_nan=True)


def test_normalized_auc_is_not_the_step_mean():
    length = 32
    values = (np.arange(length, dtype=np.float64) / (length - 1)) ** 2
    result = summarize_trajectory(values[:, None])
    np.testing.assert_allclose(result["temporal"][0, 1], 1 / 3 + 1 / (6 * (length - 1) ** 2), rtol=1e-6)
    assert not np.isclose(result["temporal"][0, 1], result["full"][0, 0])


@pytest.mark.parametrize("length", [0, 1, 6, 31])
def test_short_trajectories_do_not_fabricate_long_time_statistics(length):
    values = np.arange(length, dtype=np.float64).reshape(length, 1)
    result = summarize_trajectory(values)
    assert result["valid_count"].item() == length
    assert np.isnan(result["quarters"]).all() and np.isnan(result["bins"]).all()
    assert np.isnan(result["temporal"]).all() and (result["temporal_count"] == 0).all()
    if length == 0:
        assert np.isnan(result["full"]).all()


def synthetic_index(root, lengths=(2, 40)):
    """Small checked NPZ sources for the reducer, with unequal lengths in one data condition."""
    schema = B2TraceSchema(1, 4, 2).to_dict()
    run = root / "batch"
    entries, samples = [], {}
    for i, length in enumerate(lengths):
        sid = f"sample_{i}"
        sd = run / "samples" / sid
        sd.mkdir(parents=True)
        prefill = make_prefill(0)
        np.savez_compressed(
            sd / "prefill_consistency.npz",
            **{
                key: getattr(prefill, key) if key == "candidate_token_count" else getattr(prefill, key)[None, ...]
                for key in (
                    "candidate_token_count",
                    "critical_token_valid_count",
                    "cv_prefill",
                    "frequency_mean",
                    "frequency_std",
                    "binary_threshold",
                    "nonzero_frequency_count",
                )
            },
        )
        chunks = []
        for start in range(0, length, 13):
            stop = min(length, start + 13)
            p = np.full((stop - start, 1, 2), 0.8, dtype=np.float32)
            learned = np.full((stop - start, 1, 2, 4, 3), 0.1 if i == 0 else 0.5, dtype=np.float32)
            a = np.stack((learned, learned + 0.1), axis=-1)
            chunk = sd / f"decode_recall_{len(chunks):03d}.npz"
            np.savez_compressed(
                chunk,
                decode_steps=np.arange(start, stop, dtype=np.uint32),
                prompt_attention_mass=p,
                critical_attention_mass=a,
            )
            chunks.append(chunk)
        for path in sorted(sd.iterdir()):
            entries.append(
                {
                    "relative_path": str(path.relative_to(run)),
                    "sample_id": sid,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "file_bytes": path.stat().st_size,
                }
            )
        samples[sid] = B2SampleIndex(
            sid,
            "govreport",
            "summary",
            None,
            8192,
            384,
            "input",
            "generated",
            length + 1,
            length,
            "batch",
            sd,
            tuple(chunks),
        )
    fm = run / "file_manifest.json"
    fm.write_text(json.dumps({"files": entries}))
    batch = B2BatchIndex("batch", run, tuple(samples), hashlib.sha256(fm.read_bytes()).hexdigest(), len(entries), 0)
    return B2RunSetIndex(
        root / "manifest.json",
        root,
        "frozen-hash",
        {
            "execution_contract": {
                "axes": schema["axes"],
                "conditions": schema["conditions"],
                "dimensions": schema["dimensions"],
            }
        },
        samples,
        (batch,),
        root / "report.json",
        root / "b1.json",
        root / "b1analysis.json",
        0,
    )


def test_sample_equal_aggregation_saturation_short_steps_and_temporary_cleanup(tmp_path):
    index = synthetic_index(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = summarize_b2_coverage(index, scratch_dir=scratch)
    arrays = result.arrays
    # Equal sample weights: .1 and .5, despite 2 versus 40 observed Steps.
    np.testing.assert_allclose(arrays["condition_recall_full_mean"][..., 0, 0, 0], 0.3)
    assert not np.isclose((2 * 0.1 + 40 * 0.5) / 42, 0.3)
    assert arrays["sample_decode_lengths"].tolist() == [2, 40]
    assert arrays["short_sample_offsets"].tolist() == [0, 2, 2]
    assert arrays["short_recall"].shape[0] == 2
    assert np.isnan(arrays["sample_recall_bins"][0]).all()
    assert (arrays["condition_recall_bins_valid_samples"] == 1).all()
    assert np.isnan(arrays["sample_gain_per_added_token_full"]).all()
    assert (arrays["sample_gain_per_added_token_valid_count"] == 0).all()
    assert result.diagnostics["long_sample_count"] == result.diagnostics["short_sample_count"] == 1
    assert list(scratch.iterdir()) == []
    assert set(result.axes) == set(arrays)
    assert all(len(result.axes[key]) == value.ndim for key, value in arrays.items())


def test_reducer_handles_missing_post_index_artifact_and_cleans_scratch(tmp_path):
    index = synthetic_index(tmp_path)
    chunk = index.samples["sample_1"].decode_chunks[1]
    chunk.unlink()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(OSError):
        summarize_b2_coverage(index, scratch_dir=scratch)
    assert list(scratch.iterdir()) == []


def test_zero_decode_sample_has_nan_recall_and_zero_valid_steps(tmp_path):
    result = summarize_b2_coverage(synthetic_index(tmp_path, lengths=(0, 2)))
    assert np.isnan(result.arrays["sample_recall_full"][0]).all()
    assert (result.arrays["sample_recall_valid_count"][0] == 0).all()
    assert result.arrays["short_sample_offsets"].tolist() == [0, 0, 2]


def test_cli_stage2_binds_partial_outputs_and_does_not_complete_analysis(tmp_path, monkeypatch):
    manifest = fixture(tmp_path)
    index = load(manifest)
    monkeypatch.setattr(cli, "load_completed_b2_run_set", lambda *a, **k: index)
    output = tmp_path / "analysis/stage2"
    args = cli.make_parser().parse_args(
        [
            "--run-set",
            str(manifest),
            "--analysis-id",
            "stage2",
            "--analysis-git-commit",
            "a" * 40,
            "--output-dir",
            str(output),
            "--run-stage2",
        ]
    )
    cli.prepare_analysis(args)
    assert {p.name for p in output.iterdir()} == {
        "analysis_plan.json",
        "stage2_profiles.npz",
        "stage2_summary.json",
        "stage2_manifest.json",
    }
    partial = json.loads((output / "stage2_manifest.json").read_text())
    assert partial["status"] == "partial" and partial["source_run_set_sha256"] == index.run_set_sha256
    for entry in partial["outputs"]:
        path = output / entry["relative_path"]
        assert path.stat().st_size == entry["file_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    assert b'"NaN"' not in (output / "stage2_summary.json").read_bytes()
    assert not (output / "analysis_manifest.json").exists()
