import numpy as np
import pytest

from experiments.head_analysis.analysis.pca_subspace import (
    ANGLE_ATOL_RAD, ERROR_ATOL, OVERLAP_ATOL, captured_energy, check_nonnegative,
    diagnose_moments, diagnose_pairs, principal_geometry, spectral_boundary,
)
from experiments.head_analysis.analysis.pca_reconstruction import (
    RANKS as OLD_RANKS, condition_weights, fit_basis, fit_calibration, oracle_errors, projection_errors,
)
from experiments.head_analysis.common.io import read_json, write_json, write_npz
from experiments.head_analysis.common.statistics import correlation_1d
from experiments.head_analysis.analysis import kv_subspace_diagnostic as pipeline


def test_geometry_ignores_internal_rotation_and_has_separate_angle_tolerance():
    rng = np.random.default_rng(8)
    basis = np.linalg.qr(rng.normal(size=(128, 64)))[0]
    rotation = np.linalg.qr(rng.normal(size=(64, 64)))[0]
    result = principal_geometry(basis, basis @ rotation, keep_angles=True)
    np.testing.assert_allclose(result["overlap"][-1], 1, atol=OVERLAP_ATOL)
    for rank in (16, 32):
        # A rotation mixing retained and discarded columns changes these prefixes.
        assert result["overlap"][list(pipeline.RANKS).index(rank)] < 1
        internal = np.linalg.qr(rng.normal(size=(rank, rank)))[0]
        same = principal_geometry(basis[:, :rank], basis[:, :rank] @ internal, [rank])
        np.testing.assert_allclose(same["overlap"], 1, atol=OVERLAP_ATOL)
        np.testing.assert_allclose(same["angle_max"], 0, atol=ANGLE_ATOL_RAD)
    np.testing.assert_allclose(result["angles_r64"], 0, atol=ANGLE_ATOL_RAD)
    assert ANGLE_ATOL_RAD > ERROR_ATOL
    orthogonal = principal_geometry(np.eye(128)[:, :64], np.eye(128)[:, 64:], [64])
    np.testing.assert_allclose(orthogonal["overlap"], 0, atol=OVERLAP_ATOL)
    np.testing.assert_allclose(orthogonal["angle_max"], np.pi/2, atol=ANGLE_ATOL_RAD)


def test_equal_boundary_can_change_subspace_without_projection_penalty():
    matrix = np.diag([3., 2., 2., 1.])
    a, b = np.eye(4)[:, :2], np.eye(4)[:, [0, 2]]
    geometry = principal_geometry(a, b, [2], keep_angles=True)
    np.testing.assert_allclose(geometry["angles_r2"], [0, np.pi/2], atol=ANGLE_ATOL_RAD)
    np.testing.assert_allclose(captured_energy(matrix, a, [2]), captured_energy(matrix, b, [2]))
    np.testing.assert_allclose(spectral_boundary(np.array([3., 2., 2., 1.]), [2]), [[2, 2, 0]])


def test_signed_roundoff_losses_are_retained_and_large_negatives_raise():
    values = np.array([.1, -ERROR_ATOL/2, 0])
    before = values.copy()
    flags = check_nonnegative(values, "example")
    np.testing.assert_array_equal(values, before)
    np.testing.assert_array_equal(flags, [False, True, False])
    with pytest.raises(ValueError, match="negative normalized loss"):
        check_nonnegative(np.array([-2*ERROR_ATOL]), "bad")
    transient = {"M": np.eye(4)[None], "M_basis": np.eye(4)[None],
                 "C_basis": np.eye(4)[None], "optimal_capture": np.array([[-1.]])}
    with pytest.raises(ValueError, match="forward_gap"):
        diagnose_pairs(transient, [[0, 0]], np.array(1.), [1])


def test_moment_gap_decompositions_match_direct_reconstruction_and_are_signed():
    covariance = np.diag([5., 1., .2, .1])
    mean = np.array([0., 3., 0., 0.])
    shape = (1, 1)
    fixed = np.broadcast_to(np.eye(4), shape + (4, 4)).copy()
    data, transient = diagnose_moments(mean[None, None, None], covariance[None, None, None],
        np.zeros(shape + (4,)), fixed, np.ones(shape), fixed, [1, 2])
    assert data["gap_covariance"][0, 0, 0, 0] < 0
    assert data["gap_mean"][0, 0, 0, 0] > 0
    vectors = np.concatenate((np.sqrt(4*covariance), -np.sqrt(4*covariance))) + mean
    for ri, rank in enumerate((1, 2)):
        u = fixed[0, 0, :, :rank]
        actual = np.square(vectors @ u @ u.T - vectors).sum(-1).mean()
        np.testing.assert_allclose(actual, data["fixed_mse"][0, 0, 0, ri], atol=1e-10)
    np.testing.assert_allclose(data["gap"], data["gap_covariance"]+data["gap_mean"], atol=1e-10)
    np.testing.assert_allclose(data["gap"], data["missed_top_energy"]-data["recovered_tail_energy"], atol=1e-10)
    peer = diagnose_pairs(transient, [[0, 0]], np.ones(shape), [1, 2])
    np.testing.assert_allclose(peer["forward_gap"], 0, atol=1e-10)
    assert data["boundary_M"].shape == (1, 1, 1, 2, 3)


def test_correct_calib_fit_can_have_oracle_gap_and_pair_transfer_is_asymmetric():
    cov = np.array([np.diag([10., 1., .1, .1]), np.diag([1., 2., .1, .1])])[:, None, None]
    mu = np.zeros((2, 1, 1, 4))
    fitted = fit_calibration(mu, cov, ["one", "one"])
    basis = fit_basis(fitted["G"], fitted["scale"])["basis"]
    data, transient = diagnose_moments(mu, cov, fitted["mean"], basis, fitted["scale"], basis, [1])
    np.testing.assert_allclose(data["gap"].ravel(), [0, 1/7.2], atol=ERROR_ATOL)
    assert data["gap"].mean() > 0  # In-fit compromise, despite an optimal average fit.
    pairs = diagnose_pairs(transient, [[0, 1]], fitted["scale"], [1])
    np.testing.assert_allclose(pairs["forward_gap"], 9/7.2, atol=ERROR_ATOL)
    np.testing.assert_allclose(pairs["reverse_gap"], 1/7.2, atol=ERROR_ATOL)


def _frozen_bundle(path):
    records = []
    conditions = [("ruler", 8192), ("ruler", 32768), ("govreport", 8192),
                  ("govreport", 16384), ("pg19", 8192), ("pg19", 32768)]
    for cohort in ("calib", "validation", "test"):
        for dataset, length in conditions:
            count = 2 if dataset == "pg19" else 3 if cohort == "calib" else 4
            for source in range(count):
                group = f"pg19:{cohort}:{source}" if dataset == "pg19" else f"{dataset}:{cohort}:{length}:{source}"
                records.append({"sample_id": f"{dataset}_{cohort}_{length}_{source}",
                    "dataset": dataset, "condition": f"{dataset}_{length}", "source_group": group,
                    "pair_id": group if dataset == "pg19" else None, "split": cohort,
                    "num_layers": 1, "num_kv_heads": 2, "head_dim": 128,
                    "prompt_length": 600, "index": len(records)})
    means = np.zeros((len(records), 2, 1, 2, 128), np.float32)
    covs = np.zeros((len(records), 2, 1, 2, 128, 128), np.float32)
    for i, row in enumerate(records):
        for rep, head in np.ndindex(2, 2):
            scale = 1 + 5*rep + 3*head
            values = np.linspace(2., .05, 128) * scale
            covariance = np.diag(values)
            angle = .13*(i % 7) + .03*head
            c, s = np.cos(angle), np.sin(angle)
            block = np.array([[c, -s], [s, c]])
            covariance[np.ix_([0, 90], [0, 90])] = block @ np.diag(values[[0, 90]]) @ block.T
            covs[i, rep, 0, head] = covariance
            means[i, rep, 0, head, 1] = .2*(i % 9)
            means[i, rep, 0, head, 2] = .1*rep
    calib = [r["index"] for r in records if r["split"] == "calib"]
    fit = fit_calibration(means[calib], covs[calib], [records[i]["condition"] for i in calib],
                          [records[i]["source_group"] for i in calib])
    basis = fit_basis(fit["G"], fit["scale"])["basis"]
    write_json(path / "split.json", {"samples": records})
    write_npz(path / "calibration.npz", {"mean": fit["mean"], "scale": fit["scale"], "G": fit["G"],
        "head_basis": basis, "calibration_weights": fit["weights"],
        "calibration_sample_ids": np.array([records[i]["sample_id"] for i in calib])})
    for cohort in ("validation", "test"):
        ids = [r["index"] for r in records if r["split"] == cohort]
        old_oracle = oracle_errors(means[ids], covs[ids], fit["mean"])["same_mean"]
        old_fixed = projection_errors(means[ids], covs[ids], fit["mean"], basis)
        write_npz(path / f"{cohort}_errors.npz", {"sample_ids": np.array([records[i]["sample_id"] for i in ids]),
            "ranks": OLD_RANKS, "methods": np.array(["oracle_same_mean", "head"]),
            "ideal_mse": np.stack((old_oracle, old_fixed))})
    return records, means, covs


def test_complete_layer_pipeline_reuses_fit_and_bootstraps_paired_sources(tmp_path, monkeypatch):
    bundle, output = tmp_path / "frozen", tmp_path / "new"
    records, means, covs = _frozen_bundle(bundle)
    original = {p.name: p.read_bytes() for p in bundle.iterdir()}
    calls = []
    def read(record, *, layer_ids):
        assert layer_ids == [0]
        calls.append(record["sample_id"])
        return {"mean": means[record["index"]], "covariance": covs[record["index"]]}
    monkeypatch.setattr(pipeline, "read_sample", read)
    pipeline.run_diagnostic(bundle, output, bootstrap_replicates=8)
    summary = read_json(output / "summary.json")
    assert summary["status"] == "completed" and len(calls) == len(records)
    assert original == {p.name: p.read_bytes() for p in bundle.iterdir()}
    metadata = read_json(output / "metadata.json")
    assert len(metadata["pair_indices"]["test"]) == 190
    with np.load(output / "layers/layer_000.npz") as a:
        assert a["boundary_G"].shape == a["boundary_GC"].shape == (2, 2, 3, 3)
        assert a["boundary_M"].shape == a["boundary_C"].shape == (len(records), 2, 2, 3, 3)
        assert a["main_angles_r64"].shape == (len(records), 2, 2, 64)
        saved = {k: a[k] for k in a.files}
    with np.load(output / "bootstrap.npz") as bootstrap:
        rows = [r for r in records if r["split"] == "test"]
        draw = bootstrap["test_source_draw_indices"]
        pg8 = [i for i, r in enumerate(rows) if r["condition"] == "pg19_8192"]
        pg32 = [i for i, r in enumerate(rows) if r["condition"] == "pg19_32768"]
        source = np.array([r["source_group"] for r in rows])
        np.testing.assert_array_equal(source[draw[:, pg8]], source[draw[:, pg32]])
        ids = [r["index"] for r in rows]
        x, y = 1-saved["main_overlap"][ids, ..., 2], saved["gap"][ids, ..., 2]
        for bi, selected in enumerate(draw):
            reference = np.array([[correlation_1d(x[selected, rep, h], y[selected, rep, h])[1]
                                   for h in range(2)] for rep in range(2)])
            np.testing.assert_allclose(np.nanmean(reference, 1),
                bootstrap["test_correlation_mean_bootstrap"][bi, :, 0], atol=1e-10)
        weights = condition_weights([r["condition"] for r in rows], [r["source_group"] for r in rows])
        expected = np.einsum("s,bsr->br", weights, saved["gap"][ids, ..., 2].mean(2)[draw])
        gap_column = list(bootstrap["mean_metric_names"]).index("gap")
        np.testing.assert_allclose(expected, bootstrap["test_mean_bootstrap"][:, :, gap_column], atol=1e-10)
    with np.load(bundle / "calibration.npz") as model, np.load(bundle / "test_errors.npz") as old:
        positions = [list(old["ranks"]).index(int(r)) for r in pipeline.RANKS]
        normal = np.take(old["ideal_mse"][:, :, :, 0], positions, axis=-1) / model["scale"][:, 0, :, None]
        avg = np.einsum("s,msrk->mrk", weights, normal.mean(3))
        np.testing.assert_allclose(summary["cohorts"]["test"]["fixed_over_oracle_increase"],
                                   avg[1]/avg[0]-1, atol=1e-10)
    with pytest.raises(FileExistsError):
        pipeline.run_diagnostic(bundle, output, bootstrap_replicates=8)
