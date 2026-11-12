import numpy as np
import pytest

from experiments.head_analysis.analysis.kv_representation_metrics import (
    cka_statistics, head_cka_means, sampled_statistics, spectrum_metrics,
)


def direct_cka(left, right):
    left = left - left.mean(axis=0)
    right = right - right.mean(axis=0)
    denominator = np.linalg.norm(left.T @ left) * np.linalg.norm(right.T @ right)
    return np.linalg.norm(left.T @ right) ** 2 / denominator if denominator else np.nan


def test_spectrum_known_rank_coverage_and_zero_variance():
    covariance = np.zeros((3, 128, 128))
    covariance[0, :4, :4] = np.eye(4) * 2
    covariance[1] = np.eye(128)
    result = spectrum_metrics(covariance)
    np.testing.assert_array_equal(result["valid"], [True, True, False])
    np.testing.assert_allclose(result["effective_rank"][:2], [4, 128])
    np.testing.assert_array_equal(result["r90"][:2], [4, 116])
    np.testing.assert_array_equal(result["variance_coverage"][0], np.ones(4))
    np.testing.assert_array_equal(result["variance_coverage"][1], np.array([8, 16, 32, 64]) / 128)
    assert np.isnan(result["effective_rank"][2]) and np.isnan(result["r90"][2])
    assert np.isnan(result["variance_coverage"][2]).all()
    np.testing.assert_array_equal(result["eigenvalues"][0, :4], np.full(4, 2.))


def test_spectrum_records_roundoff_and_rejects_material_negative_values():
    covariance = np.diag([3., 1., -1e-8])
    result = spectrum_metrics(covariance)
    assert result["min_eigenvalue"] == pytest.approx(-1e-8)
    assert result["negative_tolerance"] == pytest.approx(3 * np.finfo(np.float32).eps * 3)
    assert result["eigenvalues"][-1] == 0
    np.testing.assert_array_equal(result["variance_coverage"], np.ones(4))
    with pytest.raises(ValueError, match="negative eigenvalue"):
        spectrum_metrics(np.diag([3., 1., -.1]))


def test_sampled_statistics_match_centered_reference_and_constant_data():
    rng = np.random.default_rng(3)
    x = 10000 + rng.normal(scale=.01, size=(2, 17, 4))
    result = sampled_statistics(x)
    assert result["count"] == 17
    for head in range(2):
        np.testing.assert_allclose(result["covariance"][head], np.cov(x[head], rowvar=False, ddof=0),
                                   rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(result["feature_energy"], (x ** 2).mean(axis=1), rtol=0, atol=0)
    norms = np.linalg.norm(x, axis=-1)
    np.testing.assert_array_equal(result["norm_summary"], np.stack((norms.mean(1),
        *np.quantile(norms, [.1, .5, .9], axis=1), norms.max(1)), axis=-1))
    np.testing.assert_array_equal(result["element_min"], x.min((1, 2)))
    np.testing.assert_array_equal(result["element_max"], x.max((1, 2)))
    constant = sampled_statistics(np.full((2, 17, 4), 7.))
    assert not constant["covariance"].any()
    assert not spectrum_metrics(constant["covariance"])["valid"].any()


def test_cka_gram_matches_direct_formula_and_one_sided_permutations():
    rng = np.random.default_rng(12)
    vectors = rng.normal(size=(4, 19, 5))
    permutations = np.stack([rng.permutation(19) for _ in range(7)])
    result = cka_statistics(vectors, permutations)
    np.testing.assert_array_equal(result["pairs"], [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    for row, (left, right) in enumerate(result["pairs"]):
        assert result["observed"][row] == pytest.approx(direct_cka(vectors[left], vectors[right]), abs=1e-14)
        expected = [direct_cka(vectors[left], vectors[right, permutation]) for permutation in permutations]
        np.testing.assert_allclose(result["permutation_values"][row], expected, rtol=1e-13, atol=1e-14)
        assert result["permutation_mean"][row] == pytest.approx(np.mean(expected))
        np.testing.assert_allclose([result["permutation_p05"][row], result["permutation_p95"][row]],
                                   np.quantile(expected, [.05, .95]), rtol=1e-13)
    np.testing.assert_allclose(result["excess"], result["observed"] - result["permutation_mean"])
    repeat = cka_statistics(vectors, permutations)
    np.testing.assert_array_equal(repeat["permutation_values"], result["permutation_values"])


def test_cka_self_orthogonal_scale_invariance_and_missing_neighbors():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(41, 6))
    orthogonal, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    vectors = np.stack((x, x, 4.2 * x @ orthogonal, np.ones_like(x)))
    permutations = np.stack([rng.permutation(len(x)) for _ in range(8)])
    result = cka_statistics(vectors, permutations)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        index = np.flatnonzero(np.all(result["pairs"] == (i, j), axis=1))[0]
        assert result["observed"][index] == pytest.approx(1, abs=1e-14)
    invalid = np.any(result["pairs"] == 3, axis=1)
    assert not result["valid"][invalid].any()
    assert np.isnan(result["permutation_values"][invalid]).all()
    means = head_cka_means(result, 4)
    np.testing.assert_array_equal(means["neighbor_count"], [2, 2, 2, 0])
    np.testing.assert_allclose(means["observed"][:3], 1, atol=1e-14)
    assert np.isnan(means["observed"][3]) and np.isnan(means["excess"][3])
    for head in range(3):
        rows = result["valid"] & np.any(result["pairs"] == head, axis=1)
        assert means["excess"][head] == pytest.approx(result["excess"][rows].mean())
