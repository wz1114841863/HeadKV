import numpy as np
import pytest

from experiments.head_analysis.analysis.pca_reconstruction import (
    apply_position_rope, bf16_from_bits, condition_weights, fit_basis, fit_calibration,
    oracle_errors, projection_errors, rope_coefficients, sample_encoding_errors,
)


def moments(x):
    mean = x.mean(-2)
    centered = x - mean[..., None, :]
    return mean, centered.swapaxes(-1, -2) @ centered / x.shape[-2]


def test_calibration_preserves_source_weights_and_between_input_mean_energy():
    means = np.array([0., 4., 10.])[:, None, None, None] * np.array([1., 0., 0.])
    cov = np.broadcast_to(np.diag([1., 2., 3.]), (3, 1, 1, 3, 3)).copy()
    fitted = fit_calibration(means, cov, ["a", "a", "b"])
    np.testing.assert_array_equal(fitted["weights"], [.25, .25, .5])
    np.testing.assert_allclose(fitted["mean"], [[[6., 0., 0.]]])
    # E[mu²]-E[mu]² = 54-36, not zero as averaging only C would imply.
    np.testing.assert_allclose(fitted["G"], [[np.diag([19., 2., 3.])]])
    np.testing.assert_allclose(fitted["scale"], [[60.]])
    frozen = fit_calibration(means, cov, ["a", "a", "b"], fixed_mean=np.zeros((1, 1, 3)))
    np.testing.assert_array_equal(frozen["mean"], np.zeros((1, 1, 3)))
    np.testing.assert_allclose(frozen["G"], [[np.diag([55., 2., 3.])]])
    np.testing.assert_array_equal(frozen["scale"], fitted["scale"])
    np.testing.assert_allclose(condition_weights(["a", "a", "a", "b"], ["x", "x", "y", "z"]),
                               [.125, .125, .25, .5])


def test_shared_bases_normalize_head_energy_and_preserve_representation_axis():
    g = np.array([[[[100., 0.], [0., 0.]], [[0., 0.], [0., 2.]]]])
    scale = np.array([[100., 1.]])
    layer = fit_basis(g, scale, "layer")
    # Unnormalized pooling would choose dimension 0; normalized objective chooses 1.
    np.testing.assert_allclose(np.abs(layer["compact_basis"][0, :, 0]), [0, 1])
    assert layer["basis"].shape == (1, 2, 2, 2)
    global_fit = fit_basis(np.stack((g, g[..., ::-1, ::-1])), np.stack((scale, scale)), "global")
    assert global_fit["compact_basis"].shape == (2, 2, 2)
    np.testing.assert_allclose(np.max(global_fit["orthogonality_error"]), 0)
    empty = fit_basis(np.zeros_like(g), np.zeros_like(scale), "global")
    assert np.isfinite(empty["basis"]).all()


def test_analytic_errors_match_vectors_oracles_and_nested_full_rank():
    rng = np.random.default_rng(81)
    x = rng.normal(size=(2, 3, 31, 6)) * np.arange(1, 7)
    mean, covariance = moments(x)
    fixed = mean + rng.normal(size=mean.shape)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(2, 3, 6, 6)))
    ranks = np.array([1, 2, 4, 6])
    analytic = projection_errors(mean, covariance, fixed, orthogonal, ranks)
    for k, r in enumerate(ranks):
        u = orthogonal[..., :r]
        reconstructed = ((x - fixed[..., None, :]) @ u) @ u.swapaxes(-1, -2) + fixed[..., None, :]
        expected = np.square(x - reconstructed).sum(-1).mean(-1)
        np.testing.assert_allclose(analytic[..., k], expected, atol=1e-12)
    assert np.max(analytic[..., -1]) < 1e-10
    assert (np.diff(analytic, axis=-1) <= 1e-10).all()
    oracle = oracle_errors(mean, covariance, fixed, ranks)
    assert (oracle["classic"] <= oracle["same_mean"] + 1e-12).all()
    assert (oracle["same_mean"] <= analytic + 1e-12).all()
    np.testing.assert_allclose(oracle["same_mean"][..., -1], 0)


def test_mean_matched_oracle_is_not_classic_pca_when_means_shift():
    mean = np.array([10., 0.])
    covariance = np.diag([0., 4.])
    oracle = oracle_errors(mean, covariance, np.zeros(2), [1, 2])
    np.testing.assert_allclose(oracle["classic"], [0., 0.])
    np.testing.assert_allclose(oracle["same_mean"], [4., 0.])


def test_psd_tolerance_matches_v2_and_material_negative_is_rejected():
    g = np.diag([3., 1., -1e-8])[None, None]
    result = fit_basis(g, np.ones((1, 1)))
    assert result["minimum_eigenvalue"][0, 0] == pytest.approx(-1e-8)
    assert result["negative_tolerance"][0, 0] == pytest.approx(3 * np.finfo(np.float32).eps * 3)
    with pytest.raises(ValueError, match="negative eigenvalue"):
        fit_basis(np.diag([3., 1., -.1])[None, None], np.ones((1, 1)))


def test_original_bf16_bits_and_actual_encoding_are_distinct_from_ideal():
    import torch
    rng = np.random.default_rng(13)
    original = torch.from_numpy(rng.normal(size=(2, 19, 4)).astype(np.float32)).to(torch.bfloat16)
    bits = original.view(torch.uint16).numpy()
    x = bf16_from_bits(bits)
    np.testing.assert_array_equal(x, original.float().numpy())
    mean, covariance = moments(x)
    u = np.linalg.eigh(covariance)[1][..., ::-1]
    result = sample_encoding_errors(x, mean, u, [1, 2, 4])
    analytic = projection_errors(mean, covariance, mean, u, [1, 2, 4])
    np.testing.assert_allclose(result["ideal_mse"], analytic, atol=1e-13)
    np.testing.assert_array_equal(result["coefficient_bytes_total"], [2 * 2 * 19 * r for r in (1, 2, 4)])
    assert (result["bf16_mse"][..., -1] > 0).all()
    assert (result["ideal_mse"][..., -1] < 1e-25).all()
    # Independent CPU reference for the actual coefficient storage precision.
    delta = original.float() - torch.tensor(mean, dtype=torch.float32)[:, None]
    coeff = (delta @ torch.tensor(u[..., :2].copy(), dtype=torch.float32)).bfloat16()
    restored = coeff.float() @ torch.tensor(u[..., :2].copy(), dtype=torch.float32).transpose(-1, -2)
    restored += torch.tensor(mean, dtype=torch.float32)[:, None]
    error = ((restored.double() - torch.tensor(x)) ** 2).sum(-1).mean(-1)
    # NumPy and Torch use different FP64 reduction orders after identical decode.
    np.testing.assert_allclose(result["bf16_mse"][..., 1], error.numpy(), rtol=1e-14, atol=1e-14)


def test_rope_coefficients_follow_transformers_and_inverse_is_position_specific():
    import torch
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    configuration = {"hidden_size": 32, "num_attention_heads": 4, "num_key_value_heads": 2,
        "max_position_embeddings": 32768, "rope_theta": 500000.,
        "rope_scaling": {"rope_type": "llama3", "factor": 8., "low_freq_factor": 1.,
                         "high_freq_factor": 4., "original_max_position_embeddings": 8192}}
    positions = np.array([0, 64, 1001, 8192, 32000])
    cos, sin = rope_coefficients(configuration, positions, 8)
    module = LlamaRotaryEmbedding(LlamaConfig.from_dict(configuration), device="cpu")
    c, s = module(torch.empty(1, dtype=torch.bfloat16), torch.tensor(positions[None]))
    np.testing.assert_array_equal(cos, c[0].float().numpy())
    np.testing.assert_array_equal(sin, s[0].float().numpy())
    rng = np.random.default_rng(24)
    vectors = rng.normal(size=(2, len(positions), 8))
    rotated = apply_position_rope(vectors, cos, sin)
    np.testing.assert_allclose(apply_position_rope(rotated, cos, sin, inverse=True), vectors, atol=1e-12)
    assert not np.allclose(rotated[:, 1], apply_position_rope(vectors[:, 1:2], cos[:1], sin[:1])[:, 0])
    # BF16 cos/sin are not exactly orthogonal, so transpose alone is not inverse.
    half = np.concatenate((-rotated[..., 4:], rotated[..., :4]), -1)
    naive = rotated * cos - half * sin
    assert np.max(np.abs(naive - vectors)) > 1e-5
    x = torch.tensor(vectors, dtype=torch.bfloat16)
    half = torch.cat((-x[..., 4:], x[..., :4]), -1)
    expected = (x * c[0] + half * s[0]).float().numpy()
    np.testing.assert_array_equal(apply_position_rope(vectors, cos, sin, bf16=True), expected)
    # Rounding after the rotation loses information; the linear inverse is no repair.
    inverse_rounded = apply_position_rope(expected, cos, sin, inverse=True)
    assert not np.array_equal(inverse_rounded, x.float().numpy())
    with pytest.raises(ValueError, match="distinct paths"):
        apply_position_rope(vectors, cos, sin, inverse=True, bf16=True)
