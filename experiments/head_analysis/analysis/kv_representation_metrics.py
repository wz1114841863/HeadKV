"""NumPy reference metrics for V2 covariance spectra and aligned Head CKA."""
from __future__ import annotations

import numpy as np

from experiments.head_analysis.common.statistics import finite_mean_count

VARIANCE_K = (8, 16, 32, 64)
NORM_STATISTICS = ("mean", "p10", "median", "p90", "max")


def spectrum_metrics(covariance) -> dict[str, np.ndarray]:
    """Analyze centered covariance; tolerate only FP32 storage-scale negatives.

    All numeric outputs are FP64. Zero total variance leaves rank and coverage
    undefined. ``min_eigenvalue`` precedes truncation, and its allowed negative
    magnitude is ``D * eps(float32) * max(abs(eigenvalues))``.
    """
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim < 2 or covariance.shape[-1] != covariance.shape[-2] or not covariance.shape[-1]:
        raise ValueError("covariance must have nonempty square final axes")
    if not np.isfinite(covariance).all():
        raise ValueError("covariance must be finite")
    eigenvalues = np.linalg.eigvalsh((covariance + covariance.swapaxes(-1, -2)) * .5)[..., ::-1]
    minimum = eigenvalues[..., -1].copy()
    tolerance = covariance.shape[-1] * np.finfo(np.float32).eps * np.max(np.abs(eigenvalues), axis=-1)
    if np.any(minimum < -tolerance):
        raise ValueError("covariance has a negative eigenvalue beyond FP32 storage tolerance")
    eigenvalues = np.maximum(eigenvalues, 0.)
    cumulative = np.cumsum(eigenvalues, axis=-1)
    total = cumulative[..., -1]
    valid = total > 0
    probabilities = np.full_like(eigenvalues, np.nan)
    np.divide(eigenvalues, total[..., None], out=probabilities, where=valid[..., None])
    log_probabilities = np.zeros_like(probabilities)
    np.log(probabilities, out=log_probabilities, where=probabilities > 0)
    entropy = -np.sum(np.where(probabilities > 0, probabilities * log_probabilities, 0.), axis=-1)
    coverage = np.full(total.shape + (len(VARIANCE_K),), np.nan)
    for index, k in enumerate(VARIANCE_K):
        np.divide(cumulative[..., min(k, eigenvalues.shape[-1]) - 1], total,
                  out=coverage[..., index], where=valid)
    fractions = np.full_like(cumulative, np.nan)
    np.divide(cumulative, total[..., None], out=fractions, where=valid[..., None])
    return {
        "eigenvalues": eigenvalues,
        "effective_rank": np.where(valid, np.exp(entropy), np.nan),
        "variance_coverage": coverage,
        "r90": np.where(valid, np.sum(fractions < .9, axis=-1) + 1., np.nan),
        "valid": valid,
        "min_eigenvalue": minimum,
        "negative_tolerance": tolerance,
    }


def _vectors(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 3 or not all(vectors.shape) or not np.isfinite(vectors).all():
        raise ValueError("vectors must be finite [Head,Token,Feature] with nonempty axes")
    return vectors


def sampled_statistics(vectors) -> dict[str, np.ndarray]:
    """Centered FP64 moments for one aligned [H,N,D] sample scope, ddof=0."""
    vectors = _vectors(vectors)
    mean = vectors.mean(axis=1)
    centered = vectors - mean[:, None, :]
    norms = np.linalg.norm(vectors, axis=-1)
    return {
        "count": np.asarray(vectors.shape[1], dtype=np.uint32),
        "mean": mean,
        "covariance": centered.swapaxes(-1, -2) @ centered / vectors.shape[1],
        "feature_energy": np.mean(vectors ** 2, axis=1),
        "norm_summary": np.stack((norms.mean(axis=1),
                                  *np.quantile(norms, [.1, .5, .9], axis=1, method="linear"),
                                  norms.max(axis=1)), axis=-1),
        "element_min": vectors.min(axis=(1, 2)),
        "element_max": vectors.max(axis=(1, 2)),
    }


def cka_statistics(vectors, permutations) -> dict[str, np.ndarray]:
    """Linear CKA and one-sided row-permutation reference via cached Token Gram.

    ``permutations[B,N]`` is supplied by the caller so layers and representation
    types can share precisely the same row correspondences. Each Head pair is
    ordered i<j; the permutation applies only to the second Head j.
    """
    vectors = _vectors(vectors)
    permutations = np.asarray(permutations)
    heads, tokens, _ = vectors.shape
    if (permutations.ndim != 2 or permutations.shape[1] != tokens or not len(permutations)
            or not np.issubdtype(permutations.dtype, np.integer)
            or not np.array_equal(np.sort(permutations, axis=1),
                                  np.broadcast_to(np.arange(tokens), permutations.shape))):
        raise ValueError("permutations must contain nonempty complete Token-row permutations")
    centered = vectors - vectors.mean(axis=1, keepdims=True)
    gram = centered @ centered.swapaxes(-1, -2)
    norms = np.linalg.norm(gram, axis=(1, 2))
    normalized = np.zeros_like(gram)
    np.divide(gram, norms[:, None, None], out=normalized, where=norms[:, None, None] > 0)
    pairs = np.asarray([(i, j) for i in range(heads) for j in range(i + 1, heads)], dtype=np.int64).reshape(-1, 2)
    left, right = pairs.T
    valid = (norms[left] > 0) & (norms[right] > 0)
    flat = normalized.reshape(heads, -1)
    observed = (flat @ flat.T)[left, right]
    reference = np.empty((len(pairs), len(permutations)), dtype=np.float64)
    for index, rows in enumerate(permutations):
        permuted = normalized[:, rows[:, None], rows[None, :]].reshape(heads, -1)
        reference[:, index] = (flat @ permuted.T)[left, right]
    # Linear CKA is in [0,1]; Gram inner products can cross an endpoint by roundoff.
    np.clip(observed, 0., 1., out=observed)
    np.clip(reference, 0., 1., out=reference)
    observed = np.where(valid, observed, np.nan)
    reference[~valid] = np.nan
    # Quantiles along a nonempty B axis also preserve shape when there are zero Head pairs.
    mean = reference.mean(axis=1)
    p05, p95 = np.quantile(reference, [.05, .95], axis=1, method="linear")
    return {
        "pairs": pairs,
        "observed": observed,
        "permutation_values": reference,
        "permutation_mean": mean,
        "permutation_p05": p05,
        "permutation_p95": p95,
        "excess": observed - mean,
        "valid": valid,
    }


def head_cka_means(cka: dict[str, np.ndarray], heads: int) -> dict[str, np.ndarray]:
    """Average valid same-layer neighbors without counting missing pairs as zero."""
    observed = np.full((heads, heads), np.nan)
    excess = np.full_like(observed, np.nan)
    valid = cka["valid"] & np.isfinite(cka["observed"]) & np.isfinite(cka["excess"])
    left, right = cka["pairs"][valid].T
    observed[left, right] = observed[right, left] = cka["observed"][valid]
    excess[left, right] = excess[right, left] = cka["excess"][valid]
    observed_mean, count = finite_mean_count(observed, axis=1, count_dtype=np.uint32)
    excess_mean, _ = finite_mean_count(excess, axis=1, count_dtype=np.uint32)
    return {"observed": observed_mean, "excess": excess_mean, "neighbor_count": count}
