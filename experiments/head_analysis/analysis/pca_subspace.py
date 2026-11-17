"""Principal-subspace geometry and signed A1 Oracle-gap diagnostics."""
from __future__ import annotations

import numpy as np

from experiments.head_analysis.analysis.pca_reconstruction import _eigh


RANKS = np.array([16, 32, 64], dtype=np.int64)
ANGLE_ATOL_RAD = 1e-6
OVERLAP_ATOL = 1e-10
ERROR_ATOL = 1e-10


def principal_geometry(left, right, ranks=RANKS, *, keep_angles=False):
    """Compare orthonormal feature bases; angles are ascending, in radians."""
    result = {}
    overlap, median, maximum = [], [], []
    for rank in ranks:
        cross = left[..., :, :rank].swapaxes(-1, -2) @ right[..., :, :rank]
        singular = np.linalg.svd(cross, compute_uv=False)
        if not np.isfinite(singular).all() or np.any(singular > 1 + OVERLAP_ATOL):
            raise ValueError("principal-angle singular values exceed orthonormal-basis tolerance")
        singular = np.clip(singular, 0, 1)
        angles = np.arccos(singular)
        overlap.append(np.square(singular).mean(-1))
        median.append(np.median(angles, axis=-1))
        maximum.append(angles[..., -1])
        if keep_angles:
            result[f"angles_r{rank}"] = angles
    result.update(overlap=np.stack(overlap, -1), angle_median=np.stack(median, -1),
                  angle_max=np.stack(maximum, -1))
    return result


def spectral_boundary(values, ranks=RANKS):
    """Descending spectrum -> [...,Rank,(lambda_r,lambda_next,gap)]."""
    ranks = np.asarray(ranks)
    if np.any(ranks < 1) or np.any(ranks >= values.shape[-1]):
        raise ValueError("spectral boundary requires 1 <= rank < feature dimension")
    current, following = values[..., ranks - 1], values[..., ranks]
    return np.stack((current, following, current - following), -1)


def check_nonnegative(values, label, *, tolerance=ERROR_ATOL):
    """Retain signed values; flag roundoff negatives and reject larger negatives."""
    values = np.asarray(values)
    bad = np.isfinite(values) & (values < -tolerance)
    if bad.any():
        coordinate = np.unravel_index(np.nanargmin(values), values.shape)
        raise ValueError(f"{label}: negative normalized loss {values[coordinate]:.17g} "
                         f"at {coordinate}, tolerance={tolerance}")
    return np.isfinite(values) & (values < 0)


def normalized(values, scale):
    return np.divide(values, scale[..., None], out=np.full_like(values, np.nan),
                     where=scale[..., None] > 0)


def check_identity(residual, label):
    valid = np.asarray(residual)[np.isfinite(residual)]
    if valid.size and np.max(np.abs(valid)) > ERROR_ATOL:
        raise ValueError(f"{label}: normalized residual {np.max(np.abs(valid)):.17g}")


def captured_energy(matrix, basis, ranks=RANKS):
    u = basis[..., :, :int(np.max(ranks))]
    directional = (u * (matrix @ u)).sum(-2)
    return np.cumsum(directional, axis=-1)[..., np.asarray(ranks) - 1]


def diagnose_moments(means, covariances, fixed_mean, fixed_basis, scale, centered_basis,
                     ranks=RANKS):
    """Axes [Sample,Representation,Head,...]; returns small results and transient bases."""
    covariances = np.asarray(covariances, np.float64)
    covariances = (covariances + covariances.swapaxes(-1, -2)) / 2
    delta = np.asarray(means, np.float64) - fixed_mean
    matrix = covariances + delta[..., :, None] * delta[..., None, :]
    values_m, u_m, minimum_m, _ = _eigh(matrix)
    values_c, u_c, minimum_c, _ = _eigh(covariances)
    u_m, u_c = np.ascontiguousarray(u_m), np.ascontiguousarray(u_c)
    optimal_capture = captured_energy(matrix, u_m, ranks)
    fixed_capture = captured_energy(matrix, fixed_basis, ranks)
    fixed_error = np.trace(matrix, axis1=-2, axis2=-1)[..., None] - fixed_capture
    oracle = np.stack([values_m[..., int(r):].sum(-1) for r in ranks], -1)
    gap = normalized(fixed_error - oracle, scale)
    data = {"fixed_mse": fixed_error, "oracle_mse": oracle, "gap": gap,
            "fixed_normalized_mse": normalized(fixed_error, scale),
            "oracle_normalized_mse": normalized(oracle, scale),
            "gap_roundoff_negative": check_nonnegative(gap, "fixed/Oracle gap"),
            "boundary_M": spectral_boundary(values_m, ranks),
            "boundary_C": spectral_boundary(values_c, ranks),
            "minimum_eigenvalue_M": minimum_m, "minimum_eigenvalue_C": minimum_c}
    for prefix, left, right, keep in (
        ("main", fixed_basis, u_m, True), ("centered", centered_basis, u_c, False),
        ("mean_shift", u_m, u_c, False),
    ):
        data.update({f"{prefix}_{key}": value for key, value in
                     principal_geometry(left, right, ranks, keep_angles=keep).items()})
    data["gap_covariance"] = normalized(captured_energy(covariances, u_m, ranks)
                                      - captured_energy(covariances, fixed_basis, ranks), scale)
    def mean_capture(basis):
        powers = np.square(np.einsum("...d,...dr->...r", delta, basis[..., :, :int(max(ranks))]))
        return np.cumsum(powers, -1)[..., np.asarray(ranks) - 1]
    data["gap_mean"] = normalized(mean_capture(u_m) - mean_capture(fixed_basis), scale)
    cross = u_m.swapaxes(-1, -2) @ fixed_basis[..., :, :int(max(ranks))]
    captured_fraction = np.cumsum(np.square(cross), axis=-1)
    lost, recovered = [], []
    for rank in ranks:
        fraction = captured_fraction[..., int(rank) - 1]
        lost.append((values_m[..., :rank] * (1 - fraction[..., :rank])).sum(-1))
        recovered.append((values_m[..., rank:] * fraction[..., rank:]).sum(-1))
    data["missed_top_energy"] = normalized(np.stack(lost, -1), scale)
    data["recovered_tail_energy"] = normalized(np.stack(recovered, -1), scale)
    data["mean_decomposition_residual"] = data["gap_covariance"] + data["gap_mean"] - gap
    data["energy_decomposition_residual"] = data["missed_top_energy"] - data["recovered_tail_energy"] - gap
    data["oracle_trace_residual"] = normalized(optimal_capture - fixed_capture, scale) - gap
    for key in ("mean_decomposition_residual", "energy_decomposition_residual", "oracle_trace_residual"):
        check_identity(data[key], key)
    return data, {"M": matrix, "M_basis": u_m, "C_basis": u_c,
                  "optimal_capture": optimal_capture}


def diagnose_pairs(transient, pairs, scale, ranks=RANKS):
    """Unordered pairs, with both directional losses, not clipped to zero."""
    left, right = np.asarray(pairs, dtype=np.int64).T
    data = {}
    for kind in ("M", "C"):
        basis = transient[f"{kind}_basis"]
        data.update({f"{kind}_{key}": value for key, value in
                     principal_geometry(basis[left], basis[right], ranks).items()})
    for key, target, source in (("forward_gap", left, right), ("reverse_gap", right, left)):
        candidate = captured_energy(transient["M"][target], transient["M_basis"][source], ranks)
        data[key] = normalized(transient["optimal_capture"][target] - candidate, scale)
        data[f"{key}_roundoff_negative"] = check_nonnegative(data[key], key)
    return data
