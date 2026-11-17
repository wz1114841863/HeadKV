"""V3-A fixed-mean PCA fitting and reconstruction errors from existing moments."""
from __future__ import annotations

import numpy as np


RANKS = np.arange(16, 129, 16, dtype=np.uint16)


def condition_weights(conditions, source_groups=None):
    """Condition-equal, then source-equal weights; duplicate rows share source mass."""
    conditions = np.asarray(conditions)
    groups = np.arange(len(conditions)) if source_groups is None else np.asarray(source_groups)
    if conditions.ndim != 1 or not len(conditions) or groups.shape != conditions.shape:
        raise ValueError("nonempty aligned condition/source arrays required")
    weights = np.zeros(len(conditions), dtype=np.float64)
    labels = np.unique(conditions)
    for condition in labels:
        mask = conditions == condition
        sources = np.unique(groups[mask])
        for source in sources:
            rows = mask & (groups == source)
            weights[rows] = 1. / (len(labels) * len(sources) * rows.sum())
    return weights


def fit_calibration(means, covariances, conditions, source_groups=None, *, fixed_mean=None):
    """Fit per-Head means, second moments about those means, and raw energy s.

    Input axes are [Sample,...,D] and [Sample,...,D,D]. No evaluation rows
    enter this routine; the caller chooses the Calib subset before loading it.
    An explicit fixed_mean preserves a previous Calib mean while fitting G;
    the between-input mean contribution is still included.
    """
    means, covariances = np.asarray(means, np.float64), np.asarray(covariances, np.float64)
    if covariances.shape != means.shape + (means.shape[-1],):
        raise ValueError("mean/covariance shapes differ")
    if not np.isfinite(means).all() or not np.isfinite(covariances).all():
        raise ValueError("Calib moments must be finite")
    weights = condition_weights(conditions, source_groups)
    if len(weights) != len(means):
        raise ValueError("Calib labels and moments differ")
    # Reference shifting limits cancellation for large mean offsets.
    if fixed_mean is None:
        mean = means[0] + np.einsum("s,s...d->...d", weights, means - means[0])
    else:
        mean = np.asarray(fixed_mean, dtype=np.float64)
        if mean.shape != means.shape[1:] or not np.isfinite(mean).all():
            raise ValueError("fixed mean must match the finite per-Head coordinates")
    delta = means - mean
    g = (np.einsum("s,s...de->...de", weights, covariances)
         + np.einsum("s,s...d,s...e->...de", weights, delta, delta))
    g = (g + g.swapaxes(-1, -2)) * .5
    energy = np.trace(covariances, axis1=-2, axis2=-1) + (means * means).sum(-1)
    scale = np.einsum("s,s...->...", weights, energy)
    if (scale < 0).any():
        raise ValueError("Calib raw energy is negative")
    return {"mean": mean, "G": g, "scale": scale, "weights": weights,
            "valid_scale": scale > 0}


def _eigh(matrix):
    matrix = np.asarray(matrix, np.float64)
    if matrix.ndim < 2 or matrix.shape[-2] != matrix.shape[-1] or not np.isfinite(matrix).all():
        raise ValueError("finite square moment matrices required")
    values, vectors = np.linalg.eigh((matrix + matrix.swapaxes(-1, -2)) * .5)
    minimum = values[..., 0].copy()
    tolerance = matrix.shape[-1] * np.finfo(np.float32).eps * np.max(np.abs(values), axis=-1)
    if (minimum < -tolerance).any():
        raise ValueError("negative eigenvalue beyond V2 FP32 storage tolerance")
    return np.maximum(values[..., ::-1], 0), vectors[..., :, ::-1], minimum, tolerance


def fit_basis(g, scale, sharing="head"):
    """Fit full ordered bases, returning compact and per-Head broadcast forms.

    G axes are [...,Layer,Head,D,D]; an optional representation prefix is kept
    separate for Layer/global pooling. Zero-energy Heads have no shared weight.
    """
    g, scale = np.asarray(g, np.float64), np.asarray(scale, np.float64)
    if g.ndim < 4 or scale.shape != g.shape[:-2]:
        raise ValueError("G and scale require aligned Layer/Head axes")
    if sharing == "head":
        pooled = g
    elif sharing in ("layer", "global"):
        normalized = np.divide(g, scale[..., None, None], out=np.zeros_like(g), where=scale[..., None, None] > 0)
        axes = (-3,) if sharing == "layer" else (-4, -3)
        counts = (scale > 0).sum(axis=-1 if sharing == "layer" else (-2, -1))
        pooled = np.divide(normalized.sum(axis=axes), counts[..., None, None],
                           out=np.zeros(normalized.sum(axis=axes).shape), where=counts[..., None, None] > 0)
    else:
        raise ValueError(f"unknown basis sharing: {sharing}")
    eigenvalues, compact, minimum, tolerance = _eigh(pooled)
    if sharing == "head":
        basis = compact
    elif sharing == "layer":
        basis = np.broadcast_to(compact[..., :, None, :, :], g.shape)
    else:
        basis = np.broadcast_to(compact[..., None, None, :, :], g.shape)
    identity = np.eye(g.shape[-1])
    error = np.max(np.abs(compact.swapaxes(-1, -2) @ compact - identity), axis=(-2, -1))
    return {"basis": basis, "compact_basis": compact, "eigenvalues": eigenvalues,
            "orthogonality_error": error, "minimum_eigenvalue": minimum,
            "negative_tolerance": tolerance}


def _ranks(ranks, dimension):
    ranks = np.asarray(ranks)
    if (ranks.ndim != 1 or not len(ranks) or not np.issubdtype(ranks.dtype, np.integer)
            or (ranks < 1).any() or (ranks > dimension).any()):
        raise ValueError("ranks must be integer values within the feature dimension")
    return ranks.astype(np.int64)


def projection_errors(mean, covariance, fixed_mean, basis, ranks=RANKS):
    """Every-token squared L2 error e[...,Rank] for an orthonormal prefix basis.

    This is the fixed-mean trace formula, not a sampled or quantized error.
    The caller retains N separately to obtain Frobenius SSE=N*e.
    """
    mean, covariance, fixed_mean, basis = map(lambda v: np.asarray(v, np.float64),
                                             (mean, covariance, fixed_mean, basis))
    ranks = _ranks(ranks, mean.shape[-1])
    delta = mean - fixed_mean
    covariance = (covariance + covariance.swapaxes(-1, -2)) * .5
    direction_energy = np.einsum("...di,...de,...ei->...i", basis, covariance, basis, optimize=True)
    direction_energy += np.einsum("...d,...di->...i", delta, basis, optimize=True) ** 2
    total = np.trace(covariance, axis1=-2, axis2=-1) + (delta * delta).sum(-1)
    errors = total[..., None] - np.cumsum(direction_energy, axis=-1)[..., ranks - 1]
    tolerance = 1e-10 * np.maximum(np.abs(total), np.finfo(float).tiny)
    if (errors < -tolerance[..., None]).any():
        raise ValueError("projection error is negative beyond FP64 roundoff")
    return np.maximum(errors, 0.)


def oracle_errors(mean, covariance, fixed_mean, ranks=RANKS):
    """Same-mean best projection and classical per-input affine PCA lower bound."""
    mean, covariance, fixed_mean = map(lambda v: np.asarray(v, np.float64), (mean, covariance, fixed_mean))
    ranks = _ranks(ranks, mean.shape[-1])
    delta = mean - fixed_mean
    same_values = _eigh(covariance + delta[..., :, None] * delta[..., None, :])[0]
    classic_values = _eigh(covariance)[0]
    return {"same_mean": np.stack([same_values[..., r:].sum(-1) for r in ranks], -1),
            "classic": np.stack([classic_values[..., r:].sum(-1) for r in ranks], -1)}


def bf16_from_bits(bits):
    """Restore V2 uint16 BF16 bit arrays without integer numeric conversion."""
    import torch
    bits = np.asarray(bits)
    if bits.dtype != np.uint16:
        raise ValueError("V2 BF16 raw bit arrays must have uint16 dtype")
    return torch.from_numpy(bits.copy()).view(torch.bfloat16).float().numpy().astype(np.float64)


def sample_encoding_errors(vectors, fixed_mean, basis, ranks=RANKS):
    """Same-position FP64 ideal and FP32/BF16 coefficient encode/decode errors.

    Vectors axes [...,Token,D] come from original sampled BF16 values, already
    decoded numerically. The reconstructed tensor is compared in FP64 with
    those original values; neither a full Prompt nor a model output is inferred.
    """
    import torch
    vectors, fixed_mean, basis = map(lambda v: np.asarray(v, np.float64), (vectors, fixed_mean, basis))
    ranks = _ranks(ranks, vectors.shape[-1])
    if not np.isfinite(vectors).all():
        raise ValueError("sample vectors must be finite")
    x32 = torch.from_numpy(vectors.astype(np.float32))
    mu32 = torch.from_numpy(fixed_mean.astype(np.float32))
    u32 = torch.from_numpy(np.array(basis, dtype=np.float32, copy=True))
    delta64 = vectors - fixed_mean[..., None, :]
    delta32 = x32 - mu32.unsqueeze(-2)
    ideal, actual, token_p90, byte_counts = [], [], [], []
    with torch.no_grad():
        for r in ranks:
            u = basis[..., :, :r]
            reconstructed = (delta64 @ u) @ u.swapaxes(-1, -2) + fixed_mean[..., None, :]
            ideal.append(np.square(reconstructed - vectors).sum(-1).mean(-1))
            coefficients = torch.matmul(delta32, u32[..., :, :r]).to(torch.bfloat16)
            decoded = torch.matmul(coefficients.float(), u32[..., :, :r].transpose(-1, -2)) + mu32.unsqueeze(-2)
            errors = np.square(decoded.numpy().astype(np.float64) - vectors).sum(-1)
            actual.append(errors.mean(-1)); token_p90.append(np.quantile(errors, .9, axis=-1))
            byte_counts.append(coefficients.numel() * coefficients.element_size())
    shape = vectors.shape[:-2]
    per_head_bytes = np.broadcast_to(2 * vectors.shape[-2] * ranks, shape + (len(ranks),)).copy()
    return {"ideal_mse": np.stack(ideal, -1), "bf16_mse": np.stack(actual, -1),
            "bf16_token_error_p90": np.stack(token_p90, -1),
            "coefficient_bytes_per_head": per_head_bytes,
            "coefficient_bytes_total": np.asarray(byte_counts, dtype=np.uint64),
            "ranks": ranks.astype(np.uint16)}


def rope_coefficients(model_config, positions, dimension):
    """Current Llama-3 RoPE coefficients, using CPU FP32 then BF16 rounding.

    This constructs no model or inference path. CPU/GPU trigonometric rounding
    can differ, so recomputing saved GPU K_post still needs its own baseline.
    """
    import torch
    from transformers import LlamaConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    config = LlamaConfig.from_dict(dict(model_config))
    params = config.rope_parameters
    head_dimension = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    if (params["rope_type"] != "llama3" or params.get("partial_rotary_factor", 1.) != 1.
            or head_dimension != dimension or dimension % 2):
        raise ValueError("RoPE helper requires the current full-dimensional Llama-3 configuration")
    positions = np.asarray(positions)
    if positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer) or (positions < 0).any():
        raise ValueError("RoPE requires nonnegative absolute integer positions")
    with torch.no_grad():
        inv_freq, scaling = ROPE_INIT_FUNCTIONS["llama3"](config, torch.device("cpu"))
        frequencies = (inv_freq[None, :, None].float()
                       @ torch.tensor(positions[None, None], dtype=torch.float32)).transpose(1, 2)
        embedding = torch.cat((frequencies, frequencies), -1)
        cos = (embedding.cos() * scaling).to(torch.bfloat16).float().numpy()[0].astype(np.float64)
        sin = (embedding.sin() * scaling).to(torch.bfloat16).float().numpy()[0].astype(np.float64)
    return cos, sin


def apply_position_rope(vectors, cos, sin, *, inverse=False, bf16=False):
    """Apply per-position half rotations or their true FP64 linear inverse.

    The inverse divides by c²+s² because saved BF16 coefficients are not exactly
    unit norm. It cannot undo the later BF16 product/add rounding of K_post.
    """
    vectors, cos, sin = map(lambda v: np.asarray(v, np.float64), (vectors, cos, sin))
    dimension = vectors.shape[-1]
    if dimension % 2 or cos.shape != vectors.shape[-2:] or sin.shape != cos.shape:
        raise ValueError("RoPE coefficients must match the Token/feature axes")
    if inverse and bf16:
        raise ValueError("inverse and BF16 forward rounding are distinct paths")
    if bf16:
        import torch
        x = torch.tensor(vectors, dtype=torch.bfloat16)
        c, s = (torch.tensor(a, dtype=torch.bfloat16) for a in (cos, sin))
        rotated = torch.cat((-x[..., dimension // 2:], x[..., :dimension // 2]), -1)
        return (x * c + rotated * s).float().numpy().astype(np.float64)
    rotated = np.concatenate((-vectors[..., dimension // 2:], vectors[..., :dimension // 2]), -1)
    if inverse:
        denominator = cos * cos + sin * sin
        if (denominator <= 0).any():
            raise ValueError("RoPE inverse coefficients are singular")
        return (vectors * cos - rotated * sin) / denominator
    return vectors * cos + rotated * sin
