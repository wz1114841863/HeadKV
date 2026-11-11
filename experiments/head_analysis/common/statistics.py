"""Existing statistical primitives shared without changing aggregation policies."""

from __future__ import annotations

import hashlib

import numpy as np


def finite_mean_count(values, axis=0, *, dtype=np.float64, count_dtype=None):
    valid = np.isfinite(values)
    count = valid.sum(axis=axis, dtype=count_dtype)
    result = np.full(count.shape, np.nan, dtype=dtype)
    np.divide(np.where(valid, values, 0).sum(axis=axis, dtype=np.float64), count, out=result, where=count > 0)
    return result, count


def finite_mean(values, axis=0, *, dtype=np.float64):
    return finite_mean_count(values, axis, dtype=dtype)[0]


def average_ranks(values):
    """Zero-based average ties along the final axis; NaN remains missing."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    safe = np.where(valid, values, np.inf)
    order = np.argsort(safe, axis=-1, kind="stable")
    sorted_values = np.take_along_axis(safe, order, axis=-1)
    positions = np.broadcast_to(np.arange(values.shape[-1]), values.shape)
    starts = np.ones(values.shape, dtype=bool)
    ends = np.ones(values.shape, dtype=bool)
    starts[..., 1:] = sorted_values[..., 1:] != sorted_values[..., :-1]
    ends[..., :-1] = sorted_values[..., 1:] != sorted_values[..., :-1]
    first = np.maximum.accumulate(np.where(starts, positions, 0), axis=-1)
    last = np.minimum.accumulate(np.where(ends, positions, values.shape[-1])[..., ::-1], axis=-1)[..., ::-1]
    result = np.empty_like(values)
    np.put_along_axis(result, order, (first + last) * 0.5, axis=-1)
    return np.where(valid, result, np.nan)


def pearson(left, right):
    """Batched, pairwise-complete correlation, returning valid coordinate counts."""
    valid = np.isfinite(left) & np.isfinite(right)
    count = valid.sum(-1)
    left, right = np.where(valid, left, np.nan), np.where(valid, right, np.nan)
    dx = np.where(valid, left - finite_mean(left, -1)[..., None], 0)
    dy = np.where(valid, right - finite_mean(right, -1)[..., None], 0)
    denominator = np.sqrt((dx**2).sum(-1) * (dy**2).sum(-1))
    varies_x = np.max(np.where(valid, left, -np.inf), axis=-1) > np.min(np.where(valid, left, np.inf), axis=-1)
    varies_y = np.max(np.where(valid, right, -np.inf), axis=-1) > np.min(np.where(valid, right, np.inf), axis=-1)
    result = np.full(count.shape, np.nan, dtype=np.float64)
    np.divide((dx * dy).sum(-1), denominator, out=result, where=(count >= 3) & (denominator > 0) & varies_x & varies_y)
    return np.clip(result, -1, 1), count.astype(np.uint32)


def correlation_1d(x, y):
    """Retain the original B1 scalar dot-product arithmetic and return ordering."""
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid].astype(np.float64, copy=False), y[valid].astype(np.float64, copy=False)
    count = len(x)
    if count < 3:
        return count, float("nan"), float("nan")

    def coefficient(left, right):
        left, right = left - left.mean(), right - right.mean()
        denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
        return float(np.dot(left, right) / denominator) if denominator > 0 else float("nan")

    return count, coefficient(average_ranks(x), average_ranks(y)), coefficient(x, y)


def percentile_ci(values):
    """95% percentile intervals over axis zero, with actual valid replicate counts."""
    values = np.asarray(values, dtype=np.float64)
    shape = values.shape[1:]
    flat = values.reshape(len(values), -1)
    count = np.isfinite(flat).sum(0)
    result = np.full((flat.shape[1], 2), np.nan)
    usable = count > 0
    if usable.any():
        result[usable] = np.nanquantile(flat[:, usable], [0.025, 0.975], axis=0, method="linear").T
    return result.reshape(shape + (2,)).astype(np.float32), count.reshape(shape).astype(np.uint32)


def seed_from_label(seed, label):
    # This digest is part of the frozen random-number sequence, not an integrity check.
    return int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:8], "little")


def bootstrap_draws(seed, label, replicates, sample_count):
    if replicates <= 0 or sample_count <= 0:
        raise ValueError("positive bootstrap replicate/sample counts required")
    rng = np.random.Generator(np.random.PCG64(seed_from_label(seed, label)))
    return rng.integers(0, sample_count, size=(replicates, sample_count))


def bootstrap_means(values, draws):
    """B2 weighted cluster means, preserving its original FP64 evaluation order."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    weights = np.zeros((len(draws), n), dtype=np.float64)
    np.add.at(weights, (np.arange(len(draws))[:, None], draws), 1)
    flat = values.reshape(n, -1)
    finite = np.isfinite(flat)
    numerator = weights @ np.where(finite, flat, 0)
    denominator = weights @ finite.astype(np.float64)
    result = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result.reshape((len(draws),) + values.shape[1:])


def bootstrap_sample_mean_ci(values, replicates, rng):
    """Sample-mean percentile CI with the original B1 draw and arithmetic order."""
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    flat = values.reshape(len(values), -1).astype(np.float64, copy=False)
    output = np.full((flat.shape[1], 2), np.nan, dtype=np.float32)
    for column in np.flatnonzero(np.isfinite(flat).any(axis=0)):
        selected = flat[draws, column]
        valid = np.isfinite(selected)
        counts = valid.sum(axis=1)
        means = np.full(replicates, np.nan, dtype=np.float64)
        np.divide(np.where(valid, selected, 0.0).sum(axis=1), counts, out=means, where=counts > 0)
        means = means[np.isfinite(means)]
        if means.size:
            output[column] = np.quantile(means, (0.025, 0.975))
    return output.reshape((*values.shape[1:], 2))
