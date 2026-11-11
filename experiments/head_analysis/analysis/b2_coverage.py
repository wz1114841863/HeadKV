"""B2 stage 2: exact, sample-equal Prefill and Decode coverage summaries.

No model execution or correlation inference lives here. A sample's derived
trajectories use temporary NPY memory maps for exact quantiles; only compact
sample/head summaries and Recall time bins survive the sample boundary.
"""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from experiments.head_analysis.analysis.b2_consistency import B2RunSetIndex, B2SourceError
from experiments.head_analysis.common.statistics import finite_mean_count as _mean

STATISTICS = ("mean", "median", "p10", "p90")
RECALL_SCOPES = ("all_visible", "prompt_conditional")
SET_TYPES = ("learned_middle", "augmented_prompt")
PREFILL_FEATURES = ("cv", "frequency_mean", "frequency_std", "binary_threshold", "nonzero_fraction")
DISTRIBUTION_STATISTICS = ("mean", "std", "p10", "p25", "median", "p75", "p90", "min", "max")
TIME_METRICS = ("slope", "normalized_auc", "last_minus_first_quarter")
TOLERANCE = 1e-5


def _require(value, message):
    if not value:
        raise B2SourceError(message)


def _divide(numerator, denominator):
    shape = np.broadcast_shapes(np.shape(numerator), np.shape(denominator))
    result = np.full(shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=np.asarray(denominator) > 0)
    return result


def _scopes(mass, prompt):
    return np.stack((mass, _divide(mass, prompt)), axis=-1)


def derive_coverage_metrics(
    prompt_mass: np.ndarray,
    critical_mass: np.ndarray,
    valid_counts: np.ndarray,
    candidate_counts: np.ndarray,
    budgets: tuple[int, ...],
    fixed_token_counts: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """Derive per-Step metrics from full-visible masses, using FP64 arithmetic.

    Inputs: P=[C,L,H], A=[C,L,H,M,B,U], counts=[L,H,M],
    candidates/fixed sizes=[M]. No input array is changed.
    """

    p = np.asarray(prompt_mass, dtype=np.float64)
    a = np.asarray(critical_mass, dtype=np.float64)
    counts = np.asarray(valid_counts)
    candidates = np.asarray(candidate_counts)
    fixed_sizes = np.asarray(fixed_token_counts)
    _require(p.ndim == 3 and a.ndim == 6 and a.shape[:3] == p.shape, "coverage mass shapes differ")
    m, b, u = a.shape[3:]
    _require(
        u == 2 and b == len(budgets) and b >= 2 and list(budgets) == sorted(set(budgets)),
        "coverage budget/set axes differ",
    )
    _require(
        counts.shape == p.shape[1:] + (m,) and candidates.shape == fixed_sizes.shape == (m,),
        "coverage count shapes differ",
    )
    _require(
        np.issubdtype(counts.dtype, np.integer)
        and np.issubdtype(candidates.dtype, np.integer)
        and np.issubdtype(fixed_sizes.dtype, np.integer),
        "Token counts must be integers",
    )
    _require(
        (candidates > 0).all()
        and (fixed_sizes >= 0).all()
        and (counts >= 0).all()
        and (counts <= candidates).all()
        and (counts <= budgets[-1]).all(),
        "invalid candidate/ranking counts",
    )
    _require(
        np.isfinite(p).all()
        and np.isfinite(a).all()
        and (p >= 0).all()
        and (p <= 1 + TOLERANCE).all()
        and (a >= 0).all()
        and (a <= p[..., None, None, None] + TOLERANCE).all(),
        "invalid Attention masses",
    )
    _require((np.diff(a, axis=-2) >= -TOLERANCE).all(), "non-monotonic budget coverage")
    difference = a[..., 1] - a[..., 0]
    _require((difference >= 0).all(), "negative fixed-region mass")
    fixed = difference[..., -1]
    fixed_error = np.abs(difference - fixed[..., None])
    _require((fixed_error <= TOLERANCE).all(), "fixed-region mass depends on budget")
    middle = p[..., None] - fixed
    _require((middle >= -TOLERANCE).all(), "fixed-region mass exceeds Prompt mass")
    clipped = middle < 0
    clipped_max = float(np.max(np.where(clipped, -middle, 0), initial=0))
    middle = np.maximum(middle, 0)
    effective = np.minimum(counts[..., None], np.asarray(budgets))
    sizes = np.stack((effective, effective + fixed_sizes[None, None, :, None]), axis=-1).astype(np.uint32)
    uniform_middle = middle[..., None] * (effective / candidates[None, None, :, None])[None, ...]
    uniform = np.stack((uniform_middle, uniform_middle + fixed[..., None]), axis=-1)
    excess = a[..., 0] - uniform_middle
    gains = np.diff(a, axis=-2)
    added = np.diff(effective, axis=-1)
    per_token = _divide(gains, added[None, ..., None])
    controls = np.stack((fixed, middle), axis=-1)
    metrics = {
        "recall": _scopes(a, p[..., None, None, None]),
        "uniform_expected_recall": _scopes(uniform, p[..., None, None, None]),
        "learned_excess": _scopes(excess, p[..., None, None]),
        "budget_gain": _scopes(gains, p[..., None, None, None]),
        "gain_per_added_token": _scopes(per_token, p[..., None, None, None]),
        "prompt_mass": p,
        "fixed_middle_mass": _scopes(controls, p[..., None, None]),
    }
    return (
        metrics,
        sizes,
        {
            "middle_mass_clipped_count": int(clipped.sum()),
            "middle_mass_max_negative_magnitude": clipped_max,
            "fixed_mass_max_budget_difference": float(np.max(fixed_error, initial=0)),
        },
    )


def summarize_trajectory(
    values: np.ndarray, *, column_block: int = 128, include_time: bool = True
) -> dict[str, np.ndarray]:
    """Exact temporal statistics; at most [C,column_block] is copied to RAM."""

    _require(values.ndim >= 1 and column_block > 0, "invalid trajectory or column block")
    length = values.shape[0]
    tail = values.shape[1:]
    columns = int(np.prod(tail, dtype=np.int64))
    flat = values.reshape(length, columns)
    full = np.full((columns, 4), np.nan, dtype=np.float64)
    valid_count = np.zeros(columns, dtype=np.uint32)
    quarters = np.full((4, columns), np.nan, dtype=np.float64)
    quarter_count = np.zeros((4, columns), dtype=np.uint32)
    bins = np.full((20, columns), np.nan, dtype=np.float64)
    bin_count = np.zeros((20, columns), dtype=np.uint32)
    temporal = np.full((columns, 3), np.nan, dtype=np.float64)
    temporal_count = np.zeros((columns, 3), dtype=np.uint32)
    long = include_time and length >= 32
    for start in range(0, columns, column_block):
        end = min(columns, start + column_block)
        block = np.asarray(flat[:, start:end], dtype=np.float64)
        finite = np.isfinite(block)
        count = finite.sum(0)
        valid_count[start:end] = count
        sums = np.where(finite, block, 0).sum(0, dtype=np.float64)
        full[start:end, 0] = _divide(sums, count)
        complete_columns = (count == length) & (count > 0)
        if complete_columns.any():
            full[start:end, 1:][complete_columns] = np.quantile(
                block[:, complete_columns], [0.5, 0.1, 0.9], axis=0, method="linear"
            ).T
        incomplete_columns = (count > 0) & (count < length)
        if incomplete_columns.any():
            full[start:end, 1:][incomplete_columns] = np.nanquantile(
                np.where(finite[:, incomplete_columns], block[:, incomplete_columns], np.nan),
                [0.5, 0.1, 0.9],
                axis=0,
                method="linear",
            ).T
        if not long:
            continue
        for output, output_counts, parts in ((quarters, quarter_count, 4), (bins, bin_count, 20)):
            ids = np.minimum(parts - 1, parts * np.arange(length) // length)
            for i in range(parts):
                mask = ids == i
                n = finite[mask].sum(0)
                output[i, start:end] = _divide(np.where(finite[mask], block[mask], 0).sum(0), n)
                output_counts[i, start:end] = n
        u = np.arange(length, dtype=np.float64) / (length - 1)
        mean_u = _divide((finite * u[:, None]).sum(0), count)
        mean_x = full[start:end, 0]
        du = u[:, None] - mean_u
        numerator = np.where(finite, du * (block - mean_x), 0).sum(0)
        denominator = np.where(finite, du**2, 0).sum(0)
        temporal[start:end, 0] = _divide(numerator, denominator)
        temporal_count[start:end, 0] = np.where(count >= 2, count, 0)
        complete = count == length
        if complete.any():
            temporal[start:end, 1][complete] = np.trapezoid(block[:, complete], u, axis=0)
            temporal_count[start:end, 1][complete] = length
        both = (quarter_count[0, start:end] > 0) & (quarter_count[3, start:end] > 0)
        temporal[start:end, 2][both] = quarters[3, start:end][both] - quarters[0, start:end][both]
        temporal_count[start:end, 2][both] = quarter_count[0, start:end][both] + quarter_count[3, start:end][both]
    return {
        "full": full.reshape(tail + (4,)).astype(np.float32),
        "valid_count": valid_count.reshape(tail),
        "quarters": quarters.reshape((4,) + tail).astype(np.float32),
        "quarter_count": quarter_count.reshape((4,) + tail),
        "bins": bins.reshape((20,) + tail).astype(np.float32),
        "bin_count": bin_count.reshape((20,) + tail),
        "temporal": temporal.reshape(tail + (3,)).astype(np.float32),
        "temporal_count": temporal_count.reshape(tail + (3,)),
    }


def _distribution(values, axis):
    _require(np.isfinite(values).all(), "Prefill distribution contains invalid values")
    return np.stack(
        (
            np.mean(values, axis=axis, dtype=np.float64),
            np.std(values, axis=axis, dtype=np.float64),
            *np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9], axis=axis, method="linear"),
            np.min(values, axis=axis),
            np.max(values, axis=axis),
        ),
        axis=-1,
    ).astype(np.float32)


@dataclass(frozen=True)
class B2CoverageSummary:
    arrays: Mapping[str, np.ndarray]
    axes: Mapping[str, list[str]]
    diagnostics: Mapping[str, object]


@contextmanager
def _checked_npz(path: Path, entries):
    _require(path.name in entries, f"stage2 source artifact missing from manifest: {path}")
    with np.load(path, allow_pickle=False) as archive:
        yield archive


def summarize_b2_coverage(
    index: B2RunSetIndex,
    *,
    scratch_dir: Path | str | None = None,
    on_sample: Callable[[str, int, int], None] | None = None,
) -> B2CoverageSummary:
    """Read one sample/chunk at a time, then aggregate sample-equal profiles."""

    contract = index.payload["execution_contract"]
    budgets = tuple(contract["axes"]["budgets"])
    conditions = tuple(contract["axes"]["conditions"])
    condition_names = tuple(sorted(index.condition_counts))
    samples = list(index.samples.values())
    dims = contract["dimensions"]
    layers, heads, m, b = dims["L"], dims["Hkv"], len(conditions), len(budgets)
    n = len(samples)
    _require(n > 0, "no samples to summarize")
    arrays = {}
    axes = {}
    features = np.empty((n, layers, heads, m, len(PREFILL_FEATURES)), dtype=np.float32)
    sizes_all = np.empty((n, layers, heads, m, b, 2), dtype=np.uint32)
    candidates_all = np.empty((n, m), dtype=np.uint32)
    fixed_sizes = np.asarray(
        [c["initial_exclusion"] + c["recent_exclusion"] for c in contract["conditions"]], dtype=np.uint32
    )
    sample_axes = {
        "recall": ["L", "Hkv", "M", "B", "U", "R"],
        "uniform_expected_recall": ["L", "Hkv", "M", "B", "U", "R"],
        "learned_excess": ["L", "Hkv", "M", "B", "R"],
        "budget_gain": ["L", "Hkv", "M", "D_budget", "U", "R"],
        "gain_per_added_token": ["L", "Hkv", "M", "D_budget", "U", "R"],
        "prompt_mass": ["L", "Hkv"],
        "fixed_middle_mass": ["L", "Hkv", "M", "Control", "R"],
    }
    diagnostics = {
        "middle_mass_clipped_count": 0,
        "middle_mass_max_negative_magnitude": 0.0,
        "fixed_mass_max_budget_difference": 0.0,
        "long_sample_count": 0,
        "short_sample_count": 0,
    }
    short = []
    short_offsets = [0]
    batches = {batch.batch_id: batch for batch in index.batches}
    for sample_number, sample in enumerate(samples):
        batch = batches[sample.batch_id]
        manifest_content = (batch.run_dir / "file_manifest.json").read_bytes()
        entries = {
            Path(e["relative_path"]).name: e
            for e in json.loads(manifest_content)["files"]
            if e.get("sample_id") == sample.sample_id
        }
        with _checked_npz(sample.sample_dir / "prefill_consistency.npz", entries) as prefill:
            candidates = prefill["candidate_token_count"]
            counts = prefill["critical_token_valid_count"]
            features[sample_number] = np.stack(
                (
                    prefill["cv_prefill"],
                    prefill["frequency_mean"],
                    prefill["frequency_std"],
                    prefill["binary_threshold"],
                    prefill["nonzero_frequency_count"] / candidates,
                ),
                axis=-1,
            )
            candidates_all[sample_number] = candidates
        with tempfile.TemporaryDirectory(prefix="headkv-b2-coverage-", dir=scratch_dir) as temporary:
            buffers = {}
            cursor = 0
            for chunk in sample.decode_chunks:
                with _checked_npz(chunk, entries) as archive:
                    steps = archive["decode_steps"]
                    count = len(steps)
                    _require(np.array_equal(steps, np.arange(cursor, cursor + count)), "stage2 Decode discontinuity")
                    derived, sizes, diag = derive_coverage_metrics(
                        archive["prompt_attention_mass"],
                        archive["critical_attention_mass"],
                        counts,
                        candidates,
                        budgets,
                        fixed_sizes,
                    )
                if not buffers:
                    for name, values in derived.items():
                        buffers[name] = np.lib.format.open_memmap(
                            Path(temporary) / (name + ".npy"),
                            mode="w+",
                            dtype=np.float64,
                            shape=(sample.decode_length,) + values.shape[1:],
                        )
                for name, values in derived.items():
                    buffers[name][cursor : cursor + count] = values
                sizes_all[sample_number] = sizes
                diagnostics["middle_mass_clipped_count"] += diag["middle_mass_clipped_count"]
                for key in ("middle_mass_max_negative_magnitude", "fixed_mass_max_budget_difference"):
                    diagnostics[key] = max(diagnostics[key], diag[key])
                cursor += count
            _require(cursor == sample.decode_length, "stage2 Decode coverage differs")
            if not buffers:
                derived, sizes, _ = derive_coverage_metrics(
                    np.empty((0, layers, heads)),
                    np.empty((0, layers, heads, m, b, 2)),
                    counts,
                    candidates,
                    budgets,
                    fixed_sizes,
                )
                buffers = derived
                sizes_all[sample_number] = sizes
            for name, values in buffers.items():
                reduced = summarize_trajectory(values, include_time=name == "recall")
                for statistic in ("full", "valid_count"):
                    key = f"sample_{name}_{statistic}"
                    if key not in arrays:
                        arrays[key] = np.empty((n,) + reduced[statistic].shape, dtype=reduced[statistic].dtype)
                        axes[key] = ["N"] + sample_axes[name] + (["Stat"] if statistic == "full" else [])
                    arrays[key][sample_number] = reduced[statistic]
                if name == "recall":
                    for statistic in ("quarters", "quarter_count", "bins", "bin_count", "temporal", "temporal_count"):
                        key = "sample_recall_" + statistic
                        if key not in arrays:
                            arrays[key] = np.empty((n,) + reduced[statistic].shape, dtype=reduced[statistic].dtype)
                            axes[key] = (
                                ["N", "Q" if statistic.startswith("quarter") else "ProgressBin"] + sample_axes[name]
                                if statistic not in ("temporal", "temporal_count")
                                else ["N"] + sample_axes[name] + ["TimeMetric"]
                            )
                        arrays[key][sample_number] = reduced[statistic]
                    if sample.decode_length < 32:
                        short.append(np.asarray(values, dtype=np.float32).copy())
                        short_offsets.append(short_offsets[-1] + sample.decode_length)
                    else:
                        short_offsets.append(short_offsets[-1])
            diagnostics["long_sample_count" if sample.decode_length >= 32 else "short_sample_count"] += 1
            # Drop the last loop reference as well as all memory maps before cleanup.
            del values, reduced, buffers
        if on_sample:
            on_sample(sample.sample_id, sample_number + 1, n)
    arrays["sample_prefill_features"] = features
    axes["sample_prefill_features"] = ["N", "L", "Hkv", "M", "PrefillFeature"]
    arrays["sample_effective_set_sizes"] = sizes_all
    axes["sample_effective_set_sizes"] = ["N", "L", "Hkv", "M", "B", "U"]
    arrays["sample_candidate_counts"] = candidates_all
    axes["sample_candidate_counts"] = ["N", "M"]
    arrays["condition_fixed_token_counts"] = fixed_sizes
    axes["condition_fixed_token_counts"] = ["M"]
    arrays["sample_prefill_head_distribution"] = _distribution(features.reshape(n, layers * heads, m, -1), 1)
    axes["sample_prefill_head_distribution"] = ["N", "M", "PrefillFeature", "DistributionStat"]
    # Condition means always average sample records, never pool their Step counts.
    for key in list(arrays):
        if not key.startswith("sample_") or key.endswith(
            ("valid_count", "quarter_count", "bin_count", "temporal_count")
        ):
            continue
        if key == "sample_prefill_head_distribution":
            continue
        grouped, valid_samples, stds = [], [], []
        for condition in condition_names:
            mask = np.asarray([s.condition == condition for s in samples])
            mean, count = _mean(arrays[key][mask])
            grouped.append(mean.astype(np.float32))
            valid_samples.append(count.astype(np.uint32))
            if key == "sample_prefill_features":
                stds.append(np.std(arrays[key][mask], axis=0, dtype=np.float64).astype(np.float32))
        target = "condition_" + key.removeprefix("sample_")
        arrays[target + "_mean"] = np.stack(grouped)
        arrays[target + "_valid_samples"] = np.stack(valid_samples)
        axes[target + "_mean"] = ["DataCondition"] + axes[key][1:]
        axes[target + "_valid_samples"] = ["DataCondition"] + axes[key][1:]
        if stds:
            arrays["condition_prefill_features_std"] = np.stack(stds)
            axes["condition_prefill_features_std"] = ["DataCondition", "L", "Hkv", "M", "PrefillFeature"]
    arrays["condition_prefill_head_distribution"] = _distribution(
        arrays["condition_prefill_features_mean"].reshape(len(condition_names), layers * heads, m, -1), 1
    )
    axes["condition_prefill_head_distribution"] = ["DataCondition", "M", "PrefillFeature", "DistributionStat"]
    arrays["short_recall"] = (
        np.concatenate(short) if short else np.empty((0, layers, heads, m, b, 2, 2), dtype=np.float32)
    )
    axes["short_recall"] = ["ShortStep", "L", "Hkv", "M", "B", "U", "R"]
    arrays["short_sample_offsets"] = np.asarray(short_offsets, dtype=np.uint32)
    axes["short_sample_offsets"] = ["N_plus_1"]
    arrays.update(
        sample_ids=np.asarray([s.sample_id for s in samples]),
        sample_conditions=np.asarray([s.condition for s in samples]),
        sample_pair_ids=np.asarray([s.pair_id or "" for s in samples]),
        sample_decode_lengths=np.asarray([s.decode_length for s in samples], dtype=np.uint32),
        condition_names=np.asarray(condition_names),
        condition_ids=np.asarray(conditions),
        budgets=np.asarray(budgets, dtype=np.uint32),
        set_types=np.asarray(SET_TYPES),
        recall_scopes=np.asarray(RECALL_SCOPES),
        statistics=np.asarray(STATISTICS),
        prefill_feature_names=np.asarray(PREFILL_FEATURES),
        distribution_statistics=np.asarray(DISTRIBUTION_STATISTICS),
        time_metrics=np.asarray(TIME_METRICS),
        control_names=np.asarray(("fixed", "middle")),
        budget_pairs=np.asarray(list(zip(budgets[:-1], budgets[1:])), dtype=np.uint32),
    )
    axes.update(
        sample_ids=["N"],
        sample_conditions=["N"],
        sample_pair_ids=["N"],
        sample_decode_lengths=["N"],
        condition_names=["DataCondition"],
        condition_ids=["M"],
        budgets=["B"],
        set_types=["U"],
        recall_scopes=["R"],
        statistics=["Stat"],
        prefill_feature_names=["PrefillFeature"],
        distribution_statistics=["DistributionStat"],
        time_metrics=["TimeMetric"],
        control_names=["Control"],
        budget_pairs=["D_budget", "PairEndpoint"],
    )
    return B2CoverageSummary(arrays, axes, diagnostics)
