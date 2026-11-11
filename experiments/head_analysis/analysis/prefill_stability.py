"""Exploratory leave-one-group-out tests of Prefill CV and Decode stability."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.head_analysis.analysis.b2_relationships import correlate_profiles
from experiments.head_analysis.common.io import read_json
from experiments.head_analysis.common.statistics import bootstrap_draws

CV_ID = "cv:paper_token_primary"
TARGETS = {
    "page_d32": "page_rco:prompt_only:k32:d32",
    "token_d1": "token_rco:all_visible:k32:d1",
}
METHODS = ("cv", "history", "history_cv")
METRICS = ("rho_cv", "rho_history", "rho_history_cv", "mse_history", "mse_history_cv", "delta_rho", "delta_mse")


@dataclass(frozen=True)
class Inputs:
    sample_ids: np.ndarray
    conditions: np.ndarray
    pair_ids: np.ndarray
    decode_lengths: np.ndarray
    x: np.ndarray
    ys: dict[str, np.ndarray]
    source_counts: dict[str, np.ndarray]
    source_heads: dict[str, np.ndarray]
    source_run_set: str


def load_inputs(bundle: Path | str) -> Inputs:
    root = Path(bundle)
    index = read_json(root / "sample_index.json")
    manifest = read_json(root / "analysis_manifest.json")
    stage = read_json(root / "stages/stage4/stage4_manifest.json")
    if stage.get("source_run_set_sha256") != manifest.get("source_run_set_sha256"):
        raise ValueError("stage4 and bundle sources differ")
    records = index["samples"]
    path = root / "stages/stage4/stage4_joint_profiles.npz"
    with np.load(path, allow_pickle=False) as a:
        ids, conditions, pairs = (a[k].copy() for k in ("sample_ids", "sample_conditions", "sample_pair_ids"))
        xids, yids = a["x_ids"].tolist(), a["y_ids"].tolist()
        if len(set(xids)) != len(xids) or len(set(yids)) != len(yids):
            raise ValueError("duplicate metric IDs")
        x = a["sample_x"][:, xids.index(CV_ID)].astype(np.float64)
        ys, counts, heads = {}, {}, {}
        for target, name in TARGETS.items():
            j = yids.index(name)
            value = a["sample_y"][:, j].astype(np.float64)
            support = a["sample_y_valid_source_heads"][:, j].copy()
            count = a["sample_y_source_count_max"][:, j].copy()
            ys[target] = np.where((support > 0) & (count > 0), value, np.nan)
            counts[target], heads[target] = count, support
    n = len(ids)
    if (len(set(ids.tolist())) != n or len(records) != n or len(conditions) != n or len(pairs) != n
            or x.ndim != 3 or x.shape[0] != n):
        raise ValueError("invalid sample IDs or CV shape")
    if any(y.shape != x.shape for y in ys.values()):
        raise ValueError("CV and RCO shapes differ")
    if [r["sample_id"] for r in records] != ids.tolist():
        raise ValueError("sample index and arrays differ")
    if [r.get("pair_id") or "" for r in records] != pairs.tolist():
        raise ValueError("sample pair IDs differ")
    for r, condition in zip(records, conditions):
        target = int(r["target_prompt_length"])
        expected = f"{r['dataset']}:{target // 1024}k" if target % 1024 == 0 else f"{r['dataset']}:{target}"
        if expected != condition:
            raise ValueError("sample data conditions differ")
        if r["dataset"] == "pg19" and not r.get("pair_id"):
            raise ValueError("PG19 sample is missing a pair ID")
    return Inputs(ids, conditions, pairs, np.asarray([r["decode_length"] for r in records]), x, ys, counts, heads,
                  manifest["source_run_set_sha256"])


def _correlation(x, y):
    """B2 pairwise-complete layer-centered Spearman, preserving missing values."""
    result = correlate_profiles(x, y)
    return result["values"][..., 1, 0], result["valid_heads"][..., 1]


def predict_fold(x, y, test_index, weights):
    """Weighted leave-one-group-out fit; the caller removes all held-out group copies.

    x/y are [N,L,H]; weights are [B,N]. For each Head, paired finite training
    observations define both history means. The common within-Head OLS slope is
    estimated across all Heads and weighted training observations.
    """
    x, y, weights = np.asarray(x, float), np.asarray(y, float), np.asarray(weights, float)
    valid = np.isfinite(x) & np.isfinite(y)
    xf, yf = np.where(valid, x, 0).reshape(len(x), -1), np.where(valid, y, 0).reshape(len(y), -1)
    mask = valid.reshape(len(x), -1).astype(float)
    count = weights @ mask
    sx, sy = weights @ xf, weights @ yf
    mx = np.divide(sx, count, out=np.full_like(sx, np.nan), where=count > 0)
    my = np.divide(sy, count, out=np.full_like(sy, np.nan), where=count > 0)
    sxx, sxy = weights @ (xf * xf), weights @ (xf * yf)
    centered_xx = sxx - np.divide(sx * sx, count, out=np.zeros_like(sx), where=count > 0)
    centered_xy = sxy - np.divide(sx * sy, count, out=np.zeros_like(sx), where=count > 0)
    denom = np.maximum(centered_xx, 0).sum(axis=1)
    beta = np.divide(centered_xy.sum(axis=1), denom, out=np.full(len(weights), np.nan), where=denom > 1e-12)
    history = my.reshape((-1,) + x.shape[1:])
    prediction = (my + beta[:, None] * (x[test_index].reshape(1, -1) - mx)).reshape(history.shape)
    return history, prediction, beta


def _scores(x, y, history, prediction):
    """Compare on identical valid Head coordinates, separately for each resample."""
    x, y = np.broadcast_arrays(x, y, history)[0:2]
    rcv, _ = _correlation(x, y)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(history) & np.isfinite(prediction)
    x, y, history, prediction = (np.where(valid, a, np.nan) for a in (x, y, history, prediction))
    rh, count = _correlation(history, y)
    rp, _ = _correlation(prediction, y)
    n = valid.sum(axis=(-2, -1))
    eh = np.where(valid, (history - y) ** 2, 0).sum(axis=(-2, -1))
    ep = np.where(valid, (prediction - y) ** 2, 0).sum(axis=(-2, -1))
    mh = np.divide(eh, n, out=np.full(np.shape(eh), np.nan), where=n > 0)
    mp = np.divide(ep, n, out=np.full(np.shape(ep), np.nan), where=n > 0)
    return np.stack((rcv, rh, rp, mh, mp, rp - rh, mh - mp), axis=-1), count


def evaluate_condition(x, y, groups, *, replicates=2000, seed=20261111, label="condition", batch_size=64):
    """Group LOO and paired group Bootstrap with refitting, by one condition/target."""
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups, str)
    if x.ndim != 3 or x.shape != y.shape or len(groups) != len(x) or replicates <= 0 or batch_size <= 0:
        raise ValueError("invalid profiles, groups or Bootstrap parameters")
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 3:
        raise ValueError("at least three independent groups are required")
    draws = bootstrap_draws(seed, label, replicates, len(unique))
    observed = np.full((len(x), len(METRICS)), np.nan)
    valid_heads = np.zeros(len(x), dtype=np.uint32)
    slopes = np.full(len(x), np.nan)
    boot = np.full((replicates, 2), np.nan)
    for start in [-1, *range(0, replicates, batch_size)]:
        if start < 0:
            group_weights = np.ones((1, len(unique)))
        else:
            draw = draws[start : start + batch_size]
            group_weights = np.zeros((len(draw), len(unique)))
            np.add.at(group_weights, (np.arange(len(draw))[:, None], draw), 1)
        # Give every group one unit of mass even if it contains multiple rows.
        group_sizes = np.bincount(inverse)
        sample_weights = group_weights[:, inverse] / group_sizes[inverse]
        values = np.full((len(group_weights), len(x), len(METRICS)), np.nan)
        for i in range(len(x)):
            weights = sample_weights.copy()
            weights[:, inverse == inverse[i]] = 0  # Exclude every copy of the held-out group.
            history, prediction, beta = predict_fold(x, y, i, weights)
            scores, count = _scores(x[i], y[i], history, prediction)
            values[:, i] = scores
            if start < 0:
                observed[i], valid_heads[i], slopes[i] = scores[0], count[0], beta[0]
        if start >= 0:
            # Each drawn group has equal weight; duplicate draws retain multiplicity.
            for k, metric_index in enumerate((5, 6)):
                v = values[:, :, metric_index]
                valid = np.isfinite(v) & (sample_weights > 0)
                denom = np.where(valid, sample_weights, 0).sum(axis=1)
                num = np.where(valid, v * sample_weights, 0).sum(axis=1)
                boot[start : start + len(group_weights), k] = np.divide(
                    num, denom, out=np.full(len(group_weights), np.nan), where=denom > 0
                )
    return observed, valid_heads, slopes, boot


def finite_summary(values):
    values = np.asarray(values, float)
    valid = values[np.isfinite(values)]
    return {"count": int(len(valid)), "mean": float(valid.mean()) if len(valid) else None,
            "median": float(np.median(valid)) if len(valid) else None}


def analyze(inputs: Inputs, *, replicates=2000, seed=20261111, batch_size=64, progress=None):
    rows, summaries, bootstrap = [], {}, {}
    for target, target_id in TARGETS.items():
        for condition in sorted(set(inputs.conditions.tolist())):
            mask = inputs.conditions == condition
            if target == "page_d32" and condition.startswith("ruler:"):
                summaries[f"{target}/{condition}"] = {"status": "not_applicable", "reason": "six Decode steps; delta=32"}
                continue
            sample_ids, pair_ids = inputs.sample_ids[mask], inputs.pair_ids[mask]
            groups = np.where(pair_ids != "", pair_ids, sample_ids)
            label = f"{target}/{condition}"
            values, valid_heads, slopes, boot = evaluate_condition(inputs.x[mask], inputs.ys[target][mask], groups,
                replicates=replicates, seed=seed, label=label, batch_size=batch_size)
            for i, sample_id in enumerate(sample_ids):
                row = {"sample_id": str(sample_id), "condition": condition, "pair_id": str(pair_ids[i]) or None,
                       "target": target_id, "decode_length": int(inputs.decode_lengths[mask][i]),
                       "valid_heads": int(valid_heads[i]), "beta": float(slopes[i]) if np.isfinite(slopes[i]) else None}
                row.update({name: float(v) if np.isfinite(v) else None for name, v in zip(METRICS, values[i])})
                rows.append(row)
            metrics = {name: finite_summary(values[:, j]) for j, name in enumerate(METRICS)}
            for j, name in enumerate(("delta_rho", "delta_mse")):
                valid = boot[:, j][np.isfinite(boot[:, j])]
                metrics[name]["bootstrap_ci_95"] = np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else None
                metrics[name]["bootstrap_valid_replicates"] = len(valid)
            summaries[label] = {"status": "completed", "sample_count": len(sample_ids), "group_count": len(set(groups)),
                                "metrics": metrics}
            bootstrap[label.replace("/", "__").replace(":", "_")] = boot
            if progress:
                progress(label)
    return rows, summaries, bootstrap
