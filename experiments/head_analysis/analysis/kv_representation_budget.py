"""V3-A fixed Head rank allocation and explicit representation byte budgets."""
from __future__ import annotations

import numpy as np


RANKS = tuple(range(16, 129, 16))
MODEL_LAYOUT_BYTES = 8  # L/H/D:uint16, basis granularity:uint8, dtype/layout:uint8.
CACHE_LAYOUT_BYTES = 4  # T:uint32; all other offsets follow from the fixed ranks.


def _rank_grid(ranks):
    values = np.asarray(ranks, dtype=np.int64)
    if values.ndim != 1 or not len(values) or np.any(values <= 0) or np.any(np.diff(values) <= 0):
        raise ValueError("rank grid must contain increasing positive integers")
    return values


def optimal_layer_allocation(costs, baseline_rank, *, ranks=RANKS):
    """Minimize Calib cost separately per Layer at exactly H * baseline_rank.

    costs[Layer,Head,Rank] are the already frozen calibration error curves.
    Exact ties retain the first ascending-rank path encountered by the DP.
    """
    ranks = _rank_grid(ranks)
    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim != 3 or costs.shape[-1] != len(ranks) or not np.isfinite(costs).all():
        raise ValueError("finite costs[Layer,Head,Rank] matching the rank grid required")
    if baseline_rank not in ranks:
        raise ValueError("baseline rank is not in the rank grid")
    layers, heads, _ = costs.shape
    budget = heads * int(baseline_rank)
    result = np.empty((layers, heads), dtype=np.uint16)
    for layer in range(layers):
        previous = np.full(budget + 1, np.inf)
        previous[0] = 0.
        choices = np.full((heads, budget + 1), -1, dtype=np.int64)
        for head in range(heads):
            current = np.full(budget + 1, np.inf)
            for rank_index, rank in enumerate(ranks):
                if rank > budget:
                    continue
                candidate = previous[:budget + 1 - rank] + costs[layer, head, rank_index]
                destination = current[rank:]
                better = candidate < destination
                destination[better] = candidate[better]
                choices[head, rank:][better] = rank_index
            previous = current
        if not np.isfinite(previous[budget]):
            raise ValueError("layer budget cannot be represented by the rank grid")
        remaining = budget
        for head in range(heads - 1, -1, -1):
            rank_index = choices[head, remaining]
            result[layer, head] = ranks[rank_index]
            remaining -= ranks[rank_index]
    return result


def byte_ledger(rank_matrix, prompt_lengths, *, strategy="per_head", dimension=128,
                protected_tokens=128, rank_table="auto", copies=(1, 4, 16), request_metadata_bytes=CACHE_LAYOUT_BYTES):
    """Byte accounting for one representation, without allocating cache tensors.

    BF16 coefficients/protected tokens and FP32 bases/means are counted as
    stored arrays, not NPZ container sizes. The 8-byte common model header
    specifies L/H/D, basis granularity, and the BF16/F32/fixed-layout format.
    Default metadata is four bytes for T; T/g callers explicitly charge eight.
    Offsets are derived, not stored.
    """
    rank_matrix = np.asarray(rank_matrix, dtype=np.int64)
    lengths = np.atleast_1d(np.asarray(prompt_lengths, dtype=np.int64))
    copies = np.asarray(copies, dtype=np.int64)
    if rank_matrix.ndim != 2 or not rank_matrix.size or np.any((rank_matrix <= 0) | (rank_matrix > dimension)):
        raise ValueError("ranks require a nonempty [Layer,Head] matrix within the feature dimension")
    if lengths.ndim != 1 or np.any(lengths < protected_tokens) or protected_tokens < 0:
        raise ValueError("Prompt lengths must cover the protected Token range")
    if strategy not in ("per_head", "layer", "global"):
        raise ValueError("unknown basis sharing strategy")
    if copies.ndim != 1 or np.any(copies <= 0):
        raise ValueError("resident cache counts must be positive")
    uniform = bool(np.all(rank_matrix == rank_matrix.flat[0]))
    if rank_table == "auto":
        rank_table = "uniform" if uniform else "per_head"
    if rank_table not in ("uniform", "per_head") or (rank_table == "uniform" and not uniform):
        raise ValueError("a single rank scalar can only encode a uniform rank matrix")
    layers, heads = rank_matrix.shape
    rank_sum = int(rank_matrix.sum())
    basis_elements = {"per_head": rank_sum, "layer": int(rank_matrix.max(1).sum()),
                      "global": int(rank_matrix.max())}[strategy] * dimension
    basis_bytes = 4 * basis_elements
    mean_bytes = 4 * layers * heads * dimension
    rank_bytes = 2 if rank_table == "uniform" else 2 * layers * heads
    model_bytes = basis_bytes + mean_bytes + rank_bytes + MODEL_LAYOUT_BYTES
    coefficient = 2 * (lengths - protected_tokens) * rank_sum
    protected = np.full_like(lengths, 2 * protected_tokens * layers * heads * dimension)
    cache_meta = np.full_like(lengths, request_metadata_bytes)
    cache = coefficient + protected + cache_meta
    dense = 2 * lengths * layers * heads * dimension
    return {
        "strategy": strategy, "rank_encoding": rank_table, "rank_sum": rank_sum,
        "model_basis_bytes": int(basis_bytes), "model_mean_bytes": int(mean_bytes),
        "model_rank_metadata_bytes": int(rank_bytes), "model_layout_metadata_bytes": MODEL_LAYOUT_BYTES,
        "model_total_bytes": int(model_bytes), "prompt_lengths": lengths,
        "cache_coefficient_bytes": coefficient, "cache_protected_bytes": protected,
        "cache_metadata_bytes": cache_meta, "cache_total_bytes": cache,
        "dense_payload_bytes": dense, "dense_cache_bytes": dense + cache_meta,
        "request_counts": copies,
        "resident_identical_requests_bytes": model_bytes + copies[:, None] * cache[None],
    }


def allocate_budget(costs, baseline_rank, prompt_lengths=None, *, ranks=RANKS,
                    dimension=128, protected_tokens=128, request_metadata_bytes=CACHE_LAYOUT_BYTES):
    """Return equal-coefficient diagnostics and a strictly affordable Head policy.

    The repair uses only Calib cost increments and model byte differences.
    A rank decrease also decreases request payload for every N >= 0, hence the
    frozen policy does not depend on validation/test Prompt lengths or errors.
    """
    ranks = _rank_grid(ranks)
    costs = np.asarray(costs, dtype=np.float64)
    equal = optimal_layer_allocation(costs, baseline_rank, ranks=ranks)
    uniform = np.full(equal.shape, baseline_rank, dtype=np.uint16)
    strict = equal.copy()
    lengths = [protected_tokens + 1] if prompt_lengths is None else prompt_lengths

    def ledger(values):
        return byte_ledger(values, lengths, dimension=dimension, protected_tokens=protected_tokens,
                           request_metadata_bytes=request_metadata_bytes)

    uniform_ledger, equal_ledger = ledger(uniform), ledger(equal)
    strict_ledger = equal_ledger
    deletions = []
    while strict_ledger["model_total_bytes"] > uniform_ledger["model_total_bytes"] or strict.sum() > uniform.sum():
        candidates = []
        for layer, head in np.ndindex(strict.shape):
            ri = int(np.searchsorted(ranks, strict[layer, head]))
            if ri > 0:
                change = costs[layer, head, ri - 1] - costs[layer, head, ri]
                candidates.append((float(change), layer, head, ri))
        if not candidates:
            raise ValueError("minimum ranks cannot satisfy the uniform byte budget")
        change, layer, head, ri = min(candidates)
        before = strict_ledger["model_total_bytes"]
        old_rank = int(strict[layer, head])
        strict[layer, head] = ranks[ri - 1]
        strict_ledger = ledger(strict)
        deletions.append({"layer_index": layer, "head_index": head, "from_rank": old_rank,
            "to_rank": int(strict[layer, head]), "calibration_cost_increase": change,
            "model_bytes_before": before, "model_bytes_after": strict_ledger["model_total_bytes"]})
    if np.any(strict.sum(1) > uniform.sum(1)) or np.any(strict_ledger["cache_total_bytes"] > uniform_ledger["cache_total_bytes"]):
        raise ValueError("allocation exceeded a Layer or request budget")

    def objective(values):
        ri = np.searchsorted(ranks, values)
        selected = np.take_along_axis(costs, ri[..., None], axis=-1)[..., 0]
        return float(selected.mean())

    return {"baseline_rank": int(baseline_rank), "uniform_ranks": uniform,
        "equal_coefficient_ranks": equal, "strict_ranks": strict,
        "uniform_ledger": uniform_ledger, "equal_coefficient_ledger": equal_ledger,
        "strict_ledger": strict_ledger, "deletions": deletions,
        "uniform_calibration_objective": objective(uniform),
        "equal_coefficient_calibration_objective": objective(equal),
        "strict_calibration_objective": objective(strict),
        "model_slack_bytes": uniform_ledger["model_total_bytes"] - strict_ledger["model_total_bytes"],
        "cache_slack_bytes": uniform_ledger["cache_total_bytes"] - strict_ledger["cache_total_bytes"],
        "coefficient_slack_per_middle_token_bytes": 2 * int(uniform.sum() - strict.sum())}


def combine_ledgers(*ledgers):
    """Add independent K/V costs while retaining one shared parameter set per side."""
    if not ledgers:
        raise ValueError("at least one representation ledger required")
    first = ledgers[0]
    for item in ledgers[1:]:
        if not (np.array_equal(item["prompt_lengths"], first["prompt_lengths"])
                and np.array_equal(item["request_counts"], first["request_counts"])):
            raise ValueError("K/V ledgers require matching request lengths/counts")
    keys = [key for key in first if key.endswith("_bytes")]
    return {"representations": len(ledgers), "prompt_lengths": first["prompt_lengths"],
            "request_counts": first["request_counts"],
            **{key: sum(item[key] for item in ledgers) for key in keys}}
