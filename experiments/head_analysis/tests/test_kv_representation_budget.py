from itertools import product

import numpy as np
import pytest

from experiments.head_analysis.analysis.kv_representation_budget import (
    allocate_budget, byte_ledger, combine_ledgers, optimal_layer_allocation,
)


def test_layer_dp_matches_exhaustive_minimum_and_never_moves_budget_across_layers():
    costs = np.array([[[9., 3., 1.], [4., 3., 0.], [6., 2., 1.]],
                      [[1., .9, .8], [8., 3., 0.], [7., 4., .2]]])
    ranks = (1, 2, 3)
    allocation = optimal_layer_allocation(costs, 2, ranks=ranks)
    np.testing.assert_array_equal(allocation.sum(1), 6)
    for layer in range(2):
        brute = min(sum(costs[layer, h, r - 1] for h, r in enumerate(candidate))
                    for candidate in product(ranks, repeat=3) if sum(candidate) == 6)
        actual = sum(costs[layer, h, r - 1] for h, r in enumerate(allocation[layer]))
        assert actual == pytest.approx(brute)
    np.testing.assert_array_equal(allocation, optimal_layer_allocation(costs, 2, ranks=ranks))


def test_full_byte_ledger_matches_arrays_and_all_sharing_granularities():
    values = np.full((32, 8), 64)
    ledger = byte_ledger(values, [8192, 32768])
    assert ledger["model_basis_bytes"] == 8 * 1024**2
    assert ledger["model_mean_bytes"] == 128 * 1024
    assert ledger["model_rank_metadata_bytes"] == 2
    assert ledger["model_layout_metadata_bytes"] == 8
    np.testing.assert_array_equal(ledger["cache_protected_bytes"], [8 * 1024**2] * 2)
    np.testing.assert_array_equal(ledger["cache_coefficient_bytes"], 2 * (np.array([8192, 32768]) - 128) * 32 * 8 * 64)
    np.testing.assert_array_equal(ledger["cache_total_bytes"], ledger["cache_coefficient_bytes"] + ledger["cache_protected_bytes"] + 4)
    np.testing.assert_array_equal(ledger["resident_identical_requests_bytes"][1], ledger["model_total_bytes"] + 4 * ledger["cache_total_bytes"])
    assert byte_ledger(values, 8192, strategy="layer")["model_basis_bytes"] == 1024**2
    assert byte_ledger(values, 8192, strategy="global")["model_basis_bytes"] == 32 * 1024
    values[0, 0] = 80
    assert byte_ledger(values, 8192)["model_rank_metadata_bytes"] == 512
    assert byte_ledger(values, 8192, strategy="layer")["model_basis_bytes"] == 4 * 128 * (31 * 64 + 80)
    assert byte_ledger(values, 8192, strategy="global")["model_basis_bytes"] == 4 * 128 * 80


def test_metadata_correction_removes_the_least_cost_block_and_is_length_independent():
    ranks = (16, 32, 48)
    costs = np.array([[[100., 1., 0.], [1., .9, .8]],
                      [[50., 3., 2.], [4., 3., 0.]]])
    result = allocate_budget(costs, 32, [8192, 32768], ranks=ranks)
    equal, strict = result["equal_coefficient_ranks"], result["strict_ranks"]
    assert not np.all(equal == 32)
    np.testing.assert_array_equal(equal.sum(1), [64, 64])
    assert result["equal_coefficient_ledger"]["model_total_bytes"] > result["uniform_ledger"]["model_total_bytes"]
    assert len(result["deletions"]) == 1
    assert result["model_slack_bytes"] >= 0 and (result["cache_slack_bytes"] >= 0).all()
    assert np.all(strict.sum(1) <= 64)
    candidates = [(costs[l, h, ranks.index(int(equal[l, h])) - 1] - costs[l, h, ranks.index(int(equal[l, h]))], l, h)
                  for l, h in np.ndindex(equal.shape) if equal[l, h] > 16]
    _, layer, head = min(candidates)
    assert result["deletions"][0]["layer_index"] == layer
    assert result["deletions"][0]["head_index"] == head
    assert result["deletions"][0]["model_bytes_before"] - result["deletions"][0]["model_bytes_after"] == 8192
    other = allocate_budget(costs, 32, [129, 131072], ranks=ranks)
    np.testing.assert_array_equal(strict, other["strict_ranks"])
    assert result["equal_coefficient_calibration_objective"] <= result["uniform_calibration_objective"]


def test_uniform_tie_solution_can_use_scalar_metadata_and_kv_totals_sum_once():
    costs = np.tile([3., 1., 0.], (1, 2, 1))
    result = allocate_budget(costs, 32, [8192], ranks=(16, 32, 48))
    np.testing.assert_array_equal(result["equal_coefficient_ranks"], [[32, 32]])
    assert result["deletions"] == []
    k = byte_ledger([[16, 32]], [8192], dimension=32)
    v = byte_ledger([[32, 32]], [8192], dimension=32)
    total = combine_ledgers(k, v)
    assert total["model_total_bytes"] == k["model_total_bytes"] + v["model_total_bytes"]
    np.testing.assert_array_equal(total["cache_total_bytes"], k["cache_total_bytes"] + v["cache_total_bytes"])
    np.testing.assert_array_equal(total["resident_identical_requests_bytes"], k["resident_identical_requests_bytes"] + v["resident_identical_requests_bytes"])
    with pytest.raises(ValueError, match="uniform rank"):
        byte_ledger([[16, 32]], [8192], rank_table="uniform")


def test_all_three_planned_budgets_support_eight_heads_and_strict_totals():
    rng = np.random.default_rng(17)
    eigenvalues = np.sort(rng.exponential(size=(2, 8, 128)), axis=-1)[..., ::-1]
    tails = np.cumsum(eigenvalues[..., ::-1], axis=-1)[..., ::-1]
    costs = np.stack([tails[..., rank] if rank < 128 else np.zeros((2, 8))
                      for rank in range(16, 129, 16)], axis=-1)
    for baseline in (32, 64, 96):
        result = allocate_budget(costs, baseline, [7205, 8192, 32768])
        np.testing.assert_array_equal(result["equal_coefficient_ranks"].sum(1), [8 * baseline] * 2)
        assert result["equal_coefficient_calibration_objective"] <= result["uniform_calibration_objective"]
        assert result["strict_ledger"]["model_total_bytes"] <= result["uniform_ledger"]["model_total_bytes"]
        assert (result["strict_ledger"]["cache_total_bytes"] <= result["uniform_ledger"]["cache_total_bytes"]).all()
