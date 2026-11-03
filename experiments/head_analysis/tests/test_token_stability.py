import numpy as np
import pytest

from experiments.head_analysis.analysis.token_stability import compute_token_stability
from experiments.head_analysis.collectors.attention_stats import UINT32_SENTINEL


def test_token_stability_matches_exact_overlap_rco_rank_and_age():
    tokens = np.asarray(
        [
            [[[[0, 1, UINT32_SENTINEL, UINT32_SENTINEL]]]],
            [[[[1, 2, UINT32_SENTINEL, UINT32_SENTINEL]]]],
            [[[[2, 3, UINT32_SENTINEL, UINT32_SENTINEL]]]],
        ],
        dtype=np.uint32,
    ).reshape(3, 1, 1, 4)
    visible = np.asarray([4, 5, 6], dtype=np.uint32)
    query = visible - 1

    result = compute_token_stability(
        sample_id="tiny",
        token_positions=tokens,
        visible_count=visible,
        query_positions=query,
        k_values=(2, 4),
        deltas=(1, 2, 4),
        comparison_chunk_steps=1,
    )

    metric = {name: index for index, name in enumerate(result.metrics)}
    assert result.valid_pair_count[0, 0, 0].tolist() == [2, 1, 0]
    assert result.rank_valid_pair_count[0, 0, 0].tolist() == [2, 0, 0]
    assert result.head_metrics[0, 0, 0, 0, metric["overlap_raw_mean"]] == 0.5
    assert result.head_metrics[0, 0, 0, 0, metric["rco_raw_mean"]] == pytest.approx(np.mean([1 / 6, 1 / 4]))
    assert result.head_metrics[0, 0, 0, 0, metric["token_entry_rate_mean"]] == 0.5
    assert result.head_metrics[0, 0, 0, 0, metric["token_entry_rate_p90"]] == 0.5
    assert result.head_metrics[0, 0, 0, 0, metric["rank_displacement_mean"]] == 1.0
    assert result.head_metrics[0, 0, 0, 1, metric["rco_clipped_mean"]] == 0.0
    assert np.isnan(result.head_metrics[0, 0, 0, 1, metric["rank_displacement_mean"]])
    np.testing.assert_allclose(result.head_age[0, 0, 0], [2.5, 3.0])
    assert result.age_valid_token_count[0, 0, 0] == 6
    assert not result.valid_pair_count[0, 0, 1].any()
    assert np.isnan(result.head_metrics[0, 0, 1]).all()


def test_duplicate_topk_positions_invalidate_only_affected_pairs():
    tokens = np.asarray(
        [
            [[[0, 1, 2, 3]]],
            [[[0, 0, 2, 3]]],
            [[[0, 1, 2, 3]]],
        ],
        dtype=np.uint32,
    )
    visible = np.asarray([8, 9, 10], dtype=np.uint32)

    result = compute_token_stability(
        sample_id="duplicate",
        token_positions=tokens,
        visible_count=visible,
        query_positions=visible - 1,
        k_values=(2, 4),
        deltas=(1, 2),
    )

    assert result.valid_pair_count[0, 0, 0].tolist() == [0, 1]
    assert result.valid_pair_count[0, 0, 1].tolist() == [0, 1]


def test_zero_common_tokens_keep_rank_displacement_null_but_other_metrics_valid():
    tokens = np.asarray([[[[0, 1]]], [[[2, 3]]]], dtype=np.uint32)
    visible = np.asarray([4, 5], dtype=np.uint32)

    result = compute_token_stability(
        sample_id="disjoint",
        token_positions=tokens,
        visible_count=visible,
        query_positions=visible - 1,
        k_values=(2,),
        deltas=(1,),
    )

    metric = {name: index for index, name in enumerate(result.metrics)}
    assert result.valid_pair_count[0, 0, 0, 0] == 1
    assert result.rank_valid_pair_count[0, 0, 0, 0] == 0
    assert result.head_metrics[0, 0, 0, 0, metric["token_entry_rate_mean"]] == 1.0
    assert np.isnan(result.head_metrics[0, 0, 0, 0, metric["rank_displacement_mean"]])
