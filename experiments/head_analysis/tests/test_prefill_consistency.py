import math

import numpy as np
import pytest

from experiments.head_analysis.collectors.prefill_consistency import (
    B2CollectorConfig,
    B2Condition,
    DEFAULT_B2_CONDITIONS,
    UINT32_SENTINEL,
    PrefillConsistencyError,
    build_prediction_token_sets,
    candidate_token_positions,
    compute_decode_recall_reference,
    compute_prefill_consistency_from_logits_reference,
    compute_prefill_consistency_reference,
    summarize_observation_matrix_reference,
)


def toy_condition(*, condition_id="toy", window=3, initial=2, recent=3, k=0.5, alpha=1.0):
    return B2Condition(condition_id, window, initial, recent, k, alpha)


def independent_softmax(values):
    values = np.asarray(values, dtype=np.float64)
    values = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(values)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def test_frozen_default_conditions_and_config_contract():
    config = B2CollectorConfig()

    assert [item.condition_id for item in config.conditions] == [
        "paper_token_primary",
        "paper_token_alpha1",
        "paper_token_k099",
        "paper_token_recent256",
    ]
    assert config.critical_token_budgets == (128, 512, 2048)
    assert config.critical_token_max == 2048
    assert DEFAULT_B2_CONDITIONS[0].recent_exclusion == 64

    with pytest.raises(ValueError, match="observation_window=64"):
        B2CollectorConfig(conditions=(toy_condition(),))
    with pytest.raises(ValueError, match="strictly increasing"):
        B2CollectorConfig(critical_token_budgets=(128, 128, 2048))
    with pytest.raises(ValueError, match="largest"):
        B2CollectorConfig(critical_token_max=1024)


def test_candidate_positions_exclude_initial_and_full_observation_tail():
    primary, *_, recent256 = DEFAULT_B2_CONDITIONS

    primary_positions = candidate_token_positions(8192, primary)
    recent256_positions = candidate_token_positions(8192, recent256)

    assert (primary_positions[0], primary_positions[-1], primary_positions.size) == (64, 8127, 8064)
    assert (recent256_positions[0], recent256_positions[-1], recent256_positions.size) == (64, 7935, 7872)
    with pytest.raises(PrefillConsistencyError, match="too short"):
        candidate_token_positions(128, primary)


def test_gqa_aggregation_happens_after_each_query_head_softmax():
    condition = toy_condition(window=1, initial=0, recent=1)
    logits = np.array([[[10.0, 0.0]], [[0.0, 0.0]]], dtype=np.float32)

    result = compute_prefill_consistency_from_logits_reference(
        logits,
        np.array([0, 1], dtype=np.uint32),
        condition=condition,
        critical_token_max=2,
    )

    expected = independent_softmax(logits).mean(axis=0)
    wrong = independent_softmax(logits.mean(axis=0))
    np.testing.assert_allclose(result.observation_matrix, expected, rtol=1e-6, atol=1e-6)
    assert not np.allclose(result.observation_matrix, wrong, rtol=1e-3, atol=1e-3)
    np.testing.assert_array_equal(result.frequency_vector, np.array([1, 0], dtype=np.uint8))
    assert result.cv_prefill == pytest.approx(1.0)


def test_token_frequency_cv_histogram_and_stable_ranking_match_formula():
    condition = toy_condition()
    observation = np.array(
        [
            [0.4, 0.4, 0.1, 0.1],
            [0.4, 0.1, 0.4, 0.1],
            [0.1, 0.4, 0.4, 0.1],
        ],
        dtype=np.float32,
    )
    positions = np.array([10, 11, 12, 13], dtype=np.uint32)

    result = summarize_observation_matrix_reference(
        observation,
        positions,
        condition=condition,
        critical_token_max=5,
    )

    expected_threshold = np.quantile(observation.reshape(-1), 0.5, method="linear")
    expected_frequency = (observation >= expected_threshold).sum(axis=0)
    expected_mean = expected_frequency.mean()
    expected_std = np.sqrt(np.mean((expected_frequency - expected_mean) ** 2))
    assert result.binary_threshold == pytest.approx(expected_threshold)
    np.testing.assert_array_equal(result.frequency_vector, [2, 2, 2, 0])
    np.testing.assert_array_equal(result.frequency_histogram, [1, 0, 3, 0])
    assert result.frequency_mean == pytest.approx(expected_mean)
    assert result.frequency_std == pytest.approx(expected_std)
    assert result.cv_prefill == pytest.approx(expected_std / expected_mean)
    assert result.nonzero_frequency_count == result.critical_token_valid_count == 3
    assert result.critical_token_positions.tolist() == [10, 11, 12, UINT32_SENTINEL, UINT32_SENTINEL]
    assert result.critical_token_frequencies.tolist() == [2, 2, 2, 0, 0]


def test_soft_score_breaks_frequency_ties_before_logical_position():
    condition = toy_condition(window=2, initial=0, recent=2, k=0.5)
    observation = np.array([[0.30, 0.25, 0.45], [0.25, 0.30, 0.45]], dtype=np.float32)

    result = summarize_observation_matrix_reference(
        observation,
        np.array([3, 5, 7]),
        condition=condition,
        critical_token_max=3,
    )

    # Position 7 has the greatest soft score. Positions 3 and 5 tie on both
    # frequency and soft score, so the smaller logical position comes first.
    assert result.critical_token_positions.tolist() == [7, 3, 5]


def test_qk_reference_matches_independent_logits_path():
    rng = np.random.default_rng(9)
    condition = toy_condition(window=2, initial=0, recent=2, k=0.75, alpha=0.8)
    query = rng.normal(size=(2, 2, 3)).astype(np.float32)
    keys = rng.normal(size=(4, 3)).astype(np.float32)
    positions = np.arange(4, dtype=np.uint32)
    scaling = 1 / math.sqrt(3)

    from_qk = compute_prefill_consistency_reference(
        query,
        keys,
        positions,
        scaling=scaling,
        condition=condition,
        critical_token_max=4,
    )
    logits = np.einsum("gwd,nd->gwn", query, keys) * scaling
    from_logits = compute_prefill_consistency_from_logits_reference(
        logits,
        positions,
        condition=condition,
        critical_token_max=4,
    )

    np.testing.assert_allclose(from_qk.observation_matrix, from_logits.observation_matrix, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(from_qk.frequency_vector, from_logits.frequency_vector)
    np.testing.assert_array_equal(from_qk.critical_token_positions, from_logits.critical_token_positions)


def test_prediction_sets_are_nested_and_augment_only_fixed_prompt_regions():
    condition = toy_condition(window=2, initial=2, recent=2, k=0.5)
    prompt_length = 8
    observation = np.array([[0.45, 0.30, 0.25, 0.0], [0.45, 0.25, 0.30, 0.0]], dtype=np.float32)
    result = summarize_observation_matrix_reference(
        observation,
        candidate_token_positions(prompt_length, condition),
        condition=condition,
        critical_token_max=4,
    )

    budget_one = build_prediction_token_sets(result, budget=1, prompt_length=prompt_length, condition=condition)
    budget_three = build_prediction_token_sets(result, budget=3, prompt_length=prompt_length, condition=condition)

    assert set(budget_one.learned_middle).issubset(set(budget_three.learned_middle))
    assert set(budget_three.learned_middle).issubset(set(budget_three.augmented_prompt))
    assert {0, 1, 6, 7}.issubset(set(budget_three.augmented_prompt))
    assert np.all(budget_three.augmented_prompt < prompt_length)


def test_decode_recall_uses_full_visible_softmax_then_gqa_mean():
    logits = np.array(
        [
            [2.0, 1.0, 0.0, -1.0, -2.0, -3.0],
            [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        ],
        dtype=np.float32,
    )
    positions = np.arange(6, dtype=np.uint32)
    sets = {
        "learned_1": np.array([0], dtype=np.uint32),
        "learned_2": np.array([0, 2], dtype=np.uint32),
        "augmented": np.array([0, 2, 3], dtype=np.uint32),
    }

    result = compute_decode_recall_reference(logits, positions, prompt_length=4, prediction_sets=sets)

    expected_attention = independent_softmax(logits).mean(axis=0)
    expected_prompt = expected_attention[:4].sum()
    np.testing.assert_allclose(result.attention_probabilities, expected_attention, rtol=1e-6, atol=1e-6)
    assert result.prompt_attention_mass == pytest.approx(expected_prompt)
    assert result.critical_attention_mass["learned_2"] == pytest.approx(expected_attention[[0, 2]].sum())
    assert result.critical_prompt_recall["learned_2"] == pytest.approx(
        expected_attention[[0, 2]].sum() / expected_prompt
    )
    assert result.critical_attention_mass["learned_1"] <= result.critical_attention_mass["learned_2"]
    assert result.critical_attention_mass["learned_2"] <= result.critical_attention_mass["augmented"]
    assert result.critical_attention_mass["augmented"] <= result.prompt_attention_mass + 1e-6


def test_reference_rejects_non_rectangular_or_invalid_position_inputs():
    condition = toy_condition(window=1, initial=0, recent=1)
    logits = np.zeros((2, 1, 2), dtype=np.float32)

    with pytest.raises(PrefillConsistencyError, match="strictly increasing"):
        compute_prefill_consistency_from_logits_reference(
            logits,
            np.array([1, 1]),
            condition=condition,
            critical_token_max=2,
        )
    with pytest.raises(PrefillConsistencyError, match="non-visible"):
        compute_decode_recall_reference(
            np.zeros((2, 3), dtype=np.float32),
            np.arange(3),
            prompt_length=2,
            prediction_sets={"bad": np.array([4])},
        )
    with pytest.raises(PrefillConsistencyError, match="non-Prompt"):
        compute_decode_recall_reference(
            np.zeros((2, 3), dtype=np.float32),
            np.arange(3),
            prompt_length=2,
            prediction_sets={"bad": np.array([2])},
        )
