import numpy as np
import pytest

from experiments.head_analysis.analysis.b1_offline import B1OfflineError
from experiments.head_analysis.analysis.metric_relationships import (
    PRIMARY_RELATION_DELTAS,
    QUERY_RELATION_FEATURES,
    RelationshipInputs,
    analyze_metric_relationships,
    build_primary_profiles,
)


def make_inputs() -> RelationshipInputs:
    pair_ids = ("pg19_001", "pg19_003", "pg19_005")
    sample_ids = tuple(f"{pair_id}_{length}" for length in ("8k", "32k") for pair_id in pair_ids)
    sample_conditions = ("pg19:8k",) * 3 + ("pg19:32k",) * 3
    condition_names = ("pg19:32k", "pg19:8k")
    samples, layers, kv_heads, query_heads = 6, 2, 3, 12
    features = len(QUERY_RELATION_FEATURES)
    base = np.asarray([[0.10, 0.20, 0.30], [0.60, 0.70, 0.80]], dtype=np.float32)
    query = np.empty((samples, 1, layers, query_heads, features, 1), dtype=np.float32)
    page = np.empty((samples, layers, kv_heads, 2, 1, 1), dtype=np.float32)
    page_rco = np.empty((samples, layers, kv_heads, 2, 1, 4, 1), dtype=np.float32)
    token = np.empty((samples, layers, query_heads, 1, 4, 1), dtype=np.float32)
    for sample in range(samples):
        sample_offset = (sample % 3) * 0.005
        condition_offset = 0.01 if sample >= 3 else 0.0
        profile = base + sample_offset + condition_offset
        grouped = np.repeat(profile, 4, axis=1)
        for feature in range(features):
            query[sample, 0, :, :, feature, 0] = grouped + feature * 0.001
        page[sample, :, :, 0, 0, 0] = profile
        page[sample, :, :, 1, 0, 0] = profile + 0.01
        for scope in range(2):
            for delta in range(4):
                page_rco[sample, :, :, scope, 0, delta, 0] = profile - delta * 0.01 + scope * 0.002
        for delta in range(4):
            token[sample, :, :, 0, delta, 0] = grouped - delta * 0.01
    return RelationshipInputs(
        sample_ids=sample_ids,
        sample_conditions=sample_conditions,
        condition_names=condition_names,
        query_feature_names=QUERY_RELATION_FEATURES,
        query_statistics=("mean",),
        query_phases=("decode",),
        query_sample_full=query,
        page_scopes=("all_visible", "prompt_only"),
        page_metrics=("top32_coverage",),
        page_sample_full=page,
        page_rco_k_values=(32,),
        page_rco_deltas=PRIMARY_RELATION_DELTAS,
        page_rco_metrics=("rco_raw_mean",),
        page_rco_sample_head_values=page_rco,
        token_k_values=(32,),
        token_deltas=PRIMARY_RELATION_DELTAS,
        token_metrics=("rco_raw_mean",),
        token_sample_head_metrics=token,
    )


def test_primary_profiles_apply_contiguous_four_to_one_gqa_after_sample_summary():
    inputs = make_inputs()
    profiles = build_primary_profiles(inputs)

    expected = np.asarray([[0.10, 0.20, 0.30], [0.60, 0.70, 0.80]], dtype=np.float32)
    np.testing.assert_allclose(profiles["query:entropy_normalized"][0], expected)
    np.testing.assert_allclose(profiles["token_rco:k32:d1"][0], expected)
    np.testing.assert_allclose(profiles["page_rco:all_visible:k32:d1"][0], expected)


def test_relationships_are_sample_bootstrapped_deterministic_and_layer_aware():
    inputs = make_inputs()
    first = analyze_metric_relationships(inputs, bootstrap_replicates=20, bootstrap_seed=7)
    second = analyze_metric_relationships(inputs, bootstrap_replicates=20, bootstrap_seed=7)

    assert len(first.records) == 112
    target = next(
        record
        for record in first.records
        if record.relation_id == "within:pg19:8k:token_rco:k32:d1__page_rco:all_visible:k32:d1"
    )
    assert target.bootstrap_mode == "paired_samples_within_condition"
    assert target.estimate.valid_head_count == 6
    assert target.estimate.spearman == pytest.approx(1.0)
    assert target.estimate.pearson == pytest.approx(1.0)
    assert target.estimate.layer_centered_spearman == pytest.approx(1.0)
    assert target.estimate.per_layer_valid_head_count.tolist() == [3, 3]
    np.testing.assert_allclose(target.estimate.per_layer_spearman, 1.0)
    np.testing.assert_allclose(target.estimate.spearman_bootstrap_ci, [1.0, 1.0])

    repeated = next(record for record in second.records if record.relation_id == target.relation_id)
    np.testing.assert_array_equal(
        target.estimate.layer_centered_spearman_bootstrap_ci,
        repeated.estimate.layer_centered_spearman_bootstrap_ci,
    )


def test_pg19_cross_length_profile_correlation_uses_pair_bootstrap():
    summary = analyze_metric_relationships(make_inputs(), bootstrap_replicates=10, bootstrap_seed=11)
    target = next(
        record
        for record in summary.records
        if record.relation_id == "cross:page_rco:all_visible:k32:d1:pg19:32k__pg19:8k"
    )

    assert target.bootstrap_mode == "paired_pg19_samples"
    assert target.estimate.valid_head_count == 6
    assert target.estimate.spearman == pytest.approx(1.0)
    np.testing.assert_allclose(target.estimate.spearman_bootstrap_ci, [1.0, 1.0])


def test_relationship_input_rejects_non_four_to_one_gqa():
    inputs = make_inputs()
    invalid = RelationshipInputs(
        **{
            **inputs.__dict__,
            "token_sample_head_metrics": inputs.token_sample_head_metrics[:, :, :-1],
            "query_sample_full": inputs.query_sample_full[:, :, :, :-1],
        }
    )

    with pytest.raises(B1OfflineError, match="4:1 GQA"):
        build_primary_profiles(invalid)
