import numpy as np
import pytest

from experiments.head_analysis.analysis.head_identity import (
    decompose_profiles, pairwise_and_holdout, run_analysis,
)
from experiments.head_analysis.common.io import read_json, write_json, write_npz


def test_fixed_head_pattern_survives_layer_baselines_and_group_holdout():
    pattern = np.asarray([[-3., -1., 1., 3.], [-6., -2., 2., 6.]])
    values = np.broadcast_to(np.asarray([100., 200.])[:, None] + pattern, (4, 2, 4)).copy()
    result = decompose_profiles(values)
    assert result["shares"]["fixed_head"] == pytest.approx(1)
    assert result["shares"]["input"] == pytest.approx(0)
    assert result["shares"]["interaction"] == pytest.approx(0)
    assert result["relative_fixed_share"] == pytest.approx(1)
    np.testing.assert_allclose(result["head_effect"], pattern)
    np.testing.assert_allclose(result["per_head_input_std"], 0)
    repeated = pairwise_and_holdout(values, ["a", "b", "c", "d"])
    upper = np.triu_indices(4, 1)
    np.testing.assert_allclose(repeated["pairwise_spearman"][upper], 1)
    np.testing.assert_allclose(repeated["holdout_rho"], 1)
    np.testing.assert_allclose(repeated["holdout_skill"], 1)
    assert repeated["pooled_skill"] == pytest.approx(1)
    np.testing.assert_allclose(repeated["predictions"], np.broadcast_to(pattern, values.shape))
    shifts = np.asarray([[0., 100.], [-100., 0.], [25., -50.], [75., 200.]])
    shifted = values + shifts[..., None]
    shifted_result = decompose_profiles(shifted)
    assert shifted_result["shares"]["fixed_head"] < 1
    assert shifted_result["shares"]["input"] > 0
    assert shifted_result["shares"]["interaction"] == pytest.approx(0)
    assert shifted_result["relative_fixed_share"] == pytest.approx(1)
    shifted_holdout = pairwise_and_holdout(shifted, ["a", "b", "c", "d"])
    np.testing.assert_allclose(shifted_holdout["holdout_skill"], 1)
    np.testing.assert_allclose(shifted_holdout["predictions"], np.broadcast_to(pattern, values.shape))


def test_input_layer_shift_is_not_mislabelled_as_fixed_head_variation():
    shifts = np.asarray([-3., -1., 1., 3.])[:, None] * np.asarray([1., 2.])[None]
    values = np.broadcast_to((np.asarray([100., 200.]) + shifts)[..., None], (4, 2, 4)).copy()
    result = decompose_profiles(values)
    assert result["shares"]["input"] == pytest.approx(1)
    assert result["shares"]["fixed_head"] == pytest.approx(0)
    assert result["shares"]["interaction"] == pytest.approx(0)
    assert np.isnan(result["relative_fixed_share"])
    assert result["between_head_rms"] == pytest.approx(0)
    assert result["within_head_input_rms"] > 0
    repeated = pairwise_and_holdout(values, ["a", "b", "c", "d"])
    assert np.isnan(repeated["holdout_skill"]).all()
    assert np.isnan(repeated["pooled_skill"])
    assert np.isnan(repeated["pairwise_spearman"]).all()


def test_pure_interaction_keeps_negative_holdout_skill():
    values = np.asarray([[[-1., 0., 1.]], [[1., 0., -1.]]])
    result = decompose_profiles(values)
    assert result["shares"]["interaction"] == pytest.approx(1)
    assert result["shares"]["fixed_head"] == pytest.approx(0)
    assert result["shares"]["input"] == pytest.approx(0)
    assert result["relative_fixed_share"] == pytest.approx(0)
    repeated = pairwise_and_holdout(values, ["a", "b"])
    np.testing.assert_allclose(repeated["holdout_skill"], -3)
    np.testing.assert_allclose(repeated["holdout_rho"], -1)
    assert repeated["pooled_skill"] == pytest.approx(-3)


def test_three_component_decomposition_matches_hand_calculation():
    baseline = np.asarray([10., 20.])
    sample = np.asarray([-1., 0., 1.])
    input_effect = sample[:, None] * np.asarray([1., 2.])[None]
    head_effect = np.asarray([1., 3.])[:, None] * np.asarray([-2., 0., 2.])[None]
    interaction = (sample[:, None, None] * np.asarray([.5, 1.5])[None, :, None]
                   * np.asarray([-1., 2., -1.])[None, None])
    values = baseline[None, :, None] + input_effect[..., None] + head_effect[None] + interaction
    result = decompose_profiles(values)
    assert result["ss_input"] == pytest.approx(30)
    assert result["ss_head"] == pytest.approx(240)
    assert result["ss_interaction"] == pytest.approx(30)
    assert result["ss_total"] == pytest.approx(300)
    np.testing.assert_allclose(result["per_layer_ss_total"], [33, 267])
    np.testing.assert_allclose(result["per_layer_ss_head"], [24, 216])
    np.testing.assert_allclose(result["per_layer_ss_input"], [6, 24])
    np.testing.assert_allclose(result["per_layer_ss_interaction"], [3, 27])
    np.testing.assert_allclose(result["layer_baseline"], baseline)
    np.testing.assert_allclose(result["input_effect"], input_effect)
    np.testing.assert_allclose(result["head_effect"], head_effect)
    np.testing.assert_allclose(result["residual"], interaction)
    reconstructed = (result["layer_baseline"][None, :, None] + result["input_effect"][..., None]
                     + result["head_effect"][None] + result["residual"])
    np.testing.assert_allclose(reconstructed, values)
    assert result["shares"] == pytest.approx({"fixed_head": .8, "input": .1, "interaction": .1})
    assert result["relative_fixed_share"] == pytest.approx(8 / 9)
    np.testing.assert_allclose(result["per_head_input_std"], values.std(axis=0))


def test_pair_members_cannot_predict_one_another_in_source_group_holdout():
    pattern = np.asarray([[-1., 0., 1.]])
    values = np.stack((pattern, pattern, -pattern, -pattern))
    result = pairwise_and_holdout(values, ["pg_pair", "pg_pair", "gov_a", "gov_b"])
    np.testing.assert_array_equal(result["holdout_train_count"], [2, 2, 3, 3])
    np.testing.assert_allclose(result["predictions"][:2], np.stack((-pattern, -pattern)))
    np.testing.assert_allclose(result["holdout_skill"][:2], -3)
    with pytest.raises(ValueError):
        pairwise_and_holdout(values[:2], ["pg_pair", "pg_pair"])


def test_common_missing_coordinates_and_constant_profiles_remain_explicit():
    values = np.broadcast_to(np.arange(8.).reshape(2, 4), (3, 2, 4)).copy()
    values[1, 0, 0] = np.nan
    values[0, 1, :3] = np.nan
    result = decompose_profiles(values)
    np.testing.assert_array_equal(result["valid_mask"][0], [False, True, True, True])
    assert not result["valid_mask"][1].any()
    assert np.isnan(result["head_effect"][0, 0])
    assert np.isnan(result["head_effect"][1]).all()
    assert np.isnan(result["residual"][:, 1]).all()
    assert result["shares"]["fixed_head"] == pytest.approx(1)
    constant = decompose_profiles(np.full((3, 2, 4), 7.))
    assert constant["ss_total"] == 0
    assert all(np.isnan(value) for value in constant["shares"].values())
    assert np.isnan(constant["relative_fixed_share"])


def make_numeric_bundle(root):
    ids = ["ruler_8", "ruler_32", "gov_8", "gov_16", "pg_8", "pg_32"]
    datasets = ["ruler", "ruler", "govreport", "govreport", "pg19", "pg19"]
    lengths = [8072, 32650, 7205, 15594, 8192, 32768]
    samples = []
    for index, (sid, dataset, length) in enumerate(zip(ids, datasets, lengths)):
        values = np.asarray([[10., 12., 15.], [20., 24., 29.]]) + index * .25
        arrays = {"layer_ids": np.arange(2, dtype=np.uint32), "kv_head_ids": np.arange(3, dtype=np.uint32)}
        for representation in ("k_post", "v"):
            arrays[f"full_{representation}_middle_effective_rank"] = values
            arrays[f"full_{representation}_middle_variance_coverage"] = np.stack(
                [values / denominator for denominator in (120, 100, 80, 60)], axis=-1)
            arrays[f"full_{representation}_middle_norm_summary"] = np.stack(
                [values * scale for scale in (1., .8, .95, 1.2, 1.5)], axis=-1)
        for representation in ("k_pre", "k_post", "v"):
            arrays[f"sample_{representation}_middle_effective_rank"] = values * .9
            arrays[f"cka_{representation}_head_observed"] = values / 50
            arrays[f"cka_{representation}_head_excess"] = values / 100
        arrays["rope_middle_rank_post_pre_ratio"] = 1.5 + values / 100
        relative = f"samples/{sid}.npz"
        write_npz(root / relative, arrays)
        samples.append({"sample_id": sid, "dataset": dataset, "pair_id": "pg_pair" if dataset == "pg19" else None,
                        "prompt_length": length, "numeric_file": relative})
    write_json(root / "summary.json", {"status": "completed", "sample_ids": ids, "samples": samples})
    return ids


def test_numeric_bundle_pipeline_uses_existing_values_and_preserves_sources(tmp_path):
    bundle = tmp_path / "bundle"
    ids = make_numeric_bundle(bundle)
    original = {p.relative_to(bundle): p.read_bytes() for p in bundle.rglob("*") if p.is_file()}
    output = tmp_path / "identity"
    result = run_analysis(bundle, output)
    assert result["status"] == "completed" and result["sample_ids"] == ids
    assert len(result["feature_names"]) == 16
    assert read_json(output / "summary.json") == result
    with np.load(output / "profiles.npz", allow_pickle=False) as profile:
        assert profile["values"].shape == (16, 6, 2, 3)
        feature = profile["feature_names"].tolist().index("full_v_middle_E16")
        with np.load(bundle / "samples/gov_8.npz", allow_pickle=False) as source:
            np.testing.assert_array_equal(profile["values"][feature, 2],
                                          source["full_v_middle_variance_coverage"][..., 1])
    expected_cohorts = {frozenset(ids), frozenset(ids[2:]), frozenset(ids[:-1]), frozenset(ids[:4] + ids[5:])}
    assert {frozenset(c["sample_ids"]) for c in result["cohorts"]} == expected_cohorts
    for cohort in result["cohorts"]:
        assert len(cohort["features"]) == 16
        assert (output / cohort["numeric_file"]).is_file()
    assert all((bundle / relative).read_bytes() == value for relative, value in original.items())
    published = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_analysis(bundle, output)
    assert (output / "summary.json").read_bytes() == published
