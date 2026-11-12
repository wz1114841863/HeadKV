import json

import numpy as np
import pytest
import torch

from experiments.head_analysis.analysis.kv_representation import correlate_heads, run_analysis
from experiments.head_analysis.analysis.kv_representation_metrics import sampled_statistics
from experiments.head_analysis.collectors.kv_representation import (
    KVRepresentationConfig, KVRepresentationRecord, sample_positions,
)
from experiments.head_analysis.common.io import read_json, write_json, write_npz
from experiments.head_analysis.storage.kv_representation import KVRepresentationWriter


def test_head_relation_centers_values_before_ranking_and_uses_pairwise_support():
    x = np.asarray([[0, 1, 2], [100, 101, 102]], dtype=float)
    y = np.asarray([[2, 1, 0], [102, 101, 100]], dtype=float)
    result = correlate_heads(x, y)
    assert result["spearman"] > 0
    assert result["layer_centered_spearman"] == pytest.approx(-1)
    assert result["valid_head_count"] == 6
    x[0, 0] = np.nan
    result = correlate_heads(x, y)
    assert result["valid_head_count"] == result["layer_centered_valid_head_count"] == 5
    assert result["layer_centered_spearman"] == pytest.approx(-1)


def make_inputs(tmp_path):
    layers, heads, dimension, length = 2, 3, 16, 640
    rng = np.random.default_rng(3)
    run = tmp_path / "run"
    ids = ["govreport_tiny", "ruler_tiny"]
    positions = sample_positions(length)
    regions = np.repeat(np.arange(3, dtype=np.uint8), [32, 448, 32])
    for sample_id, dataset in zip(ids, ("govreport", "ruler")):
        directory = run / "samples" / sample_id
        writer = KVRepresentationWriter(directory, sample_id=sample_id, prompt_length=length,
                                         num_layers=layers, config=KVRepresentationConfig(decode_steps=()))
        for layer in range(layers):
            key = rng.normal(size=(heads, length, dimension)).astype(np.float32)
            value = rng.normal(size=key.shape).astype(np.float32)
            # RoPE-like position-dependent orthogonal sign changes: Norm unchanged,
            # while covariance is generally different from the original sampled Key.
            post = key.copy()
            post[:, ::2, :dimension // 2] *= -1
            arrays = {"sample_positions": positions.astype(np.uint32), "sample_regions": regions,
                      "k_pre": torch.from_numpy(key[:, positions]),
                      "k_post": torch.from_numpy(post[:, positions]),
                      "v": torch.from_numpy(value[:, positions])}
            for name, vectors in (("k_post", post), ("v", value)):
                statistics = [sampled_statistics(vectors[:, lo:hi]) for lo, hi in ((0, length), (64, length - 64))]
                for field in ("count", "mean", "covariance", "feature_energy", "norm_summary", "element_min", "element_max"):
                    arrays[f"{name}_{field}"] = np.stack([s[field] for s in statistics])
            writer.write_record(KVRepresentationRecord(sample_id, length, layer, 0, None, arrays))
        writer.finalize()
        write_json(directory / "sample_metadata.json", {
            "status": "completed", "sample_id": sample_id, "dataset": dataset,
            "prompt_length": length, "pair_id": None,
        })
    write_json(run / "run_metadata.json", {
        "status": "completed", "target_sample_ids": ids, "completed_sample_ids": ids,
        "source_b1_run_set": "b1-frozen.json", "fixed_dataset": {"prompts_path": "prompts.jsonl"},
        "model": "tiny", "dtype": "float32", "attention_backend": "sdpa",
    })
    b1 = tmp_path / "b1"
    write_json(b1 / "analysis_manifest.json", {"status": "completed", "source_run_set_sha256": "historical-source-id"})
    page = np.full((2, layers, heads, 1, 1, 1, 1), np.nan, dtype=np.float32)
    page[0, ..., 0, 0, 0, 0] = np.asarray([[.2, .8, .4], [.6, .3, .9]])
    count = np.zeros(page.shape[:-1], dtype=np.uint32)
    count[0] = 100
    token = np.zeros((2, layers, heads * 4, 1, 1, 1), dtype=np.float32)
    token[1, ..., 0, 0, 0] = np.tile(np.arange(heads * 4) / 12., (layers, 1))
    write_npz(b1 / "head_profiles.npz", {
        "sample_ids": np.asarray(ids), "page_scopes": np.asarray(["prompt_only"]),
        "rco_k_values": np.asarray([32]), "rco_deltas": np.asarray([32]),
        "rco_metrics": np.asarray(["rco_raw_mean"]), "rco_sample_head_values": page,
        "rco_sample_head_valid_pair_count": count, "token_k_values": np.asarray([32]),
        "token_deltas": np.asarray([1]), "token_metrics": np.asarray(["rco_raw_mean"]),
        "token_sample_head_metrics": token,
        "token_sample_head_valid_pair_count": np.full(token.shape[:-1], 5, dtype=np.uint32),
    })
    return run, b1


def test_pipeline_preserves_scopes_coordinates_and_reference_values(tmp_path):
    run, b1 = make_inputs(tmp_path)
    output = tmp_path / "analysis"
    result = run_analysis([run], output, b1_bundle=b1, permutation_replicates=3)
    assert result["status"] == "completed"
    assert result["sample_ids"] == ["govreport_tiny", "ruler_tiny"]
    stored = read_json(output / "summary.json")
    assert stored == result and len(stored["samples"]) == 2
    lines = [json.loads(line) for line in (output / "sample_summary.jsonl").read_text().splitlines()]
    assert lines == result["samples"]
    with np.load(output / "permutations.npz") as archive:
        permutations = archive["row_permutations"]
    rng = np.random.default_rng(20261111)
    np.testing.assert_array_equal(permutations, [rng.permutation(448) for _ in range(3)])
    for i, summary in enumerate(result["samples"]):
        with np.load(output / summary["numeric_file"]) as arrays:
            assert arrays["full_k_post_middle_effective_rank"].shape == (2, 3)
            np.testing.assert_array_equal(arrays["full_k_post_middle_count"], [512, 512])
            np.testing.assert_array_equal(arrays["sample_k_pre_all_count"], [512, 512])
            np.testing.assert_array_equal(arrays["sample_k_pre_middle_count"], [448, 448])
            assert arrays["cka_k_post_permutation_values"].shape == (2, 3, 3)
            assert arrays["relation_feature_values"].dtype == np.float64
            np.testing.assert_array_equal(arrays["cka_pairs"], [[0, 1], [0, 2], [1, 2]])
            np.testing.assert_allclose(arrays["full_v_middle_normalized_feature_energy"].sum(-1), 1, rtol=1e-6)
            assert len(summary["relations"]) == 14
            assert "sampling_representativeness" in summary and "rope_comparison" in summary
            assert summary["page_rco_eligible"] is (i == 0)
            if i == 0:
                expected = correlate_heads(arrays["full_k_post_middle_effective_rank"], arrays["b1_page_rco_raw_mean"])
                assert summary["relations"][0]["layer_centered_spearman"] == pytest.approx(expected["layer_centered_spearman"])
                assert summary["relations"][0]["layer_centered_valid_head_count"] == 6
                for values, relation in zip(arrays["relation_feature_values"], summary["relations"]):
                    recomputed = correlate_heads(values, arrays["b1_page_rco_raw_mean"])
                    observed, expected_rho = relation["layer_centered_spearman"], recomputed["layer_centered_spearman"]
                    assert (observed is None and not np.isfinite(expected_rho)) or observed == expected_rho
            else:
                assert all(row["layer_centered_spearman"] is None for row in summary["relations"])
                assert all(row["valid_head_count"] == 0 for row in summary["relations"])
                assert np.isnan(arrays["b1_page_rco_raw_mean"]).all()
                np.testing.assert_allclose(arrays["b1_token_rco_kv_head_mean"],
                                           np.tile([.125, 5.5 / 12, 9.5 / 12], (2, 1)))
    original = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_analysis([run], output, b1_bundle=b1, permutation_replicates=3)
    assert (output / "summary.json").read_bytes() == original


def test_duplicate_or_incomplete_sources_are_rejected_before_output(tmp_path):
    run, b1 = make_inputs(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        run_analysis([run, run], tmp_path / "duplicate", b1_bundle=b1)
    metadata = read_json(run / "run_metadata.json")
    other = tmp_path / "other_run"
    write_json(other / "run_metadata.json", {**metadata, "model": "other-model"})
    with pytest.raises(ValueError, match="different collection protocols: model"):
        run_analysis([run, other], tmp_path / "mixed", b1_bundle=b1)
    assert not (tmp_path / "mixed").exists()
    metadata["status"] = "failed"
    write_json(run / "run_metadata.json", metadata, overwrite=True)
    with pytest.raises(ValueError, match="incomplete"):
        run_analysis([run], tmp_path / "incomplete", b1_bundle=b1)
    assert not (tmp_path / "incomplete").exists()
