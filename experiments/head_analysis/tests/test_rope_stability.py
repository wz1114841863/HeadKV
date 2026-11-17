import numpy as np
import pytest

from experiments.head_analysis.analysis.rope_stability import (
    increase_fraction, pg19_pairs, rank_changes, run_analysis,
)
from experiments.head_analysis.common.io import read_json, write_json, write_npz


def test_rank_change_sign_and_missing_support():
    values = rank_changes([2., 4., 8., 3., 0., np.nan], [4., 2., 8., np.nan, 2., 1.])
    np.testing.assert_allclose(values[2, :3], [2., -2., 0.])
    np.testing.assert_allclose(values[3, :3], [2., .5, 1.])
    assert np.isnan(values[:, 3:]).all()
    assert increase_fraction(values[2]) == pytest.approx(1 / 3)
    assert np.isnan(increase_fraction(np.array([np.nan])))
    with pytest.raises(ValueError, match="matching"):
        rank_changes([1.], [2., 3.])


def test_pg19_lengths_are_matched_by_source_in_shuffled_rows():
    datasets = np.array(["pg19", "govreport", "pg19", "pg19", "pg19"])
    groups = np.array(["b", "g", "a", "b", "a"])
    lengths = np.array([32768, 8192, 8192, 8192, 32768])
    pairs, names = pg19_pairs(datasets, groups, lengths)
    np.testing.assert_array_equal(names, ["a", "b"])
    np.testing.assert_array_equal(pairs, [[2, 4], [3, 0]])
    with pytest.raises(ValueError, match="two distinct lengths"):
        pg19_pairs(datasets[:-1], groups[:-1], lengths[:-1])


def _bundle(root, prefix, copies):
    datasets = ["ruler", "ruler", "govreport", "govreport", "pg19", "pg19"]
    lengths = [8192, 32768, 8192, 16384, 8192, 32768]
    samples, index = [], []
    for copy in range(copies):
        for row, (dataset, length) in enumerate(zip(datasets, lengths)):
            sid = f"{prefix}_{copy}_{row}"
            pre = np.array([[30., 40., 50.], [30., 40., 50.]]) + copy * .3
            delta = np.array([[-3., 8., 20.], [12., -4., 18.]]) + length / 8192 * .5
            numeric_file = f"samples/{sid}.npz"
            write_npz(root / numeric_file, {"layer_ids": np.array([25, 29]), "kv_head_ids": np.array([5, 6, 7]),
                "sample_k_pre_middle_effective_rank": pre,
                "sample_k_post_middle_effective_rank": pre + delta})
            pair_id = f"{prefix}_pair_{copy}" if dataset == "pg19" else None
            sample = {"sample_id": sid, "dataset": dataset, "pair_id": pair_id,
                      "prompt_length": length - 100, "numeric_file": numeric_file}
            samples.append(sample)
            index.append({**sample, "target_prompt_length": length})
    write_json(root / "summary.json", {"status": "completed", "sample_ids": [s["sample_id"] for s in samples],
                                      "samples": samples})
    return index


def test_rope_pipeline_reference_pairing_and_numeric_outputs(tmp_path):
    reference, extension, output = (tmp_path / name for name in ("reference", "extension", "output"))
    records = _bundle(reference, "ref", 1) + _bundle(extension, "ext", 2)
    index = tmp_path / "sample_index.json"
    write_json(index, {"samples": records[::-1]})
    original = {p: p.read_bytes() for root in (reference, extension) for p in root.rglob("*") if p.is_file()}
    result = run_analysis(extension, reference, output, sample_index=index, bootstrap_replicates=30)
    assert result["status"] == "completed" and read_json(output / "summary.json") == result
    assert len(result["extension_sample_ids"]) == 12 and len(result["reference_sample_ids"]) == 6
    for exception in result["exceptions"]:
        stats = exception["cohorts"]["extension"]
        assert stats["decrease_count"] == stats["valid_count"] == 12
    with np.load(output / "profiles.npz") as p:
        assert p["values"].shape == (4, 18, 2, 3)
        np.testing.assert_allclose(p["values"][2], p["values"][1] - p["values"][0])
        np.testing.assert_allclose(p["values"][3], p["values"][1] / p["values"][0])
    with np.load(output / "fixed_reference.npz") as fixed:
        np.testing.assert_allclose(fixed["per_layer_rho"][0], 1)
        np.testing.assert_allclose(fixed["skill"][0], 1, atol=1e-14)
        # Reconstruct the reference directly; extension inputs cannot change it.
        ref_delta = np.stack([np.array([[-3., 8., 20.], [12., -4., 18.]]) + r["target_prompt_length"] / 8192 * .5
                              for r in records[:6]])
        np.testing.assert_allclose(fixed["reference_profile"][0], (ref_delta - ref_delta.mean(-1, keepdims=True)).mean(0))
    with np.load(output / "pg19_pairs.npz") as paired:
        assert paired["long_minus_short"].shape == (4, 3, 2, 3)
        np.testing.assert_allclose(paired["long_minus_short"][2], 1.5)
        assert paired["is_reference_pair"].sum() == 1
    with np.load(output / "input_pairs.npz") as pairs:
        assert pairs["per_layer_rho"].shape == (2, 153, 2)
        assert pairs["group_extension_pg19_paired"].sum() == 2
        assert pairs["group_extension_all"].sum() == 66
    with np.load(output / "bootstrap.npz") as b:
        assert b["draws_pg19"].shape == (30, 2)
        assert b["condition_valid_source_count"].min() == 2
        assert (b["equal_condition_valid_replicates"] == 30).all()
        pg = b["length_contrast_names"].tolist().index("pg19_32768_minus_pg19_8192")
        np.testing.assert_allclose(b["length_contrast_mean"][pg, 2], 1.5)
        np.testing.assert_allclose(b["length_contrast_ci95"][pg, 2], 1.5)
    assert all(path.read_bytes() == content for path, content in original.items())
    with pytest.raises(FileExistsError):
        run_analysis(extension, reference, output, sample_index=index)
