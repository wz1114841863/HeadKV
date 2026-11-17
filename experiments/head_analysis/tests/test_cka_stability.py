from itertools import combinations

import numpy as np
import pytest

from experiments.head_analysis.analysis.cka_stability import pair_matrix, run_analysis, top_weights
from experiments.head_analysis.common.io import read_json, write_json, write_npz


def test_top_pairs_share_boundary_ties_and_exclude_undefined_profiles():
    values = np.array([[9., 8., 8., 1.], [1., 1., 1., 1.], [9., 8., np.nan, 1.]])
    weights = top_weights(values, 2)
    np.testing.assert_array_equal(weights[0], [1., .5, .5, 0.])
    assert weights[0].sum() == 2
    assert np.isnan(weights[1:]).all()
    with pytest.raises(ValueError, match="pair count"):
        top_weights(values, 4)


def test_pair_matrix_uses_physical_head_labels_and_leaves_diagonal_missing():
    heads = np.array([2, 5, 7])
    pairs = np.array(list(combinations(heads, 2)))
    matrix = pair_matrix(np.array([[.1, .2, -.3]]), pairs, heads)
    np.testing.assert_array_equal(matrix[0, np.triu_indices(3, 1)[0], np.triu_indices(3, 1)[1]], [.1, .2, -.3])
    np.testing.assert_array_equal(matrix, matrix.swapaxes(-1, -2))
    assert np.isnan(np.diagonal(matrix, axis1=-2, axis2=-1)).all()


def _bundle(root, prefix, copies):
    heads, layers = np.array([0, 2, 5, 7]), np.array([3, 9])
    pairs = np.array(list(combinations(heads, 2)))
    records = []
    for copy in range(copies):
        for i, (dataset, length) in enumerate(zip(["ruler", "ruler", "govreport", "govreport", "pg19", "pg19"],
                                                [8192, 32768, 8192, 16384, 8192, 32768])):
            sid = f"{prefix}_{copy}_{i}"
            arrays = {"layer_ids": layers, "kv_head_ids": heads, "cka_pairs": pairs}
            for rep, base in zip(("k_pre", "k_post", "v"), ([.6, .2, .5, .1, .4, .3],
                                                          [.5, .3, .6, .1, .2, .4], [.1, .6, .2, .5, .3, .4])):
                observed = np.stack((base, np.asarray(base) * .5)) + i * .001 + copy * .01
                arrays[f"cka_{rep}_observed"] = observed
                arrays[f"cka_{rep}_excess"] = observed - .15  # Retain negative excess values.
            file = f"samples/{sid}.npz"
            write_npz(root / file, arrays)
            records.append({"sample_id": sid, "dataset": dataset, "pair_id": f"{prefix}_pair_{copy}" if dataset == "pg19" else None,
                            "prompt_length": length - 100, "target_prompt_length": length, "numeric_file": file})
    write_json(root / "summary.json", {"status": "completed", "sample_ids": [r["sample_id"] for r in records], "samples": records})
    return records


def test_cka_pipeline_only_reads_existing_values_and_groups_sources(tmp_path, monkeypatch):
    reference, extension, output = (tmp_path / name for name in ("reference", "extension", "output"))
    rows = _bundle(reference, "ref", 1) + _bundle(extension, "ext", 2)
    index = tmp_path / "sample_index.json"
    write_json(index, {"samples": rows[::-1]})
    before = {p: p.read_bytes() for root in (reference, extension) for p in root.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("This stage must not recompute SVD or CKA")

    monkeypatch.setattr(np.linalg, "svd", forbidden)
    monkeypatch.setattr(np.linalg, "eigvalsh", forbidden)
    monkeypatch.setattr("experiments.head_analysis.analysis.kv_representation.cka_statistics", forbidden)
    monkeypatch.setattr("experiments.head_analysis.analysis.kv_representation.spectrum_metrics", forbidden)
    result = run_analysis(extension, reference, output, sample_index=index, top_k=2, bootstrap_replicates=30)
    assert result == read_json(output / "summary.json") and result["status"] == "completed"
    assert len(result["features"]) == 6 and len(result["comparisons"]) == 9
    with np.load(output / "profiles.npz") as a:
        assert a["values"].shape == (6, 18, 2, 6)
        assert (a["values"][1::2] < 0).any()
        for i, row in enumerate(rows):
            with np.load((reference if i < 6 else extension) / row["numeric_file"]) as src:
                np.testing.assert_array_equal(a["values"][0, i], src["cka_k_pre_observed"])
    with np.load(output / "fixed_reference.npz") as a:
        np.testing.assert_allclose(a["per_layer_rho"], 1)
        np.testing.assert_allclose(a["top_k_retention"], 1)
        np.testing.assert_allclose(a["reference_top_weights"].sum(-1), 2)
    with np.load(output / "input_pairs.npz") as a:
        assert a["per_layer_rho"].shape == (6, 153, 2)
        assert a["group_extension_pg19_paired"].sum() == 2
    for rep in ("k_pre", "k_post", "v"):
        with np.load(output / f"matrices_{rep}.npz") as a:
            assert a["mean"].shape == (15, 2, 2, 4, 4)
            np.testing.assert_array_equal(a["mean"], a["mean"].swapaxes(-1, -2))
            assert np.isnan(np.diagonal(a["mean"], axis1=-2, axis2=-1)).all()
            assert (np.diagonal(a["valid_input_count"], axis1=-2, axis2=-1) == 0).all()
    with np.load(output / "bootstrap.npz") as a:
        assert a["draws_pg19"].shape == (30, 2)
        assert a["input_scores"].shape == (12, 27)
        assert (a["condition_valid_source_count"] == 2).all()
        assert (a["equal_condition_valid_replicates"] == 30).all()
        # Per-input scores are averaged once by source, then equally by condition.
        np.testing.assert_allclose(a["equal_condition_mean"][0], a["input_scores"].mean(0))
    assert all(p.read_bytes() == data for p, data in before.items())
    with pytest.raises(FileExistsError):
        run_analysis(extension, reference, output, sample_index=index)
