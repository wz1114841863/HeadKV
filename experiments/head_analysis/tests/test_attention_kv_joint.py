import numpy as np
import pytest

from experiments.head_analysis.analysis.attention_kv_joint import (
    PRIMARY_FEATURES, RCO_ID, decompose_condition, load_inputs, response_bootstrap, run_analysis,
)
from experiments.head_analysis.common.io import read_json, write_json, write_npz


def _profiles():
    head = np.array([[-6., -2., 2., 6.], [-12., -4., 4., 12.]])
    interaction = np.array([[1., -1., -1., 1.], [-2., 2., 2., -2.]])
    t = np.array([-3., -1., 1., 3.])[:, None, None]
    x = 100 + head + t * interaction + t * 10
    y = 200 + head - t * interaction - t * 5
    return x[None], y


def test_fixed_head_relation_and_input_response_can_have_opposite_signs():
    x, y = _profiles()
    result = decompose_condition(x, y)
    np.testing.assert_allclose(result["metrics"], [[1., -1., -1.]])
    np.testing.assert_allclose(result["within_head_response_rho"], -1)
    np.testing.assert_allclose(result["x_residual"].mean(1), 0, atol=1e-14)
    # A pure fixed pattern plus whole-Layer input shifts contains no relative response.
    fixed = np.tile(np.array([[-6., -2., 2., 6.]]), (4, 2, 1)) + np.arange(4)[:, None, None] * 10
    constant = decompose_condition(fixed[None], fixed)
    assert constant["metrics"][0, 0] == pytest.approx(1)
    assert np.isnan(constant["metrics"][0, 1:]).all()
    assert constant["x_relative_fixed_share"][0] == pytest.approx(1)


def test_response_bootstrap_refits_means_and_handles_missing_and_degenerate_draws():
    x, y = _profiles()
    x[:, 1, 0, 0] = np.nan
    result = decompose_condition(x, y)
    assert not result["valid_mask"][0, 0, 0]
    draws = np.array([[0, 1, 1, 3], [3, 2, 0, 2], [1, 1, 1, 1]])
    actual = response_bootstrap(result, draws, batch_size=2)
    xx = np.where(result["valid_mask"][:, None], x, np.nan)
    yy = np.where(result["valid_mask"][0][None], y, np.nan)
    for i, draw in enumerate(draws[:2]):
        expected = decompose_condition(xx[:, draw], yy[draw])["metrics"]
        np.testing.assert_allclose(actual[i], expected, atol=1e-12)
    assert np.isnan(actual[2, :, 1:]).all()


def _packages(root):
    reference, extension, b2 = (root / name for name in ("reference", "extension", "b2"))
    all_rows, all_y = [], []
    for prefix, copies, directory in (("ref", 1, reference), ("ext", 3, extension)):
        rows = []
        for copy in range(copies):
            for condition, (dataset, length) in enumerate(zip(["ruler", "ruler", "govreport", "govreport", "pg19", "pg19"],
                                                             [8192, 32768, 8192, 16384, 8192, 32768])):
                sid = f"{prefix}_{copy}_{condition}"
                base = np.array([[-3., -1., 1., 3.], [-6., -2., 2., 6.]])
                response = copy * np.array([[1., -1., -1., 1.], [-1., 1., 1., -1.]])
                values = np.stack((60 + base + response, .6 - .01 * (base + response),
                                   60 - base + response, .6 + .01 * (base - response), .5 + .02 * base + .01 * response))
                y = (.5 + base * .02 + response * .003).astype(np.float32)
                count = np.full((2, 4), 100 + copy, dtype=np.uint64)
                if dataset == "ruler":
                    y[:] = np.nan; count[:] = 0
                relative = f"samples/{sid}.npz"
                write_npz(directory / relative, {"layer_ids": np.arange(2), "kv_head_ids": np.arange(4),
                    "relation_feature_names": np.array(PRIMARY_FEATURES), "relation_feature_values": values,
                    "full_k_post_middle_norm_summary": np.repeat(values[0, ..., None], 5, axis=-1),
                    "full_v_middle_norm_summary": np.repeat(values[2, ..., None], 5, axis=-1),
                    "b1_page_rco_raw_mean": y, "b1_page_rco_valid_pair_count": count})
                row = {"sample_id": sid, "dataset": dataset, "pair_id": f"{prefix}_pair_{copy}" if dataset == "pg19" else None,
                       "prompt_length": length - 100, "target_prompt_length": length, "numeric_file": relative}
                rows.append(row); all_rows.append(row); all_y.append((y, count))
        write_json(directory / "summary.json", {"status": "completed", "sample_ids": [r["sample_id"] for r in rows],
                                                 "samples": rows, "definitions": {"rco": RCO_ID}})
    write_json(b2 / "sample_index.json", {"samples": all_rows[::-1]})
    ys, counts = (np.stack([r[i] for r in all_y[::-1]])[:, None] for i in (0, 1))
    write_npz(b2 / "stages/stage4/stage4_joint_profiles.npz", {"sample_ids": np.array([r["sample_id"] for r in all_rows[::-1]]),
        "y_ids": np.array([RCO_ID]), "sample_y": ys, "sample_y_source_count_min": counts,
        "sample_y_source_count_max": counts, "sample_y_valid_source_heads": (counts > 0).astype(np.uint32)})
    return extension, reference, b2


def test_joint_pipeline_aligns_b1_rows_excludes_ruler_and_preserves_inputs(tmp_path):
    extension, reference, b2 = _packages(tmp_path)
    inputs = load_inputs(extension, reference, b2)
    assert inputs["values"].shape == (7, 24, 2, 4)
    before = {p: p.read_bytes() for root in (extension, reference, b2) for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "result"
    result = run_analysis(extension, reference, output, b2_bundle=b2, bootstrap_replicates=30)
    assert result == read_json(output / "summary.json") and result["status"] == "completed"
    assert result["eligible_sample_count"] == 16 and result["extension_eligible_sample_count"] == 12
    assert result["extension_source_count"] == 9 and len(result["D2"]) == 8
    with np.load(output / "d1_relations.npz") as a:
        ruler = inputs["datasets"] == "ruler"
        assert np.isnan(a["rho"][:, ruler]).all() and (a["valid_head_count"][:, ruler] == 0).all()
        assert (a["valid_head_count"][:, ~ruler] == 8).all()
    with np.load(output / "d1_bootstrap.npz") as a:
        assert a["draws_pg19"].shape == (30, 3)
        assert (a["condition_valid_source_count"] == 3).all()
    assert all(path.read_bytes() == value for path, value in before.items())
    with pytest.raises(FileExistsError):
        run_analysis(extension, reference, output, b2_bundle=b2)
    file = extension / "samples/ext_0_2.npz"
    with np.load(file) as a:
        arrays = {key: a[key] for key in a.files}
    arrays["b1_page_rco_raw_mean"][0, 0] += .01
    write_npz(file, arrays, overwrite=True)
    with pytest.raises(ValueError, match="RCO/count values differ"):
        load_inputs(extension, reference, b2)
