import json

import numpy as np

from experiments.head_analysis.analysis.page_rco import SENTINEL, analyze_completed_run, compute_page_rco


def make_arrays():
    page_ids = np.full((3, 1, 1, 2, 4), SENTINEL, dtype=np.uint32)
    page_ids[:, 0, 0, 0, :2] = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)
    page_ids[:, 0, 0, 1, :2] = np.asarray([[10, 11], [11, 12], [12, 13]], dtype=np.uint32)
    counts = np.asarray([4, 4, 4], dtype=np.uint32)
    valid = np.ones((3, 1, 1, 2), dtype=bool)
    return page_ids, counts, valid


def test_page_rco_matches_exact_overlap_and_random_baseline():
    page_ids, counts, valid = make_arrays()
    rows = compute_page_rco(
        sample_id="tiny",
        scope="all_visible",
        page_ids=page_ids[:, :, :, 0, :],
        candidate_counts=counts,
        scope_valid=valid[:, :, :, 0],
        k_values=(2,),
        deltas=(1,),
    )

    result = rows[0]
    assert result["valid_pair_count"] == 2
    assert result["overlap_raw_mean"] == 0.5
    assert result["random_baseline_mean"] == 0.5
    assert result["rco_raw_mean"] == 0.0
    assert result["rco_clipped_mean"] == 0.0


def test_page_rco_keeps_invalid_pairs_out_of_the_mean():
    page_ids, counts, valid = make_arrays()
    counts[1] = 2  # Later N_t == k is invalid by the frozen RCO definition.
    valid[1, 0, 0, 0] = False
    rows = compute_page_rco(
        sample_id="tiny",
        scope="all_visible",
        page_ids=page_ids[:, :, :, 0, :],
        candidate_counts=counts,
        scope_valid=valid[:, :, :, 0],
        k_values=(2,),
        deltas=(1,),
    )

    assert rows[0]["valid_pair_count"] == 0
    assert rows[0]["rco_clipped_mean"] is None


def test_completed_run_analyzes_the_two_scopes_independently(tmp_path):
    run_dir = tmp_path / "run"
    sample_dir = run_dir / "samples" / "tiny"
    sample_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(json.dumps({"status": "completed"}))
    (sample_dir / "sample_metadata.json").write_text(json.dumps({"status": "completed", "sample_id": "tiny"}))
    np.savez_compressed(
        sample_dir / "decode_stats_000.npz",
        decode_steps=np.asarray([0, 1], dtype=np.uint32),
        page_topk_ids_all_visible=np.asarray([[[[0, 1, 2, 3]]], [[[1, 2, 3, 4]]]], dtype=np.uint32),
        page_topk_ids_prompt_only=np.asarray([[[[8, 9, 10, 11]]], [[[9, 10, 11, 12]]]], dtype=np.uint32),
        page_candidate_count_all_visible=np.asarray([4, 4], dtype=np.uint32),
        page_candidate_count_prompt_only=np.asarray([4, 4], dtype=np.uint32),
        page_scope_valid=np.ones((2, 1, 1, 2), dtype=bool),
    )

    report = analyze_completed_run(run_dir, k_values=(2,), deltas=(1,))

    assert {record["scope"] for record in report["records"]} == {"all_visible", "prompt_only"}
    assert all(record["valid_pair_count"] == 1 for record in report["records"])
