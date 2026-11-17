"""Independent accounting, causal updates and Dense identity controls."""
import csv

import numpy as np
import pytest

from experiments.head_analysis.analysis import dense_kv_memory as dense
from experiments.head_analysis.common.io import read_json, write_npz


def total(result, field):
    return int(result["counts"][..., dense.CI[field]].sum())


def test_partial_pages_forwarding_and_write_through_do_not_reread_new_rows():
    g = dense.Geometry(1, 1, 16, 4)
    common = g.page_bytes+g.row_bytes+256
    none = dense.simulate_reference(3, 3, common, "no_retention", g=g)
    assert total(none, "read_bytes") == 64*(3+4+5) == 768
    assert total(none, "write_bytes") == 192
    assert total(none, "read_commands") == 4
    assert total(none, "forward_only") == 1
    assert total(none, "lookups") == 0
    # All historical rows fit: only the first three Prompt rows are fetched.
    result = dense.simulate_reference(3, 3, common+2*(g.page_bytes+24), "global_lru", g=g)
    assert total(result, "read_bytes") == 192
    assert total(result, "hit_bytes") == 576
    assert total(result, "write_updates") == 1
    assert total(result, "lookups") == 8  # five reads plus three append checks
    assert result["final_valid_rows"] == {(0, 0, 0): 4, (0, 0, 1): 2}
    # A new Page write does not allocate it; the later read/forward can insert.
    result = dense.simulate_reference(4, 1, common+g.page_bytes+24, "global_lru", g=g)
    assert total(result, "write_updates") == total(result, "hits") == 0
    assert total(result, "read_bytes") == 256
    assert total(result, "inserts") == 2 and total(result, "evictions_caused") == 1
    dense.check_counts(4, 1, result, g)
    # Starting another request cannot reuse the prior invocation's cache.
    again = dense.simulate_reference(3, 3, common+2*(g.page_bytes+24), "global_lru", g=g)
    assert total(again, "read_bytes") == 192
    assert again["final_valid_rows"] == {
        (0, 0, 0): 4, (0, 0, 1): 2}


@pytest.mark.parametrize("length", [40, 41, 43])
@pytest.mark.parametrize("additional_slots", [0, 3, 9])
def test_exact_scan_matches_every_reference_counter(length, additional_slots):
    g = dense.Geometry(2, 4, 16, 4)
    score = np.array([[.3, .8, -.1, .2], [.2, .2, .8, .4]])
    common = g.page_bytes+g.heads*g.row_bytes+256
    capacity = common+12*g.head_count+additional_slots*(g.page_bytes+24)
    for policy in dense.POLICIES:
        actual = dense.simulate_fast(length, 9, capacity, policy, score, g)
        reference = dense.simulate_reference(length, 9, capacity, policy, score, g)
        np.testing.assert_array_equal(actual["counts"], reference["counts"])
        assert actual["peak_resident_pages"] == reference["peak_resident_pages"]
        dense.check_counts(length, 9, actual, g)
    # One physical KV read serves all four GQA Queries, without a factor of four.
    base = dense.simulate_fast(length, 9, capacity, "no_retention", score, g)
    assert total(base, "read_bytes") == g.head_count*g.row_bytes*(9*length+36)


def test_global_write_hits_before_layer_reads_are_not_lru_promotions():
    g = dense.Geometry(1, 4, 16, 4)
    capacity = g.page_bytes+g.heads*g.row_bytes+256+18*(g.page_bytes+24)
    fast = dense.simulate_fast(20, 8, capacity, "global_lru", g=g)
    reference = dense.simulate_reference(20, 8, capacity, "global_lru", g=g)
    assert total(fast, "write_updates") > 0 and total(fast, "hits") == 0
    np.testing.assert_array_equal(fast["counts"], reference["counts"])
    with pytest.raises(ValueError, match="reuse distance"):
        dense.simulate_fast(20, 8, capacity*100, "global_lru", g=g)
    with pytest.raises(ValueError, match="fully populated"):
        dense.simulate_fast(1, 3, capacity, "equal_pin", g=g)


def test_default_byte_budgets_quotas_and_identity_equivalence():
    g = dense.Geometry()
    score = np.tile(np.arange(8), (32, 1))
    expected_slots = {16: [0, 2040, 2040, 2042, 2042, 2042],
                      32: [0, 4082, 4082, 4086, 4086, 4086],
                      64: [0, 8166, 8166, 8174, 8174, 8174]}
    for mb, expected in expected_slots.items():
        for policy, slots in zip(dense.POLICIES, expected):
            b = dense.budget(mb*1024**2, policy, g)
            assert b["slots"] == slots
            assert b["reserved_bytes"]+b["unused_bytes"] == mb*1024**2
            assert b["data_bytes"]+b["tag_bytes"]+b["policy_bytes"]+12544 == b["reserved_bytes"]
        c = mb*1024**2
        a = dense.quotas(expected[4], "profile_pin", score, g)
        shifted = dense.quotas(expected[5], "shifted_profile_pin", score, g)
        np.testing.assert_array_equal(shifted, np.roll(a, -1, axis=1))
        assert a.sum() == expected[4] and a.max() <= 57
        x = dense.simulate_fast(8072, 6, c, "profile_pin", score, g)
        y = dense.simulate_fast(8072, 6, c, "shifted_profile_pin", score, g)
        assert total(x, "hit_bytes") == total(y, "hit_bytes") == 8192*expected[4]*5
        assert total(x, "read_commands") == total(y, "read_commands")
    # Equal scores retain ascending Head ID; no numerical tie perturbation.
    q = dense.quotas(36, "profile_pin", np.zeros((1, 8)), dense.Geometry(1, 8))
    np.testing.assert_array_equal(q, [[8, 7, 6, 5, 4, 3, 2, 1]])


def test_profile_uses_calib_sources_and_condition_weights_only(tmp_path):
    records, values = [], []
    for condition, count in enumerate([12, 12, 6, 6]):
        for i in range(count):
            records.append({"sample_id": f"c{condition}_{i}", "split": "calib",
                "dataset": "govreport" if condition < 2 else "pg19", "condition": str(condition),
                "source_group": f"gov:{condition}_{i}" if condition < 2 else f"pg:{i}"})
            values.append(np.full((2, 2), float(condition)))
    records += [{"sample_id": "ruler", "split": "calib", "dataset": "ruler"},
                {"sample_id": "validation", "split": "validation", "dataset": "pg19"},
                {"sample_id": "test", "split": "test", "dataset": "pg19"}]
    values += [np.full((2, 2), np.nan)]*3
    array = np.asarray(values)[:, :, :, None, None, None, None]
    def save(v):
        write_npz(tmp_path/"head_profiles.npz", {"sample_ids": np.array([r["sample_id"] for r in records]),
            "page_scopes": np.array(["all_visible"]), "rco_k_values": np.array([32]),
            "rco_deltas": np.array([32]), "rco_metrics": np.array(["rco_raw_mean"]),
            "rco_sample_head_values": v,
            "rco_sample_head_valid_pair_count": np.ones(v.shape[:-1], dtype=np.uint32)}, overwrite=True)
    save(array)
    score, selected, weights = dense.fit_profile(tmp_path, records)
    np.testing.assert_allclose(score, 1.5)
    assert len(selected) == 36 and len({r["source_group"] for r in selected}) == 30
    array[-2:] = 999999
    save(array)
    np.testing.assert_array_equal(dense.fit_profile(tmp_path, records)[0], score)
    assert weights.sum() == pytest.approx(1.)


def test_end_to_end_freezes_before_validation_and_exports_consistent_counters(tmp_path, monkeypatch):
    g = dense.Geometry(2, 2, 16, 4)
    score = np.array([[.8, .2], [.1, .9]])
    def sources(directory, validation_shift=0):
        cohorts = []
        for phase in ("calib", "validation"):
            rows = []
            for i in range(6):
                length = 128+i+(validation_shift if phase == "validation" else 0)
                path = directory/phase/str(i)
                step = np.arange(4, dtype=np.uint32)
                write_npz(path/"decode_stats_000.npz", {"decode_steps": step,
                    "query_positions": step+length, "visible_token_count": step+length+1,
                    "page_candidate_count_all_visible": (step+length+4)//4})
                rows.append({"sample_id": f"{phase}_{i}", "split": phase, "condition": f"condition_{i}",
                    "prompt_length": length, "decode_steps": 4, "b1_sample_dir": str(path)})
            cohorts.append(rows)
        return cohorts, g, score, cohorts[0], np.ones(6)/6
    original_check = dense.check_positions
    output = tmp_path/"result"
    def guarded(row, geometry):
        if row["split"] == "validation":
            assert read_json(output/"config.json")["status"] == "frozen_before_validation"
            assert (output/"profile.npz").exists()
        original_check(row, geometry)
    monkeypatch.setattr(dense, "check_positions", guarded)
    monkeypatch.setattr(dense, "load_inputs", lambda *a: sources(tmp_path/"source"))
    result = dense.run_analysis("runset", "root", "bundle", "b1", "codec", output, capacities=(4096,8192))
    assert result["status"] == "completed" and result["identity_equivalence_passed"]
    assert len(result["reference_check"]) == 6 and len(list(output.iterdir())) == 6
    with np.load(output/"profile.npz") as z:
        profile = {k: z[k] for k in z.files}
        assert profile["quotas"].dtype == np.uint32
    with np.load(output/"head_counters.npz") as z:
        counts = z["counts"]
        assert counts.shape == (12,2,6,2,2,2,len(dense.COUNTERS))
    with (output/"calib.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        assert int(row["reserved_bytes"]) <= int(row["capacity_bytes"])
        assert int(row["full_total_bytes"]) == sum(int(row[k]) for k in ("decode_total_bytes","prefill_write_bytes","quota_load_bytes"))
    # A changed Validation set cannot alter the fitted score, quotas or protocol.
    monkeypatch.setattr(dense, "check_positions", original_check)
    monkeypatch.setattr(dense, "load_inputs", lambda *a: sources(tmp_path/"changed_source",16))
    other = tmp_path/"changed_result"
    dense.run_analysis("runset", "root", "bundle", "b1", "codec", other, capacities=(4096,8192))
    assert read_json(output/"config.json") == read_json(other/"config.json")
    with np.load(other/"profile.npz") as z:
        for k in profile: np.testing.assert_array_equal(profile[k],z[k])
    assert (output/"validation.csv").read_bytes() != (other/"validation.csv").read_bytes()
