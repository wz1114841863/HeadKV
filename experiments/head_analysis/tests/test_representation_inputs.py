import json
from pathlib import Path

import numpy as np
import pytest

from experiments.head_analysis.common.io import write_json, write_npz
from experiments.head_analysis.storage.representation_inputs import (
    discover_samples, freeze_split, preflight, read_sample,
)


def _split_records():
    rows = []
    for dataset, length, label in (("ruler", 8192, "8k"), ("ruler", 32768, "32k"),
                                    ("govreport", 8192, "8k"), ("govreport", 16384, "16k")):
        for i in range(1, 21):
            sid = f"{dataset}_{label}_{i:03d}"
            rows.append({"sample_id": sid, "dataset": dataset, "target_prompt_length": length,
                         "source_group": f"{dataset}:{sid}", "pair_id": None})
    for i in range(1, 20, 2):
        pair = f"pg19_{i:03d}"
        for length, label in ((8192, "8k"), (32768, "32k")):
            rows.append({"sample_id": f"{pair}_{label}", "dataset": "pg19", "pair_id": pair,
                         "target_prompt_length": length, "source_group": f"pg19:{pair}"})
    return rows


def test_frozen_source_split_matches_predeclared_seed_and_pair_membership(tmp_path):
    rows = _split_records()
    result = freeze_split(rows[::-1], tmp_path / "split.json")
    assert result["counts"] == {"calib": 60, "validation": 20, "test": 20}
    assert result["source_counts"] == {"calib": 54, "validation": 18, "test": 18}
    assert not any("split" in row for row in rows)
    by_id = {r["sample_id"]: r["split"] for r in result["samples"]}
    assert {sid for sid, split in by_id.items() if sid.startswith("ruler_8k") and split == "validation"} == {
        "ruler_8k_007", "ruler_8k_013", "ruler_8k_018", "ruler_8k_020"}
    assert {sid for sid, split in by_id.items() if sid.startswith("govreport_16k") and split == "test"} == {
        "govreport_16k_012", "govreport_16k_014", "govreport_16k_017", "govreport_16k_018"}
    for pair, split in (("pg19_013", "validation"), ("pg19_017", "validation"),
                        ("pg19_009", "test"), ("pg19_015", "test"), ("pg19_001", "calib")):
        assert by_id[f"{pair}_8k"] == by_id[f"{pair}_32k"] == split
    with pytest.raises(FileExistsError):
        freeze_split(rows, tmp_path / "split.json")
    duplicate = rows.copy(); duplicate[-1] = duplicate[-2]
    with pytest.raises(ValueError, match="100 unique"):
        freeze_split(duplicate, tmp_path / "bad.json")


def _collection(tmp_path):
    sid, length, heads, dimension, layers = "govreport_8k_001", 600, 2, 4, 2
    run_dir = tmp_path / "run"
    sample_dir = run_dir / "samples" / sid
    fixed = {"sample_id": sid, "dataset": "govreport", "pair_id": None, "source_id": "report_1",
             "actual_prompt_length": length, "target_prompt_length": 8192, "input_ids": [1] * length}
    inputs = tmp_path / "prompts.jsonl"
    inputs.write_text(json.dumps(fixed) + "\n")
    write_json(run_dir / "run_metadata.json", {"status": "completed", "model": "tiny-model",
        "target_sample_ids": [sid], "completed_sample_ids": [sid]})
    write_json(sample_dir / "sample_metadata.json", {"status": "completed", "sample_id": sid,
        "dataset": "govreport", "pair_id": None, "prompt_length": length, "num_layers": layers, "num_kv_heads": heads})
    metadata = {"status": "completed", "sample_id": sid, "prompt_length": length, "num_layers": layers,
                "covariance_ddof": 0, "scope_order": ["prompt", "middle"], "files": {}}
    positions = np.concatenate((np.arange(32), 64 + ((2*np.arange(448)+1)*(length-128))//896, np.arange(length-32, length)))
    regions = np.repeat(np.array([0, 1, 2], dtype=np.uint8), [32, 448, 32])
    for layer in range(layers):
        arrays = {"sample_positions": positions.astype(np.uint32), "sample_regions": regions}
        fields = {}
        for index, rep in enumerate(("k_pre", "k_post", "v")):
            # IEEE BF16 bit patterns of +1 and -2, not integer casts of their values.
            bits = np.full((heads, 512, dimension), 0x3F80 if index != 2 else 0xC000, dtype=np.uint16)
            arrays[rep] = bits
            fields[rep] = {"tensor_dtype": "bfloat16", "stored_dtype": "uint16", "shape": list(bits.shape)}
        for rep in ("k_post", "v"):
            arrays[f"{rep}_count"] = np.array([length, length-128], dtype=np.uint32)
            arrays[f"{rep}_mean"] = np.full((2, heads, dimension), layer, dtype=np.float32)
            arrays[f"{rep}_covariance"] = np.broadcast_to(np.eye(dimension, dtype=np.float32), (2, heads, dimension, dimension)).copy()
            # A sentinel verifies that the reader never loads the irrelevant Norm arrays.
            arrays[f"{rep}_norms"] = np.array(["not needed"], dtype=object)
        relative = f"prefill/layer_{layer:03d}.npz"
        write_npz(sample_dir / relative, arrays)
        metadata["files"][relative] = {"layer_id": layer, "decode_step": None, "fields": fields}
    write_json(sample_dir / "metadata.json", metadata)
    return run_dir, inputs


def test_reader_checks_both_scopes_and_restores_original_bf16_without_loading_norms(tmp_path):
    run, inputs = _collection(tmp_path)
    records = discover_samples([run], inputs)
    record = records[0]
    assert record["head_dim"] == 4 and record["prompt_length"] == 600
    values = read_sample(record, include_vectors=True)
    assert values["mean"].shape == (2, 2, 2, 4)
    assert values["covariance"].shape == (2, 2, 2, 4, 4)
    assert values["vectors"].shape == (3, 2, 2, 448, 4)
    np.testing.assert_array_equal(values["count"], [[472, 472], [472, 472]])
    np.testing.assert_array_equal(values["vectors"][:2], 1.)
    np.testing.assert_array_equal(values["vectors"][2], -2.)
    assert values["vectors"].dtype == np.float32
    events = []
    summary = preflight(records, events.append)
    assert summary["status"] == "passed" and summary["prefill_layer_files"] == 2
    assert summary["reconstruction_errors_computed"] is False
    assert events[0]["sample_id"] == record["sample_id"]


def test_reader_rejects_wrong_prompt_count_nonfinite_vectors_and_missing_layers(tmp_path):
    run, inputs = _collection(tmp_path)
    record = discover_samples([run], inputs)[0]
    file = Path(record["sample_dir"]) / "prefill/layer_000.npz"
    with np.load(file, allow_pickle=True) as a:
        arrays = {key: a[key] for key in a.files}
    arrays["k_post_count"][0] = 599
    write_npz(file, arrays, overwrite=True)
    with pytest.raises(ValueError, match="Prompt/middle"):
        read_sample(record)
    arrays["k_post_count"][0] = 600
    arrays["k_pre"][0, 0, 0] = 0x7F80
    write_npz(file, arrays, overwrite=True)
    with pytest.raises(ValueError, match="nonfinite"):
        preflight([record], lambda _: None)
    metadata_file = Path(record["sample_dir"]) / "metadata.json"
    metadata = json.loads(metadata_file.read_text())
    metadata["files"].pop("prefill/layer_001.npz")
    write_json(metadata_file, metadata, overwrite=True)
    with pytest.raises(ValueError, match="coverage"):
        read_sample(record)


def test_layer_selection_reads_only_requested_file_and_keeps_layer_axis(tmp_path):
    run, inputs = _collection(tmp_path)
    record = discover_samples([run], inputs)[0]
    full = read_sample(record)
    (Path(record["sample_dir"]) / "prefill/layer_000.npz").unlink()
    selected = read_sample(record, layer_ids=[1])
    np.testing.assert_array_equal(selected["mean"], full["mean"][:, 1:2])
    np.testing.assert_array_equal(selected["covariance"], full["covariance"][:, 1:2])
    with pytest.raises(FileNotFoundError):
        read_sample(record)
