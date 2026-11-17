import json

import numpy as np
import pytest

from experiments.head_analysis.analysis import kv_reconstruction as joint
from experiments.head_analysis.common.io import read_json, write_json


def test_test_phase_rejects_missing_or_invalid_freeze_before_reading_inputs(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Test inputs must not be read before Validation publication")

    monkeypatch.setattr(joint, "read_sample", forbidden)
    monkeypatch.setattr(joint, "_load_models", forbidden)
    with pytest.raises(FileNotFoundError):
        joint.run_test(tmp_path)
    write_json(tmp_path / "validation_freeze.json", {"status": "not_frozen"})
    with pytest.raises(ValueError, match="frozen Validation"):
        joint.run_test(tmp_path)


def test_rank_gather_broadcasts_across_input_axis():
    curve = np.arange(3 * 2 * 4 * len(joint.RANKS)).reshape(3, 2, 4, len(joint.RANKS))
    ranks = np.array([[16, 32, 48, 64], [80, 96, 112, 128]])
    expected = np.array([[[curve[s, l, h, int(ranks[l, h] // 16 - 1)] for h in range(4)]
                          for l in range(2)] for s in range(3)])
    np.testing.assert_array_equal(joint._gather(curve, ranks), expected)


def _scores_fixture():
    conditions = ["govreport_8192"] * 3 + ["pg19_8192"] * 2 + ["pg19_32768"] * 2
    groups = ["g0", "g1", "g2", "pg19:a", "pg19:b", "pg19:a", "pg19:b"]
    records = [{"sample_id": f"sample_{i}", "condition": c, "source_group": g, "prompt_length": 600 + i}
               for i, (c, g) in enumerate(zip(conditions, groups))]
    scale = np.array([[[2., 8.]], [[3., 12.]]])
    base = np.array([1., 2., 3., 10., 14., 20., 24.])
    rank_factor = (129 - joint.RANKS.astype(float)) / 128
    error = np.broadcast_to(scale[None, None, ..., None] * base[None, :, None, None, None, None]
                            * rank_factor[None, None, None, None, None],
                            (len(joint.METHODS), len(records), 2, 1, 2, len(joint.RANKS))).copy()
    for name, factor in (("oracle_same_mean", .95), ("oracle_classic", .90), ("layer", 1.04), ("global", 1.08)):
        error[joint.METHODS.index(name)] *= factor
    strict_factors = np.array([.8, .9, 1., .6, 1., .95, .7])
    for rank in (32, 64, 96):
        error[joint.METHODS.index(f"head_aware_strict_{rank}")] *= strict_factors[:, None, None, None, None]
        error[joint.METHODS.index(f"head_aware_equal_{rank}")] *= .85
    energy = np.broadcast_to(scale[None] * (1 + np.arange(len(records))[:, None, None, None] / 10),
                             (len(records), 2, 1, 2)).copy()
    return records, {"scale": scale}, error, energy, base, strict_factors


def test_scores_decisions_and_source_bootstrap_keep_axes_and_pg19_pairs(tmp_path):
    records, models, error, energy, base, strict_factors = _scores_fixture()
    summary = joint.summarize(records, error, error * 1.02, energy, models)
    head = joint.METHODS.index("head"); ri = list(joint.RANKS).index(64)
    rank_factor = (129 - 64) / 128
    assert summary["input_scores"].shape == (7, len(joint.METHODS), 2, 8, len(joint.METRICS))
    np.testing.assert_allclose(summary["input_scores"][:, head, :, ri, 0], np.broadcast_to(base[:, None] * rank_factor, (7, 2)))
    # Three conditions are equal despite GovReport having three inputs versus two PG19 Pairs.
    expected_mean = np.mean([base[:3].mean(), base[3:5].mean(), base[5:].mean()]) * rank_factor
    np.testing.assert_allclose(summary["equal_condition_scores"][head, :, ri, 0], expected_mean)
    assert expected_mean != pytest.approx(base.mean() * rank_factor)
    expected_relative = np.sqrt(base / (1 + np.arange(7) / 10) * rank_factor)
    np.testing.assert_allclose(summary["input_scores"][:, head, :, ri, 2], np.broadcast_to(expected_relative[:, None], (7, 2)))
    decisions = joint.decide(summary, models)
    assert all(row["A1"]["passed"] and row["A2"]["selected_sharing"] == "global" for row in decisions.values())
    comparison = joint.comparisons(records, summary, tmp_path, "validation")
    with np.load(tmp_path / "validation_bootstrap.npz") as a:
        pg = a["draws_pg19"]; gov = a["draws_govreport_8192"]
        assert pg.shape == (2000, 2) and gov.shape == (2000, 3)
        reference = np.mean(np.stack((base[:3][gov].mean(1), base[3:5][pg].mean(1), base[5:][pg].mean(1))), axis=0)
        candidate = np.mean(np.stack(((base[:3] * strict_factors[:3])[gov].mean(1),
            (base[3:5] * strict_factors[3:5])[pg].mean(1),
            (base[5:] * strict_factors[5:])[pg].mean(1))), axis=0)
        expected = 1 - candidate / reference
        np.testing.assert_allclose(a["improvement_64"], np.broadcast_to(expected[:, None], (2000, 2)))
        np.testing.assert_allclose(comparison["64"]["ci95"], np.broadcast_to(np.quantile(expected, [.025, .975]), (2, 2)), atol=1e-7)


def _protocol_records(model):
    records = []
    for dataset, length, label in (("ruler", 8192, "8k"), ("ruler", 32768, "32k"),
                                    ("govreport", 8192, "8k"), ("govreport", 16384, "16k")):
        for i in range(1, 21):
            sid = f"{dataset}_{label}_{i:03d}"
            records.append({"sample_id": sid, "dataset": dataset, "target_prompt_length": length,
                "condition": f"{dataset}_{length}", "source_group": f"{dataset}:{sid}", "pair_id": None,
                "prompt_length": 600, "model": str(model), "num_layers": 1, "num_kv_heads": 2, "head_dim": 128})
    for i in range(1, 20, 2):
        pair = f"pg19_{i:03d}"
        for length, label in ((8192, "8k"), (32768, "32k")):
            records.append({"sample_id": f"{pair}_{label}", "dataset": "pg19", "pair_id": pair,
                "target_prompt_length": length, "condition": f"pg19_{length}", "source_group": f"pg19:{pair}",
                "prompt_length": 600, "model": str(model), "num_layers": 1, "num_kv_heads": 2, "head_dim": 128})
    return records


def test_tiny_full_protocol_publishes_validation_before_any_test_read(tmp_path, monkeypatch):
    import torch

    model = tmp_path / "model"
    config = {"hidden_size": 128, "num_attention_heads": 1, "head_dim": 128,
        "max_position_embeddings": 131072, "rope_theta": 500000.,
        "rope_scaling": {"rope_type": "llama3", "factor": 8., "low_freq_factor": 1.,
                         "high_freq_factor": 4., "original_max_position_embeddings": 8192}}
    write_json(model / "config.json", config)
    records = _protocol_records(model)
    rng = np.random.default_rng(9)
    raw = torch.from_numpy(rng.standard_normal((3, 1, 2, 448, 128)).astype(np.float32)).to(torch.bfloat16).float().numpy()
    raw_before = raw.copy()
    mean = raw[[1, 2]].astype(float).mean(-2)
    centered = raw[[1, 2]].astype(float) - mean[..., None, :]
    covariance = centered.swapaxes(-1, -2) @ centered / 448
    positions = 64 + ((2*np.arange(448)+1)*(600-128))//896
    output = tmp_path / "result"
    calls = []

    def read(record, include_vectors=False):
        if record["split"] == "test":
            assert (output / "validation_freeze.json").exists(), "Test data was opened before Validation froze"
        calls.append((record["split"], include_vectors))
        values = {"mean": mean, "covariance": covariance, "count": np.full((2, 1), 472), "positions": positions}
        if include_vectors:
            values["vectors"] = raw
        return values

    monkeypatch.setattr(joint, "discover_samples", lambda *_: records)
    monkeypatch.setattr(joint, "preflight", lambda rows: {"status": "passed", "sample_count": len(rows)})
    monkeypatch.setattr(joint, "read_sample", read)
    joint.prepare([], tmp_path / "unused_inputs.jsonl", output)
    assert not calls and read_json(output / "split.json")["counts"] == {"calib": 60, "validation": 20, "test": 20}
    decisions = joint.run_validation(output)
    decisions = joint._json_value(decisions)
    assert not any(split == "test" for split, _ in calls)
    freeze_before = (output / "validation_freeze.json").read_bytes()
    fixed_before = {name: (output / name).read_bytes() for name in
                    ("calibration.npz", "rope_calibration.npz", "split.json", "protocol.json")}
    assert read_json(output / "validation_freeze.json")["decisions"] == decisions
    joint.run_test(output)
    assert any(split == "test" for split, _ in calls)
    assert freeze_before == (output / "validation_freeze.json").read_bytes()
    assert all((output / name).read_bytes() == value for name, value in fixed_before.items())
    np.testing.assert_array_equal(raw, raw_before)
    assert read_json(output / "summary.json")["status"] == "completed"
    report = read_json(output / "test_report.json")
    assert report["validation_decisions_unchanged"] == decisions
    with np.load(output / "test_errors.npz") as a:
        assert len(a["sample_ids"]) == 20 and a["ideal_mse"].shape == (len(joint.METHODS), 20, 2, 1, 2, 8)
        assert a["bf16_mse"].shape == a["sample_ideal_mse"].shape == a["ideal_mse"].shape
    with pytest.raises(ValueError, match="new Test result"):
        joint.run_test(output)
