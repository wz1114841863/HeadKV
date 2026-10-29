import hashlib
import json

import pytest

from experiments.head_analysis.runners import pilot_preflight


def _write_frozen_pilot(tmp_path):
    records = []
    for index, sample_id in enumerate(pilot_preflight.PILOT_SAMPLE_IDS, 1):
        input_ids = [index, index + 1]
        records.append({
            "sample_id": sample_id, "dataset": sample_id.split("_", 1)[0],
            "actual_prompt_length": len(input_ids),
            "max_new_tokens": 128 if sample_id.startswith("ruler") else 1024,
            "input_ids": input_ids,
            "input_sha256": hashlib.sha256(json.dumps(input_ids, separators=(",", ":")).encode()).hexdigest(),
        })
    input_path = tmp_path / "prompts.jsonl"
    input_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    data_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"data_sha256": data_sha}))
    return input_path, manifest_path, data_sha


def _gpu(*, free_gib=40, pids=()):
    return pilot_preflight.GPUInspection("cuda:4", 4, "A100", free_gib * 1024**3, tuple(pids))


def _build(tmp_path, **overrides):
    input_path, manifest_path, data_sha = _write_frozen_pilot(tmp_path)
    model = tmp_path / "model"
    tokenizer = tmp_path / "tokenizer"
    model.mkdir()
    tokenizer.mkdir()
    kwargs = {
        "input_path": input_path, "manifest_path": manifest_path, "tokenizer_path": tokenizer, "model_path": model,
        "device": "cuda:4", "run_dir": tmp_path / "run", "baseline_output": tmp_path / "baseline.json",
        "expected_commit": "abc123", "python_executable": ".venv/bin/python",
        "minimum_free_memory_bytes": 32 * 1024**3, "minimum_free_disk_bytes": 200 * 1024**3,
        "expected_data_sha256": data_sha, "gpu": _gpu(), "disk_free_bytes": 300 * 1024**3,
        "observed_git": {"commit": "abc123"}, "status_lines": ("?? data/",),
    }
    kwargs.update(overrides)
    return pilot_preflight.build_pilot_plan(**kwargs)


def test_build_pilot_plan_freezes_ids_configuration_and_launch_arguments(tmp_path):
    plan = _build(tmp_path)
    assert plan["status"] == "ready"
    assert plan["fixed_dataset"]["selected_sample_ids"] == list(pilot_preflight.PILOT_SAMPLE_IDS)
    assert plan["collection"] == pilot_preflight.PILOT_COLLECTION
    assert plan["resource_budget"] == pilot_preflight.PILOT_BUDGET
    assert plan["host_preflight"]["gpu"]["compute_pids"] == []
    assert plan["launch_arguments"]["baseline"][:3] == [
        ".venv/bin/python", "-m", "experiments.head_analysis.runners.baseline_decode"
    ]
    assert plan["launch_arguments"]["b1_collection"].count("--sample-id") == 10
    assert "--verify-baseline" in plan["launch_arguments"]["b1_collection"]


def test_build_pilot_plan_rejects_busy_gpu(tmp_path):
    with pytest.raises(pilot_preflight.PilotPreflightError, match="active compute PIDs"):
        _build(tmp_path, gpu=_gpu(pids=(1234,)))


def test_build_pilot_plan_rejects_unexpected_git_changes(tmp_path):
    with pytest.raises(pilot_preflight.PilotPreflightError, match="unexpected working-tree changes"):
        _build(tmp_path, status_lines=(" M experiments/head_analysis/runners/b1_runner.py",))


def test_write_json_new_refuses_overwrite(tmp_path):
    output = tmp_path / "plan.json"
    pilot_preflight.write_json_new(output, {"status": "ready"})
    assert json.loads(output.read_text()) == {"status": "ready"}
    with pytest.raises(pilot_preflight.PilotPreflightError, match="already exists"):
        pilot_preflight.write_json_new(output, {"status": "ready"})
