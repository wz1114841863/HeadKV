import hashlib
import json
from argparse import Namespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.head_analysis.runners import b1_runner
from experiments.head_analysis.runners import baseline_decode


class TinyTokenizer:
    pad_token_id = None
    eos_token_id = None


def make_record():
    input_ids = [1, 2, 3, 4]
    return {
        "sample_id": "tiny_8k",
        "dataset": "unit",
        "task": "continuation",
        "pair_id": None,
        "actual_prompt_length": len(input_ids),
        "max_new_tokens": 3,
        "input_ids": input_ids,
        "input_sha256": hashlib.sha256(json.dumps(input_ids, separators=(",", ":")).encode()).hexdigest(),
        "metadata": {},
    }


def make_model():
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            eos_token_id=None,
            pad_token_id=None,
        )
    ).eval()
    model.config._attn_implementation = "sdpa"
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = None
    return model


def test_runner_uses_frozen_ids_and_writes_probe_collector_trace(tmp_path, monkeypatch):
    record = make_record()
    input_path = tmp_path / "prompts.jsonl"
    input_path.write_text(json.dumps(record) + "\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"data_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()}))
    model = make_model()
    monkeypatch.setattr(b1_runner.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TinyTokenizer())
    monkeypatch.setattr(b1_runner, "load_model", lambda *args, **kwargs: (model, torch.device("cpu")))
    args = Namespace(
        input=str(input_path),
        manifest=str(manifest_path),
        tokenizer="unused",
        model="unused",
        run_dir=str(tmp_path / "run"),
        sample_id=[record["sample_id"]],
        device="cpu",
        dtype="float32",
        key_block_size=2,
        prefill_window=2,
        token_topk=5,
        save_token_topk=True,
        page_size=2,
        page_topk=3,
        decode_chunk_steps=128,
        decode_cap=None,
        seed=0,
        verify_baseline=None,
    )

    assert b1_runner.run_collection(args) == 0

    sample_dir = tmp_path / "run" / "samples" / record["sample_id"]
    assert (sample_dir / "prefill_stats.npz").is_file()
    with torch.no_grad(), __import__("numpy").load(sample_dir / "decode_stats_000.npz") as trace:
        assert trace["decode_steps"].tolist() == [0, 1]
        assert trace["scalar_features"].shape == (2, 2, 4, 28)
    metadata = json.loads((sample_dir / "sample_metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["decode_trace_length"] == 2
    resource = json.loads((tmp_path / "run" / "run_metadata.json").read_text())["resource_summary"]["per_sample"][0]
    assert resource["generation_end_to_end_seconds"] >= 0
    assert resource["forward_timing"]["prefill"]["count"] == 1
    assert resource["forward_timing"]["decode"]["count"] == 2
    assert resource["trace_write_wall_seconds"] >= 0
    assert resource["trace_storage"]["logical_array_bytes"] > 0
    assert resource["trace_storage"]["artifact_file_bytes"] > 0


def test_runner_requires_exact_ids_from_independent_baseline(tmp_path, monkeypatch):
    record = make_record()
    input_path = tmp_path / "prompts.jsonl"
    input_path.write_text(json.dumps(record) + "\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"data_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()}))
    model = make_model()
    baseline_output = model.generate(torch.tensor([record["input_ids"]]), max_new_tokens=3, do_sample=False)[
        0, 4:
    ].tolist()
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": "headkv-baseline-decode-v1",
                "status": "completed",
                "fixed_dataset": {
                    "data_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                },
                "model": "unused",
                "tokenizer": "unused",
                "generation": {"do_sample": False, "num_beams": 1, "use_cache": True, "decode_cap": None},
                "samples": [
                    {
                        "sample_id": record["sample_id"],
                        "input_sha256": record["input_sha256"],
                        "prompt_length": 4,
                        "generation_max_new_tokens_requested": 3,
                        "generated_ids": baseline_output,
                        "generated_ids_sha256": b1_runner.generated_ids_sha256(baseline_output),
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(b1_runner.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TinyTokenizer())
    monkeypatch.setattr(b1_runner, "load_model", lambda *args, **kwargs: (model, torch.device("cpu")))
    args = Namespace(
        input=str(input_path),
        manifest=str(manifest_path),
        tokenizer="unused",
        model="unused",
        run_dir=str(tmp_path / "run"),
        sample_id=[record["sample_id"]],
        device="cpu",
        dtype="float32",
        key_block_size=2,
        prefill_window=2,
        token_topk=5,
        save_token_topk=True,
        page_size=2,
        page_topk=3,
        decode_chunk_steps=128,
        decode_cap=None,
        seed=0,
        verify_baseline=str(baseline_path),
    )

    assert b1_runner.run_collection(args) == 0
    metadata = json.loads((tmp_path / "run" / "samples" / record["sample_id"] / "sample_metadata.json").read_text())
    assert metadata["baseline_verification"]["exact_generated_ids_match"] is True


def test_baseline_runner_persists_no_probe_generated_ids(tmp_path, monkeypatch):
    record = make_record()
    input_path = tmp_path / "prompts.jsonl"
    input_path.write_text(json.dumps(record) + "\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"data_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()}))
    model = make_model()
    monkeypatch.setattr(baseline_decode.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TinyTokenizer())
    monkeypatch.setattr(baseline_decode, "load_model", lambda *args, **kwargs: (model, torch.device("cpu")))
    output_path = tmp_path / "baseline.json"
    args = Namespace(
        input=str(input_path),
        manifest=str(manifest_path),
        tokenizer="unused",
        model="unused",
        output=str(output_path),
        sample_id=[record["sample_id"]],
        device="cpu",
        dtype="float32",
        decode_cap=None,
        seed=0,
    )

    assert baseline_decode.run_baseline(args) == 0
    output = json.loads(output_path.read_text())
    assert output["schema_version"] == baseline_decode.BASELINE_SCHEMA_VERSION
    assert output["samples"][0]["input_sha256"] == record["input_sha256"]
    assert output["samples"][0]["generated_ids_sha256"] == b1_runner.generated_ids_sha256(
        output["samples"][0]["generated_ids"]
    )
    assert output["resource_measurement"]["version"] == "headkv-resource-measurement-v1"
    timing = output["samples"][0]["forward_timing"]
    assert timing["prefill"]["count"] == 1
    assert timing["decode"]["count"] == 2


def test_frozen_selection_rejects_manifest_mismatch(tmp_path):
    record = make_record()
    input_path = tmp_path / "prompts.jsonl"
    input_path.write_text(json.dumps(record) + "\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"data_sha256": "0" * 64}))

    with pytest.raises(b1_runner.B1RunnerError, match="checksum"):
        b1_runner.load_frozen_selection(input_path, manifest_path, [record["sample_id"]])


def test_resource_budget_reports_only_post_sample_threshold_violations():
    records = [
        {
            "sample_id": "first",
            "status": "completed",
            "generation_end_to_end_seconds": 3.0,
            "peak_memory_reserved_bytes": 10,
            "trace_write_wall_seconds": 1.0,
            "trace_storage": {"logical_array_bytes": 20, "artifact_file_bytes": 30},
        },
        {
            "sample_id": "second",
            "status": "completed",
            "generation_end_to_end_seconds": 4.0,
            "peak_memory_reserved_bytes": 10,
            "trace_write_wall_seconds": 1.0,
            "trace_storage": {"logical_array_bytes": 20, "artifact_file_bytes": 30},
        },
    ]
    limits = {
        "max_batch_generation_seconds": 6.0,
        "max_batch_artifact_file_bytes": 100,
        "max_sample_peak_reserved_bytes": 11,
        "max_sample_logical_array_bytes": 21,
        "max_sample_artifact_file_bytes": 31,
        "max_sample_trace_write_seconds": 2.0,
    }

    assert b1_runner.resource_budget_violations(records[:1], limits) == []
    assert b1_runner.resource_budget_violations(records, limits) == [
        "batch_generation_seconds: observed 7.0 exceeds limit 6.0"
    ]
