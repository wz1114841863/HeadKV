import hashlib
import json
from argparse import Namespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.head_analysis.runners import b1_runner


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


def test_frozen_selection_rejects_manifest_mismatch(tmp_path):
    record = make_record()
    input_path = tmp_path / "prompts.jsonl"
    input_path.write_text(json.dumps(record) + "\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"data_sha256": "0" * 64}))

    with pytest.raises(b1_runner.B1RunnerError, match="checksum"):
        b1_runner.load_frozen_selection(input_path, manifest_path, [record["sample_id"]])
