import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.head_analysis.collectors.b2_attention import B2AttentionCollector
from experiments.head_analysis.runners import b2_runner
from experiments.head_analysis.runners.b2_runner import (
    B1ReplaySample,
    B2RunnerError,
    teacher_forced_replay,
)
from experiments.head_analysis.storage.b2_trace_writer import validate_b2_run


class TinyTokenizer:
    pass


def make_model():
    torch.manual_seed(17)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            eos_token_id=None,
            pad_token_id=None,
        )
    ).eval()
    model.config._attn_implementation = "sdpa"
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = None
    return model


def make_prompt():
    return [(index % 60) + 1 for index in range(384)]


def generate_ids(model, prompt, length=3):
    with torch.inference_mode():
        output = model.generate(
            torch.tensor([prompt]),
            max_new_tokens=length,
            do_sample=False,
            use_cache=True,
        )
    return output[0, len(prompt) :].tolist()


def test_tiny_llama_probe_replay_preserves_logits_and_token_offsets():
    model = make_model()
    prompt = make_prompt()
    replay = generate_ids(model, prompt)
    baseline = teacher_forced_replay(
        model,
        device="cpu",
        sample_id="tiny",
        prompt_ids=prompt,
        replay_generated_ids=replay,
        capture_logits=True,
    )
    collector = B2AttentionCollector()
    observed = teacher_forced_replay(
        model,
        device="cpu",
        sample_id="tiny",
        prompt_ids=prompt,
        replay_generated_ids=replay,
        collector=collector,
        capture_logits=True,
    )

    assert observed.all_predicted_token_ids == tuple(replay)
    assert len(observed.prefill_records) == 1
    assert len(observed.decode_records) == len(replay) - 1
    assert [record.decode_step for record in observed.decode_records] == list(range(len(replay) - 1))
    assert [record.query_position for record in observed.decode_records] == [384, 385]
    assert baseline.logits is not None and observed.logits is not None
    for expected, actual in zip(baseline.logits, observed.logits, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert collector._prefill_records == {}
    assert collector._rankings == {}


def test_replay_with_one_generated_token_has_no_decode_trace():
    model = make_model()
    prompt = make_prompt()
    replay = generate_ids(model, prompt, length=1)

    observed = teacher_forced_replay(
        model,
        device="cpu",
        sample_id="one",
        prompt_ids=prompt,
        replay_generated_ids=replay,
        collector=B2AttentionCollector(),
    )

    assert observed.all_predicted_token_ids == tuple(replay)
    assert observed.decode_predicted_token_ids == ()
    assert observed.decode_records == ()


def test_top1_failure_restores_probe_and_allows_next_sample():
    model = make_model()
    prompt = make_prompt()
    replay = generate_ids(model, prompt, length=2)
    bad = [int((replay[0] + 1) % 64), replay[1]]

    with pytest.raises(B2RunnerError, match="Prefill Top-1"):
        teacher_forced_replay(
            model,
            device="cpu",
            sample_id="bad",
            prompt_ids=prompt,
            replay_generated_ids=bad,
            collector=B2AttentionCollector(),
        )

    recovered = teacher_forced_replay(
        model,
        device="cpu",
        sample_id="good",
        prompt_ids=prompt,
        replay_generated_ids=replay,
        collector=B2AttentionCollector(),
    )
    assert recovered.all_predicted_token_ids == tuple(replay)


def test_runner_composes_replay_collector_writer_and_validator(tmp_path, monkeypatch):
    model = make_model()
    prompt = make_prompt()
    replay = generate_ids(model, prompt, length=2)
    fixed_manifest = tmp_path / "manifest.json"
    fixed_manifest.write_text(json.dumps({"data_sha256": "a" * 64}))
    frozen_record = {
        "sample_id": "tiny",
        "dataset": "unit",
        "task": "continuation",
        "pair_id": None,
        "input_ids": prompt,
        "input_sha256": "b" * 64,
        "actual_prompt_length": len(prompt),
        "max_new_tokens": 2,
        "metadata": {},
    }
    sample = B1ReplaySample(
        frozen_record=frozen_record,
        batch_id="batch",
        run_id="b1",
        generated_ids=tuple(replay),
        generated_ids_sha256=b2_runner.generated_ids_sha256(replay),
        generated_ids_file_sha256="c" * 64,
    )
    index = SimpleNamespace(
        payload={
            "fixed_dataset": {
                "data_sha256": "a" * 64,
                "manifest_path": str(fixed_manifest),
            }
        },
        source_root=tmp_path,
        run_set_sha256="d" * 64,
    )
    monkeypatch.setattr(b2_runner, "load_b1_replay_index", lambda *args, **kwargs: index)
    monkeypatch.setattr(b2_runner, "_validate_execution_contract", lambda *args, **kwargs: None)
    monkeypatch.setattr(b2_runner, "load_replay_samples", lambda *args, **kwargs: [sample])
    monkeypatch.setattr(b2_runner.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TinyTokenizer())
    monkeypatch.setattr(b2_runner, "load_model", lambda *args, **kwargs: (model, torch.device("cpu")))
    run_dir = tmp_path / "b2"
    args = Namespace(
        run_set="unused",
        source_root=str(tmp_path),
        expected_run_set_sha256=None,
        expected_data_sha256=None,
        sample_id=["tiny"],
        tokenizer="unused",
        model="unused",
        device="cpu",
        dtype="float32",
        key_block_size=64,
        decode_chunk_steps=128,
        run_dir=str(run_dir),
        seed=0,
        verify_no_probe=True,
        stop_on_failure=True,
    )

    assert b2_runner.run_collection(args) == 0
    validation = validate_b2_run(run_dir)
    assert validation.completed_sample_count == 1
    resources = json.loads((run_dir / "resource_metrics.json").read_text())
    assert resources["per_sample"][0]["no_probe_replay"] is not None
    assert resources["per_sample"][0]["forward_timing"]["prefill"]["count"] == 1


def test_execution_contract_accepts_frozen_nested_b1_model_schema():
    model_path = "/cache/model"
    index = SimpleNamespace(
        payload={
            "execution_contract": {
                "attention_backend": "sdpa",
                "model": {
                    "path": model_path,
                    "tokenizer_path": model_path,
                    "dtype": "bfloat16",
                },
            }
        },
        batches=(SimpleNamespace(batch_id="b", run_metadata={"tokenizer": model_path}),),
    )
    args = Namespace(model=model_path, tokenizer=model_path, dtype="bfloat16")

    b2_runner._validate_execution_contract(index, args)
