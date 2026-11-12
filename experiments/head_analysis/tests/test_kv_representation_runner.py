import json
from argparse import Namespace
from dataclasses import replace
from functools import partial

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from experiments.head_analysis.collectors.kv_representation import sample_positions
from experiments.head_analysis.runners import kv_representation_runner as runner
from experiments.head_analysis.runners.b2_runner import B1ReplaySample, B2RunnerError, generated_ids_sha256
from experiments.head_analysis.storage.kv_representation import read_record
from experiments.head_analysis.tests.fixtures import sha256, write_json, write_trace


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tiny_source():
    """An independently generated trajectory and actual Cache/projection references."""
    torch.manual_seed(37)
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
        eos_token_id=None, pad_token_id=None,
    )
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).eval().to(dtype=torch.bfloat16)
    prompt = [(i % 60) + 1 for i in range(640)]
    snapshots, keys_pre = {}, {}

    def save_projection(layer_id, module, args, output):
        if output.shape[1] == len(prompt):
            keys_pre[layer_id] = output.view(1, len(prompt), 2, 8).transpose(1, 2)[0].clone()

    def save_cache(module, args, output):
        cache = output.past_key_values
        step = cache.get_seq_length() - len(prompt) - 1
        if step in (-1, 5, 31, 255):
            snapshots[step] = [
                (layer.keys[0].clone(), layer.values[0].clone()) for layer in cache.layers
            ]

    handles = [model.register_forward_hook(save_cache)]
    handles += [
        layer.self_attn.k_proj.register_forward_hook(partial(save_projection, i))
        for i, layer in enumerate(model.model.layers)
    ]
    try:
        with torch.inference_mode():
            output = model.generate(
                torch.tensor([prompt]), max_new_tokens=260, do_sample=False, use_cache=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    ids = tuple(output[0, len(prompt):].tolist())
    sample = B1ReplaySample(
        frozen_record={
            "sample_id": "tiny", "dataset": "govreport", "task": "summarization",
            "input_ids": prompt, "actual_prompt_length": len(prompt), "max_new_tokens": 260,
            "input_sha256": "b" * 64, "metadata": {},
        },
        batch_id="b1_batch", run_id="b1_run", generated_ids=ids,
        generated_ids_sha256="c" * 64, generated_ids_file_sha256="d" * 64,
    )
    return model, sample, snapshots, keys_pre


def test_runner_replays_256_forwards_and_saves_exact_cache_samples(tmp_path, tiny_source):
    model, sample, snapshots, keys_pre = tiny_source
    directory = tmp_path / "long"
    metadata = runner.collect_sample(
        model, device="cpu", sample=sample, directory=directory, block_size=97,
    )
    assert metadata == read_json(directory / "sample_metadata.json")
    assert metadata["status"] == "completed" and metadata["top1_match"] is True
    assert metadata["decode_forward_count"] == 256
    assert metadata["checkpoints"] == [31, 255]
    assert metadata["source_b1"]["batch_id"] == sample.batch_id
    assert metadata["source_b1"]["run_id"] == sample.run_id
    assert metadata["source_b1"]["generated_length"] == 260
    timing = metadata["forward_timing"]
    assert timing["prefill"]["count"] == 1 and timing["decode"]["count"] == 256
    assert timing["incomplete_forward_count"] == 0
    replay = read_json(directory / "replay.json")
    assert replay["generated_ids"] == list(sample.generated_ids[:257])
    assert replay["prefill_predicted_token_id"] == sample.generated_ids[0]
    assert replay["decode_predicted_token_ids"] == list(sample.generated_ids[1:257])

    manifest = read_json(directory / "metadata.json")
    assert manifest["status"] == "completed"
    assert set(manifest["files"]) == {
        f"{prefix}/layer_{layer:03d}.npz"
        for prefix in ("prefill", "decode/step_031", "decode/step_255") for layer in range(2)
    }
    positions = sample_positions(640)
    for layer in range(2):
        saved = read_record(directory, f"prefill/layer_{layer:03d}.npz")
        np.testing.assert_array_equal(saved["sample_positions"], positions)
        for name, reference in (
            ("k_pre", keys_pre[layer]),
            ("k_post", snapshots[-1][layer][0]),
            ("v", snapshots[-1][layer][1]),
        ):
            assert saved[name].dtype == torch.bfloat16 and saved[name].shape == (2, 512, 8)
            assert torch.equal(saved[name].view(torch.uint16), reference[:, positions].contiguous().view(torch.uint16))
        np.testing.assert_array_equal(saved["k_post_count"], [640, 512])
        for step in (31, 255):
            relative = f"decode/step_{step:03d}/layer_{layer:03d}.npz"
            saved = read_record(directory, relative)
            np.testing.assert_array_equal(saved["token_positions"], [640 + step])
            for name, reference in zip(("k_post", "v"), snapshots[step][layer]):
                assert saved[name].shape == (2, 1, 8)
                assert torch.equal(saved[name], reference[:, 640 + step:641 + step])
                assert manifest["files"][relative]["checks"][name + "_prefix_exact"]
    original = (directory / "sample_metadata.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.collect_sample(model, device="cpu", sample=sample, directory=directory)
    assert (directory / "sample_metadata.json").read_bytes() == original


def test_short_ruler_replay_uses_six_forwards_and_one_checkpoint(tmp_path, tiny_source):
    model, sample, snapshots, _ = tiny_source
    sample = replace(sample, frozen_record={**sample.frozen_record, "dataset": "ruler"})
    directory = tmp_path / "short"
    metadata = runner.collect_sample(model, device="cpu", sample=sample, directory=directory)
    assert metadata["decode_forward_count"] == 6 and metadata["checkpoints"] == [5]
    assert metadata["forward_timing"]["decode"]["count"] == 6
    assert read_json(directory / "replay.json")["generated_ids"] == list(sample.generated_ids[:7])
    manifest = read_json(directory / "metadata.json")
    assert len(manifest["files"]) == 4
    for layer in range(2):
        saved = read_record(directory, f"decode/step_005/layer_{layer:03d}.npz")
        np.testing.assert_array_equal(saved["token_positions"], [645])
        assert torch.equal(saved["k_post"], snapshots[5][layer][0][:, 645:646])


def test_last_top1_mismatch_cannot_finalize_and_restores_hooks(tmp_path, tiny_source):
    model, sample, _, _ = tiny_source
    ids = list(sample.generated_ids)
    ids[6] = (ids[6] + 1) % model.config.vocab_size
    bad = replace(sample, generated_ids=tuple(ids),
                  frozen_record={**sample.frozen_record, "dataset": "ruler"})
    directory = tmp_path / "failed"
    backend = ALL_ATTENTION_FUNCTIONS["sdpa"]
    with pytest.raises(B2RunnerError, match="Decode step 5 Top-1"):
        runner.collect_sample(model, device="cpu", sample=bad, directory=directory)
    assert read_json(directory / "sample_metadata.json")["status"] == "failed"
    manifest = read_json(directory / "metadata.json")
    assert manifest["status"] != "completed"
    assert "decode/step_005/layer_001.npz" in manifest["files"]
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is backend
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
    good = replace(sample, frozen_record={**sample.frozen_record, "dataset": "ruler"})
    assert runner.collect_sample(model, device="cpu", sample=good,
                                 directory=tmp_path / "recovered")["status"] == "completed"


def test_replay_config_rejects_insufficient_source_without_shortening(tiny_source):
    _, sample, _, _ = tiny_source
    assert runner.replay_config(sample, block_size=37).decode_steps == (31, 255)
    assert runner.replay_config(sample, block_size=37).block_size == 37
    with pytest.raises(ValueError):
        runner.replay_config(replace(sample, generated_ids=sample.generated_ids[:256]))
    ruler = replace(sample, frozen_record={**sample.frozen_record, "dataset": "ruler"})
    assert runner.replay_config(ruler).decode_steps == (5,)
    with pytest.raises(ValueError):
        runner.replay_config(replace(ruler, generated_ids=sample.generated_ids[:6]))


def test_run_collection_assembles_selected_source_and_preserves_existing_output(tmp_path, monkeypatch, tiny_source):
    model, sample, _, _ = tiny_source
    # The existing B1 fixture schema has one Layer and one physical KV Head.
    config = LlamaConfig.from_dict(model.config.to_dict())
    config.num_hidden_layers, config.num_key_value_heads = 1, 1
    model = LlamaForCausalLM(config).eval().to(dtype=torch.bfloat16)
    record = {**sample.frozen_record, "dataset": "ruler", "pair_id": None,
              "target_prompt_length": 8192, "max_new_tokens": 7}
    with torch.inference_mode():
        output = model.generate(torch.tensor([record["input_ids"]]), max_new_tokens=7,
                                do_sample=False, use_cache=True)
    ids = output[0, len(record["input_ids"]):].tolist()
    (tmp_path / "prompts.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    write_json(tmp_path / "manifest.json", {"data_sha256": "test-fixed-data"})
    b1_directory = tmp_path / "b1_batch"
    write_trace(b1_directory, [record], data_sha256="test-fixed-data", commit="a" * 40, decode_length=6)
    generated_path = b1_directory / "samples/tiny/generated_ids.json"
    write_json(generated_path, {"dtype": "uint32-little-endian", "ids": ids, "sha256": generated_ids_sha256(ids)})
    b1_metadata = read_json(b1_directory / "run_metadata.json")
    b1_metadata.update(model="tiny-model", tokenizer="tiny-model")
    write_json(b1_directory / "run_metadata.json", b1_metadata)
    file_manifest = read_json(b1_directory / "file_manifest.json")
    for entry in file_manifest["files"]:
        path = b1_directory / entry["relative_path"]
        entry.update(sha256=sha256(path), file_bytes=path.stat().st_size)
    write_json(b1_directory / "file_manifest.json", file_manifest)
    run_set = tmp_path / "b1.json"
    write_json(run_set, {
        "schema_version": "headkv-v1-b1-run-set-v1", "status": "completed",
        "fixed_dataset": {"prompts_path": "prompts.jsonl", "manifest_path": "manifest.json",
                          "data_sha256": "test-fixed-data"},
        "execution_contract": {"attention_backend": "sdpa", "model": {
            "path": "tiny-model", "tokenizer_path": "tiny-model", "dtype": "bfloat16",
        }},
        "batches": [{"batch_id": "b1_batch", "run_dir": "b1_batch", "sample_ids": ["tiny"],
                     "status": "completed"}],
    })
    monkeypatch.setattr(runner, "load_model", lambda *args: (model, torch.device("cpu")))
    args = Namespace(
        run_set=str(run_set), source_root=str(tmp_path), sample_id=["tiny"],
        model="tiny-model", tokenizer="tiny-model", device="cpu", dtype="bfloat16",
        block_size=97, run_dir=str(tmp_path / "run"),
    )
    selection_path = tmp_path / "selected.json"
    write_json(selection_path, ["tiny"])
    args.sample_id = None
    args.sample_ids_file = selection_path
    assert runner.run_collection(args) == 0
    metadata_path = tmp_path / "run/run_metadata.json"
    metadata = read_json(metadata_path)
    assert metadata["status"] == "completed" and metadata["completed_sample_ids"] == ["tiny"]
    assert read_json(tmp_path / "run/samples/tiny/metadata.json")["status"] == "completed"
    original = metadata_path.read_bytes()
    with pytest.raises(FileExistsError):
        runner.run_collection(args)
    assert metadata_path.read_bytes() == original

    # A failure after writing the final checkpoint must fail the whole run too.
    ids[-1] = (ids[-1] + 1) % model.config.vocab_size
    write_json(generated_path, {"dtype": "uint32-little-endian", "ids": ids, "sha256": generated_ids_sha256(ids)})
    for entry in file_manifest["files"]:
        if entry["relative_path"] == "samples/tiny/generated_ids.json":
            entry.update(sha256=sha256(generated_path), file_bytes=generated_path.stat().st_size)
    write_json(b1_directory / "file_manifest.json", file_manifest)
    args.run_dir = str(tmp_path / "failed-run")
    args.sample_id, args.sample_ids_file = ["tiny"], None
    assert runner.run_collection(args) == 1
    failed = read_json(tmp_path / "failed-run/run_metadata.json")
    assert failed["status"] == "failed" and failed["failed_sample_id"] == "tiny"
    assert failed["completed_sample_ids"] == []
    assert read_json(tmp_path / "failed-run/samples/tiny/sample_metadata.json")["status"] == "failed"
