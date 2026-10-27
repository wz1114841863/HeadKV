import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.head_analysis.collectors.attention_stats import AttentionCollectorConfig, AttentionStatsCollector
from experiments.head_analysis.runtime.attention_probe import AttentionProbeContext
from experiments.head_analysis.storage.trace_writer import (
    AttentionTraceSchema,
    AttentionTraceWriter,
    SampleWriteRequest,
    TraceWriterError,
)
from kvpress.context import KVPhase


def make_context(query, key, *, prompt_length, decode_step=None):
    query_positions = torch.arange(key.shape[2] - query.shape[2], key.shape[2])
    return AttentionProbeContext(
        sample_id="sample-1",
        phase=KVPhase.PREFILL if decode_step is None else KVPhase.DECODE,
        layer_id=0,
        forward_id=0 if decode_step is None else decode_step + 1,
        decode_step=decode_step,
        query=query,
        key=key,
        attention_mask=None,
        scaling=1 / query.shape[-1] ** 0.5,
        is_causal=True,
        backend_name="sdpa",
        cache_position=query_positions,
        query_positions=query_positions,
        key_positions=torch.arange(key.shape[2]),
        prompt_length=prompt_length,
        query_dtype=query.dtype,
        key_dtype=key.dtype,
        dropout_p=0.0,
    )


def make_request():
    torch.manual_seed(17)
    collector = AttentionStatsCollector(
        AttentionCollectorConfig(prefill_window=2, key_block_size=2, token_topk=5, page_size=2, page_topk=3)
    )
    prefill = collector.collect(make_context(torch.randn(1, 4, 4, 2), torch.randn(1, 2, 4, 2), prompt_length=4))
    decode0 = collector.collect(
        make_context(torch.randn(1, 4, 1, 2), torch.randn(1, 2, 5, 2), prompt_length=4, decode_step=0)
    )
    decode1 = collector.collect(
        make_context(torch.randn(1, 4, 1, 2), torch.randn(1, 2, 6, 2), prompt_length=4, decode_step=1)
    )
    return SampleWriteRequest(
        sample_metadata={
            "sample_id": "sample-1",
            "dataset": "unit",
            "task": "continuation",
            "pair_id": None,
            "input_sha256": "0" * 64,
            "prompt_length": 4,
            "max_new_tokens": 3,
            "attention_backend": "sdpa",
        },
        generated_ids=[11, 12, 13],
        prefill_records=[replace(prefill, layer_id=0), replace(prefill, layer_id=1)],
        decode_records=[
            replace(decode0, layer_id=0),
            replace(decode0, layer_id=1),
            replace(decode1, layer_id=0),
            replace(decode1, layer_id=1),
        ],
        eos_reached=False,
        generation_stop_reason="max_new_tokens",
    )


def make_writer(tmp_path):
    writer = AttentionTraceWriter(
        tmp_path / "run-1",
        AttentionTraceSchema(
            num_layers=2,
            num_query_heads=4,
            num_kv_heads=2,
            prefill_window=2,
            page_size=2,
            token_topk=5,
            page_topk=3,
        ),
        decode_chunk_steps=1,
    )
    writer.initialize({"run_id": "run-1", "attention_backend": "sdpa"}, config_snapshot={"seed": 17})
    return writer


def test_writer_creates_chunked_npz_json_and_hashed_manifest(tmp_path):
    writer = make_writer(tmp_path)
    writer.write_sample(make_request())
    writer.finalize(status="completed", resource_summary={"cpu_test": True})

    run_dir = tmp_path / "run-1"
    sample_dir = run_dir / "samples" / "sample-1"
    assert (sample_dir / "prefill_stats.npz").is_file()
    assert (sample_dir / "decode_stats_000.npz").is_file()
    assert (sample_dir / "decode_stats_001.npz").is_file()
    assert not list(run_dir.rglob("*.tmp"))

    with np.load(sample_dir / "prefill_stats.npz") as prefill:
        assert prefill["scalar_features"].shape == (2, 2, 4, 28)
        assert prefill["scalar_features"].dtype == np.float32
    with np.load(sample_dir / "decode_stats_000.npz") as decode:
        assert decode["decode_steps"].tolist() == [0]
        assert decode["scalar_features"].shape == (1, 2, 4, 28)
        assert decode["token_topk_positions"].shape == (1, 2, 4, 5)
        assert decode["page_topk_ids_all_visible"].shape == (1, 2, 2, 3)
        assert decode["page_scope_valid"].dtype == np.bool_

    generated = json.loads((sample_dir / "generated_ids.json").read_text())
    assert generated["ids"] == [11, 12, 13]
    assert generated["sha256"] == hashlib.sha256(np.asarray([11, 12, 13], dtype="<u4").tobytes()).hexdigest()
    sample_metadata = json.loads((sample_dir / "sample_metadata.json").read_text())
    assert sample_metadata["decode_trace_length"] == 2
    assert sample_metadata["status"] == "completed"

    manifest = json.loads((run_dir / "file_manifest.json").read_text())
    entries = {entry["relative_path"]: entry for entry in manifest["files"]}
    assert "file_manifest.json" not in entries
    assert entries["samples/sample-1/decode_stats_001.npz"]["step_start"] == 1
    assert entries["samples/sample-1/decode_stats_001.npz"]["step_end"] == 1
    for relative_path, entry in entries.items():
        assert entry["sha256"] == hashlib.sha256((run_dir / relative_path).read_bytes()).hexdigest()
    assert json.loads((run_dir / "run_metadata.json").read_text())["status"] == "completed"


def test_writer_rejects_missing_layer_before_creating_sample_directory(tmp_path):
    writer = make_writer(tmp_path)
    request = make_request()
    incomplete = replace(request, decode_records=request.decode_records[:-1])

    with pytest.raises(TraceWriterError, match="expected 2 decode Layer records"):
        writer.write_sample(incomplete)

    assert not (tmp_path / "run-1" / "samples" / "sample-1").exists()


def test_writer_rejects_completed_trace_length_mismatch(tmp_path):
    writer = make_writer(tmp_path)
    request = make_request()
    malformed = replace(request, generated_ids=[11, 12])

    with pytest.raises(TraceWriterError, match="cover exactly steps"):
        writer.write_sample(malformed)
