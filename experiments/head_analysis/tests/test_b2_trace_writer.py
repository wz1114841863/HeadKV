import json

import numpy as np
import pytest

from experiments.head_analysis.collectors.b2_attention import (
    B2DecodeLayerRecord,
    B2PrefillLayerRecord,
)
from experiments.head_analysis.collectors.prefill_consistency import B2CollectorConfig, UINT32_SENTINEL
from experiments.head_analysis.storage.b2_trace_writer import (
    B2SampleWriteRequest,
    B2TraceSchema,
    B2TraceWriter,
    B2TraceWriterError,
    validate_b2_run,
)


def make_prefill(layer_id, *, kv_heads=2, prompt_length=384):
    config = B2CollectorConfig()
    conditions = len(config.conditions)
    kmax = config.critical_token_max
    counts = np.asarray(
        [prompt_length - condition.initial_exclusion - condition.recent_exclusion for condition in config.conditions],
        dtype=np.uint32,
    )
    starts = np.asarray([condition.initial_exclusion for condition in config.conditions], dtype=np.uint32)
    ends = np.asarray([prompt_length - condition.recent_exclusion for condition in config.conditions], dtype=np.uint32)
    histogram = np.zeros((kv_heads, conditions, 65), dtype=np.uint32)
    histogram[:, :, 0] = counts - 1
    histogram[:, :, 1] = 1
    positions = np.full((kv_heads, conditions, kmax), UINT32_SENTINEL, dtype=np.uint32)
    positions[:, :, 0] = starts
    frequencies = np.zeros((kv_heads, conditions, kmax), dtype=np.uint8)
    frequencies[:, :, 0] = 1
    return B2PrefillLayerRecord(
        sample_id="tiny",
        layer_id=layer_id,
        forward_id=0,
        condition_ids=tuple(condition.condition_id for condition in config.conditions),
        cv_prefill=np.ones((kv_heads, conditions), dtype=np.float32),
        frequency_mean=np.ones((kv_heads, conditions), dtype=np.float32),
        frequency_std=np.ones((kv_heads, conditions), dtype=np.float32),
        binary_threshold=np.full((kv_heads, conditions), 0.1, dtype=np.float32),
        candidate_token_count=counts,
        candidate_position_start=starts,
        candidate_position_end_exclusive=ends,
        nonzero_frequency_count=np.ones((kv_heads, conditions), dtype=np.uint32),
        frequency_histogram=histogram,
        critical_token_positions=positions,
        critical_token_frequencies=frequencies,
        critical_token_valid_count=np.ones((kv_heads, conditions), dtype=np.uint32),
        metric_valid=np.ones((kv_heads, conditions), dtype=bool),
    )


def make_decode(layer_id, step, *, kv_heads=2):
    mass = np.empty((kv_heads, 4, 3, 2), dtype=np.float32)
    mass[:, :, 0, 0] = 0.1
    mass[:, :, 1, 0] = 0.2
    mass[:, :, 2, 0] = 0.3
    mass[:, :, 0, 1] = 0.2
    mass[:, :, 1, 1] = 0.3
    mass[:, :, 2, 1] = 0.4
    return B2DecodeLayerRecord(
        sample_id="tiny",
        layer_id=layer_id,
        forward_id=step + 1,
        decode_step=step,
        query_position=384 + step,
        prompt_attention_mass=np.full(kv_heads, 0.8, dtype=np.float32),
        critical_attention_mass=mass,
    )


def metadata():
    return {
        "sample_id": "tiny",
        "dataset": "unit",
        "task": "continuation",
        "pair_id": None,
        "input_sha256": "1" * 64,
        "prompt_length": 384,
        "max_new_tokens": 3,
        "attention_backend": "sdpa",
        "b1_run_set_sha256": "2" * 64,
        "b1_generated_ids_sha256": "3" * 64,
    }


def write_completed_run(tmp_path):
    run_dir = tmp_path / "run"
    writer = B2TraceWriter(run_dir, B2TraceSchema(2, 4, 2), decode_chunk_steps=1)
    writer.initialize(
        {"run_id": "run", "status": "running", "target_sample_ids": ["tiny"]},
        config_snapshot={"selected_sample_ids": ["tiny"]},
    )
    result = writer.write_sample(
        B2SampleWriteRequest(
            sample_metadata=metadata(),
            replay_generated_ids=[10, 11, 12],
            prefill_predicted_token_id=10,
            prefill_records=[make_prefill(0), make_prefill(1)],
            decode_records=[
                make_decode(0, 0),
                make_decode(1, 0),
                make_decode(0, 1),
                make_decode(1, 1),
            ],
            decode_predicted_token_ids=[11, 12],
        )
    )
    writer.finalize(
        status="completed",
        resource_summary={
            "status": "completed",
            "samples_completed": 1,
            "samples_failed": 0,
            "per_sample": [],
        },
    )
    return run_dir, result


def test_writer_materializes_independently_validated_atomic_run(tmp_path):
    run_dir, write_result = write_completed_run(tmp_path)

    validation = validate_b2_run(run_dir)

    assert validation.status == "completed"
    assert validation.completed_sample_count == 1
    assert write_result.logical_array_bytes > 0
    assert (run_dir / "resource_metrics.json").is_file()
    schema = json.loads((run_dir / "schema.json").read_text())
    snapshot = json.loads((run_dir / "config_snapshot.yaml").read_text())
    assert schema["collector_config_sha256"] == snapshot["collector_config_sha256"]
    with np.load(run_dir / "samples/tiny/decode_recall_001.npz") as chunk:
        assert chunk["decode_steps"].tolist() == [1]
        assert chunk["critical_attention_mass"].shape == (1, 2, 2, 4, 3, 2)


def test_writer_rejects_top1_drift_and_missing_layer(tmp_path):
    writer = B2TraceWriter(tmp_path / "run", B2TraceSchema(2, 4, 2))
    writer.initialize(
        {"run_id": "run", "status": "running", "target_sample_ids": ["tiny"]},
        config_snapshot={},
    )
    request = B2SampleWriteRequest(
        sample_metadata=metadata(),
        replay_generated_ids=[10, 11],
        prefill_predicted_token_id=9,
        prefill_records=[make_prefill(0), make_prefill(1)],
        decode_records=[make_decode(0, 0), make_decode(1, 0)],
        decode_predicted_token_ids=[11],
    )
    with pytest.raises(B2TraceWriterError, match="Prefill Top-1"):
        writer.write_sample(request)

    request = B2SampleWriteRequest(
        sample_metadata=metadata(),
        replay_generated_ids=[10, 11],
        prefill_predicted_token_id=10,
        prefill_records=[make_prefill(0)],
        decode_records=[make_decode(0, 0), make_decode(1, 0)],
        decode_predicted_token_ids=[11],
    )
    with pytest.raises(B2TraceWriterError, match="expected 2 Layer"):
        writer.write_sample(request)


def test_writer_rejects_nonmonotonic_decode_mass(tmp_path):
    writer = B2TraceWriter(tmp_path / "run", B2TraceSchema(2, 4, 2))
    writer.initialize(
        {"run_id": "run", "status": "running", "target_sample_ids": ["tiny"]},
        config_snapshot={},
    )
    bad = make_decode(0, 0)
    bad.critical_attention_mass[:, :, 1, 0] = 0.01
    with pytest.raises(B2TraceWriterError, match="monotonic"):
        writer.write_sample(
            B2SampleWriteRequest(
                sample_metadata=metadata(),
                replay_generated_ids=[10, 11],
                prefill_predicted_token_id=10,
                prefill_records=[make_prefill(0), make_prefill(1)],
                decode_records=[bad, make_decode(1, 0)],
                decode_predicted_token_ids=[11],
            )
        )


def test_run_validator_detects_post_write_corruption(tmp_path):
    run_dir, _ = write_completed_run(tmp_path)
    path = run_dir / "samples/tiny/prefill_consistency.npz"
    path.write_bytes(path.read_bytes() + b"corrupt")

    with pytest.raises(B2TraceWriterError, match="hash/size"):
        validate_b2_run(run_dir)
