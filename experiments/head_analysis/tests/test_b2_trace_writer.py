import json

import numpy as np
import pytest

from experiments.head_analysis.tests.fixtures import make_decode, make_prefill
from experiments.head_analysis.storage.b2_trace_writer import (
    B2SampleWriteRequest,
    B2TraceSchema,
    B2TraceWriter,
    B2TraceWriterError,
    validate_b2_run,
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
