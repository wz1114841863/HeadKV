"""Small trace and input builders shared by research tests."""

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.head_analysis.collectors.b2_attention import B2DecodeLayerRecord, B2PrefillLayerRecord
from experiments.head_analysis.collectors.prefill_consistency import B2CollectorConfig, UINT32_SENTINEL
from experiments.head_analysis.storage.trace_writer import AttentionTraceSchema

def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar_features(visible: np.ndarray, normalized_entropy: np.ndarray) -> np.ndarray:
    values = np.zeros((len(visible), 1, 4, 28), dtype=np.float32)
    normalized = np.asarray(normalized_entropy, dtype=np.float32)
    values[..., 0] = (normalized * np.log(visible.astype(np.float32)))[:, None, None]
    values[..., 1] = normalized[:, None, None]
    values[..., 2:6] = 1.0
    values[..., 6] = 1.0
    values[..., 7] = 1.0
    values[..., 8] = 1.0
    values[..., 9] = 0.0
    values[..., 10] = 0.0
    values[..., 11] = 1.0
    return values


def write_trace(
    run_dir: Path,
    records: list[dict],
    *,
    data_sha256: str,
    commit: str,
    decode_length: int = 2,
) -> None:
    schema = AttentionTraceSchema(
        num_layers=1,
        num_query_heads=4,
        num_kv_heads=1,
        prefill_window=64,
        token_topk=64,
        page_topk=32,
    ).to_dict()
    write_json(run_dir / "schema.json", schema)
    write_json(run_dir / "config_snapshot.yaml", {"collection": {"page_topk": 32}})
    write_json(
        run_dir / "run_metadata.json",
        {
            "status": "completed",
            "schema_version": schema["schema_version"],
            "git": {"commit": commit},
            "fixed_dataset": {"data_sha256": data_sha256},
            "dtype": "bfloat16",
            "attention_backend": "sdpa",
            "generation": {"do_sample": False, "num_beams": 1, "use_cache": True},
        },
    )
    for record in records:
        sample_id = record["sample_id"]
        sample_dir = run_dir / "samples" / sample_id
        sample_dir.mkdir(parents=True)
        base = 0.2 if record["target_prompt_length"] == 8192 else 0.6
        generated = np.arange(101, 102 + decode_length, dtype="<u4")
        write_json(
            sample_dir / "generated_ids.json",
            {"ids": generated.tolist(), "sha256": hashlib.sha256(generated.tobytes()).hexdigest()},
        )
        write_json(
            sample_dir / "sample_metadata.json",
            {
                "sample_id": sample_id,
                "dataset": record["dataset"],
                "task": record["task"],
                "pair_id": record["pair_id"],
                "input_sha256": record["input_sha256"],
                "prompt_length": record["actual_prompt_length"],
                "status": "completed",
                "actual_generated_length": decode_length + 1,
                "decode_trace_length": decode_length,
            },
        )
        prefill_visible = np.asarray([record["actual_prompt_length"]], dtype=np.uint32)
        np.savez_compressed(
            sample_dir / "prefill_stats.npz",
            query_positions=np.asarray([record["actual_prompt_length"] - 1], dtype=np.uint32),
            visible_token_count=prefill_visible,
            scalar_features=scalar_features(prefill_visible, np.asarray([base])),
        )
        decode_steps = np.arange(decode_length, dtype=np.uint32)
        decode_visible = record["actual_prompt_length"] + 1 + decode_steps
        decode_normalized = (
            np.asarray([base, base + 0.2]) if decode_length == 2 else base + 0.01 * np.arange(decode_length)
        )
        coverage_value = 0.25 if record["target_prompt_length"] == 8192 else 0.75
        page_scope_mass = np.ones((decode_length, 1, 1, 2), dtype=np.float32)
        page_scope_mass[..., 1] = 0.6
        np.savez_compressed(
            sample_dir / "decode_stats_000.npz",
            decode_steps=decode_steps,
            query_positions=decode_visible - 1,
            visible_token_count=decode_visible,
            scalar_features=scalar_features(decode_visible, decode_normalized),
            token_topk_positions=np.tile(np.arange(64, dtype=np.uint32), (decode_length, 1, 4, 1)),
            page_topk_ids_all_visible=np.zeros((decode_length, 1, 1, 32), dtype=np.uint32),
            page_topk_ids_prompt_only=np.zeros((decode_length, 1, 1, 32), dtype=np.uint32),
            page_candidate_count_all_visible=np.full(decode_length, 520, dtype=np.uint32),
            page_candidate_count_prompt_only=np.full(decode_length, 512, dtype=np.uint32),
            page_scope_mass=page_scope_mass,
            page_top32_coverage=np.full((decode_length, 1, 1, 2), coverage_value, dtype=np.float32),
            page_scope_valid=np.ones((decode_length, 1, 1, 2), dtype=bool),
        )

    entries = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(run_dir).as_posix()
        entry = {
            "relative_path": relative,
            "sha256": sha256(path),
            "file_bytes": path.stat().st_size,
            "logical_array_bytes": 0,
        }
        if relative.endswith("decode_stats_000.npz"):
            entry.update({"step_start": 0, "step_end": decode_length - 1})
        entries.append(entry)
    write_json(run_dir / "file_manifest.json", {"files": entries})


def make_fixture(tmp_path: Path, *, decode_length: int = 2) -> tuple[Path, Path, Path]:
    data_sha256 = "frozen-data"
    commit = "a" * 40
    records = [
        {
            "sample_id": "pg19_001_8k",
            "dataset": "pg19",
            "task": "continuation",
            "pair_id": "pg19_001",
            "target_prompt_length": 8192,
            "actual_prompt_length": 8190,
            "input_sha256": "input-8k",
        },
        {
            "sample_id": "pg19_001_32k",
            "dataset": "pg19",
            "task": "continuation",
            "pair_id": "pg19_001",
            "target_prompt_length": 32768,
            "actual_prompt_length": 32766,
            "input_sha256": "input-32k",
        },
    ]
    prompts = tmp_path / "data" / "fixed" / "prompts.jsonl"
    prompts.parent.mkdir(parents=True)
    prompts.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    fixed_manifest = prompts.parent / "manifest.json"
    write_json(fixed_manifest, {"data_sha256": data_sha256})
    run_dir = tmp_path / "data" / "runs" / "batch-1"
    write_trace(run_dir, records, data_sha256=data_sha256, commit=commit, decode_length=decode_length)
    run_set = tmp_path / "data" / "runs" / "set.manifest.json"
    write_json(
        run_set,
        {
            "schema_version": "headkv-v1-b1-run-set-v1",
            "status": "completed",
            "fixed_dataset": {
                "prompts_path": str(prompts.relative_to(tmp_path)),
                "manifest_path": str(fixed_manifest.relative_to(tmp_path)),
                "data_sha256": data_sha256,
            },
            "execution_contract": {
                "git_commit": commit,
                "dtype": "bfloat16",
                "attention_backend": "sdpa",
            },
            "batches": [
                {
                    "batch_id": "pg19_b01",
                    "sample_ids": [record["sample_id"] for record in records],
                    "run_dir": str(run_dir.relative_to(tmp_path)),
                    "status": "completed",
                    "run_metadata_sha256": sha256(run_dir / "run_metadata.json"),
                    "file_manifest_sha256": sha256(run_dir / "file_manifest.json"),
                    "validation_reports": [],
                }
            ],
        },
    )
    report = tmp_path / "data" / "runs" / "batch-1.page-rco.json"
    rco_records = []
    for record in records:
        rco_value = 0.2 if record["target_prompt_length"] == 8192 else 0.6
        for scope in ("all_visible", "prompt_only"):
            for k in (8, 16, 32):
                for delta in (1, 2, 4, 8, 16, 32, 64, 128, 256):
                    valid_pair_count = max(0, decode_length - delta)
                    rco_records.append(
                        {
                            "sample_id": record["sample_id"],
                            "scope": scope,
                            "k": k,
                            "delta": delta,
                            "layer_id": 0,
                            "kv_head": 0,
                            "valid_pair_count": valid_pair_count,
                            "overlap_raw_mean": None if not valid_pair_count else 0.5,
                            "random_baseline_mean": None if not valid_pair_count else 0.1,
                            "rco_raw_mean": None if not valid_pair_count else rco_value,
                            "rco_clipped_mean": None if not valid_pair_count else rco_value,
                        }
                    )
    write_json(
        report,
        {
            "schema_version": "headkv-v1-b1-page-rco-v1",
            "source_run": str(run_dir),
            "source_run_metadata_sha256": sha256(run_dir / "run_metadata.json"),
            "k_values": [8, 16, 32],
            "deltas": [1, 2, 4, 8, 16, 32, 64, 128, 256],
            "records": rco_records,
        },
    )
    return run_set, report, run_dir


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
