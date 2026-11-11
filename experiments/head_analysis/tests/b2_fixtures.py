"""Shared small B2 run sets and independent correlation reference for tests."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.head_analysis.analysis import b2_consistency as b2
from experiments.head_analysis.analysis.b2_relationships import B2RelationshipInputs
from experiments.head_analysis.scripts import analyze_b2_run_set as cli
from experiments.head_analysis.storage.b2_trace_writer import B2SampleWriteRequest, B2TraceSchema, B2TraceWriter
from experiments.head_analysis.tests.fixtures import make_decode, make_prefill

def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(root, path):
    return {"path": str(path.relative_to(root)), "sha256": sha(path), "file_bytes": path.stat().st_size}


def ids_payload(ids):
    return {
        "ids": ids,
        "dtype": "uint32-little-endian",
        "sha256": hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest(),
    }


def fixture(root):
    """Two real Writer outputs, an excluded predecessor and two hash-bound B1 sources."""
    records = []
    for i, target in enumerate((8192, 32768)):
        ids = list(range(384))
        records.append(
            {
                "sample_id": f"pg19_pair_{i}",
                "dataset": "pg19",
                "task": "continuation",
                "pair_id": "pair",
                "target_prompt_length": target,
                "actual_prompt_length": len(ids),
                "input_ids": ids,
                "input_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                "max_new_tokens": 3,
            }
        )
    prompts = root / "data/fixed/prompts.jsonl"
    prompts.parent.mkdir(parents=True)
    prompts.write_text("".join(json.dumps(r) + "\n" for r in records))
    fm = prompts.with_name("manifest.json")
    write(fm, {"data_sha256": sha(prompts)})
    fixed = {
        "prompts_path": str(prompts.relative_to(root)),
        "manifest_path": str(fm.relative_to(root)),
        "data_sha256": sha(prompts),
    }
    generated = ids_payload([10, 11, 12])
    b1batches = []
    for i, record in enumerate(records):
        run = root / f"data/b1/b{i}"
        sample = run / "samples" / record["sample_id"]
        write(sample / "generated_ids.json", generated)
        write(
            sample / "sample_metadata.json",
            {"status": "completed", "input_sha256": record["input_sha256"], "actual_generated_length": 3},
        )
        write(run / "run_metadata.json", {"status": "completed"})
        files = [ref(root, p) for p in sorted(sample.iterdir())]
        entries = [
            {"relative_path": str(p.relative_to(run)), "sha256": sha(p), "file_bytes": p.stat().st_size}
            for p in sorted(sample.iterdir())
        ]
        assert len(files) == 2
        write(run / "file_manifest.json", {"files": entries})
        b1batches.append(
            {
                "batch_id": f"b{i}",
                "status": "completed",
                "sample_ids": [record["sample_id"]],
                "run_dir": str(run.relative_to(root)),
                "run_metadata_sha256": sha(run / "run_metadata.json"),
                "file_manifest_sha256": sha(run / "file_manifest.json"),
            }
        )
    b1path = root / "data/b1.manifest.json"
    write(
        b1path,
        {
            "schema_version": "headkv-v1-b1-run-set-v1",
            "status": "completed",
            "fixed_dataset": fixed,
            "execution_contract": {"model": {"path": "model", "tokenizer_path": "tokenizer", "dtype": "bfloat16"}},
            "batches": b1batches,
        },
    )
    analysis = root / "data/b1analysis/analysis_manifest.json"
    output = analysis.with_name("profiles.json")
    write(output, {"sample_ids": [r["sample_id"] for r in records]})
    write(
        analysis,
        {
            "schema_version": "headkv-v1-b1-offline-completed-v1",
            "status": "completed",
            "sample_count": 2,
            "source_run_set_sha256": sha(b1path),
            "outputs": [{"relative_path": output.name, "file_bytes": output.stat().st_size, "sha256": sha(output)}],
        },
    )
    schema = B2TraceSchema(1, 4, 2)
    contract = {
        "git_commit": "a" * 40,
        "dirty": False,
        "model": "model",
        "tokenizer": "tokenizer",
        "dtype": "bfloat16",
        "attention_backend": "sdpa",
        "teacher_forced_replay": True,
        "verify_no_probe": False,
        "environment": {"torch": "test"},
        "device": "cpu",
    }
    for key in ("dimensions", "axes", "conditions", "collector_config_sha256"):
        contract[key] = schema.to_dict()[key]
    batch_rows, sample_rows = [], []
    for i, record in enumerate(records):
        run = root / f"data/b2/b{i}"
        sid = record["sample_id"]
        rm = {
            k: contract[k]
            for k in (
                "model",
                "tokenizer",
                "dtype",
                "attention_backend",
                "environment",
                "device",
                "teacher_forced_replay",
                "verify_no_probe",
            )
        }
        rm.update(
            run_id=f"run{i}",
            target_sample_ids=[sid],
            git={"commit": "a" * 40, "dirty": False},
            b1_replay_authority=ref(root, b1path),
            fixed_dataset={"data_sha256": fixed["data_sha256"], "manifest_sha256": sha(fm)},
        )
        metadata = {
            "sample_id": sid,
            "dataset": record["dataset"],
            "task": record["task"],
            "pair_id": record["pair_id"],
            "input_sha256": record["input_sha256"],
            "prompt_length": 384,
            "max_new_tokens": 3,
            "attention_backend": "sdpa",
            "b1_run_set_sha256": sha(b1path),
            "b1_batch_id": f"b{i}",
            "b1_generated_ids_sha256": generated["sha256"],
            "b1_generated_ids_file_sha256": sha(
                root / b1batches[i]["run_dir"] / "samples" / sid / "generated_ids.json"
            ),
        }
        prefill = make_prefill(0)
        hist = prefill.frequency_histogram.astype(np.float64)
        axis = np.arange(65)
        mean = (hist * axis).sum(-1) / prefill.candidate_token_count
        std = np.sqrt((hist * axis**2).sum(-1) / prefill.candidate_token_count - mean**2)
        prefill = replace(
            prefill,
            sample_id=sid,
            frequency_mean=mean.astype(np.float32),
            frequency_std=std.astype(np.float32),
            cv_prefill=(std / mean).astype(np.float32),
        )
        writer = B2TraceWriter(run, schema, decode_chunk_steps=1)
        writer.initialize(rm, config_snapshot={})
        writer.write_sample(
            B2SampleWriteRequest(
                metadata,
                [10, 11, 12],
                10,
                [prefill],
                [replace(make_decode(0, s), sample_id=sid) for s in range(2)],
                [11, 12],
            )
        )
        writer.finalize(
            status="completed",
            resource_summary={
                "status": "completed",
                "samples_completed": 1,
                "samples_failed": 0,
                "collector_config_sha256": contract["collector_config_sha256"],
                "per_sample": [
                    {
                        "sample_id": sid,
                        "status": "completed",
                        "forward_timing": {
                            "incomplete_forward_count": 0,
                            "prefill": {"count": 1},
                            "decode": {"count": 2},
                        },
                    }
                ],
            },
        )
        result = b2.validate_b2_run(run).to_dict()
        row = {
            "batch_id": f"b{i}",
            "b1_batch_id": f"b{i}",
            "run_dir": str(run.relative_to(root)),
            "run_id": f"run{i}",
            "sample_ids": [sid],
            "recovery_of": "data/old" if i == 0 else None,
            **result,
        }
        for name, key in (
            ("run_metadata.json", "run_metadata_sha256"),
            ("file_manifest.json", "file_manifest_sha256"),
            ("schema.json", "schema_sha256"),
            ("config_snapshot.yaml", "config_snapshot_sha256"),
            ("resource_metrics.json", "resource_metrics_sha256"),
        ):
            row[key] = sha(run / name)
        batch_rows.append(row)
        sample_rows.append(read(run / "samples" / sid / "sample_metadata.json"))
    old = root / "data/old"
    write(old / "run_metadata.json", {"status": "running", "target_sample_ids": [records[0]["sample_id"]]})
    write(old / "file_manifest.json", {"files": []})
    excluded = [
        {
            "run_dir": "data/old",
            "superseded_by": batch_rows[0]["run_dir"],
            "include_in_statistics": False,
            "recorded_status": "running",
            "run_metadata_sha256": sha(old / "run_metadata.json"),
            "file_manifest_sha256": sha(old / "file_manifest.json"),
        }
    ]
    summary = {
        "generated_tokens": 6,
        "decode_steps": 4,
        "decode_chunks": 4,
        "artifact_files": sum(b["artifact_file_count"] for b in batch_rows),
        "artifact_bytes": sum(b["artifact_file_bytes"] for b in batch_rows),
        "condition_counts": {"pg19:8192": 1, "pg19:32768": 1},
    }
    payload = {
        "schema_version": b2.B2_RUN_SET_SCHEMA,
        "run_set_id": "unit",
        "status": "completed",
        "path_base": "project_root",
        "batch_count": 2,
        "sample_count": 2,
        "fixed_dataset": {**fixed, "manifest_sha256": sha(fm)},
        "execution_contract": contract,
        "batches": batch_rows,
        "excluded_runs": excluded,
        "summary": summary,
        "b1_replay_authority": ref(root, b1path),
        "b1_analysis_authority": ref(root, analysis),
    }
    report = root / "data/validation/report.json"
    write(
        report,
        {
            "schema_version": "headkv-v1-b2-collection-validation-v1",
            "status": "passed",
            "run_set_id": "unit",
            "batches": batch_rows,
            "excluded_runs": excluded,
            "summary": summary,
            "samples": sample_rows,
            "b1_replay_authority": payload["b1_replay_authority"],
            "b1_analysis_authority": payload["b1_analysis_authority"],
            "b1_analysis_outputs_verified": 1,
        },
    )
    payload["validation_report"] = ref(root, report)
    payload["validation_files"] = [ref(root, report)]
    manifest = root / "data/b2.manifest.json"
    write(manifest, payload)
    return manifest


def load(path):
    return b2.load_completed_b2_run_set(
        path,
        source_root=path.parent.parent,
        expected_run_set_sha256=sha(path),
        expected_batch_count=2,
        expected_sample_count=2,
        model_dimensions=(1, 4, 2),
    )


def average_ranks(values):
    # Independent, intentionally small O(N^2) oracle, including ties.
    return np.asarray([(values < x).sum() + ((values == x).sum() - 1) / 2 for x in values])


def oracle_correlation(x, y, centered=False, ranked=False):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.where(mask, x, np.nan).copy(), np.where(mask, y, np.nan).copy()
    if centered:
        for layer in range(len(x)):
            usable = mask[layer]
            if usable.any():
                x[layer, usable] -= x[layer, usable].mean()
                y[layer, usable] -= y[layer, usable].mean()
    x, y = x[mask], y[mask]
    if ranked:
        x, y = average_ranks(x), average_ranks(y)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def relationship_fixture(tmp_path):
    # Deliberately shuffled sample order and opaque IDs: pairing must use pair_id.
    pairs = np.asarray(["b", "a", "c", "a", "c", "b"])
    conditions = np.asarray(["pg19:32k"] * 3 + ["pg19:8k"] * 3)
    n, l, h, m, b = 6, 2, 4, 4, 3
    names = ["paper_token_primary", "alpha_1", "k_99", "recent_256"]
    a = {
        "sample_ids": np.asarray([f"opaque_{i}" for i in range(n)]),
        "sample_pair_ids": pairs,
        "sample_conditions": conditions,
        "sample_decode_lengths": np.full(n, 40),
        "condition_names": np.asarray(sorted(set(conditions))),
        "condition_ids": np.asarray(names),
        "budgets": np.asarray([128, 512, 2048]),
        "recall_scopes": np.asarray(["all_visible", "prompt_conditional"]),
        "sample_prefill_features": np.zeros((n, l, h, m, 5), dtype=np.float32),
        "sample_recall_full": np.zeros((n, l, h, m, b, 2, 2, 4), dtype=np.float32),
        "sample_learned_excess_full": np.zeros((n, l, h, m, b, 2, 4), dtype=np.float32),
        "sample_recall_quarters": np.zeros((n, 4, l, h, m, b, 2, 2), dtype=np.float32),
        "sample_recall_temporal": np.zeros((n, l, h, m, b, 2, 2, 3), dtype=np.float32),
    }
    for i, pair in enumerate(pairs):
        length_shift = 2 if conditions[i] == "pg19:32k" else 0
        base = np.arange(8).reshape(l, h) + {"a": 1, "b": 2, "c": 4}[pair] + length_shift
        a["sample_prefill_features"][i, ..., 0] = base[..., None]
        a["sample_recall_full"][i, ..., 0] = base[..., None, None, None, None] / 32
        a["sample_learned_excess_full"][i, ..., 0] = -base[..., None, None, None] / 32
        a["sample_recall_quarters"][i] = base[None, ..., None, None, None, None] / 32
        a["sample_recall_temporal"][i] = base[..., None, None, None, None, None] / 32
    return B2RelationshipInputs(a, tmp_path / "stage2_manifest.json", "stage2", "b2")


def stage2_fixture(tmp_path, monkeypatch):
    manifest = fixture(tmp_path)
    index = load(manifest)
    monkeypatch.setattr(cli, "load_completed_b2_run_set", lambda *a, **k: index)
    output = tmp_path / "analysis/stage2"
    common = ["--run-set", str(manifest), "--analysis-git-commit", "a" * 40]
    cli.prepare_analysis(
        cli.make_parser().parse_args(common + ["--analysis-id", "s2", "--output-dir", str(output), "--run-stage2"])
    )
    return index, output, common
