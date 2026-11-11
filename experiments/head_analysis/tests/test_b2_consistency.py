import shutil

import numpy as np
import pytest

from experiments.head_analysis.analysis import b2_consistency as b2
from experiments.head_analysis.scripts import analyze_b2_run_set as cli
from experiments.head_analysis.tests.b2_fixtures import fixture, load, read, sha, write


def test_indexes_real_writer_outputs_and_ignores_unregistered_diagnostic_directory(tmp_path):
    manifest = fixture(tmp_path)
    write(tmp_path / "data/b2/unregistered_broken/run_metadata.json", {"status": "running"})
    index = load(manifest)
    assert len(index.batches) == len(index.samples) == 2
    assert index.condition_counts == {"pg19:8k": 1, "pg19:32k": 1}
    assert all(s.decode_length == 2 and len(s.decode_chunks) == 2 for s in index.samples.values())
    plan = b2.build_b2_analysis_plan(
        index, analysis_id="unit-plan", analysis_code={"git_commit": "a" * 40, "dirty": False}
    )
    assert plan["status"] == "prepared"
    assert plan["pending_stages"] and plan["pg19_pair_ids"] == ["pair"]
    assert plan["source_run_set"]["sha256"] == sha(manifest)


def test_rejects_wrong_external_authority(tmp_path):
    manifest = fixture(tmp_path)
    with pytest.raises(b2.B2SourceError, match="external authority"):
        b2.load_completed_b2_run_set(manifest, source_root=tmp_path, expected_run_set_sha256="0" * 64)


def test_rejects_missing_decode_chunk(tmp_path):
    manifest = fixture(tmp_path)
    (tmp_path / "data/b2/b0/samples/pg19_pair_0/decode_recall_001.npz").unlink()
    with pytest.raises(b2.B2SourceError):
        load(manifest)


def test_normal_read_does_not_require_historical_artifacts_or_current_file_hash(tmp_path):
    manifest = fixture(tmp_path)
    source = tmp_path / "data/b2/b0/samples/pg19_pair_0/prefill_consistency.npz"
    source.write_bytes(source.read_bytes() + b"extra")
    shutil.rmtree(tmp_path / "data/b1analysis")
    shutil.rmtree(tmp_path / "data/validation")
    shutil.rmtree(tmp_path / "data/old")
    index = load(manifest)
    assert len(index.samples) == 2


@pytest.mark.parametrize("kind", ["duplicate", "path_escape", "wrong_scope"])
def test_semantic_failures_for_invalid_sample_membership(tmp_path, kind):
    manifest = fixture(tmp_path)
    payload = read(manifest)
    if kind == "duplicate":
        payload["batches"][1]["sample_ids"] = payload["batches"][0]["sample_ids"]
    elif kind == "path_escape":
        payload["batches"][0]["run_dir"] = "../outside"
    else:
        payload["batches"][0]["b1_batch_id"] = "b1"
    write(manifest, payload)
    with pytest.raises(b2.B2SourceError):
        load(manifest)


def test_cli_writes_only_plan_and_refuses_existing_output(tmp_path, monkeypatch):
    manifest = fixture(tmp_path)
    monkeypatch.setattr(cli, "load_completed_b2_run_set", lambda *a, **k: load(manifest))
    output = tmp_path / "analysis/new"
    args = cli.make_parser().parse_args(
        [
            "--run-set",
            str(manifest),
            "--analysis-id",
            "new",
            "--analysis-git-commit",
            "a" * 40,
            "--output-dir",
            str(output),
        ]
    )
    plan_path = cli.prepare_analysis(args)
    assert {p.name for p in output.iterdir()} == {"analysis_plan.json"}
    assert read(plan_path)["status"] == "prepared"
    before = plan_path.read_bytes()
    with pytest.raises(b2.B2SourceError, match="already exists"):
        cli.prepare_analysis(args)
    assert plan_path.read_bytes() == before


def test_cli_source_failure_creates_no_output(tmp_path, monkeypatch):
    manifest = fixture(tmp_path)
    monkeypatch.setattr(cli, "load_completed_b2_run_set", lambda *a, **k: load(manifest))
    (tmp_path / "data/b2/b0/samples/pg19_pair_0/decode_recall_001.npz").unlink()
    output = tmp_path / "analysis/new"
    args = cli.make_parser().parse_args(
        [
            "--run-set",
            str(manifest),
            "--analysis-id",
            "new",
            "--analysis-git-commit",
            "a" * 40,
            "--output-dir",
            str(output),
        ]
    )
    with pytest.raises(b2.B2SourceError):
        cli.prepare_analysis(args)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["cv", "sentinel", "step_gap", "top1", "mass", "shape"])
def test_rejects_semantically_invalid_arrays_with_updated_file_hashes(tmp_path, kind):
    manifest = fixture(tmp_path)
    payload = read(manifest)
    row = payload["batches"][0]
    run = tmp_path / row["run_dir"]
    name = "prefill_consistency.npz" if kind in ("cv", "sentinel") else "decode_recall_000.npz"
    path = run / "samples/pg19_pair_0" / name
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if kind == "cv":
        arrays["cv_prefill"] += 1
    elif kind == "sentinel":
        arrays["critical_token_positions"][..., -1] = 65
    elif kind == "step_gap":
        arrays["decode_steps"][:] = 7
    elif kind == "top1":
        arrays["predicted_token_ids"][:] = 99
    elif kind == "mass":
        arrays["critical_attention_mass"][..., 1, 0] = 0.01
    else:
        arrays["prompt_attention_mass"] = arrays["prompt_attention_mass"][..., :1]
    np.savez_compressed(path, **arrays)
    file_manifest = read(run / "file_manifest.json")
    for entry in file_manifest["files"]:
        source = run / entry["relative_path"]
        entry["sha256"] = sha(source)
        entry["file_bytes"] = source.stat().st_size
        if source == path:
            entry["arrays"] = [
                {"name": key, "shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()
            ]
            entry["logical_array_bytes"] = sum(value.nbytes for value in arrays.values())
    write(run / "file_manifest.json", file_manifest)
    row["file_manifest_sha256"] = sha(run / "file_manifest.json")
    row["artifact_file_bytes"] = sum(e["file_bytes"] for e in file_manifest["files"])
    write(manifest, payload)
    with pytest.raises(b2.B2SourceError):
        load(manifest)


def test_cli_accepts_local_code_without_git_or_source_fingerprints(tmp_path, monkeypatch):
    manifest = fixture(tmp_path)
    monkeypatch.setattr(cli, "load_completed_b2_run_set", lambda *a, **k: load(manifest))
    output = tmp_path / "metadata-only"
    args = cli.make_parser().parse_args(
        [
            "--run-set",
            str(manifest),
            "--analysis-id",
            "metadata-only",
            "--output-dir",
            str(output),
        ]
    )
    assert args.expected_run_set_sha256 is None
    assert args.expected_stage2_manifest_sha256 is None
    assert args.expected_stage3_manifest_sha256 is None
    cli.prepare_analysis(args)
    plan = read(output / "analysis_plan.json")
    assert plan["analysis_code"] == {"git_commit": None}
