import json

import numpy as np
import pytest

from experiments.head_analysis.analysis.b2_consistency import B2SourceError
from experiments.head_analysis.common.io import sha256_file
from experiments.head_analysis.scripts import finalize_b2_analysis as finalizer


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def make_stages(root):
    directories = []
    for number in (2, 3, 4):
        directory = root / f"arbitrary_{number}"
        directory.mkdir()
        ids = ["a", "b"]
        arrays = {"sample_ids": np.asarray(ids), "sample_pair_ids": np.asarray(["a", "b"]),
                  "condition_names": np.asarray(["pg19:8k"]), "sample_conditions": np.asarray(["pg19:8k"] * 2),
                  "metric": np.asarray([0.5, np.nan], dtype=np.float32), "valid_count": np.asarray([1, 0])}
        numeric = finalizer.NUMERIC[number]
        np.savez_compressed(directory / numeric, **arrays)
        plan = {"source_run_set": {"sha256": "source"}, "sample_ids": ids,
                "condition_counts": {"pg19:8k": 2}, "parameters": {"bootstrap_replicates": 17},
                "analysis_code": {"git_commit": None}}
        write(directory / "analysis_plan.json", plan)
        name = f"stage{number}"
        axes = {key: ["N"] for key in arrays}
        summary = {"schema_version": f"headkv-v1-b2-offline-{name}-v1", "status": "partial",
                   "source_run_set_sha256": "source", "plan_sha256": "plan", "valid_count": 1,
                   ("field_axes" if number == 2 else "axes"): axes}
        write(directory / f"{name}_summary.json", summary)
        manifest = {"schema_version": f"headkv-v1-b2-offline-{name}-manifest-v1", "status": "partial",
                    "source_run_set_sha256": "source", "plan_sha256": "plan",
                    "outputs": [{"relative_path": filename} for filename in
                                ("analysis_plan.json", f"{name}_summary.json", numeric)]}
        if number == 3:
            manifest["stage2_source"] = {"sha256": sha256_file(directories[0] / "stage2_manifest.json")}
        if number == 4:
            manifest["joint_sources"] = {"stage2": {"sha256": sha256_file(directories[0] / "stage2_manifest.json")},
                                         "stage3": {"sha256": sha256_file(directories[1] / "stage3_manifest.json")}}
        write(directory / f"{name}_manifest.json", manifest)
        directories.append(directory)
    return directories


def test_finalize_current_stages_without_historical_artifacts(tmp_path):
    stages = make_stages(tmp_path)
    output = tmp_path / "bundle"
    manifest = finalizer.finalize(*stages, output, "test")
    result = json.loads(manifest.read_text())
    assert result["sample_count"] == 2 and result["status"] == "completed"
    assert "不预设研究结论" in (output / "REPORT.md").read_text()
    for number, source in zip((2, 3, 4), stages):
        name = finalizer.NUMERIC[number]
        assert (source / name).read_bytes() == (output / "stages" / f"stage{number}" / name).read_bytes()
        assert json.loads((output / "stages" / f"stage{number}" / "analysis_plan.json").read_text())["parameters"]["bootstrap_replicates"] == 17
    catalog = json.loads((output / "array_catalog.json").read_text())
    assert catalog["stage2"]["fields"]["metric"]["axes"] == ["N"]


@pytest.mark.parametrize("kind", ["missing", "source", "coordinates", "link", "axes"])
def test_invalid_sources_fail_before_output(tmp_path, kind):
    stages = make_stages(tmp_path)
    if kind == "missing":
        (stages[2] / finalizer.NUMERIC[4]).unlink()
    elif kind == "source":
        path = stages[2] / "stage4_summary.json"
        payload = json.loads(path.read_text()); payload["source_run_set_sha256"] = "other"; write(path, payload)
    elif kind == "coordinates":
        path = stages[2] / finalizer.NUMERIC[4]
        with np.load(path) as a:
            data = dict(a)
        data["sample_ids"] = np.asarray(["b", "a"])
        np.savez_compressed(path, **data)
    elif kind == "link":
        path = stages[2] / "stage4_manifest.json"
        payload = json.loads(path.read_text()); payload["joint_sources"]["stage3"]["sha256"] = "other"; write(path, payload)
    else:
        path = stages[2] / "stage4_summary.json"
        payload = json.loads(path.read_text()); payload["axes"]["metric"] = []; write(path, payload)
    output = tmp_path / "bundle"
    with pytest.raises(B2SourceError):
        finalizer.finalize(*stages, output, "test")
    assert not output.exists()


def test_cli_accepts_optional_report_and_refuses_overwrite(tmp_path):
    stages = make_stages(tmp_path)
    report = tmp_path / "report.md"; report.write_text("my conclusion")
    output = tmp_path / "bundle"
    finalizer.main(["--stage2-dir", str(stages[0]), "--stage3-dir", str(stages[1]), "--stage4-dir", str(stages[2]),
                    "--output-dir", str(output), "--analysis-id", "test", "--report", str(report)])
    assert (output / "REPORT.md").read_text() == "my conclusion"
    with pytest.raises(B2SourceError, match="already exists"):
        finalizer.finalize(*stages, output, "test")


def test_reader_accepts_real_stage2_and_stage3_outputs(tmp_path, monkeypatch):
    from experiments.head_analysis.scripts import analyze_b2_run_set
    from experiments.head_analysis.tests.b2_fixtures import stage2_fixture

    index, stage2, common = stage2_fixture(tmp_path, monkeypatch)
    stage3 = tmp_path / "analysis/stage3"
    args = analyze_b2_run_set.make_parser().parse_args(
        common + ["--analysis-id", "s3", "--output-dir", str(stage3), "--run-stage3",
                  "--stage2-dir", str(stage2), "--bootstrap-replicates", "3"]
    )
    analyze_b2_run_set.prepare_analysis(args)
    for number, directory in ((2, stage2), (3, stage3)):
        observed = finalizer._read_stage(number, directory)
        assert observed["coordinates"]["sample_ids"] == list(index.samples)


def test_reader_accepts_real_stage4_output(tmp_path, monkeypatch):
    from experiments.head_analysis.scripts import analyze_b2_run_set
    from experiments.head_analysis.tests.test_b2_joint import source_fixture

    index, stage2, stage3 = source_fixture(tmp_path)
    monkeypatch.setattr(analyze_b2_run_set, "load_completed_b2_run_set", lambda *a, **k: index)
    output = tmp_path / "stage4"
    args = analyze_b2_run_set.make_parser().parse_args([
        "--run-set", str(index.manifest_path), "--analysis-id", "s4", "--run-stage4",
        "--stage2-dir", str(stage2), "--stage3-dir", str(stage3), "--output-dir", str(output),
        "--bootstrap-replicates", "3", "--bootstrap-batch-size", "2",
    ])
    analyze_b2_run_set.prepare_analysis(args)
    assert finalizer._read_stage(4, output)["coordinates"]["sample_ids"] == list(index.samples)
