import hashlib
import json

import pytest

from experiments.head_analysis.scripts import analyze_b1_run_set, finalize_b1_analysis
from experiments.head_analysis.tests.fixtures import make_fixture


def test_finalize_builds_completed_numeric_table_and_figure_bundle(tmp_path, monkeypatch):
    run_set, report, _ = make_fixture(tmp_path)
    stage3 = tmp_path / "stage3"
    stage4 = tmp_path / "stage4"
    stage5 = tmp_path / "stage5"
    completed = tmp_path / "completed"
    common = [
        "--run-set",
        str(run_set),
        "--source-root",
        str(tmp_path),
        "--page-rco-report",
        str(report),
        "--analysis-git-commit",
        "b" * 40,
        "--expected-batches",
        "1",
        "--expected-samples",
        "2",
        "--bootstrap-replicates",
        "2",
        "--bootstrap-seed",
        "17",
    ]
    for stage, flag in ((stage3, "--run-stage3"), (stage4, "--run-stage4")):
        arguments = common + ["--analysis-id", stage.name, "--output-dir", str(stage), flag]
        if flag == "--run-stage4":
            arguments += ["--token-comparison-chunk-steps", "1"]
        assert analyze_b1_run_set.main(arguments) == 0
    assert (
        analyze_b1_run_set.main(
            common
            + [
                "--analysis-id",
                stage5.name,
                "--output-dir",
                str(stage5),
                "--run-stage5",
                "--stage3-dir",
                str(stage3),
                "--stage4-dir",
                str(stage4),
            ]
        )
        == 0
    )

    arguments = [
        "--stage3-dir",
        str(stage3),
        "--stage4-dir",
        str(stage4),
        "--stage5-dir",
        str(stage5),
        "--analysis-id",
        completed.name,
        "--analysis-git-commit",
        "b" * 40,
        "--output-dir",
        str(completed),
    ]
    assert finalize_b1_analysis.main(arguments) == 0

    manifest = json.loads((completed / "analysis_manifest.json").read_text())
    assert manifest["status"] == "completed"
    registered = {entry["relative_path"]: entry for entry in manifest["outputs"]}
    expected = {
        "condition_summary.json",
        "head_profiles.npz",
        "page_rco_summary.json",
        "token_stability_summary.json",
        "correlations.json",
        "correlation_profiles.npz",
        "tables/page_rco.csv",
        "tables/token_stability.csv",
        "tables/relationships.csv",
        "figures/page_rco_decay_k32.png",
        "figures/token_rco_decay_k32.png",
        "figures/cross_condition_head_identity.png",
        "figures/stability_relationships.png",
    }
    assert expected <= set(registered)
    for relative, entry in registered.items():
        path = completed / relative
        assert path.stat().st_size == entry["file_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    assert all((completed / path).stat().st_size > 1000 for path in expected if path.endswith(".png"))
    with pytest.raises(SystemExit, match="already exists"):
        finalize_b1_analysis.main(arguments)
