import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from experiments.head_analysis.analysis.b1_offline import canonical_json_sha256, sha256_file
from experiments.head_analysis.analysis.b2_consistency import B2RunSetIndex, B2SampleIndex, B2SourceError
from experiments.head_analysis.analysis.b2_joint import (
    QUERY_RELATION_FEATURES,
    RCO_DELTAS,
    B2JointInputs,
    analyze_b2_joint,
    build_joint_profiles,
    correlate_profile_grid,
    load_b2_joint_inputs,
)
from experiments.head_analysis.analysis.b2_relationships import (
    B2RelationshipInputs,
    bootstrap_draws,
    correlate_profiles,
)
from experiments.head_analysis.scripts import analyze_b2_run_set as cli
from experiments.head_analysis.tests.b2_fixtures import oracle_correlation, relationship_fixture


def make_inputs(tmp_path, ruler=True):
    stage2 = relationship_fixture(tmp_path)
    a = {k: v.copy() for k, v in stage2.arrays.items()}
    if ruler:
        for key in list(a):
            if key.startswith("sample_"):
                a[key] = np.concatenate((a[key], a[key][:2]))
        a["sample_ids"][-2:] = ["ruler_a", "ruler_b"]
        a["sample_conditions"][-2:] = ["ruler:8k"] * 2
        a["sample_pair_ids"][-2:] = ""
        a["condition_names"] = np.asarray(sorted(set(a["sample_conditions"])))
    n, layers, heads = a["sample_prefill_features"].shape[:3]
    a["sample_decode_lengths"][:] = 300
    if ruler:
        a["sample_decode_lengths"][-2:] = 6
    rng = np.random.default_rng(171)
    a["sample_prefill_features"][..., 0] = rng.uniform(1, 9, (n, layers, heads, 4))
    q = rng.random((n, 1, layers, heads * 4, 7, 1)).astype(np.float32)
    qcount = np.broadcast_to(a["sample_decode_lengths"][:, None, None, None, None], q.shape[:-1]).copy()
    page = np.full((n, layers, heads, 2, 3, 9, 1), np.nan, dtype=np.float32)
    pc = np.zeros(page.shape[:-1], dtype=np.uint64)
    token = np.full((n, layers, heads * 4, 3, 9, 1), np.nan, dtype=np.float32)
    tc = np.zeros(token.shape[:-1], dtype=np.uint32)
    for i, length in enumerate(a["sample_decode_lengths"]):
        for di, delta in enumerate(RCO_DELTAS):
            if length > delta:
                page[i, ..., di, 0] = rng.random(page[i, ..., di, 0].shape)
                token[i, ..., di, 0] = rng.random(token[i, ..., di, 0].shape)
                pc[i, ..., di] = tc[i, ..., di] = length - delta
    b1 = dict(
        sample_ids=a["sample_ids"].copy(),
        sample_conditions=a["sample_conditions"].copy(),
        condition_names=a["condition_names"].copy(),
        query_phases=np.asarray(["decode"]),
        query_feature_names=np.asarray(QUERY_RELATION_FEATURES),
        query_statistics=np.asarray(["mean"]),
        query_sample_full=q,
        query_sample_full_valid_count=qcount.astype(np.uint32),
        page_scopes=np.asarray(["all_visible", "prompt_only"]),
        rco_k_values=np.asarray([8, 16, 32]),
        rco_deltas=np.asarray(RCO_DELTAS),
        rco_metrics=np.asarray(["rco_raw_mean"]),
        rco_sample_head_values=page,
        rco_sample_head_valid_pair_count=pc,
        token_k_values=np.asarray([16, 32, 64]),
        token_deltas=np.asarray(RCO_DELTAS),
        token_metrics=np.asarray(["rco_raw_mean"]),
        token_sample_head_metrics=token,
        token_sample_head_valid_pair_count=tc,
    )
    return B2JointInputs(B2RelationshipInputs(a, tmp_path / "s2/stage2_manifest.json", "s2", "b2"), b1, {})


def test_matrix_grid_matches_pairwise_reference_with_missing_and_tied_values():
    rng = np.random.default_rng(91)
    x = rng.integers(0, 6, (3, 3, 2, 4)).astype(float)
    y = rng.integers(0, 6, (3, 4, 2, 4)).astype(float)
    x[0, 1, 0, 2] = np.nan
    x[1, 2] = np.nan
    y[2, 0, 1, 0] = np.nan
    y[:, 3] = np.nan
    result = correlate_profile_grid(x, y)
    for i in range(3):
        for j in range(4):
            expected = correlate_profiles(x[:, i], y[:, j])
            for key in expected:
                np.testing.assert_allclose(result[key][:, i, j], expected[key], equal_nan=True, atol=1e-14)


def test_grid_layer_centering_and_nonbinary_float_constants():
    x = np.asarray([[[0, 1, 2, 3], [100, 101, 102, 103]]], dtype=float)
    y = x[..., ::-1].copy()
    estimate = correlate_profile_grid(x, y)
    np.testing.assert_allclose(estimate["values"][0, 0, 1], -1)
    assert (estimate["values"][0, 0, 0] > 0).all()
    constant = np.full((1, 32, 8), 0.1)
    assert np.isnan(correlate_profile_grid(constant, constant)["values"]).all()
    assert np.isnan(correlate_profiles(constant[0], constant[0])["values"]).all()


def test_gqa_mapping_and_full_configuration_scope_and_counts(tmp_path):
    inputs = make_inputs(tmp_path)
    inputs.b1["rco_sample_head_values"][0, 0, 0, 0, 0, 0, 0] = -0.05
    inputs.b1["token_sample_head_metrics"][0, 0, :4, 0, 0, 0] = [-0.01, -0.02, -0.03, -0.04]
    profiles = build_joint_profiles(inputs)
    assert profiles["x"].shape == (8, 76, 2, 4) and profiles["y"].shape == (8, 88, 2, 4)
    assert sum(m["primary"] for m in profiles["x_metadata"]) == 4
    assert sum(m["primary"] for m in profiles["y_metadata"]) == 19
    values = inputs.b1["query_sample_full"][:, 0, :, :, 0, 0].reshape(8, 2, 4, 4).mean(-1)
    np.testing.assert_allclose(profiles["y"][:, 0], values, atol=1e-7)
    for i, meta in enumerate(profiles["y_metadata"]):
        if meta["kind"] == "page_rco":
            si = 0 if meta["scope"] == "all_visible" else 1
            ki, di = [8, 16, 32].index(meta["k"]), RCO_DELTAS.index(meta["delta"])
            np.testing.assert_allclose(
                profiles["y"][:, i], inputs.b1["rco_sample_head_values"][:, :, :, si, ki, di, 0], equal_nan=True
            )
        if meta["delta"] and meta["delta"] >= 8:
            assert np.isnan(profiles["y"][-2:, i]).all()
            assert (profiles["y_source_count_max"][-2:, i] == 0).all()
    assert (profiles["y_valid_source_heads"][:, :7] == 4).all()
    token_index = next(i for i, m in enumerate(profiles["y_metadata"]) if m["id"] == "token_rco:all_visible:k16:d1")
    np.testing.assert_allclose(profiles["y"][0, token_index, 0, 0], -0.025)


def test_partial_gqa_uses_equal_finite_heads_and_retains_support(tmp_path):
    inputs = make_inputs(tmp_path)
    inputs.b1["query_sample_full"][0, 0, 0, :4, 0, 0] = [0, 1, 2, np.nan]
    inputs.b1["query_sample_full_valid_count"][0, 0, 0, :4, 0] = [2, 10, 300, 0]
    p = build_joint_profiles(inputs)
    assert p["y"][0, 0, 0, 0] == 1  # unweighted, not count-weighted
    assert p["y_valid_source_heads"][0, 0, 0, 0] == 3
    assert p["y_source_count_min"][0, 0, 0, 0] == 0
    assert p["y_source_count_max"][0, 0, 0, 0] == 300


def test_bootstrap_grid_matches_independent_sample_reconstruction_and_short_validity(tmp_path):
    inputs = make_inputs(tmp_path)
    result = analyze_b2_joint(inputs, bootstrap_replicates=7, bootstrap_batch_size=3)
    a, p = result.arrays, build_joint_profiles(inputs)
    assert set(a) == set(result.axes)
    assert all(v.ndim == len(result.axes[k]) for k, v in a.items())
    ci = a["condition_names"].tolist().index("pg19:32k")
    xi, yi = 0, 0
    rows = np.asarray([1, 0, 2])  # shuffled IDs, frozen pair order a,b,c
    x, y = p["x"][rows, xi], p["y"][rows, yi]
    expected = []
    for draw in bootstrap_draws(20261110, "pg19-pairs", 7, 3):
        expected.append(oracle_correlation(x[draw].mean(0), y[draw].mean(0), centered=True, ranked=True))
    np.testing.assert_allclose(a["correlation_ci"][ci, xi, yi, 1, 0], np.quantile(expected, [0.025, 0.975]), atol=2e-7)
    ruler = a["condition_names"].tolist().index("ruler:8k")
    for j, metadata in enumerate(result.metadata["y_profiles"]):
        if metadata["delta"] and metadata["delta"] >= 8:
            assert np.isnan(a["correlation"][ruler, :, j]).all()
            assert (a["correlation_valid_replicates"][ruler, :, j] == 0).all()
            assert (a["correlation_valid_heads"][ruler, :, j] == 0).all()
    assert result.metadata["draws_sha256"]["pg19:8k"] == result.metadata["draws_sha256"]["pg19:32k"]
    second = analyze_b2_joint(inputs, bootstrap_replicates=7, bootstrap_batch_size=4)
    for key in a:
        if a[key].dtype.kind == "f":
            np.testing.assert_allclose(second.arrays[key], a[key], equal_nan=True, atol=2e-7)


def test_identical_pg19_profiles_have_zero_paired_correlation_difference_ci(tmp_path):
    inputs = make_inputs(tmp_path, ruler=False)
    low, high = [3, 5, 4], [1, 0, 2]  # a,b,c
    for key, value in inputs.stage2.arrays.items():
        if key.startswith("sample_") and value.dtype.kind == "f":
            value[high] = value[low]
    for key, value in inputs.b1.items():
        if key.startswith(("query_sample_", "rco_sample_", "token_sample_")):
            value[high] = value[low]
    result = analyze_b2_joint(inputs, bootstrap_replicates=9, bootstrap_batch_size=4)
    assert np.ptp(result.arrays["correlation_ci"][0, 0, 0, 1, 0]) > 0
    np.testing.assert_allclose(result.arrays["pg19_correlation_difference"], 0, atol=1e-14)
    np.testing.assert_allclose(result.arrays["pg19_correlation_ci"], 0, atol=1e-14)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))


def entries(paths):
    return [{"relative_path": p.name, "file_bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in paths]


def source_fixture(tmp_path):
    inputs = make_inputs(tmp_path)
    a = {k: v.copy() for k, v in inputs.stage2.arrays.items()}
    a.update(
        statistics=np.asarray(["mean", "median", "p10", "p90"]),
        set_types=np.asarray(["learned_middle", "augmented_prompt"]),
        time_metrics=np.asarray(["slope", "normalized_auc", "last_minus_first_quarter"]),
    )
    s2, s3, b1root = tmp_path / "s2", tmp_path / "s3", tmp_path / "b1"
    s2.mkdir()
    s3.mkdir()
    b1root.mkdir()
    np.savez_compressed(s2 / "stage2_profiles.npz", **a)
    condition_counts = {str(c): int((a["sample_conditions"] == c).sum()) for c in a["condition_names"]}
    p2 = {
        "source_run_set": {"sha256": "b2"},
        "sample_ids": a["sample_ids"].tolist(),
        "stage2_parameters": {},
        "stage2_parameters_sha256": canonical_json_sha256({}),
    }
    write(s2 / "analysis_plan.json", p2)
    write(
        s2 / "stage2_summary.json",
        {
            "plan_sha256": sha256_file(s2 / "analysis_plan.json"),
            "source_run_set_sha256": "b2",
            "condition_counts": condition_counts,
            "field_axes": {k: [] for k in a},
        },
    )
    write(
        s2 / "stage2_manifest.json",
        {
            "schema_version": "headkv-v1-b2-offline-stage2-manifest-v1",
            "status": "partial",
            "completed_stages": ["source_validation", "distributions_and_recall"],
            "source_run_set_sha256": "b2",
            "plan_sha256": sha256_file(s2 / "analysis_plan.json"),
            "outputs": entries([s2 / n for n in ["analysis_plan.json", "stage2_profiles.npz", "stage2_summary.json"]]),
        },
    )
    ref2 = {"path": str(s2 / "stage2_manifest.json"), "sha256": sha256_file(s2 / "stage2_manifest.json")}
    p3 = {
        "source_run_set": {"sha256": "b2"},
        "stage2_source": ref2,
        "parameters": {},
        "stage3_parameters": {},
        "parameters_sha256": canonical_json_sha256({}),
        "stage3_parameters_sha256": canonical_json_sha256({}),
        "sample_ids": a["sample_ids"].tolist(),
    }
    write(s3 / "analysis_plan.json", p3)
    np.savez_compressed(
        s3 / "stage3_relationships.npz",
        **{k: a[k] for k in ["sample_ids", "sample_pair_ids", "condition_names", "condition_ids"]},
    )
    write(
        s3 / "stage3_summary.json",
        {
            "schema_version": "headkv-v1-b2-offline-stage3-v1",
            "status": "partial",
            "plan_sha256": sha256_file(s3 / "analysis_plan.json"),
            "stage2_source": ref2,
            "source_run_set_sha256": "b2",
        },
    )
    write(
        s3 / "stage3_manifest.json",
        {
            "schema_version": "headkv-v1-b2-offline-stage3-manifest-v1",
            "status": "partial",
            "completed_stages": ["source_validation", "cross_phase_relations"],
            "source_run_set_sha256": "b2",
            "stage2_source": ref2,
            "plan_sha256": sha256_file(s3 / "analysis_plan.json"),
            "outputs": entries(
                [s3 / n for n in ["analysis_plan.json", "stage3_relationships.npz", "stage3_summary.json"]]
            ),
        },
    )
    # Reverse B1 sample order to require an explicit ID join.
    shuffled = {}
    for key, value in inputs.b1.items():
        shuffled[key] = (
            value[::-1] if key.startswith(("sample_", "query_sample_", "rco_sample_", "token_sample_")) else value
        )
    np.savez_compressed(b1root / "head_profiles.npz", **shuffled)
    write(
        b1root / "analysis_manifest.json",
        {
            "schema_version": "headkv-v1-b1-offline-completed-v1",
            "status": "completed",
            "sample_count": 8,
            "source_run_set_sha256": "b1",
            "outputs": entries([b1root / "head_profiles.npz"]),
        },
    )
    samples = {}
    for i, (sid, condition) in enumerate(zip(a["sample_ids"], a["sample_conditions"])):
        dataset, length = str(condition).split(":")
        samples[str(sid)] = B2SampleIndex(
            str(sid),
            dataset,
            "task",
            str(a["sample_pair_ids"][i]) or None,
            int(length[:-1]) * 1024,
            int(length[:-1]) * 1024,
            "input",
            "generated",
            int(a["sample_decode_lengths"][i]) + 1,
            int(a["sample_decode_lengths"][i]),
            "batch",
            tmp_path,
            (),
        )
    payload = {
        "execution_contract": {
            "dimensions": {"L": 2, "Hkv": 4, "Hq": 16, "M": 4, "Nb": 3},
            "axes": {"conditions": a["condition_ids"].tolist()},
        },
        "b1_analysis_authority": {"sha256": sha256_file(b1root / "analysis_manifest.json")},
        "b1_replay_authority": {"sha256": "b1"},
        "validation_report": {},
        "summary": {},
    }
    index = B2RunSetIndex(
        tmp_path / "source.json",
        tmp_path,
        "b2",
        payload,
        samples,
        (),
        tmp_path / "report.json",
        tmp_path / "b1.json",
        b1root / "analysis_manifest.json",
        1,
    )
    return index, s2, s3


def load_sources(index, s2, s3):
    return load_b2_joint_inputs(
        index,
        s2,
        s3,
        expected_stage2_sha256=sha256_file(s2 / "stage2_manifest.json"),
        expected_stage3_sha256=sha256_file(s3 / "stage3_manifest.json"),
    )


def test_source_loader_hash_binds_bundle_and_joins_ids_not_row_order(tmp_path):
    index, s2, s3 = source_fixture(tmp_path)
    loaded = load_sources(index, s2, s3)
    assert loaded.b1["sample_ids"].tolist() == list(index.samples)
    assert loaded.sources["b1_row_order_in_b2_order"] == list(range(7, -1, -1))


def test_joint_normal_read_does_not_require_b1_file_hash(tmp_path):
    index, s2, s3 = source_fixture(tmp_path)
    with (index.b1_analysis_path.parent / "head_profiles.npz").open("ab") as handle:
        handle.write(b"extra")
    assert len(load_b2_joint_inputs(index, s2, s3).b1["sample_ids"]) == len(index.samples)


@pytest.mark.parametrize(
    "kind", ["stage3_hash", "b1_hash", "duplicate_id", "condition", "gqa", "count", "axis", "zero_filled"]
)
def test_joint_source_rejects_hash_or_semantic_drift(tmp_path, kind):
    index, s2, s3 = source_fixture(tmp_path)
    path = index.b1_analysis_path.parent / "head_profiles.npz"
    if kind == "stage3_hash":
        (s3 / "stage3_summary.json").write_text("{}")
    elif kind == "b1_hash":
        with path.open("ab") as f:
            f.write(b"drift")
    else:
        with np.load(path) as archive:
            arrays = {k: archive[k] for k in archive.files}
        if kind == "duplicate_id":
            arrays["sample_ids"][0] = arrays["sample_ids"][1]
        elif kind == "condition":
            arrays["sample_conditions"][0] = "pg19:8k"
        elif kind == "gqa":
            arrays["query_sample_full"] = arrays["query_sample_full"][:, :, :, :-1]
        elif kind == "count":
            arrays["token_sample_head_valid_pair_count"][:] = 9999
        elif kind == "axis":
            arrays["rco_k_values"] = np.asarray([16, 8, 32])
        else:
            arrays["rco_sample_head_values"] = np.nan_to_num(arrays["rco_sample_head_values"])
        np.savez_compressed(path, **arrays)
        m = json.loads(index.b1_analysis_path.read_text())
        m["outputs"] = entries([path])
        write(index.b1_analysis_path, m)
        payload = copy.deepcopy(index.payload)
        payload["b1_analysis_authority"]["sha256"] = sha256_file(index.b1_analysis_path)
        index = replace(index, payload=payload)
    with pytest.raises(B2SourceError):
        load_sources(index, s2, s3)


def test_cli_stage4_partial_hashes_no_overwrite_and_bad_source_no_output(tmp_path, monkeypatch):
    index, s2, s3 = source_fixture(tmp_path)
    monkeypatch.setattr(cli, "load_completed_b2_run_set", lambda *a, **k: index)
    output = tmp_path / "stage4"
    args = cli.make_parser().parse_args(
        [
            "--run-set",
            str(index.manifest_path),
            "--analysis-id",
            "stage4",
            "--analysis-git-commit",
            "a" * 40,
            "--run-stage4",
            "--stage2-dir",
            str(s2),
            "--stage3-dir",
            str(s3),
            "--output-dir",
            str(output),
            "--expected-stage2-manifest-sha256",
            sha256_file(s2 / "stage2_manifest.json"),
            "--expected-stage3-manifest-sha256",
            sha256_file(s3 / "stage3_manifest.json"),
            "--bootstrap-replicates",
            "3",
            "--bootstrap-batch-size",
            "2",
        ]
    )
    cli.prepare_analysis(args)
    assert {p.name for p in output.iterdir()} == {
        "analysis_plan.json",
        "stage4_joint_profiles.npz",
        "stage4_summary.json",
        "stage4_manifest.json",
    }
    manifest = json.loads((output / "stage4_manifest.json").read_text())
    assert manifest["status"] == "partial" and manifest["pending_stages"] == ["finalize"]
    for e in manifest["outputs"]:
        p = output / e["relative_path"]
        assert sha256_file(p) == e["sha256"] and p.stat().st_size == e["file_bytes"]
    summary = json.loads((output / "stage4_summary.json").read_text())
    assert summary["relation_count"] == 3 * 76 * 88 and len(summary["primary_records"]) == 3 * 4 * 19
    assert any(v["coefficients"] == [[None, None], [None, None]] for v in summary["primary_records"])
    assert not (output / "analysis_manifest.json").exists()
    with pytest.raises(B2SourceError, match="already exists"):
        cli.prepare_analysis(args)
    args.output_dir = str(tmp_path / "bad_input")
    args.expected_stage3_manifest_sha256 = "0" * 64
    with pytest.raises(B2SourceError):
        cli.prepare_analysis(args)
    assert not (tmp_path / "bad_input").exists()


@pytest.mark.parametrize("kind", ["counts", "two_stages", "missing_stage3"])
def test_invalid_joint_arguments_reject_before_creating_output(tmp_path, kind):
    args = cli.make_parser().parse_args(
        [
            "--run-set",
            "unused",
            "--analysis-id",
            "x",
            "--analysis-git-commit",
            "a" * 40,
            "--output-dir",
            str(tmp_path / "out"),
            "--run-stage4",
            "--stage2-dir",
            "unused",
        ]
    )
    if kind == "counts":
        args.stage3_dir = "unused"
        args.bootstrap_batch_size = 0
    elif kind == "two_stages":
        args.run_stage3 = True
    with pytest.raises(B2SourceError):
        cli.prepare_analysis(args)
    assert not (tmp_path / "out").exists()
