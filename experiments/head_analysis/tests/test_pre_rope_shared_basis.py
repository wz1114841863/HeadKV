from itertools import combinations
import struct

import numpy as np
import pytest
import torch

from experiments.head_analysis.analysis import pre_rope_shared_basis as shared
from experiments.head_analysis.analysis.pca_reconstruction import apply_position_rope
from experiments.head_analysis.common.io import read_json, write_json, write_npz
from experiments.head_analysis.tests.test_pre_rope_head_budget import fixture


def test_matching_is_complete_and_cka_and_error_are_separate_objectives():
    pairs = np.array(list(combinations(range(8), 2)))
    choices = list(shared.perfect_matchings(range(8)))
    assert len(choices) == len(set(choices)) == 105 and choices == sorted(choices)
    tied = shared.select_matching(np.ones(28), pairs)
    np.testing.assert_array_equal(pairs[tied], [[0, 1], [2, 3], [4, 5], [6, 7]])
    cka_pairs = {(0, 7), (1, 6), (2, 5), (3, 4)}
    error_pairs = {(0, 2), (1, 3), (4, 6), (5, 7)}
    cka = np.array([tuple(p) in cka_pairs for p in pairs], dtype=float)
    loss = np.array([0. if tuple(p) in error_pairs else 10. for p in pairs])
    assert set(map(tuple, pairs[shared.select_matching(cka, pairs, maximize=True)])) == cka_pairs
    chosen = pairs[shared.select_matching(loss, pairs)]
    assert set(map(tuple, chosen)) == error_pairs
    groups = shared.pair_group_map(chosen, 8)
    np.testing.assert_array_equal(np.bincount(groups), [2, 2, 2, 2])
    with pytest.raises(ValueError, match="exactly once"):
        shared.pair_group_map([[0, 1], [0, 3]], 4)


def test_same_subspace_and_cka_one_counterexample_and_energy_normalization():
    x = np.array([[1., 0, 0, 0], [-1., 0, 0, 0], [0, 1., 0, 0], [0, -1., 0, 0]])
    y = x[:, [2, 3, 0, 1]]
    cka = np.square(x.T @ y).sum() / np.sqrt(np.square(x.T @ x).sum()*np.square(y.T @ y).sum())
    assert cka == 1.
    g = np.stack((x.T @ x, y.T @ y)) / len(x)
    identical = shared.fit_layer_pairs(np.stack((g[0], g[0])), np.ones(2), 2)
    np.testing.assert_allclose(identical["delta"], 0, atol=1e-10)
    rotated = shared.fit_layer_pairs(g, np.ones(2), 2)
    np.testing.assert_allclose(rotated["head_cost"], 0, atol=1e-10)
    np.testing.assert_allclose(rotated["delta"], [.5], atol=1e-10)
    weighted = shared.fit_layer_pairs(np.array([np.diag([100., 0, 0, 0]), np.diag([0, 2., 0, 0])]),
                                     np.array([100., 1.]), 1)
    np.testing.assert_allclose(np.abs(weighted["pair_basis"][0, :, 0]), [0, 1, 0, 0])


def test_compact_parameter_roundtrip_and_real_32k_ledger(tmp_path):
    rng = np.random.default_rng(44)
    mean = rng.normal(size=(2, 8, 8))
    mapping = np.tile(shared.pair_group_map([[0, 7], [1, 6], [2, 5], [3, 4]], 8), (2, 1))
    basis = np.linalg.qr(rng.normal(size=(2, 4, 8, 8)))[0]
    path = tmp_path / "parameters.bin"
    shared.export_parameters(path, mean, basis, mapping, 4, 2)
    ledger = shared.model_ledger(mean, basis, 2, 4, [8192])
    assert path.stat().st_size == ledger["model_total_bytes"] == 4*(2*4*8*4 + mean.size)+10+16
    blob = path.read_bytes()
    assert struct.unpack('<HHHBBH', blob[:10]) == (2, 8, 8, 0, 2, 4)
    np.testing.assert_array_equal(np.frombuffer(blob[10:26], np.uint8).reshape(2, 8), mapping)
    model = shared.read_parameters(path)
    np.testing.assert_array_equal(model["basis"], basis[..., :4].astype('f4'))
    np.testing.assert_array_equal(model["mean"], mean.astype('f4'))
    # Each mapped pair reconstructs from its one stored group basis.
    for layer, head in np.ndindex(2, 8):
        np.testing.assert_array_equal(model["basis"][layer, model["groups"][layer, head]],
                                      basis[layer, mapping[layer, head], :, :4].astype('f4'))
    large_mean = np.broadcast_to(0., (32, 8, 128))
    large_basis = np.broadcast_to(0., (32, 4, 128, 128))
    budget = shared.model_ledger(large_mean, large_basis, 2, 64, [32768])
    assert budget["model_total_bytes"] == 4325642
    assert budget["cache_total_bytes"][0] == 1077936136
    np.testing.assert_array_equal(budget["kv_total_resident_bytes"][:, 0],
                                  4325642+np.array([1, 4, 16])*(1077936136+2147483648))


def _sources(directory, validation_scale=1.):
    bundle, codec, reference = fixture(directory, validation_scale=validation_scale, heads=4)
    cka = directory / "cka"
    rows = read_json(bundle / "split.json")["samples"]
    samples = []
    pairs = np.array(list(combinations(range(4), 2)))
    for row in rows:
        if row["split"] == "test":
            continue
        name = f"samples/{row['sample_id']}.npz"
        samples.append({**row, "numeric_file": name})
        if row["split"] == "calib":
            # Force CKA to prefer a pairing distinct from the index control.
            observed = np.tile([.2, .3, .95, .94, .3, .2], (2, 1))
            write_npz(cka / name, {"layer_ids": np.arange(2), "kv_head_ids": np.arange(4),
                "cka_pairs": pairs, "cka_k_pre_observed": observed, "cka_k_pre_excess": observed-.05})
        # Validation CKA paths deliberately do not exist: they must never be read.
    write_json(cka / "summary.json", {"status": "completed", "samples": samples})
    return bundle, codec, cka, reference


def test_end_to_end_freezes_before_validation_uses_stored_codec_and_preserves_sources(tmp_path, monkeypatch):
    bundle, codec, cka, reference = _sources(tmp_path / "source")
    output = tmp_path / "result"
    source_bytes = (bundle / "rope_calibration.npz").read_bytes()
    real_reader = shared.read_moments
    def guarded_reader(records, directory, ranks):
        if records[0]["split"] == "validation":
            assert read_json(output / "freeze.json")["status"] == "frozen_before_validation"
            assert all((output / f"{m}_parameters.bin").exists() for m in shared.METHODS)
        return real_reader(records, directory, ranks)
    monkeypatch.setattr(shared, "read_moments", guarded_reader)
    result = shared.run_analysis(bundle, codec, cka, output, rank=4)
    assert result["status"] == "completed" and result["inference_path_modified"] is False
    assert (bundle / "rope_calibration.npz").read_bytes() == source_bytes
    freeze = read_json(output / "freeze.json")
    ledger = read_json(output / "byte_ledger.json")
    for m, name in enumerate(shared.METHODS):
        model = shared.read_parameters(output / f"{name}_parameters.bin")
        assert (output / f"{name}_parameters.bin").stat().st_size == ledger[name]["model_total_bytes"]
        if m >= 2:
            assert ledger[name]["model_total_bytes"] < ledger['I64']["model_total_bytes"]
        for layer in range(2):
            if m >= 2:
                np.testing.assert_array_equal(np.bincount(model["groups"][layer]), [2, 2])
    assert freeze['group_maps'][2] != freeze['group_maps'][3]
    # Independent NumPy FP32/BF16 implementation of the saved payload at one Layer.
    with np.load(output / "validation_sampled.npz") as values:
        np.testing.assert_array_equal(values['coefficient_bytes'],
            np.broadcast_to(np.array([3584, 7168]), values['coefficient_bytes'].shape))
        for method, name in enumerate(shared.METHODS):
            model = shared.read_parameters(output / f"{name}_parameters.bin")
            for h in range(4):
                u = model['basis'][0, model['groups'][0, h]]
                mu = model['mean'][0, h]
                x = reference['pre'][h]
                code = torch.tensor((x-mu) @ u).bfloat16().float().numpy()
                decoded = torch.tensor(code @ u.T + mu).bfloat16().float().numpy()
                post = apply_position_rope(decoded[None], reference['c'], reference['s'], bf16=True)[0]
                expected = np.square(post-reference['post'][h]).sum(-1).mean()
                np.testing.assert_allclose(values['bf16_post'][0, method, 0, h, 0], expected, rtol=1e-12)
        np.testing.assert_array_equal(values['transform_only_normalized_mse'][:, 1], 0)
    with np.load(output / 'calib_full.npz') as values:
        assert values['normalized_mse'].shape == (6, 5, 2, 4, 2)
        np.testing.assert_allclose(values['normalized_mse'][..., 1], 0, atol=1e-10)
    # Refit after changing Validation only: all decisions and fitted payloads agree.
    monkeypatch.setattr(shared, "read_moments", real_reader)
    b2, c2, k2, _ = _sources(tmp_path / 'changed', validation_scale=3.)
    shared.run_analysis(b2, c2, k2, tmp_path / 'changed_result', rank=4)
    assert read_json(output/'freeze.json') == read_json(tmp_path/'changed_result/freeze.json')
    for name in shared.METHODS:
        assert (output/f'{name}_parameters.bin').read_bytes() == (tmp_path/f'changed_result/{name}_parameters.bin').read_bytes()
    with np.load(output/'validation_full.npz') as a, np.load(tmp_path/'changed_result/validation_full.npz') as b:
        assert not np.array_equal(a['normalized_mse'], b['normalized_mse'])


def test_validation_improvements_and_zero_denominators_remain_explicit():
    records = [{"sample_id": "x", "condition": "a", "source_group": "a"}]
    values = np.array([2., 1., .8, .9, 1.3])[None, :, None, None]
    summary = shared.summarize(values, records)
    np.testing.assert_allclose(summary['increase_over_I64'], [1., 0, -.2, -.1, .3])
    assert shared.passes_error_limits(summary)[2]
    values[:, 1] = 0
    zero = shared.summarize(values, records)
    assert np.isnan(zero['increase_over_I64']).all()
    assert zero['worst']['S-index']['zero_baseline_positive_count'] == 1
    assert not shared.passes_error_limits(zero)[2]
