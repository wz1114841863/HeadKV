import json

import numpy as np
import pytest

from experiments.head_analysis.analysis.prefill_stability import (
    CV_ID, TARGETS, _correlation, evaluate_condition, load_inputs, predict_fold,
)
from experiments.head_analysis.scripts.analyze_prefill_stability import main
from experiments.head_analysis.common.statistics import bootstrap_draws, correlation_1d


def profiles():
    rng = np.random.default_rng(14)
    x = rng.normal(size=(6, 2, 4))
    offsets = np.arange(8).reshape(2, 4) / 3
    y = 1.75 * x + offsets
    return x, y


def test_weighted_fit_matches_independent_head_centered_ols_and_excludes_test():
    x, y = profiles()
    weights = np.array([[0, 2, 1, 0, 3, 1]], dtype=float)
    history, prediction, beta = predict_fold(x, y, 0, weights)
    mean_x = np.average(x[1:], axis=0, weights=weights[0, 1:])
    mean_y = np.average(y[1:], axis=0, weights=weights[0, 1:])
    numerator = sum(weights[0, j] * ((x[j] - mean_x) * (y[j] - mean_y)).sum() for j in range(1, 6))
    denominator = sum(weights[0, j] * ((x[j] - mean_x) ** 2).sum() for j in range(1, 6))
    assert beta[0] == pytest.approx(numerator / denominator)
    np.testing.assert_allclose(history[0], mean_y)
    np.testing.assert_allclose(prediction[0], mean_y + beta[0] * (x[0] - mean_x))
    changed = y.copy(); changed[0] = 1e9
    new_history, new_prediction, new_beta = predict_fold(x, changed, 0, weights)
    np.testing.assert_array_equal(history, new_history)
    np.testing.assert_array_equal(prediction, new_prediction)
    np.testing.assert_array_equal(beta, new_beta)


def test_group_holdout_excludes_all_pair_rows_and_bootstrap_is_batch_invariant():
    x, y = profiles()
    groups = np.array(["p0", "p0", "p1", "p1", "p2", "p2"])
    a = evaluate_condition(x, y, groups, replicates=31, seed=12, batch_size=4)
    b = evaluate_condition(x, y, groups, replicates=31, seed=12, batch_size=11)
    for left, right in zip(a, b):
        np.testing.assert_allclose(left, right, equal_nan=True, rtol=1e-11, atol=1e-11)
    changed = y.copy(); changed[1] += 100
    c = evaluate_condition(x, changed, groups, replicates=31, seed=12, batch_size=4)
    # Held-out partner is excluded from both training means and beta for row 0.
    assert a[2][0] == pytest.approx(c[2][0])
    assert a[0][0, 1] == pytest.approx(c[0][0, 1])
    assert a[0][0, 2] == pytest.approx(c[0][0, 2])


def test_bootstrap_refit_matches_manual_group_resampling():
    x, y = profiles()
    groups = np.array(["p0", "p0", "p1", "p1", "p2", "p2"])
    seed, label = 21, "manual"
    _, _, _, actual = evaluate_condition(x, y, groups, replicates=1, seed=seed, label=label)
    draws = bootstrap_draws(seed, label, 1, 3)[0]
    frequency = np.bincount(draws, minlength=3)
    deltas, weights = [], []
    for i in range(len(x)):
        group = i // 2
        if frequency[group] == 0:
            continue
        w = np.repeat(frequency / 2, 2).astype(float)
        w[2 * group : 2 * group + 2] = 0
        if (w > 0).sum() < 2:
            continue
        mx = np.average(x, axis=0, weights=w)
        my = np.average(y, axis=0, weights=w)
        xx = sum(w[j] * np.sum((x[j] - mx) ** 2) for j in range(len(x)))
        xy = sum(w[j] * np.sum((x[j] - mx) * (y[j] - my)) for j in range(len(x)))
        prediction = my + xy / xx * (x[i] - mx)
        center = lambda a: (a - a.mean(axis=-1, keepdims=True)).ravel()
        rho_history = correlation_1d(center(my), center(y[i]))[1]
        rho_prediction = correlation_1d(center(prediction), center(y[i]))[1]
        delta_mse = np.mean((my - y[i]) ** 2) - np.mean((prediction - y[i]) ** 2)
        deltas.append([rho_prediction - rho_history, delta_mse])
        weights.append(frequency[group] / 2)
    np.testing.assert_allclose(actual[0], np.average(deltas, axis=0, weights=weights), atol=1e-12)


def test_sample_correlation_is_pairwise_layer_centered_and_undefined_stays_nan():
    x = np.array([[1., 2, 3, 4], [10, 12, 13, 14]])
    y = np.array([[1., 2, 3, 4], [20, 22, 23, 24]])
    rho, count = _correlation(x, y)
    assert rho == pytest.approx(1)
    assert count == 8
    rho, _ = _correlation(np.ones((2, 4)), y)
    assert np.isnan(rho)


def make_bundle(path):
    folder = path / 'stages/stage4'; folder.mkdir(parents=True)
    x, y = profiles()
    ids = np.array([f's{i}' for i in range(6)])
    pairs = np.array(['p0', 'p0', 'p1', 'p1', 'p2', 'p2'])
    conditions = np.array(['pg19:8k'] * 6)
    sample_x = x[:, None]
    sample_y = np.stack((y, y), axis=1)
    np.savez(folder / 'stage4_joint_profiles.npz', sample_ids=ids, sample_pair_ids=pairs,
             sample_conditions=conditions, x_ids=np.array([CV_ID]), y_ids=np.array(list(TARGETS.values())),
             sample_x=sample_x, sample_y=sample_y, sample_y_valid_source_heads=np.ones_like(sample_y),
             sample_y_source_count_max=np.ones_like(sample_y))
    records = [dict(sample_id=str(ids[i]), pair_id=str(pairs[i]), dataset='pg19', target_prompt_length=8192,
                    decode_length=100) for i in range(6)]
    (path / 'sample_index.json').write_text(json.dumps({'samples': records}))
    (path / 'analysis_manifest.json').write_text(json.dumps({'source_run_set_sha256': 'source'}))
    (folder / 'stage4_manifest.json').write_text(json.dumps({'source_run_set_sha256': 'source'}))


def test_cli_creates_new_result_and_refuses_overwrite(tmp_path):
    bundle = tmp_path / 'bundle'; make_bundle(bundle)
    inputs = load_inputs(bundle)
    assert inputs.x.shape == (6, 2, 4)
    output = tmp_path / 'output'
    argv = ['--bundle', str(bundle), '--output-dir', str(output), '--bootstrap-replicates', '7', '--batch-size', '3']
    main(argv)
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['conditions']['page_d32/pg19:8k']['group_count'] == 3
    assert len((output / 'sample_metrics.jsonl').read_text().splitlines()) == 12
    with np.load(output / 'bootstrap.npz') as a:
        assert a['page_d32__pg19_8k'].shape == (7, 2)
    with pytest.raises(SystemExit):
        main(argv)


def test_index_alignment_is_required(tmp_path):
    bundle = tmp_path / 'bundle'; make_bundle(bundle)
    path = bundle / 'sample_index.json'
    payload = json.loads(path.read_text()); payload['samples'][0]['sample_id'] = 'wrong'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='sample index'):
        load_inputs(bundle)
