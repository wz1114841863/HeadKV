from functools import partial
import json

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.head_analysis.collectors.kv_representation import (
    KVRepresentationCollector, KVRepresentationConfig, prompt_statistics, sample_positions,
)
from experiments.head_analysis.runtime.attention_probe import AttentionProbe
from experiments.head_analysis.storage.kv_representation import KVRepresentationWriter, read_record


def test_fixed_sampling_is_exact_unique_and_independent_of_tensor_values():
    for length in (576, 640, 8192, 32768):
        positions = sample_positions(length)
        assert len(positions) == len(np.unique(positions)) == 512
        np.testing.assert_array_equal(positions[:32], np.arange(32))
        np.testing.assert_array_equal(positions[-32:], np.arange(length - 32, length))
        assert (positions[32:480] >= 64).all() and (positions[32:480] < length - 64).all()
        reference = [64 + int(np.floor((j + 0.5) * (length - 128) / 448)) for j in range(448)]
        np.testing.assert_array_equal(positions[32:480], reference)
    with pytest.raises(ValueError, match='>= 576'):
        sample_positions(575)


def assert_statistics(observed, tensor):
    x = tensor.double().numpy()
    norms = tensor.float().norm(dim=-1).numpy()
    np.testing.assert_array_equal(observed['norms'], norms)
    for scope, (lo, hi) in enumerate(((0, x.shape[1]), (64, x.shape[1] - 64))):
        selected = x[:, lo:hi]
        mean = selected.mean(1)
        centered = selected - mean[:, None]
        covariance = centered.transpose(0, 2, 1) @ centered / (hi - lo)
        assert observed['count'][scope] == hi - lo
        np.testing.assert_allclose(observed['mean'][scope], mean.astype(np.float32), rtol=2e-6, atol=1e-8)
        np.testing.assert_allclose(observed['covariance'][scope], covariance.astype(np.float32), rtol=2e-6, atol=1e-8)
        np.testing.assert_allclose(observed['feature_energy'][scope], np.mean(selected ** 2, axis=1).astype(np.float32),
                                   rtol=2e-6, atol=1e-8)
        np.testing.assert_array_equal(observed['element_min'][scope], selected.min((1, 2)).astype(np.float32))
        np.testing.assert_array_equal(observed['element_max'][scope], selected.max((1, 2)).astype(np.float32))
        n = norms[:, lo:hi].astype(float)
        summary = np.stack((n.mean(1), *np.quantile(n, [.1, .5, .9], axis=1), n.max(1)), axis=-1).astype(np.float32)
        np.testing.assert_array_equal(observed['norm_summary'][scope], summary)
        assert observed['covariance'].dtype == np.float32


@pytest.mark.parametrize('block_size', [1, 37, 2048])
def test_chan_statistics_match_centered_fp64_with_large_mean(block_size):
    torch.manual_seed(5)
    tensor = (10000 + torch.randn(2, 640, 8) * .02).float()
    assert_statistics(prompt_statistics(tensor, block_size), tensor)
    constant = torch.full((1, 150, 3), 12.)
    result = prompt_statistics(constant, block_size)
    assert np.count_nonzero(result['covariance']) == 0
    np.testing.assert_array_equal(result['feature_energy'], 144.)


def make_model(dtype=torch.float32):
    torch.manual_seed(37)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                         eos_token_id=None, pad_token_id=None)
    config._attn_implementation = 'sdpa'
    return LlamaForCausalLM(config).eval().to(dtype=dtype)


def run_model(model, prompt, steps):
    snapshots, logits = {}, []
    cache = None
    with torch.inference_mode():
        for step in range(-1, steps):
            ids = prompt if step < 0 else torch.tensor([[(step % 60) + 1]])
            positions = torch.arange(prompt.shape[1]) if step < 0 else torch.tensor([prompt.shape[1] + step])
            result = model(input_ids=ids, cache_position=positions, past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            logits.append(result.logits[:, -1].detach().cpu().clone())
            if step in (-1, 31, 255):
                snapshots[step] = [(layer.keys[0].detach().cpu().clone(), layer.values[0].detach().cpu().clone())
                                   for layer in cache.layers]
    return logits, snapshots


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_small_model_collection_and_roundtrip_preserve_statistics_positions_and_dtype(tmp_path, dtype):
    model = make_model(dtype)
    prompt = torch.tensor([[(i % 60) + 1 for i in range(640)]])
    original_pre, handles = {}, []
    def raw_projection(layer, module, args, output):
        if output.shape[1] == 640:
            original_pre[layer] = output.view(1, 640, 2, 8).transpose(1, 2)[0].detach().cpu().clone()
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(partial(raw_projection, i)))
    try:
        baseline_logits, baseline = run_model(model, prompt, 256)
    finally:
        for handle in handles:
            handle.remove()
    config = KVRepresentationConfig(block_size=37)
    directory = tmp_path / 'sample'
    writer = KVRepresentationWriter(directory, sample_id='tiny', prompt_length=640, num_layers=2, config=config)
    records = []
    def emit(record):
        records.append(record)
        writer.write_record(record)
    collector = KVRepresentationCollector(640, emit, config)
    with AttentionProbe(sample_id='tiny', prompt_length=640, on_key_projection=collector.on_key_projection,
                        on_attention=collector.on_attention)(model):
        logits, _ = run_model(model, prompt, 256)
    assert len(records) == 6
    assert not collector._pre
    assert all(t.device.type == 'cpu' for prefix in collector._prefix.values() for t in prefix[1:])
    for observed, expected in zip(logits, baseline_logits):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    writer.finalize()
    metadata = json.loads((directory / 'metadata.json').read_text())
    assert metadata['status'] == 'completed'
    assert set(metadata['files']) == {
        'prefill/layer_000.npz', 'prefill/layer_001.npz', 'decode/step_031/layer_000.npz',
        'decode/step_031/layer_001.npz', 'decode/step_255/layer_000.npz', 'decode/step_255/layer_001.npz',
    }
    positions = sample_positions(640)
    for layer in range(2):
        saved = read_record(directory, f'prefill/layer_{layer:03d}.npz')
        np.testing.assert_array_equal(saved['sample_positions'], positions)
        np.testing.assert_array_equal(saved['sample_regions'], [0] * 32 + [1] * 448 + [2] * 32)
        checks = metadata['files'][f'prefill/layer_{layer:03d}.npz']['checks']
        norm_pre = saved['k_pre'].float().norm(dim=-1)
        norm_post = saved['k_post'].float().norm(dim=-1)
        expected_error = float(((norm_pre - norm_post).abs() / norm_pre.clamp_min(1e-12)).max())
        assert checks['rope_norm_max_relative'] == expected_error
        assert expected_error < (1e-2 if dtype is torch.bfloat16 else 1e-6)
        for name, reference in (('k_pre', original_pre[layer]), ('k_post', baseline[-1][layer][0]), ('v', baseline[-1][layer][1])):
            assert saved[name].shape == (2, 512, 8) and saved[name].dtype == dtype
            torch.testing.assert_close(saved[name], reference[:, positions], rtol=0, atol=0)
            if dtype is torch.bfloat16:
                np.testing.assert_array_equal(saved[name].view(torch.uint16).numpy(),
                                              reference[:, positions].contiguous().view(torch.uint16).numpy())
            if name != 'k_pre':
                stats = {k[len(name) + 1:]: a for k, a in saved.items() if k.startswith(name + '_')}
                assert_statistics(stats, reference)
        with np.load(directory / f'prefill/layer_{layer:03d}.npz') as archive:
            assert archive['k_pre'].nbytes == 2 * 512 * 8 * (2 if dtype is torch.bfloat16 else 4)
        for step in (31, 255):
            relative = f'decode/step_{step:03d}/layer_{layer:03d}.npz'
            saved = read_record(directory, relative)
            assert set(saved) == {'token_positions', 'k_post', 'v'}
            np.testing.assert_array_equal(saved['token_positions'], [640 + step])
            for name, reference in zip(('k_post', 'v'), baseline[step][layer]):
                assert saved[name].shape == (2, 1, 8)
                torch.testing.assert_close(saved[name], reference[:, 640 + step:641 + step], rtol=0, atol=0)
                assert metadata['files'][relative]['checks'][name + '_prefix_exact']
    collector.clear()
    assert not collector._pre and not collector._prefix
    with pytest.raises(FileExistsError):
        KVRepresentationWriter(directory, sample_id='tiny', prompt_length=640, num_layers=2)


def test_prefix_change_is_reported_and_cannot_publish_completed_sample(tmp_path):
    model = make_model()
    config = KVRepresentationConfig(block_size=128, decode_steps=(0,))
    writer = KVRepresentationWriter(tmp_path / 'sample', sample_id='tiny', prompt_length=640, num_layers=2, config=config)
    collector = KVRepresentationCollector(640, writer.write_record, config)
    with torch.inference_mode(), AttentionProbe(sample_id='tiny', prompt_length=640,
            on_key_projection=collector.on_key_projection, on_attention=collector.on_attention)(model):
        result = model(input_ids=torch.ones((1, 640), dtype=torch.long), cache_position=torch.arange(640), use_cache=True)
        result.past_key_values.layers[0].keys[:, :, 0] += 1
        with pytest.raises(ValueError, match='historical Prompt'):
            model(input_ids=torch.tensor([[2]]), cache_position=torch.tensor([640]),
                  past_key_values=result.past_key_values, use_cache=True)
    with pytest.raises(ValueError):
        writer.finalize()
    metadata = json.loads((tmp_path / 'sample/metadata.json').read_text())
    assert metadata['status'] == 'collecting'
    assert not metadata['files']['decode/step_000/layer_000.npz']['checks']['k_post_prefix_exact']


def test_short_decode_uses_only_its_requested_checkpoint(tmp_path):
    model = make_model()
    config = KVRepresentationConfig(block_size=128, decode_steps=(5,))
    writer = KVRepresentationWriter(tmp_path / 'sample', sample_id='tiny', prompt_length=640, num_layers=2, config=config)
    collector = KVRepresentationCollector(640, writer.write_record, config)
    with AttentionProbe(sample_id='tiny', prompt_length=640, on_key_projection=collector.on_key_projection,
                        on_attention=collector.on_attention)(model):
        run_model(model, torch.ones((1, 640), dtype=torch.long), 6)
    writer.finalize()
    saved = read_record(tmp_path / 'sample', 'decode/step_005/layer_000.npz')
    np.testing.assert_array_equal(saved['token_positions'], [645])
    assert saved['k_post'].shape == (2, 1, 8)


def test_incomplete_layers_do_not_publish_completed_sample(tmp_path):
    writer = KVRepresentationWriter(tmp_path / 'sample', sample_id='tiny', prompt_length=640, num_layers=2,
                                    config=KVRepresentationConfig(decode_steps=()))
    with pytest.raises(ValueError, match='missing Prefill'):
        writer.finalize()
    assert json.loads((tmp_path / 'sample/metadata.json').read_text())['status'] == 'collecting'
