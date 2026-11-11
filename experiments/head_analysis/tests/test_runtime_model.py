from types import SimpleNamespace

import pytest
import torch

from experiments.head_analysis.runtime import model as runtime_model


@pytest.mark.parametrize(
    ("dtype", "backend", "expected_dtype"),
    [("bfloat16", "sdpa", torch.bfloat16), ("auto", "flash_attention_2", None)],
)
def test_load_model_preserves_backend_and_dtype_options(monkeypatch, dtype, backend, expected_dtype):
    calls = {}

    class Model:
        def to(self, device):
            calls["device"] = device

        def eval(self):
            calls["eval"] = True

    model = Model()

    def load(path, **kwargs):
        calls.update(path=path, kwargs=kwargs)
        return model

    monkeypatch.setattr(runtime_model.AutoModelForCausalLM, "from_pretrained", load)
    result, device = runtime_model.load_model("snapshot", "cpu", dtype, backend)
    assert result is model and device == calls["device"] == torch.device("cpu")
    assert calls["eval"] and calls["path"] == "snapshot"
    assert calls["kwargs"]["local_files_only"] is True
    assert calls["kwargs"]["attn_implementation"] == backend
    assert calls["kwargs"].get("dtype") == expected_dtype


def test_model_dimensions_and_eos_fallback():
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8),
        generation_config=SimpleNamespace(eos_token_id=[2, 3]),
    )
    tokenizer = SimpleNamespace(eos_token_id=7)
    assert runtime_model.model_dimensions(model) == (32, 32, 8)
    assert runtime_model.eos_token_ids(model, tokenizer) == {2, 3}
    model.generation_config.eos_token_id = None
    assert runtime_model.eos_token_ids(model, tokenizer) == {7}
    model.config.num_key_value_heads = None
    with pytest.raises(ValueError, match="dimensions"):
        runtime_model.model_dimensions(model)


def test_unavailable_cuda_fails_before_loading(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA device requested"):
        runtime_model.load_model("snapshot", "cuda:0", "bfloat16")
