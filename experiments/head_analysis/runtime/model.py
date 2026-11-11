"""Shared model loading and configuration helpers for Head Analysis."""

import torch
from transformers import AutoModelForCausalLM


def resolve_dtype(name: str) -> torch.dtype | None:
    if name == "auto":
        return None
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_model(model_path: str, device_name: str, dtype_name: str, attn_implementation: str = "sdpa"):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested ({device_name}) but torch.cuda.is_available() is false")
    kwargs = {"local_files_only": True, "attn_implementation": attn_implementation}
    dtype = resolve_dtype(dtype_name)
    if dtype is not None:
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.to(device)
    model.eval()
    return model, device


def model_dimensions(model) -> tuple[int, int, int]:
    config = model.config
    try:
        return int(config.num_hidden_layers), int(config.num_attention_heads), int(config.num_key_value_heads)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("model config lacks valid Llama layer/query/KV-head dimensions") from exc


def eos_token_ids(model, tokenizer) -> set[int]:
    values = getattr(model.generation_config, "eos_token_id", None)
    if values is None:
        values = getattr(tokenizer, "eos_token_id", None)
    if isinstance(values, int):
        return {values}
    if isinstance(values, (tuple, list)):
        return {value for value in values if isinstance(value, int)}
    return set()
