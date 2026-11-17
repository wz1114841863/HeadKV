"""One-Forward single-Head intervention; the original Cache and Probe stay read-only."""

from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from typing import Iterator

import torch
from transformers import LlamaForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from experiments.head_analysis.runtime.attention_probe import AttentionProbe
from experiments.head_analysis.runtime.kv_quantization import quantize_and_restore, quantized_prompt_range


_BACKEND = "headkv_single_head"


@dataclass(frozen=True)
class HeadIntervention:
    layer_id: int
    head_id: int
    side: str
    bits: int | None  # None is a true bypass, not 16-bit min/max quantization.

    def __post_init__(self):
        if self.layer_id < 0 or self.head_id < 0 or self.side not in ("key", "value") or self.bits not in (None, 4, 8):
            raise ValueError("invalid single-Head intervention")


@dataclass(frozen=True)
class InterventionMeasurement:
    cache_length: int
    quantized_tokens: int
    code_bytes: int
    statistic_bytes: int
    squared_error: float
    fp32_squared_error: float
    original_squared_sum: float

    @property
    def nmse(self) -> float | None:
        return self.squared_error / self.original_squared_sum if self.original_squared_sum else None


@dataclass
class InterventionState:
    measurement: InterventionMeasurement | None = None


@contextmanager
def single_head_intervention(
    model, target: HeadIntervention, prompt_length: int, block_rows: int = 1024
) -> Iterator[InterventionState]:
    """Temporarily dispatch only the target module through a private backend.

    No sdpa registry entry or model Forward is replaced. The target's config
    is shallow-copied for this scope because Llama modules share their config.
    The real Attention receives a modified tensor copy, never a modified Cache.
    """
    if not isinstance(model, LlamaForCausalLM) or model.training:
        raise ValueError("single-Head intervention requires an eval-mode LlamaForCausalLM")
    layers = model.model.layers
    if target.layer_id >= len(layers) or target.head_id >= model.config.num_key_value_heads:
        raise ValueError("Layer or KV Head is outside this model")
    module = layers[target.layer_id].self_attn
    config = module.config
    previous = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if config._attn_implementation != "sdpa" or _BACKEND in ALL_ATTENTION_FUNCTIONS:
        raise ValueError("intervention requires an unoccupied SDPA scope")
    if isinstance(getattr(previous, "__self__", None), AttentionProbe):
        raise ValueError("run intervention branches outside the read-only Probe scope")
    start, end = quantized_prompt_range(prompt_length)
    state = InterventionState()

    def attention(attention_module, query, key, value, attention_mask, **kwargs):
        if state.measurement is not None:
            raise RuntimeError("single-Head scope must contain exactly one Decode Forward")
        if query.shape[0] != 1 or query.shape[2] != 1 or key.shape[2] < prompt_length + 1:
            raise ValueError("single-Head intervention requires batch-one full-history Decode")
        source = key if target.side == "key" else value
        changed = source
        errors = {"squared_error": 0.0, "fp32_squared_error": 0.0, "original_squared_sum": 0.0}
        code_bytes = statistic_bytes = rows = 0
        if target.bits is not None:
            original = source[0, target.head_id, start:end]
            restored, encoded, errors = quantize_and_restore(original, target.side, target.bits, block_rows)
            changed = source.clone()
            changed[0, target.head_id, start:end] = restored
            rows = encoded.rows
            code_bytes = encoded.codes.numel()
            statistic_bytes = encoded.payload_nbytes - code_bytes
        state.measurement = InterventionMeasurement(key.shape[2], rows, code_bytes, statistic_bytes, **errors)
        return previous(
            attention_module,
            query,
            changed if target.side == "key" else key,
            changed if target.side == "value" else value,
            attention_mask,
            **kwargs,
        )

    module.config = copy(config)
    module.config._attn_implementation = _BACKEND
    ALL_ATTENTION_FUNCTIONS[_BACKEND] = attention
    try:
        yield state
        if state.measurement is None:
            raise RuntimeError("target Attention was not called")
    finally:
        module.config = config
        del ALL_ATTENTION_FUNCTIONS[_BACKEND]
