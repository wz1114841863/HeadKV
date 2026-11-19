"""One-Forward single-Head intervention; the original Cache and Probe stay read-only."""

from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, field
from typing import Callable, Iterator

import torch
from transformers import LlamaForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from experiments.head_analysis.runtime.attention_probe import AttentionProbe
from experiments.head_analysis.runtime.kv_quantization import HeadBitPlan, quantize_and_restore, quantized_prompt_range


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
    mode: str = "bypass"
    encoding_performed: bool = False
    reconstruction: "FixedHeadReconstruction | None" = None
    isolation_checks: dict[str, bool] | None = None


@dataclass(frozen=True)
class FixedHeadReconstruction:
    """Owned, read-only tensors tied to the exact input of one quantization.

    Measurement describes that encoding, not work performed by fixed injection.
    """

    target: HeadIntervention
    prompt_length: int
    original: torch.Tensor
    restored: torch.Tensor
    measurement: InterventionMeasurement


@dataclass(frozen=True)
class HeadAttentionInputs:
    """Ephemeral read-only inputs; callbacks must copy any retained tensors."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention_mask: torch.Tensor | None
    scaling: float | None
    is_causal: bool  # Requested flag; SDPA additionally checks query length/mask.
    dropout_p: float
    cache_position: torch.Tensor


@contextmanager
def single_head_intervention(
    model,
    target: HeadIntervention,
    prompt_length: int,
    block_rows: int = 1024,
    *,
    clone_only: bool = False,
    fixed_reconstruction: FixedHeadReconstruction | None = None,
    capture_reconstruction: bool = False,
    check_isolation: bool = False,
    on_inputs: Callable[[HeadAttentionInputs, HeadAttentionInputs], None] | None = None,
) -> Iterator[InterventionState]:
    """Temporarily dispatch only the target module through a private backend.

    No sdpa registry entry or model Forward is replaced. The target's config
    is shallow-copied for this scope because Llama modules share their config.
    The real Attention receives a modified tensor copy, never a modified Cache.
    D1 copies/checks are opt-in; normal quantization and bypass stay unchanged.
    """
    if clone_only and (target.bits is not None or fixed_reconstruction is not None):
        raise ValueError("clone-only requires a bypass target and no fixed reconstruction")
    if fixed_reconstruction is not None and (
        target.bits is None
        or fixed_reconstruction.target != target
        or fixed_reconstruction.prompt_length != prompt_length
    ):
        raise ValueError("fixed reconstruction target/Prompt differs from this intervention")
    if capture_reconstruction and (target.bits is None or fixed_reconstruction is not None):
        raise ValueError("capture reconstruction requires a fresh quantized target")
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
    mode = (
        "fixed"
        if fixed_reconstruction is not None
        else "clone" if clone_only else "quantize" if target.bits else "bypass"
    )
    state = InterventionState(mode=mode)
    cache_position = None

    def capture_position(module, args, kwargs):
        nonlocal cache_position
        cache_position = kwargs.get("cache_position")
        if not isinstance(cache_position, torch.Tensor) or cache_position.numel() != 1:
            raise ValueError("diagnostic intervention requires one explicit Cache position")

    def attention(attention_module, query, key, value, attention_mask, **kwargs):
        if state.measurement is not None:
            raise RuntimeError("single-Head scope must contain exactly one Decode Forward")
        if query.shape[0] != 1 or query.shape[2] != 1 or key.shape[2] < prompt_length + 1:
            raise ValueError("single-Head intervention requires batch-one full-history Decode")
        source = key if target.side == "key" else value
        changed = source
        original = source[0, target.head_id, start:end]
        original_copy = original.detach().clone() if capture_reconstruction or check_isolation else None
        errors = {"squared_error": 0.0, "fp32_squared_error": 0.0, "original_squared_sum": 0.0}
        code_bytes = statistic_bytes = rows = 0
        restored = original
        if fixed_reconstruction is not None:
            frozen = fixed_reconstruction
            if (
                frozen.measurement.cache_length != key.shape[2]
                or frozen.restored.shape != original.shape
                or frozen.restored.dtype != original.dtype
                or frozen.restored.device != original.device
                or frozen.original.dtype != original.dtype
                or frozen.original.device != original.device
                or not torch.isfinite(frozen.restored).all()
                or not torch.equal(frozen.original, original)
            ):
                raise ValueError("fixed reconstruction does not match the checkpoint input/layout")
            restored = frozen.restored
            state.measurement = frozen.measurement
            state.reconstruction = frozen
        elif target.bits is not None:
            restored, encoded, errors = quantize_and_restore(original, target.side, target.bits, block_rows)
            rows = encoded.rows
            code_bytes = encoded.codes.numel()
            statistic_bytes = encoded.payload_nbytes - code_bytes
            state.encoding_performed = True
        if mode != "bypass":
            changed = source.clone()
            changed[0, target.head_id, start:end] = restored
        if state.measurement is None:
            state.measurement = InterventionMeasurement(key.shape[2], rows, code_bytes, statistic_bytes, **errors)
        if capture_reconstruction:
            state.reconstruction = FixedHeadReconstruction(
                target, prompt_length, original_copy, restored, state.measurement
            )
        if check_isolation:
            state.isolation_checks = {
                "source_middle_unchanged": torch.equal(original, original_copy),
                "other_heads_unchanged": torch.equal(source[:, : target.head_id], changed[:, : target.head_id])
                and torch.equal(source[:, target.head_id + 1 :], changed[:, target.head_id + 1 :]),
                "prefix_unchanged": torch.equal(source[:, target.head_id, :start], changed[:, target.head_id, :start]),
                "suffix_and_decode_unchanged": torch.equal(
                    source[:, target.head_id, end:], changed[:, target.head_id, end:]
                ),
                "target_matches_requested_values": torch.equal(changed[0, target.head_id, start:end], restored),
                "complete_position": int(cache_position.item()) == key.shape[2] - 1,
            }
            if not all(state.isolation_checks.values()):
                failed = [k for k, v in state.isolation_checks.items() if not v]
                raise RuntimeError(f"single-Head isolation failed: {failed}")
        actual_key = changed if target.side == "key" else key
        actual_value = changed if target.side == "value" else value
        if on_inputs is not None:
            common = dict(
                query=query,
                attention_mask=attention_mask,
                scaling=kwargs.get("scaling"),
                is_causal=bool(kwargs.get("is_causal", getattr(attention_module, "is_causal", True))),
                dropout_p=float(kwargs.get("dropout", 0.0)),
                cache_position=cache_position,
            )
            on_inputs(
                HeadAttentionInputs(key=key, value=value, **common),
                HeadAttentionInputs(key=actual_key, value=actual_value, **common),
            )
        return previous(
            attention_module,
            query,
            actual_key,
            actual_value,
            attention_mask,
            **kwargs,
        )

    module.config = copy(config)
    module.config._attn_implementation = _BACKEND
    ALL_ATTENTION_FUNCTIONS[_BACKEND] = attention
    handle = None
    try:
        if check_isolation or on_inputs is not None:
            handle = module.register_forward_pre_hook(capture_position, with_kwargs=True)
        yield state
        if state.measurement is None:
            raise RuntimeError("target Attention was not called")
    finally:
        if handle is not None:
            handle.remove()
        module.config = config
        del ALL_ATTENTION_FUNCTIONS[_BACKEND]


@dataclass
class PlanInterventionState:
    measurements: dict[tuple[int, int], InterventionMeasurement] = field(default_factory=dict)
    code_bytes: int = 0
    statistic_bytes: int = 0
    protected_residual_decode_bytes: int = 0
    dense_other_side_bytes: int = 0


@contextmanager
def multiple_head_intervention(model, plan: HeadBitPlan, prompt_length: int, block_rows: int = 1024):
    """One Dense checkpoint with a whole-model K or V plan; never modify Cache.

    The existing isolated Forward owns Cache append/restore. This scope owns
    only private Attention dispatch, actual encoding and target-side copies.
    """
    backend = "headkv_head_budget"
    layers = model.model.layers
    if (
        not isinstance(model, LlamaForCausalLM)
        or model.training
        or len(plan.bits) != len(layers)
        or len(plan.bits[0]) != model.config.num_key_value_heads
        or any(layer.self_attn.head_dim != plan.head_dim for layer in layers)
    ):
        raise ValueError("Head plan differs from the eval Llama geometry")
    previous = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if (
        backend in ALL_ATTENTION_FUNCTIONS
        or _BACKEND in ALL_ATTENTION_FUNCTIONS
        or isinstance(getattr(previous, "__self__", None), AttentionProbe)
        or any(layer.self_attn.config._attn_implementation != "sdpa" for layer in layers)
    ):
        raise ValueError("Head plan requires an unoccupied SDPA scope")
    start, end = quantized_prompt_range(prompt_length)
    state = PlanInterventionState()

    def attention(module, query, key, value, attention_mask, **kwargs):
        source, other = (key, value) if plan.side == "key" else (value, key)
        if source.dtype != torch.bfloat16 or source.shape[0] != 1 or query.shape[2] != 1:
            raise ValueError("Head budget replay requires BF16 batch-one single-token Decode")
        changed = source.clone()
        layer = module.layer_idx
        for head, bits in enumerate(plan.bits[layer]):
            if (layer, head) in state.measurements:
                raise RuntimeError("Head plan scope must contain exactly one Forward")
            restored, encoded, errors = quantize_and_restore(source[0, head, start:end], plan.side, bits, block_rows)
            changed[0, head, start:end] = restored
            state.measurements[layer, head] = InterventionMeasurement(
                source.shape[2],
                encoded.rows,
                encoded.codes.numel(),
                encoded.payload_nbytes - encoded.codes.numel(),
                **errors,
            )
            state.code_bytes += encoded.codes.numel()
            state.statistic_bytes += encoded.payload_nbytes - encoded.codes.numel()
        if not torch.equal(source[:, :, :start], changed[:, :, :start]) or not torch.equal(
            source[:, :, end:], changed[:, :, end:]
        ):
            raise RuntimeError("Head plan changed protected/residual/Decode rows")
        state.protected_residual_decode_bytes += (
            source[:, :, :start].numel() + source[:, :, end:].numel()
        ) * source.element_size()
        state.dense_other_side_bytes += other.numel() * other.element_size()
        return previous(
            module,
            query,
            changed if plan.side == "key" else key,
            changed if plan.side == "value" else value,
            attention_mask,
            **kwargs,
        )

    configs = [(layer.self_attn, layer.self_attn.config) for layer in layers]
    ALL_ATTENTION_FUNCTIONS[backend] = attention
    try:
        for module, config in configs:
            module.config = copy(config)
            module.config._attn_implementation = backend
        yield state
        if len(state.measurements) != len(plan.bits) * len(plan.bits[0]):
            raise RuntimeError("Head plan did not visit all layers/Heads")
    finally:
        for module, config in configs:
            module.config = config
        del ALL_ATTENTION_FUNCTIONS[backend]
