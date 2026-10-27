"""Scoped, read-only capture at the Llama Attention-function boundary.

This module belongs to ``experiments.head_analysis.runtime``: it is a HeadKV
B1 research probe, not a new generic KV-Base Observer API.  It captures the
Q/K tensors passed to the selected Transformers attention backend together
with lifecycle metadata supplied by an Attention-module pre-hook.  It does not
compute statistics, retain tensors after the callback, or write files.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Generator, Optional

import torch
from torch import nn
from transformers import LlamaForCausalLM, PreTrainedModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from kvpress.context import KVExecutionLifecycle, KVPhase


class AttentionProbeError(RuntimeError):
    """Raised when the B1 probe cannot preserve its capture contract."""


class UnsupportedAttentionBackendError(AttentionProbeError):
    """Raised for an attention backend that B1 has not validated."""


@dataclass(frozen=True)
class AttentionProbeEvent:
    """Tensor-free event metadata retained by :class:`AttentionProbe`."""

    sample_id: str
    phase: KVPhase
    layer_id: int
    forward_id: int
    decode_step: Optional[int]
    backend_name: str
    query_shape: tuple[int, ...]
    key_shape: tuple[int, ...]
    query_position_start: int
    query_position_end: int
    key_length: int
    attention_mask_is_none: bool
    scaling: Optional[float]
    dropout_p: float


@dataclass
class AttentionProbeContext:
    """Ephemeral Q/K capture passed to one synchronous callback invocation.

    ``query`` and ``key`` are references to the tensors supplied to the real
    attention function.  A callback may calculate and move compact results to
    CPU, but must not retain or mutate either tensor.
    """

    sample_id: str
    phase: KVPhase
    layer_id: int
    forward_id: int
    decode_step: Optional[int]
    query: torch.Tensor
    key: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    scaling: Optional[float]
    is_causal: bool
    backend_name: str
    cache_position: torch.Tensor
    query_positions: torch.Tensor
    key_positions: torch.Tensor
    prompt_length: int
    query_dtype: torch.dtype
    key_dtype: torch.dtype
    dropout_p: float


@dataclass
class _ForwardMetadata:
    phase: KVPhase
    layer_id: int
    forward_id: int
    decode_step: Optional[int]
    cache_position: torch.Tensor
    q_len: int


_PROBE_LOCK = threading.RLock()
_ACTIVE_PROBE: "AttentionProbe | None" = None


class AttentionProbe:
    """Capture Llama SDPA inputs without changing model execution.

    The first B1 implementation intentionally accepts only the current
    standard Llama SDPA path.  It avoids promising mask/backend semantics that
    have not been validated yet.  Nested and concurrent probes are rejected
    because Transformers' attention registry is process-global.
    """

    def __init__(
        self,
        *,
        sample_id: str,
        prompt_length: int,
        on_attention: Callable[[AttentionProbeContext], None] | None = None,
    ) -> None:
        if not sample_id:
            raise ValueError("sample_id must be non-empty")
        if prompt_length <= 0:
            raise ValueError("prompt_length must be positive")
        self.sample_id = sample_id
        self.prompt_length = prompt_length
        self.on_attention = on_attention
        self.lifecycle = KVExecutionLifecycle()
        self.events: list[AttentionProbeEvent] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._active_calls: dict[int, _ForwardMetadata] = {}
        self._attached_model: PreTrainedModel | None = None
        self._backend_name: str | None = None
        self._previous_attention_function = None

    @property
    def backend_name(self) -> str | None:
        """Return the validated backend while attached, otherwise ``None``."""

        return self._backend_name

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator["AttentionProbe", None, None]:
        """Attach to ``model`` for one generation scope and always restore it."""

        self.attach(model)
        try:
            yield self
        finally:
            self.detach()

    def attach(self, model: PreTrainedModel) -> None:
        """Install module pre-hooks and one temporary SDPA registry wrapper."""

        global _ACTIVE_PROBE
        if self._attached_model is not None:
            raise AttentionProbeError("this AttentionProbe is already attached")
        if not isinstance(model, LlamaForCausalLM):
            raise AttentionProbeError(f"B1 currently supports LlamaForCausalLM, got {type(model).__name__}")

        backend_name = str(getattr(model.config, "_attn_implementation", ""))
        if backend_name != "sdpa":
            raise UnsupportedAttentionBackendError(
                f"B1 AttentionProbe currently supports only sdpa, got {backend_name!r}"
            )
        if backend_name not in ALL_ATTENTION_FUNCTIONS:
            raise UnsupportedAttentionBackendError(f"attention registry has no {backend_name!r} entry")

        layers = list(model.model.layers)
        if not layers:
            raise AttentionProbeError("Llama model has no decoder layers")
        if any(str(getattr(layer.self_attn.config, "_attn_implementation", "")) != backend_name for layer in layers):
            raise AttentionProbeError("all Llama attention layers must use the same backend")

        with _PROBE_LOCK:
            if _ACTIVE_PROBE is not None:
                raise AttentionProbeError("nested or concurrent AttentionProbe instances are unsupported")
            _ACTIVE_PROBE = self

        self.lifecycle.begin_sample()
        self.events.clear()
        self._active_calls.clear()
        self._attached_model = model
        self._backend_name = backend_name
        self._previous_attention_function = ALL_ATTENTION_FUNCTIONS[backend_name]

        try:
            ALL_ATTENTION_FUNCTIONS[backend_name] = self._attention_wrapper
            for layer in layers:
                self._handles.append(layer.self_attn.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
        except Exception:
            self.detach()
            raise

    def detach(self) -> None:
        """Remove hooks and restore exactly the registry entry seen at attach."""

        global _ACTIVE_PROBE
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

        if self._backend_name is not None and self._previous_attention_function is not None:
            ALL_ATTENTION_FUNCTIONS[self._backend_name] = self._previous_attention_function

        self._active_calls.clear()
        self.lifecycle.end_sample()
        self._attached_model = None
        self._backend_name = None
        self._previous_attention_function = None
        with _PROBE_LOCK:
            if _ACTIVE_PROBE is self:
                _ACTIVE_PROBE = None

    def _pre_hook(self, module: nn.Module, args, kwargs) -> None:
        """Associate pre-function lifecycle metadata with the next layer call."""

        if not hasattr(module, "layer_idx"):
            raise AttentionProbeError("attention module is missing layer_idx")
        layer_id = int(module.layer_idx)
        if layer_id in self._active_calls:
            raise AttentionProbeError(f"layer {layer_id} entered Attention twice before function capture")

        hidden_states = kwargs.get("hidden_states")
        cache_position = kwargs.get("cache_position")
        if not isinstance(hidden_states, torch.Tensor):
            raise AttentionProbeError("attention pre-hook did not receive hidden_states")
        if not isinstance(cache_position, torch.Tensor) or cache_position.ndim != 1 or cache_position.numel() == 0:
            raise AttentionProbeError("B1 requires a non-empty one-dimensional cache_position")

        q_len = int(hidden_states.shape[1])
        if cache_position.numel() != q_len:
            raise AttentionProbeError(
                f"cache_position length {cache_position.numel()} does not match query length {q_len}"
            )
        metadata = self.lifecycle.observe(cache_position, q_len, layer_id)
        phase = metadata["phase"]
        decode_step = metadata["decode_step"]
        first_position = int(cache_position[0].item())
        last_position = int(cache_position[-1].item())

        if phase is KVPhase.PREFILL:
            if q_len != self.prompt_length or first_position != 0 or last_position != self.prompt_length - 1:
                raise AttentionProbeError("B1 requires one complete Prefill matching prompt_length")
        else:
            if q_len != 1 or decode_step is None or first_position != last_position:
                raise AttentionProbeError("B1 requires standard single-token Decode")
            expected_position = self.prompt_length + decode_step
            if first_position != expected_position:
                raise AttentionProbeError(
                    f"decode position {first_position} does not match prompt_length + decode_step {expected_position}"
                )

        self._active_calls[layer_id] = _ForwardMetadata(
            phase=phase,
            layer_id=layer_id,
            forward_id=metadata["forward_id"],
            decode_step=decode_step,
            cache_position=cache_position,
            q_len=q_len,
        )

    def _attention_wrapper(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs,
    ):
        """Capture inputs then invoke the exact entry-time registry callable once."""

        layer_id = int(getattr(module, "layer_idx", -1))
        metadata = self._active_calls.pop(layer_id, None)
        if metadata is None:
            raise AttentionProbeError(f"missing pre-hook metadata for layer {layer_id}")
        self._validate_function_inputs(query, key, metadata)

        key_length = int(key.shape[2])
        key_positions = torch.arange(key_length, device=key.device, dtype=metadata.cache_position.dtype)
        context = AttentionProbeContext(
            sample_id=self.sample_id,
            phase=metadata.phase,
            layer_id=layer_id,
            forward_id=metadata.forward_id,
            decode_step=metadata.decode_step,
            query=query,
            key=key,
            attention_mask=attention_mask,
            scaling=scaling,
            is_causal=bool(kwargs.get("is_causal", getattr(module, "is_causal", True))),
            backend_name=self._backend_name or "<detached>",
            cache_position=metadata.cache_position,
            query_positions=metadata.cache_position,
            key_positions=key_positions,
            prompt_length=self.prompt_length,
            query_dtype=query.dtype,
            key_dtype=key.dtype,
            dropout_p=float(dropout),
        )
        self.events.append(
            AttentionProbeEvent(
                sample_id=self.sample_id,
                phase=metadata.phase,
                layer_id=layer_id,
                forward_id=metadata.forward_id,
                decode_step=metadata.decode_step,
                backend_name=context.backend_name,
                query_shape=tuple(query.shape),
                key_shape=tuple(key.shape),
                query_position_start=int(metadata.cache_position[0].item()),
                query_position_end=int(metadata.cache_position[-1].item()),
                key_length=key_length,
                attention_mask_is_none=attention_mask is None,
                scaling=scaling,
                dropout_p=float(dropout),
            )
        )
        if self.on_attention is not None:
            self.on_attention(context)

        previous = self._previous_attention_function
        if previous is None:
            raise AttentionProbeError("AttentionProbe lost the entry-time attention function")
        return previous(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )

    @staticmethod
    def _validate_function_inputs(query: torch.Tensor, key: torch.Tensor, metadata: _ForwardMetadata) -> None:
        if query.ndim != 4 or key.ndim != 4:
            raise AttentionProbeError("B1 expects query and key tensors shaped [batch, head, sequence, dim]")
        if query.shape[0] != 1 or key.shape[0] != 1:
            raise AttentionProbeError("B1 currently supports batch_size=1")
        if query.shape[2] != metadata.q_len:
            raise AttentionProbeError("Attention function query length differs from its pre-hook metadata")
        if query.shape[-1] != key.shape[-1]:
            raise AttentionProbeError("query and key head dimensions differ")
        expected_key_length = int(metadata.cache_position[-1].item()) + 1
        if key.shape[2] != expected_key_length:
            raise AttentionProbeError(
                f"key length {key.shape[2]} does not match complete continuous cache length {expected_key_length}"
            )
