"""Comparable resource measurements for standard B1 greedy generation.

The B1 runners use this scoped helper around the root causal-LM ``forward``
calls made by ``model.generate``.  It deliberately records no model tensors
and does not change generation arguments.  The first root forward is the
complete Prefill; later root forwards are single-token Decode under the B1
generation contract.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


RESOURCE_MEASUREMENT_VERSION = "headkv-resource-measurement-v1"


class ResourceMeasurementError(RuntimeError):
    """Raised when a standard B1 phase-timing contract is not satisfied."""


@dataclass
class _PendingForward:
    """Host/CUDA timing state for one root-model forward invocation."""

    phase: str
    host_started_at: float
    start_event: torch.cuda.Event | None


@dataclass
class _CompletedForward:
    """Completed timing state retained until the sample boundary."""

    phase: str
    host_started_at: float
    host_ended_at: float
    start_event: torch.cuda.Event | None
    end_event: torch.cuda.Event | None


def _quantile_linear(values: list[float], quantile: float) -> float | None:
    """Return a deterministic linear-interpolation quantile without NumPy."""

    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution_summary(values: list[float]) -> dict[str, Any]:
    """Summarize duration samples in seconds, keeping empty cases explicit."""

    if not values:
        return {
            "count": 0,
            "total_seconds": 0.0,
            "mean_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
            "p50_seconds": None,
            "p95_seconds": None,
            "quantile_method": "linear",
        }
    total = sum(values)
    return {
        "count": len(values),
        "total_seconds": total,
        "mean_seconds": total / len(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "p50_seconds": _quantile_linear(values, 0.50),
        "p95_seconds": _quantile_linear(values, 0.95),
        "quantile_method": "linear",
    }


class GenerationPhaseTimer:
    """Read-only phase timer for B1's standard ``model.generate`` path.

    CUDA runs use paired events on the selected device's current stream.  CPU
    runs use host monotonic time so focused tests can validate the record
    structure without a GPU.  Measurements cover root ``model.forward`` only:
    Python generation control, token selection, input construction, and trace
    writing are reported separately by the runner.
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = torch.device(device)
        self._uses_cuda_events = self.device.type == "cuda"
        self._pre_handle: torch.utils.hooks.RemovableHandle | None = None
        self._post_handle: torch.utils.hooks.RemovableHandle | None = None
        self._pending: list[_PendingForward] = []
        self._completed: list[_CompletedForward] = []
        self._started_forward_count = 0
        self._incomplete_forward_count = 0

    def __enter__(self) -> "GenerationPhaseTimer":
        if self._pre_handle is not None or self._post_handle is not None:
            raise ResourceMeasurementError("GenerationPhaseTimer is already attached")
        self._pre_handle = self.model.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        self._post_handle = self.model.register_forward_hook(
            self._post_hook, with_kwargs=True, always_call=True
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._post_handle is not None:
            self._post_handle.remove()
            self._post_handle = None
        if self._pre_handle is not None:
            self._pre_handle.remove()
            self._pre_handle = None
        self._incomplete_forward_count += len(self._pending)
        self._pending.clear()
        return False

    def _pre_hook(self, module: nn.Module, args, kwargs) -> None:
        del module, args, kwargs
        phase = "prefill" if self._started_forward_count == 0 else "decode"
        self._started_forward_count += 1
        start_event: torch.cuda.Event | None = None
        if self._uses_cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(self.device))
        self._pending.append(
            _PendingForward(phase=phase, host_started_at=time.perf_counter(), start_event=start_event)
        )

    def _post_hook(self, module: nn.Module, args, kwargs, output) -> None:
        del module, args, kwargs
        if not self._pending:
            raise ResourceMeasurementError("root-model post-hook has no matching pre-hook")
        pending = self._pending.pop()
        if output is None:
            self._incomplete_forward_count += 1
            return
        end_event: torch.cuda.Event | None = None
        if self._uses_cuda_events:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(torch.cuda.current_stream(self.device))
        self._completed.append(
            _CompletedForward(
                phase=pending.phase,
                host_started_at=pending.host_started_at,
                host_ended_at=time.perf_counter(),
                start_event=pending.start_event,
                end_event=end_event,
            )
        )

    def summary(self) -> dict[str, Any]:
        """Return phase summaries after synchronizing CUDA at a sample boundary."""

        if self._uses_cuda_events:
            torch.cuda.synchronize(self.device)
        durations: dict[str, list[float]] = {"prefill": [], "decode": []}
        for entry in self._completed:
            if self._uses_cuda_events:
                if entry.start_event is None or entry.end_event is None:
                    raise ResourceMeasurementError("CUDA timing event was not recorded")
                seconds = entry.start_event.elapsed_time(entry.end_event) / 1000.0
            else:
                seconds = entry.host_ended_at - entry.host_started_at
            if seconds < 0:
                raise ResourceMeasurementError("forward timing cannot be negative")
            durations[entry.phase].append(seconds)
        return {
            "version": RESOURCE_MEASUREMENT_VERSION,
            "method": "cuda_event" if self._uses_cuda_events else "host_perf_counter",
            "scope": "root_model_forward_only",
            "phase_assignment": "first_forward=prefill; subsequent_forwards=decode",
            "completed_forward_count": len(self._completed),
            "incomplete_forward_count": self._incomplete_forward_count,
            "prefill": _distribution_summary(durations["prefill"]),
            "decode": _distribution_summary(durations["decode"]),
        }


def validate_standard_greedy_timing(summary: dict[str, Any], generated_length: int) -> None:
    """Require B1's one-Prefill plus ``G_out - 1`` Decode-forward mapping."""

    if generated_length < 0:
        raise ResourceMeasurementError("generated_length must be non-negative")
    prefill = summary.get("prefill")
    decode = summary.get("decode")
    if not isinstance(prefill, dict) or not isinstance(decode, dict):
        raise ResourceMeasurementError("timing summary is missing phase records")
    expected_decode_forwards = max(0, generated_length - 1)
    if (
        summary.get("incomplete_forward_count") != 0
        or prefill.get("count") != 1
        or decode.get("count") != expected_decode_forwards
    ):
        raise ResourceMeasurementError(
            "standard greedy timing requires one Prefill forward and "
            f"{expected_decode_forwards} Decode forwards; got "
            f"prefill={prefill.get('count')}, decode={decode.get('count')}, "
            f"incomplete={summary.get('incomplete_forward_count')}"
        )


def measurement_contract() -> dict[str, Any]:
    """Describe the meanings of resource fields embedded in run artifacts."""

    return {
        "version": RESOURCE_MEASUREMENT_VERSION,
        "generation_end_to_end_seconds": (
            "host wall time around model.generate plus the final CUDA synchronization; excludes trace writing"
        ),
        "forward_timing": {
            "scope": "root model.forward only; excludes Python generation control, token selection, and trace writing",
            "cuda_method": "paired CUDA events on the selected device stream, synchronized at the sample boundary",
            "cpu_method": "host perf_counter fallback for CPU regression tests",
            "phase_assignment": "first root forward is Prefill; later root forwards are Decode",
        },
        "trace_write_wall_seconds": "host wall time for B1 sample NPZ/JSON writing and manifest refresh",
        "trace_storage": (
            "sample files only, excluding mutable run-level manifest; logical bytes are uncompressed NumPy array bytes"
        ),
    }
