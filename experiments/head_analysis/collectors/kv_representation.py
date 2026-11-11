"""V2 Prompt moments, aligned K/V samples and sparse Decode checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from experiments.head_analysis.runtime.attention_probe import AttentionProbeContext, KeyProjectionContext
from kvpress.context import KVPhase

SCOPES = ("prompt", "middle")
NORM_STATISTICS = ("mean", "p10", "median", "p90", "max")


@dataclass(frozen=True)
class KVRepresentationConfig:
    block_size: int = 1024
    decode_steps: tuple[int, ...] = (31, 255)

    def __post_init__(self):
        if self.block_size <= 0 or any(s < 0 for s in self.decode_steps):
            raise ValueError("positive block size and nonnegative Decode steps required")
        if tuple(sorted(set(self.decode_steps))) != self.decode_steps:
            raise ValueError("Decode steps must be sorted and unique")


def sample_positions(prompt_length: int) -> np.ndarray:
    """32 Initial + 448 uniform middle + 32 Recent positions, without duplicates."""
    if prompt_length < 576:
        raise ValueError("512-position sampling requires Prompt length >= 576")
    middle = 64 + ((2 * np.arange(448, dtype=np.int64) + 1) * (prompt_length - 128)) // 896
    return np.concatenate((np.arange(32), middle, np.arange(prompt_length - 32, prompt_length))).astype(np.int64)


class _Moments:
    """CPU FP64 Chan merge over Token blocks; physical Head axes stay separate."""

    def __init__(self, heads, dimension):
        self.count = 0
        self.mean = torch.zeros(heads, dimension, dtype=torch.float64)
        self.m2 = torch.zeros(heads, dimension, dimension, dtype=torch.float64)
        self.minimum = torch.full((heads,), float("inf"), dtype=torch.float64)
        self.maximum = torch.full((heads,), -float("inf"), dtype=torch.float64)

    def add(self, block):
        if block.shape[1] == 0:
            return
        block = block.double()
        n = block.shape[1]
        mean = block.mean(dim=1)
        centered = block - mean[:, None, :]
        delta = mean - self.mean
        total = self.count + n
        self.m2 += centered.transpose(1, 2) @ centered + delta[:, :, None] * delta[:, None, :] * (self.count * n / total)
        self.mean += delta * (n / total)
        self.minimum = torch.minimum(self.minimum, block.amin(dim=(1, 2)))
        self.maximum = torch.maximum(self.maximum, block.amax(dim=(1, 2)))
        self.count = total


def prompt_statistics(tensor: torch.Tensor, block_size=1024) -> dict[str, np.ndarray]:
    """Summarize [Hkv,T,D] K_post or V without copying the entire vector tensor."""
    heads, length, dimension = tensor.shape
    if length <= 128 or block_size <= 0:
        raise ValueError("nonempty middle scope and positive block size required")
    scopes = ((0, length), (64, length - 64))
    moments = [_Moments(heads, dimension) for _ in scopes]
    norms = np.empty((heads, length), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, length, block_size):
            end = min(length, start + block_size)
            block = tensor[:, start:end].detach().to(device="cpu")
            if not torch.isfinite(block).all():
                raise ValueError("nonfinite K/V in Prompt")
            norms[:, start:end] = block.float().norm(dim=-1).numpy()
            for (lo, hi), moment in zip(scopes, moments):
                left, right = max(start, lo), min(end, hi)
                if left < right:
                    moment.add(block[:, left - start : right - start])
    covariance = [m.m2 / m.count for m in moments]
    summaries = []
    for lo, hi in scopes:
        values = norms[:, lo:hi].astype(np.float64)
        summaries.append(np.stack((values.mean(axis=1), *np.quantile(values, [.1, .5, .9], axis=1, method="linear"),
                                   values.max(axis=1)), axis=-1).astype(np.float32))
    return {
        "count": np.asarray([m.count for m in moments], dtype=np.uint32),
        "mean": torch.stack([m.mean for m in moments]).float().numpy(),
        "covariance": torch.stack(covariance).float().numpy(),
        "feature_energy": torch.stack([c.diagonal(dim1=-2, dim2=-1) + m.mean.square()
                                        for c, m in zip(covariance, moments)]).float().numpy(),
        "element_min": torch.stack([m.minimum for m in moments]).float().numpy(),
        "element_max": torch.stack([m.maximum for m in moments]).float().numpy(),
        "norms": norms,
        "norm_summary": np.stack(summaries),
    }


def _cpu_sample(tensor, positions):
    indices = torch.as_tensor(positions, dtype=torch.long, device=tensor.device)
    return tensor.detach().index_select(1, indices).to(device="cpu", copy=True)


@dataclass
class KVRepresentationRecord:
    sample_id: str
    prompt_length: int
    layer_id: int
    forward_id: int
    decode_step: int | None
    arrays: dict[str, np.ndarray | torch.Tensor]
    checks: dict[str, float | bool] = field(default_factory=dict)


class KVRepresentationCollector:
    """Synchronous Probe consumer; emits CPU-only records, retaining only Prefix samples."""

    def __init__(self, prompt_length: int, on_record: Callable[[KVRepresentationRecord], None],
                 config=KVRepresentationConfig()):
        self.prompt_length = prompt_length
        self.positions = sample_positions(prompt_length)
        self.on_record = on_record
        self.config = config
        self._pre = {}
        self._prefix = {}

    def clear(self):
        self._pre.clear()
        self._prefix.clear()

    def on_key_projection(self, context: KeyProjectionContext):
        if context.phase is KVPhase.PREFILL:
            actual = context.token_positions.detach().cpu().numpy()
            if not np.array_equal(actual, np.arange(self.prompt_length)):
                raise ValueError("K_pre projection positions do not cover the Prompt")
            self._pre[context.forward_id, context.layer_id] = (
                context.sample_id, _cpu_sample(context.key_pre[0], self.positions)
            )

    def on_attention(self, context: AttentionProbeContext):
        if context.value is None:
            raise ValueError("V2 collection requires the actual Attention Value tensor")
        if context.prompt_length != self.prompt_length:
            raise ValueError("collector and Probe Prompt lengths differ")
        key, value = context.key[0], context.value[0]
        if context.phase is KVPhase.PREFILL:
            coordinate = (context.forward_id, context.layer_id)
            if coordinate not in self._pre:
                raise ValueError("K_pre is missing for this layer/forward")
            sample_id, key_pre = self._pre.pop(coordinate)
            if sample_id != context.sample_id:
                raise ValueError("K_pre and Attention sample IDs differ")
            key_post, v = (_cpu_sample(t, self.positions) for t in (key, value))
            self._prefix[context.layer_id] = (context.sample_id, key_post, v)
            norm_pre, norm_post = key_pre.float().norm(dim=-1), key_post.float().norm(dim=-1)
            error = (norm_pre - norm_post).abs()
            arrays = {"sample_positions": self.positions.astype(np.uint32),
                      "sample_regions": np.repeat(np.asarray([0, 1, 2], dtype=np.uint8), [32, 448, 32]),
                      "k_pre": key_pre, "k_post": key_post, "v": v}
            for name, tensor in (("k_post", key), ("v", value)):
                arrays.update({f"{name}_{field}": a for field, a in prompt_statistics(tensor, self.config.block_size).items()})
            checks = {"rope_norm_max_abs": float(error.max()),
                      "rope_norm_max_relative": float((error / norm_pre.clamp_min(1e-12)).max())}
        elif context.decode_step in self.config.decode_steps:
            sample_id, prefix_k, prefix_v = self._prefix[context.layer_id]
            if sample_id != context.sample_id:
                raise ValueError("Decode and Prefill sample IDs differ")
            positions = context.query_positions.detach().cpu().numpy()
            if not np.array_equal(positions, [self.prompt_length + context.decode_step]):
                raise ValueError("Decode new Token position differs from T+step")
            arrays = {"token_positions": positions.astype(np.uint32),
                      "k_post": _cpu_sample(key, positions), "v": _cpu_sample(value, positions)}
            checks = {}
            for name, tensor, reference in (("k_post", key, prefix_k), ("v", value, prefix_v)):
                current = _cpu_sample(tensor, self.positions)
                checks[f"{name}_prefix_exact"] = torch.equal(current, reference)
                checks[f"{name}_prefix_max_abs"] = float((current.float() - reference.float()).abs().max())
        else:
            return
        record = KVRepresentationRecord(context.sample_id, self.prompt_length, context.layer_id, context.forward_id,
                                         context.decode_step, arrays, checks)
        self.on_record(record)
        if any(checks.get(f"{name}_prefix_exact") is False for name in ("k_post", "v")):
            raise ValueError("historical Prompt K/V changed during Decode")
