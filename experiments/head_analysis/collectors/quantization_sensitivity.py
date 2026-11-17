"""V3-B full-domain single-Head counterfactuals, retaining only compact CPU metrics."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from experiments.head_analysis.runtime.attention_probe import AttentionProbeContext
from experiments.head_analysis.runtime.kv_quantization import quantize_and_restore, quantized_prompt_range
from kvpress.context import KVPhase


SIDES = ("key", "value")
BIT_WIDTHS = (4, 8)
KL_NEGATIVE_TOLERANCE = 1e-6


def _ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else float("nan")


@dataclass(frozen=True)
class LocalQuantizationRecord:
    sample_id: str
    decode_step: int
    layer_id: int
    key_length: int
    arrays: dict[str, np.ndarray]


class QuantizationSensitivityCollector:
    def __init__(
        self, prompt_length: int, on_record: Callable[[LocalQuantizationRecord], None], block_rows: int = 1024
    ):
        self.prompt_length = prompt_length
        self.start, self.end = quantized_prompt_range(prompt_length)
        self.on_record = on_record
        self.block_rows = block_rows

    def on_attention(self, context: AttentionProbeContext) -> None:
        self.on_record(self.collect(context))

    @torch.inference_mode()
    def collect(self, context: AttentionProbeContext) -> LocalQuantizationRecord:
        if (
            context.phase is not KVPhase.DECODE
            or context.decode_step is None
            or context.query.shape[0] != 1
            or context.query.shape[2] != 1
            or context.attention_mask is not None
            or context.value is None
        ):
            raise ValueError("local quantization requires batch-one, unpadded single-query Decode with Q/K/V")
        if (
            context.prompt_length != self.prompt_length
            or context.key.shape[2] != self.prompt_length + context.decode_step + 1
        ):
            raise ValueError("local quantization requires the complete continuous visible domain")
        kv_heads, query_heads = context.key.shape[1], context.query.shape[1]
        if query_heads % kv_heads or context.key.shape != context.value.shape:
            raise ValueError("invalid GQA or mismatched K/V shape")
        group = query_heads // kv_heads
        shape = (len(SIDES), len(BIT_WIDTHS), kv_heads)
        scalar_fields = (
            "representation_squared_error",
            "representation_fp32_squared_error",
            "representation_energy",
            "representation_nmse",
            "representation_fp32_nmse",
            "output_squared_error",
            "output_energy",
            "output_nmse",
            "attention_kl",
        )
        arrays = {name: np.empty(shape, np.float64) for name in scalar_fields}
        for name in (
            "per_query_output_squared_error",
            "per_query_output_energy",
            "per_query_output_nmse",
            "per_query_attention_kl",
        ):
            arrays[name] = np.empty((*shape, group), np.float64)
        for name in ("code_bytes", "statistic_bytes"):
            arrays[name] = np.empty(shape, np.uint64)
        arrays["attention_kl_is_roundoff"] = np.empty((*shape, group), bool)
        arrays["formula_sdpa_squared_error"] = np.empty((kv_heads, group), np.float64)
        arrays["formula_sdpa_reference_energy"] = np.empty((kv_heads, group), np.float64)
        scale = context.scaling if context.scaling is not None else context.key.shape[-1] ** -0.5
        for head in range(kv_heads):
            q_raw = context.query[:, head * group : (head + 1) * group]
            k_raw, v_raw = context.key[0, head], context.value[0, head]
            q, key, value = q_raw[0, :, 0].float(), k_raw.float(), v_raw.float()
            log_p = (q @ key.T * scale).log_softmax(-1)
            p = log_p.exp()
            reference_output = p @ value
            energy = reference_output.double().square().sum(-1)
            sdpa_output = F.scaled_dot_product_attention(
                q_raw, k_raw[None, None], v_raw[None, None], is_causal=False, enable_gqa=True, scale=scale
            )[0, :, 0]
            arrays["formula_sdpa_squared_error"][head] = (
                (sdpa_output.double() - reference_output.double()).square().sum(-1).cpu().numpy()
            )
            arrays["formula_sdpa_reference_energy"][head] = energy.cpu().numpy()
            for si, side in enumerate(SIDES):
                original = k_raw if side == "key" else v_raw
                for bi, bits in enumerate(BIT_WIDTHS):
                    index = (si, bi, head)
                    restored, encoded, errors = quantize_and_restore(
                        original[self.start : self.end], side, bits, self.block_rows
                    )
                    modified = (key if side == "key" else value).clone()
                    modified[self.start : self.end] = restored.float()
                    if side == "key":
                        log_hat = (q @ modified.T * scale).log_softmax(-1)
                        output = log_hat.exp() @ value
                        kl = (p.double() * (log_p.double() - log_hat.double())).sum(-1)
                        if torch.any(kl < -KL_NEGATIVE_TOLERANCE):
                            raise FloatingPointError("local Attention KL exceeds the negative numerical tolerance")
                    else:
                        output = p @ modified
                        kl = torch.zeros(group, dtype=torch.float64, device=p.device)
                    squared_error = (output.double() - reference_output.double()).square().sum(-1)
                    arrays["representation_squared_error"][index] = errors["squared_error"]
                    arrays["representation_fp32_squared_error"][index] = errors["fp32_squared_error"]
                    arrays["representation_energy"][index] = errors["original_squared_sum"]
                    arrays["representation_nmse"][index] = _ratio(
                        errors["squared_error"], errors["original_squared_sum"]
                    )
                    arrays["representation_fp32_nmse"][index] = _ratio(
                        errors["fp32_squared_error"], errors["original_squared_sum"]
                    )
                    arrays["output_squared_error"][index] = squared_error.sum().item()
                    arrays["output_energy"][index] = energy.sum().item()
                    arrays["output_nmse"][index] = _ratio(squared_error.sum().item(), energy.sum().item())
                    arrays["attention_kl"][index] = kl.mean().item()
                    arrays["per_query_attention_kl"][index] = kl.cpu().numpy()
                    arrays["attention_kl_is_roundoff"][index] = (kl < 0).cpu().numpy()
                    arrays["per_query_output_squared_error"][index] = squared_error.cpu().numpy()
                    arrays["per_query_output_energy"][index] = energy.cpu().numpy()
                    e, n = squared_error.cpu().numpy(), energy.cpu().numpy()
                    arrays["per_query_output_nmse"][index] = np.divide(e, n, out=np.full(group, np.nan), where=n > 0)
                    arrays["code_bytes"][index] = encoded.codes.numel()
                    arrays["statistic_bytes"][index] = encoded.payload_nbytes - encoded.codes.numel()
        for prefix in ("representation", "output", "per_query_output"):
            arrays[f"{prefix}_valid"] = arrays[f"{prefix}_energy"] > 0
        return LocalQuantizationRecord(
            context.sample_id, context.decode_step, context.layer_id, context.key.shape[2], arrays
        )
