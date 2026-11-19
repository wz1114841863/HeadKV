"""Single-token, read-only propagation and arithmetic references for D2."""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F


PROJECTION_LABELS = ("B0", "Q4", "Q8")


def tensor_error(reference, current):
    """FP64 error accumulation; a zero denominator is explicitly undefined."""
    a, b = reference.double(), current.double()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("propagation tensors must have matching shapes and finite values")
    difference = b - a
    squared_error, energy = difference.square().sum().item(), a.square().sum().item()
    return {
        "squared_error": squared_error,
        "reference_energy": energy,
        "nmse": squared_error / energy if energy > 0 else None,
        "max_absolute_difference": difference.abs().max().item() if difference.numel() else 0.0,
        "equal": torch.equal(a, b),
    }


def _head_error(reference, current):
    return tensor_error(reference, current) | {
        "per_query_head": [tensor_error(a, b) for a, b in zip(reference, current)]
    }


def prediction_boundary(logits):
    """Include exact ties; argmax alone is insufficient near a BF16 boundary."""
    x = logits.detach().cpu().double()
    first = int(x.argmax())
    remaining = x.clone()
    remaining[first] = -torch.inf
    second = int(remaining.argmax())
    return {
        "predicted_token_id": first,
        "runner_up_token_id": second,
        "top1_logit": x[first].item(),
        "runner_up_logit": x[second].item(),
        "margin": (x[first] - x[second]).item(),
        "top1_tie_count": int((x == x[first]).sum()),
    }


def prediction_errors_fp64(reference, branch, expected_token):
    """Same saved logits, CPU FP64 probabilities; never clip a negative KL."""
    a, b = reference.detach().cpu().double(), branch.detach().cpu().double()
    if a.ndim != 1 or a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("FP64 prediction comparison requires finite vocabulary vectors")
    log_a, log_b = a.log_softmax(-1), b.log_softmax(-1)
    kl = (log_a.exp() * (log_a - log_b)).sum().item()
    if kl < -1e-12:
        raise FloatingPointError(f"FP64 prediction KL below numerical tolerance: {kl}")
    return {
        "prediction_kl": kl,
        "kl_is_roundoff": kl < 0,
        "delta_nll": (log_a[expected_token] - log_b[expected_token]).item(),
        "predicted_token_id": int(b.argmax()),
        "top1_match": int(b.argmax()) == int(a.argmax()),
    }


@torch.inference_mode()
def fp32_final_projection(lm_head, hidden, block_rows=4096):
    """Recompute from actual hidden/weight values, not from rounded logits.

    Only one vocabulary block of FP32 weights is resident. This auxiliary call
    disables autocast/TF32 and restores the caller's settings even on failure.
    """
    if block_rows <= 0 or hidden.ndim != 2 or hidden.shape[1] != lm_head.weight.shape[1]:
        raise ValueError("FP32 projection requires [rows, hidden] and a positive vocabulary block size")
    previous = {
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    chunks = []
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        auxiliary = {
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
        with torch.autocast(lm_head.weight.device.type, enabled=False):
            x = hidden.to(device=lm_head.weight.device, dtype=torch.float32)
            for start in range(0, lm_head.weight.shape[0], block_rows):
                weight = lm_head.weight[start : start + block_rows].float()
                bias = None if lm_head.bias is None else lm_head.bias[start : start + block_rows].float()
                chunks.append(F.linear(x, weight, bias).cpu())
    finally:
        torch.set_float32_matmul_precision(previous["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = previous["allow_tf32"]
    return torch.cat(chunks, dim=-1), {
        "before": previous,
        "auxiliary": auxiliary,
        "after": {
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "weight_dtype": str(lm_head.weight.dtype),
        "block_rows": block_rows,
        "scope": "FP32 projection of captured native hidden values and existing weights; not full-model FP32",
    }


@dataclass(frozen=True)
class PropagationResult:
    projection: dict
    arrays: dict[str, np.ndarray]


class QuantizationPropagationCollector:
    """Attach only around checkpoint forwards; retain no complete Cache or Attention.

    The real Attention output is the input of the target o_proj. All labels use
    the same hooks. Only B0/Q4/Q8 temporarily own the selected Head's actual Q/K/V
    for post-Forward formula references, and retain final hidden/output vectors.
    """

    def __init__(self, target, query_heads, kv_heads):
        self.target = target
        self.query_heads = query_heads
        self.group = query_heads // kv_heads
        self.reference = {}
        self.selected = {}
        self.reference_formulas = None
        self.reference_logits = None

    def _capture(self, name, value):
        if value.ndim != 3 or value.shape[:2] != (1, 1):
            raise ValueError("D2 hooks require a batch-one single-token checkpoint")
        if name in self.current:
            raise RuntimeError(f"D2 stage captured twice: {name}")
        self.current[name] = value.detach().to(device="cpu", dtype=torch.float32, copy=True).flatten()
        self.dtypes[name] = str(value.dtype)

    def _output_hook(self, module, args, output, *, name):
        self._capture(name, output)

    @contextmanager
    def capture(self, model, label):
        self.label, self.current, self.dtypes, self.inputs = label, {}, {}, None
        self.stage_names = [
            "attention_output",
            "o_proj_output",
            *(f"layer_{layer:02d}_output" for layer in range(self.target.layer_id, len(model.model.layers))),
            "final_norm",
            "lm_head_input",
            "native_logits",
        ]
        handles = []
        projection = model.model.layers[self.target.layer_id].self_attn.o_proj
        try:
            handles.append(
                projection.register_forward_pre_hook(lambda m, args: self._capture("attention_output", args[0]))
            )
            handles.append(projection.register_forward_hook(partial(self._output_hook, name="o_proj_output")))
            for layer in range(self.target.layer_id, len(model.model.layers)):
                handles.append(
                    model.model.layers[layer].register_forward_hook(
                        partial(self._output_hook, name=f"layer_{layer:02d}_output")
                    )
                )
            handles.append(model.model.norm.register_forward_hook(partial(self._output_hook, name="final_norm")))
            handles.append(
                model.lm_head.register_forward_pre_hook(lambda m, args: self._capture("lm_head_input", args[0]))
            )
            handles.append(model.lm_head.register_forward_hook(partial(self._output_hook, name="native_logits")))
            yield
        except Exception:
            self.inputs = None
            raise
        finally:
            for handle in handles:
                handle.remove()

    def capture_inputs(self, inputs):
        if self.label not in PROJECTION_LABELS:
            return
        if inputs.attention_mask is not None or inputs.dropout_p != 0 or inputs.query.shape[:1] != (1,):
            raise ValueError("D2 formula reference requires unpadded, zero-dropout batch-one Decode")
        head, group = self.target.head_id, self.group
        self.inputs = (
            inputs.query[0, head * group : (head + 1) * group, 0].detach().clone(),
            inputs.key[0, head].detach().clone(),
            inputs.value[0, head].detach().clone(),
            inputs.scaling if inputs.scaling is not None else inputs.key.shape[-1] ** -0.5,
        )

    def record(self, logits, expected_token):
        if not self.reference:
            if self.label != "B0":
                raise ValueError("D2 must start with B0")
            self.reference = self.current.copy()
        if list(self.current) != self.stage_names:
            raise RuntimeError("D2 checkpoint stages are incomplete")
        native = logits.detach().to(device="cpu", dtype=torch.float32, copy=True)
        if not torch.equal(native, self.current["native_logits"]):
            raise RuntimeError("D2 lm_head hook differs from returned logits")
        if self.label == "B0":
            self.reference_logits = native
        stages = {
            name: tensor_error(self.reference[name], value) | {"dtype": self.dtypes[name]}
            for name, value in self.current.items()
        }
        attention = self.current["attention_output"].reshape(self.query_heads, -1)
        baseline = self.reference["attention_output"].reshape_as(attention)
        start, end = self.target.head_id * self.group, (self.target.head_id + 1) * self.group
        actual = attention[start:end]
        result = {
            "stages": stages,
            "first_different_stage": next((name for name, metric in stages.items() if not metric["equal"]), None),
            "attention": {
                "target_query_head_ids": list(range(start, end)),
                "target_vs_baseline": _head_error(baseline[start:end], actual),
                "other_query_heads_vs_baseline": _head_error(
                    torch.cat((baseline[:start], baseline[end:])), torch.cat((attention[:start], attention[end:]))
                ),
            },
            "native_prediction_fp64": prediction_errors_fp64(self.reference_logits, native, expected_token),
            "native_boundary": prediction_boundary(native),
        }
        if self.label in PROJECTION_LABELS:
            if self.inputs is None:
                raise RuntimeError("D2 is missing the actual Attention inputs")
            q, key, value, scale = self.inputs
            # Preserve the old FP32 formula and device; do not call SDPA again.
            log32 = (q.float() @ key.float().T * scale).log_softmax(-1)
            out32 = (log32.exp() @ value.float()).cpu()
            q64, k64, v64 = (x.cpu().double() for x in (q, key, value))
            log64 = (q64 @ k64.T * scale).log_softmax(-1)
            out64 = log64.exp() @ v64
            if self.label == "B0":
                self.reference_formulas = (out32, out64, log64)
            ref32, ref64, ref_log64 = self.reference_formulas
            kl64 = (ref_log64.exp() * (ref_log64 - log64)).sum(-1)
            if (kl64 < -1e-12).any():
                raise FloatingPointError("D2 FP64 Attention KL below numerical tolerance")
            result["attention"]["formulas"] = {
                "fp32_device": str(q.device),
                "actual_vs_fp32": _head_error(out32, actual),
                "actual_vs_cpu_fp64": _head_error(out64, actual),
                "fp32_vs_cpu_fp64": _head_error(out64, out32),
                "fp32_vs_baseline": _head_error(ref32, out32),
                "cpu_fp64_vs_baseline": _head_error(ref64, out64),
                "attention_kl_fp64_per_query": kl64.tolist(),
                "attention_kl_fp64_is_roundoff": (kl64 < 0).tolist(),
            }
            self.selected[self.label] = {
                "hidden": self.current["lm_head_input"],
                "hidden_dtype": self.dtypes["lm_head_input"],
                "native_logits": native,
                "attention": attention,
                "formula32": out32,
                "formula64": out64,
            }
            self.inputs = None  # Release selected full-history Q/K/V after this reference.
        return result

    def finish(self, lm_head, expected_token, block_rows=4096):
        if tuple(self.selected) != PROJECTION_LABELS:
            raise RuntimeError("D2 requires B0/Q4/Q8 hidden vectors")
        rows = [self.selected[label] for label in PROJECTION_LABELS]
        hidden = torch.stack([row["hidden"] for row in rows])
        projected, settings = fp32_final_projection(lm_head, hidden, block_rows)
        records = []
        for label, row, vector in zip(PROJECTION_LABELS, rows, projected):
            records.append(
                {
                    "label": label,
                    **prediction_errors_fp64(projected[0], vector, expected_token),
                    "boundary": prediction_boundary(vector),
                    "native_top1_matches_projection": int(row["native_logits"].argmax()) == int(vector.argmax()),
                    "native_vs_projected_logits": tensor_error(row["native_logits"], vector),
                }
            )
        return PropagationResult(
            projection={"settings": settings, "records": records},
            arrays={
                "labels": np.asarray(PROJECTION_LABELS),
                "final_hidden": hidden.numpy(),
                "hidden_native_dtype": np.asarray([row["hidden_dtype"] for row in rows]),
                "fp32_projected_logits": projected.numpy(),
                "actual_attention_output": torch.stack([row["attention"] for row in rows]).numpy(),
                "formula_fp32_output": torch.stack([row["formula32"] for row in rows]).numpy(),
                "formula_cpu_fp64_output": torch.stack([row["formula64"] for row in rows]).numpy(),
            },
        )
