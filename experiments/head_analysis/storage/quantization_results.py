"""Compact per-checkpoint V3-B metrics; no Attention, vocabulary, or Cache tensors."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.head_analysis.collectors.quantization_sensitivity import BIT_WIDTHS, SIDES
from experiments.head_analysis.common.io import read_json, write_json, write_npz
from experiments.head_analysis.runtime.kv_quantization import quantization_byte_ledger


def _describe(values):
    values = np.asarray(values)
    valid = np.isfinite(values)
    if not valid.any():
        return {"valid_count": 0, "mean": None, "p90": None, "worst": None}
    worst = np.unravel_index(np.nanargmax(values), values.shape)
    return {
        "valid_count": int(valid.sum()),
        "mean": float(values[valid].mean()),
        "p90": float(np.quantile(values[valid], 0.9)),
        "worst": {"layer_id": int(worst[0]), "head_id": int(worst[1]), "value": float(values[worst])},
    }


class QuantizationResultWriter:
    def __init__(
        self,
        directory,
        *,
        sample_id,
        prompt_length,
        num_layers,
        query_heads,
        kv_heads,
        head_dim,
        local_steps,
        targets,
        source,
        execution_dtype,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.sample_id, self.prompt_length = sample_id, prompt_length
        self.num_layers, self.kv_heads, self.head_dim = num_layers, kv_heads, head_dim
        self.query_group = query_heads // kv_heads
        self.local_steps, self.targets = tuple(local_steps), tuple(targets)
        self.primary_step = self.local_steps[0]
        self.pending, self.local_files, self.checkpoints = {}, {}, {}
        self.model_written = False
        self.summary = {"local": {}}
        self.metadata = {
            "schema_version": "headkv-v3b-quantization-v1",
            "status": "collecting",
            "sample_id": sample_id,
            "prompt_length": prompt_length,
            "source": source,
            "execution_dtype": execution_dtype,
            "budget_storage_dtype": "bfloat16",
            "num_layers": num_layers,
            "num_query_heads": query_heads,
            "num_kv_heads": kv_heads,
            "head_dim": head_dim,
            "local_steps": list(local_steps),
            "primary_step": self.primary_step,
            "sides": list(SIDES),
            "bit_widths": list(BIT_WIDTHS),
            "local_axes": ["side", "bits", "layer", "kv_head"],
            "per_query_axes": ["side", "bits", "layer", "kv_head", "query_in_group"],
            "formula_sdpa_axes": ["layer", "kv_head", "query_in_group"],
            "local_arithmetic": "FP32 QK/log_softmax/AV on actual restored inputs; FP64 error sums",
            "invalid_ratios": "NaN in NPZ with explicit valid mask; JSON summaries use null",
            "negative_kl_tolerance": 1e-6,
            "expected_model_targets": [asdict(target) for target in targets],
        }
        write_json(self.directory / "metadata.json", self.metadata)

    def write_local(self, record):
        if (
            record.sample_id != self.sample_id
            or record.decode_step not in self.local_steps
            or not 0 <= record.layer_id < self.num_layers
            or record.key_length != self.prompt_length + record.decode_step + 1
        ):
            raise ValueError("unexpected local sample, layer, step or visible length")
        step = record.decode_step
        if step in self.local_files or record.layer_id in self.pending.get(step, {}):
            raise ValueError("duplicate local layer/step")
        for name, array in record.arrays.items():
            expected_shape = (
                (self.kv_heads, self.query_group)
                if name.startswith("formula_sdpa_")
                else (
                    (2, 2, self.kv_heads, self.query_group)
                    if name.startswith("per_query_") or name == "attention_kl_is_roundoff"
                    else (2, 2, self.kv_heads)
                )
            )
            if array.shape != expected_shape:
                raise ValueError(f"local statistic shape differs from all-Head coverage: {name}")
            if (
                "squared_error" in name or "energy" in name or name in ("attention_kl", "per_query_attention_kl")
            ) and not np.isfinite(array).all():
                raise ValueError(f"nonfinite local raw statistic: {name}")
        for bi, bits in enumerate(BIT_WIDTHS):
            ledger = quantization_byte_ledger(
                self.prompt_length, step + 1, self.num_layers, self.kv_heads, self.head_dim, bits
            )
            if not np.all(record.arrays["code_bytes"][:, bi] == ledger["code_bytes"]) or not np.all(
                record.arrays["statistic_bytes"][:, bi] == ledger["statistic_bytes"]
            ):
                raise ValueError("actual packed code/statistic bytes differ from layout budget")
        self.pending.setdefault(step, {})[record.layer_id] = record
        if len(self.pending[step]) == self.num_layers:
            records = self.pending.pop(step)
            arrays = {
                name: np.stack(
                    [records[layer].arrays[name] for layer in range(self.num_layers)],
                    axis=0 if name.startswith("formula_sdpa_") else 2,
                )
                for name in record.arrays
            }
            relative = f"local_step_{step:03d}.npz"
            write_npz(self.directory / relative, arrays)
            self.local_files[step] = relative
            self.summary["local"][str(step)] = {
                side: {
                    str(bits): {
                        metric: _describe(arrays[metric][si, bi])
                        for metric in ("representation_nmse", "output_nmse", "attention_kl")
                    }
                    for bi, bits in enumerate(BIT_WIDTHS)
                }
                for si, side in enumerate(SIDES)
            }

    def write_checkpoint(self, checkpoint):
        step = checkpoint.decode_step
        if step not in self.local_steps or step in self.checkpoints:
            raise ValueError("unexpected or duplicate model checkpoint")
        expected = self.targets if step == self.primary_step else ()
        if tuple(branch.target for branch in checkpoint.branches) != expected:
            raise ValueError("model branches do not match the frozen single-Head targets")
        self.checkpoints[step] = {
            "input_token_id": checkpoint.input_token_id,
            "expected_token_id": checkpoint.expected_token_id,
            "baseline_predicted_token_id": checkpoint.baseline_predicted_token_id,
        }
        if step != self.primary_step:
            return
        branches = checkpoint.branches
        for branch in branches:
            ledger = quantization_byte_ledger(
                self.prompt_length, step + 1, self.num_layers, self.kv_heads, self.head_dim, branch.target.bits
            )
            if (
                branch.measurement.cache_length != self.prompt_length + step + 1
                or branch.measurement.code_bytes != ledger["code_bytes"]
                or branch.measurement.statistic_bytes != ledger["statistic_bytes"]
            ):
                raise ValueError("model branch length/actual encoding bytes differ from the layout")
        arrays = {
            "layer_id": np.array([b.target.layer_id for b in branches], np.uint16),
            "head_id": np.array([b.target.head_id for b in branches], np.uint16),
            "side_id": np.array([SIDES.index(b.target.side) for b in branches], np.uint8),
            "bits": np.array([b.target.bits or 16 for b in branches], np.uint8),
            "predicted_token_id": np.array([b.predicted_token_id for b in branches], np.uint32),
            "prediction_kl": np.array([b.prediction_kl for b in branches], np.float64),
            "kl_is_roundoff": np.array([b.kl_is_roundoff for b in branches], bool),
            "delta_nll": np.array([b.delta_nll for b in branches], np.float64),
            "top1_match": np.array([b.top1_match for b in branches], bool),
        }
        for name in (
            "cache_length",
            "quantized_tokens",
            "code_bytes",
            "statistic_bytes",
            "squared_error",
            "fp32_squared_error",
            "original_squared_sum",
        ):
            dtype = np.uint64 if name.endswith("bytes") or name in ("cache_length", "quantized_tokens") else np.float64
            arrays[name] = np.array([getattr(b.measurement, name) for b in branches], dtype)
        write_npz(self.directory / "model_branches.npz", arrays)
        self.model_written = True
        self.summary["model"] = {}
        for si, side in enumerate(SIDES):
            self.summary["model"][side] = {}
            for bits in BIT_WIDTHS:
                mask = (arrays["side_id"] == si) & (arrays["bits"] == bits)
                values = np.full((self.num_layers, self.kv_heads), np.nan)
                values[arrays["layer_id"][mask], arrays["head_id"][mask]] = arrays["prediction_kl"][mask]
                self.summary["model"][side][str(bits)] = _describe(values)
        bypass = arrays["bits"] == 16
        self.summary["bypass"] = {
            "prediction_kl": arrays["prediction_kl"][bypass].tolist(),
            "delta_nll": arrays["delta_nll"][bypass].tolist(),
            "top1_match": arrays["top1_match"][bypass].tolist(),
        }

    def finalize(self, replay_ids, result, resources):
        if (
            self.pending
            or set(self.local_files) != set(self.local_steps)
            or set(self.checkpoints) != set(self.local_steps)
            or not self.model_written
        ):
            raise ValueError("missing local layers, checkpoints or model branch results")
        if result.baseline.all_predicted_token_ids != tuple(replay_ids):
            raise ValueError("unmodified replay predictions differ from B1")
        ledgers = {
            str(step): {
                str(bits or 16): quantization_byte_ledger(
                    self.prompt_length, step + 1, self.num_layers, self.kv_heads, self.head_dim, bits
                )
                for bits in (None, *BIT_WIDTHS)
            }
            for step in self.local_steps
        }
        write_json(
            self.directory / "byte_ledger.json",
            {
                "scope": "one single-Head/side candidate at a time, not all Heads compressed together",
                "by_step": ledgers,
                "shadow_gpu_memory_is_separate": True,
            },
        )
        write_json(
            self.directory / "replay.json",
            {
                "generated_ids": list(replay_ids),
                "prefill_predicted_token_id": result.baseline.prefill_predicted_token_id,
                "decode_predicted_token_ids": list(result.baseline.decode_predicted_token_ids),
                "checkpoints": self.checkpoints,
            },
        )
        self.summary["resources"] = resources
        write_json(self.directory / "summary.json", self.summary)
        self.metadata.update(
            status="completed",
            local_files={str(k): v for k, v in self.local_files.items()},
            baseline_top1_match=True,
            completed_checkpoint_steps=sorted(self.checkpoints),
        )
        write_json(self.directory / "metadata.json", self.metadata, overwrite=True)
        return self.summary

    def fail(self, exc):
        self.metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(self.directory / "metadata.json", self.metadata, overwrite=True)


def read_sample_profiles(directory):
    """Read the primary step only; late local observations do not fit the Calib profile."""
    directory = Path(directory)
    metadata = read_json(directory / "metadata.json")
    if metadata["status"] != "completed":
        raise ValueError("cannot profile an incomplete sample")
    with np.load(directory / metadata["local_files"][str(metadata["primary_step"])]) as archive:
        local = archive["output_nmse"].copy()
    prediction = np.full(local.shape, np.nan)
    with np.load(directory / "model_branches.npz") as archive:
        for row, bits in enumerate(archive["bits"]):
            if bits in BIT_WIDTHS:
                index = (
                    int(archive["side_id"][row]),
                    BIT_WIDTHS.index(int(bits)),
                    int(archive["layer_id"][row]),
                    int(archive["head_id"][row]),
                )
                prediction[index] = archive["prediction_kl"][row]
    return {"local_output_nmse": local, "prediction_kl": prediction}
