"""Layer-wise CPU NPZ storage for V2, preserving sampled BF16 bits."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from experiments.head_analysis.collectors.kv_representation import (
    KVRepresentationConfig, KVRepresentationRecord, NORM_STATISTICS, SCOPES,
)
from experiments.head_analysis.common.io import read_json, write_json, write_npz


class KVRepresentationWriter:
    def __init__(self, directory, *, sample_id, prompt_length, num_layers, config=KVRepresentationConfig()):
        self.directory = Path(directory)
        if self.directory.exists():
            raise FileExistsError(f"output already exists: {self.directory}")
        self.directory.mkdir(parents=True)
        self.sample_id, self.prompt_length, self.num_layers = sample_id, prompt_length, num_layers
        self.config = config
        self.files = {}
        self._coordinates = set()
        self._prefix_ok = True
        self._completed = False
        self._metadata = {"schema_version": "headkv-v2-kv-representation-v1", "status": "collecting",
                          "sample_id": sample_id, "prompt_length": prompt_length, "num_layers": num_layers,
                          "config": asdict(config), "scopes": SCOPES, "norm_statistics": NORM_STATISTICS,
                          "covariance_ddof": 0, "accumulation_dtype": "float64", "saved_moments_dtype": "float32",
                          "sample_axis": "Hkv,position,D", "scope_order": SCOPES,
                          "sample_regions": {"0": "initial_32", "1": "middle_448", "2": "recent_32"},
                          "raw_bfloat16_encoding": "uint16 bit view; not numeric integer conversion"}
        write_json(self.directory / "metadata.json", self._metadata)

    def write_record(self, record: KVRepresentationRecord):
        if self._completed:
            raise ValueError("writer has already completed")
        if record.sample_id != self.sample_id or record.prompt_length != self.prompt_length:
            raise ValueError("record sample or Prompt length differs")
        coordinate = (record.decode_step, record.layer_id)
        if coordinate in self._coordinates or not 0 <= record.layer_id < self.num_layers:
            raise ValueError("duplicate or invalid layer/step")
        if record.decode_step is not None and record.decode_step not in self.config.decode_steps:
            raise ValueError("unexpected Decode checkpoint")
        prefix = "prefill" if record.decode_step is None else f"decode/step_{record.decode_step:03d}"
        relative = f"{prefix}/layer_{record.layer_id:03d}.npz"
        arrays, fields = {}, {}
        for name, value in record.arrays.items():
            descriptor = {}
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu":
                    raise ValueError("Writer only accepts CPU tensors")
                descriptor["tensor_dtype"] = str(value.dtype).removeprefix("torch.")
                if value.dtype is torch.bfloat16:
                    value = value.contiguous().view(torch.uint16).numpy()
                else:
                    value = value.detach().contiguous().numpy()
            else:
                value = np.asarray(value)
            arrays[name] = value
            fields[name] = {**descriptor, "shape": list(value.shape), "stored_dtype": str(value.dtype)}
        write_npz(self.directory / relative, arrays)
        self.files[relative] = {"layer_id": record.layer_id, "forward_id": record.forward_id,
                                "decode_step": record.decode_step, "fields": fields, "checks": record.checks}
        self._coordinates.add(coordinate)
        self._prefix_ok &= all(record.checks.get(f"{name}_prefix_exact", True) for name in ("k_post", "v"))
        # Keep the layer index and diagnostic checks readable if a later callback fails.
        write_json(self.directory / "metadata.json", {**self._metadata, "files": self.files}, overwrite=True)

    def finalize(self):
        required = {(step, layer) for step in (None, *self.config.decode_steps) for layer in range(self.num_layers)}
        if self._coordinates != required:
            raise ValueError("missing Prefill layers or Decode checkpoints")
        if not self._prefix_ok:
            raise ValueError("historical Prompt K/V comparison failed")
        write_json(self.directory / "metadata.json", {**self._metadata, "status": "completed", "files": self.files},
                   overwrite=True)
        self._completed = True
        return self.directory / "metadata.json"


def read_record(directory, relative):
    """Restore raw tensors, including exact BF16 bits; statistics remain NumPy arrays."""
    directory = Path(directory)
    metadata = read_json(directory / "metadata.json")
    entry = metadata["files"][relative]
    result = {}
    with np.load(directory / relative, allow_pickle=False) as archive:
        for name, descriptor in entry["fields"].items():
            value = archive[name].copy()
            if "tensor_dtype" in descriptor:
                value = torch.from_numpy(value)
                if descriptor["tensor_dtype"] == "bfloat16":
                    value = value.view(torch.bfloat16)
            result[name] = value
    return result
