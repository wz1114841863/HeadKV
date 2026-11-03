"""Run a self-contained Tiny Llama B2 smoke test on an explicit device."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.head_analysis.collectors.b2_attention import B2AttentionCollector
from experiments.head_analysis.runners.b2_runner import teacher_forced_replay


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def run_smoke(device_name: str, dtype_name: str) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device_name}")
    torch.manual_seed(17)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(17)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            eos_token_id=None,
            pad_token_id=None,
        )
    ).to(device=device, dtype=resolve_dtype(dtype_name))
    model.eval()
    model.config._attn_implementation = "sdpa"
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = None
    prompt = [(index % 60) + 1 for index in range(384)]
    with torch.inference_mode():
        output = model.generate(
            torch.tensor([prompt], dtype=torch.long, device=device),
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
        )
    replay = output[0, len(prompt) :].tolist()
    baseline = teacher_forced_replay(
        model,
        device=device,
        sample_id="tiny_gpu_baseline",
        prompt_ids=prompt,
        replay_generated_ids=replay,
        capture_logits=True,
    )
    observed = teacher_forced_replay(
        model,
        device=device,
        sample_id="tiny_gpu_b2",
        prompt_ids=prompt,
        replay_generated_ids=replay,
        collector=B2AttentionCollector(),
        capture_logits=True,
    )
    if baseline.all_predicted_token_ids != observed.all_predicted_token_ids:
        raise RuntimeError("B2 Probe changed Tiny Llama Top-1 outputs")
    if baseline.logits is None or observed.logits is None or len(baseline.logits) != len(observed.logits):
        raise RuntimeError("Tiny Llama smoke did not retain comparable logits")
    if any(not torch.equal(expected, actual) for expected, actual in zip(baseline.logits, observed.logits)):
        raise RuntimeError("B2 Probe changed Tiny Llama logits")
    if len(observed.prefill_records) != 1 or len(observed.decode_records) != len(replay) - 1:
        raise RuntimeError("Tiny Llama B2 event coverage is incomplete")
    return {
        "status": "passed",
        "device": str(device),
        "dtype": dtype_name,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "prompt_length": len(prompt),
        "replay_length": len(replay),
        "prefill_layer_records": len(observed.prefill_records),
        "decode_layer_records": len(observed.decode_records),
        "logits_exact_match": True,
        "top1_exact_match": True,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args(argv)
    print(json.dumps(run_smoke(args.device, args.dtype), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
