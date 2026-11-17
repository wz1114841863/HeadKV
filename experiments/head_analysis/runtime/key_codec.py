"""Fixed-mean Key encoding; these functions never modify a model or Cache."""
from contextlib import contextmanager
from time import perf_counter

import torch
import numpy as np


def rotate_key(key, cos, sin):
    """Actual Llama half rotation, preserving the caller's arithmetic dtype."""
    half = key.shape[-1] // 2
    rotated = torch.cat((-key[..., half:], key[..., :half]), -1)
    return key * cos + rotated * sin


def encode_key(key, mean, basis, rank, *, ideal=False):
    dtype = torch.float64 if ideal else torch.float32
    coefficients = (key.to(dtype) - mean.to(dtype)[:, None]) @ basis.to(dtype)[..., :rank]
    return coefficients if ideal else coefficients.to(torch.bfloat16)


def decode_key(coefficients, mean, basis, *, ideal=False, output_dtype=torch.bfloat16):
    dtype = torch.float64 if ideal else torch.float32
    rank = coefficients.shape[-1]
    decoded = coefficients.to(dtype) @ basis.to(dtype)[..., :rank].transpose(-1, -2) + mean.to(dtype)[:, None]
    return decoded if ideal else decoded.to(output_dtype)


@contextmanager
def codec_timer(totals, name, device):
    """Isolated GPU event time, or CPU wall time; excludes transfers outside scope."""
    if device.type == "cuda":
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device)
        begin.record(stream)
        yield
        end.record(stream)
        end.synchronize()
        elapsed = begin.elapsed_time(end) / 1000
    else:
        start = perf_counter()
        yield
        elapsed = perf_counter() - start
    totals[name] = totals.get(name, 0.) + elapsed


def key_byte_ledger(length, consumed, layers, heads, dimension=128, rank=64):
    """One candidate: contiguous positions, BF16 coefficients, FP32 parameters."""
    middle = length - 128
    if middle <= 0 or consumed < 0 or not 0 < rank <= dimension:
        raise ValueError("invalid Key encoding dimensions/length")
    lh = layers * heads
    model = 4 * lh * (dimension * rank + dimension) + 10
    coefficient = 2 * lh * rank * (middle + consumed)
    protected = 2 * 128 * lh * dimension
    dense = 2 * lh * dimension * (length + consumed)
    cache = coefficient + protected + 8
    return {"model_bytes": model, "coefficient_bytes": coefficient, "protected_bytes": protected,
            "request_metadata_bytes": 8, "encoded_key_bytes": cache, "dense_key_bytes": dense + 8,
            "dense_value_bytes": dense, "position_table_bytes_on_demand": 0,
            "position_table_bytes_if_cached": 4 * (length + consumed) * dimension,
            "one_layer_bf16_restore_bytes": 2 * heads * (length + consumed) * dimension,
            "encode_prompt_flops": 2 * middle * lh * dimension * rank,
            "encode_new_token_flops": 2 * lh * dimension * rank,
            "restore_scan_flops": 2 * (middle + consumed) * lh * dimension * rank,
            "copies": {str(m): {"candidate_kv_resident_bytes": model + m * (cache + dense),
                                "dense_kv_resident_bytes": m * (2 * dense + 8)} for m in (1, 4, 16)},
            "scope": "array layout budget; Dense plus shadow CUDA peak is not compressed inference"}


def benchmark_restore_scan(seeds, mean, basis, target, length, inv_freq, scaling, *, block_size=1024):
    """Controlled, layer-sized fixture, not a timed model Attention call.

    Real sampled codes are tiled to the actual history length; this tests cost,
    not reconstruction quality. Trig generation, casts and scratch writes are timed.
    """
    device = target.device
    n = target.shape[1]-128
    positions = torch.cat((torch.arange(64,length-64,device=device),
                           torch.arange(length,target.shape[1],device=device)))
    scratch = torch.empty_like(target,dtype=torch.bfloat16)
    values = []
    for path in range(2):
        code = seeds[path].to(device)
        code = code.repeat(1,(n+code.shape[1]-1)//code.shape[1],1)[:,:n].contiguous()
        m,u = mean[path].float(),basis[path,...,:code.shape[-1]].float()
        samples = []
        for repeat in range(4):  # one warmup, three measured passes
            measured = {}
            with codec_timer(measured,'seconds',device):
                scratch[:,:64]=target[:,:64]
                scratch[:,length-64:length]=target[:,length-64:length]
                for start in range(0,n,block_size):
                    end=min(start+block_size,n)
                    decoded=decode_key(code[:,start:end],m,u)
                    p=positions[start:end]
                    if path==0:
                        theta=p.float()[:,None]*inv_freq.float()[None]
                        phase=torch.cat((theta,theta),-1)
                        c=(phase.cos()*scaling).to(target.dtype)
                        s=(phase.sin()*scaling).to(target.dtype)
                        decoded=rotate_key(decoded,c,s).to(torch.bfloat16)
                    scratch[:,p]=decoded
            if repeat:samples.append(measured['seconds'])
        values.append(float(np.median(samples)))
    samples=[]
    for repeat in range(4):
        measured={}
        with codec_timer(measured,'seconds',device):
            scratch.copy_(target)
        if repeat:samples.append(measured['seconds'])
    return np.array([*values,float(np.median(samples))])
