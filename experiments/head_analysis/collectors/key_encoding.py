"""Read-only full-Prompt and append-only Decode Key codec validation."""
import numpy as np
import torch

from experiments.head_analysis.analysis.pca_reconstruction import projection_errors, oracle_errors, apply_position_rope
from experiments.head_analysis.collectors.kv_representation import prompt_statistics, sample_positions, _cpu_sample
from experiments.head_analysis.runtime.key_codec import (
    encode_key, decode_key, rotate_key, codec_timer, benchmark_restore_scan,
)
from kvpress.context import KVPhase


class KeyEncodingCollector:
    """One temporary CPU Pre tensor per layer; no original GPU tensor survives a callback.

    Output error axes: path(pre/post), rank(main/full), precision(ideal/BF16), Head.
    Only r64 coefficients are retained for Decode; full-Prompt buffers are released.
    """

    def __init__(self, length, mean, basis, scale, on_record, *, block_size=1024, rank=64,
                 checkpoints=(31, 255), on_payload=None, rotary_spec=None):
        self.length, self.mean, self.basis, self.scale = length, mean, basis, scale
        self.ranks = (rank, basis.shape[-1])
        self.on_record, self.block_size = on_record, block_size
        self.checkpoints, self.on_payload = checkpoints, on_payload
        self.rotary_spec, self.code_seeds = rotary_spec, {}
        self.pending, self.prefix, self.history, self.snapshots = {}, {}, {}, {}
        self.timings, self.prompt_errors, self.decode_errors, self.checks = {}, {}, {}, []
        self.restore_costs = {}
        self.positions = sample_positions(length)

    def _parameters(self, layer, device):
        return (torch.as_tensor(self.mean[:, layer], device=device),
                torch.as_tensor(self.basis[:, layer].copy(), device=device))

    def on_key_projection(self, context):
        if context.position_embeddings is None:
            raise ValueError("Key encoding requires the real position_embeddings")
        x = context.key_pre[0]
        expected = (torch.arange(self.length) if context.phase is KVPhase.PREFILL
                    else torch.tensor([self.length + context.decode_step]))
        if not torch.equal(context.token_positions.cpu(), expected):
            raise ValueError("Key projection positions differ from the replay")
        mu, u = self._parameters(context.layer_id, x.device)
        phase = 'prefill' if context.phase is KVPhase.PREFILL else 'decode'
        lo, hi = (64, self.length - 64) if context.phase is KVPhase.PREFILL else (0, 1)
        # A single layer's original CPU values are ephemeral reference data, never written.
        raw = torch.empty(x.shape, dtype=x.dtype, device="cpu")
        z = torch.empty((x.shape[0], hi-lo, self.ranks[0]), dtype=torch.bfloat16)
        for start in range(0, x.shape[1], self.block_size):
            end = min(start + self.block_size, x.shape[1])
            raw[:, start:end] = x[:, start:end].cpu()
            a, b = max(start, lo), min(end, hi)
            if a < b:
                with codec_timer(self.timings, f"{phase}.pre_encode_r{self.ranks[0]}", x.device):
                    codes = encode_key(x[:, a:b], mu[0], u[0], self.ranks[0])
                z[:, a-lo:b-lo] = codes.cpu()
        cos, sin = context.position_embeddings
        self.pending[context.forward_id, context.layer_id] = (
            raw, z, cos[0].detach().cpu(), sin[0].detach().cpu())

    def _inverse_check(self, layer, target, cos, sin):
        x = target.detach().cpu().double().numpy()
        c, s = cos.double().numpy(), sin.double().numpy()
        inv = apply_position_rope(x, c, s, inverse=True)
        closed = apply_position_rope(inv, c, s)
        if np.max(np.abs(closed-x))>1e-10*max(1.,np.max(np.abs(x))):
            raise ValueError('inverse/forward linear RoPE closure failed')
        mu = torch.from_numpy(self.mean[2,layer])
        u = torch.from_numpy(self.basis[2,layer].copy())
        losses = []
        for r in self.ranks:
            z = encode_key(torch.from_numpy(inv),mu,u,r)
            decoded = decode_key(z,mu,u,output_dtype=torch.float32).double().numpy()
            post = apply_position_rope(decoded,c,s,bf16=True)
            losses.append(np.square(post-x).sum(-1).mean(-1))
        return {'inverse_bf16_mse':np.stack(losses),
                'inverse_linear_closure_max':np.array(np.max(np.abs(closed-x)))}

    def on_attention(self, context):
        layer, device = context.layer_id, context.key.device
        raw, pre_codes, cos, sin = self.pending.pop((context.forward_id, layer))
        prefill = context.phase is KVPhase.PREFILL
        phase = 'prefill' if prefill else 'decode'
        step = context.decode_step
        lo, hi = (64, self.length-64) if prefill else (0, 1)
        absolute = 0 if prefill else self.length + step
        target = context.key[0, :, absolute:absolute + raw.shape[1]]
        if target.shape != raw.shape:
            raise ValueError("Cache rows and Key projection rows differ")
        mu, u = self._parameters(layer, device)
        sums = np.zeros((2, 2, 2, raw.shape[0]), dtype=np.float64)
        transform = np.zeros((2, raw.shape[0]), dtype=np.float64)
        post_fp32 = np.zeros((2,raw.shape[0]),dtype=np.float64)
        exact = True
        for start in range(0, raw.shape[1], self.block_size):
            end = min(start+self.block_size, raw.shape[1])
            x = raw[:, start:end].to(device)
            c, s = cos[start:end].to(device), sin[start:end].to(device)
            baseline = rotate_key(x, c, s)
            exact &= torch.equal(baseline, target[:, start:end])
            a, b = max(start, lo), min(end, hi)
            if a < b:
                ref = target[:, a:b].double()
                ideal = rotate_key(x.double(), c.double(), s.double())[:, a-start:b-start]
                transform[0] += (ideal-ref).square().sum((1, 2)).cpu().numpy()
                transform[1] += (baseline[:, a-start:b-start].double()-ref).square().sum((1, 2)).cpu().numpy()
        if not exact:
            raise ValueError(f"layer {layer}: actual RoPE does not reproduce the Cache Key")
        # Reuse one real BF16 layer-sized scratch for both candidates, not 32 restored layers.
        scratch = torch.empty_like(target, dtype=torch.bfloat16)
        codes_out = []
        if prefill:self.code_seeds[layer]=[]
        for path in range(2):
            path_codes = []
            seed_chunks = []
            if prefill and path == 0 and self.on_payload is not None:
                self.on_payload(torch.cat((target[:, :64], target[:, -64:]), 1).cpu(), pre_codes)
            if prefill:
                scratch[:, :64] = target[:, :64]
                scratch[:, -64:] = target[:, -64:]
            for start in range(lo, hi, self.block_size):
                end = min(start+self.block_size, hi)
                x = (raw[:, start:end].to(device) if path == 0 else target[:, start:end])
                c, s = cos[start:end].to(device), sin[start:end].to(device)
                ref = target[:, start:end].double()
                for ri, rank in enumerate(self.ranks):
                    zi = encode_key(x, mu[path], u[path], rank, ideal=True)
                    ideal = decode_key(zi, mu[path], u[path], ideal=True)
                    if path == 0:
                        ideal = rotate_key(ideal, c.double(), s.double())
                    sums[path, ri, 0] += (ideal-ref).square().sum((1, 2)).cpu().numpy()
                    if path == 0 and ri == 0:
                        codes = pre_codes[:, start-lo:end-lo].to(device)
                    else:
                        with codec_timer(self.timings, f"{phase}.{('pre','post')[path]}_encode_r{rank}", device):
                            codes = encode_key(x, mu[path], u[path], rank)
                    with codec_timer(self.timings, f"{phase}.{('pre','post')[path]}_restore_r{rank}", device):
                        decoded = decode_key(codes, mu[path], u[path],
                                             output_dtype=torch.bfloat16 if path==0 else torch.float32)
                        if path == 0:
                            decoded = rotate_key(decoded, c, s).to(torch.bfloat16)
                        else:
                            decoded32=decoded
                            decoded=decoded.to(torch.bfloat16)
                        if ri == 0:
                            scratch[:, start:end] = decoded
                    sums[path, ri, 1] += (decoded.double()-ref).square().sum((1, 2)).cpu().numpy()
                    if path==1:
                        post_fp32[ri]+=(decoded32.double()-ref).square().sum((1,2)).cpu().numpy()
                    if ri == 0:
                        with codec_timer(self.timings, f"{phase}.{('pre','post')[path]}_dense_read_write", device):
                            dense_copy = target[:, start:end].clone()
                        del dense_copy
                        if not prefill:
                            path_codes.append(codes.cpu())
                        else:
                            selected=self.positions[32:-32]
                            selected=selected[(selected>=start)&(selected<end)]-start
                            seed_chunks.append(_cpu_sample(codes,selected))
            if not prefill:
                codes_out.append(torch.cat(path_codes, 1)[:, 0])
            else:
                self.code_seeds[layer].append(torch.cat(seed_chunks,1))
        mse = sums / (hi-lo)
        if prefill:
            stats = [prompt_statistics(raw, self.block_size), prompt_statistics(target, self.block_size)]
            fixed, oracle = [], []
            for path, stat in enumerate(stats):
                fixed.append(projection_errors(stat['mean'][1], stat['covariance'][1],
                                                self.mean[path, layer], self.basis[path, layer], self.ranks))
                oracle.append(oracle_errors(stat['mean'][1], stat['covariance'][1],
                                             self.mean[path, layer], self.ranks)['same_mean'])
            gap=(np.stack(fixed)-np.stack(oracle))/self.scale[layer][None,:,None]
            if not np.isfinite(mse).all() or np.min(gap)<-1e-10:
                raise ValueError('nonfinite error or fixed projection better than same-mean Oracle beyond roundoff')
            self.on_record(f"prefill/layer_{layer:03d}.npz", {
                **{f"k_pre_{k}": stats[0][k] for k in ('count','mean','covariance')},
                "middle_mse": mse, "prompt_mse": sums/self.length,
                "post_fp32_decode_mse":post_fp32/(hi-lo),
                "transform_only_mse": transform/(hi-lo), "ranks": np.array(self.ranks),
                "fixed_ideal_in_domain_mse": np.stack(fixed), "same_mean_oracle_mse": np.stack(oracle),
                "same_mean_oracle_gap":gap,
                "raw_rope_exact": np.array(exact),
                **self._inverse_check(layer,_cpu_sample(target,self.positions[32:-32]),
                                      cos[self.positions[32:-32]],sin[self.positions[32:-32]])})
            self.prompt_errors[layer] = mse
            self.prefix[layer] = (_cpu_sample(context.key[0], self.positions),
                                  _cpu_sample(context.value[0], self.positions))
            self.history[layer], self.decode_errors[layer] = [], []
        else:
            self.history[layer].append(torch.stack(codes_out))
            self.decode_errors[layer].append(mse)
            if step in self.checkpoints:
                prefix_exact = all(torch.equal(_cpu_sample(t[0], self.positions), r)
                                   for t, r in zip((context.key, context.value), self.prefix[layer]))
                previous = self.snapshots.get(layer)
                current = torch.stack(self.history[layer])
                history_exact = previous is None or torch.equal(current[:len(previous)], previous)
                if not prefix_exact or not history_exact:
                    raise ValueError("historical Prompt Cache or Decode coefficients changed")
                self.snapshots[layer] = current.clone()
                self.checks.append({"layer":layer, "step":step, "prefix_exact":prefix_exact,
                                    "history_exact":history_exact, "raw_rope_exact":exact})
                aux=self._inverse_check(layer,target,cos,sin)
                if self.rotary_spec is not None:
                    frequency,scaling=self.rotary_spec
                    aux['restore_scan_fixture_seconds_pre_post_dense']=benchmark_restore_scan(
                        self.code_seeds[layer],mu,u,context.key[0],self.length,
                        frequency.to(device),scaling,block_size=self.block_size)
                    aux['restore_scan_history_tokens']=np.array(context.key.shape[-2])
                    self.restore_costs.setdefault(step,[]).append(aux['restore_scan_fixture_seconds_pre_post_dense'])
                self.on_record(f"checkpoints/step_{step:03d}/layer_{layer:03d}.npz",aux)
        if prefill or (step+1) % 32 == 0:
            print(f"Key codec {'Prefill' if prefill else f'Decode {step+1}'} layer {layer}", flush=True)

    def finish(self, consumed):
        layers = len(self.mean[0])
        if self.pending or sorted(self.prompt_errors) != list(range(layers)):
            raise ValueError("incomplete Prompt Key encoding")
        for layer in range(layers):
            if len(self.history[layer]) != consumed:
                raise ValueError("incomplete continuous Decode encoding")
            z = torch.stack(self.history[layer]).contiguous()
            self.on_record(f"decode/layer_{layer:03d}.npz", {
                "positions":np.arange(self.length, self.length+consumed,dtype=np.uint32),
                "coefficient_bits":z.view(torch.uint16).numpy(),
                "mse":np.stack(self.decode_errors[layer]), "ranks":np.array(self.ranks)})
        prompt = np.stack([self.prompt_errors[l] for l in range(layers)], axis=3)
        decode = np.stack([self.decode_errors[l] for l in range(layers)], axis=4)
        # Prompt: path,rank,precision,layer,Head; Decode: step,path,rank,precision,layer,Head.
        scaled_prompt = prompt/self.scale[None,None,None]
        scaled_decode = decode/self.scale[None,None,None,None]
        self.on_record("errors.npz", {"middle_normalized_mse":scaled_prompt,
                                      "decode_normalized_mse":scaled_decode})
        return {"middle_normalized_mean":scaled_prompt.mean((3,4)),
                "middle_head_p90":np.quantile(scaled_prompt.reshape(2,2,2,-1),.9,axis=-1),
                "decode_normalized_mean":scaled_decode.mean((0,4,5)),
                "decode_first32_normalized_mean":scaled_decode[:32].mean((0,4,5)),
                "decode_late_normalized_mean":(scaled_decode[32:].mean((0,4,5)) if consumed>32 else None),
                "checks":self.checks, "codec_seconds":self.timings,
                "restore_scan_fixture_layer_sum_seconds":{str(k):np.sum(v,axis=0) for k,v in self.restore_costs.items()},
                "restore_scan_fixture_scope":"isolated per-layer scans, tiled real sampled codes; pre/post/dense; warmup+3 medians; not model latency",
                "codec_timing_scope":"GPU event or CPU wall; actual math only, excludes CPU copies/moments/NPZ",
                "raw_rope_exact":True, "prompt_tokens":self.length, "consumed_tokens":consumed}
