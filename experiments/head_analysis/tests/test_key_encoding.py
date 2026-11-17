from dataclasses import replace
import os

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.head_analysis.analysis.pca_reconstruction import projection_errors, sample_encoding_errors
from experiments.head_analysis.collectors.key_encoding import KeyEncodingCollector
from experiments.head_analysis.common.io import read_json
from experiments.head_analysis.runners.b2_runner import B1ReplaySample, teacher_forced_replay, B2RunnerError
from experiments.head_analysis.runners.key_encoding_runner import collect_sample
from experiments.head_analysis.runtime.key_codec import encode_key, decode_key, rotate_key, key_byte_ledger


def parameters(layers=2, heads=2, dim=16):
    rng=np.random.default_rng(71)
    mean=rng.normal(scale=.03,size=(3,layers,heads,dim))
    basis=np.linalg.qr(rng.normal(size=(3,layers,heads,dim,dim)))[0]
    return mean,basis,np.ones((layers,heads))


def test_fixed_mean_codec_trace_and_position_dependent_score_with_bias():
    mu,u,_=parameters(layers=1)
    x=torch.randn(2,37,16,dtype=torch.float64)
    m,b=torch.tensor(mu[0,0]),torch.tensor(u[0,0])
    z=encode_key(x,m,b,4,ideal=True)
    restored=decode_key(z,m,b,ideal=True)
    xc=x-x.mean(1,keepdim=True)
    analytic=projection_errors(x.mean(1).numpy(),(xc.transpose(1,2)@xc/37).numpy(),mu[0,0],u[0,0],[4])
    np.testing.assert_allclose((restored-x).square().sum(-1).mean(-1).numpy(),analytic[:,0],atol=1e-12)
    # Synthetic real orthogonal rotations: each position has its own theta.
    angle=torch.arange(37,dtype=torch.float64)[:,None]*torch.linspace(.01,.3,8)
    c,s=torch.cat((angle.cos(),angle.cos()),-1),torch.cat((angle.sin(),angle.sin()),-1)
    q=torch.randn(2,16,dtype=torch.float64)
    explicit=(rotate_key(restored,c,s)*q[:,None]).sum(-1)
    rotated_q=rotate_key(q[:,None].expand(-1,37,-1),c,-s)
    phi=rotated_q@b[...,:4]
    low=(phi*z).sum(-1)+(rotated_q*m[:,None]).sum(-1)
    torch.testing.assert_close(explicit,low,atol=1e-12,rtol=1e-12)
    assert not torch.allclose(phi[:,0],phi[:,-1])
    actual=decode_key(encode_key(x,m,b,4),m,b)
    assert actual.dtype==torch.bfloat16
    torch.testing.assert_close(actual,(((x.float()-m.float()[:,None])@b.float()[...,:4]).bfloat16().float()
        @b.float()[...,:4].transpose(-1,-2)+m.float()[:,None]).bfloat16(),atol=0,rtol=0)
    # Historical V3 stored BF16 coefficients but evaluated Post before final BF16 output rounding.
    historical=sample_encoding_errors(x.numpy(),mu[0,0],u[0,0],[4])['bf16_mse'][:,0]
    decoded32=decode_key(encode_key(x,m,b,4),m,b,output_dtype=torch.float32)
    np.testing.assert_allclose((decoded32.double()-x).square().sum(-1).mean(-1),historical,atol=1e-12)
    assert torch.any(decoded32!=decoded32.bfloat16().float())


def test_full_budget_counts_parameters_protection_decode_and_restore_workspace():
    z=key_byte_ledger(32768,256,32,8)
    assert z['encoded_key_bytes']==(1028+8)*2**20+8
    assert z['model_bytes']==8*2**20+128*2**10+10
    assert z['one_layer_bf16_restore_bytes']==2*8*(32768+256)*128
    assert z['position_table_bytes_if_cached']==4*(32768+256)*128
    assert z['copies']['1']['candidate_kv_resident_bytes']==z['model_bytes']+z['encoded_key_bytes']+z['dense_value_bytes']


@pytest.mark.parametrize('dtype,dataset,consumed',[(torch.float32,'ruler',6),(torch.bfloat16,'govreport',256)])
def test_tiny_dense_replay_full_prompt_and_continuous_append(tmp_path,dtype,dataset,consumed):
    torch.manual_seed(12)
    config=LlamaConfig(vocab_size=48,hidden_size=64,intermediate_size=96,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=2048,
        eos_token_id=None,pad_token_id=None)
    config._attn_implementation='sdpa'
    device=os.environ.get('HEADKV_TEST_DEVICE','cpu')
    model=LlamaForCausalLM(config).eval().to(device=device,dtype=dtype)
    prompt=[(i%47)+1 for i in range(640)]
    with torch.inference_mode():
        ids=model.generate(torch.tensor([prompt],device=device),max_new_tokens=consumed+1,do_sample=False)[0,640:].tolist()
    sample=B1ReplaySample({'sample_id':'tiny','dataset':dataset,'input_ids':prompt,
                          'actual_prompt_length':640},'batch','run',tuple(ids),'unused','unused')
    baseline=teacher_forced_replay(model,device=device,sample_id='tiny',prompt_ids=prompt,
                                  replay_generated_ids=ids,capture_logits=True)
    observed=[]
    raw_pre={}
    new_pre={0:[],1:[]}
    cache_rows={0:[],1:[]}
    rotations=[]
    counts=[0,0]
    def capture_pre(l,module,args,out):
        counts[l]+=1
        if out.shape[1]==640:
            raw_pre[l]=out.view(1,640,2,16).transpose(1,2)[0].clone()
        else:
            new_pre[l].append(out.view(1,1,2,16).transpose(1,2)[0].clone())
    def save_output(m,a,o):
        observed.append(o.logits[0,-1].detach().cpu().clone())
        if o.past_key_values.get_seq_length()>640:
            for l in range(2):cache_rows[l].append(o.past_key_values.layers[l].keys[0,:,-1:].clone())
    handles=[model.register_forward_hook(save_output),
             model.model.rotary_emb.register_forward_hook(lambda m,a,o:rotations.append(tuple(x[0].clone() for x in o)))]
    from functools import partial
    handles += [layer.self_attn.k_proj.register_forward_hook(partial(capture_pre,l))
                for l,layer in enumerate(model.model.layers)]
    params=parameters()
    original=[a.copy() for a in params]
    try:
        result=collect_sample(model,sample,tmp_path/'run',params,device=device,block_size=97,rank=4,
                              serialize_payload=dtype==torch.bfloat16)
    finally:
        for h in handles:h.remove()
    assert counts==[consumed+1]*2
    assert result['top1_match'] and result['raw_rope_exact']
    for a,b in zip(observed,baseline.logits):torch.testing.assert_close(a,b,atol=0,rtol=0)
    for a,b in zip(params,original):np.testing.assert_array_equal(a,b)
    for layer in range(2):
        with np.load(tmp_path/f'run/prefill/layer_{layer:03d}.npz') as z:
            assert z['raw_rope_exact']
            np.testing.assert_array_equal(z['k_pre_count'],[640,512])
            x=raw_pre[layer][:,64:-64].double().cpu().numpy()
            np.testing.assert_allclose(z['k_pre_mean'][1],x.mean(1),rtol=1e-6,atol=1e-8)
            assert z['middle_mse'].shape==(2,2,2,2)
            np.testing.assert_allclose(z['prompt_mse'],z['middle_mse']*512/640,atol=1e-12)
            assert np.max(z['transform_only_mse'][1])==0
            assert np.max(z['inverse_linear_closure_max'])<1e-10
        with np.load(tmp_path/f'run/decode/layer_{layer:03d}.npz') as z:
            assert z['coefficient_bits'].dtype==np.uint16
            assert z['coefficient_bits'].shape==(consumed,2,2,4)
            np.testing.assert_array_equal(z['positions'],np.arange(640,640+consumed))
            assert np.isfinite(z['mse']).all() and z['mse'].shape==(consumed,2,2,2,2)
            # Independent direct matrix expressions on actual projection/Cache rows.
            for step in (0,consumed-1):
                c,s=rotations[step+1]
                for path in range(2):
                    xx=new_pre[layer][step] if path==0 else cache_rows[layer][step]
                    mm=torch.tensor(params[0][path,layer],device=device)
                    uu=torch.tensor(params[1][path,layer].copy(),device=device)
                    for ri,r in enumerate((4,16)):
                        zz=((xx.float()-mm.float()[:,None])@uu.float()[...,:r]).bfloat16()
                        yy=(zz.float()@uu.float()[...,:r].transpose(-1,-2)+mm.float()[:,None]).bfloat16()
                        if path==0:yy=rotate_key(yy,c,s).bfloat16()
                        loss=(yy.double()-cache_rows[layer][step].double()).square().sum((1,2)).cpu().numpy()
                        np.testing.assert_allclose(z['mse'][step,path,ri,1],loss,atol=1e-12)
                        if ri==0:np.testing.assert_array_equal(z['coefficient_bits'][step,path],zz[:,0].contiguous().view(torch.uint16).cpu().numpy())
    assert read_json(tmp_path/'run/metadata.json')['status']=='completed'
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    assert not (tmp_path/'run/temporary_prompt_key.bin').exists()
    # A failed replay must not publish a completed sample or leave Probe hooks attached.
    bad=replace(sample,generated_ids=(*sample.generated_ids[:-1],(sample.generated_ids[-1]+1)%48))
    with pytest.raises(B2RunnerError,match='Top-1'):
        collect_sample(model,bad,tmp_path/'bad',params,device=device,block_size=127,rank=4)
    assert read_json(tmp_path/'bad/metadata.json')['status']=='failed'
    assert not (tmp_path/'bad/summary.json').exists()
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
