import struct

import numpy as np
import torch

from experiments.head_analysis.analysis.pre_rope_head_budget import run_analysis,select_ranks
from experiments.head_analysis.analysis.kv_representation_budget import byte_ledger,allocate_budget
from experiments.head_analysis.analysis.pca_reconstruction import projection_errors,rope_coefficients,apply_position_rope
from experiments.head_analysis.collectors.kv_representation import KVRepresentationConfig
from experiments.head_analysis.collectors.kv_representation import prompt_statistics,sample_positions,KVRepresentationRecord
from experiments.head_analysis.storage.kv_representation import KVRepresentationWriter
from experiments.head_analysis.common.io import read_json,write_json,write_npz


def fixture(directory,validation_scale=1.,dimension=8,heads=2):
    L,H,D,T=2,heads,dimension,600
    bundle,codec=directory/'bundle',directory/'codec'
    bundle.mkdir(parents=True)
    mu=np.zeros((3,L,H,D));basis=np.broadcast_to(np.eye(D),(3,L,H,D,D)).copy()
    write_npz(bundle/'rope_calibration.npz',{'mean':mu,'basis':basis,'scale':np.ones((L,H))})
    write_npz(bundle/'calibration.npz',{'scale':np.ones((2,L,H))})
    config={'hidden_size':4*D,'num_attention_heads':4,'num_key_value_heads':H,'max_position_embeddings':32768,
        'rope_theta':500000.,'rope_scaling':{'rope_type':'llama3','factor':8.,'low_freq_factor':1.,
                                           'high_freq_factor':4.,'original_max_position_embeddings':8192}}
    write_json(bundle/'rope_config.json',config)
    c,s=rope_coefficients(config,np.arange(T),D);positions=sample_positions(T)
    labels=('ruler_8192','ruler_32768','govreport_8192','govreport_16384','pg19_8192','pg19_32768')
    rows=[];references={}
    for phase,split in (('pilot','calib'),('validation','validation')):
        ids=[]
        for i,label in enumerate(labels):
            sid=f'{phase}_{i}';ids.append(sid);old=directory/'v2'/sid
            row={'sample_id':sid,'split':split,'condition':label,'dataset':label.rsplit('_',1)[0],
                'source_group':f'{phase}:pg' if i>=4 else sid,'pair_id':f'{phase}_pg' if i>=4 else None,
                'sample_dir':str(old),'prompt_length':T,'num_layers':L,'num_kv_heads':H,'head_dim':D}
            rows.append(row)
            writer=KVRepresentationWriter(old,sample_id=sid,prompt_length=T,num_layers=L,
                                          config=KVRepresentationConfig(decode_steps=()))
            for layer in range(L):
                rng=torch.Generator().manual_seed(80+i*3+layer+(30 if phase=='validation' else 0))
                amplitudes=torch.tensor([[8,4,2,1,.1,.05,.02,.01],[4,4,4,4,3,3,3,3]]).repeat((H+1)//2,D//8)[:H]
                x=torch.randn(H,T,D,generator=rng)*amplitudes[:,None]
                if phase=='validation':x*=validation_scale
                pre=x.bfloat16()
                post=torch.tensor(apply_position_rope(pre.float().numpy(),c,s,bf16=True),dtype=torch.bfloat16)
                pstats,qstats=prompt_statistics(pre,71),prompt_statistics(post,71)
                arrays={'sample_positions':positions.astype(np.uint32),
                    'sample_regions':np.repeat(np.array([0,1,2],dtype=np.uint8),[32,448,32]),
                    'k_pre':pre[:,positions],'k_post':post[:,positions],'v':post[:,positions]}
                for rep in ('k_post','v'):
                    arrays.update({f'{rep}_{k}':qstats[k] for k in ('count','mean','covariance')})
                writer.write_record(KVRepresentationRecord(sid,T,layer,0,None,arrays))
                raw_mu=np.stack((pstats['mean'][1],qstats['mean'][1]));cov=np.stack((pstats['covariance'][1],qstats['covariance'][1]))
                fixed=projection_errors(raw_mu,cov,mu[:2,layer],basis[:2,layer],[D//2,D])
                write_npz(codec/phase/'samples'/sid/f'prefill/layer_{layer:03d}.npz',
                    {**{f'k_pre_{k}':pstats[k] for k in ('count','mean','covariance')},
                     'ranks':np.array([D//2,D]),'fixed_ideal_in_domain_mse':fixed})
                if phase=='validation' and i==0 and layer==0:
                    references={'pre':pre[:,positions[32:-32]].float().numpy(),
                        'post':post[:,positions[32:-32]].float().numpy(),
                        'c':c[positions[32:-32]],'s':s[positions[32:-32]]}
            writer.finalize()
        write_json(codec/phase/'run_metadata.json',{'status':'completed','cohort':phase,
                  'completed_sample_ids':ids,'target_sample_ids':ids})
    # Any accidental evaluation of Test will encounter a missing original sample directory.
    write_json(bundle/'split.json',{'samples':rows+[{'sample_id':'test_guard','split':'test'}]})
    return bundle,codec,references


def test_metadata_extension_preserves_old_budget_and_counts_new_T_g():
    r=np.full((32,8),64)
    old=byte_ledger(r,[8192]);new=byte_ledger(r,[8192],request_metadata_bytes=8)
    assert old['cache_metadata_bytes'][0]==4 and new['cache_metadata_bytes'][0]==8
    assert new['cache_total_bytes'][0]-old['cache_total_bytes'][0]==4
    np.testing.assert_array_equal(new['resident_identical_requests_bytes']-old['resident_identical_requests_bytes'],
                                  np.array([[4],[16],[64]]))
    costs=np.broadcast_to(np.array([8.,4.,1.]),(2,2,3))
    a=allocate_budget(costs,4,ranks=[2,4,6],dimension=8)
    b=allocate_budget(costs,4,ranks=[2,4,6],dimension=8,request_metadata_bytes=8)
    np.testing.assert_array_equal(a['strict_ranks'],b['strict_ranks'])


def test_rank_gather_keeps_sample_position_method_layer_head_axes():
    curve=np.arange(2*2*2*3*4).reshape(2,2,2,3,4)
    r=np.array([[[[1,2,3],[4,3,2]],[[4,3,2],[1,2,3]]]]*2)
    z=select_ranks(curve,r,np.arange(1,5))
    for s,p,m,l,h in np.ndindex(z.shape):assert z[s,p,m,l,h]==curve[s,p,l,h,r[p,m,l,h]-1]


def test_end_to_end_frozen_allocation_actual_bf16_and_parameter_bytes(tmp_path):
    bundle,codec,ref=fixture(tmp_path/'source')
    output=tmp_path/'result'
    report=run_analysis(bundle,codec,output,ranks=np.array([2,4,6,8]),baseline_rank=4)
    assert report['status']=='completed' and read_json(output/'metadata.json')['means_bases_refitted'] is False
    with np.load(output/'stability.npz') as z:
        assert z['pre_reference_per_layer_rho'].shape == (3,6,2)
        assert z['oracle_pre_reference_per_layer_rho'].shape == (3,6,2)
    allocation=read_json(output/'allocation.json')
    assert allocation['status']=='frozen_before_validation'
    ranks=np.array(allocation['rank_matrices'])
    ledger=read_json(output/'byte_ledger.json')
    for p,name in enumerate(('pre','post')):
        base=ledger[name]['uniform'];strict=ledger[name]['head_aware_strict']
        assert strict['model_total_bytes']<=base['model_total_bytes']
        assert np.all(np.array(strict['cache_total_bytes'])<=base['cache_total_bytes'])
        for method,label in enumerate(('uniform','head_aware_strict')):
            blob=(output/f'{name}_{label}_parameters.bin').read_bytes()
            assert len(blob)==ledger[name][label]['model_total_bytes']
            L,H,D,pos,flag=struct.unpack('<HHHBB',blob[:8]);assert (L,H,D,pos)==(2,2,8,p)
            count=L*H if flag else 1
            table=np.frombuffer(blob[8:8+count*2],'<u2')
            np.testing.assert_array_equal(table,ranks[p,method].ravel() if flag else [ranks[p,method,0,0]])
            values=np.frombuffer(blob[8+count*2:],'<f4');assert np.all(values[:L*H*D]==0)
            cursor=L*H*D
            for l,h in np.ndindex(L,H):
                r=ranks[p,method,l,h];size=D*r
                np.testing.assert_array_equal(values[cursor:cursor+size].reshape(D,r),np.eye(D)[:,:r]);cursor+=size
            assert cursor==len(values)
    with np.load(output/'validation_sampled.npz') as z:
        for p,m,h in np.ndindex(2,2,2):
            expected=ref['pre' if p==0 else 'post'][h:h+1].copy()
            expected[...,ranks[p,m,0,h]:]=0
            if p==0:expected=apply_position_rope(expected,ref['c'],ref['s'],bf16=True)
            loss=np.square(expected.astype(np.float64)-ref['post'][h:h+1].astype(np.float64)).sum(-1).mean()
            np.testing.assert_allclose(z['bf16'][0,p,m,0,h],loss,rtol=1e-12)
    # Change all Validation values while retaining identical Calib and fixed parameters.
    other_bundle,other_codec,_=fixture(tmp_path/'other',validation_scale=3.)
    run_analysis(other_bundle,other_codec,tmp_path/'other_result',ranks=np.array([2,4,6,8]),baseline_rank=4)
    other=read_json(tmp_path/'other_result/allocation.json')
    np.testing.assert_array_equal(allocation['rank_matrices'],other['rank_matrices'])
    np.testing.assert_array_equal(allocation['costs'],other['costs'])
    with np.load(output/'validation_curves.npz') as a,np.load(tmp_path/'other_result/validation_curves.npz') as b:
        assert not np.array_equal(a['normalized_mse'],b['normalized_mse'])


def test_real_dimension_head_ranks_do_not_overflow_coefficient_byte_check(tmp_path):
    bundle,codec,_=fixture(tmp_path/'source',dimension=128)
    output=tmp_path/'result'
    run_analysis(bundle,codec,output)
    ranks=np.asarray(read_json(output/'allocation.json')['rank_matrices'],dtype=np.int64)
    assert ranks.max()>=80
    with np.load(output/'validation_sampled.npz') as z:
        expected=np.broadcast_to(2*448*ranks,(6,)+ranks.shape)
        assert expected.max()>np.iinfo(np.uint16).max
        np.testing.assert_array_equal(z['coefficient_bytes'],expected)
