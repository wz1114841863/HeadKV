"""Exploratory matched Pre/Post Head budgets, with fixed bases and no inference."""
from pathlib import Path
import struct

import numpy as np
import torch

from experiments.head_analysis.analysis.head_identity import fixed_reference_comparison, input_pair_correlations
from experiments.head_analysis.analysis.kv_representation import _json_value, describe
from experiments.head_analysis.analysis.kv_representation_budget import allocate_budget, byte_ledger
from experiments.head_analysis.analysis.pca_reconstruction import (
    RANKS, condition_weights, projection_errors, oracle_errors, rope_coefficients, apply_position_rope,
)
from experiments.head_analysis.common.io import read_json, write_json, write_npz, atomic_file
from experiments.head_analysis.common.statistics import finite_mean
from experiments.head_analysis.runtime.key_codec import encode_key, decode_key
from experiments.head_analysis.storage.representation_inputs import read_sample, load_codec_cohorts


POSITIONS = ('pre', 'post')
METHODS = ('uniform', 'head_aware_strict')


def _save(path, value):
    write_json(path, _json_value(value))


def full_curves(records, directory, mean, basis, scale, ranks):
    """Own-domain full-middle errors; no claim of full BF16 post-RoPE error."""
    curves, same, classic, energies = [], [], [], []
    for row in records:
        by_layer = [[], [], [], []]
        for layer in range(basis.shape[1]):
            old = read_sample(row, layer_ids=[layer])
            with np.load(Path(directory)/'samples'/row['sample_id']/f'prefill/layer_{layer:03d}.npz') as new:
                if not np.array_equal(new['k_pre_count'], [row['prompt_length'], row['prompt_length']-128]):
                    raise ValueError('K_pre scope counts differ from the frozen Prompt')
                mu = np.stack((new['k_pre_mean'][1], old['mean'][0,0]))
                cov = np.stack((new['k_pre_covariance'][1], old['covariance'][0,0]))
                e = projection_errors(mu, cov, mean[:2,layer], basis[:2,layer], ranks)
                oracle = oracle_errors(mu, cov, mean[:2,layer], ranks)
                old_indices = [list(ranks).index(int(r)) for r in new['ranks']]
                np.testing.assert_allclose(e[...,old_indices]/scale[layer,:,None],
                    new['fixed_ideal_in_domain_mse']/scale[layer,:,None],atol=1e-10,rtol=0)
                if (np.max(e[...,-1]/scale[layer])>1e-10
                        or np.max(np.diff(e,axis=-1)/scale[layer,:,None])>1e-10
                        or np.max((oracle['same_mean']-e)/scale[layer,:,None])>1e-10
                        or np.max((oracle['classic']-oracle['same_mean'])/scale[layer,:,None])>1e-10):
                    raise ValueError('nested ranks/full-rank/Oracle ordering failed')
                for target,value in zip(by_layer,(e,oracle['same_mean'],oracle['classic'],
                        np.trace(cov,axis1=-2,axis2=-1)+np.square(mu).sum(-1))):
                    target.append(value)
        curves.append(np.stack(by_layer[0],1));same.append(np.stack(by_layer[1],1))
        classic.append(np.stack(by_layer[2],1));energies.append(np.stack(by_layer[3],1))
        print(f"Full curves: {row['sample_id']}",flush=True)
    return {'sample_ids':np.asarray([r['sample_id'] for r in records]),'ranks':np.asarray(ranks),
        'raw_mse':np.stack(curves),'normalized_mse':np.stack(curves)/scale[None,None,:,:,None],
        'oracle_same_normalized':np.stack(same)/scale[None,None,:,:,None],
        'oracle_classic_normalized':np.stack(classic)/scale[None,None,:,:,None],
        'energy':np.stack(energies)}


def select_ranks(curve, rank_matrices, ranks):
    """[sample,position,layer,head,rank] -> [sample,position,method,layer,head]."""
    indices = np.searchsorted(ranks,rank_matrices)
    return np.take_along_axis(curve[:,:,None],indices[None,...,None],axis=-1)[...,0]


def sampled_errors(records, mean, basis, scale, rank_matrices, config):
    """448 matched original vectors, FP64 projection and FP32/BF16 real coding."""
    layers,heads,dimension = basis.shape[1:4]
    result = {k:np.empty((len(records),2,2,layers,heads)) for k in ('ideal','bf16')}
    result['coefficient_bytes'] = np.empty((len(records),2,2,layers,heads),dtype=np.uint64)
    floors = np.empty((len(records),2,layers,heads))
    for sample,row in enumerate(records):
        for layer in range(layers):
            raw = read_sample(row,include_vectors=True,layer_ids=[layer])
            vectors = raw['vectors'][:2,0].astype(np.float64)
            c,s = rope_coefficients(config,raw['positions'],dimension)
            target = vectors[1]
            floors[sample,0,layer] = np.square(apply_position_rope(vectors[0],c,s)-target).sum(-1).mean(-1)
            floors[sample,1,layer] = np.square(apply_position_rope(vectors[0],c,s,bf16=True)-target).sum(-1).mean(-1)
            for position,method,head in np.ndindex(2,2,heads):
                rank = int(rank_matrices[position,method,layer,head])
                x = vectors[position,head:head+1]
                mu = mean[position,layer,head:head+1]
                u = basis[position,layer,head:head+1].copy()
                ideal = ((x-mu[:,None])@u[...,:rank])@u[...,:rank].swapaxes(-1,-2)+mu[:,None]
                with torch.no_grad():
                    code = encode_key(torch.from_numpy(x),torch.from_numpy(mu),torch.from_numpy(u),rank)
                    actual = decode_key(code,torch.from_numpy(mu),torch.from_numpy(u)).float().numpy().astype(np.float64)
                if position==0:
                    ideal = apply_position_rope(ideal,c,s)
                    actual = apply_position_rope(actual,c,s,bf16=True)
                index = sample,position,method,layer,head
                result['ideal'][index] = np.square(ideal-target[head:head+1]).sum(-1).mean()
                result['bf16'][index] = np.square(actual-target[head:head+1]).sum(-1).mean()
                result['coefficient_bytes'][index] = code.numel()*code.element_size()
        print(f"BF16 samples: {row['sample_id']}",flush=True)
    result['sample_ids'] = np.asarray([r['sample_id'] for r in records])
    result['ideal'] /= scale[None,None,None]
    result['bf16'] /= scale[None,None,None]
    result['transform_only_normalized_mse'] = floors/scale[None,None]
    if not all(np.isfinite(result[k]).all() for k in ('ideal','bf16','coefficient_bytes')):
        raise ValueError('nonfinite sampled codec results')
    return result


def gain(left, right):
    left,right = np.asarray(left),np.asarray(right)
    return np.divide(left-right,left,out=np.full(np.broadcast_shapes(left.shape,right.shape),np.nan),where=left>0)


def scores(values, records):
    weights = condition_weights([r['condition'] for r in records],[r['source_group'] for r in records])
    mean = values.mean((3,4))
    p90 = np.quantile(values.reshape(*values.shape[:3],-1),.9,axis=-1)
    overall = np.einsum('s,spm->pm',weights,mean)
    tail = np.einsum('s,spm->pm',weights,p90)
    groups = sorted({r['condition'] for r in records})
    condition = {c:mean[[i for i,r in enumerate(records) if r['condition']==c]].mean(0) for c in groups}
    condition_gains = np.stack([gain(v[:,0],v[:,1]) for v in condition.values()])
    improvement = gain(overall[:,0],overall[:,1])
    p90_improvement = gain(tail[:,0],tail[:,1])
    worst = {}
    for position,name in enumerate(POSITIONS):
        worst[name] = {}
        for method,label in enumerate(METHODS):
            s,l,h = np.unravel_index(np.argmax(values[:,position,method]),values[:,position,method].shape)
            worst[name][label] = {'sample_id':records[s]['sample_id'],'layer':int(l),'head':int(h),
                                 'normalized_mse':float(values[s,position,method,l,h])}
        ratio = gain(values[:,position,0],values[:,position,1])
        valid = np.isfinite(ratio)
        if valid.any():
            s,l,h = np.unravel_index(np.nanargmin(ratio),ratio.shape)
            worst[name]['largest_relative_degradation'] = {'sample_id':records[s]['sample_id'],
                'layer':int(l),'head':int(h),'increase':float(-ratio[s,l,h]),
                'uniform_mse':float(values[s,position,0,l,h]),'head_aware_mse':float(values[s,position,1,l,h])}
    return {'mean':overall,'input_mean':mean,'p90':tail,'condition_mean':condition,
            'improvement':improvement,'p90_improvement':p90_improvement,'worst':worst,
            'degraded_head_fraction':(values[:,:,1]>values[:,:,0]).mean((0,2,3)),
            'descriptive_5pct_goal':(improvement>=.05)&(condition_gains.min(0)>=-.05)&(p90_improvement>=-.05),
            'no_confirmatory_ci':True}


def stability(calib, validation, records, ranks, primary):
    result,summary = {},{}
    i64 = list(ranks).index(primary)
    for position,name in enumerate(POSITIONS):
        references=[];pairs=[]
        for ri in range(len(ranks)-1):
            refs = fixed_reference_comparison(calib[:,position,:,:,ri],validation[:,position,:,:,ri])
            combined = np.concatenate((calib[:,position,:,:,ri],validation[:,position,:,:,ri]))
            pair = input_pair_correlations(combined)
            references.append(refs);pairs.append(pair)
        result[f'{name}_reference_per_layer_rho'] = np.stack([x['per_layer_rho'] for x in references])
        result[f'{name}_reference_per_layer_skill'] = np.stack([x['per_layer_skill'] for x in references])
        result[f'{name}_reference_valid_head_count'] = np.stack([x['per_layer_valid_count'] for x in references])
        result[f'{name}_reference_profile'] = np.stack([x['reference_profile'] for x in references])
        result[f'{name}_pair_per_layer_rho'] = np.stack([x['per_layer_rho'] for x in pairs])
        summary[name] = {'validation_layer_rho_r64':describe(references[i64]['per_layer_rho']),
                         'validation_layer_skill_r64':describe(references[i64]['per_layer_skill']),
                         'weakest_layers_r64':np.argsort(finite_mean(references[i64]['per_layer_rho'],0))[:3]}
    indices = pairs[0]['pair_indices']
    descriptions=[]
    for a,b in indices:
        left,right=records[a],records[b]
        kind = ('pg19_same_source' if left['source_group']==right['source_group'] else
                'same_condition' if left['condition']==right['condition'] else 'cross_condition')
        descriptions.append((kind,'|'.join(sorted((left['condition'],right['condition'])))))
    for name in POSITIONS:
        value=result[f'{name}_pair_per_layer_rho'][i64]
        summary[name]['pair_rho_r64']={}
        for kind in sorted({x[0] for x in descriptions}):
            groups=sorted({x[1] for x in descriptions if x[0]==kind})
            grouped=np.stack([finite_mean(value[[i for i,x in enumerate(descriptions) if x==(kind,g)]],0) for g in groups])
            summary[name]['pair_rho_r64'][kind]=describe(finite_mean(grouped,0))
    result.update(pair_indices=indices,pair_kind=np.asarray([x[0] for x in descriptions]),
                  pair_condition_group=np.asarray([x[1] for x in descriptions]),stability_ranks=np.asarray(ranks[:-1]))
    return result,summary


def export_parameters(path, mean, basis, rank_matrix, position):
    layers,heads,dimension = mean.shape
    uniform = np.all(rank_matrix==rank_matrix.flat[0])
    with atomic_file(path,binary=True) as handle:
        handle.write(struct.pack('<HHHBB',layers,heads,dimension,position,int(not uniform)))
        handle.write(np.asarray(rank_matrix.flat[0] if uniform else rank_matrix,dtype='<u2').tobytes())
        handle.write(mean.astype('<f4').tobytes())
        for l,h in np.ndindex(rank_matrix.shape):
            handle.write(basis[l,h,:,:int(rank_matrix[l,h])].astype('<f4').tobytes())


def run_analysis(bundle, codec_root, output, *, ranks=RANKS, baseline_rank=64):
    bundle,codec_root,output = map(Path,(bundle,codec_root,output))
    calib_records,validation_records=load_codec_cohorts(bundle,codec_root)
    with np.load(bundle/'rope_calibration.npz') as z:mean,basis=z['mean'],z['basis']
    with np.load(bundle/'calibration.npz') as z:scale=z['scale'][0]
    if not np.all(scale>0):raise ValueError('positive frozen K_post scale required')
    output.mkdir(parents=True,exist_ok=False)
    _save(output/'metadata.json',{'status':'running','source_bundle':str(bundle.resolve()),
        'codec_root':str(codec_root.resolve()),'calib':calib_records,'validation':validation_records,
        'positions':POSITIONS,'methods':METHODS,'ranks':ranks,'primary_rank':baseline_rank,
        'means_bases_refitted':False,'full_scope':'own-domain ideal full middle [64,T-64)',
        'sample_scope':'same 448 positions; final BF16 K_post; CPU reference',
        'request_metadata_bytes':8,'curve_axes':['sample','position','layer','head','rank'],
        'score_axes':['sample','position','method','layer','head'],'exploratory':True,'bootstrap':False})
    calib=full_curves(calib_records,codec_root/'pilot',mean,basis,scale,ranks)
    write_npz(output/'calib_curves.npz',calib)
    weights=condition_weights([r['condition'] for r in calib_records],[r['source_group'] for r in calib_records])
    costs=np.einsum('s,splhr->plhr',weights,calib['normalized_mse'])
    allocations=[allocate_budget(costs[p],baseline_rank,[r['prompt_length'] for r in calib_records],
                ranks=ranks,dimension=basis.shape[-1],request_metadata_bytes=8) for p in range(2)]
    rank_matrices=np.stack([np.stack((a['uniform_ranks'],a['strict_ranks'])) for a in allocations])
    # Publish the only candidate decisions before loading Validation moments/errors.
    _save(output/'allocation.json',{'status':'frozen_before_validation','calib_sample_ids':calib['sample_ids'],
        'costs':costs,'positions':POSITIONS,'methods':METHODS,'rank_matrices':rank_matrices,'details':allocations})
    validation=full_curves(validation_records,codec_root/'validation',mean,basis,scale,ranks)
    write_npz(output/'validation_curves.npz',validation)
    records=calib_records+validation_records
    ledger={}
    for p,name in enumerate(POSITIONS):
        ledger[name]={}
        for method,label in enumerate(METHODS):
            r=rank_matrices[p,method]
            item=byte_ledger(r,[x['prompt_length'] for x in records],dimension=basis.shape[-1],request_metadata_bytes=8)
            path=output/f'{name}_{label}_parameters.bin'
            export_parameters(path,mean[p],basis[p],r,p)
            if path.stat().st_size!=item['model_total_bytes']:raise ValueError('parameter byte mismatch')
            lh=r.size;D=basis.shape[-1];lengths=np.array([x['prompt_length'] for x in records])
            item.update(ideal_projection_flops_per_token=2*D*int(r.sum()),
                padded_projection_flops_per_token=2*D*int(r.max(1).sum()*r.shape[1]),
                rank_group_counts=[len(np.unique(x)) for x in r],
                dense_value_bytes=2*lengths*lh*D,
                kv_total_resident_bytes=item['model_total_bytes']+item['request_counts'][:,None]*(
                    item['cache_total_bytes'][None]+(2*lengths*lh*D)[None]),
                one_layer_bf16_restore_bytes=2*lengths*r.shape[1]*D,
                optional_rope_table_bytes=4*lengths*D if p==0 else np.zeros_like(lengths))
            ledger[name][label]=item
        uniform,strict=ledger[name].values()
        if strict['model_total_bytes']>uniform['model_total_bytes'] or np.any(strict['cache_total_bytes']>uniform['cache_total_bytes']):
            raise ValueError('strict policy exceeded uniform model/request byte ceiling')
    _save(output/'byte_ledger.json',ledger)
    phase_summaries={}
    for phase,data,selected in (('calib',calib,calib_records),('validation',validation,validation_records)):
        full=select_ranks(data['normalized_mse'],rank_matrices,ranks)
        sample=sampled_errors(selected,mean,basis,scale,rank_matrices,read_json(bundle/'rope_config.json'))
        for p in range(2):
            expected=2*448*rank_matrices[p].astype(np.int64)
            np.testing.assert_array_equal(sample['coefficient_bytes'][:,p],np.broadcast_to(expected,sample['coefficient_bytes'][:,p].shape))
        write_npz(output/f'{phase}_sampled.npz',sample)
        write_npz(output/f'{phase}_scores.npz',{'full_ideal_normalized_mse':full})
        phase_summaries[phase]={'full_ideal':scores(full,selected),
            'sampled_ideal_post_target':scores(sample['ideal'],selected),
            'sampled_bf16_post_target':scores(sample['bf16'],selected)}
    stable,stable_summary=stability(calib['normalized_mse'],validation['normalized_mse'],records,ranks,baseline_rank)
    oracle_stable,oracle_summary=stability(calib['oracle_same_normalized'],validation['oracle_same_normalized'],records,ranks,baseline_rank)
    stable.update({f'oracle_{k}':v for k,v in oracle_stable.items() if k not in ('pair_indices','pair_kind','pair_condition_group','stability_ranks')})
    write_npz(output/'stability.npz',stable)
    result={'status':'completed','positions':POSITIONS,'methods':METHODS,'phases':phase_summaries,
        'stability':stable_summary,'oracle_same_stability':oracle_summary,
        'exploratory':True,'historical_K_post_A3_only_reference':True,
        'full_bf16_or_decode_quality_collected':False,'inference_path_modified':False}
    _save(output/'summary.json',result)
    metadata=read_json(output/'metadata.json');metadata['status']='completed'
    write_json(output/'metadata.json',metadata,overwrite=True)
    print(f'Pre/Post Head budget analysis completed: {output}',flush=True)
    return _json_value(result)
