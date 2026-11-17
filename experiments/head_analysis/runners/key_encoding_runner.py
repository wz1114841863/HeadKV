"""V3 Pre-RoPE codec validation, using the existing unmodified Dense replay."""
import argparse
from pathlib import Path
import struct
import time

import numpy as np
import torch

from experiments.head_analysis.analysis.kv_representation import _json_value
from experiments.head_analysis.collectors.key_encoding import KeyEncodingCollector
from experiments.head_analysis.common.io import read_json, write_json, write_npz, atomic_file
from experiments.head_analysis.runners.b2_runner import (
    load_b1_replay_index, load_replay_samples, _validate_execution_contract, teacher_forced_replay,
)
from experiments.head_analysis.runners.kv_representation_runner import replay_config
from experiments.head_analysis.runners.input_selection import frozen_length
from experiments.head_analysis.runners.resource_metrics import GenerationPhaseTimer, validate_standard_greedy_timing
from experiments.head_analysis.runtime.attention_probe import AttentionProbe
from experiments.head_analysis.runtime.key_codec import key_byte_ledger
from experiments.head_analysis.runtime.model import load_model


PILOT_IDS = ('ruler_8k_001','ruler_32k_001','govreport_8k_001','govreport_16k_001',
             'pg19_001_8k','pg19_001_32k')
VALIDATION_IDS = ('ruler_8k_007','ruler_32k_003','govreport_8k_003','govreport_16k_003',
                  'pg19_013_8k','pg19_013_32k')


def load_parameters(bundle):
    with np.load(Path(bundle)/'rope_calibration.npz') as z:
        mean, basis = z['mean'], z['basis']
    with np.load(Path(bundle)/'calibration.npz') as z:
        scale = z['scale'][0]
    if not np.isfinite(mean).all() or not np.isfinite(basis).all() or not np.all(scale>0):
        raise ValueError('nonfinite parameters or invalid frozen K_post scale')
    if np.max(np.abs(basis.swapaxes(-1,-2)@basis-np.eye(basis.shape[-1])))>1e-10:
        raise ValueError('frozen basis is not orthonormal')
    return mean, basis, scale


def collect_sample(model, sample, directory, parameters, *, device, block_size=1024,
                   rank=64, serialize_payload=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    config = replay_config(sample, block_size)
    consumed = config.decode_steps[-1]+1
    length = frozen_length(sample.frozen_record)
    mean, basis, scale = parameters
    metadata = {'status':'running','sample_id':sample.frozen_record['sample_id'],
                'prompt_length':length,'decode_forward_count':consumed, 'rank':rank,
                'paths':['pre','post'],'ranks':[rank,basis.shape[-1]],
                'precisions':['ideal','BF16'],'error_axes':['path','rank','precision','layer','head'],
                'decode_axes':['step','path','rank','precision','layer','head'],
                'coefficient_axes':['step','path','head','rank'],
                'protected_prompt_tokens':128,'teacher_forced_dense_replay':True,
                'source_batch':sample.batch_id,'source_run':sample.run_id,
                'codec_temporary_cpu_pre_scope':'one layer only, released after Attention callback'}
    write_json(directory/'metadata.json', metadata)
    payload = None
    if serialize_payload:
        payload = (directory/'temporary_prompt_key.bin').open('xb')
        payload.write(struct.pack('<II',length,0))

    def on_payload(protected, codes):
        for a in (protected, codes):
            payload.write(a.contiguous().view(torch.uint16).numpy().tobytes())

    collector = KeyEncodingCollector(length, *parameters,
        lambda relative, arrays: write_npz(directory/relative, arrays),
        block_size=block_size, rank=rank, checkpoints=config.decode_steps,
        on_payload=on_payload if payload else None,
        rotary_spec=(model.model.rotary_emb.inv_freq.detach(),model.model.rotary_emb.attention_scaling))
    timer = GenerationPhaseTimer(model, device)
    started = time.perf_counter()
    try:
        if device.type=='cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        with timer, AttentionProbe(sample_id=sample.frozen_record['sample_id'], prompt_length=length,
                                  on_key_projection=collector.on_key_projection,
                                  on_attention=collector.on_attention)(model):
            replay = teacher_forced_replay(model, device=device, sample_id=sample.frozen_record['sample_id'],
                prompt_ids=sample.frozen_record['input_ids'],
                replay_generated_ids=sample.generated_ids[:consumed+1], require_top1_match=True)
        timing = timer.summary()
        validate_standard_greedy_timing(timing, consumed+1)
        result = collector.finish(consumed)
        ledger = key_byte_ledger(length, consumed, mean.shape[1], mean.shape[2], mean.shape[3], rank)
        if payload is not None:
            payload.close()
            observed = (directory/'temporary_prompt_key.bin').stat().st_size
            expected = key_byte_ledger(length,0,mean.shape[1],mean.shape[2],mean.shape[3],rank)['encoded_key_bytes']
            if observed!=expected:
                raise ValueError(f'actual serialized Key bytes {observed} != budget {expected}')
            result['serialized_prompt_bytes_verified'] = observed
            (directory/'temporary_prompt_key.bin').unlink()
        result.update(byte_ledger=ledger, forward_timing=timing,
                      replay_wall_seconds=time.perf_counter()-started, top1_match=True)
        if device.type=='cuda':
            result.update(shadow_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          shadow_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        for path,name in enumerate(('pre','post')):
            v = collector.prompt_errors
            values = np.stack([v[l][path,0,1] for l in range(len(v))])/scale
            l,h = np.unravel_index(np.argmax(values),values.shape)
            result[f'{name}_worst_head'] = {'layer':int(l),'head':int(h),'normalized_mse':float(values[l,h])}
        write_json(directory/'summary.json', _json_value(result))
        write_json(directory/'replay.json', {'prefill_prediction':replay.prefill_predicted_token_id,
                    'decode_predictions':list(replay.decode_predicted_token_ids)})
        metadata.update(status='completed', top1_match=True)
        write_json(directory/'metadata.json',metadata,overwrite=True)
        return _json_value(result)
    except Exception as exc:
        metadata.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        write_json(directory/'metadata.json',metadata,overwrite=True)
        raise
    finally:
        if payload is not None:
            payload.close()


def run(args):
    bundle, output = Path(args.bundle), Path(args.run_dir)
    ids = PILOT_IDS if args.cohort=='pilot' else VALIDATION_IDS
    records = read_json(bundle/'split.json')['samples']
    by_id = {r['sample_id']:r for r in records}
    cohort = 'calib' if args.cohort=='pilot' else 'validation'
    if any(by_id[s]['split']!=cohort for s in ids):
        raise ValueError('requested samples differ from the frozen V3 split')
    if args.cohort=='validation':
        if args.pilot_dir is None:
            raise ValueError('Validation requires --pilot-dir with a completed six-sample Pilot')
        pilot = read_json(Path(args.pilot_dir)/'run_metadata.json')
        if (pilot['status']!='completed' or pilot['completed_sample_ids']!=list(PILOT_IDS)
                or pilot['source_bundle']!=str(bundle.resolve())):
            raise ValueError('Pilot is incomplete or uses a different source bundle')
    parameters = load_parameters(bundle)
    if parameters[1].shape!=(3,32,8,128,128):
        raise ValueError('real-model run requires the frozen 32-layer/8-KV-Head/128-dimensional parameters')
    index = load_b1_replay_index(args.run_set,source_root=args.source_root,
                               expected_run_set_sha256=None,expected_data_sha256=None)
    _validate_execution_contract(index,args)
    samples = load_replay_samples(index,ids)
    for sample in samples:
        replay_config(sample,args.block_size)
    output.mkdir(parents=True,exist_ok=False)
    state = {'status':'running','cohort':args.cohort,'source_bundle':str(bundle.resolve()),
             'source_b1_run_set':str(Path(args.run_set).resolve()),'model':args.model,
             'device':args.device,'dtype':args.dtype,'target_sample_ids':list(ids),
             'completed_sample_ids':[],'parameters_refitted':False,'attention_backend':'sdpa',
             'torch_version':torch.__version__,
             'float32_matmul_precision':torch.get_float32_matmul_precision(),
             'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32}
    write_json(output/'run_metadata.json',state)
    # Store only deployable r64 parameters; r128 is a diagnostic reference, not another Cache candidate.
    for path,name in enumerate(('pre','post')):
        with atomic_file(output/f'{name}_parameters.bin',binary=True) as handle:
            handle.write(struct.pack('<HHHBBH',32,8,128,path,0,64))
            handle.write(parameters[0][path].astype('<f4').tobytes())
            handle.write(parameters[1][path,...,:64].astype('<f4').tobytes())
        expected = key_byte_ledger(8192,0,32,8)['model_bytes']
        if (output/f'{name}_parameters.bin').stat().st_size!=expected:
            raise ValueError('serialized parameter bytes differ from ledger')
    try:
        model, device = load_model(args.model,args.device,args.dtype)
        summaries = []
        for sample in samples:
            sid = sample.frozen_record['sample_id']
            print(f'Key codec sample started: {sid}',flush=True)
            summaries.append(collect_sample(model,sample,output/'samples'/sid,parameters,
                device=device,block_size=args.block_size,serialize_payload='32k' in sid))
            state['completed_sample_ids'].append(sid)
            write_json(output/'run_metadata.json',state,overwrite=True)
            print(f'Key codec sample completed: {sid} ({len(summaries)}/6)',flush=True)
        means = np.array([s['middle_normalized_mean'] for s in summaries])
        write_json(output/'summary.json',_json_value({
            'scope':'six conditions, one input each; PG19 paired; no bootstrap or quality claims',
            'sample_ids':list(ids),'middle_normalized_equal_condition_mean':means.mean(0),
            'pre_vs_post_bf16_improvement':1-means[:,0,0,1].mean()/means[:,1,0,1].mean(),
            'condition_results':{by_id[sid]['condition']:s for sid,s in zip(ids,summaries)}}))
        state['status']='completed'
        write_json(output/'run_metadata.json',state,overwrite=True)
        print(f'Key codec {args.cohort} completed: {output}',flush=True)
    except Exception as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        write_json(output/'run_metadata.json',state,overwrite=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('bundle','run-set','model','tokenizer','run-dir'):
        parser.add_argument('--'+option,required=True)
    parser.add_argument('--cohort',choices=('pilot','validation'),default='pilot')
    parser.add_argument('--pilot-dir')
    parser.add_argument('--source-root',default='.')
    parser.add_argument('--device',default='cuda:4')
    parser.add_argument('--dtype',choices=('bfloat16',),default='bfloat16')
    parser.add_argument('--block-size',type=int,default=1024)
    args = parser.parse_args(argv)
    if args.block_size<1:
        parser.error('--block-size must be positive')
    run(args)


if __name__=='__main__':
    main()
