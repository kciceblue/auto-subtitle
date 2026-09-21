"""Pinned Voxtral offline audio capture; the caller owns GPU lifecycle/restoration.

No Qwen generation, EOS or language-parser assumptions are reused. CPU replay
rebuilds the actual native serializer/tokenizer/frontend, never acoustic weights.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import version
import inspect
import math
import os
from pathlib import Path
import re
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src import auto_source_native as geometry
from src import evidence_context as ec
from src import contextual_asr_native as base
from src.contextual_review_contract import require
from src.contextual_asr_native import (_same, _sha, _file, _read, _save, _sealed, _now, _time,
    _deadline, WorkerDeadline, tensor_record, _load_tensor, tensor_identity)
VERSION = 'voxtral-source-worker-1'
PLAN_VERSION = 'voxtral-source-plan-1'
REQUEST_VERSION = 'voxtral-source-request-1'
OBSERVATIONS_VERSION = 'voxtral-source-observations-1'
MODEL_ID = 'mistralai/Voxtral-Mini-4B-Realtime-2602'
MODEL_REVISION = '2769294da9567371363522aac9bbcfdd19447add'
MODEL_SHA256 = 'e745e4902df6a4c48f29f2f8dc1f6d0fb4cc73c7156bc45923451a5bcdfcd1d6'
MODEL_FILES = ('config.json', 'generation_config.json', 'model.safetensors', 'processor_config.json', 'tekken.json')
PACKAGES = ('torch', 'transformers', 'mistral-common', 'numpy', 'soundfile', 'soxr', 'accelerate', 'safetensors')
POLICY = {'maximum_calls':68,'control_calls':2,'real_calls':66,'model_loads':1,'retries':0,
    'local_seconds':1800,'work_seconds':1380,'cleanup_reserve_seconds':420,'recap_calls':0,
    'candidate_generated':False,'accuracy_verified':False,'language':None}
CONDITIONING = {'context':'','language':None,'hotwords':[]}
GENERATION = {'do_sample':False,'num_beams':1,'num_return_sequences':1,'use_cache':True,
    'return_dict_in_generate':True,'output_scores':False}
PROCESSING = {'sampling_rate':16000,'is_streaming':False,'is_first_audio_chunk':True,
    'return_tensors':'pt','truncation':False}
RUNTIME_POLICY = {'dtype':'torch.bfloat16','device_type':'cuda','attention_implementation':'sdpa',
    'num_delay_tokens':6,'audio_length_per_tok':8,'downsample_factor':4,'length_policy':'native_audio_duration'}


def _hash(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value), 'Invalid digest')
    return value


def _runtime_paths():
    import torch, transformers, mistral_common, soundfile, soxr
    tf=Path(transformers.__file__).parent; mc=Path(mistral_common.__file__).parent
    relative=('models/voxtral_realtime/modeling_voxtral_realtime.py','models/voxtral_realtime/configuration_voxtral_realtime.py',
        'models/voxtral_realtime/processing_voxtral_realtime.py','models/voxtral_realtime/feature_extraction_voxtral_realtime.py',
        'tokenization_mistral_common.py','processing_utils.py','feature_extraction_sequence_utils.py','audio_utils.py',
        'generation/utils.py','generation/configuration_utils.py','generation/stopping_criteria.py','generation/logits_process.py',
        'cache_utils.py','masking_utils.py','integrations/sdpa_attention.py','modeling_utils.py')
    paths=[tf/name for name in relative]+list(mc.rglob('*.py'))
    paths += [Path(torch.__file__), Path(torch.__file__).parent/'version.py',Path(torch._C.__file__),
        Path(soundfile.__file__),Path(soxr.__file__)]
    return sorted({p.resolve(strict=True) for p in paths})


def build_assets(model_dir, interpreter=None):
    """Read and pin local files/package identity, with no model construction."""
    model_dir=Path(model_dir).resolve(strict=True)
    result={'version':'voxtral-source-assets-1','model_id':MODEL_ID,'revision':MODEL_REVISION,
        'model_dir':str(model_dir),'interpreter':os.path.abspath(interpreter or sys.executable),
        'model_files':[{'path':str(model_dir/name),'sha256':_file(model_dir/name)} for name in MODEL_FILES],
        'runtime_files':[{'path':str(p),'sha256':_file(p)} for p in _runtime_paths()],
        'versions':{name:version(name) for name in PACKAGES}}
    validate_assets(result, runtime_check=True)
    return result


def execution_identity(assets):
    return {'family':'voxtral_realtime','model_id':MODEL_ID,'model_revision':MODEL_REVISION,
        'model_identity_sha256':ec._hash(assets['model_files']),
        'runtime_identity_sha256':ec._hash({'files':assets['runtime_files'],'versions':assets['versions']}),
        'tokenizer_sha256':next(p['sha256'] for p in assets['model_files'] if Path(p['path']).name=='tekken.json'),
        'worker_sha256':_file(__file__),'assets_sha256':ec._hash(assets)}


def validate_assets(assets, identity=None, *, runtime_check=False):
    ec._keys(assets,{'version','model_id','revision','model_dir','interpreter','model_files','runtime_files','versions'})
    require(assets['version']=='voxtral-source-assets-1' and assets['model_id']==MODEL_ID
        and assets['revision']==MODEL_REVISION and Path(assets['model_dir']).is_absolute()
        and Path(assets['interpreter']).is_absolute(), 'Voxtral asset identity changed')
    expected=[str(Path(assets['model_dir'])/name) for name in MODEL_FILES]
    require([p['path'] for p in assets['model_files']]==expected and assets['runtime_files'], 'Incomplete model/runtime manifest')
    for group in ('model_files','runtime_files'):
        paths=[]
        for row in assets[group]:
            ec._keys(row,{'path','sha256'});_hash(row['sha256'])
            require(Path(row['path']).is_absolute() and _file(row['path'])==row['sha256'], 'Voxtral asset bytes changed')
            paths.append(row['path'])
        require(len(paths)==len(set(paths)), 'Duplicated runtime/model pin')
    require(next(p['sha256'] for p in assets['model_files'] if Path(p['path']).name=='model.safetensors')==MODEL_SHA256,
        'Wrong native BF16 model weights')
    versions=assets['versions'];ec._keys(versions,set(PACKAGES))
    require(versions['transformers']=='5.3.0' and versions['mistral-common']=='1.11.7'
        and versions['torch'].split('+')[0]=='2.10.0', 'Pinned framework versions changed')
    if runtime_check:
        require(_same(versions,{n:version(n) for n in PACKAGES}) and
            [r['path'] for r in assets['runtime_files']]==[str(p) for p in _runtime_paths()], 'Active native runtime changed')
    if identity is not None: require(_same(identity,execution_identity(assets)), 'Native request identity changed')
    return execution_identity(assets)


def model_config(assets):
    folder=Path(assets['model_dir']);config=_read(folder/'config.json');gen=_read(folder/'generation_config.json')
    require(config['model_type']=='voxtral_realtime' and config['audio_length_per_tok']==8
        and config['default_num_delay_tokens']==6 and config['downsample_factor']==4
        and config['audio_config']['hidden_size']==1280 and config['audio_config']['num_hidden_layers']==32
        and config['text_config']['hidden_size']==3072 and config['text_config']['num_hidden_layers']==26,
        'Native model architecture changed')
    require(gen['bos_token_id']==1 and gen['eos_token_id']==2 and gen['pad_token_id']==11, 'Native protocol IDs changed')
    return {'config':config,'generation_config':gen}


def build_plan(rows, detector, mono_sha256, audio_bindings, assets, input_proof_sha256):
    owners=geometry.owner_geometry(rows,detector)
    slots=[{'slot_id':k,'kind':'control','owner_id':None} for k in ('silence','noise')]
    slots += [{'slot_id':f'owner-{i:03d}','kind':'real','owner_id':i} for i in range(1,67)]
    require(type(audio_bindings) is dict and set(audio_bindings)=={s['slot_id'] for s in slots}, 'All68 fixed audio bindings required')
    for slot in slots:
        a=audio_bindings[slot['slot_id']];ec._keys(a,{'path','sha256','pcm_sha256','frames','sample_rate','dtype'})
        expected=192000 if slot['kind']=='control' else owners[slot['owner_id']-1]['crop_end_frame']-owners[slot['owner_id']-1]['crop_start_frame']
        require(type(a['frames']) is int and a['frames']==expected and type(a['sample_rate']) is int
            and a['sample_rate']==16000 and a['dtype']=='float32' and Path(a['path']).is_absolute(), 'Voxtral audio scope changed')
        _hash(a['sha256']);_hash(a['pcm_sha256'])
    value={'version':PLAN_VERSION,'source_rows_sha256':ec._hash(rows),'original_mono_sha256':_hash(mono_sha256),
        'original_mono_frames':detector['mono']['frames'],'sample_rate':16000,'geometry':owners,
        'detector_record_sha256':ec._hash(detector),'audio_bindings':deepcopy(audio_bindings),
        'execution_identity':execution_identity(assets),'input_proof_sha256':_hash(input_proof_sha256),
        'model_config':model_config(assets),'runtime_policy':deepcopy(RUNTIME_POLICY),'policy':deepcopy(POLICY),'slots':slots}
    value['plan_sha256']=ec._hash(value);return value


def check_plan(plan, assets):
    ec._keys(plan,{'version','source_rows_sha256','original_mono_sha256','original_mono_frames','sample_rate',
        'geometry','detector_record_sha256','audio_bindings','execution_identity','input_proof_sha256',
        'model_config','runtime_policy','policy','slots','plan_sha256'})
    require(plan['version']==PLAN_VERSION and plan['plan_sha256']==ec._hash({k:v for k,v in plan.items() if k!='plan_sha256'})
        and _same(plan['execution_identity'],execution_identity(assets)) and _same(plan['model_config'],model_config(assets))
        and _same(plan['runtime_policy'],RUNTIME_POLICY) and _same(plan['policy'],POLICY), 'Voxtral plan changed')
    slots=[{'slot_id':k,'kind':'control','owner_id':None} for k in ('silence','noise')]+[
        {'slot_id':f'owner-{i:03d}','kind':'real','owner_id':i} for i in range(1,67)]
    require(_same(plan['slots'],slots) and len(plan['geometry'])==66 and set(plan['audio_bindings'])=={s['slot_id'] for s in slots}, 'Fixed68 slots changed')
    previous=0
    for i,row in enumerate(plan['geometry'],1):
        start,end=map(geometry._stamp,row['owner_ts_line'].split(' --> '))
        physical=min(end,plan['original_mono_frames']);tail=end-physical
        require(row['owner_id']==i and start==previous and row['owner_start_frame']==row['crop_start_frame']==start
            and row['owner_end_frame']==end and row['crop_end_frame']==physical and row['clamped_tail_frames']==tail
            and 0<physical-start<=480000 and (tail==0 or i==66 and 0<tail<=2)
            and plan['audio_bindings'][f'owner-{i:03d}']['frames']==physical-start, 'Original owner physical scope changed')
        previous=end
    require(plan['geometry'][-1]['crop_end_frame']==plan['original_mono_frames'], 'Incomplete mono domain')


def build_request(plan, slot_id):
    slot=next((s for s in plan['slots'] if s['slot_id']==slot_id),None);require(slot is not None,'Unknown fixed Voxtral slot')
    view={'kind':'synthetic_control','control_id':slot_id,'crop_start_frame':0,'crop_end_frame':192000,'sample_rate':16000} if slot['kind']=='control' else plan['geometry'][slot['owner_id']-1]
    return {'version':REQUEST_VERSION,'plan_sha256':plan['plan_sha256'],'slot':deepcopy(slot),
        'observation_id':'voxtral-source-'+slot_id,'view':deepcopy(view),'audio':deepcopy(plan['audio_bindings'][slot_id]),
        'execution_identity':deepcopy(plan['execution_identity']),'conditioning':deepcopy(CONDITIONING),
        'processing':deepcopy(PROCESSING),'generation':deepcopy(GENERATION),'runtime_policy':deepcopy(RUNTIME_POLICY)}


def validate_request(request, plan):
    require(_same(request,build_request(plan,request['slot']['slot_id'])),'Voxtral request changed')


_pcm=geometry._pcm
_read_audio=geometry._read_audio


def load_processor(assets):
    validate_assets(assets,runtime_check=True)
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(assets['model_dir'],local_files_only=True)
    require(type(processor).__name__=='VoxtralRealtimeProcessor','Wrong native processor')
    return processor


def _frontend(processor,audio):
    """Actual public processor plus independent native serializer/padding check."""
    import numpy as np
    from mistral_common.audio import Audio
    from mistral_common.protocol.instruct.chunk import RawAudio
    native=Audio(audio_array=np.array(audio,copy=True),sampling_rate=16000,format='wav')
    serialized=Audio.from_raw_audio(RawAudio.from_audio(native)).audio_array
    encoder=processor.tokenizer.tokenizer.instruct_tokenizer.audio_encoder
    config=processor.mistral_common_audio_config
    left,right=encoder._get_streaming_pad(len(audio))
    require(config.get_num_delay_tokens()==6 and config.n_left_pad_tokens*1280==left
        and config.raw_audio_length_per_tok==1280 and config.audio_length_per_tok==8,
        'Native padding/delay clock changed')
    expected=np.pad(serialized,(left,right));calls=[];extractor=processor.feature_extractor
    class CaptureExtractor:
        def __getattr__(self,key):return getattr(extractor,key)
        def __call__(self,values,**kwargs):
            require(len(calls)==0 and isinstance(values,list) and len(values)==1 and kwargs.get('center') is True
                and kwargs.get('truncation') is False,'Frontend whole-wave call changed')
            actual=np.ascontiguousarray(values[0]);require(actual.dtype==np.float32 and np.array_equal(actual,expected),'Native serializer/padding waveform changed')
            calls.append(np.array(actual,copy=True));return extractor(values,**kwargs)
    processor.feature_extractor=CaptureExtractor()
    try: inputs=processor(audio,**PROCESSING)
    finally:processor.feature_extractor=extractor
    require(len(calls)==1 and set(inputs)=={'input_ids','attention_mask','input_features','num_delay_tokens'}
        and inputs['num_delay_tokens']==6,'Native frontend output fields changed')
    frames=len(expected)//160;ids=inputs['input_ids'];features=inputs['input_features'];mask=inputs['attention_mask']
    require(tuple(features.shape)==(1,128,frames) and frames%8==0 and tuple(ids.shape)==tuple(mask.shape)
        and ids.shape[0]==1 and ids.shape[1]>0 and bool((mask==1).all()),'Native frontend truncated or batched')
    return inputs, {'serialized_pcm':np.ascontiguousarray(serialized),'padded_pcm':calls[0],
        'left_pad_samples':left,'right_pad_samples':right,'physical_start_sample':left,
        'physical_end_sample':left+len(audio),'feature_frames':frames,'native_total_positions':frames//8,
        'prefix_positions':int(ids.shape[1]),'num_delay_tokens':6,'audio_samples_per_position':1280}


def _record(value, folder):
    import torch
    source_dtype=str(value.dtype)
    if isinstance(value,torch.Tensor) and value.dtype==torch.bfloat16:value=value.float()
    return {'source_dtype':source_dtype,'tensor':tensor_record(value,folder)}


def _load(record, folder):
    ec._keys(record,{'source_dtype','tensor'})
    require(record['source_dtype'] in {'torch.bfloat16','torch.float32','torch.int64','torch.bool','float32','int64'},'Unsupported tensor dtype')
    return _load_tensor(record['tensor'],folder)


def _id(value):return ec._hash({'source_dtype':value['source_dtype'],'tensor':{k:value['tensor'][k] for k in ('dtype','shape','bytes_sha256')}})


def _generation_state(config):
    return {k:deepcopy(getattr(config,k,None)) for k in ('do_sample','num_beams','num_return_sequences','use_cache',
        'max_length','max_new_tokens','bos_token_id','eos_token_id','pad_token_id','return_dict_in_generate','output_scores')}


def runtime_identity(model,processor):
    return {'model_class':type(model).__name__,'processor_class':type(processor).__name__,
        'device':str(model.device),'dtype':str(model.dtype),'audio_attention':model.config.audio_config._attn_implementation,
        'text_attention':model.config.text_config._attn_implementation,'audio_layers':len(model.audio_tower.layers),
        'text_layers':model.config.text_config.num_hidden_layers,'audio_hidden_size':model.config.audio_config.hidden_size,'num_delay_tokens':processor.num_delay_tokens,
        'generation_config':model.generation_config.to_dict()}


def _validate_runtime(value):
    require(value['model_class']=='VoxtralRealtimeForConditionalGeneration' and value['processor_class']=='VoxtralRealtimeProcessor'
        and value['device'].startswith('cuda') and value['dtype']=='torch.bfloat16'
        and value['audio_attention']==value['text_attention']=='sdpa' and value['audio_layers']==32
        and value['text_layers']==26 and value['audio_hidden_size']==1280 and value['num_delay_tokens']==6, 'Actual Voxtral runtime changed')
    return value


def _runtime_check(model,processor):
    return _validate_runtime(runtime_identity(model,processor))


def decode_tokens(processor,ids):
    from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
    tokenizer=processor.tokenizer.tokenizer
    raw=tokenizer.decode(ids,special_token_policy=SpecialTokenPolicy.KEEP)
    parsed=tokenizer.decode(ids,special_token_policy=SpecialTokenPolicy.IGNORE)
    # No HF convenience regex removing ordinary lang:xx, and no whitespace cleanup.
    is_special=tokenizer.instruct_tokenizer.tokenizer.is_special
    groups=[]
    for value in ids:
        special=is_special(value)
        if groups and groups[-1][0]==special:groups[-1][1].append(value)
        else:groups.append((special,[value]))
    raw_parts=[];ordinary=[];insertions=[];offset=0
    for special,tokens in groups:
        text=tokenizer.decode(tokens,special_token_policy=SpecialTokenPolicy.KEEP);raw_parts.append(text)
        if special:insertions.append({'parsed_offset':offset,'special_text':text,'token_ids':tokens})
        else:ordinary.append(text);offset+=len(text)
    if ''.join(raw_parts)!=raw or ''.join(ordinary)!=parsed:insertions=None
    return {'raw_text':raw,'parsed_text':parsed,'special_token_insertions':insertions}


def _coverage(capture, folder):
    import numpy as np
    layout=capture['layout'];full=_load(capture['audio_embeddings'],folder);calls=capture['audio_calls']
    require(full.ndim==3 and full.shape[0]==1 and full.shape[1]==layout['feature_frames']//2 and full.shape[2]==capture['runtime']['audio_hidden_size']
        and calls and capture['embedder_calls']==1 and capture['generation_calls']==1,'Missing full native audio embedding')
    previous=0
    for number,call in enumerate(calls):
        ec._keys(call,{'position_start','position_end','audio_start','audio_end','slice_sha256','encoder_completed','layer_calls'})
        start,end=call['position_start'],call['position_end']
        require(type(start) is int and type(end) is int and start==previous and end>start
            and (end==layout['prefix_positions'] if number==0 else end-start==1)
            and call['audio_start']==start*4 and call['audio_end']==end*4
            and end*4<=full.shape[1] and call['slice_sha256']==_sha(np.ascontiguousarray(full[:,start*4:end*4]).tobytes())
            and call['encoder_completed'] is True and call['layer_calls']==capture['runtime']['audio_layers'],
            'Actual audio consumption ranges changed')
        previous=end
    ids=capture['output_token_ids'];prefix=capture['prefix_token_ids'];N=layout['native_total_positions']
    require(type(ids) is list and all(type(v) is int and v>=0 for v in ids) and ids[:len(prefix)]==prefix
        and len(ids)>len(prefix) and len(ids)<=N and previous==len(ids)-1, 'Native output/position cardinality changed')
    generated=ids[len(prefix):];eos=capture['resolved_generation']['eos_token_id']
    require(eos==2 and (eos not in generated or generated.index(eos)==len(generated)-1), 'EOS before final generated position')
    stopped_eos=generated[-1]==eos;duration_end=len(ids)==N
    require(stopped_eos or duration_end,'Native generation stopped without EOS or audio clock completion')
    consumed_samples=previous*1280
    complete=consumed_samples>=layout['physical_end_sample']
    require(complete,'Native EOS stopped before complete physical audio consumption')
    return {'stop_reason':'eos' if stopped_eos else 'audio_duration','total_positions':N,
        'output_positions':len(ids),'generated_positions':len(generated),'audio_forward_positions':previous,
        'audio_embedding_positions':previous*4,'conservative_consumed_padded_samples':consumed_samples,
        'physical_end_sample':layout['physical_end_sample'],'physical_audio_complete':True}


def capture_call(model,processor,audio,request,*,artifact_root,save_capture,deadline):
    import numpy as np
    import torch
    require(request.get('version')==REQUEST_VERSION and _same(request['conditioning'],CONDITIONING)
        and _same(request['processing'],PROCESSING) and _same(request['generation'],GENERATION)
        and _same(request['runtime_policy'],RUNTIME_POLICY),'Voxtral capture policy changed')
    audio=_pcm(audio);require(len(audio)==request['audio']['frames'] and _sha(audio.tobytes())==request['audio']['pcm_sha256'],'Bound PCM changed')
    runtime=_runtime_check(model,processor);model.eval();before=deepcopy(model.generation_config.to_dict())
    capture={'runtime':runtime,'generation_calls':0,'embedder_calls':0,'audio_calls':[],
        'input_pcm':_record(audio,artifact_root),'processor_inputs':{},'generation_inputs':{},'layout':None,
        'audio_embeddings':None,'resolved_generation':None,'generation_kwargs':deepcopy(GENERATION),
        'frontend_finished_utc':None,'generation_started_utc':None,'generation_finished_utc':None,
        'prefix_token_ids':None,'output_token_ids':None,'raw_text':None,'parsed_text':None,'special_token_insertions':None,'coverage':None}
    def persist():save_capture(deepcopy(capture))
    persist()
    inputs,layout=_frontend(processor,audio)
    for key in ('serialized_pcm','padded_pcm'):layout[key]=_record(layout[key],artifact_root)
    capture['layout']=layout;capture['processor_inputs']={k:_record(v,artifact_root) for k,v in inputs.items() if k!='num_delay_tokens'}
    capture['prefix_token_ids']=inputs['input_ids'][0].tolist()
    device_inputs=inputs.to(model.device,dtype=model.dtype)
    capture['generation_inputs']={k:_record(v,artifact_root) for k,v in device_inputs.items() if k!='num_delay_tokens'}
    capture['frontend_finished_utc']=_now();persist()
    handles=[];originals={};pending=[]
    def patch_method(name,wrapper):
        originals[name]=(name in model.__dict__,model.__dict__.get(name));setattr(model,name,wrapper)
    original_length=model._prepare_generated_length
    def length(*args,**kwargs):
        result=original_length(*args,**kwargs);capture['resolved_generation']=_generation_state(result)
        require(result.max_length==layout['native_total_positions'] and result.max_new_tokens is None
            and result.eos_token_id==2 and result.pad_token_id==11,'Native duration/protocol changed')
        return result
    def embedding(module,args,kwargs,output):
        capture['embedder_calls']+=1;require(capture['embedder_calls']==1,'Repeated whole audio embedder')
        require(_id(_record(args[0],artifact_root))==_id(capture['generation_inputs']['input_features']), 'Actual frontend embedding input changed')
        capture['audio_embeddings']=_record(output,artifact_root)
    def forward(module,args,kwargs):
        require(int(kwargs.get('num_delay_tokens',-1))==6,'Actual delay conditioning changed')
        positions=kwargs.get('cache_position');value=kwargs.get('encoder_inputs_embeds')
        require(positions is not None and value is not None,'Audio positions or native embedding slice missing')
        ids=positions.detach().cpu().tolist();require(ids==list(range(ids[0],ids[-1]+1)),'Noncontiguous generation positions')
        array=value.detach().float().cpu().contiguous().numpy() if value.dtype==torch.bfloat16 else value.detach().cpu().contiguous().numpy()
        row={'position_start':ids[0],'position_end':ids[-1]+1,'audio_start':ids[0]*4,'audio_end':(ids[-1]+1)*4,
            'slice_sha256':_sha(array.tobytes()),'encoder_completed':False,'layer_calls':0}
        require(not pending,'Nested audio forward');capture['audio_calls'].append(row);pending.append(row)
    def layer(module,args,kwargs,output):
        require(len(pending)==1,'Audio layer outside native forward');pending[0]['layer_calls']+=1
    def encoder_done(module,args,kwargs,output):
        require(len(pending)==1 and kwargs.get('inputs_embeds') is not None,'Audio encoder scope missing')
        row=pending.pop();value=kwargs['inputs_embeds'];array=value.detach().float().cpu().contiguous().numpy() if value.dtype==torch.bfloat16 else value.detach().cpu().contiguous().numpy()
        require(_sha(array.tobytes())==row['slice_sha256'] and output.last_hidden_state.shape[1]==row['audio_end']-row['audio_start'], 'Audio tower consumed different embedding slice')
        row['encoder_completed']=True
    try:
        handles.append(model.audio_tower.embedder.register_forward_hook(embedding,with_kwargs=True))
        handles.append(model.register_forward_pre_hook(forward,with_kwargs=True))
        handles.append(model.audio_tower.register_forward_hook(encoder_done,with_kwargs=True))
        for module in model.audio_tower.layers:handles.append(module.register_forward_hook(layer,with_kwargs=True))
        patch_method('_prepare_generated_length',length)
        with _deadline(deadline),torch.inference_mode():
            capture['generation_calls']=1;capture['generation_started_utc']=_now();persist()
            result=model.generate(**device_inputs,**GENERATION)
            capture['generation_finished_utc']=_now()
            output=result.sequences
            require(output.ndim==2 and output.shape[0]==1,'Native output batch changed')
            capture['output_token_ids']=output[0].detach().cpu().tolist()
            capture.update(decode_tokens(processor,capture['output_token_ids']))
            persist() # preserve genuine output before any completion/content gate
            capture['coverage']=_coverage(capture,artifact_root);persist()
    finally:
        for handle in reversed(handles):handle.remove()
        for name,(had,value) in originals.items():
            if had:setattr(model,name,value)
            else:delattr(model,name)
        if capture['generation_started_utc'] and capture['generation_finished_utc'] is None:capture['generation_finished_utc']=_now()
        persist()
    require(_same(before,model.generation_config.to_dict()),'Native generation defaults mutated')
    return capture


def validate_native_receipt(receipt, request, plan, *, assets, artifact_root, processor):
    import numpy as np
    import torch
    _sealed(receipt);validate_request(request,plan);validate_assets(assets,request['execution_identity'])
    ec._keys(receipt,{'version','request','request_sha256','assets_sha256','started_utc','finished_utc','seconds',
        'status','error_type','capture','receipt_sha256'})
    require(receipt['version']==VERSION and _same(receipt['request'],request) and receipt['request_sha256']==ec._hash(request)
        and receipt['assets_sha256']==ec._hash(assets) and receipt['status'] in ('complete','empty') and receipt['error_type'] is None
        and type(receipt['seconds']) in (float,int) and math.isfinite(receipt['seconds']) and receipt['seconds']>=0,'Native receipt failed or changed')
    c=receipt['capture'];audio=_read_audio(request['audio']);require(c['input_pcm']['source_dtype']=='float32' and np.array_equal(_load(c['input_pcm'],artifact_root),audio),'Native input tensor changed')
    inputs,layout=_frontend(processor,audio)
    for key in ('serialized_pcm','padded_pcm'):
        require(np.array_equal(_load(c['layout'][key],artifact_root),layout[key]),'Native padded PCM changed')
        layout[key]=c['layout'][key]
    require(_same(c['layout'],layout),'Native padding/length layout changed')
    require(c['prefix_token_ids']==inputs['input_ids'][0].tolist() and set(c['processor_inputs'])==set(c['generation_inputs'])=={'input_ids','attention_mask','input_features'},'Native tensor fields changed')
    for key in c['processor_inputs']:
        expected_source='torch.float32' if key=='input_features' else 'torch.int64'
        expected_delivered='torch.bfloat16' if key=='input_features' else 'torch.int64'
        require(c['processor_inputs'][key]['source_dtype']==expected_source and c['generation_inputs'][key]['source_dtype']==expected_delivered,'Native tensor delivery dtype changed')
        original=inputs[key].detach().cpu().numpy();actual=_load(c['processor_inputs'][key],artifact_root)
        require(np.array_equal(actual,original),'Actual native frontend tensor changed')
        cast=inputs[key].to(dtype=torch.bfloat16).float().numpy() if inputs[key].is_floating_point() else original
        require(np.array_equal(_load(c['generation_inputs'][key],artifact_root),cast),'Actual native generation tensor changed')
    _validate_runtime(c['runtime'])
    from transformers import GenerationConfig
    expected_defaults=GenerationConfig.from_dict(plan['model_config']['generation_config']).to_dict()
    require(_same(c['runtime']['generation_config'],expected_defaults),'Native generation defaults changed')
    resolved=c['resolved_generation'];require(_same(c['generation_kwargs'],GENERATION)
        and resolved['max_length']==layout['native_total_positions'] and resolved['max_new_tokens'] is None
        and resolved['eos_token_id']==2 and resolved['pad_token_id']==11 and resolved['bos_token_id']==1
        and all(_same(resolved[k],v) for k,v in GENERATION.items()),'Actual generation settings changed')
    require(_time(receipt['started_utc'])<=_time(c['frontend_finished_utc'])<=_time(c['generation_started_utc'])
        <=_time(c['generation_finished_utc'])<=_time(receipt['finished_utc']),'Native timestamps changed')
    decoded=decode_tokens(processor,c['output_token_ids'])
    require(_same({k:c[k] for k in decoded},decoded),'Native token decoding or special-only projection changed')
    coverage=_coverage(c,artifact_root);require(_same(coverage,c['coverage']),'Native completion proof changed')
    require(receipt['status']==('empty' if c['parsed_text']=='' else 'complete'),'Empty native output status changed')
    return {'raw_text':c['raw_text'],'parsed_text':c['parsed_text'],'special_token_insertions':c['special_token_insertions'],'native_language':None,
        'audio_features_sha256':_id(c['processor_inputs']['input_features']),
        'audio_mask_sha256':None,'coverage':coverage,'protocol_category':'special_tokens_only' if c['parsed_text']=='' else 'nonempty_transcript'}


def control_result(output):
    return {'passed':output['parsed_text']=='','parsed_characters':len(output['parsed_text']),
        'raw_characters':len(output['raw_text']),'physical_audio_complete':output['coverage']['physical_audio_complete']}


def cuda_memory(*,reset=False):
    from src.long_source_native import cuda_memory as measure
    return measure(reset=reset)


def _load_model(assets):
    import torch
    from transformers import VoxtralRealtimeForConditionalGeneration
    model=VoxtralRealtimeForConditionalGeneration.from_pretrained(assets['model_dir'],local_files_only=True,
        dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
    return model,load_processor(assets)


def run_slots(plan,assets,folder,model,processor,audios,*,deadline,save_state):
    check_plan(plan,assets);folder=Path(folder)
    slots=[{**s,'number':i,'status':'undispatched','receipt':None} for i,s in enumerate(plan['slots'],1)]
    summary={'version':VERSION,'plan_sha256':plan['plan_sha256'],'status':'running','slots':slots,
        'asr_calls':0,'real_audio_calls':0,'controls':{},'error_type':None};save_state(deepcopy(summary))
    for slot in slots:
        request=build_request(plan,slot['slot_id']);path=folder/'receipts'/f'{slot["number"]:02d}-{slot["slot_id"]}.json'
        path.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic();require(started<deadline,'Work budget exhausted')
        with path.with_suffix('.started.json').open('x',encoding='utf-8') as stream:
            import json
            json.dump({'version':VERSION,'request_sha256':ec._hash(request),'started_utc':_now()},stream)
        require(not path.exists(),'No native reroll allowed')
        receipt={'version':VERSION,'request':request,'request_sha256':ec._hash(request),'assets_sha256':ec._hash(assets),
            'started_utc':_now(),'finished_utc':None,'seconds':0.,'status':'running','error_type':None,'capture':{}}
        slot.update(status='started',receipt=str(path));summary['asr_calls']+=1
        if slot['kind']=='real':summary['real_audio_calls']+=1
        def save_capture(value):receipt['capture']=value;_save(path,receipt)
        _save(path,receipt);save_state(deepcopy(summary));failure=None
        try:
            capture_call(model,processor,audios[slot['slot_id']],request,artifact_root=folder,save_capture=save_capture,deadline=deadline)
            receipt.update(status='empty' if receipt['capture']['parsed_text']=='' else 'complete',finished_utc=_now(),seconds=time.monotonic()-started)
            _save(path,receipt)
            output=validate_native_receipt(receipt,request,plan,assets=assets,artifact_root=folder,processor=processor)
            if slot['kind']=='control':summary['controls'][slot['slot_id']]=control_result(output)
        except BaseException as exc:failure=exc;receipt.update(status='failed',error_type=type(exc).__name__)
        finally:
            receipt.update(finished_utc=_now(),seconds=time.monotonic()-started);_save(path,receipt);slot['status']=receipt['status'];save_state(deepcopy(summary))
        if failure is not None:summary.update(status='failed',error_type=type(failure).__name__);break
        if slot['kind']=='control' and not summary['controls'][slot['slot_id']]['passed']:
            summary['status']='negative_control_output';break
    else:summary['status']='complete'
    save_state(deepcopy(summary));return summary


def run_worker(spec_path):
    started=time.monotonic();spec=_read(spec_path);ec._keys(spec,base._SPEC_KEYS)
    require(spec['version']==VERSION and type(spec['worker_seconds']) in (float,int) and math.isfinite(spec['worker_seconds'])
        and 0<spec['worker_seconds']<=1380,'Invalid worker budget')
    deadline=started+min(spec['worker_seconds'],(_time(spec['work_deadline_utc'])-datetime.now(timezone.utc)).total_seconds())
    folder=Path(spec['output_dir']);require(folder.is_absolute(),'Absolute worker output required')
    folder.mkdir(parents=True,exist_ok=True);require(not any(folder.iterdir()),'Worker already attempted')
    state={'version':VERSION,'status':'prepared','started_utc':_now(),'finished_utc':None,'seconds':0.,'asr_calls':0,
        'model_loads':0,'model_load_attempts':0,'error_type':None,'spec_sha256':_file(spec_path),
        'cuda_memory_before_load':None,'cuda_memory_after_load':None,'cuda_memory_final':None,'cuda_memory_error_type':None}
    def save(value):state.update(value);state['seconds']=time.monotonic()-started;_save(folder/'worker-result.json',state)
    save({});memory_started=False
    try:
        with _deadline(deadline):
            require(_file(__file__)==spec['worker_sha256'] and _file(spec['plan_path'])==spec['plan_file_sha256'],'Worker/plan pin changed')
            require(os.path.abspath(sys.executable)==spec['assets']['interpreter'],'Wrong isolated worker interpreter')
            plan=_read(spec['plan_path']);check_plan(plan,spec['assets']);validate_assets(spec['assets'],runtime_check=True)
            for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):os.environ[key]='1'
            audios={s['slot_id']:_read_audio(build_request(plan,s['slot_id'])['audio']) for s in plan['slots']}
            memory_started=True;save({'cuda_memory_before_load':cuda_memory(reset=True)})
            save({'model_load_attempts':1,'model_load_started_utc':_now()});model,processor=_load_model(spec['assets'])
            save({'cuda_memory_after_load':cuda_memory(),'model_loads':1,'model_load_finished_utc':_now(),
                'runtime_before':_runtime_check(model,processor)})
            save(run_slots(plan,spec['assets'],folder,model,processor,audios,deadline=deadline,save_state=save))
            after=_runtime_check(model,processor);save({'runtime_after':after})
            require(_same(state['runtime_before'],after),'Runtime changed during owned load')
    except BaseException as exc:save({'status':'failed','error_type':type(exc).__name__})
    finally:
        if memory_started:
            try:state['cuda_memory_final']=cuda_memory()
            except BaseException as exc:state.update(status='failed',error_type=state['error_type'] or type(exc).__name__,cuda_memory_error_type=type(exc).__name__)
        state['finished_utc']=_now();save({})
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',type=Path,required=True)
    state=run_worker(parser.parse_args().spec)
    print({k:state.get(k) for k in ('status','asr_calls','model_loads','seconds','error_type')})
    return 1 if state['status']=='failed' else 0
if __name__=='__main__':raise SystemExit(main())
