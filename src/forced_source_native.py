"""Nine fixed language-conditioned views; automatic source evidence is immutable."""
from __future__ import annotations
import argparse
from copy import deepcopy
from collections import Counter
from datetime import datetime,timezone
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src import contextual_asr_native as base
from src import auto_source_native as automatic
from src import evidence_context as ec
from src.contextual_review_contract import require
from src.contextual_asr_native import (_apply_policy,_same,_sha,_CAPTURE_KEYS,tensor_record,_TENSOR_KEYS,_config,
    _array,_now,_deadline,WorkerDeadline,_sealed,_load_tensor,_describe,_GEN_KEYS,_time,tensor_identity,_file,_read,_save,native_parse)
from src.auto_source_native import _pcm,_read_audio
VERSION='forced-source-worker-1'
REQUEST_VERSION='forced-source-request-1'
PLAN_VERSION='forced-source-plan-1'
OBSERVATIONS_VERSION='forced-source-observations-1'
OWNERS=(3,4,5,6,11,44,45,59,64)
POLICY={'language':'Japanese','maximum_calls':9,'model_loads':1,'retries':0,'local_seconds':600,
        'work_seconds':240,'cleanup_reserve_seconds':360,'control_calls':0,'recap_calls':0,
        'candidate_generated':False,'accuracy_verified':False}
CONDITIONING={'context':'','language':'Japanese','hotwords':[]}
ELIGIBILITY={'proposed_condition':'nonempty auto transcript, non-Japanese label, positive saved VAD overlap',
    'eligible_owner_ids':list(OWNERS),'eligible_count':9,'new_inference_calls':0,
    'selected_for_execution':False,'source_accuracy_verified':False}


def execution_identity(assets):
    value=base.execution_identity(assets);value['worker_sha256']=_file(__file__);return value


def validate_assets(assets,identity,*,runtime_check=False):
    require(_same(identity,execution_identity(assets)),'Forced-source worker identity changed')
    return base.validate_assets(assets,base.execution_identity(assets),runtime_check=runtime_check)


def eligible_owner_ids(envelope):
    rows=envelope['observations'];require(type(rows) is list and len(rows)==66,'Complete66 automatic records required')
    require(envelope['observations_sha256']==ec._hash(rows) and _same(envelope['conditioning'],automatic.CONDITIONING),
        'Automatic source evidence changed')
    ids=[]
    for number,row in enumerate(rows,1):
        require(type(row['owner_id']) is int and row['owner_id']==number
            and row['observation_id']==f'auto-source-owner-{number:03d}' and type(row['text']) is str
            and type(row['detected_language']) is str and row['native_status']==('empty' if row['text']=='' else 'complete')
            and type(row['vad']['detected_frames']) is int and row['vad']['detected_frames']>=0,'Invalid automatic observation')
        if row['native_status']=='complete' and row['text']!='' and row['detected_language'] not in ('','Japanese') and row['vad']['detected_frames']>0:
            ids.append(number)
    return ids


def build_plan(automatic_source,automatic_plan,assets,eligibility_record):
    automatic.check_plan(automatic_plan,assets)
    require(_same(eligibility_record,ELIGIBILITY),'Pre-score eligibility declaration changed')
    require(eligible_owner_ids(automatic_source)==list(OWNERS),'Exact predeclared nine-owner predicate changed')
    require(automatic_source['version']==automatic.OBSERVATIONS_VERSION
        and automatic_source['plan_sha256']==automatic_plan['plan_sha256']
        and automatic_source['source_rows_sha256']==automatic_plan['source_rows_sha256']
        and automatic_source['original_mono_sha256']==automatic_plan['original_mono_sha256']
        and automatic_source['original_mono_frames']==automatic_plan['original_mono_frames']
        and automatic_source['sample_rate']==16000 and _same(automatic_source['execution_identity'],automatic_plan['execution_identity']),
        'Automatic native source/plan binding changed')
    geometry=[];bindings={};parent_receipts={}
    for i in OWNERS:
        row=automatic_source['observations'][i-1];scope=automatic_plan['geometry'][i-1];a=automatic_plan['audio_bindings'][str(i)]
        require(all(_same(row[k],v) for k,v in scope.items()) and row['audio_sha256']==a['sha256']
            and row['pcm_sha256']==a['pcm_sha256'],'Parent automatic physical scope changed')
        geometry.append(deepcopy(scope));bindings[str(i)]=deepcopy(a);parent_receipts[str(i)]=deepcopy(row['receipt'])
    value={'version':PLAN_VERSION,'automatic_source_sha256':ec._hash(automatic_source),
        'parent_observations_sha256':automatic_source['observations_sha256'],
        'source_rows_sha256':automatic_source['source_rows_sha256'],'original_mono_sha256':automatic_source['original_mono_sha256'],
        'original_mono_frames':automatic_source['original_mono_frames'],'sample_rate':16000,'eligible_owner_ids':list(OWNERS),
        'eligibility_sha256':ec._hash(eligibility_record),'geometry':geometry,'audio_bindings':bindings,'parent_receipts':parent_receipts,
        'execution_identity':execution_identity(assets),'automatic_worker_sha256':_file(automatic.__file__),
        'base_worker_sha256':_file(base.__file__),'policy':deepcopy(POLICY),
        'slots':[{'slot_id':f'owner-{i:03d}','owner_id':i} for i in OWNERS]}
    value['plan_sha256']=ec._hash(value);return value


def check_plan(plan,assets):
    ec._keys(plan,{'version','automatic_source_sha256','parent_observations_sha256','source_rows_sha256','original_mono_sha256',
        'original_mono_frames','sample_rate','eligible_owner_ids','eligibility_sha256','geometry','audio_bindings','parent_receipts',
        'execution_identity','automatic_worker_sha256','base_worker_sha256','policy','slots','plan_sha256'})
    require(plan['version']==PLAN_VERSION and plan['plan_sha256']==ec._hash({k:v for k,v in plan.items() if k!='plan_sha256'})
        and _same(plan['policy'],POLICY) and plan['automatic_worker_sha256']==_file(automatic.__file__)
        and plan['base_worker_sha256']==_file(base.__file__) and _same(plan['execution_identity'],execution_identity(assets))
        and _same(plan['eligible_owner_ids'],list(OWNERS)) and plan['eligibility_sha256']==ec._hash(ELIGIBILITY)
        and _same(plan['slots'],[{'slot_id':f'owner-{i:03d}','owner_id':i} for i in OWNERS])
        and len(plan['geometry'])==9 and set(plan['audio_bindings'])=={str(i) for i in OWNERS},'Forced source plan changed')
    for i,row in zip(OWNERS,plan['geometry']):
        require(row['owner_id']==i and 0<=row['crop_start_frame']<row['crop_end_frame']<=plan['original_mono_frames']
            and row['sample_rate']==16000 and plan['audio_bindings'][str(i)]['frames']==row['crop_end_frame']-row['crop_start_frame'],
            'Forced physical scope changed')


def build_request(plan,owner_id):
    require(type(owner_id) is int and owner_id in OWNERS,'Owner outside fixed forced treatment')
    return {'version':REQUEST_VERSION,'plan_sha256':plan['plan_sha256'],'owner_id':owner_id,
        'observation_id':f'forced-source-owner-{owner_id:03d}','parent_observation_id':f'auto-source-owner-{owner_id:03d}',
        'view':deepcopy(plan['geometry'][OWNERS.index(owner_id)]),'audio':deepcopy(plan['audio_bindings'][str(owner_id)]),
        'execution_identity':deepcopy(plan['execution_identity']),'policy':deepcopy(base.requests.POLICY),
        'language':'Japanese','conditioning':deepcopy(CONDITIONING)}


def validate_request(request,plan):
    require(_same(request,build_request(plan,request['owner_id'])),'Forced source request changed')


def protocol_category(raw,text):
    if not raw.strip():return 'empty_raw'
    return 'nonempty_transcript' if text else 'unrecognized_empty_protocol'


def summarize(plan,outputs):
    require(set(outputs)=={s['slot_id'] for s in plan['slots']},'Complete nine forced observations required')
    return {'available_slots':9,'unavailable_slots':0,'empty_slots':sum(v['parsed_text']=='' for v in outputs.values()),
        'parser_language_counts':dict(sorted(Counter(v['parser_language'] for v in outputs.values()).items())),
        'language_origin':'forced','forced_language':'Japanese',
        'protocol_counts':dict(sorted(Counter(protocol_category(v['raw_text'],v['parsed_text']) for v in outputs.values()).items())),
        'parser_changed_slots':sum(v['raw_text']!=v['parsed_text'] for v in outputs.values()),
        'input_normalization_changed_slots':0,'accuracy_verified':False}

def capture_call(asr, audio, request, *, artifact_root, save_capture, deadline):
    """Transparent single public normalization/processor/outer/inner capture."""
    import numpy as np
    import torch
    require(request.get('version') == REQUEST_VERSION and _same(request.get('policy'), base.requests.POLICY),
            'Native request policy changed')
    require(request.get('language') == 'Japanese', 'Forced Japanese must be explicit')
    audio = _pcm(audio); expected = _apply_policy(asr); context = request['conditioning']['context']
    require(_same(request['conditioning'],CONDITIONING) and context == ''
            and not any(token in context for token in request['execution_identity']['reserved_tokens']), 'Invalid exact ASR context')
    require(_sha(audio.tobytes()) == request['audio']['pcm_sha256'], 'ASR input PCM does not match bound view')
    capture = {key: None for key in _CAPTURE_KEYS}
    for key in ('public_calls', 'normalization_calls', 'split_calls', 'infer_calls', 'outer_calls', 'inner_calls', 'processor_calls', 'decode_calls'):
        capture[key] = 0
    capture['input_pcm'] = tensor_record(audio, artifact_root)
    def persist(): save_capture(deepcopy(capture))
    persist()
    public = inspect.unwrap(asr.transcribe).__func__ if hasattr(inspect.unwrap(asr.transcribe), '__func__') else inspect.unwrap(asr.transcribe)
    namespace = public.__globals__
    originals = {name: namespace[name] for name in ('normalize_audios', 'split_audio_into_chunks')}
    infer = asr._infer_asr; outer = asr.model.generate; inner = asr.model.thinker.generate
    prompt_builder = asr._build_text_prompt; processor = asr.processor
    def normalize(value):
        capture['normalization_calls'] += 1
        require(capture['normalization_calls'] == 1 and isinstance(value, tuple) and len(value) == 2
                and value[0] is audio and type(value[1]) is int and value[1] == 16000, 'Native normalization request changed')
        result = originals['normalize_audios'](value)
        require(isinstance(result, list) and len(result) == 1, 'Native normalization cardinality changed')
        normalized = _pcm(result[0]); capture['normalized_pcm'] = tensor_record(normalized, artifact_root); persist()
        require(_same(capture['input_pcm'], capture['normalized_pcm']), 'Native normalization changed PCM')
        return result
    def split(*args, **kwargs):
        capture['split_calls'] += 1
        require(not args and set(kwargs) == {'wav', 'sr', 'max_chunk_sec'} and capture['split_calls'] == 1
                and kwargs['sr'] == 16000 and kwargs['max_chunk_sec'] == 1200, 'Native chunk policy changed')
        result = originals['split_audio_into_chunks'](**kwargs)
        require(isinstance(result, list) and len(result) == 1 and result[0][1] == 0, 'Native chunk mapping changed')
        descriptor = tensor_record(_pcm(result[0][0]), artifact_root)
        capture['chunk'] = {'count': 1, 'offset_seconds': 0, 'sample_rate': 16000, 'pcm': descriptor}; persist()
        require(_same(descriptor, capture['input_pcm']), 'Native chunk changed waveform')
        return result
    def build_prompt(*args, **kwargs):
        require(not args and kwargs == {'context': context, 'force_language': 'Japanese'} and capture['prompt'] is None,
                'Native context/prompt multiplicity changed')
        result = prompt_builder(**kwargs); require(type(result) is str and bool(result), 'Native prompt missing')
        capture['prompt'] = result; persist(); return result
    def captured_infer(contexts, wavs, languages):
        capture['infer_calls'] += 1
        require(capture['infer_calls'] == 1 and contexts == [context] and languages == ['Japanese'] and len(wavs) == 1
                and _sha(_pcm(wavs[0]).tobytes()) == request['audio']['pcm_sha256'], 'Native infer scope changed')
        capture['contexts'] = contexts; capture['languages'] = languages; persist()
        return infer(contexts, wavs, languages)
    class Processor:
        def __getattr__(self, name): return getattr(processor, name)
        def __call__(self, *args, **kwargs):
            capture['processor_calls'] += 1
            require(not args and set(kwargs) == {'text', 'audio', 'return_tensors', 'padding'}
                    and capture['processor_calls'] == 1 and kwargs['text'] == [capture['prompt']]
                    and kwargs['return_tensors'] == 'pt' and kwargs['padding'] is True
                    and len(kwargs['audio']) == 1 and _sha(_pcm(kwargs['audio'][0]).tobytes()) == request['audio']['pcm_sha256'],
                    'Native processor request changed')
            result = processor(**kwargs)
            require(set(result) == _TENSOR_KEYS, 'Native processor tensor fields changed')
            capture['processor_inputs'] = {key: tensor_record(value, artifact_root) for key, value in result.items()}
            persist(); return result
        def batch_decode(self, tokens, **kwargs):
            capture['decode_calls'] += 1
            require(capture['decode_calls'] == 1 and kwargs == {'skip_special_tokens': True, 'clean_up_tokenization_spaces': False},
                    'Native raw decoding policy changed')
            require(_array(tokens).tolist() == [capture['output_token_ids']], 'Native decoded token IDs changed')
            result = processor.batch_decode(tokens, **kwargs)
            require(isinstance(result, list) and len(result) == 1 and type(result[0]) is str, 'Native raw decode missing')
            capture['raw_text'] = result[0]; persist(); return result
    def generate_outer(*args, **kwargs):
        capture['outer_calls'] += 1
        require(not args and capture['outer_calls'] == 1 and set(kwargs) == _TENSOR_KEYS | {'max_new_tokens'}
                and type(kwargs['max_new_tokens']) is int and kwargs['max_new_tokens'] == 512, 'Native outer generation changed')
        capture['outer_inputs'] = {k: tensor_record(kwargs[k], artifact_root) for k in _TENSOR_KEYS}
        capture['outer_kwargs'] = {'max_new_tokens': kwargs['max_new_tokens']}; capture['outer_config'] = _config(asr.model)
        persist(); return outer(**kwargs)
    def generate_inner(*args, **kwargs):
        capture['inner_calls'] += 1
        require(not args and capture['inner_calls'] == 1 and set(kwargs) == _TENSOR_KEYS | {'max_new_tokens', 'eos_token_id', 'return_dict_in_generate'},
                'Native delegated generation changed')
        capture['inner_inputs'] = {k: tensor_record(kwargs[k], artifact_root) for k in _TENSOR_KEYS}
        capture['inner_kwargs'] = {k: deepcopy(kwargs[k]) for k in ('max_new_tokens', 'eos_token_id', 'return_dict_in_generate')}
        capture['inner_config'] = _config(asr.model.thinker)
        require(_same(capture['outer_config'], expected) and _same(capture['inner_config'], expected)
                and _same(capture['inner_kwargs'], {'max_new_tokens': 512, 'eos_token_id': [151645, 151643], 'return_dict_in_generate': True})
                and _same(capture['outer_inputs'], capture['inner_inputs']), 'Native effective inner policy differs')
        tokens = processor.tokenizer.encode(context, add_special_tokens=False)
        require(isinstance(tokens, list) and all(type(t) is int and t >= 0 for t in tokens) and len(tokens) <= 1024, 'ASR hint token limit exceeded')
        capture['hint_token_ids'] = tokens
        ids = _array(kwargs['input_ids']); mask = _array(kwargs['attention_mask']); features = _array(kwargs['input_features']); fmask = _array(kwargs['feature_attention_mask'])
        require(ids.ndim == 2 and ids.shape[0] == 1 and mask.shape == ids.shape and ids.dtype.kind in 'iu'
                and ids.shape[1] > 0 and (ids >= 0).all() and np.isin(mask, [0, 1]).all() and features.ndim == 3 and features.shape[:2] == (1, 128)
                and fmask.ndim == 2 and fmask.shape[0] == 1 and fmask.shape[1] == features.shape[2]
                and np.isin(fmask, [0, 1]).all() and 0 < int(fmask.sum()) <= 3000, 'Native expanded audio/text geometry changed')
        capture['capacity'] = {'input_positions': int(ids.shape[1]), 'feature_frames': int(fmask.sum()),
            'position_limit': 65536, 'reserved_output': 512, 'margin': 64, 'hint_tokens': len(tokens),
            'fits': ids.shape[1]+512+64 <= 65536}
        capture['capacity_finished_utc'] = _now(); persist()
        require(capture['capacity']['fits'] is True, 'Expanded ASR input exceeds pinned capacity')
        if time.monotonic() >= deadline: raise WorkerDeadline('No native generation time remains')
        capture['generation_started_utc'] = _now(); persist()
        try:
            output = inner(**kwargs)
            ids = _array(output.sequences)
            require(ids.ndim == 2 and ids.shape[0] == 1, 'Native output sequence cardinality changed')
            prefix = _array(kwargs['input_ids'])
            require(np.array_equal(ids[:, :prefix.shape[1]], prefix), 'Native output prompt prefix changed')
            generated = ids[0, prefix.shape[1]:].tolist(); capture['output_token_ids'] = generated
            capture['normal_stop'] = bool(generated and generated[-1] in base.requests.POLICY['eos_token_id']
                and not any(token in base.requests.POLICY['eos_token_id'] for token in generated[:-1]) and len(generated) < 512)
            persist(); return output
        finally:
            capture['generation_finished_utc'] = _now(); persist()
    namespace['normalize_audios'] = normalize; namespace['split_audio_into_chunks'] = split
    asr._infer_asr = captured_infer; asr._build_text_prompt = build_prompt; asr.processor = Processor()
    asr.model.generate = generate_outer; asr.model.thinker.generate = generate_inner
    try:
        with _deadline(deadline):
            torch.manual_seed(20260915); torch.cuda.manual_seed_all(20260915)
            capture['public_calls'] = 1; persist()
            with torch.inference_mode():
                results = asr.transcribe((audio, 16000), context=context, language='Japanese', return_time_stamps=False)
            require(isinstance(results, list) and len(results) == 1, 'Native public result missing')
            result = {key: getattr(results[0], key) for key in ('text', 'language', 'time_stamps')}
            capture['native_result'] = result; capture['parser_changed'] = capture['raw_text'] != result['text']; persist()
            require(capture['normal_stop'] is True and all(capture[key] == 1 for key in ('public_calls', 'normalization_calls',
                'split_calls', 'infer_calls', 'outer_calls', 'inner_calls', 'processor_calls', 'decode_calls')), 'Incomplete or truncated native call')
            return capture
    finally:
        namespace.update(originals); asr._infer_asr = infer; asr._build_text_prompt = prompt_builder
        asr.processor = processor; asr.model.generate = outer; asr.model.thinker.generate = inner


def validate_native_receipt(receipt, request, plan, *, assets, artifact_root, processor):
    """Independent pinned CPU processor/parser replay; never calls a model."""
    import numpy as np
    validate_request(request, plan); validate_assets(assets, plan['execution_identity'])
    _sealed(receipt)
    ec._keys(receipt, {'version', 'request', 'request_sha256', 'assets_sha256', 'started_utc', 'finished_utc',
                       'seconds', 'status', 'error_type', 'capture', 'receipt_sha256'})
    require(receipt['version'] == VERSION and _same(receipt['request'], request)
        and receipt['request_sha256'] == ec._hash(request) and receipt['assets_sha256'] == ec._hash(assets)
        and receipt['status'] in ('complete', 'empty') and receipt['error_type'] is None
        and type(receipt['seconds']) in (int, float) and math.isfinite(receipt['seconds'])
        and 0 <= receipt['seconds'] <= 60, 'Native receipt identity/status/time changed')
    c = receipt['capture']; ec._keys(c, _CAPTURE_KEYS)
    for key in ('public_calls', 'normalization_calls', 'split_calls', 'infer_calls', 'outer_calls', 'inner_calls', 'processor_calls', 'decode_calls'):
        require(type(c[key]) is int and c[key] == 1, 'Native ASR call or chunk count changed')
    arrays = {}
    for key in ('input_pcm', 'normalized_pcm'):
        arrays[key] = _load_tensor(c[key], artifact_root)
        _pcm(arrays[key]); require(c[key]['bytes_sha256'] == request['audio']['pcm_sha256'], 'Native PCM differs from physical view')
    require(_same(c['input_pcm'], c['normalized_pcm']) and _same(c['chunk'],
        {'count': 1, 'offset_seconds': 0, 'sample_rate': 16000, 'pcm': c['input_pcm']}), 'Native normalization/chunk changed')
    source_pcm = _read_audio(request['audio'])
    require(source_pcm.tobytes() == arrays['input_pcm'].tobytes(), 'Captured PCM differs from pinned WAV')
    context = request['conditioning']['context']
    require(_same(c['contexts'], [context]) and _same(c['languages'], ['Japanese']), 'Native conditioning changed')
    hint_tokens = processor.tokenizer.encode(context, add_special_tokens=False)
    require(isinstance(hint_tokens, list) and all(type(t) is int and t >= 0 for t in hint_tokens)
        and len(hint_tokens) <= 1024 and _same(c['hint_token_ids'], hint_tokens), 'Native hint tokens changed')
    expected_prompt = processor.apply_chat_template([
        {'role': 'system', 'content': context}, {'role': 'user', 'content': [{'type': 'audio', 'audio': ''}]}],
        add_generation_prompt=True, tokenize=False)+'language Japanese<asr_text>'
    require(c['prompt'] == expected_prompt, 'Native rendered context/template changed')
    inputs = processor(text=[expected_prompt], audio=[arrays['input_pcm']], return_tensors='pt', padding=True)
    require(set(inputs) == _TENSOR_KEYS, 'Replayed processor fields changed')
    expected_processor = {key: _describe(value) for key, value in inputs.items()}
    expected_outer = {key: _describe(_array(value).astype(np.float16) if _array(value).dtype.kind == 'f'
                                   else _array(value)) for key, value in inputs.items()}
    require(_same(c['processor_inputs'], expected_processor) and _same(c['outer_inputs'], expected_outer)
        and _same(c['inner_inputs'], expected_outer), 'Native processor/features/input positions differ from PCM')
    for collection in ('processor_inputs', 'outer_inputs', 'inner_inputs'):
        for value in c[collection].values(): _load_tensor(value, artifact_root)
    policy = {key: deepcopy(base.requests.POLICY[key]) for key in _GEN_KEYS}
    require(_same(c['outer_config'], policy) and _same(c['inner_config'], policy)
        and _same(c['outer_kwargs'], {'max_new_tokens': 512})
        and _same(c['inner_kwargs'], {'max_new_tokens': 512, 'eos_token_id': [151645, 151643], 'return_dict_in_generate': True}),
        'Native effective decoder policy changed')
    ids = _array(inputs['input_ids']); mask = _array(inputs['feature_attention_mask'])
    expected_capacity = {'input_positions': int(ids.shape[1]), 'feature_frames': int(mask.sum()), 'position_limit': 65536,
        'reserved_output': 512, 'margin': 64, 'hint_tokens': len(hint_tokens), 'fits': ids.shape[1]+512+64 <= 65536}
    require(_same(c['capacity'], expected_capacity) and expected_capacity['fits'] is True, 'Native capacity changed')
    times = [_time(receipt['started_utc']), _time(c['capacity_finished_utc']), _time(c['generation_started_utc']),
             _time(c['generation_finished_utc']), _time(receipt['finished_utc'])]
    require(times == sorted(times), 'Native capacity/generation timestamps reordered')
    generated = c['output_token_ids']
    require(isinstance(generated, list) and 0 < len(generated) < 512
        and all(type(token) is int and token >= 0 for token in generated)
        and generated[-1] in base.requests.POLICY['eos_token_id']
        and not any(token in base.requests.POLICY['eos_token_id'] for token in generated[:-1])
        and c['normal_stop'] is True, 'Native EOS/count failed')
    decoded = processor.batch_decode([generated], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    require(isinstance(decoded, list) and len(decoded) == 1 and type(decoded[0]) is str and decoded[0] == c['raw_text'],
            'Native raw text differs from saved token IDs')
    language, text = native_parse(c['raw_text'], assets['parser']['path'], assets['parser']['sha256'])
    require(_same(c['native_result'], {'text': text, 'language': language, 'time_stamps': None})
        and type(c['parser_changed']) is bool and c['parser_changed'] == (c['raw_text'] != text)
        and receipt['status'] == ('empty' if text == '' else 'complete'), 'Native parser result/empty status changed')
    return {'raw_text': c['raw_text'], 'parsed_text': text, 'parser_language': language,
        'audio_features_sha256': tensor_identity(c['outer_inputs']['input_features']),
        'audio_mask_sha256': tensor_identity(c['outer_inputs']['feature_attention_mask'])}


def run_slots(plan,assets,folder,asr,audios,*,deadline,save_state):
    check_plan(plan,assets);folder=Path(folder);outputs={}
    slots=[{**s,'number':i,'status':'undispatched','receipt':None} for i,s in enumerate(plan['slots'],1)]
    summary={'version':VERSION,'plan_sha256':plan['plan_sha256'],'status':'running','slots':slots,
        'asr_calls':0,'error_type':None,'available_slots':0,'unavailable_slots':9}
    save_state(deepcopy(summary))
    for slot in slots:
        request=build_request(plan,slot['owner_id'])
        tokens=asr.processor.tokenizer.encode(request['conditioning']['context'],add_special_tokens=False)
        require(type(tokens) is list and all(type(t) is int and t>=0 for t in tokens) and len(tokens)<=1024,'Hint capacity exceeded')
    for slot in slots:
        request=build_request(plan,slot['owner_id']);validate_request(request,plan)
        path=folder/'receipts'/f'{slot["number"]:02d}-owner-{slot["owner_id"]:03d}.json';path.parent.mkdir(parents=True,exist_ok=True)
        started=time.monotonic();call_deadline=min(deadline,started+60)
        if started>=deadline:raise WorkerDeadline('No real-pair call time remains')
        with path.with_suffix('.started.json').open('x',encoding='utf-8') as stream:
            json.dump({'version':VERSION,'request_sha256':ec._hash(request),'started_utc':_now()},stream)
        require(not path.exists(),'Real-pair receipt exists; no reroll')
        receipt={'version':VERSION,'request':request,'request_sha256':ec._hash(request),'assets_sha256':ec._hash(assets),
            'started_utc':_now(),'finished_utc':None,'seconds':0.,'status':'running','error_type':None,'capture':{}}
        slot.update(status='started',receipt=str(path));summary['asr_calls']+=1
        def save_capture(capture):receipt['capture']=capture;_save(path,receipt)
        _save(path,receipt);save_state(deepcopy(summary));failure=None
        try:
            with _deadline(call_deadline):
                capture_call(asr,audios[str(slot['owner_id'])],request,artifact_root=folder,save_capture=save_capture,deadline=call_deadline)
                receipt.update(status='empty' if receipt['capture']['native_result']['text']=='' else 'complete',
                    finished_utc=_now(),seconds=time.monotonic()-started);_save(path,receipt)
                outputs[slot['slot_id']]=validate_native_receipt(receipt,request,plan,assets=assets,artifact_root=folder,processor=asr.processor)
                summary['available_slots']=len(outputs);summary['unavailable_slots']=9-len(outputs)
        except BaseException as exc:failure=exc;receipt.update(status='failed',error_type=type(exc).__name__)
        finally:
            receipt.update(finished_utc=_now(),seconds=time.monotonic()-started);_save(path,receipt)
            slot['status']=receipt['status'];save_state(deepcopy(summary))
        if failure is not None:
            summary.update(status='failed',error_type=type(failure).__name__);save_state(deepcopy(summary));return summary
    metrics=summarize(plan,outputs);summary.update(metrics)
    summary['status']='complete'
    save_state(deepcopy(summary));return summary


def run_worker(spec_path):
    started=time.monotonic();spec=_read(spec_path);ec._keys(spec,base._SPEC_KEYS)
    require(spec['version']==VERSION and type(spec['worker_seconds']) in (int,float)
        and math.isfinite(spec['worker_seconds']) and 0<spec['worker_seconds']<=240,'Invalid real-pair worker budget')
    remaining=(_time(spec['work_deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
    deadline=started+min(spec['worker_seconds'],remaining);folder=Path(spec['output_dir'])
    require(folder.is_absolute(),'Worker folder must be absolute');folder.mkdir(parents=True,exist_ok=True)
    require(not any(folder.iterdir()),'Worker already attempted')
    state={'version':VERSION,'status':'prepared','started_utc':_now(),'finished_utc':None,'seconds':0.,'asr_calls':0,
        'model_loads':0,'model_load_attempts':0,'error_type':None,'spec_sha256':_file(spec_path),'available_slots':0,'unavailable_slots':9}
    def save(value):state.update(value);state['seconds']=time.monotonic()-started;_save(folder/'worker-result.json',state)
    save({})
    try:
        with _deadline(deadline):
            require(_file(__file__)==spec['worker_sha256'],'Real-pair worker changed')
            require(base.late_audio._interpreter_path(sys.executable)==spec['assets']['interpreter'],'Wrong worker interpreter')
            require(_file(spec['plan_path'])==spec['plan_file_sha256'],'Real-pair plan changed')
            plan=_read(spec['plan_path']);check_plan(plan,spec['assets']);validate_assets(spec['assets'],plan['execution_identity'],runtime_check=True)
            for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):os.environ[key]='1'
            audios={str(i):_read_audio(plan['audio_bindings'][str(i)]) for i in OWNERS}
            save({'model_load_attempts':1,'model_load_started_utc':_now()})
            asr=base._load_asr(spec['assets']);base._check_processor(spec['assets'],asr.processor)
            save({'model_loads':1,'model_load_finished_utc':_now()});save(run_slots(plan,spec['assets'],folder,asr,audios,deadline=deadline,save_state=save))
    except BaseException as exc:save({'status':'failed','error_type':type(exc).__name__})
    finally:state['finished_utc']=_now();save({})
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);value=run_worker(p.parse_args().spec)
    print(json.dumps({k:value.get(k) for k in ('status','asr_calls','model_loads','seconds','error_type')}))
    return 1 if value['status']=='failed' else 0
if __name__=='__main__':raise SystemExit(main())
