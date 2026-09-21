"""Long automatic-language ASR with exact frontend and scoped window-mask proof.

All inputs are local and unhinted. This new worker never assigns words to owners;
its controller owns physical-grid provenance and the inclusive restore budget.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src import contextual_asr_native as base
from src import auto_source_native as automatic
from src import masked_source_native as masked
from src import qwen_audio_attention as attention
from src import evidence_context as ec
from src.contextual_review_contract import require
from src.contextual_asr_native import (_same, _sha, _CAPTURE_KEYS, tensor_record, _TENSOR_KEYS, _config,
    _array, _now, _deadline, WorkerDeadline, _sealed, _load_tensor, _describe, _GEN_KEYS, _time,
    tensor_identity, _file, _read, _save)
from src.contextual_asr_auto_native import native_parse
VERSION = 'long-source-worker-1'
REQUEST_VERSION = 'long-source-request-1'
PLAN_VERSION = 'long-source-plan-1'
OBSERVATIONS_VERSION = 'long-source-observations-1'
CONDITIONING = {'context': '', 'language': None, 'hotwords': []}
POLICY = {'language': None, 'maximum_calls': 14, 'real_calls': 12, 'control_calls': 2,
    'model_loads': 1, 'retries': 0, 'local_seconds': 1800, 'work_seconds': 1380,
    'cleanup_reserve_seconds': 420, 'recap_calls': 0, 'candidate_generated': False, 'accuracy_verified': False}
ENCODER_CONFIG = deepcopy(masked.ENCODER_CONFIG)
ATTENTION_POLICY = deepcopy(masked.ATTENTION_POLICY)


@dataclass(frozen=True)
class CapturePolicy:
    max_audio_seconds: int = 120
    max_new_tokens: int = 4096

    def __post_init__(self):
        require(type(self.max_audio_seconds) is int and 1 <= self.max_audio_seconds <= 1200
            and type(self.max_new_tokens) is int and 1 <= self.max_new_tokens < 65536-64, 'Invalid capture bounds')

    def as_dict(self):
        return {'max_audio_seconds': self.max_audio_seconds, 'max_new_tokens': self.max_new_tokens,
            'sdk_chunk_seconds': 1200, 'sample_rate': 16000, 'feature_hop_samples': 160,
            'position_limit': 65536, 'margin': 64, 'language': None,
            'generation': {k: (self.max_new_tokens if k == 'max_new_tokens' else deepcopy(base.requests.POLICY[k])) for k in _GEN_KEYS}}

    @classmethod
    def from_dict(cls, value):
        result = cls(value['max_audio_seconds'], value['max_new_tokens'])
        require(_same(value, result.as_dict()), 'Capture policy changed')
        return result


CAPTURE_POLICY = CapturePolicy()
load_processor = base.load_processor
protocol_category = automatic.protocol_category


def execution_identity(assets):
    value = base.execution_identity(assets); value['worker_sha256'] = _file(__file__); return value


def validate_assets(assets, identity, *, runtime_check=False):
    require(_same(identity, execution_identity(assets)), 'Long worker identity changed')
    return base.validate_assets(assets, base.execution_identity(assets), runtime_check=runtime_check)


def model_config(assets):
    return masked.model_config(assets)


def _grid(grid):
    from src import long_source_grid
    long_source_grid.validate_geometry(grid)
    require(len(grid['windows']) == 12 and len(grid['source_owners']) == 66
        and grid['window_frames'] == 120*16000, 'Declared long grid requires 12 windows and 66 owners')


def build_plan(grid, audio_bindings, assets, input_proof_sha256):
    _grid(grid)
    slots = [{'slot_id': name, 'kind': 'control', 'window_id': None} for name in ('silence', 'noise')]
    slots += [{'slot_id': w['window_id'], 'kind': 'real', 'window_id': w['window_id']} for w in grid['windows']]
    require(type(audio_bindings) is dict and set(audio_bindings) == {s['slot_id'] for s in slots}, 'All 14 audio bindings required')
    for slot in slots:
        binding = audio_bindings[slot['slot_id']]
        ec._keys(binding, {'path','sha256','pcm_sha256','frames','sample_rate','dtype'})
        expected = 120*16000 if slot['kind'] == 'control' else next(w['crop_end_frame']-w['crop_start_frame']
            for w in grid['windows'] if w['window_id'] == slot['window_id'])
        require(type(binding['frames']) is int and binding['frames'] == expected
            and type(binding['sample_rate']) is int and binding['sample_rate'] == 16000 and binding['dtype'] == 'float32'
            and Path(binding['path']).is_absolute(), 'Long audio binding geometry changed')
        for key in ('sha256','pcm_sha256'): require(type(binding[key]) is str and re.fullmatch('[0-9a-f]{64}',binding[key]), 'Invalid audio hash')
    require(type(input_proof_sha256) is str and re.fullmatch('[0-9a-f]{64}',input_proof_sha256), 'Invalid input proof hash')
    value = {'version':PLAN_VERSION, 'grid':deepcopy(grid), 'audio_bindings':deepcopy(audio_bindings),
        'execution_identity':execution_identity(assets), 'input_proof_sha256':input_proof_sha256,
        'attention_source':attention.source_identity(), 'encoder_config':deepcopy(ENCODER_CONFIG),
        'model_config':model_config(assets), 'attention_policy':deepcopy(ATTENTION_POLICY),
        'capture_policy':CAPTURE_POLICY.as_dict(), 'policy':deepcopy(POLICY), 'slots':slots}
    value['plan_sha256'] = ec._hash(value); return value


def check_plan(plan, assets):
    require(_same(plan, build_plan(plan['grid'],plan['audio_bindings'],assets,plan['input_proof_sha256'])), 'Long plan changed')


def build_request(plan, slot_id):
    slot = next((s for s in plan['slots'] if s['slot_id'] == slot_id),None)
    require(slot is not None, 'Unknown long slot')
    view = {'kind':'synthetic_control','control_id':slot_id,'crop_start_frame':0,
            'crop_end_frame':120*16000,'sample_rate':16000} if slot['kind']=='control' else next(
            w for w in plan['grid']['windows'] if w['window_id']==slot_id)
    return {'version':REQUEST_VERSION,'plan_sha256':plan['plan_sha256'],'slot':deepcopy(slot),
        'observation_id':'long-source-'+slot_id,'view':deepcopy(view),'audio':deepcopy(plan['audio_bindings'][slot_id]),
        'execution_identity':deepcopy(plan['execution_identity']),'language':None,'conditioning':deepcopy(CONDITIONING),
        'capture_policy':deepcopy(plan['capture_policy']),'attention_source':deepcopy(plan['attention_source']),
        'encoder_config':deepcopy(plan['encoder_config']),'model_config':deepcopy(plan['model_config']),
        'attention_policy':deepcopy(ATTENTION_POLICY)}


def validate_request(request, plan):
    require(_same(request,build_request(plan,request['slot']['slot_id'])), 'Long request changed')


def _pcm(audio, policy=CAPTURE_POLICY):
    import numpy as np
    require(isinstance(audio,np.ndarray) and audio.dtype==np.dtype('float32') and audio.ndim==1
        and 160 <= len(audio) <= policy.max_audio_seconds*16000 and np.isfinite(audio).all()
        and float(np.max(np.abs(audio)))<=1., 'Invalid long finite FLOAT PCM')
    return np.ascontiguousarray(audio)


def _read_audio(binding, policy=CAPTURE_POLICY):
    import numpy as np
    import soundfile as sf
    path=Path(binding['path']); require(_file(path)==binding['sha256'], 'Long WAV changed')
    info=sf.info(path)
    require(info.channels==1 and info.frames==binding['frames'] and info.samplerate==16000 and info.subtype=='FLOAT', 'Long WAV geometry changed')
    audio,rate=sf.read(path,dtype='float32',always_2d=False); audio=_pcm(audio,policy)
    require(rate==16000 and _sha(audio.tobytes())==binding['pcm_sha256'] and _file(path)==binding['sha256'], 'Long PCM changed')
    return np.array(audio,copy=True)


def _apply_policy(asr, policy):
    require(asr.backend=='transformers' and asr.max_inference_batch_size==1 and asr.max_new_tokens==policy.max_new_tokens
        and asr.forced_aligner is None and str(asr.model.device).startswith('cuda')
        and str(asr.model.dtype)=='torch.float16', 'Wrong long ASR backend/dtype/reserve')
    asr.model.eval(); asr.model.thinker.eval()
    settings=policy.as_dict()['generation']
    for model in (asr.model,asr.model.thinker):
        for key,value in settings.items(): setattr(model.generation_config,key,deepcopy(value))
    return settings


def _feature_geometry(inputs, pcm_frames, policy):
    import numpy as np
    ids=_array(inputs['input_ids']); mask=_array(inputs['attention_mask'])
    features=_array(inputs['input_features']); fmask=_array(inputs['feature_attention_mask'])
    expected=pcm_frames//160
    require(ids.ndim==2 and ids.shape[0]==1 and ids.dtype.kind in 'iu' and ids.shape[1]>0 and (ids>=0).all()
        and mask.shape==ids.shape and np.isin(mask,[0,1]).all()
        and features.ndim==3 and features.shape==(1,128,expected) and fmask.shape==(1,expected)
        and np.all(fmask==1) and 0<expected<=policy.max_audio_seconds*100, 'Full long frontend frames missing or truncated')
    return ids, fmask


def capture_call(asr, audio, request, *, artifact_root, save_capture, deadline):
    """Transparent single public normalization/processor/outer/inner capture."""
    import numpy as np
    import torch
    require(request.get('version') == REQUEST_VERSION, 'Long request version changed')
    policy = CapturePolicy.from_dict(request['capture_policy'])
    require(request.get('language', 'missing') is None, 'Automatic language must be explicit')
    audio = _pcm(audio, policy); expected = _apply_policy(asr, policy); context = request['conditioning']['context']
    require(_same(request['conditioning'],CONDITIONING) and context == ''
            and not any(token in context for token in request['execution_identity']['reserved_tokens']), 'Invalid exact ASR context')
    require(len(audio) == request['audio']['frames'] and _sha(audio.tobytes()) == request['audio']['pcm_sha256'], 'ASR input PCM does not match bound view')
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
        normalized = _pcm(result[0], policy); capture['normalized_pcm'] = tensor_record(normalized, artifact_root); persist()
        require(_same(capture['input_pcm'], capture['normalized_pcm']), 'Native normalization changed PCM')
        return result
    def split(*args, **kwargs):
        capture['split_calls'] += 1
        require(not args and set(kwargs) == {'wav', 'sr', 'max_chunk_sec'} and capture['split_calls'] == 1
                and kwargs['sr'] == 16000 and kwargs['max_chunk_sec'] == 1200, 'Native chunk policy changed')
        result = originals['split_audio_into_chunks'](**kwargs)
        require(isinstance(result, list) and len(result) == 1 and result[0][1] == 0, 'Native chunk mapping changed')
        descriptor = tensor_record(_pcm(result[0][0], policy), artifact_root)
        capture['chunk'] = {'count': 1, 'offset_seconds': 0, 'sample_rate': 16000, 'pcm': descriptor}; persist()
        require(_same(descriptor, capture['input_pcm']), 'Native chunk changed waveform')
        return result
    def build_prompt(*args, **kwargs):
        require(not args and kwargs == {'context': context, 'force_language': None} and capture['prompt'] is None,
                'Native context/prompt multiplicity changed')
        result = prompt_builder(**kwargs); require(type(result) is str and bool(result), 'Native prompt missing')
        capture['prompt'] = result; persist(); return result
    def captured_infer(contexts, wavs, languages):
        capture['infer_calls'] += 1
        require(capture['infer_calls'] == 1 and contexts == [context] and languages == [None] and len(wavs) == 1
                and _sha(_pcm(wavs[0], policy).tobytes()) == request['audio']['pcm_sha256'], 'Native infer scope changed')
        capture['contexts'] = contexts; capture['languages'] = languages; persist()
        return infer(contexts, wavs, languages)
    class Processor:
        def __getattr__(self, name): return getattr(processor, name)
        def __call__(self, *args, **kwargs):
            capture['processor_calls'] += 1
            require(not args and set(kwargs) == {'text', 'audio', 'return_tensors', 'padding'}
                    and capture['processor_calls'] == 1 and kwargs['text'] == [capture['prompt']]
                    and kwargs['return_tensors'] == 'pt' and kwargs['padding'] is True
                    and len(kwargs['audio']) == 1 and _sha(_pcm(kwargs['audio'][0], policy).tobytes()) == request['audio']['pcm_sha256'],
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
                and type(kwargs['max_new_tokens']) is int and kwargs['max_new_tokens'] == policy.max_new_tokens, 'Native outer generation changed')
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
                and _same(capture['inner_kwargs'], {'max_new_tokens': policy.max_new_tokens, 'eos_token_id': [151645, 151643], 'return_dict_in_generate': True})
                and _same(capture['outer_inputs'], capture['inner_inputs']), 'Native effective inner policy differs')
        tokens = processor.tokenizer.encode(context, add_special_tokens=False)
        require(isinstance(tokens, list) and all(type(t) is int and t >= 0 for t in tokens) and tokens == [], 'ASR hint token limit exceeded')
        capture['hint_token_ids'] = tokens
        ids, fmask = _feature_geometry(kwargs, len(audio), policy)
        capture['capacity'] = {'input_positions': int(ids.shape[1]), 'feature_frames': int(fmask.sum()),
            'position_limit': 65536, 'reserved_output': policy.max_new_tokens, 'margin': 64, 'hint_tokens': len(tokens),
            'fits': ids.shape[1]+policy.max_new_tokens+64 <= 65536}
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
                and not any(token in base.requests.POLICY['eos_token_id'] for token in generated[:-1]) and len(generated) < policy.max_new_tokens)
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
                results = asr.transcribe((audio, 16000), context=context, language=None, return_time_stamps=False)
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
    policy = CapturePolicy.from_dict(request['capture_policy'])
    validate_request(request, plan); validate_assets(assets, plan['execution_identity'])
    _sealed(receipt)
    ec._keys(receipt, {'version', 'request', 'request_sha256', 'assets_sha256', 'started_utc', 'finished_utc',
                       'seconds', 'status', 'error_type', 'capture', 'attention', 'receipt_sha256'})
    require(receipt['version'] == VERSION and _same(receipt['request'], request)
        and receipt['request_sha256'] == ec._hash(request) and receipt['assets_sha256'] == ec._hash(assets)
        and receipt['status'] in ('complete', 'empty') and receipt['error_type'] is None
        and type(receipt['seconds']) in (int, float) and math.isfinite(receipt['seconds'])
        and 0 <= receipt['seconds'] <= 120, 'Native receipt identity/status/time changed')
    c = receipt['capture']; ec._keys(c, _CAPTURE_KEYS)
    for key in ('public_calls', 'normalization_calls', 'split_calls', 'infer_calls', 'outer_calls', 'inner_calls', 'processor_calls', 'decode_calls'):
        require(type(c[key]) is int and c[key] == 1, 'Native ASR call or chunk count changed')
    arrays = {}
    for key in ('input_pcm', 'normalized_pcm'):
        arrays[key] = _load_tensor(c[key], artifact_root)
        _pcm(arrays[key], policy); require(c[key]['bytes_sha256'] == request['audio']['pcm_sha256'], 'Native PCM differs from physical view')
    require(_same(c['input_pcm'], c['normalized_pcm']) and _same(c['chunk'],
        {'count': 1, 'offset_seconds': 0, 'sample_rate': 16000, 'pcm': c['input_pcm']}), 'Native normalization/chunk changed')
    source_pcm = _read_audio(request['audio'], policy)
    require(source_pcm.tobytes() == arrays['input_pcm'].tobytes(), 'Captured PCM differs from pinned WAV')
    context = request['conditioning']['context']
    require(_same(c['contexts'], [context]) and _same(c['languages'], [None]), 'Native conditioning changed')
    hint_tokens = processor.tokenizer.encode(context, add_special_tokens=False)
    require(isinstance(hint_tokens, list) and all(type(t) is int and t >= 0 for t in hint_tokens)
        and hint_tokens == [] and _same(c['hint_token_ids'], hint_tokens), 'Native hint tokens changed')
    expected_prompt = processor.apply_chat_template([
        {'role': 'system', 'content': context}, {'role': 'user', 'content': [{'type': 'audio', 'audio': ''}]}],
        add_generation_prompt=True, tokenize=False)
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
    generation = policy.as_dict()['generation']
    require(_same(c['outer_config'], generation) and _same(c['inner_config'], generation)
        and _same(c['outer_kwargs'], {'max_new_tokens': policy.max_new_tokens})
        and _same(c['inner_kwargs'], {'max_new_tokens': policy.max_new_tokens, 'eos_token_id': [151645, 151643], 'return_dict_in_generate': True}),
        'Native effective decoder policy changed')
    ids, mask = _feature_geometry(inputs, len(arrays['input_pcm']), policy)
    expected_capacity = {'input_positions': int(ids.shape[1]), 'feature_frames': int(mask.sum()), 'position_limit': 65536,
        'reserved_output': policy.max_new_tokens, 'margin': 64, 'hint_tokens': len(hint_tokens), 'fits': ids.shape[1]+policy.max_new_tokens+64 <= 65536}
    require(_same(c['capacity'], expected_capacity) and expected_capacity['fits'] is True, 'Native capacity changed')
    times = [_time(receipt['started_utc']), _time(c['capacity_finished_utc']), _time(c['generation_started_utc']),
             _time(c['generation_finished_utc']), _time(receipt['finished_utc'])]
    require(times == sorted(times), 'Native capacity/generation timestamps reordered')
    generated = c['output_token_ids']
    require(isinstance(generated, list) and 0 < len(generated) < policy.max_new_tokens
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
    record = attention.validate_attestation(receipt['attention'], feature_frames=c['capacity']['feature_frames'],
        config=plan['encoder_config'], source=plan['attention_source'])
    require(record['mask_dtype']=='float16' and record['device'].startswith('cuda'), 'Long attention dtype/device changed')
    return {'attention_sha256':receipt['attention']['attestation_sha256'], 'raw_text': c['raw_text'], 'parsed_text': text, 'native_language': language,
        'audio_features_sha256': tensor_identity(c['outer_inputs']['input_features']),
        'audio_mask_sha256': tensor_identity(c['outer_inputs']['feature_attention_mask'])}



def control_result(output):
    return masked.control_result(output)


def runtime_identity(asr, plan, assets):
    result=masked.runtime_identity(asr,plan,assets)
    require(asr.max_new_tokens==CAPTURE_POLICY.max_new_tokens, 'Runtime output reserve changed')
    result['max_new_tokens']=asr.max_new_tokens
    return result


def _load_asr(assets, policy=CAPTURE_POLICY):
    import torch
    from qwen_asr import Qwen3ASRModel
    return Qwen3ASRModel.from_pretrained(assets['model']['model_dir'],dtype=torch.float16,device_map='cuda',
        local_files_only=True,max_inference_batch_size=1,max_new_tokens=policy.max_new_tokens)


def run_slots(plan, assets, folder, asr, audios, *, deadline, save_state):
    check_plan(plan, assets); folder = Path(folder)
    require(_same(attention.encoder_config(asr.model.thinker.audio_tower), plan['encoder_config']), 'Actual encoder backend/config changed')
    slots = [{**s, 'number': i, 'status': 'undispatched', 'receipt': None} for i, s in enumerate(plan['slots'], 1)]
    summary = {'version': VERSION, 'plan_sha256': plan['plan_sha256'], 'status': 'running', 'slots': slots,
               'asr_calls': 0, 'controls': {}, 'real_audio_calls': 0, 'error_type': None}
    save_state(deepcopy(summary))
    for slot in slots:
        request = build_request(plan, slot['slot_id']); path = folder/'receipts'/f'{slot["number"]:02d}-{slot["slot_id"]}.json'
        path.parent.mkdir(parents=True, exist_ok=True); started = time.monotonic(); end = min(deadline, started+120)
        require(started < deadline, 'Long worker time exhausted')
        with path.with_suffix('.started.json').open('x', encoding='utf-8') as stream:
            import json
            json.dump({'version': VERSION, 'request_sha256': ec._hash(request), 'started_utc': _now()}, stream)
        require(not path.exists(), 'Long receipt already attempted; no reroll')
        receipt = {'version': VERSION, 'request': request, 'request_sha256': ec._hash(request), 'assets_sha256': ec._hash(assets),
                   'started_utc': _now(), 'finished_utc': None, 'seconds': 0., 'status': 'running', 'error_type': None,
                   'capture': {}, 'attention': None}
        slot.update(status='started', receipt=str(path)); summary['asr_calls'] += 1
        if slot['kind'] == 'real': summary['real_audio_calls'] += 1
        def save_capture(value): receipt['capture'] = value; _save(path, receipt)
        _save(path, receipt); save_state(deepcopy(summary)); failure = None
        try:
            with _deadline(end):
                audit = None
                try:
                    with attention.scoped_window_attention(asr.model.thinker.audio_tower) as audit:
                        capture_call(asr, audios[slot['slot_id']], request,
                            artifact_root=folder, save_capture=save_capture, deadline=end)
                finally:
                    receipt['attention'] = deepcopy(audit); _save(path, receipt)
                receipt.update(status='empty' if receipt['capture']['native_result']['text'] == '' else 'complete',
                               finished_utc=_now(), seconds=time.monotonic()-started)
                _save(path, receipt)
                output = validate_native_receipt(receipt, request, plan, assets=assets, artifact_root=folder, processor=asr.processor)
                if slot['kind'] == 'control': summary['controls'][slot['slot_id']] = control_result(output)
        except BaseException as exc:
            failure = exc; receipt.update(status='failed', error_type=type(exc).__name__)
        finally:
            receipt.update(finished_utc=_now(), seconds=time.monotonic()-started); _save(path, receipt)
            slot['status'] = receipt['status']; save_state(deepcopy(summary))
        if failure is not None:
            summary.update(status='failed', error_type=type(failure).__name__); break
        if slot['kind'] == 'control' and not summary['controls'][slot['slot_id']]['passed']:
            summary['status'] = 'negative_control_output'; break
    else: summary['status'] = 'complete'
    save_state(deepcopy(summary)); return summary


def cuda_memory(*, reset: bool = False) -> dict:
    """Actual owned-worker CUDA allocator peaks; never used by CPU replay."""
    import torch
    device = torch.cuda.current_device()
    if reset: torch.cuda.reset_peak_memory_stats(device)
    properties = torch.cuda.get_device_properties(device)
    return {'scope':'worker_cuda_reset_before_model_load','device_index':device,
        'device_name':properties.name,'device_total_bytes':properties.total_memory,
        'allocated_bytes':torch.cuda.memory_allocated(device),'reserved_bytes':torch.cuda.memory_reserved(device),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)}


def run_worker(spec_path):
    started = time.monotonic(); spec = _read(spec_path); ec._keys(spec, base._SPEC_KEYS)
    require(spec['version'] == VERSION and type(spec['worker_seconds']) in (int, float)
            and math.isfinite(spec['worker_seconds']) and 0 < spec['worker_seconds'] <= 1380, 'Invalid long worker budget')
    deadline = started+min(spec['worker_seconds'], (_time(spec['work_deadline_utc'])-datetime.now(timezone.utc)).total_seconds())
    folder = Path(spec['output_dir']); require(folder.is_absolute(), 'Absolute worker folder required')
    folder.mkdir(parents=True, exist_ok=True); require(not any(folder.iterdir()), 'Worker already attempted')
    state = {'version': VERSION, 'status': 'prepared', 'started_utc': _now(), 'finished_utc': None, 'seconds': 0.,
             'asr_calls': 0, 'model_loads': 0, 'model_load_attempts': 0, 'error_type': None, 'spec_sha256': _file(spec_path),
             'cuda_memory_before_load':None, 'cuda_memory_after_load':None, 'cuda_memory_final':None, 'cuda_memory_error_type':None}
    def save(value): state.update(value); state['seconds'] = time.monotonic()-started; _save(folder/'worker-result.json', state)
    save({}); memory_started = False
    try:
        with _deadline(deadline):
            require(_file(__file__) == spec['worker_sha256'] and _file(spec['plan_path']) == spec['plan_file_sha256'], 'Worker/plan changed')
            require(base.late_audio._interpreter_path(sys.executable) == spec['assets']['interpreter'], 'Wrong worker interpreter')
            plan = _read(spec['plan_path']); check_plan(plan, spec['assets'])
            base.validate_assets(spec['assets'], base.execution_identity(spec['assets']), runtime_check=True)
            for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'): os.environ[key] = '1'
            audios = {s['slot_id']: _read_audio(build_request(plan, s['slot_id'])['audio']) for s in plan['slots']}
            memory_started = True
            save({'cuda_memory_before_load':cuda_memory(reset=True)})
            save({'model_load_attempts': 1, 'model_load_started_utc': _now()})
            asr = _load_asr(spec['assets'])
            save({'cuda_memory_after_load':cuda_memory()})
            base._check_processor(spec['assets'], asr.processor)
            save({'model_loads': 1, 'model_load_finished_utc': _now(), 'runtime_before': runtime_identity(asr, plan, spec['assets'])})
            save(run_slots(plan, spec['assets'], folder, asr, audios, deadline=deadline, save_state=save))
            after = runtime_identity(asr, plan, spec['assets']); save({'runtime_after': after})
            require(_same(state['runtime_before'], after), 'Runtime backend changed during acquisition')
    except BaseException as exc: save({'status': 'failed', 'error_type': type(exc).__name__})
    finally:
        if memory_started:
            try: state['cuda_memory_final'] = cuda_memory()
            except BaseException as exc:
                state.update(status='failed',error_type=state['error_type'] or type(exc).__name__,cuda_memory_error_type=type(exc).__name__)
        state['finished_utc'] = _now(); save({})
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec', type=Path, required=True)
    value = run_worker(parser.parse_args().spec)
    print({k: value.get(k) for k in ('status', 'asr_calls', 'model_loads', 'seconds', 'error_type')})
    return 1 if value['status'] == 'failed' else 0
if __name__ == '__main__': raise SystemExit(main())
