"""One native long-context Gemma stream for the immutable compact B request.

The controller owns the declared profile, weights, process lifecycle and global
budget. Only ask performs I/O to a model endpoint. Replay is network-free and
retains the exact response; there is no fallback, normalization or retry.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
import time

import jsonschema
from src import compact_raw_draft_requests as contract
from src import contextual_asr_recap_native as shared
from src import evidence_context as ec
from src import revisit_workflow as rw
from src.config import TranslateConfig
from src.contextual_review_contract import require, strict_json
from src.translate import StreamResult, _build_payload, _record_request, _stream_response
from src.workflow_state import write_json

VERSION='long-context-gemma-native-1'
MODEL='gemma4-31b-qat-q4'
MODEL_SHA256='179cfb99212709597eae5929112cfca677e1bbf566178b479ae1da0c4772874b'
CONTEXT=163840
RESERVE=16384
MAX_ATTEMPT_SECONDS=1980
CAPACITY_BASE='http://127.0.0.1:18101'
ENDPOINT=CAPACITY_BASE+'/v1/chat/completions'
SAMPLER={'temperature':1.0,'top_p':.95,'top_k':64,'repeat_penalty':1.0,'seed':20260913}
_ROOT={'version','native','endpoint','key','backend_identity','backend_identity_sha256',
       'status','attempts','parsed','receipt_sha256'}
_ATTEMPT={'number','started_utc','finished_utc','seconds','payload_sha256','capacity_started_utc',
          'capacity_finished_utc','capacity','generation_started_utc','generation_finished_utc',
          'metrics_start','metrics_end','telemetry','raw','exception_type','failure_kind'}
_CAPACITY={'endpoint','props','payload_sha256','rendered_prompt','input_ids','prompt_tokens',
           'reserve','context','margin','fits'}


def _same(left,right):return ec._hash(left)==ec._hash(right)
def _track(path,raw=False):return Path(path).read_bytes() if raw else strict_json(Path(path).read_bytes())


def prepare_request(request):
    contract.validate_writer_request(request)
    payload={**SAMPLER,'model':MODEL,'max_tokens':RESERVE,'cache_prompt':False,
        'chat_template_kwargs':{'enable_thinking':False},'reasoning_budget_tokens':0,'reasoning_effort':'none',
        'response_format':{'type':'json_object','schema':deepcopy(request['schema'])}}
    cfg=TranslateConfig(endpoint=ENDPOINT,max_tokens=RESERVE,retries=0,separate_instruction=True,
                        telemetry_path=Path('metrics.jsonl'),extra_payload=payload)
    payload=_build_payload(json.dumps(request['body'],ensure_ascii=False),request['instruction'],cfg,{},
                           RESERVE,stream=True,with_thinking=False)
    return {'version':VERSION,'request':deepcopy(request),'payload':payload}


def _identity(identity):
    require(type(identity) is dict and identity.get('model_alias')==MODEL
        and identity.get('model_sha256')==MODEL_SHA256 and type(identity.get('context_size')) is int
        and identity['context_size']==CONTEXT and identity.get('full_swa') is False
        and type(identity.get('pid')) is int and identity['pid']>0
        and type(identity.get('model_path')) is str and Path(identity['model_path']).is_absolute(),
        'Wrong owned long-context Gemma identity')


def _props(props,identity):
    _identity(identity)
    require(type(props) is dict and props.get('model_alias') in (MODEL,[MODEL])
        and props.get('model_path')==identity['model_path']
        and type(props.get('default_generation_settings',{}).get('n_ctx')) is int
        and props['default_generation_settings']['n_ctx']==CONTEXT,'Native served Gemma context/weights differ')


def _capacity(payload,identity):
    props=rw._request_json(CAPACITY_BASE+'/props',timeout=20);_props(props,identity)
    rendered=rw._request_json(CAPACITY_BASE+'/apply-template',payload,timeout=20).get('prompt')
    require(type(rendered) is str and bool(rendered),'Native full template missing')
    ids=rw._request_json(CAPACITY_BASE+'/tokenize',{'content':rendered,'add_special':True,'parse_special':True},timeout=20).get('tokens')
    require(type(ids) is list,'Native token IDs missing')
    return {'endpoint':CAPACITY_BASE,'props':props,'payload_sha256':ec._hash(payload),'rendered_prompt':rendered,
        'input_ids':ids,'prompt_tokens':len(ids),'reserve':RESERVE,'context':CONTEXT,'margin':64,
        'fits':len(ids)+RESERVE+64<=CONTEXT}


def _check_capacity(value,payload,identity):
    ec._keys(value,_CAPACITY);_props(value['props'],identity)
    require(type(value['rendered_prompt']) is str and bool(value['rendered_prompt']),'Native rendered prompt missing')
    ids=value['input_ids'];require(type(ids) is list and bool(ids) and all(type(token) is int and token>=0 for token in ids),
                                 'Native exact token sequence missing')
    expected={'endpoint':CAPACITY_BASE,'payload_sha256':ec._hash(payload),'prompt_tokens':len(ids),
              'reserve':RESERVE,'context':CONTEXT,'margin':64,'fits':True}
    require(all(_same(value[key],item) for key,item in expected.items()) and len(ids)+RESERVE+64<=CONTEXT,
            'Long-context complete prompt and reserved output do not fit')


def _parse(raw,request):
    value=strict_json(raw);jsonschema.validate(value,request['schema'])
    ec._keys(value,{'owners'});ec._keys(value['owners'],set(map(str,contract.OWNER_IDS)))
    for owner in value['owners'].values():
        ec._keys(owner,{'chinese'});ec._text(owner['chinese'],maximum=contract.MAX_CHINESE_CHARS)
    return value


def _telemetry(attempt,native,key):
    rows=attempt['telemetry'];require(type(rows) is list and len(rows)==1,'Exactly one shared native stream required')
    row=rows[0];ec._keys(row,shared._TELEMETRY_KEYS);payload=native['payload']
    settings={name:payload[name] for name in ('model','max_tokens','reasoning_budget_tokens','reasoning_effort',
                                            'temperature','top_p','top_k','seed','stream')}
    settings['enable_thinking']=False
    require(row['stage']==key and type(row['attempt']) is int and row['attempt']==0 and row['thinking'] is False
        and type(row['reasoning_chars']) is int and row['reasoning_chars']==0
        and type(attempt['raw']) is str and type(row['answer_chars']) is int and row['answer_chars']==len(attempt['raw'])
        and row['finish_reason']=='stop' and shared._number(row['seconds']) and _same(row['request_settings'],settings),
        'Native raw/stop/telemetry settings differ')
    usage=row['usage'];require(type(usage) is dict and all(type(usage.get(k)) is int for k in
        ('prompt_tokens','completion_tokens','total_tokens')) and usage['prompt_tokens']>0
        and 0<usage['completion_tokens']<=RESERVE and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
        and usage['prompt_tokens']+RESERVE+64<=CONTEXT,'Native token usage absent or outside declared context')


def replay_native(path,native_request,backend_identity,track=_track):
    path=Path(path);saved=track(path);ec._keys(saved,_ROOT)
    expected=prepare_request(native_request['request']);_identity(backend_identity)
    require(_same(native_request,expected) and saved['version']==VERSION and saved['endpoint']==ENDPOINT
        and saved['key']==path.stem and _same(saved['native'],expected) and _same(saved['backend_identity'],backend_identity)
        and saved['backend_identity_sha256']==ec._hash(backend_identity) and saved['status']=='complete'
        and saved['receipt_sha256']==ec._hash({k:v for k,v in saved.items() if k!='receipt_sha256'}),
        'Native long-context request identity/completion changed')
    require(type(saved['attempts']) is list and len(saved['attempts'])==1,'One native attempt required')
    attempt=saved['attempts'][0];ec._keys(attempt,_ATTEMPT)
    require(type(attempt['number']) is int and attempt['number']==1 and attempt['payload_sha256']==ec._hash(expected['payload'])
        and shared._number(attempt['seconds']) and attempt['seconds']<=MAX_ATTEMPT_SECONDS
        and attempt['exception_type'] is None and attempt['failure_kind'] is None,'Native attempt failed or exceeded budget')
    times=[shared._timestamp(attempt[key]) for key in ('started_utc','capacity_started_utc','capacity_finished_utc',
                                                     'generation_started_utc','generation_finished_utc','finished_utc')]
    require(times==sorted(times),'Native generation preceded capacity admission')
    _check_capacity(attempt['capacity'],expected['payload'],backend_identity)
    metrics=shared._metrics(track(path.parent/'metrics.jsonl',raw=True));start,end=attempt['metrics_start'],attempt['metrics_end']
    require(type(start) is int and type(end) is int and 0<=start<end<=len(metrics) and end-start==1
        and [i for i,row in enumerate(metrics) if row.get('stage')==path.stem]==[start]
        and _same(attempt['telemetry'],metrics[start:end]),'Missing/extra native stream or changed telemetry')
    _telemetry(attempt,expected,path.stem)
    value=_parse(attempt['raw'],expected['request']);require(_same(saved['parsed'],value),'Parsed output differs from exact raw')
    return value


def ask(directory,key,endpoint,native_request,backend_identity,deadline):
    require(endpoint==ENDPOINT and type(key) is str and re.fullmatch('[A-Za-z0-9_-]+',key),'Wrong native endpoint/key')
    require(type(deadline) in (int,float) and math.isfinite(deadline),'Invalid absolute work deadline')
    directory=Path(directory);path=directory/(key+'.json');expected=prepare_request(native_request['request'])
    require(_same(native_request,expected),'Native request changed');_identity(backend_identity)
    if path.exists():return replay_native(path,native_request,backend_identity)
    started=time.monotonic();remaining=min(MAX_ATTEMPT_SECONDS,deadline-started)
    require(remaining>=1,'No complete native attempt time remains')
    cfg=TranslateConfig(endpoint=ENDPOINT,max_tokens=RESERVE,timeout=max(1,int(remaining)),retries=0,
        separate_instruction=True,response_guard_floor=65536,stage=key,telemetry_path=directory/'metrics.jsonl',
        extra_payload=deepcopy(expected['payload']))
    directory.mkdir(parents=True,exist_ok=True);metrics_path=directory/'metrics.jsonl'
    prior=shared._metrics(metrics_path.read_bytes()) if metrics_path.exists() else []
    require(not any(row.get('stage')==key for row in prior),'Orphan native telemetry blocks dispatch')
    attempt={name:None for name in _ATTEMPT};attempt.update(number=1,started_utc=shared._now(),
        payload_sha256=ec._hash(expected['payload']),telemetry=[])
    saved={'version':VERSION,'native':expected,'endpoint':ENDPOINT,'key':key,'backend_identity':deepcopy(backend_identity),
        'backend_identity_sha256':ec._hash(backend_identity),'status':'running','attempts':[attempt],'parsed':None}
    with path.open('x',encoding='utf-8') as stream:json.dump(saved,stream)
    def save():
        saved['receipt_sha256']=ec._hash({name:value for name,value in saved.items() if name!='receipt_sha256'});write_json(path,saved)
    failure=None;phase='capacity'
    try:
        with shared._within(min(deadline,started+MAX_ATTEMPT_SECONDS)):
            attempt['capacity_started_utc']=shared._now();save()
            try:attempt['capacity']=_capacity(expected['payload'],backend_identity);save();_check_capacity(attempt['capacity'],expected['payload'],backend_identity)
            finally:attempt['capacity_finished_utc']=shared._now();save()
            before=shared._metrics(metrics_path.read_bytes()) if metrics_path.exists() else []
            attempt['metrics_start']=len(before);attempt['generation_started_utc']=shared._now();phase='transport';save()
            result=None;generation_start=time.monotonic()
            try:
                result=_stream_response(endpoint,deepcopy(expected['payload']),cfg.timeout,cfg.response_guard_floor)
                attempt['raw']=result.content;save()
            finally:
                attempt['generation_finished_utc']=shared._now()
                _record_request(cfg,0,generation_start,False,result or StreamResult(content='',finish_reason='capture_unavailable'),payload=expected['payload'])
                after=shared._metrics(metrics_path.read_bytes()) if metrics_path.exists() else []
                attempt['metrics_end']=len(after);attempt['telemetry']=after[len(before):];save()
                require(_same(before,after[:len(before)]),'Native telemetry prefix changed')
            phase='response';_telemetry(attempt,expected,key);saved['parsed']=_parse(attempt['raw'],expected['request'])
    except BaseException as exc:
        failure=exc;attempt.update(exception_type=type(exc).__name__,failure_kind='budget' if isinstance(exc,rw.LocalBudgetExceeded) else phase)
    finally:
        attempt.update(finished_utc=shared._now(),seconds=time.monotonic()-started)
        saved['status']='failed' if failure else 'complete';save()
    if failure:raise failure
    return replay_native(path,native_request,backend_identity)
