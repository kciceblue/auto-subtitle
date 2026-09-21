"""Exactly one native stream per fixed source/translation stage.

The controller owns model loading, asset/runtime identity and restoration. This
adapter binds the actual request, local backend identity, capacity and shared
transport receipt. Replay never contacts a server and never normalizes text.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
import time

import jsonschema
from src import contextual_asr_recap_native as shared
from src import evidence_context as ec
from src import revisit_workflow as rw
from src import staged_source_requests as contract
from src.config import TranslateConfig
from src.contextual_review_contract import require, strict_json
from src.translate import StreamResult, _build_payload, _record_request, _stream_response
from src.workflow_state import write_json

VERSION='staged-source-native-1'
FAMILIES={
    'qwen':{'model':'qwen3.8-27b-dflash','served_alias':'local/qwen3.8-27b-dflash',
        'model_path':'/home/kciceblue/models/qwen3.8-27b-dflash/Qwen3.8-27B-NVFP4-MTP-LOW.gguf',
        'context':196608,'reserve':16384,
        'endpoint':'http://127.0.0.1:8089/v1/chat/completions','capacity_base':'http://127.0.0.1:18089',
        'sampler':{'temperature':.3,'top_p':.95,'seed':20260915}},
    'gemma':{'model':'gemma4-31b-qat-q4','served_alias':'gemma4-31b-qat-q4','context':32768,'reserve':8192,
        'endpoint':'http://127.0.0.1:18101/v1/chat/completions','capacity_base':'http://127.0.0.1:18101',
        'sampler':{'temperature':1.0,'top_p':.95,'top_k':64,'repeat_penalty':1.0,'seed':20260913}}}
_ROOT={'version','native','family','endpoint','key','backend_identity','backend_identity_sha256',
       'status','attempts','parsed','receipt_sha256'}
_ATTEMPT={'number','started_utc','finished_utc','seconds','payload_sha256','capacity_started_utc',
          'capacity_finished_utc','capacity','generation_started_utc','generation_finished_utc',
          'metrics_start','metrics_end','telemetry','raw','exception_type','failure_kind'}
_CAPACITY={'endpoint','props','payload_sha256','rendered_prompt','input_ids','prompt_tokens',
           'reserve','context','margin','fits'}


def _same(a,b):return ec._hash(a)==ec._hash(b)
def _read(path):return strict_json(Path(path).read_bytes())
def _track(path,raw=False):return Path(path).read_bytes() if raw else _read(path)


def _policy(family):
    require(type(family) is str and family in FAMILIES,'Unknown fixed text stage family')
    return FAMILIES[family]


def prepare_request(request,family):
    policy=_policy(family);contract.validate_stage_request(request,family)
    extra={**policy['sampler'],'model':policy['model'],'max_tokens':policy['reserve'],
        'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':False},
        'reasoning_budget_tokens':0,'reasoning_effort':'none',
        'response_format':{'type':'json_object','schema':deepcopy(request['schema'])}}
    cfg=TranslateConfig(endpoint=policy['endpoint'],max_tokens=policy['reserve'],retries=0,
        separate_instruction=True,telemetry_path=Path('metrics.jsonl'),extra_payload=extra)
    payload=_build_payload(json.dumps(request['body'],ensure_ascii=False),request['instruction'],cfg,{},
        policy['reserve'],stream=True,with_thinking=False)
    return {'version':VERSION,'family':family,'request':deepcopy(request),'payload':payload}


def config_for(endpoint,directory,key,request,family,remaining=600):
    policy=_policy(family)
    require(endpoint==policy['endpoint'] and type(key) is str and re.fullmatch('[A-Za-z0-9_-]+',key),'Wrong stage endpoint/key')
    require(type(remaining) in (int,float) and math.isfinite(remaining) and remaining>=1,'No stage execution time remains')
    return TranslateConfig(endpoint=endpoint,max_tokens=policy['reserve'],retries=0,
        timeout=max(1,int(min(600,remaining))),separate_instruction=True,response_guard_floor=65536,
        stage=key,telemetry_path=Path(directory)/'metrics.jsonl',extra_payload=deepcopy(request['payload']))


def _identity(identity,family):
    policy=_policy(family);require(type(identity) is dict,'Missing backend identity')
    if family=='qwen':
        require(identity.get('loaded_model')==policy['model'],'Wrong active Qwen identity')
    else:
        require(identity.get('model_alias')==policy['model'] and identity.get('model_sha256')==
            '179cfb99212709597eae5929112cfca677e1bbf566178b479ae1da0c4772874b'
            and type(identity.get('context_size')) is int and identity['context_size']==policy['context']
            and type(identity.get('model_path')) is str and Path(identity['model_path']).is_absolute()
            and type(identity.get('pid')) is int and identity['pid']>0,'Wrong owned Gemma writer identity')


def _props(props,family,identity):
    policy=_policy(family);_identity(identity,family)
    require(type(props) is dict and props.get('model_alias') in (policy['served_alias'],[policy['served_alias']])
        and type(props.get('default_generation_settings',{}).get('n_ctx')) is int
        and props['default_generation_settings']['n_ctx']==policy['context'],'Native served model/context differs')
    expected_path=identity['model_path'] if family=='gemma' else policy['model_path']
    require(props.get('model_path')==expected_path,'Native served model weights differ')


def _capacity(payload,family,identity):
    policy=_policy(family);base=policy['capacity_base']
    props=rw._request_json(base+'/props',timeout=20);_props(props,family,identity)
    rendered=rw._request_json(base+'/apply-template',payload,timeout=20).get('prompt')
    require(type(rendered) is str and bool(rendered),'Native complete template missing')
    ids=rw._request_json(base+'/tokenize',{'content':rendered,'add_special':True,'parse_special':True},timeout=20).get('tokens')
    require(type(ids) is list,'Missing native token IDs')
    return {'endpoint':base,'props':props,'payload_sha256':ec._hash(payload),'rendered_prompt':rendered,
        'input_ids':ids,'prompt_tokens':len(ids),'reserve':policy['reserve'],'context':policy['context'],'margin':64,
        'fits':len(ids)+policy['reserve']+64<=policy['context']}


def _check_capacity(c,payload,family,identity):
    ec._keys(c,_CAPACITY);policy=_policy(family);_props(c['props'],family,identity)
    require(type(c['rendered_prompt']) is str and bool(c['rendered_prompt']),'Rendered request missing')
    ids=c['input_ids'];require(type(ids) is list and bool(ids) and all(type(t) is int and t>=0 for t in ids),'Native input IDs invalid')
    expected={'endpoint':policy['capacity_base'],'payload_sha256':ec._hash(payload),'prompt_tokens':len(ids),
        'reserve':policy['reserve'],'context':policy['context'],'margin':64,'fits':True}
    require(all(_same(c[k],v) for k,v in expected.items()) and len(ids)+policy['reserve']+64<=policy['context'],
        'Native complete request does not fit fixed capacity')


def _parse(raw,request,family):
    value=strict_json(raw);jsonschema.validate(value,request['schema'])
    checked=contract.validate_stage_response(value,request,family)
    require(_same(value,checked),'Stage validator must not normalize or repair native output')
    return value


def _telemetry(a,native,key):
    rows=a['telemetry'];require(type(rows) is list and len(rows)==1,'Exactly one native transport required')
    row=rows[0];ec._keys(row,shared._TELEMETRY_KEYS)
    payload=native['payload'];policy=_policy(native['family'])
    settings={k:payload[k] for k in ('model','max_tokens','reasoning_budget_tokens','reasoning_effort','temperature','top_p','seed','stream')}
    if 'top_k' in payload:settings['top_k']=payload['top_k']
    settings['enable_thinking']=False
    require(row['stage']==key and type(row['attempt']) is int and row['attempt']==0 and row['thinking'] is False
        and type(row['reasoning_chars']) is int and row['reasoning_chars']==0
        and type(a['raw']) is str and type(row['answer_chars']) is int and row['answer_chars']==len(a['raw'])
        and row['finish_reason']=='stop' and shared._number(row['seconds'])
        and _same(row['request_settings'],settings),'Native raw/stop/settings differ')
    usage=row['usage']
    require(type(usage) is dict and all(type(usage.get(k)) is int for k in ('prompt_tokens','completion_tokens','total_tokens'))
        and 0<usage['prompt_tokens'] and 0<usage['completion_tokens']<=policy['reserve']
        and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
        and usage['prompt_tokens']+policy['reserve']+64<=policy['context'],'Native token usage absent or out of budget')
    return row


def replay_native(path,request,family,backend_identity,track=_track):
    path=Path(path);policy=_policy(family);saved=track(path);ec._keys(saved,_ROOT)
    expected=prepare_request(request['request'],family);_identity(backend_identity,family)
    require(_same(request,expected) and saved['version']==VERSION and saved['family']==family
        and _same(saved['native'],expected) and saved['endpoint']==policy['endpoint'] and saved['key']==path.stem
        and _same(saved['backend_identity'],backend_identity) and saved['backend_identity_sha256']==ec._hash(backend_identity)
        and saved['status']=='complete' and saved['receipt_sha256']==ec._hash({k:v for k,v in saved.items() if k!='receipt_sha256'}),
        'Native stage identity or completion changed')
    require(type(saved['attempts']) is list and len(saved['attempts'])==1,'Stage may have only one native attempt')
    a=saved['attempts'][0];ec._keys(a,_ATTEMPT)
    require(type(a['number']) is int and a['number']==1 and a['payload_sha256']==ec._hash(expected['payload'])
        and shared._number(a['seconds']) and a['seconds']<=600
        and a['exception_type'] is None and a['failure_kind'] is None,'Stage attempt failed or exceeded budget')
    times=[shared._timestamp(a[k]) for k in ('started_utc','capacity_started_utc','capacity_finished_utc',
        'generation_started_utc','generation_finished_utc','finished_utc')]
    require(times==sorted(times),'Stage generation preceded native capacity')
    _check_capacity(a['capacity'],expected['payload'],family,backend_identity)
    metrics=shared._metrics(track(path.parent/'metrics.jsonl',raw=True));start,end=a['metrics_start'],a['metrics_end']
    require(type(start) is int and type(end) is int and 0<=start<end<=len(metrics) and end-start==1
        and [i for i,row in enumerate(metrics) if row.get('stage')==path.stem]==[start]
        and _same(a['telemetry'],metrics[start:end]),'Missing/extra stage transport or changed telemetry')
    _telemetry(a,expected,path.stem)
    value=_parse(a['raw'],expected['request'],family);require(_same(saved['parsed'],value),'Stage parsed value differs from raw')
    return value


def ask(directory,key,endpoint,request,family,backend_identity,deadline):
    policy=_policy(family);directory=Path(directory);path=directory/(key+'.json')
    require(endpoint==policy['endpoint'],'Wrong native stage endpoint')
    expected=prepare_request(request['request'],family);require(_same(request,expected),'Stage request mutated')
    _identity(backend_identity,family)
    if path.exists():return replay_native(path,request,family,backend_identity)
    started=time.monotonic();cfg=config_for(endpoint,directory,key,request,family,deadline-started)
    directory.mkdir(parents=True,exist_ok=True);metrics_path=directory/'metrics.jsonl'
    prior=shared._metrics(metrics_path.read_bytes()) if metrics_path.exists() else []
    require(not any(row.get('stage')==key for row in prior),'Orphan stage telemetry blocks dispatch')
    a={k:None for k in _ATTEMPT};a.update(number=1,started_utc=shared._now(),payload_sha256=ec._hash(expected['payload']),telemetry=[])
    saved={'version':VERSION,'native':expected,'family':family,'endpoint':endpoint,'key':key,
        'backend_identity':deepcopy(backend_identity),'backend_identity_sha256':ec._hash(backend_identity),
        'status':'running','attempts':[a],'parsed':None}
    with path.open('x',encoding='utf-8') as f:json.dump(saved,f)
    def save():
        saved['receipt_sha256']=ec._hash({k:v for k,v in saved.items() if k!='receipt_sha256'});write_json(path,saved)
    failure=None;phase='capacity'
    try:
        with shared._within(min(deadline,started+600)):
            a['capacity_started_utc']=shared._now();save()
            try:a['capacity']=_capacity(expected['payload'],family,backend_identity);save();_check_capacity(a['capacity'],expected['payload'],family,backend_identity)
            finally:a['capacity_finished_utc']=shared._now();save()
            before=shared._metrics(metrics_path.read_bytes()) if metrics_path.exists() else []
            a['metrics_start']=len(before);phase='transport';a['generation_started_utc']=shared._now();save()
            result=None;generation_start=time.monotonic()
            try:
                result=_stream_response(endpoint,deepcopy(expected['payload']),cfg.timeout,cfg.response_guard_floor)
                a['raw']=result.content;save()
            finally:
                a['generation_finished_utc']=shared._now()
                _record_request(cfg,0,generation_start,False,result or StreamResult(content='',finish_reason='capture_unavailable'),payload=expected['payload'])
                after=shared._metrics(metrics_path.read_bytes()) if metrics_path.exists() else []
                a['metrics_end']=len(after);a['telemetry']=after[len(before):];save()
                require(_same(before,after[:len(before)]),'Native telemetry prefix changed')
            phase='response';_telemetry(a,expected,key);saved['parsed']=_parse(a['raw'],expected['request'],family)
    except BaseException as exc:
        failure=exc;a.update(exception_type=type(exc).__name__,failure_kind='budget' if isinstance(exc,rw.LocalBudgetExceeded) else phase)
    finally:
        a.update(finished_utc=shared._now(),seconds=time.monotonic()-started);saved['status']='failed' if failure else 'complete';save()
    if failure:raise failure
    return replay_native(path,request,family,backend_identity)
