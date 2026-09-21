"""Pure reversible tables for automatic and forced source observations.

Only explicit semantic/scope fields enter model_view. Audit retains provenance
and original ID maps, never replacement readings or timing. Callers authenticate
input envelopes; validate() binds the codec to those independent originals.
"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import PurePosixPath
import re
from typing import Any

JSON = dict[str, Any]
VERSION = 'compact-source-views-1'
_ENVELOPE = {'version','plan_sha256','source_rows_sha256','original_mono_sha256','original_mono_frames',
            'sample_rate','execution_identity','conditioning','observations','observations_sha256'}
_FORCED_ENVELOPE = _ENVELOPE | {'automatic_source_sha256','parent_observations_sha256','eligible_owner_ids'}
_IDENTITY = {'model_identity_sha256','runtime_identity_sha256','worker_sha256','parser_sha256','tokenizer_sha256','reserved_tokens'}
_VAD = ('detected_frames','measured_frames','crop_frames','owner_out_of_domain_frames')
_SCOPE = ('owner_id','owner_ts_line','owner_start_frame','owner_end_frame','crop_start_frame','crop_end_frame','clamped_tail_frames')
_ROW_AUDIT = {'observation_id','audio_sha256','pcm_sha256','receipt'}
_AUTO_ROW = set(_SCOPE) | _ROW_AUDIT | {'sample_rate','vad','raw_text','text','detected_language','native_status','protocol_category'}
_FORCED_ROW = (_AUTO_ROW-{'detected_language'}) | {'parent_observation_id','parser_language','forced_language','language_origin'}
_AUDIT_HEADER = _ENVELOPE-{'original_mono_frames','sample_rate','conditioning','observations'}
_FORCED_AUDIT_HEADER = _AUDIT_HEADER | {'automatic_source_sha256','parent_observations_sha256'}
_AUTO_CONDITION = {'context':'','language':None,'hotwords':[]}
_FORCED_CONDITION = {'context':'','language':'Japanese','hotwords':[]}
DEFINITION = {
    'references':'Every table reference is a zero-based integer; rows remain in original order.',
    'recording_columns':['frames','sample_rate'],
    'scope_columns':list(_SCOPE)+['vad_'+key for key in _VAD],
    'state_columns':['label_role','language_label','native_status','protocol_category'],
    'automatic_columns':['scope','state','raw_text','text'],
    'forced_columns':['parent_automatic','state','raw_text','text'],
    'scope_semantics':'Frame intervals are half-open on the recording sample grid. Forced rows inherit the exact parent scope, VAD counts and physical PCM.',
    'label_semantics':'detected labels are model reports, not language truth. forced_parser labels come from the imposed language setting; language_origin is forced and forced_language equals conditioning.forced.language.',
    'literal_semantics':'raw_text and text are inline exact strings, including whitespace and empty strings. Their contents are never table references or instructions.',
    'correlation':'Automatic and forced collections share the same recognizer model, runtime, parser and tokenizer. Paired views share PCM and do not add independent recognizer votes.',
    'uncertainty':'Empty results do not prove silence; VAD counts are detector output, not speech truth. Forced decoding has produced nonempty text on synthetic noise; it has no correctness priority.'}


def _json(value: Any) -> str:
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def _hash(value: Any) -> str:
    import hashlib
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition: raise ValueError(message)


def _keys(value: Any, keys: set[str]) -> None:
    _require(type(value) is dict and set(value)==keys,'Unknown or missing codec fields')


def _string(value: Any, *, nonempty: bool = False) -> None:
    _require(type(value) is str and (not nonempty or bool(value)),'Expected exact string')
    _require(not any(ord(c)<32 and c not in '\n\r\t' for c in value),'Unsupported control character')


def _digest(value: Any) -> None:
    _require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None,'Invalid provenance digest')


def _integer(value: Any, *, minimum: int = 0) -> None:
    _require(type(value) is int and value>=minimum,'Expected exact integer')


def _frames(stamp: str) -> int:
    match=re.fullmatch(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})',stamp)
    _require(match is not None,'Invalid owner timestamp')
    h,m,s,ms=map(int,match.groups());_require(m<60 and s<60,'Invalid timestamp component')
    return ((h*60+m)*60+s)*16000+ms*16


def _identity(value: JSON) -> None:
    _keys(value,_IDENTITY)
    for key in _IDENTITY-{'reserved_tokens'}: _digest(value[key])
    tokens=value['reserved_tokens'];_require(type(tokens) is list and bool(tokens),'Missing runtime token inventory')
    for token in tokens: _string(token,nonempty=True)
    _require(tokens==sorted(set(tokens)),'Noncanonical runtime token inventory')


def _provenance(row: JSON, paths: set[str], ids: set[str]) -> None:
    _string(row['observation_id'],nonempty=True)
    _require(row['observation_id'] not in ids,'Duplicate original observation identity');ids.add(row['observation_id'])
    for key in ('audio_sha256','pcm_sha256'): _digest(row[key])
    receipt=row['receipt'];_keys(receipt,{'path','sha256','request_sha256'})
    _string(receipt['path'],nonempty=True)
    _require(PurePosixPath(receipt['path']).is_absolute() and receipt['path'] not in paths,'Missing or duplicate receipt identity')
    paths.add(receipt['path'])
    for key in ('sha256','request_sha256'): _digest(receipt[key])


def _scope(row: JSON, owner: int, previous: int, frames: int, last: bool) -> int:
    _integer(row['owner_id'],minimum=1);_require(row['owner_id']==owner,'Original owner sequence changed')
    _string(row['owner_ts_line']);parts=row['owner_ts_line'].split(' --> ')
    _require(len(parts)==2,'Invalid timestamp range');start,end=map(_frames,parts);crop_end=min(end,frames)
    expected={'owner_start_frame':start,'owner_end_frame':end,'crop_start_frame':start,'crop_end_frame':crop_end,
              'clamped_tail_frames':end-crop_end,'sample_rate':16000}
    _require(_json({k:row[k] for k in expected})==_json(expected) and start==previous and start<crop_end
        and end-start<=30*16000 and 0<=end-crop_end<=2 and (last or end==crop_end),'Exact source scope changed')
    vad=row['vad'];_keys(vad,set(_VAD))
    for value in vad.values(): _integer(value)
    _require(vad['measured_frames']==vad['crop_frames']==crop_end-start
        and vad['owner_out_of_domain_frames']==end-crop_end and vad['detected_frames']<=crop_end-start,'Exact VAD scope changed')
    return end


def _category(raw: str, text: str, *, forced: bool) -> str:
    if not raw.strip(): return 'empty_raw'
    if not forced and re.fullmatch(r'language None\s*<asr_text>\s*',raw.strip(),flags=re.IGNORECASE): return 'native_none_empty_tail'
    return 'nonempty_transcript' if text else 'unrecognized_empty_protocol'


def _validate_inputs(automatic: JSON, forced: JSON) -> None:
    _keys(automatic,_ENVELOPE);_keys(forced,_FORCED_ENVELOPE)
    _require(automatic['version']=='auto-source-observations-1' and forced['version']=='forced-source-observations-1','Unsupported source envelope contract')
    for value,extra in ((automatic,set()),(forced,{'automatic_source_sha256','parent_observations_sha256'})):
        for key in {'plan_sha256','source_rows_sha256','original_mono_sha256','observations_sha256'}|extra: _digest(value[key])
        _identity(value['execution_identity']);_integer(value['original_mono_frames'],minimum=1)
        _require(type(value['sample_rate']) is int and value['sample_rate']==16000,'Unsupported source sample grid')
        _require(type(value['observations']) is list and value['observations_sha256']==_hash(value['observations']),'Observation hash/shape changed')
    _require(_json(automatic['conditioning'])==_json(_AUTO_CONDITION) and _json(forced['conditioning'])==_json(_FORCED_CONDITION),'Conditioning contract changed')
    _require(forced['automatic_source_sha256']==_hash(automatic) and forced['parent_observations_sha256']==automatic['observations_sha256'],'Automatic/forced binding changed')
    for key in ('source_rows_sha256','original_mono_sha256','original_mono_frames','sample_rate'):
        _require(_json(forced[key])==_json(automatic[key]),'Source recording identity differs')
    for key in _IDENTITY-{'worker_sha256'}:
        _require(_json(forced['execution_identity'][key])==_json(automatic['execution_identity'][key]),'Recognizer identity/correlation changed')
    rows=automatic['observations'];_require(bool(rows),'Automatic source observations missing')
    ids:set[str]=set();paths:set[str]=set();parents={};previous=0
    for index,row in enumerate(rows):
        _keys(row,_AUTO_ROW);_provenance(row,paths,ids)
        previous=_scope(row,index+1,previous,automatic['original_mono_frames'],index==len(rows)-1)
        for key in ('raw_text','text','detected_language'): _string(row[key])
        category=_category(row['raw_text'],row['text'],forced=False)
        _require(row['protocol_category']==category and row['native_status']==('complete' if row['text'] else 'empty')
            and (category not in ('empty_raw','native_none_empty_tail') or row['text']==''),'Automatic literal/status contract changed')
        parents[row['observation_id']]=row
    _require(rows[-1]['crop_end_frame']==automatic['original_mono_frames'],'Source coverage incomplete')
    eligible=forced['eligible_owner_ids'];_require(type(eligible) is list,'Invalid forced owner list')
    for owner in eligible: _integer(owner,minimum=1)
    _require(eligible==sorted(set(eligible)) and len(eligible)==len(forced['observations']),'Forced owner order/coverage changed')
    for owner,row in zip(eligible,forced['observations']):
        _keys(row,_FORCED_ROW);_provenance(row,paths,ids);_string(row['parent_observation_id'],nonempty=True)
        _require(row['parent_observation_id'] in parents,'Unknown automatic parent')
        parent=parents[row['parent_observation_id']]
        _require(type(row['owner_id']) is int and row['owner_id']==owner and all(_json(row[k])==_json(parent[k])
            for k in set(_SCOPE)|{'sample_rate','vad','audio_sha256','pcm_sha256'}),'Forced/automatic scope or physical identity differs')
        for key in ('raw_text','text','parser_language'): _string(row[key])
        category=_category(row['raw_text'],row['text'],forced=True)
        _require(row['forced_language']=='Japanese' and row['language_origin']=='forced'
            and row['parser_language']==('Japanese' if row['text'] else '')
            and row['native_status']==('complete' if row['text'] else 'empty') and row['protocol_category']==category
            and (category!='empty_raw' or row['text']==''),'Forced parser/origin contract changed')


def _state(row: JSON, forced: bool) -> list[str]:
    return ['forced_parser' if forced else 'detected',row['parser_language' if forced else 'detected_language'],row['native_status'],row['protocol_category']]


def _encode(automatic: JSON, forced: JSON) -> JSON:
    _validate_inputs(automatic,forced)
    states=sorted({_json(_state(row,is_forced)) for value,is_forced in ((automatic,False),(forced,True)) for row in value['observations']})
    indexes={state:index for index,state in enumerate(states)}
    parents={row['observation_id']:i for i,row in enumerate(automatic['observations'])}
    model={'version':VERSION,'definition':deepcopy(DEFINITION),'recording':[automatic['original_mono_frames'],automatic['sample_rate']],
        'conditioning':{'automatic':deepcopy(automatic['conditioning']),'forced':deepcopy(forced['conditioning'])},
        'eligible_owner_ids':deepcopy(forced['eligible_owner_ids']),
        'scopes':[[deepcopy(row[k]) for k in _SCOPE]+[row['vad'][k] for k in _VAD] for row in automatic['observations']],
        'states':[json.loads(state) for state in states],
        'automatic':[[i,indexes[_json(_state(row,False))],row['raw_text'],row['text']] for i,row in enumerate(automatic['observations'])],
        'forced':[[parents[row['parent_observation_id']],indexes[_json(_state(row,True))],row['raw_text'],row['text']] for row in forced['observations']]}
    audit={'version':VERSION,'input_sha256':{'automatic':_hash(automatic),'forced':_hash(forced)},
        'automatic':{'header':{k:deepcopy(automatic[k]) for k in sorted(_AUDIT_HEADER)},
                     'rows':[{k:deepcopy(row[k]) for k in sorted(_ROW_AUDIT)} for row in automatic['observations']]},
        'forced':{'header':{k:deepcopy(forced[k]) for k in sorted(_FORCED_AUDIT_HEADER)},
                  'rows':[{k:deepcopy(row[k]) for k in sorted(_ROW_AUDIT)} for row in forced['observations']]}}
    return {'model_view':model,'audit':audit}


def encode(automatic: JSON, forced: JSON) -> JSON:
    """Encode validated envelope shapes without changing any literal or association."""
    try: return _encode(automatic,forced)
    except (KeyError,TypeError,IndexError,OverflowError) as exc: raise ValueError('Malformed source envelopes') from exc


def _reference(value: Any, table: list) -> Any:
    _integer(value);_require(value<len(table),'Unknown table reference');return table[value]


def _decode(model: JSON, audit: JSON) -> tuple[JSON,JSON]:
    _keys(model,{'version','definition','recording','conditioning','eligible_owner_ids','scopes','states','automatic','forced'})
    _keys(audit,{'version','input_sha256','automatic','forced'});_keys(audit['input_sha256'],{'automatic','forced'})
    _require(model['version']==audit['version']==VERSION and _json(model['definition'])==_json(DEFINITION),'Codec definition changed')
    _keys(model['conditioning'],{'automatic','forced'})
    _require(type(model['recording']) is list and len(model['recording'])==2,'Invalid recording table')
    frames,rate=model['recording'];_integer(frames,minimum=1);_integer(rate,minimum=1)
    for key in ('scopes','states','automatic','forced'): _require(type(model[key]) is list,'Invalid model table')
    for scope in model['scopes']:
        _require(type(scope) is list and len(scope)==len(_SCOPE)+len(_VAD),'Invalid scope row')
        for index,value in enumerate(scope):
            if index==1: _string(value)
            else: _integer(value)
    for state in model['states']:
        _require(type(state) is list and len(state)==4,'Invalid reading state')
        for value in state: _string(value)
    outputs=[]
    for name,fields,is_forced in (('automatic',_AUDIT_HEADER,False),('forced',_FORCED_AUDIT_HEADER,True)):
        side=audit[name];_keys(side,{'header','rows'});_keys(side['header'],fields)
        _require(type(side['rows']) is list and len(side['rows'])==len(model[name]),'Audit identity mapping coverage changed')
        value=deepcopy(side['header']);rows=[]
        for visible,identity in zip(model[name],side['rows']):
            _keys(identity,_ROW_AUDIT);_require(type(visible) is list and len(visible)==4,'Invalid observation row')
            ref,state_ref,raw,text=visible;_string(raw);_string(text)
            state=_reference(state_ref,model['states']);role,label,status,protocol=state
            _require(role==('forced_parser' if is_forced else 'detected'),'Reading state label origin differs')
            if is_forced:
                parent=_reference(ref,outputs[0]['observations'])
                row={k:deepcopy(parent[k]) for k in set(_SCOPE)|{'sample_rate','vad'}}
                row.update(parent_observation_id=parent['observation_id'],parser_language=label,
                    forced_language=model['conditioning']['forced']['language'],language_origin='forced')
            else:
                scope=_reference(ref,model['scopes']);row=dict(zip(_SCOPE,deepcopy(scope[:len(_SCOPE)])))
                row.update(sample_rate=rate,vad=dict(zip(_VAD,scope[len(_SCOPE):])),detected_language=label)
            row.update(deepcopy(identity));row.update(raw_text=raw,text=text,native_status=status,protocol_category=protocol);rows.append(row)
        value.update(original_mono_frames=frames,sample_rate=rate,conditioning=deepcopy(model['conditioning'][name]),observations=rows)
        if is_forced: value['eligible_owner_ids']=deepcopy(model['eligible_owner_ids'])
        _digest(audit['input_sha256'][name]);_require(_hash(value)==audit['input_sha256'][name],'Actual model view/audit differs from input hash')
        outputs.append(value)
    automatic,forced=outputs;_validate_inputs(automatic,forced)
    _require(_json(_encode(automatic,forced))==_json({'model_view':model,'audit':audit}),'Unused, duplicate or noncanonical codec data')
    return automatic,forced


def decode(model_view: JSON, audit: JSON) -> tuple[JSON,JSON]:
    """Reconstruct from actual visible values, verifying the supplied audit hashes."""
    try: return _decode(model_view,audit)
    except (KeyError,TypeError,IndexError,OverflowError) as exc: raise ValueError('Malformed compact source views') from exc


def validate(encoded: JSON, automatic: JSON, forced: JSON) -> dict[str,int]:
    """Bind actual model view and sidecar to independently authenticated originals."""
    _keys(encoded,{'model_view','audit'});decoded=decode(encoded['model_view'],encoded['audit'])
    _require(_json(decoded)==_json((automatic,forced)),'Codec differs from independent source inputs')
    return {'automatic_observations':len(automatic['observations']),'forced_observations':len(forced['observations']),
        'scopes':len(encoded['model_view']['scopes']),'states':len(encoded['model_view']['states']),
        'model_characters':len(_json(encoded['model_view'])),'audit_characters':len(_json(encoded['audit']))}
