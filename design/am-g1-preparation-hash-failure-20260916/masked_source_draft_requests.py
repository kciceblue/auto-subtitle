"""AU-G1 layout with a distinct mask-corrected source collection.

No audio, model or reviewer access. Native execution is authenticated by the
caller; the legacy projection below checks common source shape only and is never
sent to the writer or saved as an acquisition receipt.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import PurePosixPath
from src import auto_source_draft_requests as automatic
from src import compact_raw_draft_requests as compact
from src import compact_raw_evidence as codec
from src import evidence_context as ec
from src.workflow_state import fingerprint

VERSION = 'masked-source-draft-requests-1'
OBSERVATIONS_VERSION = 'masked-source-observations-1'
ATTENTION_POLICY = {'version':'qwen-audio-window-mask-1','change':'encoder_window_mask_passed_to_every_layer',
                    'backend_changed':False,'independent_recognizer':False,'accuracy_verified':False}
ENCODER_CONFIG = {'backend':'sdpa','layers':24,'hidden_size':1024,'heads':16,'head_dim':64,
                  'n_window':50,'n_window_infer':800,'max_source_positions':1500}
OWNER_IDS = automatic.OWNER_IDS
MAX_CHINESE_CHARS = automatic.MAX_CHINESE_CHARS
SETTINGS = deepcopy(automatic.SETTINGS)
writer_schema = automatic.writer_schema
validate_writer_response = automatic.validate_writer_response
BACKEND_DEFINITION = (
    'automatic_source的masked-source-observations-1版本是同一Qwen语音识别器在修正音频编码器'
    '分块注意力掩码后生成的另一次无提示、自动语言观察。audio_attention记录该执行方式，'
    '不是独立识别器，也不证明新文字更正确；旧证据和原假说仍全部保留，不能自动优先新观察。'
    '除这一实际后端说明外，正文布局、以下任务说明和输出约束不变。\n\n'
)
WRITER_INSTRUCTION = BACKEND_DEFINITION + automatic.WRITER_INSTRUCTION


def _same(a,b): return fingerprint(a) == fingerprint(b)


def _attention(value, rows):
    ec._keys(value,{'policy','source_identity','encoder_config','model_config','control_receipts','observation_attestations'})
    ec._require(_same(value['policy'],ATTENTION_POLICY) and _same(value['encoder_config'],ENCODER_CONFIG),
                'Declared attention correction/backend changed')
    identity = value['source_identity']; ec._keys(identity,{'files','versions'})
    ec._require(type(identity['files']) is dict and bool(identity['files']),'Attention source identity missing')
    for path,digest in identity['files'].items():
        ec._require(type(path) is str and PurePosixPath(path).is_absolute(),'Attention code path missing')
        automatic._hash(digest)
    ec._keys(identity['versions'],{'qwen-asr','transformers','torch'})
    for version in identity['versions'].values():
        ec._require(type(version) is str and bool(version),'Attention runtime version missing')
    model = value['model_config']; ec._keys(model,{'path','sha256'})
    ec._require(type(model['path']) is str and PurePosixPath(model['path']).is_absolute(),'Model configuration identity missing')
    automatic._hash(model['sha256'])
    controls = value['control_receipts']
    ec._require(type(controls) is list and len(controls) == 2,'Both native negative controls required')
    paths = {row['receipt']['path'] for row in rows}
    for name,control in zip(('silence','noise'),controls):
        ec._keys(control,{'slot_id','path','sha256','attention_sha256'})
        ec._require(control['slot_id'] == name and type(control['path']) is str
            and PurePosixPath(control['path']).is_absolute() and control['path'] not in paths,'Control receipt identity changed')
        paths.add(control['path'])
        for key in ('sha256','attention_sha256'): automatic._hash(control[key])
    attestations = value['observation_attestations']
    ec._require(type(attestations) is list and len(attestations) == len(rows),'Every source mask attestation required')
    for row,attestation in zip(rows,attestations):
        ec._keys(attestation,{'observation_id','attention_sha256'})
        ec._require(attestation['observation_id'] == row['observation_id'],'Attention observation association changed')
        automatic._hash(attestation['attention_sha256'])


def _masked(value, source):
    ec._keys(value,automatic._ENVELOPE | {'audio_attention'})
    ec._require(value['version'] == OBSERVATIONS_VERSION,'Source collection treatment version differs')
    ec._require(type(value['observations']) is list and len(value['observations']) == len(OWNER_IDS),
                'All 66 mask-corrected observations are required')
    for owner,row in zip(OWNER_IDS,value['observations']):
        ec._require(row.get('observation_id') == f'masked-source-owner-{owner:03d}', 'Masked source ID changed')
    ec._require(value['observations_sha256'] == ec._hash(value['observations']), 'Masked observation hash changed')
    # The unchanged legacy validator checks exact scopes, literals, statuses,
    # source joins, hash fields and unhinted/automatic conditioning. Only its
    # hardcoded version/ID spelling is projected; actual request bytes stay new.
    shadow = {key:deepcopy(value[key]) for key in automatic._ENVELOPE}
    shadow['version'] = automatic.OBSERVATIONS_VERSION
    for owner,row in zip(OWNER_IDS,shadow['observations']): row['observation_id'] = f'auto-source-owner-{owner:03d}'
    shadow['observations_sha256'] = ec._hash(shadow['observations'])
    empty = automatic._validate_automatic(shadow,source)
    _attention(value['audio_attention'],value['observations'])
    return empty


def validate_writer_request(request):
    ec._keys(request,{'instruction','body','schema'})
    ec._require(request['instruction'] == WRITER_INSTRUCTION and _same(request['schema'],writer_schema()),
                'Masked-source task/schema changed')
    body = request['body']; ec._keys(body,{'compact_evidence','automatic_source'})
    b = {'instruction':compact.WRITER_INSTRUCTION,'body':body['compact_evidence'],'schema':request['schema']}
    counts = compact.validate_writer_request(b)
    ec._require(counts['observations'] == 799,'All original B observations must survive')
    decoded = codec.decode_body(b['body'])
    empty = _masked(body['automatic_source'],decoded['source_evidence']['source_rows'])
    return {**counts,'automatic_observations':66,'automatic_empty':empty,'total_observation_records':865,
            'source_treatment':OBSERVATIONS_VERSION}


def build_writer_request(b_request, observations):
    compact.validate_writer_request(b_request)
    request = {'instruction':WRITER_INSTRUCTION,'body':{'compact_evidence':deepcopy(b_request['body']),
        'automatic_source':deepcopy(observations)},'schema':deepcopy(b_request['schema'])}
    validate_writer_request(request); return request


def validate_extension(request,b_request,observations):
    ec._require(_same(request,build_writer_request(b_request,observations)), 'Writer differs from independently bound inputs')
    return validate_writer_request(request)
