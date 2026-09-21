"""Pure AU-G2 extension; preserve AU-G1 and add all nine imposed-language records.

Callers authenticate the original AU request and acquisition native files.
Hashes/shape checks cannot prove native execution or correctness. No file I/O,
filtering, text normalization, prior Chinese, score or review inputs occur here.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import PurePosixPath

from src import auto_source_draft_requests as automatic
from src import evidence_context as ec
from src.workflow_state import fingerprint

VERSION = 'forced-source-draft-requests-1'
OBSERVATIONS_VERSION = 'forced-source-observations-1'
ELIGIBLE_OWNER_IDS = (3, 4, 5, 6, 11, 44, 45, 59, 64)
OWNER_IDS = automatic.OWNER_IDS
MAX_CHINESE_CHARS = automatic.MAX_CHINESE_CHARS
SETTINGS = deepcopy(automatic.SETTINGS)
writer_schema = automatic.writer_schema
validate_writer_response = automatic.validate_writer_response
_EXTENSION = (
    '本正文的automatic_evidence完整保留上一次请求正文。以下原任务说明中的compact_evidence和'
    'automatic_source均位于automatic_evidence内，其文字、编号、范围和全部观察保持不变。'
    '另一个独立部分forced_source保留9条额外的强制日语识别观察；所有9条都要参考，不能替换、'
    '覆盖或丢弃原799条证据及66条自动语言观察。'
    'forced_source.conditioning.language、forced_language和language_origin明确表示施加的语言条件。'
    'parser_language来自强制设置下的解析器，不能当作模型独立检测到日语的证据。'
    '这些记录与parent_observation_id指定的自动记录使用同一Qwen识别器和完全相同的音频范围，'
    '不是新增的独立识别器投票。强制语言设置已在合成噪声控制中产生过非空文字；因此强制输出、'
    '与背景或其他读法的表面一致均不证明听写正确，不能自动优先采用。'
    '保留raw_text、text及其不确定性；空文字不证明无语音，强制输出中的语言标记也不转换为静音结论。'
    '禁止据此推定词句归属、说话人身份或源字幕错误。仍只执行以下原有完整中文翻译任务和原有JSON格式，'
    '不要输出解释、评分、来源修订或额外字段。\n\n'
)
WRITER_INSTRUCTION = _EXTENSION + automatic.WRITER_INSTRUCTION
_ENVELOPE = automatic._ENVELOPE | {'automatic_source_sha256', 'parent_observations_sha256', 'eligible_owner_ids'}
_ROW = (automatic._ROW - {'detected_language'}) | {
    'parent_observation_id', 'parser_language', 'forced_language', 'language_origin'}
_SAME_FIELDS = {'owner_id', 'owner_ts_line', 'owner_start_frame', 'owner_end_frame',
    'crop_start_frame', 'crop_end_frame', 'sample_rate', 'clamped_tail_frames', 'vad', 'audio_sha256', 'pcm_sha256'}


def _same(a, b): return fingerprint(a) == fingerprint(b)


def _forced(value, parent):
    ec._keys(value, _ENVELOPE)
    eligible = [row['owner_id'] for row in parent['observations']
                if row['native_status'] == 'complete' and row['text'] != ''
                and row['detected_language'] not in ('', 'Japanese') and row['vad']['detected_frames'] > 0]
    ec._require(eligible == list(ELIGIBLE_OWNER_IDS), 'Forced eligibility differs from the fixed native metadata rule')
    ec._require(value['version'] == OBSERVATIONS_VERSION
        and _same(value['eligible_owner_ids'], list(ELIGIBLE_OWNER_IDS)), 'Wrong forced-source owner set')
    for key in ('plan_sha256', 'source_rows_sha256', 'original_mono_sha256', 'observations_sha256',
                'automatic_source_sha256', 'parent_observations_sha256'): automatic._hash(value[key])
    ec._require(value['automatic_source_sha256'] == ec._hash(parent)
        and value['parent_observations_sha256'] == parent['observations_sha256']
        and all(_same(value[key], parent[key]) for key in
                ('source_rows_sha256', 'original_mono_sha256', 'original_mono_frames', 'sample_rate')),
        'Forced source is not bound to the preserved automatic collection')
    ec._require(_same(value['conditioning'], {'context': '', 'language': 'Japanese', 'hotwords': []}),
                'Forced source must contain only the declared imposed Japanese condition')
    identity = value['execution_identity']; ec._keys(identity, automatic._IDENTITY)
    for key in automatic._IDENTITY - {'reserved_tokens'}: automatic._hash(identity[key])
    ec._require(all(_same(identity[key], parent['execution_identity'][key])
                   for key in automatic._IDENTITY - {'worker_sha256'}),
                'Forced source recognizer/runtime/parser/tokenizer differs from automatic source')
    rows = value['observations']
    ec._require(type(rows) is list and len(rows) == len(ELIGIBLE_OWNER_IDS)
        and value['observations_sha256'] == ec._hash(rows), 'Every forced-source record must survive exactly')
    paths = {row['receipt']['path'] for row in parent['observations']}; empty = 0
    for owner, row in zip(ELIGIBLE_OWNER_IDS, rows):
        ec._keys(row, _ROW); original = parent['observations'][owner-1]
        ec._require(type(row['owner_id']) is int and row['owner_id'] == owner
            and row['observation_id'] == f'forced-source-owner-{owner:03d}'
            and row['parent_observation_id'] == original['observation_id']
            and all(_same(row[key], original[key]) for key in _SAME_FIELDS),
            'Forced source owner, original PCM or exact sample scope changed')
        for key in ('raw_text', 'text', 'parser_language'): automatic._string(row[key])
        raw, text = row['raw_text'], row['text']
        category = 'empty_raw' if not raw.strip() else 'nonempty_transcript' if text else 'unrecognized_empty_protocol'
        ec._require(row['forced_language'] == 'Japanese' and row['language_origin'] == 'forced'
            and row['parser_language'] == ('Japanese' if text else '')
            and row['native_status'] == ('complete' if text else 'empty') and row['protocol_category'] == category
            and (category != 'empty_raw' or text == ''), 'Imposed-language parser/status claim changed')
        empty += text == ''
        receipt = row['receipt']; ec._keys(receipt, {'path', 'sha256', 'request_sha256'})
        ec._require(type(receipt['path']) is str and PurePosixPath(receipt['path']).is_absolute()
                    and receipt['path'] not in paths, 'Forced receipt is missing, duplicated or reused from automatic ASR')
        paths.add(receipt['path'])
        for key in ('sha256', 'request_sha256'): automatic._hash(receipt[key])
    return empty


def _validate(request):
    ec._keys(request, {'instruction', 'body', 'schema'})
    ec._require(request['instruction'] == WRITER_INSTRUCTION and _same(request['schema'], writer_schema()),
                'Forced-source writer instruction or schema changed')
    body = request['body']; ec._keys(body, {'automatic_evidence', 'forced_source'})
    parent_request = {'instruction': automatic.WRITER_INSTRUCTION, 'body': body['automatic_evidence'], 'schema': request['schema']}
    counts = automatic.validate_writer_request(parent_request)
    empty = _forced(body['forced_source'], body['automatic_evidence']['automatic_source'])
    return {**counts, 'forced_observations': len(ELIGIBLE_OWNER_IDS), 'forced_empty': empty,
            'total_observation_records': counts['total_observation_records']+len(ELIGIBLE_OWNER_IDS)}


def validate_writer_request(request):
    """Validate closed visible shape; caller authenticates original/native receipts."""
    try: return _validate(request)
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError('Malformed forced-source writer request') from exc


def build_writer_request(auto_request, forced_source):
    """Add the exact conditioned collection to independently pinned AU-G1 inputs."""
    automatic.validate_writer_request(auto_request)
    request = {'instruction': WRITER_INSTRUCTION,
               'body': {'automatic_evidence': deepcopy(auto_request['body']), 'forced_source': deepcopy(forced_source)},
               'schema': deepcopy(auto_request['schema'])}
    validate_writer_request(request)
    return request


def validate_extension(request, auto_request, forced_source):
    """Compare actual model-visible contents with separately authenticated inputs."""
    ec._require(_same(request, build_writer_request(auto_request, forced_source)),
                'Forced-source extension differs from independent inputs')
    return validate_writer_request(request)
