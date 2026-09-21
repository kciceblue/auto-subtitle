"""Pure writer-only ablation of R1's generated frames and source map.

The source-bound entry points independently rebuild the frozen R request. The
sidecar contains only removed fields/clauses, never replacement retained text.
No file access, inference, campaign registration, or scoring occurs here.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from src import readable_draft_requests as original
from src import readable_evidence as readable
from src import evidence_context as ec
from src.workflow_state import fingerprint

VERSION = 'raw-evidence-draft-requests-1'
OWNER_IDS = original.OWNER_IDS
MAX_CHINESE_CHARS = original.MAX_CHINESE_CHARS
writer_schema = original.writer_schema
build_verifier_request = original.build_verifier_request
build_global_request = original.build_global_request
VERIFIER_INSTRUCTION = original.VERIFIER_INSTRUCTION
GLOBAL_INSTRUCTION = original.GLOBAL_INSTRUCTION
_REMOVED_FIELDS = {'frames', 'source_map'}
_EVIDENCE_KEYS = {'source_rows', 'original_context', 'views', 'authority',
                  'temporal_authority', 'scope_rules', 'additional_literals'}
_EVIDENCE_ORDER = ('source_rows', 'original_context', 'frames', 'source_map',
                   'authority', 'temporal_authority', 'scope_rules', 'additional_literals', 'views')
# These exact spans alone are removed. No replacement source-specific wording.
_REMOVED_CLAUSES = (
    'source_evidence.frames保留每个编号的summary_ja、unknowns及不确定性；这里延续既有写作输入边界，不提供辅助逐条reading解释，但并未删除任何原始声音文字观察。',
    'source_evidence.source_map是完整冻结的暂定源地图，保留支持、冲突、所有替代分支及不确定性；其中结构化observation_ids、supporting_ids和conflicting_ids使用与readings相同的E编号。',
    '地图文字内出现的其他字符串仍按原文理解，不自行当作结构化引用。',
    '、每个编号的摘要与unknowns，以及完整地图',
    '地图、摘要与',
)


def _same(left: Any, right: Any) -> bool:
    return fingerprint(left) == fingerprint(right)


def _instruction_clauses() -> list[dict]:
    spans = []
    for text in _REMOVED_CLAUSES:
        ec._require(original.WRITER_INSTRUCTION.count(text) == 1, 'Frozen R instruction clause changed')
        spans.append({'offset': original.WRITER_INSTRUCTION.index(text), 'text': text})
    spans.sort(key=lambda row: row['offset'])
    ec._require(all(left['offset']+len(left['text']) <= right['offset']
                    for left, right in zip(spans, spans[1:])), 'Instruction removals overlap')
    return spans


def _raw_instruction() -> str:
    result = original.WRITER_INSTRUCTION
    for span in reversed(_instruction_clauses()):
        start = span['offset']; result = result[:start]+result[start+len(span['text']):]
    return result


WRITER_INSTRUCTION = _raw_instruction()


def validate_writer_request(request: dict) -> dict[str, int]:
    """Check closed model-input shape; source authenticity needs validate_ablation."""
    ec._keys(request, {'instruction', 'body', 'schema'})
    ec._require(request['instruction'] == WRITER_INSTRUCTION
                and _same(request['schema'], writer_schema()), 'Raw writer instruction or all-owner schema changed')
    body = request['body']; ec._keys(body, {'source_evidence', 'focus_owner_ids'})
    ec._require(_same(body['focus_owner_ids'], list(OWNER_IDS)), 'Raw writer owner coverage changed')
    evidence = body['source_evidence']; ec._keys(evidence, _EVIDENCE_KEYS)
    ec._require(_same(evidence['authority'], ec.AUTHORITY)
        and _same(evidence['temporal_authority'], {'word_ownership_inferred': False, 'acoustic_truth_verified': False})
        and _same(evidence['scope_rules'], readable._RULES), 'Raw evidence capability or correlation rules changed')
    ec._text(evidence['original_context'], nonempty=False)
    rows = evidence['source_rows']
    ec._require(isinstance(rows, list) and len(rows) == len(OWNER_IDS), 'Raw source owner rows missing')
    for owner, row in zip(OWNER_IDS, rows):
        ec._keys(row, {'index', 'ts_line', 'text'})
        ec._require(type(row['index']) is int and row['index'] == owner, 'Raw source order changed')
        ec._text(row['ts_line']); ec._text(row['text'])
        bounds = row['ts_line'].split(' --> ')
        ec._require(len(bounds) == 2 and readable.temporal._milliseconds(bounds[0])
                    < readable.temporal._milliseconds(bounds[1]), 'Invalid raw owner interval')
    ec._require(isinstance(evidence['additional_literals'], dict), 'Raw additional literals missing')
    for key, value in evidence['additional_literals'].items():
        ec._text(key); ec._text(value)
    views = evidence['views']; ec._require(isinstance(views, list) and bool(views), 'Raw views missing')
    seen_views: set[str] = set(); seen_observations: set[str] = set(); observers: dict[str, Any] = {}
    for view in views:
        ec._keys(view, readable._VIEW_KEYS | {'readings'})
        identifier = view['id']; ec._text(identifier)
        owner = view['owner_id']; rate = view['sample_rate']
        ec._require(identifier not in seen_views and type(owner) is int and owner in OWNER_IDS
            and type(rate) is int and rate > 0 and type(view['duplicate_geometry']) is bool
            and view['kind'] in readable._KINDS, 'Raw view identity or geometry changed')
        seen_views.add(identifier)
        for key in ('crop_seconds', 'core_seconds', 'owner_intersection_seconds'):
            interval = view[key]
            ec._require((interval is None and key != 'crop_seconds')
                or isinstance(interval, list) and len(interval) == 2, 'Invalid exact raw interval')
            if interval is not None:
                ec._require(readable._fraction(interval[0]) < readable._fraction(interval[1]), 'Nonpositive raw interval')
        start, end = [readable._frames(value, rate) for value in view['crop_seconds']]
        ec._require(_same(view['owner_intersection_seconds'], readable._intersection(start, end, rate, rows[owner-1])),
                    'Raw owner intersection changed')
        core = view['core_seconds']
        if view['kind'] == 'raw_short':
            ec._require(core is not None, 'Short view core missing')
            a, b = [readable._frames(value, rate) for value in core]
            ec._require(start <= a < b <= end, 'Raw core exceeds crop')
        else:
            ec._require(core is None, 'Unexpected raw view core')
        transform = None if view['kind'] == 'original_transcript' else (
            'raw' if view['kind'].startswith('raw') else view['kind'])
        ec._require(view['transform'] == transform, 'Raw transform changed')
        readings = view['readings']; ec._require(isinstance(readings, list) and bool(readings), 'Raw readings missing')
        local_ids = []
        for reading in readings:
            ec._keys(reading, readable._READING_KEYS, {'origin', 'owner_exact', 'original_observation_id'})
            key = reading['observation_id']; ec._text(key); ec._text(reading['text'])
            observer = reading['observer_instance']; ec._text(observer)
            family = reading['family']
            ec._require(key not in seen_observations and type(reading['owner_id']) is int
                and reading['owner_id'] == owner and family in {None, 'zipformer-ja', 'qwen3-asr-1.7b'},
                'Raw observation identity or owner changed')
            ec._require(observer not in observers or observers[observer] == family, 'Observer family changed')
            observers[observer] = family; seen_observations.add(key); local_ids.append(key)
            for optional in ('origin', 'original_observation_id'):
                if optional in reading: ec._text(reading[optional])
            if 'owner_exact' in reading: ec._require(type(reading['owner_exact']) is bool, 'Invalid raw owner flag')
            if view['kind'] == 'original_transcript':
                ec._require(family is None and reading.get('origin') == 'immutable_first_asr'
                    and reading.get('owner_exact') is True and view['duplicate_geometry'] is False,
                    'Original recognizer identity was inferred')
            else:
                ec._require(family is not None and reading.get('origin') != 'immutable_first_asr', 'Native observer identity missing')
        ec._require(local_ids == sorted(local_ids), 'Raw reading order changed')
    ec._require(views == sorted(views, key=readable._view_order), 'Raw view chronology changed')
    return {'source_owners': len(rows), 'observations': len(seen_observations), 'views': len(views),
            'observers': len(observers)}


def reconstruct_request(request: dict, audit: dict) -> dict:
    """Restore removed values; this alone does not authenticate caller inputs."""
    validate_writer_request(request)
    ec._keys(audit, {'removed_fields', 'removed_instruction_clauses'})
    ec._keys(audit['removed_fields'], _REMOVED_FIELDS)
    ec._require(_same(audit['removed_instruction_clauses'], _instruction_clauses()), 'Removed instruction clauses changed')
    restored = deepcopy(request); retained = restored['body']['source_evidence']
    fields = audit['removed_fields']
    restored['body']['source_evidence'] = {
        key: deepcopy(fields[key] if key in _REMOVED_FIELDS else retained[key]) for key in _EVIDENCE_ORDER}
    pieces = []; cursor = 0; removed = 0
    for span in audit['removed_instruction_clauses']:
        offset = span['offset']-removed
        pieces.extend([request['instruction'][cursor:offset], span['text']])
        cursor = offset; removed += len(span['text'])
    pieces.append(request['instruction'][cursor:]); restored['instruction'] = ''.join(pieces)
    ec._require(restored['instruction'] == original.WRITER_INSTRUCTION, 'Original R instruction did not reconstruct')
    return restored


def validate_ablation(request: dict, audit: dict, pack: dict, bound: dict, identity_map: dict,
                      *, projection: dict | None = None) -> dict[str, int]:
    """Authenticate every retained/removed value against independently rebuilt R input."""
    expected = original.build_writer_request(pack, bound, identity_map, projection=projection)
    ec._require(_same(reconstruct_request(request, audit), expected), 'Ablation differs from source-bound original R request')
    return validate_writer_request(request)


def build_ablation(pack: dict, bound: dict, identity_map: dict,
                   *, projection: dict | None = None) -> dict:
    original_request = original.build_writer_request(pack, bound, identity_map, projection=projection)
    request = deepcopy(original_request); evidence = request['body']['source_evidence']
    audit = {'removed_fields': {key: evidence.pop(key) for key in sorted(_REMOVED_FIELDS)},
             'removed_instruction_clauses': _instruction_clauses()}
    request['instruction'] = WRITER_INSTRUCTION
    ec._require(_same(reconstruct_request(request, audit), original_request), 'Raw ablation round trip changed')
    return {'version': VERSION, 'request': request, 'audit': audit}


def build_writer_request(pack: dict, bound: dict, identity_map: dict,
                         *, projection: dict | None = None) -> dict:
    """Return only the model request; build_ablation also returns its audit sidecar."""
    return build_ablation(pack, bound, identity_map, projection=projection)['request']
