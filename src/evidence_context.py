"""Pure source-evidence context maps; no models, I/O, translation or source edits.

A pack contains only explicit source inputs. Its hashes bind all observations,
fallible readings, background and caller-supplied pool versions. They establish
mechanical consistency, not authenticity or acoustic truth. Map owner coverage
is a checked declaration of input coverage, never an understanding attestation.
Citations are IDs: expansion retrieves original text without model recopying.
The future caller must check actual native prompt/output token capacity.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
import re
from typing import Any, Callable, Sequence

PACK_VERSION = 'source-evidence-context-pack-1'
MAP_VERSION = 'fallible-source-context-map-1'
MAX_CLAIMS = 32
MAX_ALTERNATIVES = 8
RAW_MAP_KEYS = {'pack_sha256', 'coverage_owner_ids', 'claims'}
AUTHORITY = {'hypotheses_only': True, 'source_accuracy_verified': False,
             'acoustic_ownership_verified': False, 'coverage_is_read_attestation': False}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8', errors='strict')).hexdigest()


def _text(value: Any, *, maximum: int | None = None, nonempty: bool = True) -> None:
    _require(isinstance(value, str), 'Expected text')
    _require(not nonempty or bool(value.strip()), 'Empty text')
    _require(maximum is None or len(value) <= maximum, 'Text exceeds declared bound')
    try:
        value.encode('utf-8', errors='strict')
    except UnicodeError as exc:
        raise ValueError('Text is not valid UTF-8') from exc


def _keys(value: Any, required: set[str], optional: set[str] = frozenset()) -> None:
    _require(isinstance(value, dict) and required <= set(value) <= required | optional,
             'Unexpected or missing record fields')


def _ids(value: Any, allowed: set, *, integer: bool = False, minimum: int = 0) -> list:
    _require(isinstance(value, list) and len(value) >= minimum, 'Invalid reference list')
    kind = int if integer else str
    _require(all(type(item) is kind and item in allowed for item in value), 'Unknown or invalid reference')
    _require(len(value) == len(set(value)), 'Duplicate reference')
    return deepcopy(value)


def build_source_pack(source_rows: Sequence[Any], observations: list[dict], frames: dict,
                      original_context: str, *, pool_versions: Sequence[str]) -> dict:
    """Copy SrtBlock dataclasses or {index, ts_line, text} rows into a bound pack.

    Frames use the existing per-owner readings/summary_ja/unknowns/unresolved
    shape. An optional reading quote must be literal; if absent, its complete
    observation text is inserted mechanically. No language heuristic rejects
    Japanese Kanji. No target-text or external-feedback fields are accepted.
    """
    _require(isinstance(source_rows, (list, tuple)) and bool(source_rows), 'Source rows missing')
    source = []
    for row in source_rows:
        value = asdict(row) if is_dataclass(row) and not isinstance(row, type) else deepcopy(row)
        _keys(value, {'index', 'ts_line', 'text'})
        _require(type(value['index']) is int and value['index'] == len(source)+1, 'Source owner sequence changed')
        _text(value['ts_line']); _text(value['text'])
        source.append(value)
    owners = {row['index'] for row in source}; source_by_owner = {row['index']: row for row in source}
    _text(original_context, nonempty=False)
    _require(isinstance(pool_versions, (list, tuple)) and bool(pool_versions), 'Pool versions missing')
    _require(all(isinstance(v, str) and re.fullmatch(r'[0-9a-f]{64}', v) for v in pool_versions), 'Invalid pool version')
    _require(len(pool_versions) == len(set(pool_versions)), 'Duplicate pool version')
    _require(isinstance(observations, list) and bool(observations), 'Observations missing')
    observed = []; by_id = {}; by_owner: dict[int, list[str]] = {owner: [] for owner in owners}
    required = {'observation_id', 'owner_id', 'text', 'crop_start_frame', 'crop_end_frame', 'sample_rate'}
    for item in observations:
        _keys(item, required, {'origin', 'owner_exact'})
        _text(item['observation_id']); _text(item['text'])
        _require(item['observation_id'] not in by_id, 'Duplicate observation ID')
        _require(type(item['owner_id']) is int and item['owner_id'] in owners, 'Unknown observation owner')
        _require(all(type(item[k]) is int for k in ('crop_start_frame', 'crop_end_frame', 'sample_rate'))
            and 0 <= item['crop_start_frame'] < item['crop_end_frame'] and item['sample_rate'] > 0,
            'Invalid observation crop geometry')
        if 'origin' in item:
            _text(item['origin'])
        if 'owner_exact' in item:
            _require(type(item['owner_exact']) is bool, 'Invalid owner scope flag')
        if item.get('origin') == 'immutable_first_asr':
            _require(item['text'] == source_by_owner[item['owner_id']]['text'], 'Raw source observation differs')
        copy = deepcopy(item); observed.append(copy); by_id[copy['observation_id']] = copy
        by_owner[copy['owner_id']].append(copy['observation_id'])
    _require(all(by_owner.values()), 'Observation coverage omits a source owner')
    _require(isinstance(frames, dict) and set(frames) == {str(i) for i in owners}, 'Frame owner coverage changed')
    packed_frames = {}
    frame_keys = {'readings', 'summary_ja', 'unknowns', 'unresolved'}
    for owner in sorted(owners):
        frame = frames[str(owner)]
        _keys(frame, frame_keys, {'source_accuracy_verified', 'literal_observation_coverage_valid'})
        _require(type(frame['unresolved']) is bool, 'Invalid uncertainty flag')
        _text(frame['summary_ja'], nonempty=False)
        _require(isinstance(frame['unknowns'], list), 'Invalid unknowns')
        for unknown in frame['unknowns']:
            _text(unknown)
        if 'source_accuracy_verified' in frame:
            _require(frame['source_accuracy_verified'] is False, 'Frame claims source authority')
        if 'literal_observation_coverage_valid' in frame:
            _require(type(frame['literal_observation_coverage_valid']) is bool, 'Invalid frame coverage flag')
        _require(isinstance(frame['readings'], list), 'Invalid frame readings')
        readings = []; seen = set()
        for reading in frame['readings']:
            _keys(reading, {'observation_id', 'interpretation_ja', 'speech_act', 'viable'}, {'quote'})
            identifier = reading['observation_id']
            _require(isinstance(identifier, str) and identifier in by_id
                and by_id[identifier]['owner_id'] == owner and identifier not in seen,
                'Frame reading has fabricated, duplicate or cross-owner observation')
            _text(reading['interpretation_ja']); _text(reading['speech_act'])
            _require(type(reading['viable']) is bool, 'Invalid reading viability')
            quote = reading.get('quote', by_id[identifier]['text']); _text(quote)
            _require(quote in by_id[identifier]['text'], 'Frame quote is not literal observation text')
            readings.append({**deepcopy(reading), 'quote': quote}); seen.add(identifier)
        _require(seen == set(by_owner[owner]), 'Frame dropped an observation or alternative reading')
        packed_frames[str(owner)] = {**deepcopy(frame), 'readings': readings}
    pack = {'version': PACK_VERSION, 'source_rows': source, 'observations': observed, 'frames': packed_frames,
        'original_context': original_context, 'pool_versions': list(pool_versions), 'authority': deepcopy(AUTHORITY),
        'source_sha256': _hash(source), 'observations_sha256': _hash(observed),
        'frames_sha256': _hash(packed_frames), 'context_sha256': _hash(original_context)}
    return {**pack, 'pack_sha256': _hash(pack)}


def _check_pack(pack: dict) -> None:
    _keys(pack, {'version', 'source_rows', 'observations', 'frames', 'original_context', 'pool_versions',
        'authority', 'source_sha256', 'observations_sha256', 'frames_sha256', 'context_sha256', 'pack_sha256'})
    rebuilt = build_source_pack(pack['source_rows'], pack['observations'], pack['frames'],
        pack['original_context'], pool_versions=pack['pool_versions'])
    _require(_hash(pack) == _hash(rebuilt), 'Source evidence pack or binding changed')


def _object(properties: dict) -> dict:
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def context_schema(pack: dict) -> dict:
    """Bounded JSON schema; runtime must additionally call validate_context_map."""
    _check_pack(pack)
    owners = [row['index'] for row in pack['source_rows']]
    observations = [row['observation_id'] for row in pack['observations']]
    def ids(values: list, kind: str, minimum: int = 0) -> dict:
        return {'type': 'array', 'items': {'type': kind, 'enum': values}, 'minItems': minimum,
                'maxItems': len(values), 'uniqueItems': True}
    branch = _object({'interpretation_ja': {'type': 'string', 'minLength': 1, 'maxLength': 400},
                      'observation_ids': ids(observations, 'string', 1)})
    claim = _object({'claim_id': {'type': 'string', 'pattern': r'^[A-Za-z][A-Za-z0-9_-]{0,63}$'},
        'claim_ja': {'type': 'string', 'minLength': 1, 'maxLength': 600}, 'owner_ids': ids(owners, 'integer', 1),
        'supporting_ids': ids(observations, 'string', 1), 'conflicting_ids': ids(observations, 'string'),
        'uncertain': {'type': 'boolean'}, 'alternatives': {'type': 'array', 'items': branch, 'maxItems': MAX_ALTERNATIVES}})
    return _object({'pack_sha256': {'const': pack['pack_sha256'], 'type': 'string'},
        'coverage_owner_ids': ids(owners, 'integer', len(owners)),
        'claims': {'type': 'array', 'items': claim, 'maxItems': MAX_CLAIMS}})


def validate_context_map(value: dict, pack: dict) -> dict:
    """Validate references and bind a fallible map; never verify its meanings."""
    _check_pack(pack); _keys(value, RAW_MAP_KEYS)
    _require(value['pack_sha256'] == pack['pack_sha256'], 'Map belongs to another source evidence pack')
    owners = {row['index'] for row in pack['source_rows']}
    observations = {row['observation_id']: row for row in pack['observations']}
    covered = _ids(value['coverage_owner_ids'], owners, integer=True, minimum=len(owners))
    _require(set(covered) == owners, 'Map input coverage declaration is incomplete')
    _require(isinstance(value['claims'], list) and len(value['claims']) <= MAX_CLAIMS, 'Too many or invalid claims')
    seen = set(); claim_keys = {'claim_id', 'claim_ja', 'owner_ids', 'supporting_ids', 'conflicting_ids', 'uncertain', 'alternatives'}
    for claim in value['claims']:
        _keys(claim, claim_keys)
        identifier = claim['claim_id']
        _require(isinstance(identifier, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', identifier)
            and identifier not in seen, 'Invalid or duplicate claim ID')
        seen.add(identifier); _text(claim['claim_ja'], maximum=600)
        declared = _ids(claim['owner_ids'], owners, integer=True, minimum=1)
        support = _ids(claim['supporting_ids'], set(observations), minimum=1)
        conflicts = _ids(claim['conflicting_ids'], set(observations))
        _require(not set(support) & set(conflicts), 'Support and conflict roles overlap in one claim')
        _require(type(claim['uncertain']) is bool, 'Invalid claim uncertainty flag')
        _require(isinstance(claim['alternatives'], list) and len(claim['alternatives']) <= MAX_ALTERNATIVES,
                 'Too many or invalid alternatives')
        cited = set(support) | set(conflicts)
        for branch in claim['alternatives']:
            _keys(branch, {'interpretation_ja', 'observation_ids'})
            _text(branch['interpretation_ja'], maximum=400)
            cited.update(_ids(branch['observation_ids'], set(observations), minimum=1))
        _require(set(declared) == {observations[key]['owner_id'] for key in cited},
                 'Claim owner IDs do not exactly cover all cited evidence')
    bound = {'version': MAP_VERSION, **deepcopy(value), 'pool_versions': deepcopy(pack['pool_versions']),
             'authority': deepcopy(AUTHORITY)}
    return {**bound, 'map_sha256': _hash(bound)}


def expand_claim(context_map: dict, claim_id: str, case_owner_id: int, pack: dict) -> dict:
    """Return all cited raw observations plus all case-owner observations.

    Cross-owner conflicts and every alternative branch remain present. No
    relevance filter, winner selection, source rewrite or text normalization is
    performed. Overlapping crop content remains an uncertain ownership hint.
    """
    _keys(context_map, RAW_MAP_KEYS | {'version', 'pool_versions', 'authority', 'map_sha256'})
    rebuilt = validate_context_map({key: context_map[key] for key in RAW_MAP_KEYS}, pack)
    _require(_hash(context_map) == _hash(rebuilt), 'Context map or pool binding changed')
    owners = {row['index'] for row in pack['source_rows']}
    _require(type(case_owner_id) is int and case_owner_id in owners, 'Unknown case owner')
    _require(isinstance(claim_id, str), 'Invalid claim ID')
    matches = [claim for claim in context_map['claims'] if claim['claim_id'] == claim_id]
    _require(len(matches) == 1, 'Unknown claim ID')
    claim = matches[0]
    cited = set(claim['supporting_ids']) | set(claim['conflicting_ids'])
    for branch in claim['alternatives']:
        cited.update(branch['observation_ids'])
    case_ids = [row['observation_id'] for row in pack['observations'] if row['owner_id'] == case_owner_id]
    expanded = [deepcopy(row) for row in pack['observations'] if row['observation_id'] in cited | set(case_ids)]
    return {'pack_sha256': pack['pack_sha256'], 'map_sha256': context_map['map_sha256'],
        'pool_versions': deepcopy(pack['pool_versions']), 'case_owner_id': case_owner_id,
        'case_observation_ids': case_ids, 'claim': deepcopy(claim), 'observations': expanded,
        'authority': deepcopy(AUTHORITY)}
