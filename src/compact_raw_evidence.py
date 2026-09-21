"""Reversible grouped encoding of the validated A1 raw writer body.

Only metadata is interned. Each observation ID and its complete literal remain
inline under their actual view; equal texts, views and observer families are
never merged. Definition references are zero based and use first-use order.
Array order, optional-key presence, nulls and exact time strings survive decode.
JSON object member order is not an identity claim.

``decode_body`` checks structure, geometry and canonical encoding, not source
authenticity. ``validate_body`` additionally compares the decoded actual prompt
with a fresh request built from independently pinned source inputs. There is no
retained-literal backup, audit override, I/O, or model call in this module.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from src import evidence_context as ec
from src import raw_evidence_draft_requests as raw
from src.workflow_state import fingerprint

VERSION = 'compact-raw-evidence-1'
_ATTRIBUTES = {'original_observation_id', 'origin', 'owner_exact'}
_COLUMNS = {
    'source_rows': ['index', 'ts_line', 'text'],
    'observers': ['id', 'family'],
    'view_types': ['kind', 'transform'],
    'views': ['id', 'owner_id', 'view_type_ref', 'sample_rate_ref', 'crop_seconds',
              'core_seconds', 'owner_intersection_seconds', 'duplicate_geometry', 'readings'],
    'readings': ['observation_id', 'observer_instance', 'text', 'attributes_ref'],
}
_COPIED = ('authority', 'temporal_authority', 'scope_rules', 'additional_literals')
_KEYS = {'version', 'reference_index_base', 'columns', 'source_rows', 'original_context',
         'observers', 'view_types', 'sample_rates', 'reading_attributes', 'views',
         'focus_owner_ids', *_COPIED}


def _same(left: Any, right: Any) -> bool:
    return fingerprint(left) == fingerprint(right)


def _validate_raw(body: dict) -> dict[str, int]:
    return raw.validate_writer_request({'instruction': raw.WRITER_INSTRUCTION,
                                       'body': body, 'schema': raw.writer_schema()})


def _row(value: Any, width: int) -> list:
    ec._require(isinstance(value, list) and len(value) == width, 'Invalid compact row width')
    return value


def _table(value: Any) -> list:
    ec._require(isinstance(value, list) and bool(value), 'Missing compact definition or row table')
    return value


def _reference(value: Any, table: list) -> Any:
    ec._require(type(value) is int and 0 <= value < len(table), 'Unknown compact definition reference')
    return table[value]


def _encode(body: dict) -> dict:
    evidence = body['source_evidence']
    observers: list[list] = []; observer_ids: set[str] = set()
    view_types: list[list] = []; rates: list[int] = []; attributes: list[dict] = []
    registries: dict[str, dict[str, int]] = {'types': {}, 'rates': {}, 'attributes': {}}

    def intern(table: list, registry: str, value: Any) -> int:
        key = fingerprint(value); known = registries[registry]
        if key not in known:
            known[key] = len(table); table.append(deepcopy(value))
        return known[key]

    views = []
    for view in evidence['views']:
        kind = intern(view_types, 'types', [view['kind'], view['transform']])
        rate = intern(rates, 'rates', view['sample_rate'])
        readings = []
        for reading in view['readings']:
            observer = reading['observer_instance']
            if observer not in observer_ids:
                observers.append([observer, reading['family']]); observer_ids.add(observer)
            optional = {key: value for key, value in reading.items() if key in _ATTRIBUTES}
            attr = intern(attributes, 'attributes', optional)
            readings.append([reading['observation_id'], observer, reading['text'], attr])
        views.append([view['id'], view['owner_id'], kind, rate,
                      deepcopy(view['crop_seconds']), deepcopy(view['core_seconds']),
                      deepcopy(view['owner_intersection_seconds']), view['duplicate_geometry'], readings])
    return {'version': VERSION, 'reference_index_base': 0, 'columns': deepcopy(_COLUMNS),
            'source_rows': [[row[key] for key in _COLUMNS['source_rows']] for row in evidence['source_rows']],
            'original_context': evidence['original_context'], 'observers': observers,
            'view_types': view_types, 'sample_rates': rates, 'reading_attributes': attributes,
            'views': views, **{key: deepcopy(evidence[key]) for key in _COPIED},
            'focus_owner_ids': deepcopy(body['focus_owner_ids'])}


def encode_body(body: dict) -> dict:
    """Encode an A1 body after closed raw-input validation; leave it untouched."""
    try:
        _validate_raw(body)
        return _encode(body)
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError('Malformed raw evidence body') from error


def _decode(compact: dict) -> dict:
    ec._keys(compact, _KEYS)
    ec._require(compact['version'] == VERSION and type(compact['reference_index_base']) is int
                and compact['reference_index_base'] == 0, 'Unknown compact encoding version or index base')
    ec._require(_same(compact['columns'], _COLUMNS), 'Compact column definitions changed')
    observers = {}
    for item in _table(compact['observers']):
        identifier, family = _row(item, 2); ec._text(identifier)
        ec._require(identifier not in observers, 'Duplicate compact observer ID')
        ec._require(family in {None, 'zipformer-ja', 'qwen3-asr-1.7b'}, 'Unknown compact observer family')
        observers[identifier] = family
    kinds = _table(compact['view_types'])
    for item in kinds: _row(item, 2)
    rates = _table(compact['sample_rates'])
    ec._require(all(type(rate) is int and rate > 0 for rate in rates), 'Invalid compact sample rate')
    attributes = _table(compact['reading_attributes'])
    for item in attributes:
        ec._keys(item, set(), _ATTRIBUTES)
        for name in ('original_observation_id', 'origin'):
            if name in item: ec._text(item[name])
        if 'owner_exact' in item:
            ec._require(type(item['owner_exact']) is bool, 'Invalid compact optional owner flag')
    source = [dict(zip(_COLUMNS['source_rows'], deepcopy(_row(item, 3))))
              for item in _table(compact['source_rows'])]
    views = []
    for item in _table(compact['views']):
        identifier, owner, kind_ref, rate_ref, crop, core, intersection, duplicate, encoded_readings = _row(item, 9)
        kind, transform = _reference(kind_ref, kinds)
        rate = _reference(rate_ref, rates)
        readings = []
        for item in _table(encoded_readings):
            observation, observer, literal, attribute_ref = _row(item, 4)
            ec._require(isinstance(observer, str) and observer in observers, 'Unknown compact observer reference')
            optional = _reference(attribute_ref, attributes)
            readings.append({'observation_id': observation, 'owner_id': owner, 'text': literal,
                             'observer_instance': observer, 'family': observers[observer], **deepcopy(optional)})
        views.append({'id': identifier, 'owner_id': owner, 'kind': kind, 'transform': transform,
                      'sample_rate': rate, 'crop_seconds': deepcopy(crop), 'core_seconds': deepcopy(core),
                      'owner_intersection_seconds': deepcopy(intersection), 'duplicate_geometry': duplicate,
                      'readings': readings})
    evidence = {'source_rows': source, 'original_context': compact['original_context'],
                **{key: deepcopy(compact[key]) for key in _COPIED}, 'views': views}
    body = {'source_evidence': evidence, 'focus_owner_ids': deepcopy(compact['focus_owner_ids'])}
    _validate_raw(body)
    # Reject unused/duplicate definitions and alternative table ordering, rather
    # than silently repairing a model-visible table or sorting supplied rows.
    ec._require(_same(compact, _encode(body)), 'Noncanonical compact definitions or references')
    return body


def decode_body(compact: dict) -> dict:
    """Reconstruct actual inline literals; reject malformed or noncanonical input.

    A structurally valid substituted literal can decode. Use ``validate_body``
    with independent source inputs before treating the result as authenticated.
    """
    try:
        return _decode(compact)
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError('Malformed compact raw evidence body') from error


def validate_body(compact: dict, pack: dict, bound: dict, identity_map: dict,
                  *, projection: dict | None = None) -> dict[str, int]:
    """Bind decoded prompt text/identity/scope to independently rebuilt A1 input."""
    decoded = decode_body(compact)
    expected = raw.build_writer_request(pack, bound, identity_map, projection=projection)['body']
    ec._require(_same(decoded, expected), 'Compact body differs from source-bound raw writer input')
    coverage = _validate_raw(decoded)
    return {**coverage, 'additional_literals': len(decoded['source_evidence']['additional_literals'])}
