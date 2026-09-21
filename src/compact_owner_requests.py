"""Pure W2 writer representation; W1 translation and verification rules unchanged.

The envelope's ``body`` contains JSON objects ``{"$o": n}`` and ``{"$s": n}``
only where a string was replaced. They look up zero-based entries in
``table.observation_ids`` and ``table.strings`` respectively. Table entries are
literal, complete strings, never another reference. Ordinary strings, lists,
keys, numbers and null values retain their original meaning. Both tables are
sorted by the exact Unicode string, without normalization. All identified raw
observation IDs are tabled, even when different IDs have identical text.

``decoded_body_sha256`` binds the decoded JSON; release replay must additionally
compare it with the independently reconstructed W1 body. This checksum cannot
authenticate an envelope whose contents and checksum were both replaced.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from typing import Any, Sequence

from src import whole_owner_requests as original
from src.workflow_state import fingerprint

VERSION = 'compact-owner-body-1'
MIN_SHARED_STRING_CHARS = 32
_MARKERS = {'$o', '$s'}
_ID_FIELDS = {'observation_id', 'source_observation_id'}
_IDS_FIELDS = {'observation_ids', 'source_observation_ids', 'case_observation_ids',
               'supporting_ids', 'conflicting_ids'}
_ENVELOPE_KEYS = {'version', 'focus_owner_ids', 'table', 'body', 'decoded_body_sha256'}

REFERENCE_INSTRUCTION = '''
输入body使用无损引用封装。先按以下规则还原完整资料，再执行以上全部翻译要求。封装的body字段才是上述原始资料对象；外层focus_owner_ids是同一焦点编号的副本。table.observation_ids与table.strings是从0开始编号的字符串数组。资料中恰好只有一个键的对象{"$o":n}表示table.observation_ids[n]的完整原始观察ID；{"$s":n}表示table.strings[n]的完整原始字符串。每次出现均用对应完整字符串理解，不省略、不总结、不拆词，不把引用序号当作原始观察ID。表内字符串只按原样读取，不再次解析引用；普通字符串中看似引用的文字也只是原文。所有其他字段、对象键、列表次序、数字、布尔值和null均保留原义。decoded_body_sha256是完整性元数据，不是语义证据。
同一个引用可出现在不同位置，必须在每个位置完整读取；不同原始观察ID始终是不同观察，即使其文字完全相同。表只消除传输中的重复字符串，不合并观察、时间片段、所有编号、支持、冲突或不确定性。必须通读还原后的完整source_rows、original_context、source_readings和focused_context以及其中的全部地图、观察、替代读法与时间信息。只输出既定owners中文结果，不输出引用符号、引用序号或解码说明。'''

WRITER_INSTRUCTION = original.WRITER_INSTRUCTION + REFERENCE_INSTRUCTION
VERIFIER_INSTRUCTION = original.VERIFIER_INSTRUCTION
GLOBAL_INSTRUCTION = original.GLOBAL_INSTRUCTION
build_verifier_request = original.build_verifier_request
build_global_request = original.build_global_request


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError('Compact body must be finite acyclic JSON') from exc


def _scan(value: Any, counts: Counter[str], identifiers: set[str]) -> None:
    """Validate plain JSON and discover strings by exact value, without guesses."""
    if type(value) is str:
        counts[value] += 1
    elif type(value) is list:
        for item in value:
            _scan(item, counts, identifiers)
    elif type(value) is dict:
        _require(all(type(key) is str for key in value), 'Body keys must be strings')
        _require(not (_MARKERS & value.keys()), 'Reserved reference marker collision')
        for key, item in value.items():
            if key in _ID_FIELDS and type(item) is str:
                identifiers.add(item)
            elif key in _IDS_FIELDS and type(item) is list:
                identifiers.update(entry for entry in item if type(entry) is str)
            _scan(item, counts, identifiers)
    else:
        _require(value is None or type(value) in (bool, int, float), 'Body is not plain JSON')
        _require(type(value) is not float or math.isfinite(value), 'Nonfinite body number')


def encode_body(body: dict[str, Any]) -> dict[str, Any]:
    """Encode a complete JSON body deterministically, retaining every value."""
    _require(type(body) is dict, 'Writer body must be an object')
    _canonical(body)  # Reject cyclic containers and non-JSON values before walking.
    counts: Counter[str] = Counter()
    identifiers: set[str] = set()
    _scan(body, counts, identifiers)
    observation_ids = sorted(identifiers)
    strings = sorted(value for value, count in counts.items()
                     if count > 1 and len(value) >= MIN_SHARED_STRING_CHARS and value not in identifiers)
    refs = {value: {'$o': index} for index, value in enumerate(observation_ids)}
    refs.update({value: {'$s': index} for index, value in enumerate(strings)})

    def encode(value: Any) -> Any:
        if type(value) is str:
            return deepcopy(refs.get(value, value))
        if type(value) is list:
            return [encode(item) for item in value]
        if type(value) is dict:
            return {key: encode(value[key]) for key in sorted(value)}
        return value

    return {'version': VERSION, 'focus_owner_ids': json.loads(_canonical(body.get('focus_owner_ids'))),
            'table': {'observation_ids': observation_ids, 'strings': strings},
            'body': encode(body), 'decoded_body_sha256': fingerprint(body)}


def decode_body(envelope: dict[str, Any]) -> dict[str, Any]:
    """Reject ambiguous, stale or noncanonical references; return fresh plain JSON."""
    _require(type(envelope) is dict and set(envelope) == _ENVELOPE_KEYS, 'Invalid compact envelope fields')
    _canonical(envelope)
    _require(envelope['version'] == VERSION, 'Unknown compact envelope version')
    table = envelope['table']
    _require(type(table) is dict and set(table) == {'observation_ids', 'strings'}, 'Invalid reference table')
    for values in table.values():
        _require(type(values) is list and all(type(value) is str for value in values), 'Invalid table strings')
        _require(values == sorted(set(values)), 'Reference table is not sorted and unique')
    _require(not (set(table['observation_ids']) & set(table['strings'])), 'Ambiguous reference tables')
    tables = {'$o': table['observation_ids'], '$s': table['strings']}
    used: dict[str, set[int]] = {marker: set() for marker in _MARKERS}

    def decode(value: Any) -> Any:
        if type(value) is list:
            return [decode(item) for item in value]
        if type(value) is dict:
            markers = _MARKERS & value.keys()
            if markers:
                _require(len(value) == 1, 'Reference object contains extra fields')
                marker = next(iter(markers)); index = value[marker]
                _require(type(index) is int and 0 <= index < len(tables[marker]), 'Unknown reference index')
                used[marker].add(index)
                return tables[marker][index]
            _require(all(type(key) is str for key in value), 'Encoded body keys must be strings')
            return {key: decode(value[key]) for key in sorted(value)}
        _require(value is None or type(value) in (str, bool, int, float), 'Encoded body is not plain JSON')
        _require(type(value) is not float or math.isfinite(value), 'Nonfinite encoded number')
        return value

    body = decode(envelope['body'])
    _require(type(body) is dict, 'Decoded writer body must be an object')
    for marker, values in tables.items():
        _require(used[marker] == set(range(len(values))), 'Unused reference table entry')
    _require(envelope['decoded_body_sha256'] == fingerprint(body), 'Decoded body hash differs')
    _require(_canonical(envelope['focus_owner_ids']) == _canonical(body.get('focus_owner_ids')),
             'Focus owner metadata differs')
    _require(_canonical(encode_body(body)) == _canonical(envelope), 'Noncanonical compact representation')
    return body


def build_writer_request(pack: dict, bound_map: dict, owner_ids: Sequence[int]) -> dict:
    """Wrap the exact W1 two-owner request; only input representation changes."""
    request = original.build_writer_request(pack, bound_map, owner_ids)
    body = encode_body(request['body'])
    _require(_canonical(decode_body(body)) == _canonical(request['body']), 'Compact writer round trip differs')
    return {'instruction': request['instruction'] + REFERENCE_INSTRUCTION,
            'body': body, 'schema': request['schema']}
