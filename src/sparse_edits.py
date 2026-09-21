"""Pure, source-grounded sparse patch transactions; never a semantic approval.

Public inputs are lists of SrtBlock and observation dictionaries. ``validate_case``
returns a deep copy with all readings/uncertainty unchanged. ``prepare_patch``
returns valid/rejection plus exact bilingual before/after owner text, immutable
case/proposal/used-observation copies, and SHA256 bindings. ``apply_transactions``
validates every transaction against the same complete input pair before applying
any, and returns copied source/target rows and a transaction ledger. One owner may
occur at most once. A valid transaction establishes mechanical provenance only.

Target span lengths are Unicode code points, not UTF8 bytes. Only the established
_target_text whitespace policy may normalize a proposed Chinese replacement;
source replacements never normalize non-whitespace or whitespace characters.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import re
from typing import Any

from src.coherence import _edited_text
from src.revisit_workflow import _target_text, anchored
from src.translate import SrtBlock

VERSION = 'sparse-edits-1'
CASE_KEYS = {'case_id', 'owner_id', 'target_span', 'source_span', 'category',
             'severity', 'issue', 'unresolved', 'readings'}
READING_KEYS = {'observation_id', 'quote', 'interpretation_ja', 'speech_act', 'viable'}
PROPOSAL_KEYS = {'old_chinese', 'new_chinese', 'old_japanese', 'new_japanese',
                 'source_observation_id', 'source_support_quote'}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _string(value: Any, *, nonempty: bool = True) -> None:
    _require(isinstance(value, str), 'expected_string')
    _require(not nonempty or bool(value.strip()), 'empty_string')
    value.encode('utf-8', errors='strict')


def _rows(source_rows: list[SrtBlock], target_rows: list[SrtBlock]) -> dict[int, tuple[SrtBlock, SrtBlock]]:
    _require(isinstance(source_rows, list) and isinstance(target_rows, list), 'rows_must_be_lists')
    _require(bool(source_rows) and len(source_rows) == len(target_rows), 'row_count_mismatch')
    result = {}
    for source, target in zip(source_rows, target_rows):
        _require(isinstance(source, SrtBlock) and isinstance(target, SrtBlock), 'expected_srtblock')
        _require(type(source.index) is int and type(target.index) is int and source.index > 0, 'invalid_owner_id')
        _require(source.index == target.index and source.ts_line == target.ts_line, 'geometry_mismatch')
        _require(source.index not in result, 'duplicate_input_owner')
        _string(source.ts_line)
        _string(source.text)
        _string(target.text)
        result[source.index] = (source, target)
    return result


def _observations(observations: list[dict]) -> dict[str, dict]:
    _require(isinstance(observations, list), 'observations_must_be_list')
    result = {}
    for row in observations:
        _require(isinstance(row, dict), 'invalid_observation')
        _string(row.get('observation_id'))
        _require(type(row.get('owner_id')) is int and row['owner_id'] > 0, 'invalid_observation_owner')
        _string(row.get('text'), nonempty=False)
        _require(row['observation_id'] not in result, 'duplicate_observation_id')
        result[row['observation_id']] = row
    return result


def _witness(index: dict[str, dict], observation_id: str, owner_id: int, quote: str) -> dict:
    _string(observation_id)
    _string(quote)
    witness = index.get(observation_id)
    _require(witness is not None, 'unknown_observation')
    _require(witness['owner_id'] == owner_id, 'wrong_owner_observation')
    _require(witness.get('status', 'ok') == 'ok', 'unavailable_observation')
    _require(quote in witness['text'], 'quote_not_observed')
    return witness


def _unique_span(span: str, owner: str) -> bool:
    # str.count ignores overlapping occurrences, which are ambiguous edit anchors.
    return bool(span) and sum(1 for _ in re.finditer('(?=' + re.escape(span) + ')', owner)) == 1


def _span_limit(span: str, owner: str) -> None:
    _require(len(span) <= 160, 'target_span_over_160_codepoints')
    _require(len(owner) <= 100 or len(span) * 5 <= len(owner) * 3,
             'target_span_over_60_percent')


def validate_case(case: dict, source_rows: list[SrtBlock], target_rows: list[SrtBlock],
                  observations: list[dict]) -> dict:
    """Validate exact anchors; retain every reading and every uncertainty flag."""
    owners = _rows(source_rows, target_rows)
    index = _observations(observations)
    _require(isinstance(case, dict) and set(case) == CASE_KEYS, 'case_fields_mismatch')
    for key in ('case_id', 'target_span', 'source_span', 'category', 'issue'):
        _string(case[key])
    _require(type(case['owner_id']) is int and case['owner_id'] in owners, 'unknown_owner')
    _require(type(case['severity']) is int and 0 <= case['severity'] <= 3, 'invalid_severity')
    _require(type(case['unresolved']) is bool, 'invalid_unresolved')
    source, target = owners[case['owner_id']]
    _require(_unique_span(case['target_span'], target.text), 'target_span_not_unique')
    _require(case['source_span'] in source.text, 'source_span_not_in_owner')
    _span_limit(case['target_span'], target.text)
    _require(isinstance(case['readings'], list), 'readings_must_be_list')
    for reading in case['readings']:
        _require(isinstance(reading, dict) and set(reading) == READING_KEYS, 'reading_fields_mismatch')
        _require(type(reading['viable']) is bool, 'invalid_viable')
        _string(reading['interpretation_ja'])
        _string(reading['speech_act'])
        _witness(index, reading['observation_id'], case['owner_id'], reading['quote'])
    return deepcopy(case)


def prepare_patch(case: dict, proposal: dict, source_rows: list[SrtBlock], target_rows: list[SrtBlock],
                  observations: list[dict], allow_source: bool = False) -> dict:
    """Return a mechanically valid bound transaction, or valid=False/rejection.

    No source amendment uses four empty source proposal fields. A nonempty source
    amendment requires allow_source=True, an exact unique old span, a same-owner
    observed support quote containing the new span, and the existing textual
    neighbor-anchor check. This never certifies acoustic ownership or truth.
    """
    try:
        normalized_case = validate_case(case, source_rows, target_rows, observations)
        _require(type(allow_source) is bool, 'invalid_allow_source')
        _require(isinstance(proposal, dict) and set(proposal) == PROPOSAL_KEYS, 'proposal_fields_mismatch')
        for value in proposal.values():
            _string(value, nonempty=False)
        source, target = _rows(source_rows, target_rows)[case['owner_id']]
        old_chinese = proposal['old_chinese']
        _require(old_chinese == case['target_span'], 'old_chinese_differs_from_case')
        new_chinese = _target_text(proposal['new_chinese'], source.text)
        _require(re.sub(r'\s', '', new_chinese) == re.sub(r'\s', '', proposal['new_chinese']),
                 'normalization_changed_nonwhitespace')
        _require(len(new_chinese) <= max(64, 2 * len(old_chinese)), 'replacement_too_long')
        old_japanese, new_japanese = proposal['old_japanese'], proposal['new_japanese']
        index = _observations(observations)
        if old_japanese or new_japanese:
            _require(allow_source, 'source_edit_disabled')
            _require(bool(old_japanese.strip()) and bool(new_japanese.strip()), 'incomplete_source_edit')
            _require(_unique_span(old_japanese, source.text), 'old_japanese_not_unique')
            _edited_text(new_japanese, source.text)  # Validate only; never use normalized return.
            witness = _witness(index, proposal['source_observation_id'], case['owner_id'],
                               proposal['source_support_quote'])
            _require(new_japanese in proposal['source_support_quote'], 'new_japanese_not_supported')
            _require(anchored(source.text, old_japanese, new_japanese, witness['text']),
                     'source_edit_not_owner_anchored')
            after_source = source.text.replace(old_japanese, new_japanese, 1)
        else:
            _require(not proposal['source_observation_id'] and not proposal['source_support_quote'],
                     'source_support_without_edit')
            after_source = source.text
        after_target = target.text.replace(old_chinese, new_chinese, 1)
        _edited_text(after_source, source.text)
        _edited_text(after_target, source.text)
        used_ids = {r['observation_id'] for r in case['readings']}
        if proposal['source_observation_id']:
            used_ids.add(proposal['source_observation_id'])
        used_observations = [deepcopy(index[key]) for key in sorted(used_ids)]
        tx = {'version': VERSION, 'valid': True, 'rejection': None,
              'case_id': case['case_id'], 'owner_id': case['owner_id'],
              'old_chinese': old_chinese, 'new_chinese': new_chinese,
              'old_japanese': old_japanese, 'new_japanese': new_japanese,
              'before_source_text': source.text, 'after_source_text': after_source,
              'before_target_text': target.text, 'after_target_text': after_target,
              'source_before_sha256': _text_hash(source.text),
              'target_before_sha256': _text_hash(target.text),
              'source_context_sha256': _hash([asdict(r) for r in source_rows]),
              'target_context_sha256': _hash([asdict(r) for r in target_rows]),
              'geometry_sha256': _hash([(r.index, r.ts_line) for r in source_rows]),
              'case_sha256': _hash(normalized_case), 'proposal_sha256': _hash(proposal),
              'observations_sha256': _hash(used_observations),
              'case': normalized_case, 'proposal': deepcopy(proposal), 'observations': used_observations,
              'allow_source': allow_source,
              'no_op': after_source == source.text and after_target == target.text,
              'source_interpretations_verified': False,
              'observation_owner_scope': 'crop_owner_hint_not_acoustic_alignment',
              'target_normalization': 'existing_target_text_guard',
              'target_normalized': new_chinese != proposal['new_chinese']}
        tx['transaction_sha256'] = _hash(tx)
        return tx
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        return {'version': VERSION, 'valid': False, 'rejection': str(exc),
                'case_id': case.get('case_id') if isinstance(case, dict) else None,
                'owner_id': case.get('owner_id') if isinstance(case, dict) else None,
                'source_interpretations_verified': False}


def apply_transactions(source_rows: list[SrtBlock], target_rows: list[SrtBlock],
                       transactions: list[dict]) -> tuple[list[SrtBlock], list[SrtBlock], list[dict]]:
    """Validate all bindings atomically, then apply minimal independent patches.

    Transactions must share the exact complete input context. Duplicate owners
    (including no-op duplicates), invalid/tampered transactions, or any stale row
    reject the entire batch. Caller-owned input rows are never modified.
    """
    _rows(source_rows, target_rows)
    _require(isinstance(transactions, list), 'transactions_must_be_list')
    by_owner = {}
    for tx in transactions:
        _require(isinstance(tx, dict) and tx.get('valid') is True, 'invalid_transaction')
        owner = tx.get('owner_id')
        _require(type(owner) is int and owner not in by_owner, 'duplicate_or_invalid_transaction_owner')
        unsigned = {k: v for k, v in tx.items() if k != 'transaction_sha256'}
        _require(tx.get('transaction_sha256') == _hash(unsigned), 'transaction_hash_mismatch')
        rebuilt = prepare_patch(tx.get('case'), tx.get('proposal'), source_rows, target_rows,
                                tx.get('observations'), allow_source=tx.get('allow_source'))
        _require(rebuilt.get('valid') is True, 'transaction_replay_invalid')
        _require(rebuilt == tx, 'stale_or_modified_transaction')
        by_owner[owner] = tx
    source = [replace(row, text=by_owner[row.index]['after_source_text'])
              if row.index in by_owner else replace(row) for row in source_rows]
    target = [replace(row, text=by_owner[row.index]['after_target_text'])
              if row.index in by_owner else replace(row) for row in target_rows]
    _rows(source, target)
    ledger = [{'case_id': tx['case_id'], 'owner_id': tx['owner_id'],
               'transaction_sha256': tx['transaction_sha256'], 'applied': True, 'no_op': tx['no_op'],
               'source_changed': tx['before_source_text'] != tx['after_source_text'],
               'target_changed': tx['before_target_text'] != tx['after_target_text'],
               'source_interpretations_verified': False}
              for tx in transactions]
    return source, target, ledger
