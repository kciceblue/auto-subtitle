"""Pure complete-owner target transactions, independent of sparse diagnoses.

The source evidence pack and every original source row stay immutable. A valid
transaction proves mechanical bindings only; accept_owner_decision additionally
checks the strict, fallible native comparison contract, never acoustic truth.
No models, files, translation requests, or source edits are performed here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import re
from typing import Any

from src import evidence_context as ec
from src import sparse_revisit as sr
from src.revisit_workflow import _target_text
from src.translate import SrtBlock

VERSION = 'whole-owner-translation-1'
MAX_OUTPUT_CHARS = 2500
_VERDICTS = {'supported', 'contradicted', 'uncertain'}
_DECISION_KEYS = {'problem_exists', 'patch_resolves', 'preserves_uncertainty',
                  'new_material_error', 'reading_checks', 'source_observation_id',
                  'source_quote', 'before_quote', 'after_quote', 'reason'}
_CHECK_KEYS = {'observation_id', 'quote', 'before_support', 'after_support'}
# Process-specific forms, not dialogue words such as "sorry", "cannot", or "无".
_ASSISTANT_META = re.compile(
    r'\A\s*(?:'
    r'as\s+an?\s+(?:ai(?:\s+language)?\s+(?:model|assistant)|language\s+model)\b|'
    r'(?:sorry[,!\s]*|i[\u2019\']m\s+sorry[,!\s]*)?'
    r'i\s+(?:cannot|can[\u2019\']t|am\s+unable\s+to)\s+'
    r'(?:assist\s+with\s+(?:this|that|your)\s+(?:request|translation)|'
    r'provide\s+(?:a\s+|the\s+)?translation\b|'
    r'comply\s+with\s+(?:this|that|your)\s+request)|'
    r'(?:很?抱歉[，,。\s]*)?(?:作为(?:一个)?(?:AI|人工智能|语言模型)|'
    r'我(?:无法|不能)(?:协助|帮助)(?:你|您)?(?:完成|处理)?(?:此|该|这个|你的|您的)请求|'
    r'我(?:无法|不能)(?:提供|完成)(?:此|该|这段|这个)?(?:内容的)?翻译)|'
    r'(?:中文译文|翻译结果|译文如下|翻译如下)\s*[:：]'
    r')', re.IGNORECASE)


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8', errors='strict')).hexdigest()


def _row_values(rows: list[SrtBlock]) -> list[dict]:
    ec._require(isinstance(rows, list) and bool(rows), 'Owner rows missing')
    result = []
    for row in rows:
        ec._require(isinstance(row, SrtBlock), 'Expected SrtBlock owner')
        value = asdict(row)
        ec._keys(value, {'index', 'ts_line', 'text'})
        ec._require(type(value['index']) is int and value['index'] == len(result) + 1,
                    'Owner sequence changed')
        ec._text(value['ts_line']); ec._text(value['text'])
        result.append(value)
    return result


def _inputs(source: list[SrtBlock], target: list[SrtBlock], pack: dict) -> tuple[list, list]:
    ec._check_pack(pack)
    source_values, target_values = _row_values(source), _row_values(target)
    ec._require(ec._hash(source_values) == ec._hash(pack['source_rows']),
                'Source rows differ from bound pack')
    ec._require([(row['index'], row['ts_line']) for row in source_values]
                == [(row['index'], row['ts_line']) for row in target_values],
                'Source and target geometry differ')
    return source_values, target_values


def _replacement(value: Any) -> str:
    ec._text(value, maximum=MAX_OUTPUT_CHARS)
    # _target_text supplies established format/control guards and whitespace-only
    # normalization. Using the proposed text for its length argument makes its
    # legacy source-relative runaway bound nonbinding; W1 uses the absolute cap.
    normalized = _target_text(value, value)
    ec._require(not _ASSISTANT_META.search(normalized), 'Assistant refusal or process output')
    ec._require(re.sub(r'\s', '', value) == re.sub(r'\s', '', normalized),
                'Normalization changed lexical characters')
    return normalized


def _prepare(owner_id: int, replacement: str, source_values: list[dict],
             target_values: list[dict], pack: dict) -> dict:
    ec._require(type(owner_id) is int and 1 <= owner_id <= len(source_values),
                'Unknown owner')
    source, target = source_values[owner_id - 1], target_values[owner_id - 1]
    normalized = _replacement(replacement)
    # An exactly unchanged answer is a no-op, including original whitespace.
    if replacement == target['text']:
        normalized = target['text']
    proposal = {'owner_id': owner_id, 'replacement': replacement}
    observations = [deepcopy(item) for item in pack['observations'] if item['owner_id'] == owner_id]
    frame = deepcopy(pack['frames'][str(owner_id)])
    tx = {
        'version': VERSION, 'transaction_id': f'owner-{owner_id:03d}',
        'owner_id': owner_id, 'ts_line': source['ts_line'],
        'valid': True, 'no_op': normalized == target['text'], 'rejection': None,
        'old_chinese': target['text'], 'new_chinese': normalized,
        'old_japanese': '', 'new_japanese': '',
        'before_source_text': source['text'], 'after_source_text': source['text'],
        'before_target_text': target['text'], 'after_target_text': normalized,
        'source_before_sha256': _text_hash(source['text']),
        'target_before_sha256': _text_hash(target['text']),
        'source_context_sha256': ec._hash(source_values),
        'target_context_sha256': ec._hash(target_values),
        'geometry_sha256': ec._hash([(row['index'], row['ts_line']) for row in source_values]),
        'pack_sha256': pack['pack_sha256'],
        'base_target_rows': deepcopy(target_values),
        'proposal': proposal, 'proposal_sha256': ec._hash(proposal),
        'observations': observations, 'observations_sha256': ec._hash(observations),
        'frame': frame, 'frame_sha256': ec._hash(frame),
        'allow_source': False, 'source_interpretations_verified': False,
        'observation_owner_scope': 'crop_owner_hint_not_acoustic_alignment',
        'target_normalization': 'existing_whitespace_guards_absolute_2500_chars',
        'target_normalized': normalized != replacement,
    }
    return {**tx, 'transaction_sha256': ec._hash(tx)}


def prepare_owner(owner_id: int, replacement: str, source: list[SrtBlock],
                  target: list[SrtBlock], pack: dict) -> dict:
    """Prepare one whole-owner replacement or return an explicit invalid record.

    The complete raw proposal survives invalid output and no-ops. Replacement
    bounds never depend on the old Chinese length or the sparse 160/60% policy.
    Owner observations include refuted readings, fragments, and full uncertainty;
    the complete pack hash binds all other source/context/evidence as well.
    """
    proposal = {'owner_id': deepcopy(owner_id), 'replacement': deepcopy(replacement)}
    try:
        source_values, target_values = _inputs(source, target, pack)
        return _prepare(owner_id, replacement, source_values, target_values, pack)
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        return {'version': VERSION,
                'transaction_id': f'owner-{owner_id:03d}' if type(owner_id) is int and owner_id > 0 else None,
                'owner_id': deepcopy(owner_id), 'valid': False, 'no_op': False,
                'rejection': str(exc), 'proposal': proposal,
                'allow_source': False, 'source_interpretations_verified': False}


def validate_owner_transaction(transaction: dict, pack: dict, *,
                               source: list[SrtBlock] | None = None,
                               target: list[SrtBlock] | None = None) -> dict:
    """Rebuild every field against its pack/proposal and optional external base.

    base_target_rows binds the complete comparison target even for callers that
    only have the transaction and source pack. Applying to actual owner rows
    additionally supplies both source and target to reject stale external bases.
    A transaction hash is an integrity binding, not a native producer signature;
    campaign replay must separately match proposal text to its native receipt.
    """
    try:
        ec._check_pack(pack)
        ec._require(isinstance(transaction, dict) and transaction.get('valid') is True,
                    'Invalid whole-owner transaction')
        ec._require(transaction.get('pack_sha256') == pack['pack_sha256'], 'Stale evidence pack')
        ec._keys(transaction.get('proposal'), {'owner_id', 'replacement'})
        proposal = transaction['proposal']
        ec._require(type(transaction.get('owner_id')) is int
                    and proposal['owner_id'] == transaction['owner_id']
                    and type(proposal['owner_id']) is int, 'Proposal owner changed')
        base_target = transaction.get('base_target_rows')
        ec._require(isinstance(base_target, list), 'Target baseline missing')
        for row in base_target:
            ec._keys(row, {'index', 'ts_line', 'text'})
        frozen_source = [SrtBlock(**row) for row in pack['source_rows']]
        frozen_target = [SrtBlock(**row) for row in base_target]
        source_values, target_values = _inputs(frozen_source, frozen_target, pack)
        rebuilt = _prepare(proposal['owner_id'], proposal['replacement'],
                           source_values, target_values, pack)
        ec._require(ec._hash(transaction) == ec._hash(rebuilt),
                    'Whole-owner transaction or proposal binding changed')
        ec._require((source is None) == (target is None), 'Both external owner bases are required')
        if source is not None:
            actual_source, actual_target = _inputs(source, target, pack)
            ec._require(ec._hash(actual_source) == rebuilt['source_context_sha256']
                        and ec._hash(actual_target) == rebuilt['target_context_sha256'],
                        'Stale external owner base')
        return rebuilt
    except (TypeError, KeyError, UnicodeError) as exc:
        raise ValueError('Malformed whole-owner transaction') from exc


def apply_owner_transactions(source: list[SrtBlock], target: list[SrtBlock],
                             transactions: list[dict], pack: dict) -> tuple[list, list, list]:
    """Validate the complete batch before replacing any target owner.

    Rollback consists of applying a retained subset to the exact original base,
    so removed owners recover every original character and whitespace. Applying
    old transactions to an already changed target is rejected as a stale base.
    """
    _inputs(source, target, pack)
    ec._require(isinstance(transactions, list), 'Transactions must be a list')
    validated = {}; ledger = []
    for tx in transactions:
        rebuilt = validate_owner_transaction(tx, pack, source=source, target=target)
        owner = rebuilt['owner_id']
        ec._require(owner not in validated, 'Duplicate transaction owner')
        validated[owner] = rebuilt
    new_source = [replace(row) for row in source]
    new_target = [replace(row, text=validated[row.index]['after_target_text'])
                  if row.index in validated else replace(row) for row in target]
    for owner in sorted(validated):
        tx = validated[owner]
        ledger.append({'transaction_id': tx['transaction_id'], 'owner_id': owner,
                       'transaction_sha256': tx['transaction_sha256'],
                       'pack_sha256': tx['pack_sha256'], 'geometry_sha256': tx['geometry_sha256'],
                       'source_context_sha256': tx['source_context_sha256'],
                       'target_context_sha256': tx['target_context_sha256'],
                       'before_target_sha256': tx['target_before_sha256'],
                       'after_target_sha256': _text_hash(tx['after_target_text']),
                       'applied': True, 'no_op': tx['no_op'], 'source_changed': False,
                       'target_changed': not tx['no_op'], 'source_interpretations_verified': False})
    return new_source, new_target, ledger


def accept_owner_decision(verdict: dict, transaction: dict, pack: dict) -> bool:
    """Require strict material-improvement proof for a bound non-noop owner.

    Every viable observation must occur exactly once with a nonempty literal
    quote; refuted observations remain in the transaction but cannot serve as
    the primary supporting witness. No missing, foreign, invented, duplicated,
    or weakened reading is silently accepted. This is still a fallible model
    comparison, not a claim of source accuracy or speaker/word ownership.
    """
    try:
        tx = validate_owner_transaction(transaction, pack)
        ec._require(not tx['no_op'], 'No-op has no material replacement')
        ec._keys(verdict, _DECISION_KEYS)
        for name in ('problem_exists', 'patch_resolves', 'preserves_uncertainty', 'new_material_error'):
            ec._require(type(verdict[name]) is str and verdict[name] in _VERDICTS,
                        'Invalid semantic decision')
        for name in ('source_observation_id', 'source_quote', 'before_quote', 'after_quote', 'reason'):
            ec._text(verdict[name])
        observed = {item['observation_id']: item for item in tx['observations']}
        viable = {reading['observation_id'] for reading in tx['frame']['readings'] if reading['viable']}
        ec._require(verdict['source_observation_id'] in viable, 'Primary witness is not a viable reading')
        ec._require(isinstance(verdict['reading_checks'], list), 'Reading checks missing')
        for check in verdict['reading_checks']:
            ec._keys(check, _CHECK_KEYS)
            ec._text(check['observation_id']); ec._text(check['quote'])
            ec._require(check['observation_id'] in observed, 'Unknown reading witness')
            for name in ('before_support', 'after_support'):
                ec._require(type(check[name]) is str and check[name] in _VERDICTS,
                            'Invalid reading verdict')
        mechanical_owner = {'owner_id': tx['owner_id'], 'readings': tx['frame']['readings']}
        return sr.accept_decision(verdict, mechanical_owner, tx, pack['observations'])
    except (ValueError, TypeError, KeyError, UnicodeError):
        return False
