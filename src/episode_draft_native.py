"""Bounded L1 native writer and network-free replay of its exact request ledger.

The outer controller owns backend identity, producer/input pins and the local
stage deadline. Only this module's ``ask`` dispatches inference. A complete
cached response is replayed; an incomplete saved request is never resumed.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Callable

import jsonschema

from src import episode_draft_requests as drafts
from src import revisit_workflow as rw
from src import sparse_revisit as sr
from src.config import TranslateConfig
from src.contextual_review_contract import require, strict_json
from src.translate import _build_payload, call_llm
from src.workflow_state import fingerprint, write_json

VERSION = 'episode-draft-native-1'
MAX_OUTPUT_TOKENS = 16384
CONTEXT_SIZE = 196608
MAX_ATTEMPTS = 2
_CAPACITY_KEYS = {'prompt_tokens', 'reserved_output', 'context', 'rendered_prompt_sha256', 'fits'}
_ROOT_KEYS = {'request', 'request_sha256', 'status', 'attempts', 'parsed', 'receipt_sha256'}
_ATTEMPT_KEYS = {'number', 'started_utc', 'finished_utc', 'seconds', 'native_request_sha256',
                 'payload_sha256', 'capacity_started_utc', 'capacity_finished_utc', 'capacity',
                 'generation_started_utc', 'generation_finished_utc', 'metrics_start', 'metrics_end',
                 'native_receipts', 'raw', 'answer_sha256', 'error_type', 'failure_kind'}
_TELEMETRY_KEYS = {'stage', 'attempt', 'seconds', 'thinking', 'answer_chars', 'reasoning_chars',
                   'finish_reason', 'usage', 'timings', 'request_settings'}
Track = Callable[..., Any]


def _same(left: Any, right: Any) -> bool:
    return fingerprint(left) == fingerprint(right)


def _seconds(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _timestamp(value: Any) -> datetime:
    require(isinstance(value, str), 'Native timestamp missing')
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError('Native timestamp invalid') from exc
    require(parsed.utcoffset() is not None, 'Native timestamp lacks timezone')
    return parsed


def prepare_request(request: dict) -> dict:
    """Pure fixed Qwen request identity, independent of sparse ask defaults."""
    require(isinstance(request, dict) and set(request) == {'instruction', 'body', 'schema'},
            'Invalid episode writer request')
    require(request['instruction'] == drafts.WRITER_INSTRUCTION and _same(request['schema'], drafts.writer_schema()),
            'Episode instruction or all-owner output schema changed')
    body = request['body']
    require(isinstance(body, dict) and set(body) == {'source_evidence', 'source_map', 'focus_owner_ids'}
            and _same(body['focus_owner_ids'], list(drafts.OWNER_IDS)), 'Episode writer input scope changed')
    # Full input authenticity is checked by the pure builder and release reconstruction.
    require(isinstance(body['source_evidence'], dict) and isinstance(body['source_map'], dict),
            'Episode source evidence or map missing')
    json.dumps(request, allow_nan=False, ensure_ascii=False)
    extra = {'model': rw.QWEN, 'temperature': .3, 'seed': 20260915, 'top_p': .95,
             'max_tokens': MAX_OUTPUT_TOKENS, 'chat_template_kwargs': {'enable_thinking': False},
             'reasoning_budget_tokens': 0, 'reasoning_effort': 'none',
             'response_format': {'type': 'json_object', 'schema': deepcopy(request['schema'])}}
    return {'version': VERSION, 'family': 'qwen', 'instruction': request['instruction'],
            'body': deepcopy(body), 'schema': deepcopy(request['schema']), 'extra': extra, 'thinking': False}


def config_for(endpoint: str, directory: Path, key: str, request: dict, remaining: float = 600) -> TranslateConfig:
    """Pure common template/config for capacity and the single HTTP attempt."""
    require(type(remaining) in (int, float) and math.isfinite(remaining), 'Invalid episode deadline')
    if remaining < 1:
        raise rw.LocalBudgetExceeded('No remaining episode draft budget')
    return TranslateConfig(endpoint=endpoint, max_tokens=MAX_OUTPUT_TOKENS,
        timeout=max(1, int(min(600, remaining))), retries=0, separate_instruction=True,
        response_guard_floor=65536, stage=key, telemetry_path=Path(directory)/'metrics.jsonl',
        extra_payload=prepare_request(request)['extra'])


def _payload(request: dict, cfg: TranslateConfig) -> dict:
    return _build_payload(json.dumps(request['body'], ensure_ascii=False), request['instruction'], cfg,
                          {}, MAX_OUTPUT_TOKENS, stream=True, with_thinking=False)


def _capacity(value: Any) -> dict:
    require(isinstance(value, dict) and set(value) == _CAPACITY_KEYS, 'Native capacity fields changed')
    require(all(type(value[key]) is int for key in ('prompt_tokens', 'reserved_output', 'context'))
            and value['prompt_tokens'] > 0 and value['reserved_output'] == MAX_OUTPUT_TOKENS
            and value['context'] == CONTEXT_SIZE and value['fits'] is True
            and value['prompt_tokens'] + MAX_OUTPUT_TOKENS + 64 <= CONTEXT_SIZE
            and isinstance(value['rendered_prompt_sha256'], str)
            and re.fullmatch(r'[0-9a-f]{64}', value['rendered_prompt_sha256']) is not None,
            'Native episode capacity or template identity failed')
    return value


def _settings(native: dict, stream: bool) -> dict:
    extra = native['extra']
    return {**{key: extra[key] for key in ('model', 'max_tokens', 'reasoning_budget_tokens',
            'reasoning_effort', 'temperature', 'top_p', 'seed')}, 'stream': stream, 'enable_thinking': False}


def _receipts(attempt: dict, native: dict, key: str) -> None:
    receipts = attempt['native_receipts']
    require(isinstance(receipts, list) and len(receipts) == 1, 'Expected one native receipt per HTTP helper call')
    receipt = receipts[0]
    require(isinstance(receipt, dict) and set(receipt) == _TELEMETRY_KEYS
            and receipt['stage'] == key and type(receipt['attempt']) is int and receipt['attempt'] == 0
            and _seconds(receipt['seconds']) and receipt['thinking'] is False
            and type(receipt['answer_chars']) is int and receipt['answer_chars'] >= 0
            and type(receipt['reasoning_chars']) is int and receipt['reasoning_chars'] == 0,
            'Native receipt identity, counts or thinking changed')
    settings = receipt['request_settings']
    require(isinstance(settings, dict) and type(settings.get('stream')) is bool
            and _same(settings, _settings(native, settings['stream'])), 'Native sampling or reserve settings changed')
    require(isinstance(receipt['finish_reason'], str), 'Native finish reason missing')
    if isinstance(receipt['usage'], dict) and 'completion_tokens' in receipt['usage']:
        count = receipt['usage']['completion_tokens']
        require(type(count) is int and 0 <= count <= MAX_OUTPUT_TOKENS, 'Native completion token budget changed')
    raw = attempt['raw']
    require(raw is None or isinstance(raw, str), 'Native raw answer is not text')
    require(attempt['answer_sha256'] == (None if raw is None else fingerprint(raw)), 'Native answer hash changed')
    if raw is not None:
        require(receipt['answer_chars'] == len(raw), 'Native answer length differs from telemetry')


def _parse_complete(attempt: dict, schema: dict) -> dict:
    require(attempt['native_receipts'][-1]['finish_reason'] == 'stop', 'Episode draft did not stop normally')
    require(isinstance(attempt['raw'], str), 'Episode draft raw response missing')
    parsed = strict_json(attempt['raw'])
    jsonschema.validate(parsed, schema)
    return parsed


def _is_complete(attempt: dict, schema: dict) -> bool:
    try:
        _parse_complete(attempt, schema)
    except (ValueError, TypeError, jsonschema.ValidationError):
        return False
    return True


def _technical_failure(attempt: dict, native: dict, key: str) -> None:
    """Permit only an evidenced failed response, never a successful-answer reroll."""
    require(attempt['failure_kind'] in ('transport', 'response'), 'Failure is not eligible for a technical retry')
    _capacity(attempt['capacity']); _receipts(attempt, native, key)
    require(not _is_complete(attempt, native['schema']), 'Successful episode response was rerolled')
    receipt = attempt['native_receipts'][0]
    if attempt['failure_kind'] == 'transport':
        require(attempt['raw'] is None and receipt['finish_reason'] == 'request_failed'
                and receipt['answer_chars'] == 0, 'Unverifiable transport failure cannot be retried')
    else:
        require(isinstance(attempt['raw'], str) and receipt['finish_reason'] in ('stop', 'length'),
                'Response failure lacks a verifiable native result')


def _metrics(raw: bytes) -> list[dict]:
    require(isinstance(raw, bytes), 'Native telemetry must be read as original bytes')
    text = raw.decode('utf-8', errors='strict')
    if text:
        require(text.endswith('\n'), 'Native telemetry has an incomplete line')
    values = [strict_json(line) for line in text.splitlines()]
    require(all(isinstance(value, dict) for value in values), 'Native telemetry row is not an object')
    return values


def _local_track(path: Path, raw: bool = False) -> Any:
    data = Path(path).read_bytes()
    return data if raw else strict_json(data)


def replay_native(path: Path, request: dict, track: Track) -> dict:
    """Reconstruct all successful-writer provenance from supplied artifacts only."""
    path = Path(path); saved = track(path); native = prepare_request(request); pin = fingerprint(native)
    require(isinstance(saved, dict) and set(saved) == _ROOT_KEYS
            and saved['receipt_sha256'] == fingerprint({key: value for key, value in saved.items() if key != 'receipt_sha256'})
            and _same(saved['request'], native) and saved['request_sha256'] == pin,
            'Episode native request or sealed ledger changed')
    attempts = saved['attempts']
    require(saved['status'] == 'complete' and isinstance(attempts, list) and 1 <= len(attempts) <= MAX_ATTEMPTS,
            'Episode request lacks bounded complete execution')
    metrics = _metrics(track(path.parent/'metrics.jsonl', raw=True))
    cfg = config_for('http://127.0.0.1/unused', path.parent, path.stem, request)
    payload_hash = fingerprint(_payload(request, cfg))
    previous_end = None; previous_metric_end = None; capacity = None; used_lines = []
    parsed = None
    for number, attempt in enumerate(attempts, 1):
        require(isinstance(attempt, dict) and set(attempt) == _ATTEMPT_KEYS
                and type(attempt['number']) is int and attempt['number'] == number
                and attempt['native_request_sha256'] == pin and attempt['payload_sha256'] == payload_hash
                and _seconds(attempt['seconds']), 'Native attempt identity changed')
        timestamps = [_timestamp(attempt[key]) for key in ('started_utc', 'capacity_started_utc',
            'capacity_finished_utc', 'generation_started_utc', 'generation_finished_utc', 'finished_utc')]
        require(timestamps == sorted(timestamps) and (previous_end is None or timestamps[0] >= previous_end),
                'Native capacity did not complete before ordered generation')
        previous_end = timestamps[-1]
        checked = _capacity(attempt['capacity'])
        require(capacity is None or _same(capacity, checked), 'Native capacity changed between technical attempts')
        capacity = checked
        start, end = attempt['metrics_start'], attempt['metrics_end']
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(metrics)
                and (previous_metric_end is None or start == previous_metric_end)
                and _same(metrics[start:end], attempt['native_receipts']), 'Native receipt range changed')
        previous_metric_end = end; used_lines.extend(range(start, end))
        _receipts(attempt, native, path.stem)
        if number < len(attempts):
            require(isinstance(attempt['error_type'], str) and bool(attempt['error_type']), 'Failed attempt error missing')
            _technical_failure(attempt, native, path.stem)
        else:
            require(attempt['error_type'] is None and attempt['failure_kind'] is None, 'Completed attempt retains an error')
            parsed = _parse_complete(attempt, native['schema'])
    require(used_lines == [index for index, metric in enumerate(metrics) if metric.get('stage') == path.stem],
            'Unregistered native episode invocation or hidden retry')
    require(_same(parsed, saved['parsed']), 'Parsed episode differs from raw complete answer')
    return deepcopy(parsed)


def _remaining() -> float:
    remaining = 600 if sr._DEADLINE is None else sr._DEADLINE - time.monotonic()
    if remaining < 1:
        raise rw.LocalBudgetExceeded('No remaining episode draft budget')
    return remaining


def ask(directory: Path, key: str, endpoint: str, request: dict) -> dict:
    """One whole-episode generation, with at most one identical technical retry."""
    directory = Path(directory)
    require(isinstance(key, str) and re.fullmatch(r'[A-Za-z0-9_-]+', key) is not None, 'Invalid native request key')
    native = prepare_request(request); pin = fingerprint(native); path = directory/(key+'.json')
    if path.exists():
        saved = _local_track(path)
        require(saved.get('status') == 'complete', 'Incomplete saved episode request cannot be resumed')
        return replay_native(path, request, _local_track)
    directory.mkdir(parents=True, exist_ok=True)
    metric_path = directory/'metrics.jsonl'
    prior_metrics = _metrics(metric_path.read_bytes()) if metric_path.exists() else []
    require(not any(row.get('stage') == key for row in prior_metrics), 'Orphan native writer receipts exist')
    saved = {'request': native, 'request_sha256': pin, 'status': 'prepared', 'attempts': [], 'parsed': None}

    def save() -> None:
        saved['receipt_sha256'] = fingerprint({name: value for name, value in saved.items() if name != 'receipt_sha256'})
        write_json(path, saved)

    save(); previous_capacity = None
    for number in range(1, MAX_ATTEMPTS + 1):
        started = time.monotonic()
        attempt = {field: None for field in _ATTEMPT_KEYS}
        attempt.update(number=number, started_utc=sr.now(), seconds=0., native_request_sha256=pin, native_receipts=[])
        saved['attempts'].append(attempt); saved['status'] = 'running'; save()
        phase = 'budget'; failure = None; parsed = None
        try:
            cfg = config_for(endpoint, directory, key, request, _remaining())
            attempt['payload_sha256'] = fingerprint(_payload(request, cfg))
            phase = 'capacity'; attempt['capacity_started_utc'] = sr.now(); save()
            try:
                attempt['capacity'] = rw.native_capacity(json.dumps(request['body'], ensure_ascii=False),
                                                         request['instruction'], cfg, 'qwen')
                _capacity(attempt['capacity'])
                require(previous_capacity is None or _same(previous_capacity, attempt['capacity']),
                        'Native capacity changed before retry generation')
                previous_capacity = deepcopy(attempt['capacity'])
            finally:
                attempt['capacity_finished_utc'] = sr.now(); save()
            phase = 'budget'; _remaining()
            phase = 'provenance'
            before = _metrics(metric_path.read_bytes()) if metric_path.exists() else []
            attempt['metrics_start'] = len(before)
            attempt['generation_started_utc'] = sr.now(); save()
            phase = 'transport'
            try:
                raw = call_llm(json.dumps(request['body'], ensure_ascii=False), request['instruction'], cfg,
                               with_thinking=False)
                attempt['raw'] = raw; attempt['answer_sha256'] = fingerprint(raw)
            finally:
                attempt['generation_finished_utc'] = sr.now()
                # Preserve every response/failure receipt before stop/schema validation.
                after = _metrics(metric_path.read_bytes()) if metric_path.exists() else []
                attempt['metrics_end'] = len(after); attempt['native_receipts'] = after[len(before):]
                save()
                require(_same(after[:len(before)], before), 'Native telemetry prefix changed during generation')
            phase = 'provenance'; _receipts(attempt, native, key)
            phase = 'response'; parsed = _parse_complete(attempt, request['schema'])
        except BaseException as exc:
            failure = exc; attempt['error_type'] = type(exc).__name__
            attempt['failure_kind'] = ('budget' if isinstance(exc, rw.LocalBudgetExceeded) else
                                       'interrupted' if not isinstance(exc, Exception) else phase)
        finally:
            attempt['finished_utc'] = sr.now(); attempt['seconds'] = time.monotonic() - started
            saved['status'] = 'complete' if failure is None else 'failed'
            if failure is None:
                saved['parsed'] = parsed
            save()
        if failure is None:
            return replay_native(path, request, _local_track)
        if attempt['failure_kind'] not in ('transport', 'response'):
            raise failure
        try:
            _technical_failure(attempt, native, key)
        except ValueError:
            attempt['failure_kind'] = 'provenance'; save()
            raise
        if number == MAX_ATTEMPTS:
            raise RuntimeError('One episode writer technical retry exhausted') from failure
    raise AssertionError('Unreachable episode draft retry state')
