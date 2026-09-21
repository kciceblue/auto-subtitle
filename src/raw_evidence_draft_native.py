"""Raw-evidence native writer with exact request binding and frozen L1 mechanics.

The controller owns validated raw-evidence/input/producer pins and the local
stage deadline. Only ask dispatches generation. Capacity, receipt, stop/schema,
technical-retry and payload validation reuse frozen episode native helpers;
request-bound orchestration is separate and never patches predecessor modules.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
import time

from src import episode_draft_native as shared
from src import raw_evidence_draft_requests as drafts
from src import revisit_workflow as rw
from src import sparse_revisit as sr
from src.config import TranslateConfig
from src.contextual_review_contract import require
from src.translate import call_llm
from src.workflow_state import fingerprint, write_json

VERSION = 'raw-evidence-draft-native-1'
MAX_OUTPUT_TOKENS = shared.MAX_OUTPUT_TOKENS
CONTEXT_SIZE = shared.CONTEXT_SIZE
MAX_ATTEMPTS = shared.MAX_ATTEMPTS
Track = shared.Track
_ROOT_KEYS = shared._ROOT_KEYS
_ATTEMPT_KEYS = shared._ATTEMPT_KEYS
_same = shared._same
_seconds = shared._seconds
_timestamp = shared._timestamp
_payload = shared._payload
_capacity = shared._capacity
_receipts = shared._receipts
_parse_complete = shared._parse_complete
_technical_failure = shared._technical_failure
_metrics = shared._metrics
_local_track = shared._local_track
_remaining = shared._remaining


def prepare_request(request: dict) -> dict:
    """Bind the closed raw-evidence request to the fixed Qwen payload."""
    drafts.validate_writer_request(request)
    # The controller and release independently reconstruct the source-bound body.
    json.dumps(request, allow_nan=False, ensure_ascii=False)
    body = request['body']
    extra = {'model': rw.QWEN, 'temperature': .3, 'seed': 20260915, 'top_p': .95,
             'max_tokens': MAX_OUTPUT_TOKENS, 'chat_template_kwargs': {'enable_thinking': False},
             'reasoning_budget_tokens': 0, 'reasoning_effort': 'none',
             'response_format': {'type': 'json_object', 'schema': deepcopy(request['schema'])}}
    return {'version': VERSION, 'family': 'qwen', 'instruction': request['instruction'],
            'body': deepcopy(body), 'schema': deepcopy(request['schema']), 'extra': extra, 'thinking': False}


def config_for(endpoint: str, directory: Path, key: str, request: dict, remaining: float = 600) -> TranslateConfig:
    """Pure common template/config for capacity and the single HTTP attempt."""
    require(type(remaining) in (int, float) and math.isfinite(remaining), 'Invalid raw-evidence deadline')
    if remaining < 1:
        raise rw.LocalBudgetExceeded('No remaining raw-evidence draft budget')
    return TranslateConfig(endpoint=endpoint, max_tokens=MAX_OUTPUT_TOKENS,
        timeout=max(1, int(min(600, remaining))), retries=0, separate_instruction=True,
        response_guard_floor=65536, stage=key, telemetry_path=Path(directory)/'metrics.jsonl',
        extra_payload=prepare_request(request)['extra'])


def replay_native(path: Path, request: dict, track: Track) -> dict:
    """Reconstruct all successful-writer provenance from supplied artifacts only."""
    path = Path(path); saved = track(path); native = prepare_request(request); pin = fingerprint(native)
    require(isinstance(saved, dict) and set(saved) == _ROOT_KEYS
            and saved['receipt_sha256'] == fingerprint({key: value for key, value in saved.items() if key != 'receipt_sha256'})
            and _same(saved['request'], native) and saved['request_sha256'] == pin,
            'Raw-evidence native request or sealed ledger changed')
    attempts = saved['attempts']
    require(saved['status'] == 'complete' and isinstance(attempts, list) and 1 <= len(attempts) <= MAX_ATTEMPTS,
            'Raw-evidence request lacks bounded complete execution')
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
            'Unregistered native raw-evidence invocation or hidden retry')
    require(_same(parsed, saved['parsed']), 'Parsed raw-evidence draft differs from raw complete answer')
    return deepcopy(parsed)


def ask(directory: Path, key: str, endpoint: str, request: dict) -> dict:
    """One whole-episode generation, with at most one identical technical retry."""
    directory = Path(directory)
    require(isinstance(key, str) and re.fullmatch(r'[A-Za-z0-9_-]+', key) is not None, 'Invalid native request key')
    native = prepare_request(request); pin = fingerprint(native); path = directory/(key+'.json')
    if path.exists():
        saved = _local_track(path)
        require(saved.get('status') == 'complete', 'Incomplete saved raw-evidence request cannot be resumed')
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
            raise RuntimeError('One raw-evidence writer technical retry exhausted') from failure
    raise AssertionError('Unreachable raw-evidence draft retry state')
