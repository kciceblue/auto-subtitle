"""One source-only C-ASR1 recap; backend ownership stays with the outer runner.

Only ask dispatches. Replay is network-free. The shared streaming helper is used
unchanged, with no internal fallback. Lost partial output on a transport exception
is recorded as unavailable and is terminal. A single identical retry is permitted
only for a captured empty response or lone opening JSON brace without a finish marker.
Normal stops, length stops and invalid recap content never authorize another call.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import signal
import time
from typing import Callable

import jsonschema

from src import contextual_asr_requests as contract
from src import evidence_context as ec
from src import revisit_workflow as rw
from src.config import TranslateConfig
from src.contextual_review_contract import strict_json
from src.translate import StreamResult, _build_payload, _record_request, _stream_response
from src.workflow_state import write_json

VERSION = 'contextual-asr-recap-native-1'
CONTEXT = 196608
RESERVE = 4096
CAPACITY_ENDPOINT = 'http://127.0.0.1:18089'
_ENDPOINTS = {'http://127.0.0.1:8089/v1/chat/completions',
              'http://127.0.0.1:18089/v1/chat/completions'}
_ROOT_KEYS = {'version', 'native', 'endpoint', 'key', 'status', 'attempts', 'parsed', 'receipt_sha256'}
_ATTEMPT_KEYS = {'number', 'payload_sha256', 'started_utc', 'finished_utc', 'seconds',
                 'capacity_started_utc', 'capacity_finished_utc', 'capacity',
                 'generation_started_utc', 'generation_finished_utc', 'raw',
                 'metrics_start', 'metrics_end', 'telemetry', 'exception_type', 'failure_kind'}
_TELEMETRY_KEYS = {'stage', 'attempt', 'seconds', 'thinking', 'answer_chars', 'reasoning_chars',
                   'finish_reason', 'usage', 'timings', 'request_settings'}
_CAPACITY_KEYS = {'endpoint', 'payload_sha256', 'props', 'rendered_prompt', 'input_ids',
                  'prompt_tokens', 'reserved_output', 'context', 'margin', 'fits'}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _same(left: object, right: object) -> bool:
    return ec._hash(left) == ec._hash(right)


def _number(value: object) -> bool:
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def _timestamp(value: object) -> datetime:
    ec._require(type(value) is str, 'Missing native timestamp')
    parsed = datetime.fromisoformat(value)
    ec._require(parsed.utcoffset() is not None, 'Native timestamp lacks timezone')
    return parsed


def _config(endpoint: str, directory: Path, key: str, preparation: dict, timeout: int = 300) -> TranslateConfig:
    settings = preparation['recap_settings']
    extra = {'model': settings['model'], 'temperature': settings['temperature'], 'top_p': settings['top_p'],
             'seed': settings['seed'], 'max_tokens': RESERVE,
             'chat_template_kwargs': {'enable_thinking': False}, 'reasoning_budget_tokens': 0,
             'reasoning_effort': 'none',
             'response_format': {'type': 'json_object', 'schema': deepcopy(preparation['recap_request']['schema'])}}
    return TranslateConfig(endpoint=endpoint, max_tokens=RESERVE, timeout=timeout, retries=0,
                           separate_instruction=True, response_guard_floor=16384,
                           stage=key, telemetry_path=directory/'metrics.jsonl', extra_payload=extra)


def prepare_request(preparation: dict) -> dict:
    """Pure exact native body/schema/payload, fixed to 4096 total output tokens."""
    contract._check_preparation(preparation)
    request = preparation['recap_request']
    cfg = _config('http://127.0.0.1:8089/v1/chat/completions', Path('.'), 'unused', preparation)
    payload = _build_payload(json.dumps(request['body'], ensure_ascii=False), request['instruction'],
                             cfg, {}, RESERVE, stream=True, with_thinking=False)
    return {'version': VERSION, 'preparation_sha256': preparation['preparation_sha256'],
            'request': deepcopy(request), 'payload': payload}


def _check_capacity(capacity: dict, payload: dict) -> None:
    ec._keys(capacity, _CAPACITY_KEYS)
    ec._require(capacity['endpoint'] == CAPACITY_ENDPOINT and capacity['payload_sha256'] == ec._hash(payload),
                'Native capacity endpoint or payload differs')
    props = capacity['props']
    settings = props.get('default_generation_settings') if type(props) is dict else None
    ec._require(type(settings) is dict and type(settings.get('n_ctx')) is int and settings['n_ctx'] == CONTEXT,
                'Native Qwen context is not the declared 196608')
    ec._text(capacity['rendered_prompt'])
    ids = capacity['input_ids']
    ec._require(type(ids) is list and bool(ids) and all(type(token) is int and token >= 0 for token in ids),
                'Native template tokens missing')
    ec._require(type(capacity['prompt_tokens']) is int and capacity['prompt_tokens'] == len(ids) and
                type(capacity['reserved_output']) is int and capacity['reserved_output'] == RESERVE and
                type(capacity['context']) is int and capacity['context'] == CONTEXT and
                type(capacity['margin']) is int and capacity['margin'] == 64 and capacity['fits'] is True and
                len(ids)+RESERVE+64 <= CONTEXT, 'Native recap capacity failed')


def _capacity(payload: dict) -> dict:
    props = rw._request_json(CAPACITY_ENDPOINT+'/props', timeout=20)
    rendered = rw._request_json(CAPACITY_ENDPOINT+'/apply-template', payload, timeout=20).get('prompt')
    ec._text(rendered)
    ids = rw._request_json(CAPACITY_ENDPOINT+'/tokenize',
                           {'content': rendered, 'add_special': True, 'parse_special': True}, timeout=20).get('tokens')
    ec._require(type(ids) is list, 'Native token list unavailable')
    value = {'endpoint': CAPACITY_ENDPOINT, 'payload_sha256': ec._hash(payload), 'props': props,
             'rendered_prompt': rendered, 'input_ids': ids, 'prompt_tokens': len(ids),
             'reserved_output': RESERVE, 'context': props.get('default_generation_settings', {}).get('n_ctx'),
             'margin': 64, 'fits': len(ids)+RESERVE+64 <= CONTEXT}
    # The caller saves this raw capacity receipt before applying the gate.
    return value


def _metrics(raw: bytes) -> list:
    text = raw.decode('utf-8', errors='strict')
    ec._require(not text or text.endswith('\n'), 'Incomplete native telemetry line')
    rows = [strict_json(line) for line in text.splitlines()]
    ec._require(all(type(row) is dict for row in rows), 'Invalid native telemetry row')
    return rows


def _track(path: Path, raw: bool = False):
    data = Path(path).read_bytes()
    return data if raw else strict_json(data)


def _telemetry(attempt: dict, native: dict, key: str) -> dict:
    rows = attempt['telemetry']
    ec._require(type(rows) is list and len(rows) == 1, 'Exactly one shared transport receipt required')
    row = rows[0]; ec._keys(row, _TELEMETRY_KEYS)
    extra = native['payload']
    expected = {name: extra[name] for name in ('model', 'max_tokens', 'reasoning_budget_tokens',
                'reasoning_effort', 'temperature', 'top_p', 'seed', 'stream')}
    expected['enable_thinking'] = False
    ec._require(row['stage'] == key and type(row['attempt']) is int and row['attempt'] == attempt['number']-1
                and _number(row['seconds']) and row['thinking'] is False
                and type(row['answer_chars']) is int and row['answer_chars'] >= 0
                and type(row['reasoning_chars']) is int and row['reasoning_chars'] == 0
                and _same(row['request_settings'], expected), 'Native recap telemetry settings changed')
    raw = attempt['raw']
    ec._require(type(raw) is str and row['answer_chars'] == len(raw), 'Native raw capture differs from receipt')
    ec._require(row['finish_reason'] is None or type(row['finish_reason']) is str, 'Invalid finish marker')
    if row['usage'] is not None:
        ec._require(type(row['usage']) is dict, 'Invalid native usage')
        for name in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            if name in row['usage']:
                ec._require(type(row['usage'][name]) is int and row['usage'][name] >= 0, 'Invalid native token count')
        if 'completion_tokens' in row['usage']:
            ec._require(row['usage']['completion_tokens'] <= RESERVE, 'Recap output reserve exceeded')
    return row


def _parsed(raw: str, preparation: dict) -> dict:
    response = strict_json(raw)
    jsonschema.validate(response, preparation['recap_request']['schema'])
    contract.format_hint(preparation, response)
    return response


def _retryable(attempt: dict, native: dict, key: str) -> bool:
    row = _telemetry(attempt, native, key)
    if row['finish_reason'] is not None:
        return False
    # A substantive partial answer may already be a complete but malformed
    # answer. Do not infer retry permission from its JSON/schema failure.
    return attempt['raw'].strip() in ('', '{')


@contextmanager
def _within(deadline: float):
    """Bound actual walltime while respecting and restoring an earlier outer alarm."""
    started = time.monotonic(); remaining = deadline-started
    if remaining <= 0: raise rw.LocalBudgetExceeded('C-ASR recap deadline exhausted')
    previous = signal.getsignal(signal.SIGALRM); timer = signal.getitimer(signal.ITIMER_REAL)
    armed = not timer[0] or remaining < timer[0]
    def expired(signum, frame):
        raise rw.LocalBudgetExceeded('C-ASR recap deadline exhausted')
    try:
        if armed:
            signal.signal(signal.SIGALRM, expired)
            signal.setitimer(signal.ITIMER_REAL, remaining)
        yield
        if time.monotonic() > deadline: raise rw.LocalBudgetExceeded('C-ASR recap deadline exhausted')
    finally:
        if armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            if timer[0]: signal.setitimer(signal.ITIMER_REAL, max(1e-6, timer[0]-(time.monotonic()-started)), timer[1])


def replay_native(path: Path, preparation: dict, track: Callable) -> dict:
    """Replay complete local native artifacts without contacting an endpoint."""
    path = Path(path); saved = track(path); ec._keys(saved, _ROOT_KEYS)
    native = prepare_request(preparation)
    ec._require(saved['version'] == VERSION and _same(saved['native'], native) and
                saved['endpoint'] in _ENDPOINTS and saved['key'] == path.stem and saved['status'] == 'complete' and
                saved['receipt_sha256'] == ec._hash({key: value for key, value in saved.items() if key != 'receipt_sha256'}),
                'Native recap binding or completion changed')
    attempts = saved['attempts']
    ec._require(type(attempts) is list and 1 <= len(attempts) <= 2, 'Native recap attempt ceiling exceeded')
    metrics = _metrics(track(path.parent/'metrics.jsonl', raw=True))
    prior_end = None; prior_metric_end = None; prior_capacity = None; used = []; parsed = None
    for number, attempt in enumerate(attempts, 1):
        ec._keys(attempt, _ATTEMPT_KEYS)
        ec._require(type(attempt['number']) is int and attempt['number'] == number and
                    attempt['payload_sha256'] == ec._hash(native['payload']) and
                    _number(attempt['seconds']) and attempt['seconds'] <= 300,
                    'Native recap attempt identity or time budget changed')
        times = [_timestamp(attempt[key]) for key in ('started_utc', 'capacity_started_utc',
            'capacity_finished_utc', 'generation_started_utc', 'generation_finished_utc', 'finished_utc')]
        ec._require(times == sorted(times) and (prior_end is None or times[0] >= prior_end), 'Native stage ordering changed')
        prior_end = times[-1]
        _check_capacity(attempt['capacity'], native['payload'])
        ec._require(prior_capacity is None or _same(attempt['capacity'], prior_capacity), 'Capacity changed on technical retry')
        prior_capacity = attempt['capacity']
        start, end = attempt['metrics_start'], attempt['metrics_end']
        ec._require(type(start) is int and type(end) is int and 0 <= start < end <= len(metrics) and
                    (prior_metric_end is None or start == prior_metric_end) and
                    _same(attempt['telemetry'], metrics[start:end]), 'Native telemetry range changed')
        prior_metric_end = end; used.extend(range(start, end))
        row = _telemetry(attempt, native, path.stem)
        if number < len(attempts):
            ec._require(attempt['failure_kind'] == 'incomplete_response' and
                        attempt['exception_type'] == 'IncompleteRecapResponse' and _retryable(attempt, native, path.stem),
                        'A completed or unauthenticated response was rerolled')
        else:
            ec._require(attempt['failure_kind'] is None and attempt['exception_type'] is None and
                        row['finish_reason'] == 'stop', 'Recap lacks normal successful stop')
            ec._require(type(row['usage']) is dict and type(row['usage'].get('completion_tokens')) is int and
                        row['usage']['completion_tokens'] > 0,
                        'Native completed response lacks output-token accounting')
            parsed = _parsed(attempt['raw'], preparation)
    ec._require(used == [index for index, row in enumerate(metrics) if row.get('stage') == path.stem],
                'Unregistered recap call or hidden retry')
    ec._require(_same(saved['parsed'], parsed), 'Parsed recap differs from preserved raw')
    return deepcopy(parsed)


class IncompleteRecapResponse(ValueError):
    pass


def ask(directory: Path, key: str, endpoint: str, preparation: dict, deadline: float) -> dict:
    """Return the one raw-parsed recap; at most one evidenced identical retry.

    deadline is the runner's absolute monotonic work cutoff. No backend action,
    automatic resume of incomplete attempts, schema repair or changed prompt.
    """
    ec._require(endpoint in _ENDPOINTS and type(key) is str and re.fullmatch(r'[A-Za-z0-9_-]+', key) is not None,
                'Invalid local recap endpoint or key')
    ec._require(_number(deadline), 'Invalid absolute recap deadline')
    native = prepare_request(preparation); directory = Path(directory); path = directory/(key+'.json')
    if path.exists():
        saved = _track(path)
        ec._require(saved.get('status') == 'complete' and saved.get('endpoint') == endpoint,
                    'Incomplete recap cannot be resumed or endpoint changed')
        return replay_native(path, preparation, _track)
    directory.mkdir(parents=True, exist_ok=True)
    metric_path = directory/'metrics.jsonl'
    prior = _metrics(metric_path.read_bytes()) if metric_path.exists() else []
    ec._require(not any(row.get('stage') == key for row in prior), 'Orphan recap telemetry exists')
    saved = {'version': VERSION, 'native': native, 'endpoint': endpoint, 'key': key,
             'status': 'prepared', 'attempts': [], 'parsed': None}
    def save():
        saved['receipt_sha256'] = ec._hash({name: value for name, value in saved.items() if name != 'receipt_sha256'})
        write_json(path, saved)
    save(); prior_capacity = None
    for number in range(1, 3):
        started = time.monotonic(); limit = min(deadline, started+300)
        attempt = {name: None for name in _ATTEMPT_KEYS}
        attempt.update(number=number, started_utc=_now(), payload_sha256=ec._hash(native['payload']), telemetry=[])
        saved['attempts'].append(attempt); saved['status'] = 'running'; save()
        failure = None; phase = 'budget'; retry = False
        try:
            with _within(limit):
                phase = 'capacity'; attempt['capacity_started_utc'] = _now(); save()
                try:
                    attempt['capacity'] = _capacity(native['payload']); save()
                    _check_capacity(attempt['capacity'], native['payload'])
                    ec._require(prior_capacity is None or _same(prior_capacity, attempt['capacity']),
                                'Native capacity changed before retry')
                    prior_capacity = deepcopy(attempt['capacity'])
                finally:
                    attempt['capacity_finished_utc'] = _now(); save()
                phase = 'provenance'
                before = _metrics(metric_path.read_bytes()) if metric_path.exists() else []
                attempt['metrics_start'] = len(before)
                cfg = _config(endpoint, directory, key, preparation, max(1, math.ceil(limit-time.monotonic())))
                phase = 'transport'; attempt['generation_started_utc'] = _now(); save()
                generation_start = time.monotonic()
                result = None
                try:
                    result = _stream_response(endpoint, deepcopy(native['payload']), cfg.timeout, cfg.response_guard_floor)
                    attempt['raw'] = result.content
                    save()  # Save failed or successful raw before any parser/stop/telemetry gate.
                finally:
                    attempt['generation_finished_utc'] = _now()
                    recorded = result or StreamResult(content='', finish_reason='capture_unavailable')
                    _record_request(cfg, number-1, generation_start, False, recorded, payload=native['payload'])
                    after = _metrics(metric_path.read_bytes()) if metric_path.exists() else []
                    attempt['metrics_end'] = len(after); attempt['telemetry'] = after[len(before):]
                    save()
                    ec._require(_same(after[:len(before)], before), 'Telemetry prefix changed')
                phase = 'provenance'; row = _telemetry(attempt, native, key)
                if _retryable(attempt, native, key):
                    phase = 'incomplete_response'; retry = True
                    raise IncompleteRecapResponse('Captured empty response or lone opening brace lacks a finish marker')
                phase = 'response'
                ec._require(row['finish_reason'] == 'stop', 'Recap stopped abnormally; no retry')
                ec._require(type(row['usage']) is dict and type(row['usage'].get('completion_tokens')) is int and
                        row['usage']['completion_tokens'] > 0,
                            'Completed response lacks output-token accounting')
                saved['parsed'] = _parsed(attempt['raw'], preparation)
        except BaseException as error:
            failure = error
            attempt['exception_type'] = type(error).__name__
            attempt['failure_kind'] = ('budget' if isinstance(error, rw.LocalBudgetExceeded) else
                                       'interrupted' if not isinstance(error, Exception) else phase)
        finally:
            attempt['finished_utc'] = _now(); attempt['seconds'] = time.monotonic()-started
            saved['status'] = 'complete' if failure is None else 'failed'; save()
        if failure is None:
            return replay_native(path, preparation, _track)
        if not retry or number == 2:
            raise failure
    raise AssertionError('Unreachable recap request state')
