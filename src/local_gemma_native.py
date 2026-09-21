"""One local Gemma text request with exact capacity, stream capture and replay.

No model loading, experiment ancestry or evaluator lives here. The caller binds
source/contract code and owns unconditional backend shutdown/restoration. A saved
failed/incomplete request is terminal; complete requests may only be replayed.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import time
from typing import Callable

import jsonschema
import requests
from src.config import TranslateConfig
from src.local_backend import _request_json
from src.translate import _build_payload
from src.workflow_state import fingerprint, write_json

VERSION = 'local-gemma-native-1'
ENDPOINT = 'http://127.0.0.1:18101/v1/chat/completions'
CAPACITY_BASE = 'http://127.0.0.1:18101'
# SSE framing and metadata are additional to the declared generation token limit.
MAX_STREAM_BYTES = 32 * 1024 * 1024


class TreatmentAdherenceError(ValueError):
    """A valid answer did not exhibit the declared reasoning treatment."""


def require(condition, message):
    if not condition: raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'Duplicate JSON key'); value[key] = item
        return value
    def invalid(value): raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def same(a, b): return fingerprint(a) == fingerprint(b)
def now(): return datetime.now(timezone.utc).isoformat()
def digest(raw): return hashlib.sha256(raw).hexdigest()
def number(value): return type(value) in (int, float) and math.isfinite(value) and value >= 0


@dataclass(frozen=True)
class GemmaSettings:
    model: str = 'gemma4-31b-qat-q4'
    model_sha256: str = '179cfb99212709597eae5929112cfca677e1bbf566178b479ae1da0c4772874b'
    context_size: int = 163840
    max_tokens: int = 16384
    full_swa: bool = False
    temperature: float = 1.0
    top_p: float = .95
    top_k: int = 64
    repeat_penalty: float = 1.0
    seed: int = 20260913
    thinking: bool = False
    reasoning_budget_tokens: int = 0
    maximum_seconds: int = 1980

    def __post_init__(self):
        require(type(self.model) is str and re.fullmatch(r'[A-Za-z0-9_.-]+', self.model), 'Invalid local model alias')
        require(type(self.model_sha256) is str and re.fullmatch(r'[0-9a-f]{64}', self.model_sha256), 'Invalid weight identity')
        for key in ('context_size', 'max_tokens', 'top_k', 'seed', 'reasoning_budget_tokens', 'maximum_seconds'):
            require(type(getattr(self, key)) is int and getattr(self, key) >= 0, 'Invalid integer setting: '+key)
        require(self.context_size >= 512 and 0 < self.max_tokens < self.context_size and self.maximum_seconds > 0,
                'Invalid context/output/time limit')
        require(type(self.thinking) is bool and type(self.full_swa) is bool, 'Invalid boolean setting')
        require(number(self.temperature) and self.temperature <= 2 and number(self.top_p) and 0 < self.top_p <= 1
                and number(self.repeat_penalty) and self.repeat_penalty > 0, 'Invalid sampler')
        require((self.thinking and 0 < self.reasoning_budget_tokens <= self.max_tokens)
                or (not self.thinking and self.reasoning_budget_tokens == 0), 'Thinking policy/budget mismatch')


def prepare_request(request: dict, settings: GemmaSettings, *, contract_id: str,
                    validate_request: Callable | None = None) -> dict:
    require(type(settings) is GemmaSettings, 'Explicit GemmaSettings required')
    require(type(request) is dict and set(request) == {'instruction', 'body', 'schema'}
            and type(request['instruction']) is str and bool(request['instruction'])
            and type(request['body']) is dict and type(request['schema']) is dict, 'Invalid complete request')
    require(type(contract_id) is str and re.fullmatch(r'[A-Za-z0-9_.:-]+', contract_id), 'Invalid contract identity')
    copied = deepcopy(request)
    if validate_request is not None:
        validate_request(copied); require(same(copied, request), 'Contract validator mutated the request')
    jsonschema.Draft202012Validator.check_schema(copied['schema'])
    values = asdict(settings)
    extra = {key: values[key] for key in ('model', 'max_tokens', 'temperature', 'top_p', 'top_k', 'repeat_penalty', 'seed')}
    extra.update(cache_prompt=False, reasoning_budget_tokens=settings.reasoning_budget_tokens,
                 chat_template_kwargs={'enable_thinking': settings.thinking},
                 response_format={'type': 'json_object', 'schema': copied['schema']})
    if not settings.thinking: extra['reasoning_effort'] = 'none'
    cfg = TranslateConfig(endpoint=ENDPOINT, max_tokens=settings.max_tokens, retries=0,
                          separate_instruction=True, telemetry_path=Path('unused-metrics.jsonl'), extra_payload=extra)
    payload = _build_payload(json.dumps(copied['body'], ensure_ascii=False, allow_nan=False), copied['instruction'],
                             cfg, {}, settings.max_tokens, stream=True, with_thinking=settings.thinking)
    return {'version': VERSION, 'contract_id': contract_id, 'settings': values, 'request': copied, 'payload': payload}


def _prepared(value):
    require(type(value) is dict and set(value) == {'version', 'contract_id', 'settings', 'request', 'payload'}, 'Prepared fields changed')
    expected = prepare_request(value['request'], GemmaSettings(**value['settings']), contract_id=value['contract_id'])
    require(same(value, expected), 'Prepared native settings/payload changed'); return expected


def _identity(identity, policy):
    require(type(identity) is dict and identity.get('model_alias') == policy.model
        and identity.get('model_sha256') == policy.model_sha256 and type(identity.get('pid')) is int and identity['pid'] > 0
        and same(identity.get('context_size'), policy.context_size) and identity.get('full_swa') is policy.full_swa
        and type(identity.get('model_path')) is str and Path(identity['model_path']).is_absolute()
        and type(identity.get('chat_template_sha256')) is str
        and re.fullmatch(r'[0-9a-f]{64}', identity['chat_template_sha256']), 'Owned backend identity differs')


def _props(props, identity, policy):
    _identity(identity, policy)
    require(type(props) is dict and props.get('model_alias') in (policy.model, [policy.model])
        and props.get('model_path') == identity['model_path']
        and same(props.get('default_generation_settings', {}).get('n_ctx'), policy.context_size)
        and type(props.get('chat_template')) is str
        and digest(props['chat_template'].encode('utf-8')) == identity['chat_template_sha256'],
        'Native served model/context/template differs')


def _capacity(prepared, identity):
    policy = GemmaSettings(**prepared['settings']); payload = prepared['payload']
    props = _request_json(CAPACITY_BASE+'/props', timeout=20); _props(props, identity, policy)
    prompt = _request_json(CAPACITY_BASE+'/apply-template', payload, timeout=20).get('prompt')
    require(type(prompt) is str and bool(prompt), 'Native full template missing')
    tokens = _request_json(CAPACITY_BASE+'/tokenize', {'content': prompt, 'add_special': True, 'parse_special': True}, timeout=20).get('tokens')
    require(type(tokens) is list and bool(tokens) and all(type(t) is int and t >= 0 for t in tokens), 'Native token IDs missing')
    return {'props': props, 'payload_sha256': fingerprint(payload), 'rendered_prompt': prompt, 'input_ids': tokens,
        'prompt_tokens': len(tokens), 'context': policy.context_size, 'reserve': policy.max_tokens, 'margin': 64,
        'fits': len(tokens)+policy.max_tokens+64 <= policy.context_size}


def _check_capacity(value, prepared, identity):
    policy = GemmaSettings(**prepared['settings']); _props(value['props'], identity, policy)
    require(set(value) == {'props', 'payload_sha256', 'rendered_prompt', 'input_ids', 'prompt_tokens', 'context', 'reserve', 'margin', 'fits'}, 'Capacity fields changed')
    ids = value['input_ids']; require(type(ids) is list and bool(ids) and all(type(t) is int and t >= 0 for t in ids), 'Token IDs changed')
    wanted = {'payload_sha256': fingerprint(prepared['payload']), 'prompt_tokens': len(ids), 'context': policy.context_size,
              'reserve': policy.max_tokens, 'margin': 64, 'fits': True}
    require(type(value['rendered_prompt']) is str and bool(value['rendered_prompt'])
        and all(same(value[k], v) for k, v in wanted.items()) and len(ids)+policy.max_tokens+64 <= policy.context_size,
        'Complete request/output reserve exceeds native capacity')


def decode_stream(raw: bytes, *, complete: bool = True) -> dict:
    """Replay exact SSE deltas; reasoning is retained separately from the answer."""
    require(type(raw) is bytes and len(raw) <= MAX_STREAM_BYTES, 'Invalid captured stream')
    answer, reasoning = [], []; finish = usage = timings = None; done = False
    for line in raw.decode('utf-8', errors='strict').splitlines():
        if not line or line.startswith(':'): continue
        require(not done and line.startswith('data:'), 'Unexpected SSE framing/trailing data')
        data = line[5:].lstrip(' ')
        if data == '[DONE]': done = True; continue
        chunk = strict_json(data); require(type(chunk) is dict, 'Invalid SSE event')
        if chunk.get('usage') is not None: usage = chunk['usage']
        if chunk.get('timings') is not None: timings = chunk['timings']
        choices = chunk.get('choices', []); require(type(choices) is list and len(choices) <= 1, 'Unexpected multiple stream choices')
        if not choices: continue
        choice = choices[0]
        require(type(choice) is dict and type(choice.get('index', 0)) is int
                and choice.get('index', 0) == 0, 'Unexpected stream choice')
        delta = choice.get('delta') or {}; require(type(delta) is dict, 'Invalid stream delta')
        for key, target in (('content', answer), ('reasoning_content', reasoning)):
            part = delta.get(key)
            if key == 'content' and part is None: part = choice.get('text')
            require(part is None or type(part) is str, 'Invalid text delta')
            if part is not None: target.append(part)
        value = choice.get('finish_reason')
        if value is not None:
            require(type(value) is str and (finish is None or finish == value), 'Conflicting native finish reason'); finish = value
    require(not complete or done, 'Native SSE did not finish')
    return {'content': ''.join(answer), 'reasoning': ''.join(reasoning), 'finish_reason': finish,
            'usage': usage, 'timings': timings, 'done': done}


def _response(value, prepared):
    policy = GemmaSettings(**prepared['settings']); usage = value['usage']
    require(value['done'] is True and value['finish_reason'] == 'stop', 'Native response did not stop normally')
    require(type(usage) is dict and all(type(usage.get(k)) is int for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
        and usage['prompt_tokens'] > 0 and 0 < usage['completion_tokens'] <= policy.max_tokens
        and usage['total_tokens'] == usage['prompt_tokens']+usage['completion_tokens']
        and usage['prompt_tokens']+policy.max_tokens+64 <= policy.context_size, 'Native usage missing or over capacity')
    parsed = strict_json(value['content']); jsonschema.validate(parsed, prepared['request']['schema'])
    if not (bool(value['reasoning']) if policy.thinking else value['reasoning'] == ''):
        raise TreatmentAdherenceError('Observed reasoning does not match treatment')
    return parsed


@contextmanager
def _within(deadline):
    start = time.monotonic(); remaining = deadline-start; require(remaining > 0, 'Native deadline exhausted')
    handler = signal.getsignal(signal.SIGALRM); timer = signal.getitimer(signal.ITIMER_REAL)
    if timer[0]: remaining = min(remaining, timer[0])
    def expired(*_): raise TimeoutError('Local native deadline exceeded')
    signal.signal(signal.SIGALRM, expired); signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
        if time.monotonic()-start >= remaining: raise TimeoutError('Local native deadline exceeded')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, handler)
        if timer[0]: signal.setitimer(signal.ITIMER_REAL, max(.001, timer[0]-(time.monotonic()-start)), timer[1])


def _read(path, raw=False): return Path(path).read_bytes() if raw else strict_json(Path(path).read_bytes())


def replay_native(path, prepared, backend_identity, track=None):
    track = track or _read; path = Path(path); expected = _prepared(prepared); saved = track(path)
    require(set(saved) == {'version', 'key', 'prepared', 'prepared_sha256', 'backend_identity', 'status', 'attempt', 'response', 'parsed', 'receipt_sha256'}, 'Receipt fields changed')
    require(saved['version'] == VERSION and saved['key'] == path.stem and saved['status'] == 'complete'
        and same(saved['prepared'], expected) and saved['prepared_sha256'] == fingerprint(expected)
        and same(saved['backend_identity'], backend_identity)
        and saved['receipt_sha256'] == fingerprint({k:v for k,v in saved.items() if k != 'receipt_sha256'}), 'Native receipt identity/completion changed')
    policy = GemmaSettings(**expected['settings']); _identity(backend_identity, policy); attempt = saved['attempt']
    require(set(attempt) == {'number', 'started_utc', 'finished_utc', 'seconds', 'maximum_seconds', 'capacity_started_utc', 'capacity_finished_utc',
        'capacity', 'generation_started_utc', 'generation_finished_utc', 'generation_http_calls', 'http_status', 'stream_sha256', 'error_type'}, 'Attempt fields changed')
    require(type(attempt['number']) is int and attempt['number'] == 1 and type(attempt['generation_http_calls']) is int
        and attempt['generation_http_calls'] == 1 and attempt['http_status'] == 200 and attempt['error_type'] is None
        and number(attempt['seconds']) and number(attempt['maximum_seconds'])
        and 0 < attempt['maximum_seconds'] <= policy.maximum_seconds and attempt['seconds'] <= attempt['maximum_seconds'], 'Native attempt failed, repeated or over budget')
    times = [datetime.fromisoformat(attempt[k]) for k in ('started_utc','capacity_started_utc','capacity_finished_utc',
             'generation_started_utc','generation_finished_utc','finished_utc')]
    require(all(t.utcoffset() is not None for t in times) and times == sorted(times), 'Native temporal order differs')
    _check_capacity(attempt['capacity'], expected, backend_identity)
    raw = track(path.with_suffix('.sse'), raw=True); require(digest(raw) == attempt['stream_sha256'], 'Native SSE capture changed')
    response = decode_stream(raw); parsed = _response(response, expected)
    require(same(response, saved['response']) and same(parsed, saved['parsed']), 'Raw stream/parsed output differ'); return parsed


def ask(directory, key, prepared, backend_identity, deadline):
    expected = _prepared(prepared); policy = GemmaSettings(**expected['settings']); _identity(backend_identity, policy)
    require(type(key) is str and re.fullmatch(r'[A-Za-z0-9_-]+', key), 'Invalid native request key')
    require(number(deadline), 'Invalid absolute deadline'); directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/(key+'.json'); stream_path = path.with_suffix('.sse')
    if path.exists(): return replay_native(path, expected, backend_identity)
    require(not stream_path.exists(), 'Orphan native stream blocks dispatch')
    start = time.monotonic(); limit = min(policy.maximum_seconds, deadline-start); require(limit >= 1, 'No native attempt time remains')
    fields = ('finished_utc','seconds','capacity_started_utc','capacity_finished_utc','capacity','generation_started_utc',
              'generation_finished_utc','http_status','stream_sha256','error_type')
    attempt = {key:None for key in fields}; attempt.update(number=1, started_utc=now(), maximum_seconds=limit, generation_http_calls=0)
    saved = {'version':VERSION, 'key':key, 'prepared':expected, 'prepared_sha256':fingerprint(expected),
             'backend_identity':deepcopy(backend_identity), 'status':'running', 'attempt':attempt, 'response':None, 'parsed':None}
    def save():
        saved['receipt_sha256'] = fingerprint({k:v for k,v in saved.items() if k != 'receipt_sha256'}); write_json(path, saved)
    with path.open('x', encoding='utf-8') as handle: json.dump(saved, handle)
    failure = None
    try:
        with _within(start+limit):
            attempt['capacity_started_utc'] = now(); save()
            try: attempt['capacity'] = _capacity(expected, backend_identity); save(); _check_capacity(attempt['capacity'], expected, backend_identity)
            finally: attempt['capacity_finished_utc'] = now(); save()
            attempt['generation_started_utc'] = now(); attempt['generation_http_calls'] = 1; save()
            response = None
            try:
                response = requests.post(ENDPOINT, json=deepcopy(expected['payload']), timeout=(10, max(1, math.ceil(start+limit-time.monotonic()))),
                                         stream=True, allow_redirects=False)
                attempt['http_status'] = response.status_code; save(); require(response.status_code == 200, 'Native HTTP request failed')
                total = 0
                with stream_path.open('xb') as handle:
                    for line in response.iter_lines(decode_unicode=False):
                        require(type(line) is bytes, 'Transport must preserve native bytes'); total += len(line)+1
                        handle.write(line+b'\n'); handle.flush(); require(total <= MAX_STREAM_BYTES, 'Native stream capture limit exceeded')
            finally:
                if response is not None: response.close()
                attempt['generation_finished_utc'] = now(); save()
            saved['response'] = decode_stream(stream_path.read_bytes()); save(); saved['parsed'] = _response(saved['response'], expected)
    except BaseException as exc:
        failure = exc; attempt['error_type'] = type(exc).__name__
        if stream_path.exists():
            try: saved['response'] = decode_stream(stream_path.read_bytes(), complete=False)
            except (ValueError, UnicodeError, TypeError): pass
    finally:
        if stream_path.exists(): attempt['stream_sha256'] = digest(stream_path.read_bytes())
        elapsed = time.monotonic()-start
        if failure is None and elapsed > limit:
            failure = TimeoutError('Local native deadline exceeded')
            attempt['error_type'] = type(failure).__name__
        attempt.update(finished_utc=now(), seconds=elapsed); saved['status'] = 'failed' if failure else 'complete'; save()
    if failure: raise failure
    return replay_native(path, expected, backend_identity)
