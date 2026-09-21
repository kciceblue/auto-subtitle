"""Authenticate a failed C-ASR recap before a separate deterministic projection.

Read-only replay via the supplied tracker. The original native attempt remains
failed; the derived response is not represented as its original successful output.
There is no inference, retry, tokenizer, backend or file-writing API here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Callable

import jsonschema

from src import contextual_asr_recap_native as original
from src import contextual_asr_recap_projection as projection
from src import contextual_asr_requests as contract
from src import evidence_context as ec
from src.contextual_review_contract import strict_json

VERSION = 'contextual-asr-projected-recap-native-1'
ORIGINAL_ENDPOINT = 'http://127.0.0.1:8089/v1/chat/completions'


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def replay_native(path: Path, preparation: dict, track: Callable) -> dict:
    """Return derived response/audit after exact failed-native content validation.

    ``track(path, raw=True)`` must return the pinned file bytes and independently
    authenticate ancestor identity. This function validates their association and
    mechanics; it neither invents provenance nor changes the upstream receipt.
    Only the original one-attempt normal-stop response/ValueError case is admitted.
    """
    ec._require(callable(track), 'Pinned artifact tracker required')
    path = Path(path)
    native_bytes = track(path, raw=True)
    ec._require(type(native_bytes) is bytes, 'Tracker must supply exact native bytes')
    saved = strict_json(native_bytes); ec._keys(saved, original._ROOT_KEYS)
    expected = original.prepare_request(preparation)
    ec._require(saved['version'] == original.VERSION and original._same(saved['native'], expected)
        and saved['endpoint'] == ORIGINAL_ENDPOINT and saved['key'] == path.stem
        and saved['status'] == 'failed' and saved['parsed'] is None
        and saved['receipt_sha256'] == ec._hash({k: v for k, v in saved.items() if k != 'receipt_sha256'}),
        'Original failed recap binding or seal changed')
    attempts = saved['attempts']
    ec._require(type(attempts) is list and len(attempts) == 1, 'Projection requires the single original attempt')
    attempt = attempts[0]; ec._keys(attempt, original._ATTEMPT_KEYS)
    ec._require(type(attempt['number']) is int and attempt['number'] == 1
        and attempt['payload_sha256'] == ec._hash(expected['payload'])
        and original._number(attempt['seconds']) and attempt['seconds'] <= 300
        and attempt['failure_kind'] == 'response' and attempt['exception_type'] == 'ValueError',
        'Original failure is not the bounded content-contract ValueError')
    times = [original._timestamp(attempt[key]) for key in ('started_utc', 'capacity_started_utc',
        'capacity_finished_utc', 'generation_started_utc', 'generation_finished_utc', 'finished_utc')]
    ec._require(times == sorted(times), 'Original native stage ordering changed')
    original._check_capacity(attempt['capacity'], expected['payload'])
    metrics_bytes = track(path.parent/'metrics.jsonl', raw=True)
    ec._require(type(metrics_bytes) is bytes, 'Tracker must supply exact telemetry bytes')
    metrics = original._metrics(metrics_bytes)
    ec._require(type(attempt['metrics_start']) is int and attempt['metrics_start'] == 0
        and type(attempt['metrics_end']) is int and attempt['metrics_end'] == 1
        and len(metrics) == 1 and original._same(attempt['telemetry'], metrics),
        'Original telemetry range or complete call coverage changed')
    telemetry = original._telemetry(attempt, expected, path.stem)
    ec._require(telemetry['finish_reason'] == 'stop'
        and telemetry['seconds'] <= attempt['seconds']
        and type(telemetry['usage']) is dict
        and type(telemetry['usage'].get('completion_tokens')) is int
        and 0 < telemetry['usage']['completion_tokens'] <= original.RESERVE,
        'Original response lacks normal stop or bounded output-token accounting')
    raw = attempt['raw']; response = strict_json(raw)
    try:
        jsonschema.validate(response, preparation['recap_request']['schema'])
    except jsonschema.ValidationError:
        raise ValueError('Original raw recap fails the complete native schema') from None
    # Schema-valid content must reproduce the failed content check. A forged
    # failure label on a valid recap, or a failed transport, is never recoverable.
    try:
        contract.format_hint(preparation, response)
    except ValueError as error:
        ec._require(type(error) is ValueError, 'Original failure type does not match the content contract')
    else:
        raise ValueError('Original recap does not reproduce its content-contract failure')
    native_sha256 = _sha(native_bytes)
    compiled = projection.compile_recap(preparation, raw, native_sha256=native_sha256)
    projection.validate_projection(compiled, preparation, raw, native_sha256=native_sha256)
    result = {'version': VERSION, 'response': deepcopy(compiled['response']), 'audit': deepcopy(compiled['audit']),
        'projection_version': compiled['version'], 'projection_sha256': compiled['projection_sha256'],
        'parent': {'native_path': str(path), 'native_file_sha256': native_sha256,
            'native_receipt_sha256': saved['receipt_sha256'], 'raw_sha256': _sha(raw.encode('utf-8')),
            'metrics_file_sha256': _sha(metrics_bytes), 'native_request_sha256': ec._hash(expected),
            'payload_sha256': attempt['payload_sha256'], 'preparation_sha256': preparation['preparation_sha256'],
            'status': 'failed', 'parsed': None, 'failure_kind': 'response', 'exception_type': 'ValueError',
            'failure_reproduced': True, 'attempts': 1, 'finish_reason': 'stop',
            'answer_chars': telemetry['answer_chars'], 'reasoning_chars': telemetry['reasoning_chars'],
            'completion_tokens': telemetry['usage']['completion_tokens'], 'new_inference_calls': 0}}
    result['receipt_sha256'] = ec._hash(result)
    return result
