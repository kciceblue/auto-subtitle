"""Immutable, text-only subtitle quality review with the independent v5 rubric.

Preparation and receipt validation are local. Only explicit execution exports
the three supplied texts. A score never certifies audio, playback, or release.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time
import uuid

from src import subtitle_quality_contract as contract

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_SECONDS = 600
MAX_PROMPT_BYTES = 2_000_000
FILENAMES = {'source': 'input-source.srt', 'target': 'input-target.srt',
             'context': 'input-context.txt'}
CERTIFICATIONS = ('audio_reviewed', 'video_reviewed', 'audio_source_fidelity_certified',
                  'full_media_coverage_verified', 'playback_verified', 'release_gate_checked',
                  'overall_quality_certified')
IDENTITY = {'model': contract.MODEL, 'provider': 'openai', 'reasoning effort': 'high'}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file(value: str | Path) -> Path:
    path = Path(value)
    contract.require(path.is_absolute() and '..' not in path.parts and path.is_file()
                     and not any(part.is_symlink() for part in (path, *path.parents)),
                     'Input must be a plain absolute file without symlink components')
    return path


def _directory(value: str | Path) -> Path:
    path = Path(value).absolute()
    contract.require('..' not in path.parts
                     and not any(part.is_symlink() for part in (path, *path.parents)),
                     'Review directory cannot use symlinks')
    return path


def _read(path: str | Path) -> dict:
    value = contract.strict_json(_file(path).read_bytes())
    contract.require(isinstance(value, dict), 'Expected a JSON object')
    return value


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def _write(path: Path, value: bytes | dict) -> None:
    with path.open('xb') as stream:
        stream.write(value if isinstance(value, bytes) else _json_bytes(value))


def _bound_bytes(spec: dict) -> bytes:
    contract.require(isinstance(spec, dict) and set(spec) == {'path', 'sha256'}
                     and isinstance(spec['path'], str) and isinstance(spec['sha256'], str)
                     and re.fullmatch(r'[0-9a-f]{64}', spec['sha256']) is not None,
                     'Invalid manifest file binding')
    raw = _file(spec['path']).read_bytes()
    contract.require(sha256(raw) == spec['sha256'], 'Manifest input hash changed')
    return raw


def _cue_count(text: str) -> int:
    """Validate every cue independently; simultaneous dialogue/lyrics are valid.

    ASR windows and authored captions need not share cue boundaries, counts, or
    ordering by time. IDs remain consecutive within each complete supplied SRT.
    """
    normalized = text.removeprefix('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
    stamp = r'[0-9]{2}:[0-5][0-9]:[0-5][0-9],[0-9]{3}'
    chunks = re.split(r'\n[ \t]*\n', normalized.strip())
    for index, chunk in enumerate(chunks, 1):
        lines = chunk.strip().splitlines()
        contract.require(len(lines) >= 3 and re.fullmatch(r'[0-9]+', lines[0]) is not None
                         and int(lines[0]) == index, 'Malformed or nonconsecutive SRT cue ID')
        match = re.fullmatch(f'({stamp}) --> ({stamp})', lines[1])
        contract.require(match is not None and match[1] < match[2], 'Invalid SRT interval')
        body = lines[2:]
        contract.require(any(line.strip() for line in body), 'Empty SRT cue')
        contract.require(not any(re.match(r'\s*\d{2}:\d{2}:\d{2},\d{3}\s*-->', line)
                                 for line in body), 'Embedded cue timestamp in SRT text')
    return len(chunks)


def read_bundle(manifest: str | Path) -> dict:
    """Read exact private texts and provenance; do not print the returned bundle."""
    path = _file(manifest)
    raw_manifest = path.read_bytes()
    value = contract.strict_json(raw_manifest)
    contract.require(isinstance(value, dict) and set(value) == {'version', *contract.ROLES}
                     and value['version'] == contract.MANIFEST_VERSION,
                     'Invalid subtitle quality manifest')
    inputs = {}; texts = {}; raw_inputs = {}
    for role in contract.ROLES:
        raw = _bound_bytes(value[role])
        text = raw.decode('utf-8', errors='strict')
        contract.require(not any(ord(char) < 32 and char not in '\r\n\t' for char in text),
                         'Control character in review text')
        counts = {'bytes': len(raw), 'characters': len(text)}
        if role != 'context':
            counts['cues'] = _cue_count(text)
        inputs[role] = {**value[role], 'counts': counts}
        texts[role] = text; raw_inputs[role] = raw
    contract.require(path.read_bytes() == raw_manifest
                     and all(_bound_bytes(value[role]) == raw_inputs[role] for role in contract.ROLES),
                     'Bundle changed while reading')
    return {'manifest_path': str(path), 'manifest_sha256': sha256(raw_manifest),
            'manifest_raw': raw_manifest, 'inputs': inputs, 'texts': texts, 'raw_inputs': raw_inputs}


def _prompt(bundle: dict) -> bytes:
    prompt = contract.build_prompt(**bundle['texts']).encode('utf-8')
    contract.require(len(prompt) <= MAX_PROMPT_BYTES, 'Complete prompt exceeds size bound')
    return prompt


def _command(directory: Path) -> list[str]:
    return ['codex', 'exec', '--model', contract.MODEL, '-c', 'model_reasoning_effort="high"',
            '--ephemeral', '--sandbox', 'read-only', '--skip-git-repo-check', '--cd', '/tmp',
            '--color', 'never', '--output-schema', str(directory / 'review-schema.json'),
            '--output-last-message', str(directory / 'review-response.json'), '-']


def _producer_files() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), Path(contract.__file__).resolve(),
            ROOT / 'scripts/review_subtitle_quality.py', ROOT / 'src/contextual_review_contract.py')


def prepare_review(manifest: str | Path, output_dir: str | Path) -> dict:
    """Freeze all inputs, prompt, schema and producing code; never overwrite."""
    directory = _directory(output_dir)
    if directory.exists():
        return _validate_preparation(directory, manifest, current_code=True)
    bundle = read_bundle(manifest)
    prompt = _prompt(bundle); schema = _json_bytes(contract.SCHEMA)
    directory.mkdir(parents=True)
    _write(directory / 'bundle-manifest.json', bundle['manifest_raw'])
    for role in contract.ROLES:
        _write(directory / FILENAMES[role], bundle['raw_inputs'][role])
    _write(directory / 'review-prompt.txt', prompt)
    _write(directory / 'review-schema.json', schema)
    code = {}
    for index, path in enumerate(_producer_files()):
        raw = _file(path).read_bytes(); snapshot = f'code-{index:02d}-{path.name}'
        _write(directory / snapshot, raw)
        code[str(path)] = {'sha256': sha256(raw), 'snapshot': snapshot}
    preparation = {'version': contract.PREPARATION_VERSION, 'benchmark': contract.BENCHMARK,
        'assessment_scope': contract.ASSESSMENT_SCOPE, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_path': bundle['manifest_path'], 'manifest_sha256': bundle['manifest_sha256'],
        'inputs': bundle['inputs'], 'prompt_sha256': sha256(prompt), 'schema_sha256': sha256(schema),
        'command': _command(directory), 'code': code, 'timeout_seconds': TIMEOUT_SECONDS,
        'source_evidence': 'unverified_asr_possibly_locally_repaired',
        'audio_supplied': False, 'video_supplied': False, 'reviewer_tools_forbidden_by_prompt': True}
    _write(directory / 'preparation.json', preparation)
    return _validate_preparation(directory, manifest, current_code=True)


def _validate_preparation(directory: Path, manifest=None, *, current_code=False) -> dict:
    preparation = _read(directory / 'preparation.json')
    contract.require(preparation.get('version') == contract.PREPARATION_VERSION
                     and preparation.get('benchmark') == contract.BENCHMARK
                     and preparation.get('assessment_scope') == contract.ASSESSMENT_SCOPE,
                     'Wrong subtitle quality preparation contract')
    path = Path(manifest) if manifest is not None else Path(preparation['manifest_path'])
    contract.require(str(path) == preparation['manifest_path'], 'Current manifest path differs')
    bundle = read_bundle(path)
    contract.require(all(bundle[key] == preparation[key] for key in ('manifest_sha256', 'inputs')),
                     'Prepared bundle changed')
    contract.require(_file(directory / 'bundle-manifest.json').read_bytes() == bundle['manifest_raw'],
                     'Manifest snapshot changed')
    for role in contract.ROLES:
        contract.require(_file(directory / FILENAMES[role]).read_bytes() == bundle['raw_inputs'][role],
                         'Input snapshot changed')
    for name, expected, key in [('review-prompt.txt', _prompt(bundle), 'prompt_sha256'),
                              ('review-schema.json', _json_bytes(contract.SCHEMA), 'schema_sha256')]:
        contract.require(_file(directory / name).read_bytes() == expected
                         and preparation[key] == sha256(expected), 'Prompt or schema changed')
    contract.require(preparation.get('command') == _command(directory)
                     and preparation.get('timeout_seconds') == TIMEOUT_SECONDS,
                     'Prepared native dispatch changed')
    contract.require(preparation.get('source_evidence') == 'unverified_asr_possibly_locally_repaired'
                     and preparation.get('audio_supplied') is False
                     and preparation.get('video_supplied') is False
                     and preparation.get('reviewer_tools_forbidden_by_prompt') is True,
                     'Prepared evidence scope changed')
    expected_code = {str(path) for path in _producer_files()}
    contract.require(set(preparation.get('code', {})) == expected_code, 'Missing producing code pins')
    for index, path in enumerate(_producer_files()):
        spec = preparation['code'][str(path)]
        contract.require(isinstance(spec, dict) and set(spec) == {'sha256', 'snapshot'}
                         and spec['snapshot'] == f'code-{index:02d}-{path.name}'
                         and sha256(_file(directory / spec['snapshot']).read_bytes()) == spec['sha256'],
                         'Producing code snapshot changed')
        if current_code:
            contract.require(sha256(_file(path).read_bytes()) == spec['sha256'],
                             'Reviewer code changed before dispatch')
    return preparation


def _identity_headers(path: Path) -> dict:
    """Read native identity headers without exporting the reviewer's reasoning."""
    headers = {}
    with path.open(encoding='utf-8') as stream:
        for _, line in zip(range(30), stream):
            if line.startswith(('model:', 'provider:', 'reasoning effort:')):
                key, value = line.split(':', 1)
                contract.require(key not in headers, 'Duplicate native identity header')
                headers[key] = value.strip()
    return headers


def _unchanged(directory: Path, manifest: Path) -> bool:
    try:
        _validate_preparation(directory, manifest, current_code=True)
        return True
    except (ValueError, OSError, KeyError, TypeError):
        return False


def _coverage(preparation: dict) -> dict:
    return {'supplied_counts': {role: preparation['inputs'][role]['counts'] for role in contract.ROLES},
            'whole_bundle_read': True, 'basis': 'reviewer_whole_text_read_self_report'}


def _reviewer() -> dict:
    return {'model': contract.MODEL, 'provider': 'openai', 'scope': 'evaluation_only',
            'feedback_to_writer': 'scores_only', 'independent_of_writer': True, 'prior_exposure': False}


def review(manifest: str | Path, output_dir: str | Path, *, execute: bool = False) -> dict:
    """Prepare by default; explicitly execute one native attempt with no retries."""
    contract.require(type(execute) is bool, 'execute must be boolean')
    directory = _directory(output_dir); manifest = _file(manifest)
    preparation = prepare_review(manifest, directory)
    if not execute:
        return {'prepared': True, 'executed': False, 'output_dir': str(directory),
                'manifest_sha256': preparation['manifest_sha256'], 'inputs': preparation['inputs']}
    contract.require(not any((directory / name).exists() or (directory / name).is_symlink()
                             for name in ('dispatch-reservation.json',
        'review-run.json', 'review-response.json', 'review-process.log', 'assessment.json')),
        'Existing reviewer attempt cannot be overwritten or restarted')
    prep_sha = sha256(_file(directory / 'preparation.json').read_bytes())
    dispatch_id = str(uuid.uuid4())
    _write(directory / 'dispatch-reservation.json', {'dispatch_id': dispatch_id,
        'preparation_sha256': prep_sha, 'reserved_utc': datetime.now(timezone.utc).isoformat()})
    started = datetime.now(timezone.utc).isoformat(); beginning = time.monotonic()
    run = {'version': contract.RECEIPT_VERSION, 'benchmark': contract.BENCHMARK,
        'assessment_scope': contract.ASSESSMENT_SCOPE, 'dispatch_id': dispatch_id,
        'preparation_sha256': prep_sha,
        **{key: preparation[key] for key in
           ('manifest_sha256', 'inputs', 'prompt_sha256', 'schema_sha256', 'command')},
        'scope': 'evaluation_only', 'feedback_to_writer': 'scores_only',
        'quality_status': 'provisional_text_only', 'started_utc': started,
        'explicit_source_target_context_export_authorized': True,
        **dict.fromkeys(CERTIFICATIONS, False),
        'status': 'failed', 'exit_code': None, 'error_type': None, 'observed_identity_headers': {}}
    response = None
    try:
        with (directory / 'review-process.log').open('x', encoding='utf-8') as log:
            process = subprocess.run(preparation['command'],
                input=_file(directory / 'review-prompt.txt').read_text(encoding='utf-8'),
                text=True, stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUT_SECONDS)
        run['exit_code'] = process.returncode
        run['observed_identity_headers'] = _identity_headers(directory / 'review-process.log')
        contract.require(type(process.returncode) is int and process.returncode == 0,
                         'Native subtitle quality reviewer failed')
        contract.require(run['observed_identity_headers'] == IDENTITY,
                         'Unexpected native subtitle quality reviewer identity')
        response = _read(directory / 'review-response.json'); contract.validate_answer(response)
        contract.require(_unchanged(directory, manifest), 'Frozen review inputs changed during dispatch')
        run['status'] = 'completed'
    except BaseException as error:
        run['error_type'] = type(error).__name__
    finally:
        run['finished_utc'] = datetime.now(timezone.utc).isoformat()
        run['seconds'] = time.monotonic() - beginning
        if (directory / 'review-process.log').is_file():
            try:
                run['observed_identity_headers'] = _identity_headers(
                    _file(directory / 'review-process.log'))
            except (ValueError, OSError):
                run.update(status='failed', error_type='InvalidNativeIdentityLog')
        run['frozen_inputs_unchanged'] = _unchanged(directory, manifest)
        if not run['frozen_inputs_unchanged']:
            run.update(status='failed', error_type='FrozenInputChanged')
        for key, name in [('response_sha256', 'review-response.json'),
                          ('process_log_sha256', 'review-process.log')]:
            if (directory / name).is_file():
                run[key] = sha256(_file(directory / name).read_bytes())
        _write(directory / 'review-run.json', run)
    if run['status'] != 'completed':
        raise RuntimeError('Subtitle quality review failed; attempt receipt preserved')
    assessment = {key: run[key] for key in ('version', 'benchmark', 'assessment_scope',
        'dispatch_id', 'preparation_sha256', 'manifest_sha256', 'inputs', 'quality_status', *CERTIFICATIONS)}
    assessment.update(answer=response, reviewer=_reviewer(), coverage=_coverage(preparation),
                      reviewed_date=datetime.fromisoformat(run['finished_utc']).date().isoformat())
    _write(directory / 'assessment.json', assessment)
    return validate_receipt(directory, manifest)


def _instant(value: str) -> datetime:
    contract.require(isinstance(value, str), 'Missing review timestamp')
    result = datetime.fromisoformat(value)
    contract.require(result.tzinfo is not None and result.utcoffset() is not None, 'Naive review timestamp')
    return result


def validate_receipt(output_dir: str | Path, manifest: str | Path | None = None) -> dict:
    """Check a completed saved review against native output and current inputs."""
    directory = _directory(output_dir)
    preparation = _validate_preparation(directory, manifest)
    prep_sha = sha256(_file(directory / 'preparation.json').read_bytes())
    run = _read(directory / 'review-run.json'); assessment = _read(directory / 'assessment.json')
    response = _read(directory / 'review-response.json'); contract.validate_answer(response)
    reservation = _read(directory / 'dispatch-reservation.json')
    contract.require(run.get('status') == 'completed' and run.get('error_type') is None
                     and type(run.get('exit_code')) is int and run['exit_code'] == 0
                     and run.get('frozen_inputs_unchanged') is True, 'Review did not complete unchanged')
    contract.require(str(uuid.UUID(run['dispatch_id'])) == run['dispatch_id'], 'Invalid dispatch UUID')
    for item in (run, assessment):
        contract.require(item.get('version') == contract.RECEIPT_VERSION
                         and item.get('benchmark') == contract.BENCHMARK
                         and item.get('assessment_scope') == contract.ASSESSMENT_SCOPE
                         and item.get('dispatch_id') == run['dispatch_id']
                         and item.get('preparation_sha256') == prep_sha
                         and all(item.get(key) == preparation[key] for key in ('manifest_sha256', 'inputs')),
                         'Receipt protocol or input binding changed')
        contract.require(item.get('quality_status') == 'provisional_text_only'
                         and all(item.get(key) is False for key in CERTIFICATIONS),
                         'Text quality receipt cannot certify audio, playback, or release')
    contract.require(run.get('scope') == 'evaluation_only' and run.get('feedback_to_writer') == 'scores_only'
                     and run.get('explicit_source_target_context_export_authorized') is True
                     and all(run.get(key) == preparation[key] for key in
                             ('prompt_sha256', 'schema_sha256', 'command')),
                     'Dispatched prompt, schema or scope changed')
    contract.require(reservation.get('dispatch_id') == run['dispatch_id']
                     and reservation.get('preparation_sha256') == prep_sha, 'Reservation binding changed')
    contract.require(_identity_headers(_file(directory / 'review-process.log')) == IDENTITY
                     and run.get('observed_identity_headers') == IDENTITY, 'Native reviewer identity changed')
    for key, name in [('response_sha256', 'review-response.json'),
                      ('process_log_sha256', 'review-process.log')]:
        contract.require(run.get(key) == sha256(_file(directory / name).read_bytes()),
                         'Native response or log changed')
    contract.require(assessment.get('answer') == response and assessment.get('reviewer') == _reviewer()
                     and assessment.get('coverage') == _coverage(preparation),
                     'Scores, reviewer role or supplied coverage changed')
    reserved = _instant(reservation.get('reserved_utc')); started = _instant(run.get('started_utc'))
    finished = _instant(run.get('finished_utc')); seconds = run.get('seconds')
    contract.require(reserved <= started <= finished
                     and assessment.get('reviewed_date') == finished.date().isoformat()
                     and type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0
                     and abs((finished - started).total_seconds() - seconds) < 10, 'Invalid review time/date')
    filenames = ('preparation.json', 'dispatch-reservation.json', 'review-run.json', 'review-response.json',
                 'review-process.log', 'assessment.json', 'review-prompt.txt', 'review-schema.json',
                 'bundle-manifest.json')
    return {**assessment, **response, 'started_utc': run['started_utc'], 'finished_utc': run['finished_utc'],
            'seconds': seconds, 'output_dir': str(directory),
            'receipt_hashes': {name: sha256(_file(directory / name).read_bytes()) for name in filenames}}
