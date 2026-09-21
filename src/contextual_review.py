"""Hash-bound contextual score-only review. Preparation and validation are local.

Only explicit execute=True exports the three authorized text inputs through the
installed ephemeral reviewer CLI. No audio/video, repair, retry or tool request.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import uuid

from src import contextual_review_contract as contract
from src.release import _read_release_srt

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_SECONDS = 600
FILENAMES = {'source': 'input-source.srt', 'target': 'input-target.srt', 'context': 'input-context.txt'}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file(value: str | Path) -> Path:
    path = Path(value)
    contract.require(path.is_absolute() and '..' not in path.parts and path.is_file()
                     and not any(part.is_symlink() for part in (path, *path.parents)),
                     'Input must be a plain absolute file without symlink components')
    return path


def _read(path: str | Path) -> dict:
    return contract.strict_json(_file(path).read_bytes())


def _write(path: Path, value: bytes | dict) -> None:
    raw = value if isinstance(value, bytes) else (json.dumps(value, ensure_ascii=False, indent=2,
                                                            allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as stream:
        stream.write(raw)


def _bound_bytes(spec: dict) -> bytes:
    contract.require(isinstance(spec, dict) and set(spec) == {'path', 'sha256'}
                     and isinstance(spec['path'], str) and isinstance(spec['sha256'], str)
                     and re.fullmatch(r'[0-9a-f]{64}', spec['sha256']) is not None,
                     'Invalid manifest file binding')
    raw = _file(spec['path']).read_bytes()
    contract.require(sha256(raw) == spec['sha256'], 'Manifest input hash changed')
    return raw


def read_bundle(manifest: str | Path) -> dict:
    """Return private exact texts and numeric provenance; never print this object."""
    manifest = _file(manifest); raw_manifest = manifest.read_bytes()
    value = contract.strict_json(raw_manifest)
    contract.require(isinstance(value, dict) and set(value) == {'version', *contract.ROLES}
                     and value['version'] == contract.MANIFEST_VERSION, 'Invalid contextual bundle manifest')
    inputs = {}; texts = {}; raw_inputs = {}; geometry = {}
    for role in contract.ROLES:
        raw = _bound_bytes(value[role]); text = raw.decode('utf-8', errors='strict')
        counts = {'bytes': len(raw), 'characters': len(text)}
        if role != 'context':
            rows = _read_release_srt(Path(value[role]['path']))
            contract.require(not any(re.fullmatch(r'\s*\d{2}:\d{2}:\d{2},\d{3}\s*-->.*', line)
                                     for row in rows for line in row.text.splitlines()),
                             'Embedded cue timestamp in contextual input')
            counts['cues'] = len(rows)
            geometry[role] = [(row.index, row.ts_line) for row in rows]
        contract.require(_file(value[role]['path']).read_bytes() == raw, 'Input changed while counting')
        inputs[role] = {**value[role], 'counts': counts}; texts[role] = text; raw_inputs[role] = raw
    contract.require(geometry['source'] == geometry['target'], 'Source and target cue ownership differ')
    contract.require(manifest.read_bytes() == raw_manifest, 'Manifest changed while preparing')
    return {'manifest_path': str(manifest), 'manifest_sha256': sha256(raw_manifest),
            'manifest_raw': raw_manifest, 'inputs': inputs, 'texts': texts, 'raw_inputs': raw_inputs}


def _command(directory: Path) -> list[str]:
    return ['codex', 'exec', '--model', contract.MODEL, '-c', 'model_reasoning_effort="high"',
            '--ephemeral', '--sandbox', 'read-only', '--skip-git-repo-check', '--cd', '/tmp',
            '--color', 'never', '--output-schema', str(directory / 'review-schema.json'),
            '--output-last-message', str(directory / 'review-response.json'), '-']


def _directory(path: str | Path) -> Path:
    result = Path(path).absolute()
    contract.require('..' not in result.parts
                     and not any(part.is_symlink() for part in (result, *result.parents)),
                     'Review directory cannot use symlinks')
    return result


def prepare_review(manifest: str | Path, output_dir: str | Path) -> dict:
    """CPU-only snapshot; an existing complete preparation is validated read-only."""
    directory = _directory(output_dir)
    if directory.exists():
        return _validate_preparation(directory, manifest, current_code=True)
    bundle = read_bundle(manifest)
    prompt = contract.build_prompt(**bundle['texts']).encode('utf-8')
    schema = (json.dumps(contract.SCHEMA, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    directory.mkdir(parents=True)
    _write(directory/'bundle-manifest.json', bundle['manifest_raw'])
    for role in contract.ROLES:
        _write(directory/FILENAMES[role], bundle['raw_inputs'][role])
    _write(directory/'review-prompt.txt', prompt); _write(directory/'review-schema.json', schema)
    code = {}
    for index, path in enumerate((Path(__file__), Path(contract.__file__), ROOT/'scripts/review_contextual.py',
                                  ROOT/'src/release.py', ROOT/'src/translate.py')):
        raw = path.read_bytes(); snapshot = f'code-{index:02d}-{path.name}'
        _write(directory/snapshot, raw)
        code[str(path)] = {'sha256': sha256(raw), 'snapshot': snapshot}
    preparation = {'version': contract.PREPARATION_VERSION, 'benchmark': contract.BENCHMARK,
        'assessment_scope': contract.ASSESSMENT_SCOPE, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_path': bundle['manifest_path'], 'manifest_sha256': bundle['manifest_sha256'],
        'inputs': bundle['inputs'], 'prompt_sha256': sha256(prompt), 'schema_sha256': sha256(schema),
        'command': _command(directory), 'code': code, 'timeout_seconds': TIMEOUT_SECONDS,
        'source_evidence': 'unverified_asr_possibly_locally_repaired', 'audio_supplied': False,
        'video_supplied': False, 'reviewer_tools_forbidden_by_prompt': True}
    _write(directory/'preparation.json', preparation)
    return _validate_preparation(directory, manifest, current_code=True)


def _validate_preparation(directory: Path, manifest=None, *, current_code=False) -> dict:
    preparation = _read(directory/'preparation.json')
    contract.require(preparation.get('version') == contract.PREPARATION_VERSION
                     and preparation.get('benchmark') == contract.BENCHMARK
                     and preparation.get('assessment_scope') == contract.ASSESSMENT_SCOPE,
                     'Wrong contextual preparation contract')
    manifest = Path(manifest) if manifest is not None else Path(preparation['manifest_path'])
    contract.require(str(manifest) == preparation['manifest_path'], 'Current manifest path differs')
    bundle = read_bundle(manifest)
    contract.require(bundle['manifest_sha256'] == preparation['manifest_sha256']
                     and bundle['inputs'] == preparation['inputs'], 'Prepared bundle changed')
    contract.require(_file(directory/'bundle-manifest.json').read_bytes() == bundle['manifest_raw'],
                     'Manifest snapshot changed')
    for role in contract.ROLES:
        contract.require(_file(directory/FILENAMES[role]).read_bytes() == bundle['raw_inputs'][role],
                         'Contextual input snapshot changed')
    prompt = _file(directory/'review-prompt.txt').read_bytes()
    schema = _file(directory/'review-schema.json').read_bytes()
    contract.require(prompt == contract.build_prompt(**bundle['texts']).encode('utf-8')
                     and sha256(prompt) == preparation['prompt_sha256'], 'Contextual prompt changed')
    contract.require(contract.strict_json(schema) == contract.SCHEMA
                     and sha256(schema) == preparation['schema_sha256'], 'Contextual schema changed')
    contract.require(preparation.get('command') == _command(directory)
                     and preparation.get('timeout_seconds') == TIMEOUT_SECONDS,
                     'Prepared native dispatch changed')
    expected_code = {str(path) for path in (Path(__file__), Path(contract.__file__),
                     ROOT/'scripts/review_contextual.py', ROOT/'src/release.py', ROOT/'src/translate.py')}
    contract.require(set(preparation.get('code', {})) == expected_code, 'Missing producing code pins')
    for path, spec in preparation['code'].items():
        snapshot = spec['snapshot']
        contract.require(isinstance(snapshot, str) and Path(snapshot).name == snapshot
                         and sha256(_file(directory/snapshot).read_bytes()) == spec['sha256'],
                         'Producing code snapshot changed')
        if current_code:
            contract.require(sha256(_file(path).read_bytes()) == spec['sha256'], 'Reviewer code changed before dispatch')
    return preparation


def _identity_headers(path: Path) -> dict:
    """Inspect header keys only; reviewer reasoning is never returned or evaluated."""
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


def review(manifest: str | Path, output_dir: str | Path, *, execute: bool = False) -> dict:
    directory = _directory(output_dir); manifest = _file(manifest)
    preparation = prepare_review(manifest, directory)
    if not execute:
        return {'prepared': True, 'executed': False, 'output_dir': str(directory),
                'manifest_sha256': preparation['manifest_sha256'], 'inputs': preparation['inputs']}
    contract.require(not any((directory/name).exists() for name in ('dispatch-reservation.json',
        'review-run.json', 'review-response.json', 'review-process.log', 'assessment.json')),
        'Existing reviewer attempt cannot be overwritten or restarted')
    prep_sha = sha256(_file(directory/'preparation.json').read_bytes())
    dispatch_id = str(uuid.uuid4())
    _write(directory/'dispatch-reservation.json', {'dispatch_id': dispatch_id,
        'preparation_sha256': prep_sha, 'reserved_utc': datetime.now(timezone.utc).isoformat()})
    started = datetime.now(timezone.utc).isoformat(); beginning = time.monotonic()
    run = {'version': contract.RECEIPT_VERSION, 'benchmark': contract.BENCHMARK,
        'assessment_scope': contract.ASSESSMENT_SCOPE, 'dispatch_id': dispatch_id,
        'preparation_sha256': prep_sha, 'manifest_sha256': preparation['manifest_sha256'],
        'inputs': preparation['inputs'], 'prompt_sha256': preparation['prompt_sha256'],
        'schema_sha256': preparation['schema_sha256'], 'command': preparation['command'],
        'scope': 'evaluation_only', 'feedback_to_writer': 'score_only', 'started_utc': started,
        'explicit_source_target_context_export_authorized': True,
        'status': 'failed', 'exit_code': None, 'error_type': None, 'observed_identity_headers': {}}
    response = None
    try:
        with (directory/'review-process.log').open('x', encoding='utf-8') as log:
            process = subprocess.run(preparation['command'],
                input=_file(directory/'review-prompt.txt').read_text(encoding='utf-8'),
                text=True, stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUT_SECONDS)
        run['exit_code'] = process.returncode
        run['observed_identity_headers'] = _identity_headers(directory/'review-process.log')
        contract.require(process.returncode == 0, 'Native contextual reviewer failed')
        contract.require(run['observed_identity_headers'] == {'model': contract.MODEL, 'provider': 'openai',
                          'reasoning effort': 'high'}, 'Unexpected native contextual reviewer identity')
        response = _read(directory/'review-response.json'); contract.validate_answer(response)
        contract.require(_unchanged(directory, manifest), 'Contextual inputs changed during review')
        run['status'] = 'completed'
    except BaseException as error:
        run['error_type'] = type(error).__name__
    finally:
        run['finished_utc'] = datetime.now(timezone.utc).isoformat()
        run['seconds'] = time.monotonic()-beginning
        run['frozen_inputs_unchanged'] = _unchanged(directory, manifest)
        if not run['frozen_inputs_unchanged']:
            run['status'] = 'failed'; run['error_type'] = 'FrozenInputChanged'
        for key, name in (('response_sha256','review-response.json'),('process_log_sha256','review-process.log')):
            if (directory/name).is_file():
                run[key] = sha256(_file(directory/name).read_bytes())
        _write(directory/'review-run.json', run)
    if run['status'] != 'completed':
        raise RuntimeError('Contextual reviewer attempt failed; numeric receipt preserved')
    assessment = {'version': contract.RECEIPT_VERSION, 'benchmark': contract.BENCHMARK,
        'assessment_scope': contract.ASSESSMENT_SCOPE, 'dispatch_id': dispatch_id,
        'preparation_sha256': prep_sha, 'manifest_sha256': preparation['manifest_sha256'],
        'inputs': preparation['inputs'], 'answer': response,
        'reviewer': {'model': contract.MODEL, 'provider': 'openai', 'scope': 'evaluation_only',
            'feedback_to_writer': 'score_only', 'independent_of_writer': True, 'prior_exposure': False},
        'reviewed_date': datetime.fromisoformat(run['finished_utc']).date().isoformat(),
        'coverage': {'supplied_counts': {role:preparation['inputs'][role]['counts'] for role in contract.ROLES},
                     'whole_bundle_read': True, 'basis': 'reviewer_whole_text_read_self_report'},
        'audio_reviewed': False, 'video_reviewed': False, 'audio_source_fidelity_certified': False,
        'release_gate_checked': False}
    contract.validate_receipt_payloads(assessment, run, response, preparation, prep_sha)
    _write(directory/'assessment.json', assessment)
    return validate_receipt(directory, manifest)


def validate_receipt(output_dir: str | Path, manifest: str | Path | None = None) -> dict:
    """Read-only current-input and immutable-receipt validation for release callers."""
    directory = _directory(output_dir)
    preparation = _validate_preparation(directory, manifest)
    prep_sha = sha256(_file(directory/'preparation.json').read_bytes())
    assessment = _read(directory/'assessment.json'); run = _read(directory/'review-run.json')
    response = _read(directory/'review-response.json'); reservation = _read(directory/'dispatch-reservation.json')
    contract.require(reservation.get('dispatch_id') == run.get('dispatch_id')
                     and reservation.get('preparation_sha256') == prep_sha,
                     'Contextual reservation binding mismatch')
    reserved = datetime.fromisoformat(reservation.get('reserved_utc', ''))
    started = datetime.fromisoformat(run.get('started_utc', ''))
    contract.require(reserved.tzinfo is not None and started.tzinfo is not None and reserved <= started,
                     'Invalid contextual reservation timestamp')
    contract.require(_identity_headers(_file(directory/'review-process.log')) == run.get('observed_identity_headers'),
                     'Saved native identity headers differ from run receipt')
    for key, name in (('response_sha256','review-response.json'),('process_log_sha256','review-process.log')):
        contract.require(run.get(key) == sha256(_file(directory/name).read_bytes()), 'Native receipt file hash changed')
    contract.validate_receipt_payloads(assessment, run, response, preparation, prep_sha)
    return {**assessment, **response, 'started_utc': run['started_utc'], 'finished_utc': run['finished_utc'],
            'seconds': run['seconds'], 'output_dir': str(directory),
            'receipt_hashes': {name:sha256(_file(directory/name).read_bytes()) for name in
                ('preparation.json','dispatch-reservation.json','review-run.json','review-response.json',
                 'assessment.json','review-prompt.txt','review-schema.json','bundle-manifest.json')}}
