"""Run the provisional whole-source local subtitle workflow on explicit media.

CPU preflight is the default. --execute creates a fresh output directory and
runs native Anime Whisper, one whole-source Gemma request, then the existing
exact-text display formatter. No benchmark paths, model downloads, automatic
context discovery, external review, or release approval belong to this runner.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit

from src import selected_asr
from src.audio import extract_audio
from src.coherence import _json, _parse_response
from src.coherence_workflow import load_recipe
from src.config import TranslateConfig
from src.display import build_display
from src.local_backend import _request_json, _stop_owned_process, _weight_bundle, temporary_local_writer
from src.quality import seconds, validate_blocks
from src.translate import SrtBlock, _build_payload, call_llm, parse_srt, write_translated_srt
from src.workflow_state import file_hash, fingerprint, write_json

VERSION = 'selected-whole-source-pipeline-1'
MODEL = 'gemma4-31b-qat-q4'
WEIGHT_SHA256 = '179cfb99212709597eae5929112cfca677e1bbf566178b479ae1da0c4772874b'
SAMPLER = {'temperature':1, 'top_p':.95, 'top_k':64, 'repeat_penalty':1, 'seed':20260913}
CONTEXT_SIZE = 32768
MAX_TOKENS = 16384
REASONING_BUDGET = 1536
DEFAULT_REQUEST_TIMEOUT = 600



INSTRUCTION = '''请先通读提供的完整日语对白及作品背景，再把每个编号的日语原文译成自然、准确的简体中文字幕。利用全片日语上下文理解说话人的意图、指代、语气与承接，保持人物称呼和术语一致。使用自然的中文口语和语序，避免逐词硬译。
忠实保留每条原文的实际含义、否定、疑问、动作主体与对象、数字和人物关系。原文来自未经完全核实的语音识别；不把猜测写成事实，不编造情节或擅自补出缺失的台词。没有充分依据的省略或歧义应保留。短答、感叹、歌词和语句片段保持简洁，不强行补成完整句。
每个编号只承载该编号原文的内容。不得跨编号挪动、合并、遗漏或重复对白，不添加说话人标签。不存在可供照抄的中文草稿。所有中文都必须依据本次日语原文和背景生成。
只输出一个完整JSON对象，键为全部原文编号的字符串，值为该编号的完整中文译文。所有编号必须恰好出现一次，值必须非空。不要输出Markdown、说明、时间戳、审校意见或额外编号。'''


def validate_request_timeout(value: int) -> int:
    if type(value) is not int or not 1 <= value <= 3600:
        raise ValueError('request timeout must be an integer from 1 through 3600 seconds')
    return value


def configuration(endpoint: str, recipe: dict, telemetry: Path | None = None, *,
                  request_timeout: int = DEFAULT_REQUEST_TIMEOUT) -> TranslateConfig:
    return TranslateConfig(endpoint=endpoint, max_tokens=MAX_TOKENS,
                           timeout=validate_request_timeout(request_timeout), retries=0,
                           separate_instruction=True, response_guard_floor=16384,
                           telemetry_path=telemetry, stage='whole_source_translation',
                           extra_payload={**recipe['sampler'], 'model': recipe['model'], 'max_tokens': MAX_TOKENS,
                                          'cache_prompt': True, 'reasoning_budget_tokens': REASONING_BUDGET,
                                          'chat_template_kwargs': {'enable_thinking': True}})


def capacity_check(body: str, instruction: str, config: TranslateConfig, identity: dict) -> dict:
    endpoint = urlsplit(config.endpoint)
    if endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1' or endpoint.username or endpoint.query or endpoint.fragment:
        raise ValueError('Native capacity check requires the owned direct loopback backend')
    observed = identity.get('default_generation_settings', {}).get('n_ctx')
    if type(observed) is not int or observed < CONTEXT_SIZE:
        raise RuntimeError('Backend did not confirm the requested 32K context')
    payload = _build_payload(body, instruction, config, {}, MAX_TOKENS, stream=True, with_thinking=True)
    rendered = _request_json(endpoint._replace(path='/apply-template').geturl(), payload, timeout=30)
    prompt = rendered.get('prompt') if isinstance(rendered, dict) else None
    if not isinstance(prompt, str) or not prompt:
        raise RuntimeError('Native chat template returned no complete prompt')
    response = _request_json(endpoint._replace(path='/tokenize').geturl(),
                             {'content': prompt, 'add_special': True, 'parse_special': True}, timeout=30)
    tokens = response.get('tokens') if isinstance(response, dict) else None
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
        raise RuntimeError('Native tokenizer returned invalid token IDs')
    return {'request_sha256': fingerprint(payload), 'rendered_prompt_sha256': fingerprint(prompt),
            'prompt_tokens': len(tokens), 'reserved_completion_tokens': MAX_TOKENS,
            'reasoning_budget_included': REASONING_BUDGET, 'context_limit': CONTEXT_SIZE,
            'backend_reported_context': observed, 'safety_margin_tokens': 64,
            'fits': len(tokens) + MAX_TOKENS + 64 <= CONTEXT_SIZE,
            'native_template_used': True, 'full_context_preserved': True, 'generation_performed': False}


def parse_translation(raw: str, source: list[SrtBlock]) -> tuple[list[SrtBlock], dict]:
    parsed = _json(raw)
    if not isinstance(parsed, dict) or any(not isinstance(value, str) for value in parsed.values()):
        raise ValueError('Whole-source response must be a JSON object of strings')
    clean, changed = {}, []
    for key, value in parsed.items():
        normalized = re.sub(r'\n(?: *\n)+', '\n', value)
        if ''.join(c for c in value if not c.isspace()) != ''.join(c for c in normalized if not c.isspace()):
            raise RuntimeError('Blank-line normalization changed non-whitespace characters')
        clean[key] = normalized
        if normalized != value:
            changed.append(key)
    canonical = json.dumps(clean, ensure_ascii=False)
    values = _parse_response(canonical, source)
    final = [replace(row, text=value) for row, value in zip(source, values)]
    validate_blocks(final)
    return final, {'policy': 'json-blank-lines-only-1', 'changed_local_ids': changed,
                   'non_whitespace_unchanged': True, 'canonical_response_sha256': fingerprint(canonical)}


def write_checked(rows: list[SrtBlock], path: Path) -> None:
    write_translated_srt(rows, path)
    if parse_srt(path, preserve_text_whitespace=True) != rows:
        raise RuntimeError('SRT round trip changed text, IDs or timestamps')


def exception_text(exc: BaseException) -> str:
    result = f'{type(exc).__name__}: {exc}'
    notes = getattr(exc, '__notes__', [])
    return result + ('; ' + '; '.join(notes) if notes else '')


@dataclass(frozen=True)
class PipelineConfig:
    media: Path
    output_dir: Path
    context_file: Path
    asr_model_dir: Path
    writer_recipe: Path
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT
    asr_timeout: int = 1200


def _path_pins() -> dict[str, str]:
    names = ('selected_pipeline.py', 'selected_asr.py', 'audio.py', 'display.py', 'pause_layout.py',
             'translate.py', 'coherence.py', 'coherence_workflow.py', 'local_backend.py',
             'quality.py', 'config.py', 'workflow_state.py', 'asr_consensus.py')
    return {str(Path(__file__).with_name(name).resolve()): file_hash(Path(__file__).with_name(name))
            for name in names}


def load_writer_recipe(path: Path) -> dict:
    """Accept a backend-only profile; old complete recipes remain readable."""
    path = Path(path).expanduser().resolve(strict=True)
    data = _json(path.read_text(encoding='utf-8'))
    if isinstance(data, dict) and 'stages' in data:
        return load_recipe(path)
    required = {'version','model','weights','sha256','server_binary','sampler'}
    allowed = required | {'context_size','cpu_moe_layers','gpu_layers','threads','full_swa',
                          'admin_url','restore_model'}
    if (not isinstance(data,dict) or not required <= set(data) or set(data)-allowed
            or type(data.get('version')) is not int or data['version'] != 1):
        raise ValueError('Invalid selected backend recipe fields')
    for key, default in {'context_size':32768,'cpu_moe_layers':0,'gpu_layers':99,'threads':4}.items():
        value=data.setdefault(key,default)
        if type(value) is not int or value < (1 if key in {'context_size','threads'} else 0):
            raise ValueError('Invalid selected backend integer: '+key)
    if type(data.setdefault('full_swa',False)) is not bool:
        raise ValueError('Invalid selected backend full_swa')
    for key,default in {'admin_url':'http://127.0.0.1:8089/admin','restore_model':'qwen3.8-27b-dflash'}.items():
        if not isinstance(data.setdefault(key,default),str):
            raise ValueError('Invalid selected backend identity: '+key)
    if not isinstance(data['model'],str) or data['model'] != MODEL:
        raise ValueError('Selected backend must retain the pinned Gemma identity')
    if (not isinstance(data['sha256'],str) or not re.fullmatch('[a-fA-F0-9]{64}',data['sha256'])):
        raise ValueError('Invalid selected backend weight SHA256')
    data['sha256']=data['sha256'].lower()
    sampler=data['sampler']
    if (not isinstance(sampler,dict) or set(sampler)!=set(SAMPLER)
            or any(type(value) not in (int,float) or not math.isfinite(value) for value in sampler.values())
            or any(type(sampler[key]) is not int for key in ('top_k','seed'))):
        raise ValueError('Invalid selected backend sampler')
    if sampler != SAMPLER:
        raise ValueError('Selected backend sampler differs')
    for key in ('weights','server_binary'):
        if not isinstance(data[key],str) or not data[key]:
            raise ValueError('Invalid selected backend path: '+key)
        local=Path(data[key]).expanduser()
        local=(local if local.is_absolute() else path.parent/local).resolve(strict=True)
        if not local.is_file() or (key=='server_binary' and not os.access(local,os.X_OK)):
            raise ValueError('Missing selected backend file: '+key)
        data[key]=str(local)
    _weight_bundle(Path(data['weights']),data['sha256'],None)
    return data


def prepare(config: PipelineConfig) -> dict:
    """Read-only CPU readiness and immutable input identities; no model imports."""
    validate_request_timeout(config.request_timeout)
    if type(config.asr_timeout) is not int or not 1 <= config.asr_timeout <= 3600:
        raise ValueError('ASR timeout must be an integer1..3600 seconds')
    media = config.media.expanduser().resolve(strict=True)
    context_path = config.context_file.expanduser().resolve(strict=True)
    recipe_path = config.writer_recipe.expanduser().resolve(strict=True)
    destination = config.output_dir.expanduser().resolve()
    if destination.exists():
        raise ValueError('Output directory already exists; preserve it and choose a fresh destination')
    if not all(path.is_file() for path in (media, context_path, recipe_path)):
        raise ValueError('Media, original context and writer recipe must be existing files')
    if shutil.which('ffmpeg') is None:
        raise ValueError('ffmpeg is not available')
    background = context_path.read_bytes().decode('utf-8')
    if not background.strip():
        raise ValueError('Provide the complete nonempty original context explicitly')
    recipe = load_writer_recipe(recipe_path)
    if (recipe['model'] != MODEL or recipe.get('sha256') != WEIGHT_SHA256
            or recipe['sampler'] != SAMPLER or recipe['context_size'] != CONTEXT_SIZE
            or recipe['cpu_moe_layers'] != 0 or recipe['threads'] != 4 or recipe['full_swa'] is not False
            or recipe.get('gpu_layers',99) != 99):
        raise ValueError('Selected whole-source writer recipe differs from the frozen baseline')
    if (recipe['restore_model'] != 'qwen3.8-27b-dflash'
            or recipe['admin_url'] != 'http://127.0.0.1:8089/admin'):
        raise ValueError('The exact baseline formatter requires the Qwen DFlash restore profile')
    identity = selected_asr.bundle_identity(config.asr_model_dir)
    paths = [media, context_path, recipe_path, Path(recipe['server_binary']), Path(sys.executable).resolve()]
    pins = {str(path): file_hash(path) for path in paths}
    pins.update(_path_pins()); pins.update(identity['input_hashes'])
    pins[str(Path(recipe['weights']))] = recipe['sha256']
    return {'version': VERSION, 'status': 'prepared_no_inference', 'media': str(media),
            'output_dir': str(destination), 'context_file': str(context_path),
            'writer_recipe': str(recipe_path), 'recipe': recipe, 'asr_identity': identity,
            'input_hashes': pins, 'code_hashes': _path_pins(),
            'context_characters': len(background), 'source_uncertainty_cleared': False,
            'generation_performed': False, 'external_review_performed': False,
            'native_writer_capacity_checked': False,
            'native_writer_capacity_deferred_until_source_and_owned_backend': True,
            'asr_batch_size': 8, 'asr_maximum_attempts_per_window': 3,
            'asr_attempt_policy': 'greedy_first_then_seeded_sampling_only_for_structurally_rejected_windows',
            'logical_whole_source_writer_requests': 1, 'writer_configured_retries': 0,
            'shared_client_same_attempt_http400_fallback_preserved': True,
            'display_configured_retries': 2, 'display_max_tokens_per_attempt': 2048,
            'context_size': CONTEXT_SIZE, 'max_tokens': MAX_TOKENS,
            'reasoning_budget_tokens': REASONING_BUDGET,
            'request_timeout_seconds': config.request_timeout, 'asr_timeout_seconds': config.asr_timeout,
            'request_timeout_semantics': 'socket_read_inactivity'}


def _unchanged(pins: dict[str, str]) -> bool:
    try:
        return all(file_hash(Path(path)) == digest for path, digest in pins.items())
    except OSError:
        return False


def _snapshot(preparation: dict, output: Path) -> None:
    folder = output/'measured-code'
    folder.mkdir()
    for name in preparation['code_hashes']:
        shutil.copy2(name, folder/Path(name).name)
    shutil.copy2(preparation['writer_recipe'], output/'writer-recipe.json')
    shutil.copy2(preparation['context_file'], output/'original-context.txt')


def _confirm_original_backend(recipe: dict) -> None:
    status = _request_json(recipe['admin_url'].rstrip('/')+'/status', admin=True, timeout=10)
    if status.get('loaded_model') != recipe['restore_model']:
        raise RuntimeError('Required original backend is not confirmed active; nothing was unloaded')


def _run_asr(plan: dict, preparation: dict, output: Path, *, timeout: int) -> dict:
    """Own only this worker; restore only after its termination is confirmed."""
    folder = output/'asr'; folder.mkdir()
    identity = preparation['asr_identity']; recipe = preparation['recipe']
    write_json(folder/'backend.json', identity)
    write_json(folder/'windows.json', plan)
    spec = {'version': selected_asr.VERSION, 'plan_path': str(folder/'windows.json'),
            'plan_sha256': file_hash(folder/'windows.json'), 'output_dir': str(folder),
            'model_dir': identity['model_dir'], 'bundle_identity_sha256': selected_asr.json_hash(identity),
            'runtime_identity_sha256': file_hash(folder/'backend.json'),
            'worker_sha256': file_hash(Path(selected_asr.__file__)), 'batch_size':8,
            'max_new_tokens':440, 'seed':20260914, 'max_attempts':3,
            'repetition_guard':dict(selected_asr.REPETITION_POLICY)}
    write_json(folder/'worker-spec.json', spec)
    selected_asr.validate_spec(spec)
    started = time.monotonic(); process = None; unloaded = False; stopped = True
    restored = None; error = None; result = None
    try:
        _confirm_original_backend(recipe)
        unloaded = True
        _request_json(recipe['admin_url'].rstrip('/')+'/unload', {}, admin=True, timeout=45)
        with (folder/'worker.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-m', 'src.selected_asr', '--spec',
                                        str(folder/'worker-spec.json')],
                                       cwd=Path(__file__).resolve().parents[1], stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait(timeout=timeout)
        result = _json((folder/'worker-result.json').read_text(encoding='utf-8'))
        if code != 0 or result.get('complete') is not True or result.get('error') is not None:
            raise RuntimeError('Native ASR did not complete every window')
    except BaseException as exc:
        error = exc
    finally:
        if process is not None:
            try:
                _stop_owned_process(process)
            except BaseException as exc:
                stopped = False
                if error is None: error = exc
                else: error.add_note('Owned ASR worker shutdown failed: '+exception_text(exc))
        if unloaded and stopped:
            try:
                restored = _request_json(recipe['admin_url'].rstrip('/')+'/load',
                    {'model':recipe['restore_model']}, admin=True, timeout=180).get('loaded') == recipe['restore_model']
                if not restored: raise RuntimeError('Original local backend restoration was not confirmed')
            except BaseException as exc:
                restored = False
                if error is None: error = exc
                else: error.add_note('Backend restoration failed: '+exception_text(exc))
        write_json(folder/'lifecycle.json', {'wall_seconds':time.monotonic()-started,
            'owned_worker_stopped':stopped, 'original_backend_restored':restored,
            'error':exception_text(error) if error is not None else None})
    if error is not None: raise error
    return result


def _translation_inputs(source: list[SrtBlock], background: str) -> tuple[str, str, dict]:
    validate_blocks(source)
    instruction = INSTRUCTION + '\n完整原始作品背景（只读）：\n' + background
    body = json.dumps({str(row.index): row.text for row in source}, ensure_ascii=False)
    schema = {'type':'object', 'properties': {
        str(row.index): {'type':'string', 'minLength':1, 'maxLength':max(160,4*len(row.text))}
        for row in source}, 'required':[str(row.index) for row in source], 'additionalProperties':False}
    return body, instruction, schema


def _translate(source: list[SrtBlock], preparation: dict, output: Path) -> list[SrtBlock]:
    recipe = preparation['recipe']
    background = Path(preparation['context_file']).read_bytes().decode('utf-8')
    body, instruction, schema = _translation_inputs(source, background)
    result = None; body_error = None; restored = None; semantic_saved = False
    started = time.monotonic()
    try:
        _confirm_original_backend(recipe)
        with temporary_local_writer(recipe['weights'], recipe['server_binary'], alias=recipe['model'],
                expected_sha256=recipe['sha256'], expected_shards=recipe.get('shard_sha256'),
                context_size=CONTEXT_SIZE, cpu_moe_layers=recipe['cpu_moe_layers'],
                gpu_layers=recipe.get('gpu_layers',99), threads=recipe['threads'], full_swa=recipe['full_swa'],
                admin_url=recipe['admin_url'], restore_model=recipe['restore_model'],
                log_path=output/'writer-server.log') as (endpoint, identity):
            write_json(output/'writer-backend.json', identity)
            cfg = configuration(endpoint, recipe, output/'writer-metrics.jsonl',
                                request_timeout=preparation['request_timeout_seconds'])
            cfg.extra_payload['response_format'] = {'type':'json_object', 'schema':schema}
            request = {'body':body, 'instruction':instruction, 'settings':cfg.extra_payload,
                       'with_thinking':True, 'generation_started':False, 'raw_text':None}
            write_json(output/'writer-request.json', request)
            try:
                capacity = capacity_check(body, instruction, cfg, identity)
                write_json(output/'capacity.json', capacity)
                if not capacity['fits']:
                    raise ValueError('Complete source/context plus reserved output do not fit; no truncation')
                request['generation_started'] = True
                write_json(output/'writer-request.json', request)
                raw = call_llm(body, instruction, cfg, with_thinking=True)
                request.update(raw_text=raw, raw_response_sha256=hashlib.sha256(raw.encode('utf-8')).hexdigest())
                write_json(output/'writer-request.json', request)
                records = [_json(line) for line in (output/'writer-metrics.jsonl').read_text(encoding='utf-8').splitlines() if line]
                if (len(records) != 1 or records[0].get('stage') != cfg.stage
                        or records[0].get('thinking') is not True
                        or records[0].get('finish_reason') != 'stop'
                        or records[0].get('answer_chars') != len(raw)):
                    raise RuntimeError('Whole-source writer did not produce one complete normal-stop receipt')
                result, normalization = parse_translation(raw, source)
                request['normalization'] = normalization
                write_json(output/'writer-request.json', request)
                write_json(output/'translation.json', [asdict(row) for row in result])
                write_checked(result, output/'translation.utterances.srt')
                semantic_saved = True
            except BaseException as exc:
                body_error = exc
        restored = True
        if body_error is not None: raise body_error
    except BaseException as exc:
        if body_error is not None and body_error is not exc:
            exc.add_note('Original translation failure: '+exception_text(body_error))
        raise
    finally:
        write_json(output/'writer-lifecycle.json', {'wall_seconds':time.monotonic()-started,
                    'original_backend_restored':restored, 'semantic_artifact_preserved':semantic_saved})
    return result


def _display(source: list[SrtBlock], target: list[SrtBlock], output: Path) -> dict:
    """Use the unchanged formatter and settings, with no stale lexical timings."""
    metadata = {'utterance_tokens':[[] for _ in source], 'sample_rate':16000}
    layout = TranslateConfig(endpoint='http://127.0.0.1:8089/v1/chat/completions', max_tokens=2048,
                             extra_payload={'model':'qwen3.8-27b-dflash', 'temperature':.3, 'max_tokens':2048},
                             pause_layout=True, telemetry_path=output/'display-metrics.jsonl')
    ds, dt, mapping = build_display(source, target, metadata, layout,
                                   output/'display-requests.json', model_failure_fallback=True)
    if (''.join(row.text for row in ds) != ''.join(row.text for row in source)
            or ''.join(row.text for row in dt) != ''.join(row.text for row in target)):
        raise RuntimeError('Display formatter changed semantic text')
    write_checked(ds, output/'source.srt'); write_checked(dt, output/'subtitles.zh.srt')
    warnings = [dict(line=row['line'], utterance=row['utterance'], **row['layout_warning'])
                for row in mapping if row.get('layout_warning')]
    document = {'version':VERSION, 'cues':mapping, 'warnings':warnings,
                'semantic_text_changed':False, 'new_word_alignment_available':False}
    write_json(output/'display.json', document)
    return {'display_cues':len(dt), 'layout_warning_count':len(warnings)}


def run(config: PipelineConfig) -> dict:
    """Execute one fresh media run; retained artifacts cannot be silently rerolled."""
    before = time.monotonic(); preparation = prepare(config)
    preparation_seconds = time.monotonic()-before
    output = Path(preparation['output_dir']); output.mkdir(parents=True, exist_ok=False)
    write_json(output/'preparation.json', preparation); _snapshot(preparation, output)
    started = datetime.now(timezone.utc).isoformat(); beginning = time.monotonic()
    receipt = {'version':VERSION, 'status':'running', 'started_utc':started,
               'cpu_preparation_seconds_separate':preparation_seconds,
               'source_uncertainty_cleared':False, 'external_review_performed':False, 'score':None,
               'semantic_artifact_preserved':False, 'display_cues':0}
    write_json(output/'run.json', receipt)
    error = None
    try:
        if not _unchanged(preparation['input_hashes']):
            raise RuntimeError('Prepared media/context/model/runtime/code changed')
        audio_started = time.monotonic()
        audio = extract_audio(Path(preparation['media']), output/'audio.wav')
        plan = selected_asr.prepare_window_plan(Path(preparation['media']), audio, output/'windows')
        receipt['audio_preparation_seconds'] = time.monotonic()-audio_started
        generated = _run_asr(plan, preparation, output, timeout=config.asr_timeout)
        source, ledger = selected_asr.assemble_windows(plan, generated['rows'],
                                                      repetition_guard=selected_asr.REPETITION_POLICY)
        receipt.update(source_cues=len(source), audio_windows=len(plan['windows']),
                       empty_audio_windows=sum(row['empty'] for row in ledger), coverage=plan['coverage'])
        write_json(output/'source.json', {'source':[asdict(row) for row in source], 'windows':ledger,
                                         'source_uncertainty_cleared':False, 'new_word_alignment_available':False})
        if not source:
            receipt['status'] = 'completed_no_nonempty_source'
        else:
            write_checked(source, output/'source.utterances.srt')
            target = _translate(source, preparation, output)
            receipt['semantic_artifact_preserved'] = True
            receipt.update(_display(source, target, output), status='completed_unscored')
        if not _unchanged(preparation['input_hashes']):
            raise RuntimeError('Input/model/runtime/code changed during the run')
    except BaseException as exc:
        error = exc; receipt.update(status='failed', error=exception_text(exc))
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                       wall_seconds=time.monotonic()-beginning)
        for stage in ('asr', 'writer'):
            path = output/'asr/lifecycle.json' if stage=='asr' else output/'writer-lifecycle.json'
            if path.is_file():
                receipt[stage+'_lifecycle'] = _json(path.read_text(encoding='utf-8'))
        write_json(output/'run.json', receipt)
    if error is not None: raise error
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('media', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--profile', type=Path, help='Optional local path configuration; CLI overrides it')
    parser.add_argument('--context-file', type=Path)
    parser.add_argument('--asr-model-dir', type=Path)
    parser.add_argument('--writer-recipe', type=Path)
    parser.add_argument('--request-timeout', type=int, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument('--asr-timeout', type=int, default=1200)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    profile = {}
    try:
        if args.profile:
            profile = _json(args.profile.read_text(encoding='utf-8'))
            if (not isinstance(profile,dict) or profile.get('version') != 1
                    or set(profile)-{'version','asr_model_dir','writer_recipe'}):
                raise ValueError('Profile supports version1, asr_model_dir and writer_recipe only')
        paths = {}
        for name in ('context_file','asr_model_dir','writer_recipe'):
            value = getattr(args,name)
            if value is None and name in profile:
                value = Path(profile[name]).expanduser()
                if not value.is_absolute(): value = args.profile.resolve().parent/value
            if value is None: raise ValueError('--'+name.replace('_','-')+' is required')
            paths[name] = value
        config = PipelineConfig(media=args.media, output_dir=args.output_dir, **paths,
                                request_timeout=args.request_timeout, asr_timeout=args.asr_timeout)
        result = run(config) if args.execute else prepare(config)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error('%s: %s', type(exc).__name__, exc)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
