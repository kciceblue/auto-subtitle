"""Evidence-first local subtitle workflow: one media file -> Simplified-Chinese SRT.

The best fully local recipe of the September 2026 quality campaign
(design/local-ceiling-20260925.md): Astra v5 6/6/7 in both reviews on the
episode-10 benchmark, one pairwise step below the human fansub (7). Every model
runs locally on one GPU. There is no cloud API and no human input.

Stages. Each stage is resumable: a completed stage is verified and reused, never redone.
  source    Anime Whisper over ~20 s RMS windows: the primary Japanese   -> source/
  evidence  Zipformer and Qwen3-ASR readings of blind, short and BandIt
            crops; Qwen3-ASR auto/forced/masked and Voxtral window readings
                                                                           -> acoustic/ extras/
  draft     whole-episode Gemma 4 31B QAT Q4 draft (temporary llama-server) -> draft/
  align     Qwen3-ForcedAligner word timing of the primary Japanese       -> align/
  pieces    Japanese display units; exact partition of the draft (Qwen)   -> pieces/
  write     Qwen3.8-27B, reasoning on, writes every slot from all readings
            and the draft, twelve windows per request                     -> write/
  build     word-aligned cues and the subtitle punctuation convention     -> final/

    python -m src.evidence_first MEDIA [--out DIR] [--title TEXT] [--until STAGE]
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from src import aligned_display as ad
from src import fresh_six_requests as contracts
from src import fresh_writer_inputs as projection
from src import late_audio, local_gemma_native as native, selected_asr, short_audio
from src.audio import extract_audio
from src.config import TranslateConfig
from src.display import timestamp
from src.local_backend import _stop_owned_process, temporary_local_writer
from src.quality import mode
from src.selected_pipeline import load_writer_recipe, write_checked
from src.translate import call_llm
from src.workflow_state import file_hash, fingerprint, read_json, write_json

log = logging.getLogger('evidence_first')

ROOT = Path(__file__).resolve().parent.parent
ADMIN = 'http://127.0.0.1:8089/admin'
QWEN_ENDPOINT = 'http://127.0.0.1:8089/v1/chat/completions'
QWEN_MODEL = 'qwen3.8-27b-dflash'
ASR_MODELS = Path.home() / 'HF/asr-models'
WRITER_PROFILE = ROOT / 'profiles/long-context-gemma.json'
STAGES = ('source', 'evidence', 'draft', 'align', 'pieces', 'write', 'build')
EXTRAS = ('auto', 'forced', 'masked', 'voxtral')
WINDOWS_PER_PART = 12
# Request-cache key prefix of the writer; kept unchanged so the benchmark caches replay.
CACHE_VERSION = 'wording-pipeline-1'
CONTEXT = ('Original media title: {title}. Source audio language: Japanese. Target subtitles: '
           'Simplified Chinese. No synopsis, script, character glossary or reference translation is '
           'supplied. The title is background only, not proof of any line of dialogue.')
_RELEASE_TAG = re.compile(r'(?:\s*\[[^\]]*\]|\s+(?:HDTV|WEB(?:-?DL|Rip)?|BD(?:Rip)?|Blu-?Ray)\b[-\w]*'
                          r'|\s+\d{3,4}p)+$', re.I)


# ------------------------------------------------------------------ helpers

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def frozen(path: Path, value) -> None:
    """Write once; an existing file must already hold exactly this value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f'Frozen artifact changed: {path}')
        return
    with path.open('xb') as stream:
        stream.write(raw)


def verified(receipt: Path) -> bool:
    """True when a completion receipt exists and every file it pins is unchanged."""
    if not receipt.exists():
        return False
    for name, digest in read(receipt)['files'].items():
        if file_hash(Path(name)) != digest:
            raise RuntimeError(f'Completed artifact changed: {name} (delete {receipt.parent} to redo)')
    return True


def set_aside(folder: Path) -> None:
    """Keep an interrupted attempt for inspection and start clean."""
    if folder.exists():
        target = folder.with_name(f'{folder.name}.failed-{time.time_ns()}')
        folder.rename(target)
        log.warning('Previous incomplete %s kept as %s', folder.name, target.name)


def default_title(media: Path) -> str:
    return _RELEASE_TAG.sub('', media.stem).strip() or media.stem


def run_gpu_worker(argv: list[str], folder: Path, timeout: float, python: Path | str = sys.executable) -> None:
    """One short-lived GPU worker with the Warden LLM unloaded; Warden is restored afterwards."""
    folder.mkdir(parents=True, exist_ok=True)
    process = None
    with late_audio._gpu_lifecycle(ADMIN, folder / f'lifecycle-{time.time_ns()}.json') as lifecycle:
        try:
            with (folder / 'worker.log').open('a') as output:
                env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
                process = subprocess.Popen([str(python), *argv], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                           start_new_session=True, env=env)
                code = process.wait(timeout=timeout)
                if code:
                    raise RuntimeError(f'{" ".join(argv[:2])} failed with exit {code}; see {folder / "worker.log"}')
        finally:
            if process is not None:
                try:
                    _stop_owned_process(process)
                except BaseException:
                    lifecycle['owned_workers_stopped'] = False
                    raise


# ------------------------------------------------------------------ source

def stage_source(media: Path, work: Path) -> None:
    folder = work / 'source'
    if verified(folder / 'complete.json'):
        return
    set_aside(folder)
    folder.mkdir(parents=True)
    frozen(folder / 'started.json', {'started_utc': now(), 'media_sha256': file_hash(media)})
    start = time.monotonic()
    identity = selected_asr.bundle_identity(ROOT / 'models/anime-whisper')
    audio = extract_audio(media, folder / 'audio.wav')
    plan = selected_asr.prepare_window_plan(media, audio, folder / 'window-plan')
    worker = folder / 'native'
    worker.mkdir()
    selected_asr.write_json(worker / 'backend.json', identity)
    selected_asr.write_json(worker / 'windows.json', plan)
    spec = {'version': selected_asr.VERSION, 'plan_path': str(worker / 'windows.json'),
            'plan_sha256': selected_asr.file_hash(worker / 'windows.json'), 'output_dir': str(worker),
            'model_dir': identity['model_dir'], 'bundle_identity_sha256': selected_asr.json_hash(identity),
            'runtime_identity_sha256': selected_asr.file_hash(worker / 'backend.json'),
            'worker_sha256': selected_asr.file_hash(Path(selected_asr.__file__)),
            'batch_size': 8, 'max_new_tokens': 440, 'seed': 20260914, 'max_attempts': 3,
            'repetition_guard': dict(selected_asr.REPETITION_POLICY)}
    selected_asr.validate_spec(spec)
    frozen(worker / 'spec.json', spec)
    run_gpu_worker(['-m', 'src.selected_asr', '--spec', str(worker / 'spec.json')], worker, 1200)
    generated = read(worker / 'worker-result.json')
    if not generated['complete'] or generated['error']:
        raise RuntimeError('Anime Whisper worker did not complete')
    nonempty, ledger = selected_asr.assemble_windows(plan, generated['rows'], repetition_guard=spec['repetition_guard'])
    rows, owners = [], []
    for packet, result in zip(plan['windows'], ledger):
        a, b = packet['start_frame'] / selected_asr.RATE, packet['end_frame'] / selected_asr.RATE
        owners.append({'id': packet['number'], 'start': a, 'end': b})
        rows.append({'index': packet['number'], 'ts_line': timestamp(a, b), 'text': result['text']})
    if owners[0]['start'] != 0 or any(a['end'] != b['start'] for a, b in zip(owners, owners[1:])):
        raise RuntimeError('Window plan does not tile the media')
    write_checked(nonempty, folder / 'source-nonempty.srt')
    for name, value in (('source-all.json', rows), ('owners.json', owners), ('ledger.json', ledger)):
        selected_asr.write_json(folder / name, value)
    files = [folder / 'source-all.json', folder / 'owners.json', folder / 'source-nonempty.srt',
             worker / 'worker-result.json', folder / 'ledger.json']
    frozen(folder / 'complete.json', {'complete': True, 'windows': len(owners), 'nonempty_source_cues': len(nonempty),
                                      'seconds': time.monotonic() - start,
                                      'files': {str(p): file_hash(p) for p in files}})
    log.info('source: %d windows, %d non-empty', len(owners), len(nonempty))


# ------------------------------------------------------------------ evidence

def _normalize(pool: dict, prefix: str) -> list[dict]:
    rows = []
    crops = {c['view_id']: c for c in read(pool['plan_path'])['crops']}
    for item in pool['observations']:
        rate, provenance = item['sample_rate'], item['provenance']
        crop = crops[provenance['view_id']]
        rows.append({'observation_id': prefix + ':' + item['observation_id'], 'owner_id': item['owner_id'],
                     'start': item['crop_start_frame'] / rate, 'end': item['crop_end_frame'] / rate,
                     'observer': item['engine'], 'view': crop['kind'], 'transform': provenance['transform'],
                     'text': item['text'], 'status': item['status'],
                     'core_start': crop.get('core_start_frame', round(provenance['owner_start'] * rate)) / rate,
                     'core_end': crop.get('core_end_frame', round(provenance['owner_end'] * rate)) / rate,
                     'crop_start_frame': item['crop_start_frame'], 'crop_end_frame': item['crop_end_frame'],
                     'sample_rate': rate, 'word_ownership_verified': False,
                     'scope': 'full_crop_including_possible_neighbor_speech',
                     'duplicate_geometry': item.get('duplicate_geometry', False),
                     'receipt_sha256': item['receipt_sha256']})
    return rows


def _base_pools(media: Path, work: Path) -> None:
    if (work / 'base-complete.json').exists():
        done = read(work / 'base-complete.json')
        for pool in done['pools']:
            if file_hash(Path(pool['path'])) != pool['sha256']:
                raise RuntimeError(f"Acoustic pool changed: {pool['path']}")
        return
    started = time.monotonic()
    owners = read(work / 'source/owners.json')
    n = len(owners)
    count = min(n, max(1, round(n * 21 / 66)))
    bandit = sorted({1 + round(i * (n - 1) / (count - 1)) for i in range(count)}) if count > 1 else [1]
    frozen(work / 'acquisition-plan.json', {
        'media_sha256': file_hash(media), 'owner_count': n,
        'source_all_sha256': file_hash(work / 'source/source-all.json'),
        'bandit_selection': 'Uniform window indices over the full timeline; round(N*21/66).',
        'bandit_owner_ids': bandit, 'max_deep_batch': 20, 'full_audio_short_core': dict(short_audio.POLICY)})
    pools = []
    folder = work / 'acoustic/blind'
    pools.append((folder / 'evidence.json', late_audio.acquire(media, owners, folder, execute=True,
                                                               python_executable=sys.executable, worker_timeout=1800), 'blind'))
    folder = work / 'acoustic/short'
    short_audio.prepare(folder, work / 'acoustic/blind/preparation.json')
    pools.append((folder / 'evidence.json', short_audio.acquire(folder, timeout=2400), 'short'))
    for number, first in enumerate(range(0, len(bandit), 20), 1):
        ids = bandit[first:first + 20]
        folder = work / 'acoustic' / f'bandit-{number}'
        result = late_audio.acquire(media, owners, folder, mode='deep', flagged_ids=ids, masking_ids=ids,
                                    bandit_manifest=ROOT / 'models/bandit-v2/manifest.json', execute=True,
                                    python_executable=sys.executable, worker_timeout=2400)
        pools.append((folder / 'evidence.json', result, f'bandit-{number}'))
    observations = []
    for _, pool, prefix in pools:
        for engine in late_audio.ENGINES:
            if not any(i['engine'] == engine and i['status'] in ('ok', 'empty') for i in pool['observations']):
                raise RuntimeError(f'{prefix} has no usable {engine} observations')
        observations.extend(_normalize(pool, prefix))
    selected_asr.write_json(work / 'base-observations.json', observations)
    frozen(work / 'base-complete.json', {
        'complete': True, 'seconds': time.monotonic() - started, 'acoustic_observations': len(observations),
        'observation_counts': {s: sum(o['status'] == s for o in observations) for s in ('ok', 'empty', 'unavailable')},
        'normalized_observations_sha256': file_hash(work / 'base-observations.json'),
        'pools': [{'path': str(p), 'sha256': file_hash(p), 'pool_version': r['pool_version']} for p, r, _ in pools]})
    log.info('evidence: %d base observations', len(observations))


def _extra(spec: dict, work: Path, python: Path) -> dict:
    directory = Path(spec['output_dir'])
    meta = work / 'extra-specs' / directory.name
    frozen(meta / 'spec.json', spec)
    result_path = directory / 'evidence.json'
    if result_path.exists():
        result = read(result_path)
        if result.get('complete') is True:
            return result
        raise RuntimeError(f'Incomplete {spec["method"]} acquisition retained at {directory}; delete it to redo')
    run_gpu_worker(['-m', 'src.fresh_source_extras', '--spec', str(meta / 'spec.json')], meta, 2000, python)
    result = read(result_path)
    if result.get('complete') is not True:
        raise RuntimeError(f'{spec["method"]} acquisition did not complete')
    log.info('evidence: %s %d observations', spec['method'], len(result.get('observations', [])))
    return result


def stage_evidence(media: Path, work: Path) -> None:
    _base_pools(media, work)
    if (work / 'extras-complete.json').exists():
        for spec in read(work / 'extras-complete.json')['files'].values():
            if file_hash(Path(spec['path'])) != spec['sha256']:
                raise RuntimeError(f"Extra evidence changed: {spec['path']}")
        return
    owners = read(work / 'source/owners.json')
    prep = read(work / 'acoustic/short/preparation.json')
    common = {'mono_path': prep['mono']['path'], 'owners': owners, 'vad_speech': prep['vad_speech'], 'worker_seconds': 1800}
    results = {}
    for method in EXTRAS:
        destination = work / 'extras' / method
        resume = None
        if method == 'auto' and (destination / 'evidence.json').exists() and not read(destination / 'evidence.json').get('complete'):
            resume, destination = destination / 'evidence.json', work / 'extras/auto-continuation-1'
        spec = {**common, 'method': method, 'output_dir': str(destination),
                'model_dir': str(ROOT / 'models/voxtral-mini-4b-realtime-2602') if method == 'voxtral'
                else str(ASR_MODELS / 'Qwen3-ASR-1.7B')}
        if resume:
            spec['resume_from'] = str(resume)
        if method == 'forced':
            spec['automatic_evidence'] = results['auto']['path']
        python = ROOT / ('.venv-voxtral/bin/python' if method == 'voxtral' else '.venv/bin/python')
        _extra(spec, work, python)
        results[method] = {'path': str(destination / 'evidence.json'), 'sha256': file_hash(destination / 'evidence.json')}
    frozen(work / 'extras-complete.json', {'complete': True, 'files': results})


def collections(work: Path) -> list[tuple[str, list[dict]]]:
    """Every acoustic observation, in the fixed order base, auto, forced, masked, voxtral."""
    extras = read(work / 'extras-complete.json')['files']
    return [('base', read(work / 'base-observations.json'))] + [
        (name, read(extras[name]['path'])['observations']) for name in EXTRAS]


def readings_by_window(work: Path) -> dict[int, list[dict]]:
    """Usable readings per window: status ok, non-empty, first occurrence of each text."""
    out: dict[int, list[dict]] = {}
    for _, records in collections(work):
        for row in records:
            if row['status'] != 'ok' or not row['text']:
                continue
            bucket = out.setdefault(row['owner_id'], [])
            if all(r['text'] != row['text'] for r in bucket):
                bucket.append({'observer': row['observer'], 'start': float(row['start']),
                               'end': float(row['end']), 'text': row['text']})
    return out


# ------------------------------------------------------------------ draft

def stage_draft(work: Path, context: str) -> None:
    folder = work / 'draft'
    if verified(folder / 'complete.json'):
        return
    source = read(work / 'source/source-all.json')
    base = read(work / 'base-observations.json')
    clean, sidecar = projection.sanitize(base)
    if projection.restore(clean, sidecar) != base:
        raise RuntimeError('Writer input projection is not reversible')
    frozen(folder / 'context.json', {'context': context})
    request = contracts.build_request('LC', source, clean, context)
    settings = native.GemmaSettings(thinking=False, reasoning_budget_tokens=0, maximum_seconds=1980)
    prepared = native.prepare_request(request, settings, contract_id='fresh-six-requests-1:LC')
    frozen(folder / 'request.json', prepared)
    recipe = load_writer_recipe(WRITER_PROFILE)
    receipt = folder / 'native/draft.json'
    started = time.monotonic()
    with temporary_local_writer(recipe['weights'], recipe['server_binary'], alias=recipe['model'],
                                expected_sha256=recipe['sha256'], context_size=recipe['context_size'],
                                full_swa=recipe['full_swa'], threads=recipe['threads'], gpu_layers=recipe['gpu_layers'],
                                cpu_moe_layers=recipe['cpu_moe_layers'], restore_previous=True, admin_url=ADMIN,
                                restore_model=recipe['restore_model'], log_path=folder / 'server.log') as (_, identity):
        native._check_capacity(native._capacity(prepared, identity), prepared, identity)
        if receipt.exists():
            response = native.replay_native(receipt, prepared, read(receipt)['backend_identity'])
        else:
            response = native.ask(folder / 'native', 'draft', prepared, identity,
                                  time.monotonic() + prepared['settings']['maximum_seconds'])
    target = contracts.compile_target(response, source)
    frozen(folder / 'target-owners.json', target)
    frozen(folder / 'complete.json', {'complete': True, 'seconds': time.monotonic() - started,
                                      'empty_targets': sum(not r['text'] for r in target),
                                      'files': {str(folder / 'target-owners.json'): file_hash(folder / 'target-owners.json')}})
    log.info('draft: %d windows', len(target))


# ------------------------------------------------------------------ align + pieces

def stage_align(work: Path) -> None:
    result = work / 'align/alignment.json'
    if result.exists():
        return
    source = read(work / 'source/source-all.json')
    jobs = [{'owner': o['index'], 'text': o['text'], 'wav': str(work / 'source/window-plan' / f"window-{o['index']:05d}.wav")}
            for o in source if o['text']]
    spec = {'aligner_model': str(ASR_MODELS / 'Qwen3-ForcedAligner-0.6B'), 'warden_admin_url': ADMIN, 'jobs': jobs}
    write_json(work / 'align/spec.json', spec)
    partial = result.with_name('alignment.partial.json')
    subprocess.run([sys.executable, '-m', 'src.aligned_display', '--align', str(work / 'align/spec.json'), str(partial)],
                   cwd=ROOT, check=True)
    partial.replace(result)
    rows = read(result)['rows']
    log.info('align: %d windows, %d errors', len(rows), sum('error' in r for r in rows.values()))


def layout_config(work: Path) -> TranslateConfig:
    return TranslateConfig(endpoint=QWEN_ENDPOINT, max_tokens=2048,
                           extra_payload={'model': QWEN_MODEL, 'temperature': .3, 'max_tokens': 2048},
                           pause_layout=True, telemetry_path=work / 'pieces/partition-metrics.jsonl')


def owner_pieces(source: list[dict], target: list[dict], alignment: dict, config: TranslateConfig,
                 cache: Path) -> list[dict]:
    """Per window: timed Japanese units and the draft partitioned onto them, text unchanged."""
    rows = []
    for s, t in zip(source, target):
        if not t['text'] or not s['text']:
            rows.append({'owner': s['index'], 'units': None, 'pieces': None, 'source': s, 'target': t})
            continue
        start, end = (ad._seconds(v) for v in s['ts_line'].split(' --> '))
        tokens = None
        try:
            tokens = ad.owner_tokens(s['text'], alignment[str(s['index'])]['items'], start, end)
            units, _ = ad.repair_timing(ad.japanese_units(s['text'], tokens), start, end)
        except Exception:  # noqa: BLE001 - no usable alignment: proportional timing over a plausible extent
            low, high = (tokens[0]['start'], max(x['end'] for x in tokens)) if tokens else (start, end)
            if high - low < 0.1 * ad.lexical_count(s['text']):
                low, high = start, end
            units = ad.proportional_units(s['text'], low, high)
        pieces, status = ad.partition_target(units, t['text'], config, cache)
        rows.append({'owner': s['index'], 'units': units, 'pieces': pieces, 'partition': status, 'source': s, 'target': t})
    return rows


def stage_pieces(work: Path) -> None:
    out = work / 'pieces/pieces.json'
    if out.exists():
        return
    rows = owner_pieces(read(work / 'source/source-all.json'), read(work / 'draft/target-owners.json'),
                        read(work / 'align/alignment.json')['rows'], layout_config(work),
                        work / 'pieces/partition-cache.json')
    write_json(out, rows)
    log.info('pieces: %d slots', len(slots(rows)))


def slots(rows: list[dict]) -> list[dict]:
    """One slot per draft piece, in episode order: key ``window.piece``."""
    out = []
    for row in rows:
        for number, piece in enumerate(row['pieces'] or []):
            a, b = piece['units']
            units = row['units'][a:b + 1]
            out.append({'key': f"{row['owner']}.{number}", 'owner': row['owner'], 'piece': number,
                        'start': units[0]['start'], 'end': units[-1]['end'],
                        'japanese': ''.join(u['text'] for u in units), 'draft': piece['target']})
    return out


# ------------------------------------------------------------------ write

WRITE = '''You are given part of a Japanese anime episode (about 20-second windows). For each window you get every fallible speech-recognition reading (recognizer, time span, text) and numbered SLOTs. Each SLOT is one future Simplified-Chinese subtitle line with its primary-ASR Japanese (often misheard, sometimes only a gasp or humming when the readings show real words or sung lyrics) and an older literal machine DRAFT.

Write the professional Simplified-Chinese subtitle for every SLOT:
- Reconstruct what was really said/sung from all readings (agreement across recognizers is strong evidence). If readings attest a line the primary missed near a slot's time, include it in that slot's text.
- Natural, concise spoken Mandarin as in professional anime subtitles; faithful meaning, speaker/addressee and point of view, sentence-final nuance; no Japanese calques, no 桑/酱, no 您 between close friends; lyrics as concise poetic lines without exclamation marks; keep a character's recurring verbal tic (e.g. a signature sound word) with one consistent rendering.
- Slots that are only breathing, laughter, sobbing, gasps or wordless humming: output the slot key with nothing after it.
- Do NOT use outside knowledge of this series (character names, lyrics, plot): use only the evidence. For names, keep the draft's form when it matches the attested reading; when the ASR clearly misheard a name that other readings attest differently, use the attested one consistently.

Output every slot key of this part exactly once, in order, one per line, nothing else:
[slot key] Chinese subtitle'''
_ANSWER_LINE = re.compile(r'^\s*(?:SLOT\s*)?\[?(\d+\.\d+)\]?(?:\s+|$)(.*)$')


def _clock(t: float) -> str:
    return f'{int(t // 60):02d}:{t % 60:05.2f}'


def part_text(windows: range, readings: dict[int, list[dict]], items: list[dict]) -> str:
    lines = []
    for w in windows:
        lines.append(f'## WINDOW {w}')
        lines += [f"  reading {r['observer']} [{_clock(r['start'])}-{_clock(r['end'])}]: {r['text']}" for r in readings.get(w, [])]
        mine = [i for i in items if i['owner'] == w]
        if not mine:
            lines.append('  (no subtitle slots in this window)')
        lines += [f"  SLOT {i['key']} [{_clock(i['start'])}-{_clock(i['end'])}] JA: {i['japanese']} | DRAFT: {i['draft']}"
                  for i in mine]
        lines.append('')
    return '\n'.join(lines)


def part_bounds(windows: int, spec: str | None = None) -> list[tuple[int, int]]:
    """``1-12,13-24,...``; default: consecutive runs of WINDOWS_PER_PART windows."""
    if spec:
        bounds = [(int(item.split('-')[0]), int(item.split('-')[1])) for item in spec.split(',')]
    else:
        bounds = [(a, min(a + WINDOWS_PER_PART - 1, windows)) for a in range(1, windows + 1, WINDOWS_PER_PART)]
    if [w for a, b in bounds for w in range(a, b + 1)] != list(range(1, windows + 1)):
        raise ValueError('Part bounds must cover every window once, in order')
    return bounds


def writer_config(telemetry: Path, endpoint: str = QWEN_ENDPOINT, model: str = QWEN_MODEL) -> TranslateConfig:
    """Qwen3.8-27B with reasoning on; answers need ~5-15k tokens per part."""
    base = TranslateConfig(endpoint=endpoint, max_tokens=4096,
                           extra_payload={'model': model, 'temperature': .3, 'max_tokens': 4096}, telemetry_path=telemetry)
    config = mode(base, 'mirror', thinking=True)
    extra = dict(config.extra_payload or {})
    extra['max_tokens'] = 32768
    return replace(config, max_tokens=32768, extra_payload=extra,
                   response_guard_floor=max(config.response_guard_floor, 2048))


def cached_call(body: str, config: TranslateConfig, cache: Path) -> str:
    key = fingerprint([CACHE_VERSION, 'mirror', WRITE, body, config.endpoint, config.extra_payload])
    saved = read_json(cache, {})
    if isinstance(saved, dict) and isinstance(saved.get(key), str):
        return saved[key]
    answer = call_llm(body, WRITE, config)
    saved = read_json(cache, {})
    saved = saved if isinstance(saved, dict) else {}
    saved[key] = answer
    write_json(cache, saved)
    return answer


def write_part(body: str, config: TranslateConfig, cache: Path) -> dict[str, str]:
    """``{slot key: Chinese}``; a reasoning call that exhausts its retries is retried once with thinking off."""
    try:
        answer = cached_call(body, config, cache)
    except Exception:  # noqa: BLE001 - reasoning ran past the budget or ended without content
        answer = cached_call(body, mode(config, 'mirror', thinking=False), cache)
    found = {}
    for raw in answer.splitlines():
        match = _ANSWER_LINE.match(raw)
        if match and match.group(1) not in found:
            found[match.group(1)] = match.group(2).strip()
    return found


def stage_write(work: Path, bounds_spec: str | None = None, endpoint: str = QWEN_ENDPOINT, model: str = QWEN_MODEL) -> None:
    out = work / 'write'
    if (out / 'texts.json').exists():
        return
    rows = read(work / 'pieces/pieces.json')
    items = slots(rows)
    readings = readings_by_window(work)
    config = writer_config(out / 'metrics.jsonl', endpoint, model)
    texts, missing, began = {}, [], time.monotonic()
    for number, (a, b) in enumerate(part_bounds(len(rows), bounds_spec), 1):
        body = part_text(range(a, b + 1), readings, items)
        (out / 'parts').mkdir(parents=True, exist_ok=True)
        (out / 'parts' / f'part-{number}.txt').write_text(body, encoding='utf-8')
        keys = re.findall(r'SLOT (\d+\.\d+) ', body)
        found = write_part(body, config, out / 'cache.json')
        rest = [k for k in keys if k not in found]
        if rest:  # one retry restricted to the unanswered slots of this part
            lines = [l for l in body.splitlines() if not l.strip().startswith('SLOT ') or any(f'SLOT {k} ' in l for k in rest)]
            found.update({k: v for k, v in write_part('\n'.join(lines), config, out / 'cache.json').items() if k in rest})
        for key in keys:
            if key in found:
                texts[key] = found[key]
            else:
                missing.append(key)
        log.info('write: part %d (windows %d-%d) %d/%d slots, %.0fs', number, a, b,
                 sum(k in found for k in keys), len(keys), time.monotonic() - began)
    drafts = {i['key']: i['draft'] for i in items}
    for key in missing:
        texts[key] = drafts[key]
    write_json(out / 'texts.json', {'texts': texts, 'missing_fell_back_to_draft': missing})


# ------------------------------------------------------------------ build

def write_srt(cues: list[dict], path: Path) -> None:
    body = '\n'.join(f"{i}\n{timestamp(c['start'], c['end'])}\n{c['text']}\n" for i, c in enumerate(cues, 1))
    path.write_text(body, encoding='utf-8')


def build_cues(rows: list[dict], texts: dict[str, str]) -> list[dict]:
    """Place each slot's Chinese on its Japanese units; an empty slot is dropped."""
    pieces_out = []
    for row in rows:
        if row['pieces'] is None:
            if row['target']['text']:  # draft text for a window without primary Japanese
                start, end = (ad._seconds(v) for v in row['source']['ts_line'].split(' --> '))
                pieces_out.append({'owner': row['owner'], 'text': row['target']['text'], 'start': start, 'end': end})
            continue
        for number, piece in enumerate(row['pieces']):
            text = texts[f"{row['owner']}.{number}"]
            if not text.strip():
                continue
            a, b = piece['units']
            for part in ad.place_piece(text, row['units'][a:b + 1]):
                pieces_out.append({'owner': row['owner'], **part})
    media_end = ad._seconds(rows[-1]['source']['ts_line'].split(' --> ')[1])
    cues = ad.finalize_timing(pieces_out, media_end)
    for cue in cues:
        cue['raw'] = cue['text']
        cue['text'] = ad.convention(cue['text'])
    return [c for c in cues if c['text']]


def stage_build(media: Path, work: Path, final: Path) -> Path:
    rows = read(work / 'pieces/pieces.json')
    cues = build_cues(rows, read(work / 'write/texts.json')['texts'])
    final.mkdir(parents=True, exist_ok=True)
    write_json(work / 'write/cues.json', cues)
    target = final / f'{media.stem}.zh.srt'
    write_srt(cues, target)
    shutil.copyfile(work / 'source/source-nonempty.srt', final / f'{media.stem}.ja.srt')
    durations = [c['end'] - c['start'] for c in cues]
    log.info('build: %d cues, %d over 6 s, longest %d chars -> %s', len(cues), sum(d > 6 for d in durations),
             max((len(c['text']) for c in cues), default=0), target)
    return target


# ------------------------------------------------------------------ CLI

def run(media: Path, out: Path, title: str | None = None, until: str = 'build', part_spec: str | None = None) -> None:
    media = media.expanduser().resolve(strict=True)
    work = out / 'work'
    work.mkdir(parents=True, exist_ok=True)
    context = CONTEXT.format(title=title or default_title(media))
    steps = {'source': lambda: stage_source(media, work), 'evidence': lambda: stage_evidence(media, work),
             'draft': lambda: stage_draft(work, context), 'align': lambda: stage_align(work),
             'pieces': lambda: stage_pieces(work), 'write': lambda: stage_write(work, part_spec),
             'build': lambda: stage_build(media, work, out / 'final')}
    for stage in STAGES[:STAGES.index(until) + 1]:
        began = time.monotonic()
        log.info('== %s', stage)
        steps[stage]()
        log.info('== %s done in %.0fs', stage, time.monotonic() - began)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('media', type=Path)
    parser.add_argument('--out', type=Path, help='output folder (default: output/<media stem>)')
    parser.add_argument('--title', help='title given to the draft writer as background (default: cleaned file name)')
    parser.add_argument('--until', choices=STAGES, default='build', help='stop after this stage')
    parser.add_argument('--parts', help='explicit window ranges per writer request, e.g. 1-12,13-24')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    run(args.media, (args.out or ROOT / 'output' / args.media.stem).resolve(), args.title, args.until, args.parts)


if __name__ == '__main__':
    main()
