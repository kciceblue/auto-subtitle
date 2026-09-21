"""Native Anime Whisper for the selected local pipeline; imports never load weights.

The historical native generation/decoding and full-PCM window algorithms are
preserved. A final historical 25..30-second tail is split evenly for arbitrary
media because this native decoder accepts at most25 seconds. All source wording
and raw native token receipts stay local. Only the parent owns GPU admission.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import wave
import zlib
import numpy as np
from src.coherence import _json, _edited_text
from src.display import timestamp
from src.quality import validate_blocks
from src.translate import SrtBlock

VERSION = 'selected-native-anime-asr-1'
ALIAS = 'anime-whisper-local'
REVISION = '22e2008a8182b357da3922a6308d095008f72973'
RATE = 16000
MAX_SECONDS = 30
MAX_TOKENS = 440
MAX_DECODER_TOKENS = 448
TIMING = 'whole_window_approximate'
REPETITION_POLICY = {'version':'utf8-zlib-repetition-1','threshold':2.4,'max_attempts':3}



MODEL_PINS = {'added_tokens.json': ('3c51f66c4c21f9e126970078f11ae77a78c74aee8df606ee9daba86e467108e0', 34648),
 'config.json': ('1b0bf54206d648763dbadbab3fc7ff0577b0f80224f002dbbe8d8c2caa147360', 1268),
 'generation_config.json': ('4e5bccc702f2322fcb497967cff54f890dcb3053c2993884e44313655fa951e0', 3898),
 'merges.txt': ('2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6', 493869),
 'model.safetensors': ('15c672f0bf687b1c67aa14325f9c382c6919f1fce976f990513618b464a3c626', 3025686376),
 'normalizer.json': ('bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd', 52666),
 'preprocessor_config.json': ('7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711', 340),
 'special_tokens_map.json': ('baea4ea09372eb4fca86b4e4346139fd73cb807d5087e9de0948e971739c3e74', 2186),
 'tokenizer_config.json': ('c86c7398ba7f81f784569ea0ae165e252c4a006acc56906f04e5ee3300cf043e', 282863),
 'vocab.json': ('e2aa043ef015641d363d8288e7c241c85e36a5c761fb303598e0710233344387', 1036558)}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def json_hash(value) -> str:
    return text_hash(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def write_json(path: Path, value) -> None:
    """Atomic JSON persistence; no model output is logged to the console."""
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def validate_repetition_policy(policy) -> dict | None:
    if policy is None:
        return None
    if not isinstance(policy, dict) or policy != REPETITION_POLICY or type(policy.get('max_attempts')) is not int:
        raise ValueError('Invalid repetition guard policy')
    return dict(policy)


def seed_for(window: int, attempt: int, base_seed: int = 20260914) -> int:
    """One seed per native batch: window is its first ordered member, attempt is1-based."""
    if any(type(x) is not int or x < 1 for x in (window, attempt, base_seed)) or attempt > 3:
        raise ValueError('Invalid seed coordinates')
    return base_seed + 100000 * (attempt - 1) + window


def compression_check(text: str, policy: dict | None) -> dict | None:
    if policy is None:
        return None
    validate_repetition_policy(policy)
    data = text.encode('utf-8')
    ratio = len(data) / len(zlib.compress(data))
    return {'version': policy['version'], 'ratio': ratio, 'threshold': policy['threshold'],
            'passed': ratio <= policy['threshold']}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def segmentation_policy() -> dict:
    return {'target_seconds': 22, 'search_min_seconds': 18, 'search_max_seconds': 25,
            'rms_window_seconds': .1, 'candidate_step_seconds': .01,
            'tail_policy': 'finish_at_most30s_or_balance_final_pair_at_least15s',
            'maximum_seconds': MAX_SECONDS, 'vad_dropping': False, 'padding_frames': 0}


def _historical_partition_pcm(pcm: bytes) -> list[dict]:
    """Deterministic RMS cuts; retain all frames, avoiding a tiny final tail."""
    if not pcm or len(pcm) % 2:
        raise ValueError('Expected nonempty complete PCM16 frames')
    samples = np.frombuffer(pcm, dtype='<i2')
    total, cursor, windows = len(samples), 0, []
    while cursor < total:
        remaining = total - cursor
        if remaining <= MAX_SECONDS * RATE:
            cut, rms, strategy = total, None, 'final_whole_window'
        else:
            if remaining <= 40 * RATE:
                lo, hi = max(15 * RATE, remaining - 25 * RATE), min(25 * RATE, remaining - 15 * RATE)
                target, strategy = remaining // 2, 'balanced_final_pair_rms'
            else:
                lo, hi, target, strategy = 18 * RATE, 25 * RATE, 22 * RATE, 'target22s_rms'
            low, high = cursor + lo, cursor + hi
            candidates = np.unique(np.r_[np.arange(low, high + 1, RATE // 100, dtype=np.int64),
                                          low, high, cursor + target])
            half = RATE // 20
            region_start, region_end = max(0, low - half), min(total, high + half)
            segment = samples[region_start:region_end].astype(np.float64)
            prefix = np.r_[0., np.cumsum(segment * segment)]
            left = np.maximum(candidates - half, region_start) - region_start
            right = np.minimum(candidates + half, region_end) - region_start
            power = (prefix[right] - prefix[left]) / (right - left)
            # Equal RMS prefers the cut closest to22s (or the balanced tail).
            chosen = min(range(len(candidates)), key=lambda i: (power[i], abs(int(candidates[i]) - cursor - target), int(candidates[i])))
            cut, rms = int(candidates[chosen]), float(math.sqrt(power[chosen]))
        windows.append({'number': len(windows) + 1, 'start_frame': cursor, 'end_frame': cut,
                        'sample_rate': RATE, 'cut_strategy': strategy, 'cut_rms_pcm16': rms})
        cursor = cut
    return windows


def partition_pcm(pcm: bytes) -> list[dict]:
    """Preserve historical cuts except a tail too long for the native decoder."""
    windows = _historical_partition_pcm(pcm)
    last = windows[-1]
    if last['end_frame'] - last['start_frame'] > 25 * RATE:
        midpoint = (last['start_frame'] + last['end_frame']) // 2
        original = windows.pop()
        for start, end in ((original['start_frame'], midpoint), (midpoint, original['end_frame'])):
            windows.append(dict(original, number=len(windows)+1, start_frame=start, end_frame=end,
                                cut_strategy='native_max25_final_tail_split', cut_rms_pcm16=None))
    return windows


def read_pcm(path: Path, expected_sha256: str) -> bytes:
    raw = path.read_bytes()
    if sha_bytes(raw) != expected_sha256:
        raise ValueError('Original WAV hash differs')
    with wave.open(io.BytesIO(raw), 'rb') as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, RATE, 'NONE'):
            raise ValueError('Original WAV must be16kHz mono PCM16')
        frames = wav.getnframes()
        pcm = wav.readframes(frames)
    if frames <= 0 or len(pcm) != frames * 2:
        raise ValueError('Original WAV has empty or truncated PCM')
    return pcm


def verify_window_plan(plan: dict, *, compare_pcm: bool = True) -> dict:
    if plan.get('version') != VERSION or plan.get('segmentation') != segmentation_policy():
        raise ValueError('Continuous-window plan version or segmentation policy differs')
    pcm = read_pcm(Path(plan['audio']), plan['audio_sha256'])
    expected, cursor = partition_pcm(pcm), 0
    if not isinstance(plan.get('windows'), list) or len(plan['windows']) != len(expected):
        raise ValueError('Continuous-window count differs')
    for wanted, packet in zip(expected, plan['windows']):
        if any(packet.get(key) != value for key, value in wanted.items()):
            raise ValueError('Continuous-window cuts/identity differ from original PCM policy')
        start, end = packet['start_frame'], packet['end_frame']
        if start != cursor or not 0 < end - start <= MAX_SECONDS * RATE:
            raise ValueError('Continuous windows have a gap, overlap or invalid duration')
        raw = Path(packet['audio']).read_bytes()
        if sha_bytes(raw) != packet['sha256']:
            raise ValueError('Window WAV hash differs')
        with wave.open(io.BytesIO(raw), 'rb') as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes(), wav.getcomptype()) != (1, 2, RATE, end - start, 'NONE'):
                raise ValueError('Window WAV header differs')
            local = wav.readframes(end - start)
        if len(local) != (end - start) * 2 or sha_bytes(local) != packet['pcm_sha256']:
            raise ValueError('Window PCM length or hash differs')
        if compare_pcm and local != pcm[start * 2:end * 2]:
            raise ValueError('Window is not the exact original PCM frame interval')
        cursor = end
    total = len(pcm) // 2
    if cursor != total or plan.get('media_frames') != total or plan.get('pcm_sha256') != sha_bytes(pcm):
        raise ValueError('Continuous windows do not cover the entire original PCM')
    durations = [(w['end_frame'] - w['start_frame']) / RATE for w in expected]
    return {'scope': 'full_media', 'sample_rate': RATE, 'media_frames': total, 'processed_frames': total,
            'unprocessed_frames': 0, 'gaps': 0, 'overlaps': 0, 'padding_frames': 0,
            'window_count': len(expected), 'minimum_seconds': min(durations), 'maximum_seconds': max(durations),
            'exact_original_pcm_checked': compare_pcm, 'vad_dropping': False,
            'fresh_full_media_coverage': True, 'speech_coverage_verified': False}


def parse_japanese(raw: str, bound: int) -> tuple[str, dict]:
    value = _json(raw)
    if not isinstance(value, dict) or set(value) != {'1'} or not isinstance(value['1'], str):
        raise ValueError('ASR response must contain exactly string key1')
    text = value['1']
    if text:
        _edited_text(text, 'x' * math.ceil(bound / 4))  # Validate only; retain the original string verbatim.
        if len(text) > bound or '\n' in text or '\r' in text:
            raise ValueError('ASR output exceeds duration bound or is not one line')
    return text, {'policy': 'strict_single_string_verbatim_or_empty', 'accepted_text_sha256': sha_bytes(text.encode('utf-8')),
                  'empty': text == '', 'normalization_performed': False, 'language_verified': False}


def assemble_windows(plan: dict, generated: list[dict], *, repetition_guard: dict | None = None
                     ) -> tuple[list[SrtBlock], list[dict]]:
    policy = validate_repetition_policy(repetition_guard)
    if len(generated) != len(plan['windows']):
        raise ValueError('Every audio window, including empty windows, must finish before source commit')
    source, ledger = [], []
    for packet, result in zip(plan['windows'], generated):
        if (result['window'] != packet['number'] or result['start_frame'] != packet['start_frame']
                or result['end_frame'] != packet['end_frame'] or result['audio_sha256'] != packet['sha256']):
            raise ValueError('ASR window result identity differs')
        duration = (packet['end_frame'] - packet['start_frame']) / RATE
        text, parsed = parse_japanese(json.dumps({'1': result['text']}, ensure_ascii=False), max(160, math.ceil(duration * 35)))
        if parsed['accepted_text_sha256'] != result['accepted_text_sha256']:
            raise ValueError('Accepted ASR text changed')
        if policy is not None:
            checked = compression_check(text, policy)
            if result.get('repetition_check') != checked or not checked['passed']:
                raise ValueError('Accepted ASR window failed its repetition guard')
        index = len(source) + 1 if text else None
        if index is not None:
            source.append(SrtBlock(index, timestamp(packet['start_frame'] / RATE, packet['end_frame'] / RATE), text))
        ledger.append({**result, 'source_index': index, 'empty': text == '', 'status': 'UNVERIFIED',
                       'model_reported_no_intelligible_speech': text == '', 'silence_verified': False,
                       'timing': TIMING, 'source_uncertainty_cleared': False, 'new_word_alignment_available': False})
    if source:
        validate_blocks(source)
    return source, ledger


def bundle_identity(model_dir: Path) -> dict:
    """Pin the existing official bundle and the active interpreter's native APIs."""
    model_dir = Path(model_dir).resolve(strict=True)
    files, paths = {}, {}
    for name, (digest, size) in MODEL_PINS.items():
        path = model_dir / name
        if not path.is_file() or path.stat().st_size != size or file_hash(path) != digest:
            raise ValueError('Official Anime Whisper asset differs: ' + name)
        files[name] = {'sha256': digest, 'bytes': size}
        paths[str(path)] = digest
    wanted = {'transformers', 'torch', 'numpy', 'safetensors', 'tokenizers'}
    versions = {}
    for name in sorted(wanted):
        distribution = importlib.metadata.distribution(name)
        versions[name] = distribution.version
        for item in distribution.files or []:
            if str(item).endswith(('.dist-info/METADATA', '.dist-info/RECORD')):
                path = Path(distribution.locate_file(item)).resolve()
                paths[str(path)] = file_hash(path)
    if versions['transformers'] != '4.57.6':
        raise ValueError('Selected native Whisper requires Transformers4.57.6')
    package = Path(importlib.metadata.distribution('transformers').locate_file('transformers'))
    api = [*sorted((package/'models/whisper').glob('*.py')),
           *[package/name for name in ('generation/utils.py', 'generation/configuration_utils.py',
              'generation/logits_process.py', 'generation/stopping_criteria.py',
              'tokenization_utils.py', 'tokenization_utils_base.py')]]
    if len(api) < 7:
        raise ValueError('Native Whisper API files missing')
    paths.update({str(path.resolve()): file_hash(path) for path in api})
    return {'model_alias': ALIAS, 'model_dir': str(model_dir), 'publisher': 'litagin/anime-whisper',
            'publisher_revision': REVISION, 'model_files': files, 'package_versions': versions,
            'input_hashes': paths, 'python': str(Path(sys.executable).resolve()),
            'api': 'WhisperProcessor + WhisperForConditionalGeneration.generate'}


def read_window(packet: dict):
    import numpy as np
    raw = Path(packet['audio']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != packet['sha256']:
        raise ValueError('Window WAV changed before request')
    with wave.open(io.BytesIO(raw), 'rb') as wav:
        frames = packet['end_frame'] - packet['start_frame']
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes(), wav.getcomptype()) != (1, 2, RATE, frames, 'NONE'):
            raise ValueError('Window WAV header differs')
        pcm = wav.readframes(frames)
    if len(pcm) != frames * 2 or hashlib.sha256(pcm).hexdigest() != packet['pcm_sha256']:
        raise ValueError('Window PCM changed before request')
    return np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.


def generation_policy(attempt: int, seed: int) -> dict:
    if type(attempt) is not int or not 1 <= attempt <= 3 or type(seed) is not int or not 0 < seed < 2**32:
        raise ValueError('Invalid attempt or seed')
    return {'attempt':attempt,'seed':seed,'max_new_tokens':MAX_TOKENS,
        'language':'ja','task':'transcribe','return_timestamps':False,
        'condition_on_prev_tokens':False,'do_sample':attempt>1,'temperature':0.3 if attempt>1 else 0.0,
        'top_p':0.95 if attempt>1 else 1.0,'top_k':50,'num_beams':1,'num_return_sequences':1,
        'no_repeat_ngram_size':0,'repetition_penalty':1.0,'return_dict_in_generate':True,
        'return_token_timestamps':False,'force_unique_generate_call':True,
        'compression_ratio_threshold':None,'logprob_threshold':None,'no_speech_threshold':None}


def decoder_identity(config: dict, tokenizer, generation_config: dict | None = None) -> dict:
    config=dict(config)
    if generation_config is not None:
        for key in ('decoder_start_token_id','eos_token_id','pad_token_id'):
            if key in generation_config:config[key]=generation_config[key]
    prefix=[config['decoder_start_token_id']]
    prompts=tokenizer.get_decoder_prompt_ids(language='ja',task='transcribe',no_timestamps=True)
    if [p[0] for p in prompts]!=[1,2,3]:
        raise ValueError('Unexpected Japanese/transcribe native prefix')
    prefix.extend(p[1] for p in prompts)
    result={'prefix_ids':prefix,'eos_token_id':config['eos_token_id'],'pad_token_id':config['pad_token_id'],
        'max_target_positions':config['max_target_positions'],'vocab_size':config['vocab_size']}
    if (result['max_target_positions']!=MAX_DECODER_TOKENS or len(prefix)+MAX_TOKENS>MAX_DECODER_TOKENS
            or any(type(x) is not int or x<0 for x in prefix)):
        raise ValueError('Native decoder prefix exceeds bounded budget')
    return result


def token_completion(token_ids: list[int], decoder: dict) -> dict:
    if (not isinstance(token_ids,list) or not token_ids or len(token_ids)>MAX_DECODER_TOKENS
            or any(type(x) is not int or not 0<=x<decoder['vocab_size'] for x in token_ids)):
        raise ValueError('Invalid or oversized native token sequence')
    prefix=decoder['prefix_ids'];eos=decoder['eos_token_id'];pad=decoder['pad_token_id']
    if token_ids[:len(prefix)]!=prefix:
        raise ValueError('Native output prefix is not Japanese/transcribe without context')
    try:end=token_ids.index(eos,len(prefix))
    except ValueError:raise ValueError('Native output missing EOS; truncated generation') from None
    if any(x not in {eos,pad} for x in token_ids[end+1:]):
        raise ValueError('Native output contains data after EOS')
    generated=end+1-len(prefix)
    if generated>MAX_TOKENS:
        raise ValueError('Native decoder exceeded new-token budget')
    return {'eos_observed':True,'generated_tokens_through_eos':generated,
        'sequence_tokens_including_padding':len(token_ids),'decoder_budget':MAX_DECODER_TOKENS,
        'max_new_tokens':MAX_TOKENS,'padding_tokens_after_eos':len(token_ids)-end-1}


def decode_native(token_ids: list[int], tokenizer) -> tuple[str,dict]:
    raw=tokenizer.decode(token_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    text=tokenizer.decode(token_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    if not isinstance(raw,str) or not isinstance(text,str):raise ValueError('Native decode did not return strings')
    return raw,{'text':text,'language':'Japanese','time_stamps':None}


def check_output(token_ids: list[int], native_raw: str, native_result: dict, duration: float,
                 tokenizer, decoder: dict, repetition_guard: dict = REPETITION_POLICY) -> dict:
    policy=validate_repetition_policy(repetition_guard)
    if policy is None:raise ValueError('Anime-Whisper requires its structural repetition guard')
    if not isinstance(duration,(int,float)) or isinstance(duration,bool) or not math.isfinite(duration) or not 0<duration<=25:
        raise ValueError('Anime-Whisper accepts exact windows of at most25 seconds')
    completion=token_completion(token_ids,decoder)
    raw,expected=decode_native(token_ids,tokenizer)
    if native_raw!=raw or native_result!=expected:
        raise ValueError('Stored text differs from exact native token decoding')
    text=expected['text'];bound=max(160,math.ceil(duration*35))
    if len(text)>bound or len(raw)>MAX_DECODER_TOKENS*128:
        raise ValueError('Native text exceeds audio-duration guard')
    parse_japanese(json.dumps({'1':text},ensure_ascii=False),bound)
    if text and (not text.strip() or not re.search(r'[\u3040-\u30ff\u3400-\u9fff0-9]',text)):
        raise ValueError('Native source lacks expected Japanese-script content')
    if any(ord(c)<32 and c not in '\n\t\r' for c in text) or '<|' in text or '<think' in text.lower():
        raise ValueError('Native source contains control or role markup')
    checked=compression_check(text,policy)
    return {'accepted':checked['passed'],'text':text,'empty':text=='','accepted_text_sha256':text_hash(text),
        'native_raw_sha256':text_hash(raw),'native_token_ids_sha256':json_hash(token_ids),
        'completion':completion,'normalization_owner':'native_WhisperTokenizer_decode_skip_special_tokens_cleanup_false',
        'worker_text_edits':False,'maximum_source_characters':bound,'repetition_check':checked,
        'raw_repetition_check':None,'raw_repetition_scope':'special_token_and_batch_padding_decode_not_comparable_to_text',
        'language_verified':False,'source_fidelity_verified':False}


def capture_batch(model,processor,audios,settings: dict,decoder: dict,evidence: dict | None = None):
    """One native short-form generation call, no prompt/history or internal fallback loop."""
    import torch
    evidence={} if evidence is None else evidence
    if settings!=generation_policy(settings['attempt'],settings['seed']):raise ValueError('Generation policy differs')
    if any(len(a)==0 or len(a)>25*RATE for a in audios):raise ValueError('Invalid native waveform length')
    inputs=processor(audios,sampling_rate=RATE,return_tensors='pt',padding='max_length',
        truncation=False,return_attention_mask=True)
    features=inputs['input_features']
    if tuple(features.shape)!=(len(audios),model.config.num_mel_bins,model.config.max_source_positions*2):
        raise ValueError('Native features changed batch/window mapping')
    if 'attention_mask' not in inputs:raise ValueError('Batched native features require attention masks')
    kwargs={k:v for k,v in settings.items() if k not in {'attempt','seed'}}
    # Explicitly disable the publisher's optional fallback thresholds and previous
    # token conditioning in a private copy, so no hidden per-window retries occur.
    from copy import deepcopy
    generation=deepcopy(model.generation_config)
    for key in ('compression_ratio_threshold','logprob_threshold','no_speech_threshold'):
        setattr(generation,key,None)
    generation.condition_on_prev_tokens=False
    generation.forced_decoder_ids=None
    generation.no_repeat_ngram_size=0
    generation.repetition_penalty=1.0
    with torch.inference_mode():
        output=model.generate(input_features=features.to(device='cuda',dtype=torch.float16),
            attention_mask=inputs['attention_mask'].to('cuda'),generation_config=generation,**kwargs)
    sequences=output.sequences.detach().cpu().tolist()
    evidence['token_ids']=sequences
    if len(sequences)!=len(audios):raise ValueError('Native generated batch cardinality differs')
    decoded=[decode_native(ids,processor.tokenizer) for ids in sequences]
    evidence.update(raw_outputs=[x[0] for x in decoded],native_results=[x[1] for x in decoded])
    return sequences,evidence['raw_outputs'],evidence['native_results'],generation.to_dict()


def prepare_window_plan(media: Path, audio: Path, folder: Path) -> dict:
    """Partition already extracted PCM, without VAD dropping or model calls."""
    audio = audio.resolve(strict=True)
    pcm = read_pcm(audio, file_hash(audio))
    folder.mkdir(parents=True, exist_ok=False)
    windows = []
    for packet in partition_pcm(pcm):
        path = folder / f"window-{packet['number']:05d}.wav"
        local = pcm[packet['start_frame']*2:packet['end_frame']*2]
        with wave.open(str(path), 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(RATE); wav.writeframes(local)
        windows.append(dict(packet, audio=str(path), sha256=file_hash(path), pcm_sha256=sha_bytes(local)))
    plan = {'version': VERSION, 'audio': str(audio), 'audio_sha256': file_hash(audio),
            'video': str(media.resolve()), 'video_sha256': file_hash(media),
            'pcm_sha256': sha_bytes(pcm), 'media_frames': len(pcm)//2, 'sample_rate': RATE,
            'segmentation': segmentation_policy(), 'windows': windows,
            'native_max25_tail_adaptation': any(w['cut_strategy']=='native_max25_final_tail_split' for w in windows)}
    plan['coverage'] = verify_window_plan(plan)
    return plan


def validate_spec(spec: dict) -> tuple[dict, dict]:
    if not isinstance(spec, dict) or spec.get('version') != VERSION:
        raise ValueError('Invalid selected native worker version')
    for key, value in {'batch_size':8, 'max_attempts':3, 'max_new_tokens':440, 'seed':20260914}.items():
        if type(spec.get(key)) is not int or spec[key] != value:
            raise ValueError('Selected native worker policy differs: ' + key)
    if validate_repetition_policy(spec.get('repetition_guard')) is None:
        raise ValueError('Missing structural repetition policy')
    if file_hash(Path(__file__)) != spec.get('worker_sha256'):
        raise ValueError('Worker code changed')
    plan_path = Path(spec['plan_path']).resolve(strict=True)
    if file_hash(plan_path) != spec.get('plan_sha256'):
        raise ValueError('Worker plan changed')
    plan = _json(plan_path.read_text(encoding='utf-8'))
    if verify_window_plan(plan) != plan.get('coverage'):
        raise ValueError('Full PCM window coverage differs')
    if any(not 0 < p['end_frame']-p['start_frame'] <=25*RATE for p in plan['windows']):
        raise ValueError('Native worker window exceeds25 seconds')
    if file_hash(Path(plan['video'])) != plan['video_sha256']:
        raise ValueError('Original media changed')
    identity = bundle_identity(Path(spec['model_dir']))
    if json_hash(identity) != spec.get('bundle_identity_sha256'):
        raise ValueError('Worker native runtime/model bundle changed')
    output = Path(spec['output_dir']).resolve(strict=True)
    if file_hash(output/'backend.json') != spec.get('runtime_identity_sha256'):
        raise ValueError('Worker backend receipt changed')
    if (output/'worker-result.json').exists() or ((output/'requests').exists() and any((output/'requests').iterdir())):
        raise ValueError('Worker output already exists; use a fresh output directory')
    return plan, identity


def run(spec: dict) -> dict:
    started=time.monotonic();plan,identity=validate_spec(spec);output=Path(spec['output_dir']).resolve();model_dir=Path(spec['model_dir'])
    (output/'requests').mkdir(exist_ok=True);policy=validate_repetition_policy(spec['repetition_guard'])
    result={'version':VERSION,'status':'incomplete','complete':False,'rows':[],'error':None,
        'model':ALIAS,'source_language':'ja','language':'Japanese','task':'transcribe','context':'','initial_prompt':None,
        'plan_sha256':spec['plan_sha256'],'bundle_identity_sha256':spec['bundle_identity_sha256'],
        'runtime_identity_sha256':spec['runtime_identity_sha256'],'repetition_guard':policy,'max_attempts':spec['max_attempts'],
        'expected_window_count':len(plan['windows']),'window_count':0,'load_seconds':None,'inference_seconds':0.,
        'validation_seconds':time.monotonic()-started,'worker_wall_seconds':None,'repetition_rejections':0,
        'source_fidelity_verified':False,'source_uncertainty_cleared':False,'new_word_alignment_available':False,
        'model_unload_and_original_restoration_owner':'parent_driver','batches':[]}
    requests={p['number']:{'version':VERSION,'window':p,'language':'Japanese','task':'transcribe','context':'',
        'initial_prompt':None,'return_time_stamps':False,'runtime_identity_sha256':spec['runtime_identity_sha256'],
        'bundle_identity_sha256':spec['bundle_identity_sha256'],'status':'UNVERIFIED','repetition_guard':policy,
        'max_attempts':spec['max_attempts'],'attempts':[]} for p in plan['windows']}
    accepted={}
    def save():
        result['rows']=[accepted[n] for n in sorted(accepted)];result['window_count']=len(accepted)
        result['worker_wall_seconds']=time.monotonic()-started;write_json(output/'worker-result.json',result)
    save()
    try:
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
        import numpy as np
        import random
        import torch
        from transformers import WhisperProcessor,WhisperForConditionalGeneration
        before=time.monotonic()
        processor=WhisperProcessor.from_pretrained(str(model_dir),local_files_only=True)
        model=WhisperForConditionalGeneration.from_pretrained(str(model_dir),torch_dtype=torch.float16,
            local_files_only=True,use_safetensors=True,low_cpu_mem_usage=True).to('cuda').eval()
        result['load_seconds']=time.monotonic()-before
        decoder=decoder_identity(model.config.to_dict(),processor.tokenizer,model.generation_config.to_dict());result['decoder_identity']=decoder
        pending=list(plan['windows'])
        for attempt in range(1,spec['max_attempts']+1):
            retry=[]
            for offset in range(0,len(pending),spec['batch_size']):
                packets=pending[offset:offset+spec['batch_size']];numbers=[p['number'] for p in packets]
                seed=seed_for(numbers[0],attempt,spec['seed']);settings=generation_policy(attempt,seed)
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                before=time.monotonic();tokens=None;raws=None;native=None;effective=None;error=None;evidence={}
                try:
                    tokens,raws,native,effective=capture_batch(model,processor,[read_window(p) for p in packets],settings,decoder,evidence)
                except Exception as exc:
                    error=type(exc).__name__+': '+str(exc)
                    tokens=evidence.get('token_ids');raws=evidence.get('raw_outputs');native=evidence.get('native_results')
                seconds=time.monotonic()-before;result['inference_seconds']+=seconds
                batch={'number':len(result['batches'])+1,'windows':numbers,'attempt':attempt,'seed':seed,
                    'settings':settings,'seconds':seconds,'error':error,'effective_generation_config':effective,
                    'generation_config_owner':'explicit_private_native_Whisper_generation_config',
                    'token_ids':tokens,'raw_outputs':raws,'native_results':native,'decoder_identity':decoder}
                result['batches'].append(batch)
                for i,p in enumerate(packets):
                    number=p['number'];request=requests[number]
                    item={'attempt':attempt,'batch':batch['number'],'batch_windows':numbers,'batch_seed':seed,
                        'settings':settings,'seconds_batch_shared':seconds,'decoder_identity':decoder,
                        'token_ids':tokens[i] if isinstance(tokens,list) and i<len(tokens) else None,
                        'raw_text':raws[i] if isinstance(raws,list) and i<len(raws) else None,
                        'native_result':native[i] if isinstance(native,list) and i<len(native) else None,
                        'checked':None,'error':error}
                    if error is None:
                        try:
                            item['checked']=check_output(item['token_ids'],item['raw_text'],item['native_result'],
                                (p['end_frame']-p['start_frame'])/RATE,processor.tokenizer,decoder,policy)
                            if not item['checked']['accepted']:
                                result['repetition_rejections']+=1;item['error']='Repetition guard rejected decoded native text'
                        except Exception as exc:item['error']=type(exc).__name__+': '+str(exc)
                    request['attempts'].append(item);path=output/'requests'/f'window-{number:03d}.json';write_json(path,request)
                    if item['error'] is not None:retry.append(p)
                    else:
                        checked=item['checked'];accepted[number]={'window':number,'start_frame':p['start_frame'],'end_frame':p['end_frame'],
                            'text':checked['text'],'empty':checked['empty'],'accepted_text_sha256':checked['accepted_text_sha256'],
                            'audio_sha256':p['sha256'],'request_path':str(path),'request_sha256':file_hash(path),
                            'model':ALIAS,'status':'UNVERIFIED','source_fidelity_verified':False,
                            'repetition_check':checked['repetition_check'],'raw_repetition_check':None,
                            'native_token_ids_sha256':checked['native_token_ids_sha256']}
                save()
            pending=retry
            if not pending:break
        if pending:raise RuntimeError('One or more native ASR windows exhausted their bounded attempts')
        if sorted(accepted)!=[p['number'] for p in plan['windows']]:raise ValueError('Completed worker omitted or duplicated a window')
        verify_window_plan(plan)
        if json_hash(bundle_identity(Path(spec['model_dir'])))!=spec['bundle_identity_sha256']:raise ValueError('Model/API changed during inference')
        if file_hash(Path(__file__))!=spec['worker_sha256']:raise ValueError('Worker changed during inference')
        result.update(status='unverified_source_complete',complete=True,frozen_inputs_unchanged=True)
    except Exception as exc:result.update(status='incomplete',complete=False,error=type(exc).__name__+': '+str(exc))
    finally:save()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    result = run(_json(args.spec.read_text(encoding='utf-8')))
    print(json.dumps({'complete': result['complete'], 'window_count': result['window_count']}))
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
