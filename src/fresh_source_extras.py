"""Fresh, source-only Qwen/Voxtral observations for arbitrary owner counts.

This worker never manages the shared backend. Its parent must unload/restore
that backend and join this short-lived process. Historical native capture
helpers are reused with their exact policies, without claiming that a new
episode satisfies the old fixed-66-owner experiment contracts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

from src.contextual_review_contract import require, strict_json
from src import contextual_asr_native as qwen_base
from src import auto_source_native as automatic
from src import forced_source_native as forced
from src import late_audio
from src import qwen_audio_attention as attention
from src.workflow_state import write_json

VERSION = 'fresh-source-extras-2'
ACCEPTED_VERSIONS = ('fresh-source-extras-1', VERSION)
FAILURE_POLICY = {
    'version': 'fresh-source-missing-observation-1',
    'known_native_output_truncation_or_early_stop': 'retain_unavailable_and_continue_independent_owners',
    'negative_control_error': 'stop', 'systemic_or_deadline_error': 'stop',
    'retries': 0, 'truncated_text_salvaged': False,
    'continuation': 'authenticated_unattempted_owners_only', 'maximum_continuation_depth': 1,
}
RATE = 16000
METHODS = ('auto', 'forced', 'masked', 'voxtral')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pin(path: Path | str) -> dict:
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': late_audio.file_hash(path)}


def read(path: Path | str):
    return strict_json(Path(path).read_bytes())


def owner_geometry(owners: list[dict], frames: int) -> list[dict]:
    """Validate the complete continuous physical timeline, retaining silence."""
    require(type(frames) is int and frames > 0 and type(owners) is list and owners,
            'Missing complete acoustic timeline')
    result = []
    previous = 0
    for number, row in enumerate(owners, 1):
        require(type(row) is dict and set(row) == {'id', 'start', 'end'}
            and type(row['id']) is int and row['id'] == number,
            'Owners must be contiguous IDs with source-only geometry')
        require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in ('start', 'end')),
                'Nonfinite owner timestamp')
        start, end = (round(row[k]*RATE) for k in ('start', 'end'))
        require(start == previous and 0 < end-start <= 30*RATE and start < frames,
                'Owner timeline has a gap, overlap, or overlong window')
        # Millisecond SRT endpoints can round at most eight 16-kHz frames.
        physical_end = frames if number == len(owners) and abs(end-frames) <= 8 else min(end, frames)
        require(end <= frames or number == len(owners) and end-frames <= 8,
                'Owner extends beyond physical PCM')
        result.append({'owner_id': number, 'start': start/RATE, 'end': physical_end/RATE,
            'start_frame': start, 'end_frame': physical_end})
        previous = end
    require(result[-1]['end_frame'] == frames, 'Owner timeline omits the end of the recording')
    return result


def vad_overlaps(geometry: list[dict], speech: list[dict], frames: int) -> dict[int, float]:
    previous = 0
    require(type(speech) is list, 'VAD intervals must be a list')
    for row in speech:
        require(type(row) is dict and set(row) == {'start', 'end'}
            and all(type(v) is int for v in row.values())
            and previous <= row['start'] < row['end'] <= frames, 'Invalid VAD intervals')
        previous = row['end']
    return {row['owner_id']: sum(max(0, min(row['end_frame'], s['end'])
        - max(row['start_frame'], s['start'])) for s in speech)/RATE for row in geometry}


def eligible_owner_ids(observations: list[dict]) -> list[int]:
    """The original forced-language predicate applied to fresh observations."""
    require(type(observations) is list, 'Automatic observations must be a list')
    seen = set()
    result = []
    for row in observations:
        require(type(row) is dict and type(row.get('owner_id')) is int and row['owner_id'] > 0
            and row['owner_id'] not in seen and type(row.get('text')) is str
            and row.get('status') in ('ok', 'complete', 'empty', 'unavailable'), 'Invalid automatic observation')
        seen.add(row['owner_id'])
        language = row.get('detected_language')
        overlap = row.get('vad_overlap_seconds')
        require(language is None or type(language) is str, 'Invalid detected language')
        require(type(overlap) in (int, float) and math.isfinite(overlap) and overlap >= 0,
                'Invalid VAD overlap')
        if row['status'] in ('ok', 'complete') and row['text'] != '' and language not in (None, '', 'Japanese') and overlap > 0:
            result.append(row['owner_id'])
    return sorted(result)


def validate_spec(spec: dict) -> dict:
    required = {'method', 'mono_path', 'owners', 'output_dir', 'model_dir'}
    optional = {'version', 'vad_speech', 'eligible_ids', 'automatic_evidence', 'worker_seconds', 'resume_from'}
    require(type(spec) is dict and required <= set(spec) <= required | optional, 'Invalid worker spec fields')
    require(spec.get('version', VERSION) == VERSION and spec['method'] in METHODS, 'Unknown fresh acquisition method')
    if 'resume_from' in spec:
        value = spec['resume_from']
        require(type(value) is str and Path(value).is_absolute() or type(value) is dict
            and set(value) == {'path', 'sha256'} and type(value['path']) is str and Path(value['path']).is_absolute(),
            'Continuation requires an absolute prior evidence path or pin')
    for key in ('mono_path', 'output_dir', 'model_dir'):
        require(type(spec[key]) is str and Path(spec[key]).is_absolute(), 'Absolute paths required')
    budget = spec.get('worker_seconds', 1800)
    require(type(budget) in (int, float) and math.isfinite(budget) and budget > 0, 'Invalid worker time budget')
    if spec['method'] == 'forced':
        require(type(spec.get('automatic_evidence')) is str and Path(spec['automatic_evidence']).is_absolute(),
                'Forced selection requires fresh automatic evidence')
        if 'eligible_ids' in spec:
            ids = spec['eligible_ids']
            require(type(ids) is list and all(type(i) is int and i > 0 for i in ids)
                and ids == sorted(set(ids)), 'Forced IDs must be unique sorted positive integers')
    else:
        require('eligible_ids' not in spec and 'automatic_evidence' not in spec,
                'Only forced acquisition accepts a selected subset')
    return deepcopy(spec)


def qwen_assets(model_dir: Path) -> dict:
    distribution = importlib.metadata.distribution('qwen-asr')
    model = late_audio._model_pins(model_dir, late_audio.ENGINES[1])
    require(model.get('available') is True, 'Qwen model assets unavailable')
    names = {'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
             'added_tokens.json', 'vocab.json', 'merges.txt', 'chat_template.json'}
    assets = {'model': model, 'runtime': late_audio._runtime_identity(),
        'parser': pin(distribution.locate_file('qwen_asr/inference/utils.py')),
        'api': pin(distribution.locate_file('qwen_asr/inference/qwen3_asr.py')),
        'tokenizer_files': [p for p in model['files'] if Path(p['path']).name in names],
        'interpreter': late_audio._interpreter_path(sys.executable)}
    qwen_base.validate_assets(assets, qwen_base.execution_identity(assets), runtime_check=True)
    return assets


def _write_audio(path: Path, array) -> dict:
    import soundfile as sf
    sf.write(path, array, RATE, subtype='FLOAT', format='WAV')
    binding = {**pin(path), 'pcm_sha256': hashlib.sha256(array.tobytes()).hexdigest(),
               'frames': len(array), 'sample_rate': RATE, 'dtype': 'float32'}
    require(automatic._read_audio(binding).tobytes() == array.tobytes(), 'FLOAT audio roundtrip changed')
    return binding


def prepare_inputs(spec: dict, folder: Path) -> dict:
    """CPU-only preparation; no source or target transcript is a model input."""
    import numpy as np
    import soundfile as sf
    mono = pin(spec['mono_path'])
    audio, rate = sf.read(mono['path'], dtype='float32', always_2d=False)
    require(rate == RATE and audio.ndim == 1 and len(audio) > 0 and np.isfinite(audio).all(),
            'Expected finite 16kHz mono PCM')
    geometry = owner_geometry(spec['owners'], len(audio))
    if 'vad_speech' in spec:
        speech = spec['vad_speech']
    else:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        speech = [{k: int(v) for k, v in row.items()} for row in get_speech_timestamps(
            audio, VadOptions(**late_audio.VAD_OPTIONS), sampling_rate=RATE)]
    overlaps = vad_overlaps(geometry, speech, len(audio))
    eligible = [r['owner_id'] for r in geometry]
    parent = None
    if spec['method'] == 'forced':
        parent = pin(spec['automatic_evidence'])
        evidence = read(parent['path'])
        require(evidence.get('version') in ACCEPTED_VERSIONS and evidence.get('method') == 'auto'
            and evidence.get('complete') is True and evidence.get('mono') == mono
            and evidence.get('geometry') == geometry, 'Forced evidence is not complete automatic evidence for this recording')
        observations = evidence['observations']
        require([r['owner_id'] for r in observations] == eligible, 'Automatic source omits owners')
        require(all(row['vad_overlap_seconds'] == overlaps[row['owner_id']] for row in observations),
                'Forced VAD selection differs from automatic acquisition')
        eligible = eligible_owner_ids(observations)
        require('eligible_ids' not in spec or spec['eligible_ids'] == eligible, 'Supplied forced subset violates fresh eligibility')
    peak = float(np.max(np.abs(audio)))
    gain = min(1.0, .99/peak) if peak else 1.0
    directory = folder/'audio'
    directory.mkdir()
    slots = []
    if spec['method'] != 'forced':
        controls = {'silence': np.zeros(12*RATE, dtype=np.float32),
            'noise': np.random.Generator(np.random.PCG64(20260915)).uniform(-.01, .01, 12*RATE).astype(np.float32)}
        for name, pcm in controls.items():
            slots.append({'slot_id': name, 'kind': 'control', 'owner_id': None,
                          'audio': _write_audio(directory/(name+'.wav'), pcm)})
    for row in geometry:
        if row['owner_id'] not in eligible:
            continue
        name = f"owner-{row['owner_id']:04d}"
        pcm = np.multiply(audio[row['start_frame']:row['end_frame']], np.float32(gain), dtype=np.float32)
        slots.append({'slot_id': name, 'kind': 'real', **row, 'vad_overlap_seconds': overlaps[row['owner_id']],
                      'audio': _write_audio(directory/(name+'.wav'), pcm)})
    require(pin(spec['mono_path']) == mono, 'Original PCM changed during preparation')
    result = {'mono': mono, 'geometry': geometry, 'vad_speech': speech, 'eligible_ids': eligible,
        'automatic_evidence': parent, 'gain': gain, 'gain_float32': float(np.float32(gain)),
        'global_peak': peak, 'slots': slots, 'source_accuracy_verified': False}
    write_json(folder/'preparation.json', result)
    return result


def build_request(method: str, slot: dict, assets: dict, preparation: dict) -> dict:
    if method == 'voxtral':
        from src import voxtral_source_native as voxtral
        return {'version': voxtral.REQUEST_VERSION, 'audio': deepcopy(slot['audio']),
            'conditioning': deepcopy(voxtral.CONDITIONING), 'processing': deepcopy(voxtral.PROCESSING),
            'generation': deepcopy(voxtral.GENERATION), 'runtime_policy': deepcopy(voxtral.RUNTIME_POLICY),
            'execution_identity': voxtral.execution_identity(assets)}
    adapter = forced if method == 'forced' else automatic
    return {'version': adapter.REQUEST_VERSION, 'owner_id': slot['owner_id'],
        'audio': deepcopy(slot['audio']), 'language': 'Japanese' if method == 'forced' else None,
        'conditioning': deepcopy(adapter.CONDITIONING), 'policy': deepcopy(qwen_base.requests.POLICY),
        'execution_identity': adapter.execution_identity(assets), 'fresh_preparation_sha256': digest(preparation)}


def recoverable_observation_error(exc: BaseException) -> bool:
    """A failed independent recognition may be absent; broken execution stops."""
    # Native contract failures are terminal, including unrecognized ValueErrors.
    # These exact messages describe a single decoder stopping before completion.
    return type(exc) is ValueError and str(exc) in {
        'Incomplete or truncated native call',
        'Native output did not end normally',
        'Native generation stopped without EOS or audio clock completion',
        'Native EOS stopped before complete physical audio consumption',
    }


def _checked_pin(value: dict) -> Path:
    require(type(value) is dict and set(value) == {'path', 'sha256'} and pin(value['path']) == value,
            'Inherited artifact hash changed')
    return Path(value['path'])


def _check_tensor_artifacts(value, folder: Path) -> None:
    """Authenticate every recursively referenced native tensor without a model."""
    if isinstance(value, dict):
        if {'artifact', 'bytes_sha256', 'dtype', 'shape'} <= set(value):
            artifact = Path(value['artifact'])
            require(not artifact.is_absolute() and '..' not in artifact.parts, 'Invalid inherited tensor path')
            require(late_audio.file_hash(folder/artifact) == value['bytes_sha256'], 'Inherited native tensor changed')
        for child in value.values():
            _check_tensor_artifacts(child, folder)
    elif isinstance(value, list):
        for child in value:
            _check_tensor_artifacts(child, folder)


def continuation_inputs(spec: dict, preparation: dict) -> dict | None:
    """Authenticate the previous attempt before allowing any new owner call.

    Receipt and tensor replay here establishes immutable native provenance; the
    original capture's transcript remains fallible source evidence.
    """
    if 'resume_from' not in spec:
        return None
    supplied = spec['resume_from']
    source_pin = pin(supplied) if isinstance(supplied, str) else supplied
    path = _checked_pin(source_pin)
    prior = read(path)
    require(prior.get('version') in ACCEPTED_VERSIONS and prior.get('method') == spec['method']
        and prior.get('status') == 'failed' and prior.get('complete') is False
        and prior.get('finished_utc') is not None, 'Only a terminal incomplete acquisition can continue')
    require(prior.get('continuation') is None, 'Chained continuation is unsupported; preserve all original attempts')
    old_spec = read(_checked_pin(prior['spec']))
    require(old_spec['method'] == spec['method'] and Path(old_spec['model_dir']).resolve() == Path(spec['model_dir']).resolve()
        and old_spec['owners'] == spec['owners'] and Path(old_spec['mono_path']).resolve() == Path(spec['mono_path']).resolve()
        and Path(old_spec['output_dir']).resolve() == path.parent.resolve(), 'Continuation changed original request')
    old_preparation = read(_checked_pin(prior['preparation']))
    require(prior['mono'] == preparation['mono'] and prior['geometry'] == preparation['geometry']
        and prior['eligible_ids'] == preparation['eligible_ids'], 'Continuation changed recording or owner coverage')
    for key in ('mono', 'geometry', 'vad_speech', 'eligible_ids', 'automatic_evidence', 'gain', 'gain_float32', 'global_peak'):
        require(old_preparation[key] == preparation[key], 'Continuation changed source preparation: '+key)
    _checked_pin(prior['mono'])
    assets_path = _checked_pin(prior['assets'])
    assets = read(assets_path)
    groups = (assets.get('model', {}).get('files', []), assets.get('runtime', {}).get('files', []),
              assets.get('model_files', []), assets.get('runtime_files', []))
    require(any(groups), 'Prior model/runtime identity missing')
    for group in groups:
        for item in group:
            _checked_pin(item)
    # A saved old worker is accepted only at the hash already frozen in state.
    snapshots = path.parent/'failed-code-snapshots/manifest.json'
    saved = read(snapshots) if snapshots.is_file() else []
    require(type(saved) is list, 'Invalid old code snapshot manifest')
    snapshot_pins = []
    for code in prior['code_pins']:
        if pin(code['path']) == code:
            continue
        match = [item for item in saved if item.get('original_path') == code['path'] and item.get('sha256') == code['sha256']]
        require(len(match) == 1, 'Original acquisition code changed without an authenticated snapshot')
        snapshot = Path(match[0]['snapshot'])
        if not snapshot.is_absolute():
            snapshot = Path(__file__).resolve().parents[1]/snapshot
        require(late_audio.file_hash(snapshot) == code['sha256'], 'Old worker snapshot differs from frozen hash')
        snapshot_pins.append(pin(snapshot))
    require(prior['worker'] in prior['code_pins'], 'Original worker was not pinned')
    observations = prior['observations']
    ids = [row['owner_id'] for row in observations]
    require(ids == preparation['eligible_ids'][:len(ids)], 'Prior real observations are not an exact attempted prefix')
    expected_controls = [] if spec['method'] == 'forced' else ['silence', 'noise']
    require([c['slot_id'] for c in prior['controls']] == expected_controls
        and all(c['passed'] is True for c in prior['controls']), 'Prior negative controls did not pass')
    prior_slots = {s['slot_id']: s for s in old_preparation['slots']}
    expected_receipts = set()
    for slot_id, record, control in [(r['slot_id'], r, True) for r in prior['controls']] + [
            (f"owner-{r['owner_id']:04d}", r, False) for r in observations]:
        receipt_pin = {'path': record['path'], 'sha256': record['sha256']} if control else {
            'path': record['receipt_path'], 'sha256': record['receipt_sha256']}
        receipt_path = _checked_pin(receipt_pin)
        require(receipt_path.parent == path.parent/'receipts' and receipt_path.name == slot_id+'.json',
                'Prior receipt belongs to another acquisition')
        receipt = read(receipt_path)
        reservation = read(receipt_path.with_suffix('.started.json'))
        expected_receipts.update((receipt_path, receipt_path.with_suffix('.started.json')))
        require(receipt['version'] in ACCEPTED_VERSIONS and receipt['method'] == spec['method']
            and receipt['slot'] == prior_slots[slot_id]
            and receipt['request_sha256'] == digest(receipt['request'])
            and receipt['assets_sha256'] == digest(assets)
            and reservation == {'request_sha256': receipt['request_sha256'], 'started_utc': receipt['started_utc']}
            and receipt['finished_utc'] is not None, 'Prior native receipt/request changed')
        audio = receipt['request']['audio']
        _checked_pin({'path': audio['path'], 'sha256': audio['sha256']})
        automatic._read_audio(audio)
        require(audio == receipt['slot']['audio'], 'Prior native audio binding changed')
        _check_tensor_artifacts(receipt['capture'], path.parent)
        if control:
            require(receipt['status'] == 'empty' and receipt['error'] is None and receipt['result']['text'] == '',
                    'Prior negative control was not empty')
        else:
            require(record['status'] == receipt['status'] and record['start'] == receipt['slot']['start']
                and record['end'] == receipt['slot']['end']
                and record['vad_overlap_seconds'] == receipt['slot']['vad_overlap_seconds'], 'Prior observation scope/status changed')
            if receipt['status'] == 'unavailable':
                require(record['text'] == '' and record['detected_language'] is None
                    and receipt['result'] is None and type(receipt['error']) is str, 'Failed native text must remain unavailable')
            else:
                require(receipt['status'] in ('ok', 'empty') and receipt['error'] is None
                    and record['text'] == receipt['result']['text']
                    and record['detected_language'] == receipt['result']['detected_language'], 'Prior native text changed')
    require(set((path.parent/'receipts').glob('*.json')) == expected_receipts,
            'Unaccounted prior native attempt; refusing any possible reroll')
    require(prior['asr_calls'] == len(observations)+len(prior['controls']), 'Prior call ledger differs from attempted owners')
    require(pin(path) == source_pin, 'Prior acquisition changed during authentication')
    return {'evidence': source_pin, 'spec': prior['spec'], 'preparation': prior['preparation'],
        'assets': prior['assets'], 'code_snapshots': snapshot_pins,
        'observations': deepcopy(observations), 'controls': deepcopy(prior['controls']),
        'inherited_calls': prior['asr_calls'], 'inherited_seconds': prior.get('total_seconds', prior['seconds']),
        'inherited_model_loads': prior.get('total_model_loads', prior['model_loads']),
        'remaining_owner_ids': preparation['eligible_ids'][len(observations):]}


def validate_qwen_capture(capture: dict, request: dict, assets: dict, folder: Path, processor,
                          *, language: str | None) -> dict:
    """Replay native PCM, processor, decoder and parser under the fresh contract."""
    import numpy as np
    c = capture
    require(set(c) == qwen_base._CAPTURE_KEYS, 'Incomplete native capture')
    for key in ('public_calls', 'normalization_calls', 'split_calls', 'infer_calls',
                'outer_calls', 'inner_calls', 'processor_calls', 'decode_calls'):
        require(type(c[key]) is int and c[key] == 1, 'Native call repeated or missing')
    audio = automatic._read_audio(request['audio'])
    for key in ('input_pcm', 'normalized_pcm'):
        require(qwen_base._load_tensor(c[key], folder).tobytes() == audio.tobytes(), 'Captured input PCM changed')
    require(c['input_pcm'] == c['normalized_pcm'] and c['chunk'] == {
        'count': 1, 'offset_seconds': 0, 'sample_rate': RATE, 'pcm': c['input_pcm']}, 'Native chunk changed')
    require(c['contexts'] == [''] and c['languages'] == [language], 'Native conditioning changed')
    prompt = processor.apply_chat_template([
        {'role': 'system', 'content': ''}, {'role': 'user', 'content': [{'type': 'audio', 'audio': ''}]}],
        add_generation_prompt=True, tokenize=False)
    if language is not None:
        prompt += 'language Japanese<asr_text>'
    require(c['prompt'] == prompt and c['hint_token_ids'] == processor.tokenizer.encode('', add_special_tokens=False),
            'Native prompt or hint changed')
    inputs = processor(text=[prompt], audio=[audio], return_tensors='pt', padding=True)
    require(set(inputs) == qwen_base._TENSOR_KEYS, 'Native processor fields changed')
    for key, value in inputs.items():
        expected = qwen_base._array(value)
        require(np.array_equal(qwen_base._load_tensor(c['processor_inputs'][key], folder), expected),
                'Native processor tensor changed')
        delivered = expected.astype(np.float16) if expected.dtype.kind == 'f' else expected
        for collection in ('outer_inputs', 'inner_inputs'):
            require(np.array_equal(qwen_base._load_tensor(c[collection][key], folder), delivered),
                    'Native delivered tensor changed')
    policy = {k: deepcopy(qwen_base.requests.POLICY[k]) for k in qwen_base._GEN_KEYS}
    require(c['outer_config'] == c['inner_config'] == policy
        and c['outer_kwargs'] == {'max_new_tokens': 512}
        and c['inner_kwargs'] == {'max_new_tokens': 512, 'eos_token_id': [151645, 151643], 'return_dict_in_generate': True},
        'Historical native decoder settings changed')
    input_positions = int(inputs['input_ids'].shape[1])
    expected_capacity = {'input_positions': input_positions,
        'feature_frames': int(qwen_base._array(inputs['feature_attention_mask']).sum()),
        'position_limit': 65536, 'reserved_output': 512, 'margin': 64,
        'hint_tokens': len(c['hint_token_ids']), 'fits': input_positions+512+64 <= 65536}
    require(c['capacity'] == expected_capacity and expected_capacity['fits'] is True,
            'Native capacity differs from actual input tensors')
    generated = c['output_token_ids']
    require(type(generated) is list and 0 < len(generated) < 512
        and all(type(i) is int and i >= 0 for i in generated)
        and generated[-1] in policy['eos_token_id']
        and not any(i in policy['eos_token_id'] for i in generated[:-1]) and c['normal_stop'] is True,
        'Native output did not end normally')
    require(processor.batch_decode([generated], skip_special_tokens=True,
        clean_up_tokenization_spaces=False) == [c['raw_text']], 'Native token decoding changed')
    adapter = forced if language else automatic
    parsed_language, text = adapter.native_parse(c['raw_text'], assets['parser']['path'], assets['parser']['sha256'])
    require(c['native_result'] == {'text': text, 'language': parsed_language, 'time_stamps': None},
            'Native parser result changed')
    require(type(c['parser_changed']) is bool and c['parser_changed'] == (c['raw_text'] != text),
            'Native parser projection changed')
    return {'text': text, 'detected_language': parsed_language, 'raw_text': c['raw_text']}


def validate_voxtral_capture(c: dict, request: dict, assets: dict, folder: Path, processor) -> dict:
    import numpy as np
    import torch
    from src import voxtral_source_native as native
    native.validate_assets(assets, request['execution_identity'])
    audio = native._read_audio(request['audio'])
    require(np.array_equal(native._load(c['input_pcm'], folder), audio), 'Voxtral captured PCM changed')
    inputs, layout = native._frontend(processor, audio)
    for key in ('serialized_pcm', 'padded_pcm'):
        require(np.array_equal(native._load(c['layout'][key], folder), layout[key]), 'Voxtral padding changed')
        layout[key] = c['layout'][key]
    require(c['layout'] == layout and c['prefix_token_ids'] == inputs['input_ids'][0].tolist(), 'Voxtral frontend changed')
    for key in ('input_ids', 'attention_mask', 'input_features'):
        original = inputs[key].detach().cpu().numpy()
        delivered = inputs[key].to(dtype=torch.bfloat16).float().numpy() if inputs[key].is_floating_point() else original
        require(np.array_equal(native._load(c['processor_inputs'][key], folder), original)
            and np.array_equal(native._load(c['generation_inputs'][key], folder), delivered), 'Voxtral input tensor changed')
    native._validate_runtime(c['runtime'])
    from transformers import GenerationConfig
    expected_defaults = GenerationConfig.from_dict(native.model_config(assets)['generation_config']).to_dict()
    require(c['runtime']['generation_config'] == expected_defaults, 'Voxtral generation defaults changed')
    resolved = c['resolved_generation']
    require(c['generation_kwargs'] == native.GENERATION and resolved['max_length'] == layout['native_total_positions']
        and resolved['max_new_tokens'] is None and resolved['eos_token_id'] == 2
        and resolved['pad_token_id'] == 11 and resolved['bos_token_id'] == 1
        and all(resolved[k] == v for k, v in native.GENERATION.items()),
        'Voxtral native generation settings changed')
    decoded = native.decode_tokens(processor, c['output_token_ids'])
    require(all(c[k] == v for k, v in decoded.items()), 'Voxtral native decoded text changed')
    coverage = native._coverage(c, folder)
    require(coverage == c['coverage'] and coverage['physical_audio_complete'] is True, 'Voxtral physical audio incomplete')
    return {'text': c['parsed_text'], 'detected_language': None, 'raw_text': c['raw_text'], 'coverage': coverage}


def load_session(method: str, model_dir: Path):
    """Called once inside the short-lived worker, after CPU input preparation."""
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        os.environ[key] = '1'
    if method == 'voxtral':
        from src import voxtral_source_native as native
        assets = native.build_assets(model_dir)
        model, processor = native._load_model(assets)
        return assets, (model, processor)
    assets = qwen_assets(model_dir)
    if method == 'masked':
        attention.source_identity()
    asr = qwen_base._load_asr(assets)
    qwen_base._check_processor(assets, asr.processor)
    return assets, asr


def capture_one(method: str, session, request: dict, assets: dict, folder: Path,
                deadline: float, save_capture, save_attention) -> dict:
    audio = automatic._read_audio(request['audio'])
    if method == 'voxtral':
        from src import voxtral_source_native as native
        model, processor = session
        c = native.capture_call(model, processor, audio, request, artifact_root=folder,
            save_capture=save_capture, deadline=deadline)
        return validate_voxtral_capture(c, request, assets, folder, processor)
    adapter = forced if method == 'forced' else automatic
    audit = None
    try:
        if method == 'masked':
            with attention.scoped_window_attention(session.model.thinker.audio_tower) as audit:
                c = adapter.capture_call(session, audio, request, artifact_root=folder,
                    save_capture=save_capture, deadline=deadline)
        else:
            c = adapter.capture_call(session, audio, request, artifact_root=folder,
                save_capture=save_capture, deadline=deadline)
    finally:
        save_attention(deepcopy(audit))
    if method == 'masked':
        attention.validate_attestation(audit, feature_frames=c['capacity']['feature_frames'],
            config=attention.encoder_config(session.model.thinker.audio_tower))
    return validate_qwen_capture(c, request, assets, folder, session.processor,
        language='Japanese' if method == 'forced' else None)


def run(spec_path: Path | str) -> dict:
    spec_path = Path(spec_path).resolve(strict=True)
    spec_pin = pin(spec_path)
    spec = validate_spec(read(spec_path))
    folder = Path(spec['output_dir'])
    require(not folder.exists() or not any(folder.iterdir()), 'Fresh worker output already attempted')
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started+spec.get('worker_seconds', 1800)
    state = {'version': VERSION, 'method': spec['method'], 'spec': spec_pin, 'worker': pin(__file__),
        'status': 'preparing', 'complete': False, 'started_utc': now(), 'finished_utc': None,
        'model_loads': 0, 'model_load_attempts': 0, 'asr_calls': 0, 'additional_calls': 0,
        'inherited_calls': 0, 'inherited_seconds': 0., 'inherited_model_loads': 0,
        'observations': [], 'controls': [], 'continuation': None, 'failure_policy': deepcopy(FAILURE_POLICY),
        'source_accuracy_verified': False, 'audio_backend_restoration_owner': 'parent_driver',
        'backend_managed_by_worker': False, 'error': None}
    write_json(folder/'execution-started.json', {'spec': spec_pin, 'started_utc': state['started_utc']})

    def save():
        state['seconds'] = time.monotonic()-started
        state['total_seconds'] = state['seconds']+state['inherited_seconds']
        state['total_model_loads'] = state['model_loads']+state['inherited_model_loads']
        state['counts'] = {key: sum(r['status'] == key for r in state['observations'])
                           for key in ('ok', 'empty', 'unavailable')}
        state['all_observations_available'] = state['complete'] and state['counts']['unavailable'] == 0
        write_json(folder/'evidence.json', state)

    save()
    try:
        preparation = prepare_inputs(spec, folder)
        state.update({k: deepcopy(preparation[k]) for k in ('mono', 'geometry', 'eligible_ids', 'automatic_evidence')})
        state['preparation'] = pin(folder/'preparation.json')
        continuation = continuation_inputs(spec, preparation)
        slots = preparation['slots']
        if continuation is not None:
            state['continuation'] = deepcopy(continuation)
            for key in ('observations', 'controls', 'inherited_calls', 'inherited_seconds', 'inherited_model_loads'):
                state[key] = deepcopy(continuation[key])
            state['asr_calls'] = continuation['inherited_calls']
            slots = [s for s in slots if s['kind'] == 'real' and s['owner_id'] in continuation['remaining_owner_ids']]
            write_json(folder/'continuation.json', continuation)
        if not slots:
            state.update(status='complete' if continuation is not None else 'skipped_no_eligible_owners', complete=True)
            return state
        with qwen_base._deadline(deadline):
            state.update(status='loading', model_load_attempts=1)
            save()
            assets, session = load_session(spec['method'], Path(spec['model_dir']))
            if continuation is not None:
                require(assets == read(_checked_pin(continuation['assets'])), 'Continuation model/runtime assets differ from original')
            write_json(folder/'assets.json', assets)
            code_paths = [__file__, automatic.__file__, forced.__file__, qwen_base.__file__,
                          attention.__file__, late_audio.__file__]
            if spec['method'] == 'voxtral':
                from src import voxtral_source_native
                code_paths.append(voxtral_source_native.__file__)
            state['code_pins'] = [pin(p) for p in code_paths]
            state.update(assets=pin(folder/'assets.json'), model_loads=1, status='running')
            save()
            receipts = folder/'receipts'
            receipts.mkdir()
            for slot in slots:
                request = build_request(spec['method'], slot, assets, preparation)
                path = receipts/(slot['slot_id']+'.json')
                receipt = {'version': VERSION, 'method': spec['method'], 'slot': slot,
                    'request': request, 'request_sha256': digest(request), 'assets_sha256': digest(assets),
                    'started_utc': now(), 'finished_utc': None, 'status': 'running',
                    'capture': {}, 'attention': None, 'result': None, 'error': None}
                write_json(path.with_suffix('.started.json'), {'request_sha256': digest(request), 'started_utc': receipt['started_utc']})
                write_json(path, receipt)
                state['asr_calls'] += 1
                state['additional_calls'] += 1
                save()

                def save_capture(value):
                    receipt['capture'] = value
                    write_json(path, receipt)

                def save_attention(value):
                    receipt['attention'] = value
                    write_json(path, receipt)

                before = time.monotonic()
                try:
                    end = deadline if spec['method'] == 'voxtral' else min(deadline, before+60)
                    result = capture_one(spec['method'], session, request, assets, folder, end, save_capture, save_attention)
                    receipt.update(status='ok' if result['text'] != '' else 'empty', result=result)
                except BaseException as exc:
                    receipt.update(status='unavailable', error=type(exc).__name__+': '+str(exc))
                    if slot['kind'] == 'control' or not recoverable_observation_error(exc):
                        raise
                finally:
                    receipt.update(finished_utc=now(), seconds=time.monotonic()-before)
                    write_json(path, receipt)
                    if slot['kind'] == 'real':
                        result = receipt['result'] or {'text': '', 'detected_language': None}
                        state['observations'].append({'owner_id': slot['owner_id'], 'start': slot['start'], 'end': slot['end'],
                            'observer': 'voxtral-mini-4b-realtime-2602' if spec['method'] == 'voxtral' else 'qwen3-asr-1.7b',
                            'view': spec['method'], 'text': result['text'], 'status': receipt['status'],
                            'detected_language': result['detected_language'], 'language_origin': 'forced' if spec['method'] == 'forced'
                                else 'not_reported' if spec['method'] == 'voxtral' else 'automatic',
                            'vad_overlap_seconds': slot['vad_overlap_seconds'], 'receipt_path': str(path),
                            'receipt_sha256': late_audio.file_hash(path)})
                    else:
                        state['controls'].append({'slot_id': slot['slot_id'], 'passed': receipt['status'] == 'empty', **pin(path)})
                    save()
                require(slot['kind'] != 'control' or receipt['status'] == 'empty', 'Negative control produced text')
        require(pin(spec_path) == spec_pin and pin(spec['mono_path']) == preparation['mono']
            and pin(__file__) == state['worker'], 'Worker inputs changed during inference')
        require(pin(folder/'preparation.json') == state['preparation']
            and pin(folder/'assets.json') == state['assets']
            and all(pin(p['path']) == p for p in state['code_pins']), 'Acquisition code or frozen preparation changed')
        if preparation['automatic_evidence']:
            require(pin(preparation['automatic_evidence']['path']) == preparation['automatic_evidence'], 'Automatic evidence changed')
        if continuation is not None:
            _checked_pin(continuation['evidence'])
        require([r['owner_id'] for r in state['observations']] == preparation['eligible_ids'], 'Real owner coverage incomplete')
        state.update(status='complete_with_unavailable' if any(r['status'] == 'unavailable' for r in state['observations'])
                     else 'complete', complete=True)
    except BaseException as exc:
        state.update(status='failed', complete=False, error=type(exc).__name__+': '+str(exc))
    finally:
        state['finished_utc'] = now()
        save()
        write_json(folder/'worker-lifecycle.json', {'started_utc': state['started_utc'], 'finished_utc': state['finished_utc'],
            'model_load_attempts': state['model_load_attempts'], 'model_loads': state['model_loads'],
            'backend_managed_by_worker': False, 'process_exit_and_backend_restoration_owner': 'parent_driver',
            'complete': state['complete'], 'error': state['error']})
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    result = run(parser.parse_args().spec)
    print(json.dumps({k: result[k] for k in ('status', 'complete', 'asr_calls', 'model_loads', 'error')}))
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
