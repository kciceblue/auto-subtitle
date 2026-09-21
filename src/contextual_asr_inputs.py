"""Read-only C-ASR1 input lineage; never decode audio, tokenize or run a model.

Authenticate the retained original ASR recording and its literal association,
not a fresh reconstruction of native token decoding. Saved VAD provenance is
limited to its preparation, code/runtime pins and original mono ancestry.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

from src import contextual_asr_requests as requests
from src import contextual_evidence_review as review
from src import evidence_context as ec
from src import short_audio
from src.contextual_review_contract import require, strict_json
from src.translate import parse_srt
from src.workflow_state import fingerprint

VERSION = 'contextual-asr-inputs-1'
ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/'archive/experiments-20260914/workspace'
RUN_RELATIVE = Path('docs/benchmarks/quality-loop/anime-whisper-setup/runs/native-anime-whisper-01')
RUN = ARCHIVE/RUN_RELATIVE
SHORT = ROOT/'output/quality-short-20260915/evidence/short'
CONTEXT = ROOT/'output/quality-readable-draft-20260915/original-context.txt'
SHORT_PREPARATION_SHA256 = 'c53aa8568f94b4f268f88eb1c4b16bf65dd9847da036d307b8c014bafdecc729'
SOURCE_SHA256 = '807e6b35f2099421f68ef326813123d8f4a9c958afe179a0e68540641dbf33a1'
CONTEXT_SHA256 = '7ea692a93e6c2a9bc64bab0d7e8c58ddc47b07a268055de25c11d4c85f3bbf21'
MEDIA_SHA256 = '70aee6e71f4e3d4f116ca7a1ff7c2e77f3e485044817e89b1929c41354d3af6e'
ANCHORS = {
    'episode.utterances.ja.srt': '7b6bb80777a084ebe78c1cad390c7ed8f2cf446dd19258cf697e1a688b29daeb',
    'preparation.json': '1f94fe34d5b05e09c72e27e0eacbe0a4d51568f2aa741d652baa55dc453a55b8',
    'run.json': '438dd9812a7c0178dc01ad829547407355f97e7959814465fb3d140d60812382',
    'worker-result.json': '58b4245b03fec4f3846bc730aa6ba0aaf559c830469a280b5f07ecd84a2c1427',
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _same(left: object, right: object) -> bool:
    return fingerprint(left) == fingerprint(right)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class _Inputs:
    def __init__(self) -> None:
        self.pins: dict[str, str] = {}

    def read(self, path: Path, expected: str | None = None) -> bytes:
        path = Path(path).resolve(strict=True)
        raw = path.read_bytes(); digest = _sha(raw)
        require(expected is None or digest == expected, 'Input artifact hash changed')
        require(str(path) not in self.pins or self.pins[str(path)] == digest, 'Input changed during validation')
        self.pins[str(path)] = digest
        return raw

    def pin(self, path: Path, expected: str) -> None:
        path = Path(path).resolve(strict=True)
        require(_file_hash(path) == expected, 'Input artifact hash changed')
        require(str(path) not in self.pins or self.pins[str(path)] == expected, 'Input changed during validation')
        self.pins[str(path)] = expected

    def json(self, path: Path, expected: str | None = None):
        return strict_json(self.read(path, expected))

    def resolve(self, saved: str, expected: str) -> Path:
        """Exact prefix remaps only; no basename/suffix or arbitrary file search."""
        path = Path(saved)
        require(path.is_absolute() and '..' not in path.parts, 'Noncanonical historical path')
        candidates = [path]
        if path.is_relative_to(ROOT):
            relative = path.relative_to(ROOT)
            candidates += [ARCHIVE/relative, RUN/'measured-code'/relative]
        for candidate in dict.fromkeys(candidates):
            if candidate.is_file() and _file_hash(candidate) == expected:
                self.pin(candidate, expected)
                return candidate.resolve()
        raise ValueError('Pinned historical artifact unavailable under exact remaps')

    def finish(self) -> None:
        for path, digest in list(self.pins.items()):
            self.pin(Path(path), digest)


def _unhinted(value: dict) -> None:
    require(value.get('context') == '' and value.get('initial_prompt', 'missing') is None
            and value.get('language') == 'Japanese' and value.get('task') == 'transcribe',
            'Original ASR conditioning is not the pinned empty policy')


def _native_source(inputs: _Inputs) -> tuple[list, dict, dict]:
    prep = inputs.json(RUN/'preparation.json', ANCHORS['preparation.json'])
    run = inputs.json(RUN/'run.json', ANCHORS['run.json'])
    result = inputs.json(RUN/'worker-result.json', ANCHORS['worker-result.json'])
    spec = inputs.json(RUN/'worker-spec.json')
    plan = inputs.json(RUN/'windows.json', spec['plan_sha256'])
    backend = inputs.json(RUN/'backend.json', spec['runtime_identity_sha256'])
    inputs.read(RUN/'worker.py', spec['worker_sha256'])
    require(run['status'] == 'unverified_source_ready_for_separate_translation'
            and run['error'] is None and run['original_qwen_restored'] is True
            and run['frozen_inputs_unchanged'] is True and result['complete'] is True
            and result['status'] == 'unverified_source_complete' and result['error'] is None,
            'Original ASR did not finish its recorded acquisition')
    require(result['plan_sha256'] == spec['plan_sha256']
            and result['runtime_identity_sha256'] == spec['runtime_identity_sha256']
            and result['bundle_identity_sha256'] == spec['bundle_identity_sha256']
            and fingerprint(prep['bundle_identity']) == spec['bundle_identity_sha256']
            and _same(backend['bundle_identity'], prep['bundle_identity']), 'Native acquisition binding differs')
    for record in (prep, prep['bundle_identity'], backend, result):
        _unhinted(record)
    require(all(prep[key] is False for key in ('old_asr_hints_provided', 'prior_generated_context_provided',
            'original_work_context_provided', 'chinese_draft_used', 'external_material_used')),
            'Original ASR includes later or external conditioning')
    # All retained producer/audio inputs are checked. The old model/runtime
    # manifest remains pinned in preparation; its current environment is irrelevant
    # to replaying this saved literal input, and is not loaded or relabelled current.
    for path, digest in prep['input_hashes'].items():
        inputs.resolve(path, digest)
    for name in ('runner.py', 'geometry-helper.py'):
        digest = _sha(inputs.read(RUN/name))
        require(digest in prep['input_hashes'].values(), 'Archived producer was not acquisition-pinned')
    require(plan['video_sha256'] == MEDIA_SHA256 and plan['sample_rate'] == 16000
            and len(plan['windows']) == len(result['rows']) == 66, 'Wrong original media or window coverage')
    inputs.resolve(plan['video'], MEDIA_SHA256)
    inputs.resolve(plan['audio'], plan['audio_sha256'])
    source_path = RUN/'episode.utterances.ja.srt'
    inputs.read(source_path, ANCHORS['episode.utterances.ja.srt'])
    source = parse_srt(source_path, preserve_text_whitespace=True)
    rows = [asdict(row) for row in source]
    document = inputs.json(RUN/'episode.source.json')
    require(len(rows) == 66 and review.owner_rows_sha256(source) == SOURCE_SHA256
            and _same(rows, document['source']), 'Original source literal or geometry changed')
    require(sorted(path.name for path in (RUN/'requests').iterdir()) ==
            [f'window-{i:03d}.json' for i in range(1, 67)], 'Original native request coverage differs')
    batches = {item['number']: item for item in result['batches']}
    require(len(batches) == len(result['batches']) and sorted(batches) == list(range(1, len(batches)+1)),
            'Duplicate or missing original batch')
    seen = set(); attempts = 0; retried = 0
    for index, (packet, accepted, row) in enumerate(zip(plan['windows'], result['rows'], rows), 1):
        require(packet['number'] == accepted['window'] == row['index'] == index
                and accepted['start_frame'] == packet['start_frame']
                and accepted['end_frame'] == packet['end_frame']
                and accepted['audio_sha256'] == packet['sha256'], 'Original window association changed')
        path = RUN/'requests'/f'window-{index:03d}.json'
        require(inputs.resolve(accepted['request_path'], accepted['request_sha256']) == path.resolve(),
                'Original request points outside its recorded run')
        request = inputs.json(path, accepted['request_sha256']); _unhinted(request)
        require(_same(request['window'], packet) and request['bundle_identity_sha256'] == spec['bundle_identity_sha256']
                and request['runtime_identity_sha256'] == spec['runtime_identity_sha256'], 'Original request binding changed')
        history = request['attempts']; attempts += len(history); retried += len(history) > 1
        require(type(request['max_attempts']) is int and request['max_attempts'] == spec['max_attempts']
                and 1 <= len(history) <= request['max_attempts'] <= 3, 'Invalid historical retry ledger')
        for ordinal, attempt in enumerate(history, 1):
            require(attempt['attempt'] == ordinal and attempt['batch'] in batches, 'Original attempt order differs')
            batch = batches[attempt['batch']]; numbers = batch['windows']
            require(numbers.count(index) == 1 and _same(attempt['batch_windows'], numbers), 'Batch/window association differs')
            position = numbers.index(index); seen.add((batch['number'], index))
            for key, batch_key in (('batch_seed', 'seed'), ('settings', 'settings'),
                    ('seconds_batch_shared', 'seconds'), ('decoder_identity', 'decoder_identity')):
                require(_same(attempt[key], batch[batch_key]), 'Native batch receipt differs')
            for key, batch_key in (('token_ids', 'token_ids'), ('raw_text', 'raw_outputs'), ('native_result', 'native_results')):
                expected = batch[batch_key][position] if isinstance(batch[batch_key], list) else None
                require(_same(attempt[key], expected), 'Native attempt content differs from batch')
            require(attempt['settings']['condition_on_prev_tokens'] is False
                    and attempt['settings']['language'] == 'ja' and attempt['settings']['task'] == 'transcribe',
                    'Historical decoder conditioning changed')
            if ordinal < len(history):
                require(attempt['error'] is not None, 'Successful historical output was rerolled')
        final = history[-1]; checked = final['checked']
        require(final['error'] is None and checked['accepted'] is True
                and checked['worker_text_edits'] is False and final['native_result']['text'] == checked['text']
                and checked['text'] == accepted['text'] == row['text']
                and checked['accepted_text_sha256'] == accepted['accepted_text_sha256'] == _sha(row['text'].encode('utf-8'))
                and checked['native_raw_sha256'] == _sha(final['raw_text'].encode('utf-8'))
                and checked['native_token_ids_sha256'] == accepted['native_token_ids_sha256'] == fingerprint(final['token_ids']),
                'Saved native literal/hash association changed')
    require(seen == {(batch['number'], number) for batch in batches.values() for number in batch['windows']},
            'Unaccounted original generation batch/window')
    require(plan['media_frames'] == plan['windows'][-1]['end_frame']
            and plan['windows'][0]['start_frame'] == 0
            and all(a['end_frame'] == b['start_frame'] for a, b in zip(plan['windows'], plan['windows'][1:])),
            'Original acquisition geometry has a gap or overlap')
    bias = {'context': '', 'initial_prompt': None, 'condition_on_prev_tokens': False,
            'model_identity_sha256': spec['bundle_identity_sha256'], 'max_attempts': spec['max_attempts'],
            'repetition_guard': prep['repetition_guard'], 'attempt_settings': [b['settings'] for b in result['batches']]}
    return rows, plan, {'bias': bias, 'attempts': attempts, 'retried_windows': retried,
                       'batches': len(batches), 'source_file_sha256': inputs.pins[str(source_path.resolve())]}


def load_inputs() -> dict:
    """Authenticate fixed inputs and return a capsule; caller charges all elapsed time.

    No files are written. Every returned pin is rechecked before return. The caller
    must persist this capsule and recheck its pins before crop creation/dispatch.
    Actual real-crop PCM equality and controls remain the runner/native responsibility.
    """
    inputs = _Inputs(); rows, original_plan, native = _native_source(inputs)
    acquisition = fingerprint(inputs.pins)
    context = inputs.read(CONTEXT, CONTEXT_SHA256).decode('utf-8', errors='strict')
    saved_short = inputs.json(SHORT/'preparation.json', SHORT_PREPARATION_SHA256)
    short = short_audio.validate_preparation(SHORT)
    require(_same(short, saved_short), 'Short preparation changed during validation')
    old = inputs.json(Path(short['original_preparation']['path']), short['original_preparation']['sha256'])
    start = inputs.json(SHORT/'preparation-started.json')
    require(_same(start['original_preparation'], short['original_preparation'])
            and _same(start['policy'], short['short_policy'])
            and _same(short['master'], old['master']) and _same(short['mono'], old['mono'])
            and short['master']['media']['sha256'] == original_plan['video_sha256'] == MEDIA_SHA256,
            'Saved VAD preparation does not inherit the original media/mono')
    for pin in [short['mono'], short['master']['media'], short['master']['master'], *short['code_pins']]:
        inputs.pin(Path(pin['path']), pin['sha256'])
    runtime = short['runtime']; selected = {}
    for basename in ('silero_vad_v6.onnx', 'vad.py'):
        matches = [pin for pin in runtime['files'] if Path(pin['path']).name == basename]
        require(len(matches) == 1, 'Missing or ambiguous saved VAD dependency pin')
        selected[basename] = matches[0]; inputs.pin(Path(matches[0]['path']), matches[0]['sha256'])
    require(short['mono']['sample_rate'] == 16000 and short['mono']['channels'] == 1
            and short['mono']['frames'] == original_plan['media_frames'], 'Saved detector domain changed')
    detector = {'mono': {key: short['mono'][key] for key in ('frames', 'sample_rate')},
                'vad_speech': short['vad_speech'], 'vad_options': short['vad_options']}
    provenance = {'short_preparation_sha256': inputs.pins[str((SHORT/'preparation.json').resolve())],
        'original_preparation_sha256': short['original_preparation']['sha256'],
        'mono_sha256': short['mono']['sha256'], 'vad_model_sha256': selected['silero_vad_v6.onnx']['sha256'],
        'vad_implementation_sha256': selected['vad.py']['sha256'],
        'runtime_metadata_sha256': fingerprint(runtime), 'detector_record_sha256': fingerprint(detector)}
    source_provenance = {'source_file_sha256': native['source_file_sha256'], 'context_file_sha256': CONTEXT_SHA256,
        'source_rows_sha256': fingerprint(rows), 'context_text_sha256': _sha(context.encode('utf-8')),
        'acquisition_sha256': acquisition, 'bias_status': 'pinned', 'bias_policy_sha256': fingerprint(native['bias'])}
    requests.activity._check_record(detector, provenance)
    inputs.finish()
    return {'source_rows': rows, 'original_context': context, 'source_provenance': source_provenance,
        'detector_record': detector, 'detector_provenance': provenance, 'mono_path': short['mono']['path'],
        'input_hashes': dict(inputs.pins), 'numeric_audit': {'version': VERSION, 'source_owners': len(rows),
            'canonical_source_sha256': SOURCE_SHA256, 'original_native_attempts': native['attempts'],
            'original_native_batches': native['batches'], 'original_retried_windows': native['retried_windows'],
            'bias_policy': native['bias'], 'vad_intervals': len(detector['vad_speech']),
            'mono_frames': short['mono']['frames'], 'sample_rate': 16000,
            'native_token_decoding_reexecuted': False, 'historical_model_runtime_reloaded': False,
            'vad_execution_independently_verified': False, 'frame_probabilities_available': False,
            'waveform_decoded': False, 'crop_pcm_verified': False, 'acoustic_accuracy_verified': False}}
