"""One offline C-ASR1 worker; import/replay never loads an acoustic model.

The runner owns GPU admission, the sole child process and backend restoration.
This worker owns twenty predetermined slots, transparent native capture, a
single model load and terminal failure/no-retry behavior. No translation occurs.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src import contextual_asr_requests as requests
from src import evidence_context as ec
from src import late_audio
from src.contextual_review_contract import require, strict_json
from src.workflow_state import write_json

VERSION = 'contextual-asr-worker-1'
MODEL_DIR = Path('/home/kciceblue/HF/asr-models/Qwen3-ASR-1.7B')
_GEN_KEYS = ('max_new_tokens', 'do_sample', 'temperature', 'top_p', 'top_k', 'num_beams',
             'num_return_sequences', 'repetition_penalty', 'eos_token_id', 'pad_token_id')
_TENSOR_KEYS = {'input_ids', 'attention_mask', 'input_features', 'feature_attention_mask'}
_SPEC_KEYS = {'version', 'plan_path', 'plan_file_sha256', 'output_dir', 'assets',
              'worker_seconds', 'work_deadline_utc', 'worker_sha256'}
_ASSET_KEYS = {'model', 'runtime', 'parser', 'api', 'tokenizer_files', 'interpreter'}
_CAPTURE_KEYS = {'public_calls', 'normalization_calls', 'split_calls', 'infer_calls', 'outer_calls',
    'inner_calls', 'processor_calls', 'decode_calls', 'input_pcm', 'normalized_pcm', 'chunk',
    'contexts', 'languages', 'prompt', 'hint_token_ids', 'processor_inputs', 'outer_inputs',
    'inner_inputs', 'outer_config', 'inner_config', 'outer_kwargs', 'inner_kwargs',
    'capacity', 'capacity_finished_utc', 'generation_started_utc', 'generation_finished_utc',
    'output_token_ids', 'raw_text', 'native_result', 'parser_changed', 'normal_stop'}


class WorkerDeadline(BaseException):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _same(a, b):
    return ec._hash(a) == ec._hash(b)


def _time(value):
    require(isinstance(value, str), 'Missing native timestamp')
    parsed = datetime.fromisoformat(value)
    require(parsed.utcoffset() is not None, 'Native timestamp lacks timezone')
    return parsed


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _stat(path):
    value = Path(path).stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


@lru_cache(maxsize=512)
def _verified_file(path, stamp):
    digest = late_audio.file_hash(path)
    require(_stat(path) == stamp, 'Native asset changed during hashing')
    return digest


def _file(path):
    path = Path(path).resolve(strict=True)
    return _verified_file(path, _stat(path))


def _pin(pin):
    ec._keys(pin, {'path', 'sha256'}); requests._hash(pin['sha256'])
    path = Path(pin['path'])
    require(path.is_absolute() and _file(path) == pin['sha256'], 'Native file pin changed')
    return path


def _read(path):
    return strict_json(Path(path).read_bytes())


def _save(path, value):
    value['receipt_sha256'] = ec._hash({k: v for k, v in value.items() if k != 'receipt_sha256'})
    write_json(Path(path), value)


def _sealed(value):
    require(isinstance(value, dict) and value.get('receipt_sha256') == ec._hash(
        {k: v for k, v in value.items() if k != 'receipt_sha256'}), 'Native receipt seal changed')
    return value


def native_parse(raw, parser_path, parser_sha256):
    """Run only two hash-pinned upstream pure parser definitions, without torch."""
    require(type(raw) is str, 'Raw ASR output must be a string')
    code = Path(parser_path).read_bytes(); require(_sha(code) == parser_sha256, 'Native parser changed')
    tree = ast.parse(code)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('detect_and_fix_repetitions', 'parse_asr_output')]
    require(len(functions) == 2, 'Pinned native parser definitions missing')
    namespace = {'Optional': Optional, 'Tuple': Tuple}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(parser_path), 'exec'), namespace)
    return namespace['parse_asr_output'](raw, user_language='Japanese')


def _reserved(model):
    cfg = _read(Path(model['model_dir'])/'tokenizer_config.json')
    tokens = {row['content'] for row in cfg.get('added_tokens_decoder', {}).values()
              if isinstance(row, dict) and isinstance(row.get('content'), str) and row['content']}
    for key in ('bos_token', 'eos_token', 'pad_token', 'unk_token'):
        value = cfg.get(key)
        if isinstance(value, dict): value = value.get('content')
        if isinstance(value, str) and value: tokens.add(value)
    require(bool(tokens), 'Native reserved token inventory missing')
    return sorted(tokens)


def _execution_identity(assets):
    return {'model_identity_sha256': ec._hash(assets['model']),
        'runtime_identity_sha256': ec._hash(assets['runtime']), 'worker_sha256': _file(__file__),
        'parser_sha256': assets['parser']['sha256'], 'tokenizer_sha256': ec._hash(assets['tokenizer_files']),
        'reserved_tokens': _reserved(assets['model'])}


def execution_identity(assets):
    """CPU-only identity construction; caller must still authenticate lineage."""
    ec._keys(assets, _ASSET_KEYS)
    return _execution_identity(assets)


def _check_processor(assets, processor):
    reserved = set(_reserved(assets['model']))
    require(set(processor.tokenizer.all_special_tokens) <= reserved
        and set(processor.tokenizer.get_added_vocab()) <= reserved, 'Processor has unpinned reserved/added spellings')


def load_processor(assets):
    """Load only the local CPU tokenizer/feature extractor, never an ASR model."""
    validate_assets(assets, execution_identity(assets), runtime_check=True)
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        os.environ[key] = '1'
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor
    processor = Qwen3ASRProcessor.from_pretrained(assets['model']['model_dir'], local_files_only=True,
                                                 fix_mistral_regex=True)
    _check_processor(assets, processor)
    return processor


def validate_assets(assets, execution_identity, *, runtime_check=False):
    ec._keys(assets, _ASSET_KEYS)
    model = assets['model']; ec._keys(model, {'available', 'model_dir', 'files'})
    require(model['available'] is True and Path(model['model_dir']).resolve() == MODEL_DIR.resolve(), 'Wrong local ASR model')
    require(isinstance(model['files'], list) and bool(model['files']), 'Missing local ASR files')
    paths = [_pin(row) for row in model['files']]
    require(len(paths) == len(set(paths)) and all(path.parent == MODEL_DIR.resolve() for path in paths), 'ASR model file coverage changed')
    actual = sorted(p.resolve() for p in MODEL_DIR.rglob('*') if p.is_file() and '.cache' not in p.parts
                    and p.suffix in {'.json', '.safetensors', '.txt', '.model'})
    require(sorted(paths) == actual, 'Incomplete native model/processor/template assets')
    for pin in assets['runtime']['files']: _pin(pin)
    if runtime_check:
        require(_same(late_audio._runtime_identity(), assets['runtime']), 'ASR worker runtime changed')
    parser = _pin(assets['parser']); api = _pin(assets['api'])
    runtime_pins = {row['path']: row['sha256'] for row in assets['runtime']['files']}
    require(runtime_pins.get(str(parser)) == assets['parser']['sha256']
            and runtime_pins.get(str(api)) == assets['api']['sha256'], 'ASR API/parser outside runtime manifest')
    tokenizer = assets['tokenizer_files']; require(isinstance(tokenizer, list) and bool(tokenizer), 'Missing tokenizer manifest')
    model_pins = {row['path']: row['sha256'] for row in model['files']}
    expected_tokenizer = [row for row in model['files'] if Path(row['path']).name in
        {'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
         'vocab.json', 'merges.txt', 'chat_template.json'}]
    require(_same(tokenizer, expected_tokenizer), 'Tokenizer/template manifest incomplete or reordered')
    for pin in tokenizer:
        require(model_pins.get(pin['path']) == pin['sha256'], 'Tokenizer pin outside model manifest')
    require(late_audio._interpreter_path(assets['interpreter']) == assets['interpreter'], 'Interpreter lexical path changed')
    expected = _execution_identity(assets)
    require(_same(execution_identity, expected), 'ASR execution identity differs from native assets')
    config = _read(MODEL_DIR/'config.json')
    thinker = config.get('thinker_config', config)
    text = thinker.get('text_config', {}); audio = thinker.get('audio_config', {})
    require(type(text.get('max_position_embeddings')) is int and text['max_position_embeddings'] == 65536
            and type(audio.get('max_source_positions')) is int and audio['max_source_positions'] == 1500,
            'Pinned native ASR position limits changed')
    return expected


def _pcm(audio):
    import numpy as np
    require(isinstance(audio, np.ndarray) and audio.dtype == np.dtype('float32') and audio.ndim == 1
            and len(audio) == requests.CLIP_FRAMES and np.isfinite(audio).all()
            and float(np.max(np.abs(audio))) <= 1., 'Invalid native finite mono float32 PCM')
    return np.ascontiguousarray(audio)


def _array(value):
    import numpy as np
    if hasattr(value, 'detach'):
        value = value.detach().cpu().contiguous().numpy()
    array = np.ascontiguousarray(value)
    require(array.dtype.kind in 'fiu b'.replace(' ', '') and np.isfinite(array).all(), 'Invalid native tensor bytes')
    return array


def tensor_record(value, folder):
    array = _array(value); raw = array.tobytes(order='C'); digest = _sha(raw)
    path = Path(folder)/'tensors'/(digest+'.bin'); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == raw, 'Native tensor artifact changed')
    else:
        with path.open('xb') as stream: stream.write(raw)
    return {'dtype': array.dtype.str, 'shape': list(array.shape), 'bytes_sha256': digest,
            'artifact': 'tensors/'+digest+'.bin'}


def tensor_identity(record):
    return ec._hash({key: record[key] for key in ('dtype', 'shape', 'bytes_sha256')})


def _load_tensor(record, folder):
    import numpy as np
    ec._keys(record, {'dtype', 'shape', 'bytes_sha256', 'artifact'})
    requests._hash(record['bytes_sha256'])
    require(record['artifact'] == 'tensors/'+record['bytes_sha256']+'.bin', 'Native tensor artifact escaped scope')
    shape = record['shape']; require(isinstance(shape, list) and all(type(v) is int and v >= 0 for v in shape), 'Invalid tensor shape')
    dtype = np.dtype(record['dtype']); require(dtype.kind in 'fiub' and dtype.str == record['dtype'], 'Invalid tensor dtype')
    raw = (Path(folder)/record['artifact']).read_bytes()
    require(_sha(raw) == record['bytes_sha256'] and len(raw) == math.prod(shape)*dtype.itemsize, 'Native tensor bytes changed')
    array = np.frombuffer(raw, dtype=dtype).reshape(shape)
    require(np.isfinite(array).all(), 'Nonfinite native tensor')
    return array


def _config(model):
    return {key: deepcopy(getattr(model.generation_config, key, None)) for key in _GEN_KEYS}


def _apply_policy(asr):
    require(asr.backend == 'transformers' and asr.max_inference_batch_size == 1 and asr.max_new_tokens == 512
            and asr.forced_aligner is None and str(asr.model.device).startswith('cuda')
            and str(asr.model.dtype) == 'torch.float16', 'Wrong ASR native backend/dtype/scope')
    asr.model.eval(); asr.model.thinker.eval()
    for model in (asr.model, asr.model.thinker):
        for key in _GEN_KEYS: setattr(model.generation_config, key, deepcopy(requests.POLICY[key]))
    return {key: deepcopy(requests.POLICY[key]) for key in _GEN_KEYS}


@contextmanager
def _deadline(deadline):
    remaining = deadline-time.monotonic()
    if remaining <= 0: raise WorkerDeadline('ASR work allowance exhausted')
    old_handler = signal.getsignal(signal.SIGALRM); old_timer = signal.getitimer(signal.ITIMER_REAL)
    start = time.monotonic(); effective = min(remaining, old_timer[0] if old_timer[0] else remaining)
    def expired(signum, frame): raise WorkerDeadline('ASR fixed deadline reached')
    signal.signal(signal.SIGALRM, expired); signal.setitimer(signal.ITIMER_REAL, effective)
    try: yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > time.monotonic()-start:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0]-(time.monotonic()-start), old_timer[1])


def capture_call(asr, audio, request, *, artifact_root, save_capture, deadline):
    """Transparent single public normalization/processor/outer/inner capture."""
    import numpy as np
    import torch
    require(request.get('version') == requests.NATIVE_REQUEST_VERSION and _same(request.get('policy'), requests.POLICY),
            'Native request policy changed')
    audio = _pcm(audio); expected = _apply_policy(asr); context = request['conditioning']['context']
    require(type(context) is str and len(context) <= 1200 and len(context.encode()) <= 8192
            and not any(token in context for token in request['execution_identity']['reserved_tokens']), 'Invalid exact ASR context')
    require(_sha(audio.tobytes()) == request['audio']['pcm_sha256'], 'ASR input PCM does not match bound view')
    capture = {key: None for key in _CAPTURE_KEYS}
    for key in ('public_calls', 'normalization_calls', 'split_calls', 'infer_calls', 'outer_calls', 'inner_calls', 'processor_calls', 'decode_calls'):
        capture[key] = 0
    capture['input_pcm'] = tensor_record(audio, artifact_root)
    def persist(): save_capture(deepcopy(capture))
    persist()
    public = inspect.unwrap(asr.transcribe).__func__ if hasattr(inspect.unwrap(asr.transcribe), '__func__') else inspect.unwrap(asr.transcribe)
    namespace = public.__globals__
    originals = {name: namespace[name] for name in ('normalize_audios', 'split_audio_into_chunks')}
    infer = asr._infer_asr; outer = asr.model.generate; inner = asr.model.thinker.generate
    prompt_builder = asr._build_text_prompt; processor = asr.processor
    def normalize(value):
        capture['normalization_calls'] += 1
        require(capture['normalization_calls'] == 1 and isinstance(value, tuple) and len(value) == 2
                and value[0] is audio and type(value[1]) is int and value[1] == 16000, 'Native normalization request changed')
        result = originals['normalize_audios'](value)
        require(isinstance(result, list) and len(result) == 1, 'Native normalization cardinality changed')
        normalized = _pcm(result[0]); capture['normalized_pcm'] = tensor_record(normalized, artifact_root); persist()
        require(_same(capture['input_pcm'], capture['normalized_pcm']), 'Native normalization changed PCM')
        return result
    def split(*args, **kwargs):
        capture['split_calls'] += 1
        require(not args and set(kwargs) == {'wav', 'sr', 'max_chunk_sec'} and capture['split_calls'] == 1
                and kwargs['sr'] == 16000 and kwargs['max_chunk_sec'] == 1200, 'Native chunk policy changed')
        result = originals['split_audio_into_chunks'](**kwargs)
        require(isinstance(result, list) and len(result) == 1 and result[0][1] == 0, 'Native chunk mapping changed')
        descriptor = tensor_record(_pcm(result[0][0]), artifact_root)
        capture['chunk'] = {'count': 1, 'offset_seconds': 0, 'sample_rate': 16000, 'pcm': descriptor}; persist()
        require(_same(descriptor, capture['input_pcm']), 'Native chunk changed waveform')
        return result
    def build_prompt(*args, **kwargs):
        require(not args and kwargs == {'context': context, 'force_language': 'Japanese'} and capture['prompt'] is None,
                'Native context/prompt multiplicity changed')
        result = prompt_builder(**kwargs); require(type(result) is str and bool(result), 'Native prompt missing')
        capture['prompt'] = result; persist(); return result
    def captured_infer(contexts, wavs, languages):
        capture['infer_calls'] += 1
        require(capture['infer_calls'] == 1 and contexts == [context] and languages == ['Japanese'] and len(wavs) == 1
                and _sha(_pcm(wavs[0]).tobytes()) == request['audio']['pcm_sha256'], 'Native infer scope changed')
        capture['contexts'] = contexts; capture['languages'] = languages; persist()
        return infer(contexts, wavs, languages)
    class Processor:
        def __getattr__(self, name): return getattr(processor, name)
        def __call__(self, *args, **kwargs):
            capture['processor_calls'] += 1
            require(not args and set(kwargs) == {'text', 'audio', 'return_tensors', 'padding'}
                    and capture['processor_calls'] == 1 and kwargs['text'] == [capture['prompt']]
                    and kwargs['return_tensors'] == 'pt' and kwargs['padding'] is True
                    and len(kwargs['audio']) == 1 and _sha(_pcm(kwargs['audio'][0]).tobytes()) == request['audio']['pcm_sha256'],
                    'Native processor request changed')
            result = processor(**kwargs)
            require(set(result) == _TENSOR_KEYS, 'Native processor tensor fields changed')
            capture['processor_inputs'] = {key: tensor_record(value, artifact_root) for key, value in result.items()}
            persist(); return result
        def batch_decode(self, tokens, **kwargs):
            capture['decode_calls'] += 1
            require(capture['decode_calls'] == 1 and kwargs == {'skip_special_tokens': True, 'clean_up_tokenization_spaces': False},
                    'Native raw decoding policy changed')
            require(_array(tokens).tolist() == [capture['output_token_ids']], 'Native decoded token IDs changed')
            result = processor.batch_decode(tokens, **kwargs)
            require(isinstance(result, list) and len(result) == 1 and type(result[0]) is str, 'Native raw decode missing')
            capture['raw_text'] = result[0]; persist(); return result
    def generate_outer(*args, **kwargs):
        capture['outer_calls'] += 1
        require(not args and capture['outer_calls'] == 1 and set(kwargs) == _TENSOR_KEYS | {'max_new_tokens'}
                and type(kwargs['max_new_tokens']) is int and kwargs['max_new_tokens'] == 512, 'Native outer generation changed')
        capture['outer_inputs'] = {k: tensor_record(kwargs[k], artifact_root) for k in _TENSOR_KEYS}
        capture['outer_kwargs'] = {'max_new_tokens': kwargs['max_new_tokens']}; capture['outer_config'] = _config(asr.model)
        persist(); return outer(**kwargs)
    def generate_inner(*args, **kwargs):
        capture['inner_calls'] += 1
        require(not args and capture['inner_calls'] == 1 and set(kwargs) == _TENSOR_KEYS | {'max_new_tokens', 'eos_token_id', 'return_dict_in_generate'},
                'Native delegated generation changed')
        capture['inner_inputs'] = {k: tensor_record(kwargs[k], artifact_root) for k in _TENSOR_KEYS}
        capture['inner_kwargs'] = {k: deepcopy(kwargs[k]) for k in ('max_new_tokens', 'eos_token_id', 'return_dict_in_generate')}
        capture['inner_config'] = _config(asr.model.thinker)
        require(_same(capture['outer_config'], expected) and _same(capture['inner_config'], expected)
                and _same(capture['inner_kwargs'], {'max_new_tokens': 512, 'eos_token_id': [151645, 151643], 'return_dict_in_generate': True})
                and _same(capture['outer_inputs'], capture['inner_inputs']), 'Native effective inner policy differs')
        tokens = processor.tokenizer.encode(context, add_special_tokens=False)
        require(isinstance(tokens, list) and all(type(t) is int and t >= 0 for t in tokens) and len(tokens) <= 1024, 'ASR hint token limit exceeded')
        capture['hint_token_ids'] = tokens
        ids = _array(kwargs['input_ids']); mask = _array(kwargs['attention_mask']); features = _array(kwargs['input_features']); fmask = _array(kwargs['feature_attention_mask'])
        require(ids.ndim == 2 and ids.shape[0] == 1 and mask.shape == ids.shape and ids.dtype.kind in 'iu'
                and ids.shape[1] > 0 and (ids >= 0).all() and np.isin(mask, [0, 1]).all() and features.ndim == 3 and features.shape[:2] == (1, 128)
                and fmask.ndim == 2 and fmask.shape[0] == 1 and fmask.shape[1] == features.shape[2]
                and np.isin(fmask, [0, 1]).all() and 0 < int(fmask.sum()) <= 3000, 'Native expanded audio/text geometry changed')
        capture['capacity'] = {'input_positions': int(ids.shape[1]), 'feature_frames': int(fmask.sum()),
            'position_limit': 65536, 'reserved_output': 512, 'margin': 64, 'hint_tokens': len(tokens),
            'fits': ids.shape[1]+512+64 <= 65536}
        capture['capacity_finished_utc'] = _now(); persist()
        require(capture['capacity']['fits'] is True, 'Expanded ASR input exceeds pinned capacity')
        if time.monotonic() >= deadline: raise WorkerDeadline('No native generation time remains')
        capture['generation_started_utc'] = _now(); persist()
        try:
            output = inner(**kwargs)
            ids = _array(output.sequences)
            require(ids.ndim == 2 and ids.shape[0] == 1, 'Native output sequence cardinality changed')
            prefix = _array(kwargs['input_ids'])
            require(np.array_equal(ids[:, :prefix.shape[1]], prefix), 'Native output prompt prefix changed')
            generated = ids[0, prefix.shape[1]:].tolist(); capture['output_token_ids'] = generated
            capture['normal_stop'] = bool(generated and generated[-1] in requests.POLICY['eos_token_id']
                and not any(token in requests.POLICY['eos_token_id'] for token in generated[:-1]) and len(generated) < 512)
            persist(); return output
        finally:
            capture['generation_finished_utc'] = _now(); persist()
    namespace['normalize_audios'] = normalize; namespace['split_audio_into_chunks'] = split
    asr._infer_asr = captured_infer; asr._build_text_prompt = build_prompt; asr.processor = Processor()
    asr.model.generate = generate_outer; asr.model.thinker.generate = generate_inner
    try:
        with _deadline(deadline):
            torch.manual_seed(20260915); torch.cuda.manual_seed_all(20260915)
            capture['public_calls'] = 1; persist()
            with torch.inference_mode():
                results = asr.transcribe((audio, 16000), context=context, language='Japanese', return_time_stamps=False)
            require(isinstance(results, list) and len(results) == 1, 'Native public result missing')
            result = {key: getattr(results[0], key) for key in ('text', 'language', 'time_stamps')}
            capture['native_result'] = result; capture['parser_changed'] = capture['raw_text'] != result['text']; persist()
            require(capture['normal_stop'] is True and all(capture[key] == 1 for key in ('public_calls', 'normalization_calls',
                'split_calls', 'infer_calls', 'outer_calls', 'inner_calls', 'processor_calls', 'decode_calls')), 'Incomplete or truncated native call')
            return capture
    finally:
        namespace.update(originals); asr._infer_asr = infer; asr._build_text_prompt = prompt_builder
        asr.processor = processor; asr.model.generate = outer; asr.model.thinker.generate = inner


def _describe(value):
    array = _array(value); digest = _sha(array.tobytes(order='C'))
    return {'dtype': array.dtype.str, 'shape': list(array.shape), 'bytes_sha256': digest,
            'artifact': 'tensors/'+digest+'.bin'}


def validate_native_receipt(receipt, request, plan, *, assets, artifact_root, processor):
    """Independent pinned CPU processor/parser replay; never calls a model."""
    import numpy as np
    requests.validate_native_request(request, plan); validate_assets(assets, plan['execution_identity'])
    _sealed(receipt)
    ec._keys(receipt, {'version', 'request', 'request_sha256', 'assets_sha256', 'started_utc', 'finished_utc',
                       'seconds', 'status', 'error_type', 'capture', 'receipt_sha256'})
    require(receipt['version'] == VERSION and _same(receipt['request'], request)
        and receipt['request_sha256'] == ec._hash(request) and receipt['assets_sha256'] == ec._hash(assets)
        and receipt['status'] in ('complete', 'empty') and receipt['error_type'] is None
        and type(receipt['seconds']) in (int, float) and math.isfinite(receipt['seconds'])
        and 0 <= receipt['seconds'] <= 60, 'Native receipt identity/status/time changed')
    c = receipt['capture']; ec._keys(c, _CAPTURE_KEYS)
    for key in ('public_calls', 'normalization_calls', 'split_calls', 'infer_calls', 'outer_calls', 'inner_calls', 'processor_calls', 'decode_calls'):
        require(type(c[key]) is int and c[key] == 1, 'Native ASR call or chunk count changed')
    arrays = {}
    for key in ('input_pcm', 'normalized_pcm'):
        arrays[key] = _load_tensor(c[key], artifact_root)
        _pcm(arrays[key]); require(c[key]['bytes_sha256'] == request['audio']['pcm_sha256'], 'Native PCM differs from physical view')
    require(_same(c['input_pcm'], c['normalized_pcm']) and _same(c['chunk'],
        {'count': 1, 'offset_seconds': 0, 'sample_rate': 16000, 'pcm': c['input_pcm']}), 'Native normalization/chunk changed')
    source_pcm = _read_audio(request['audio'])
    require(source_pcm.tobytes() == arrays['input_pcm'].tobytes(), 'Captured PCM differs from pinned WAV')
    context = request['conditioning']['context']
    require(_same(c['contexts'], [context]) and _same(c['languages'], ['Japanese']), 'Native conditioning changed')
    hint_tokens = processor.tokenizer.encode(context, add_special_tokens=False)
    require(isinstance(hint_tokens, list) and all(type(t) is int and t >= 0 for t in hint_tokens)
        and len(hint_tokens) <= 1024 and _same(c['hint_token_ids'], hint_tokens), 'Native hint tokens changed')
    expected_prompt = processor.apply_chat_template([
        {'role': 'system', 'content': context}, {'role': 'user', 'content': [{'type': 'audio', 'audio': ''}]}],
        add_generation_prompt=True, tokenize=False)+'language Japanese<asr_text>'
    require(c['prompt'] == expected_prompt, 'Native rendered context/template changed')
    inputs = processor(text=[expected_prompt], audio=[arrays['input_pcm']], return_tensors='pt', padding=True)
    require(set(inputs) == _TENSOR_KEYS, 'Replayed processor fields changed')
    expected_processor = {key: _describe(value) for key, value in inputs.items()}
    expected_outer = {key: _describe(_array(value).astype(np.float16) if _array(value).dtype.kind == 'f'
                                   else _array(value)) for key, value in inputs.items()}
    require(_same(c['processor_inputs'], expected_processor) and _same(c['outer_inputs'], expected_outer)
        and _same(c['inner_inputs'], expected_outer), 'Native processor/features/input positions differ from PCM')
    for collection in ('processor_inputs', 'outer_inputs', 'inner_inputs'):
        for value in c[collection].values(): _load_tensor(value, artifact_root)
    policy = {key: deepcopy(requests.POLICY[key]) for key in _GEN_KEYS}
    require(_same(c['outer_config'], policy) and _same(c['inner_config'], policy)
        and _same(c['outer_kwargs'], {'max_new_tokens': 512})
        and _same(c['inner_kwargs'], {'max_new_tokens': 512, 'eos_token_id': [151645, 151643], 'return_dict_in_generate': True}),
        'Native effective decoder policy changed')
    ids = _array(inputs['input_ids']); mask = _array(inputs['feature_attention_mask'])
    expected_capacity = {'input_positions': int(ids.shape[1]), 'feature_frames': int(mask.sum()), 'position_limit': 65536,
        'reserved_output': 512, 'margin': 64, 'hint_tokens': len(hint_tokens), 'fits': ids.shape[1]+512+64 <= 65536}
    require(_same(c['capacity'], expected_capacity) and expected_capacity['fits'] is True, 'Native capacity changed')
    times = [_time(receipt['started_utc']), _time(c['capacity_finished_utc']), _time(c['generation_started_utc']),
             _time(c['generation_finished_utc']), _time(receipt['finished_utc'])]
    require(times == sorted(times), 'Native capacity/generation timestamps reordered')
    generated = c['output_token_ids']
    require(isinstance(generated, list) and 0 < len(generated) < 512
        and all(type(token) is int and token >= 0 for token in generated)
        and generated[-1] in requests.POLICY['eos_token_id']
        and not any(token in requests.POLICY['eos_token_id'] for token in generated[:-1])
        and c['normal_stop'] is True, 'Native EOS/count failed')
    decoded = processor.batch_decode([generated], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    require(isinstance(decoded, list) and len(decoded) == 1 and type(decoded[0]) is str and decoded[0] == c['raw_text'],
            'Native raw text differs from saved token IDs')
    language, text = native_parse(c['raw_text'], assets['parser']['path'], assets['parser']['sha256'])
    require(_same(c['native_result'], {'text': text, 'language': language, 'time_stamps': None})
        and type(c['parser_changed']) is bool and c['parser_changed'] == (c['raw_text'] != text)
        and receipt['status'] == ('empty' if text == '' else 'complete'), 'Native parser result/empty status changed')
    return {'raw_text': c['raw_text'], 'parsed_text': text,
        'audio_features_sha256': tensor_identity(c['outer_inputs']['input_features']),
        'audio_mask_sha256': tensor_identity(c['outer_inputs']['feature_attention_mask'])}


def _read_audio(binding):
    import numpy as np
    import soundfile as sf
    path = Path(binding['path']); require(_file(path) == binding['sha256'], 'Prepared ASR WAV changed')
    info = sf.info(path)
    require(info.channels == 1 and info.frames == 192000 and info.samplerate == 16000 and info.subtype == 'FLOAT',
            'Prepared ASR WAV is not exact FLOAT mono geometry')
    audio, rate = sf.read(path, dtype='float32', always_2d=False); audio = _pcm(audio)
    require(type(rate) is int and rate == 16000 and _sha(audio.tobytes()) == binding['pcm_sha256']
            and _file(path) == binding['sha256'], 'Prepared ASR PCM changed')
    return np.array(audio, copy=True)


def _load_asr(assets):
    import torch
    from qwen_asr import Qwen3ASRModel
    return Qwen3ASRModel.from_pretrained(assets['model']['model_dir'], dtype=torch.float16, device_map='cuda',
        local_files_only=True, max_inference_batch_size=1, max_new_tokens=512)


def run_slots(plan, assets, folder, asr, audios, *, deadline, save_state):
    """Ordered capture after authenticated startup; no retries, controls precede reals."""
    folder = Path(folder); results = {}; slots = []
    for number, slot in enumerate(plan['slots'], 1):
        slots.append({**slot, 'number': number, 'status': 'undispatched', 'receipt': None})
    summary = {'version': VERSION, 'plan_sha256': plan['plan_sha256'], 'status': 'running',
               'slots': slots, 'asr_calls': 0, 'controls': {}, 'error_type': None}
    save_state(deepcopy(summary))
    # Admission for every context before the first public ASR call.
    for slot in slots:
        request = requests.build_native_request(plan, slot['pair_id'], slot['arm'])
        tokens = asr.processor.tokenizer.encode(request['conditioning']['context'], add_special_tokens=False)
        require(isinstance(tokens, list) and all(type(token) is int and token >= 0 for token in tokens)
                and len(tokens) <= 1024, 'A planned ASR hint exceeds tokenizer bounds')
    for slot in slots:
        request = requests.build_native_request(plan, slot['pair_id'], slot['arm'])
        path = folder/'receipts'/f'{slot["number"]:02d}-{slot["pair_id"]}-{slot["arm"]}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        if time.monotonic() >= deadline:
            summary.update(status='failed', error_type='WorkerDeadline'); save_state(deepcopy(summary)); return summary
        started = time.monotonic()
        with path.with_suffix('.started.json').open('x', encoding='utf-8') as stream:
            json.dump({'version': VERSION, 'request_sha256': ec._hash(request), 'started_utc': _now()}, stream)
        require(not path.exists(), 'ASR receipt already exists; cannot reroll')
        receipt = {'version': VERSION, 'request': request, 'request_sha256': ec._hash(request), 'assets_sha256': ec._hash(assets),
            'started_utc': _now(), 'finished_utc': None, 'seconds': 0., 'status': 'running', 'error_type': None, 'capture': {}}
        slot.update(status='started', receipt=str(path)); summary['asr_calls'] += 1
        def save_capture(capture):
            receipt['capture'] = capture; _save(path, receipt)
        _save(path, receipt); save_state(deepcopy(summary))
        failure = None; result = None
        try:
            with _deadline(min(deadline, started+60)):
                capture_call(asr, audios[slot['pair_id']], request, artifact_root=folder, save_capture=save_capture,
                             deadline=min(deadline, started+60))
                receipt.update(status='empty' if receipt['capture']['native_result']['text'] == '' else 'complete',
                               finished_utc=_now(), seconds=time.monotonic()-started)
                _save(path, receipt)
                result = validate_native_receipt(receipt, request, plan, assets=assets, artifact_root=folder, processor=asr.processor)
        except BaseException as exc:
            failure = exc; receipt.update(status='failed', error_type=type(exc).__name__)
        finally:
            receipt.update(finished_utc=_now(), seconds=time.monotonic()-started); _save(path, receipt)
            slot['status'] = receipt['status']; save_state(deepcopy(summary))
        if failure is not None:
            summary.update(status='failed', error_type=type(failure).__name__); save_state(deepcopy(summary)); return summary
        results[slot['slot_id']] = result
        if slot['pair_id'] in ('silence', 'noise'):
            summary['controls'][slot['slot_id']] = requests.control_metrics(plan, result['raw_text'], result['parsed_text'])
        if slot['number'] == 4 and any(v['raw_nonempty'] or v['parsed_nonempty'] for v in summary['controls'].values()):
            summary['status'] = 'negative_control_output'; save_state(deepcopy(summary)); return summary
    changed = [pair for pair in requests.PAIR_IDS[2:]
               if results[pair+':unhinted']['parsed_text'] != results[pair+':hinted']['parsed_text']]
    summary.update(status='complete' if len(changed) >= 2 else 'conditioning_inactive', changed_real_pairs=changed)
    save_state(deepcopy(summary)); return summary


def run_worker(spec_path):
    """Direct-script worker. Parent owns process termination and restoration."""
    started = time.monotonic(); spec = _read(spec_path); ec._keys(spec, _SPEC_KEYS)
    require(spec['version'] == VERSION and type(spec['worker_seconds']) in (int, float)
        and math.isfinite(spec['worker_seconds']) and 0 < spec['worker_seconds'] <= 900,
        'Invalid fixed worker budget')
    remaining = (_time(spec['work_deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
    deadline = started+min(spec['worker_seconds'], remaining)
    folder = Path(spec['output_dir']); require(folder.is_absolute(), 'Worker output must be absolute')
    folder.mkdir(parents=True, exist_ok=True); require(not any(folder.iterdir()), 'Worker output already exists; cannot resume')
    state = {'version': VERSION, 'status': 'prepared', 'started_utc': _now(), 'finished_utc': None,
             'seconds': 0., 'asr_calls': 0, 'model_loads': 0, 'model_load_attempts': 0, 'error_type': None, 'spec_sha256': _file(spec_path)}
    def save(value):
        state.update(value); state['seconds'] = time.monotonic()-started; _save(folder/'worker-result.json', state)
    save({})
    try:
        with _deadline(deadline):
            require(_file(__file__) == spec['worker_sha256'], 'ASR worker code changed')
            require(late_audio._interpreter_path(sys.executable) == spec['assets']['interpreter'], 'Wrong worker interpreter')
            require(_file(spec['plan_path']) == spec['plan_file_sha256'], 'ASR plan bytes changed')
            plan = _read(spec['plan_path']); requests._check_plan(plan)
            save({'plan_sha256': plan['plan_sha256'], 'slots': [{**slot, 'number': number,
                'status': 'undispatched', 'receipt': None} for number, slot in enumerate(plan['slots'], 1)]})
            validate_assets(spec['assets'], plan['execution_identity'], runtime_check=True)
            os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'; os.environ['HF_DATASETS_OFFLINE'] = '1'
            audios = {pair: _read_audio(plan['audio_bindings'][pair]) for pair in requests.PAIR_IDS}
            import numpy as np
            expected = {'silence': np.zeros(192000, dtype=np.float32),
                'noise': np.random.Generator(np.random.PCG64(20260915)).uniform(-.01, .01, 192000).astype(np.float32)}
            require(all(audios[k].tobytes() == v.tobytes() for k, v in expected.items()), 'Synthetic control generation changed')
            save({'model_load_attempts': 1, 'model_load_started_utc': _now()})
            asr = _load_asr(spec['assets']); _check_processor(spec['assets'], asr.processor)
            save({'model_loads': 1, 'model_load_finished_utc': _now()})
            result = run_slots(plan, spec['assets'], folder, asr, audios, deadline=deadline, save_state=save)
            save(result)
    except BaseException as exc:
        save({'status': 'failed', 'error_type': type(exc).__name__})
    finally:
        state['finished_utc'] = _now(); save({})
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec', type=Path, required=True)
    result = run_worker(parser.parse_args().spec)
    print(json.dumps({key: result.get(key) for key in ('status', 'asr_calls', 'model_loads', 'seconds', 'error_type')}))
    return 1 if result['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
