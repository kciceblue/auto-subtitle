"""Isolated window-mask acquisition; old native helpers prove I/O capture only."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src import auto_source_native as capture
from src import qwen_audio_attention as attention
from src import evidence_context as ec
from src.contextual_review_contract import require
from src.contextual_asr_native import _read, _save, _file, _now, _time, _same, _sealed, _deadline, WorkerDeadline
VERSION = 'masked-source-worker-1'
PLAN_VERSION = 'masked-source-plan-1'
REQUEST_VERSION = 'masked-source-request-1'
OBSERVATIONS_VERSION = 'masked-source-observations-1'
CONDITIONING = capture.CONDITIONING
POLICY = {**capture.POLICY, 'maximum_calls': 68, 'control_calls': 2, 'real_calls': 66}
ENCODER_CONFIG = {'backend': 'sdpa', 'layers': 24, 'hidden_size': 1024, 'heads': 16, 'head_dim': 64,
                  'n_window': 50, 'n_window_infer': 800, 'max_source_positions': 1500}
ATTENTION_POLICY = {'version': attention.VERSION, 'change': 'encoder_window_mask_passed_to_every_layer',
                    'backend_changed': False, 'independent_recognizer': False, 'accuracy_verified': False}


def execution_identity(assets):
    result = capture.base.execution_identity(assets)
    result['worker_sha256'] = _file(__file__)
    return result


def model_config(assets):
    path = Path(assets['model']['model_dir'])/'config.json'
    record = {'path': str(path.resolve()), 'sha256': _file(path)}
    require(any(v['path'] == record['path'] and v['sha256'] == record['sha256'] for v in assets['model']['files']), 'Model configuration not pinned')
    cfg = _read(path)['thinker_config']['audio_config']
    mapping = {'layers': 'encoder_layers', 'hidden_size': 'd_model', 'heads': 'encoder_attention_heads', 'n_window': 'n_window', 'n_window_infer': 'n_window_infer', 'max_source_positions': 'max_source_positions'}
    require(all(cfg.get(field) == ENCODER_CONFIG[key] for key, field in mapping.items()), 'Checkpoint encoder geometry differs')
    return record


def build_plan(parent, controls, assets, input_proof_sha256):
    capture.check_plan(parent, assets)
    ec._keys(controls, {'silence', 'noise'})
    for value in controls.values():
        ec._keys(value, {'path', 'sha256', 'pcm_sha256', 'frames', 'sample_rate', 'dtype'})
        require(value['frames'] == 192000 and value['sample_rate'] == 16000 and value['dtype'] == 'float32', 'Fixed short control geometry changed')
    slots = [{'slot_id': name, 'kind': 'control', 'owner_id': None} for name in ('silence', 'noise')]
    slots += [{'slot_id': f'owner-{i:03d}', 'kind': 'real', 'owner_id': i} for i in range(1, 67)]
    value = {'version': PLAN_VERSION, 'parent_plan': deepcopy(parent), 'controls': deepcopy(controls),
             'input_proof_sha256': input_proof_sha256, 'execution_identity': execution_identity(assets),
             'attention_source': attention.source_identity(), 'encoder_config': deepcopy(ENCODER_CONFIG), 'model_config': model_config(assets),
             'attention_policy': deepcopy(ATTENTION_POLICY), 'capture_helper_sha256': _file(capture.__file__),
             'policy': deepcopy(POLICY), 'slots': slots}
    value['plan_sha256'] = ec._hash(value)
    return value


def check_plan(plan, assets):
    require(_same(plan, build_plan(plan['parent_plan'], plan['controls'], assets, plan['input_proof_sha256'])), 'Masked acquisition plan changed')


def capture_contract(plan, slot_id):
    """Mechanical input contract for frozen I/O validation, never new-model proof.

    Its control index is only an internal transport position; the wrapper carries
    the actual control identity and no control is exported as an owner reading.
    """
    slot = next((s for s in plan['slots'] if s['slot_id'] == slot_id), None)
    require(slot is not None, 'Unknown masked slot')
    parent = deepcopy(plan['parent_plan']); owner = slot['owner_id'] or 1
    if slot['kind'] == 'control':
        parent['audio_bindings']['1'] = deepcopy(plan['controls'][slot_id])
        parent['geometry'][0] = {'kind': 'synthetic_control', 'control_id': slot_id,
                                 'crop_start_frame': 0, 'crop_end_frame': 192000, 'sample_rate': 16000}
    return parent, capture.build_request(parent, owner)


def build_request(plan, slot_id):
    _, low = capture_contract(plan, slot_id)
    slot = next(s for s in plan['slots'] if s['slot_id'] == slot_id)
    return {'version': REQUEST_VERSION, 'plan_sha256': plan['plan_sha256'], 'slot': deepcopy(slot),
            'execution_identity': deepcopy(plan['execution_identity']), 'attention_source': deepcopy(plan['attention_source']),
            'encoder_config': deepcopy(plan['encoder_config']), 'model_config': deepcopy(plan['model_config']), 'attention_policy': deepcopy(ATTENTION_POLICY),
            'capture_role': 'low_level_io_only_not_attention_semantics', 'capture_request': low}


def validate_native_receipt(receipt, request, plan, *, assets, artifact_root, processor):
    require(_same(request, build_request(plan, request['slot']['slot_id'])), 'Masked request changed')
    _sealed(receipt)
    ec._keys(receipt, {'version', 'request', 'request_sha256', 'assets_sha256', 'started_utc', 'finished_utc',
                      'seconds', 'status', 'error_type', 'low_level_capture', 'attention', 'receipt_sha256'})
    require(receipt['version'] == VERSION and _same(receipt['request'], request)
            and receipt['request_sha256'] == ec._hash(request) and receipt['assets_sha256'] == ec._hash(assets)
            and receipt['status'] in ('complete', 'empty') and receipt['error_type'] is None
            and type(receipt['seconds']) in (int, float) and math.isfinite(receipt['seconds']) and 0 <= receipt['seconds'] <= 60,
            'Masked receipt status/binding changed')
    # Explicit projection into the old mechanical validator, not an old native receipt.
    low = {k: deepcopy(receipt[k]) for k in ('assets_sha256', 'started_utc', 'finished_utc', 'seconds', 'status', 'error_type')}
    low.update(version=capture.VERSION, request=deepcopy(request['capture_request']),
               request_sha256=ec._hash(request['capture_request']), capture=deepcopy(receipt['low_level_capture']))
    low['receipt_sha256'] = ec._hash(low)
    parent, low_request = capture_contract(plan, request['slot']['slot_id'])
    output = capture.validate_native_receipt(low, low_request, parent, assets=assets, artifact_root=artifact_root, processor=processor)
    record = attention.validate_attestation(receipt['attention'],
        feature_frames=receipt['low_level_capture']['capacity']['feature_frames'],
        config=plan['encoder_config'], source=plan['attention_source'])
    require(record['mask_dtype'] == 'float16' and record['device'].startswith('cuda'), 'Production attention dtype/device changed')
    output['attention_sha256'] = receipt['attention']['attestation_sha256']
    return output


def control_result(output):
    category = capture.protocol_category(output['raw_text'], output['parsed_text'])
    return {'protocol_category': category, 'parsed_nonempty': output['parsed_text'] != '',
            'passed': output['parsed_text'] == '' and category in ('empty_raw', 'native_none_empty_tail')}


def run_slots(plan, assets, folder, asr, audios, *, deadline, save_state):
    check_plan(plan, assets); folder = Path(folder)
    require(_same(attention.encoder_config(asr.model.thinker.audio_tower), plan['encoder_config']), 'Actual encoder backend/config changed')
    slots = [{**s, 'number': i, 'status': 'undispatched', 'receipt': None} for i, s in enumerate(plan['slots'], 1)]
    summary = {'version': VERSION, 'plan_sha256': plan['plan_sha256'], 'status': 'running', 'slots': slots,
               'asr_calls': 0, 'controls': {}, 'real_audio_calls': 0, 'error_type': None}
    save_state(deepcopy(summary))
    for slot in slots:
        request = build_request(plan, slot['slot_id']); path = folder/'receipts'/f'{slot["number"]:02d}-{slot["slot_id"]}.json'
        path.parent.mkdir(parents=True, exist_ok=True); started = time.monotonic(); end = min(deadline, started+60)
        require(started < deadline, 'Masked worker time exhausted')
        with path.with_suffix('.started.json').open('x', encoding='utf-8') as stream:
            import json
            json.dump({'version': VERSION, 'request_sha256': ec._hash(request), 'started_utc': _now()}, stream)
        require(not path.exists(), 'Masked receipt already attempted; no reroll')
        receipt = {'version': VERSION, 'request': request, 'request_sha256': ec._hash(request), 'assets_sha256': ec._hash(assets),
                   'started_utc': _now(), 'finished_utc': None, 'seconds': 0., 'status': 'running', 'error_type': None,
                   'low_level_capture': {}, 'attention': None}
        slot.update(status='started', receipt=str(path)); summary['asr_calls'] += 1
        if slot['kind'] == 'real': summary['real_audio_calls'] += 1
        def save_capture(value): receipt['low_level_capture'] = value; _save(path, receipt)
        _save(path, receipt); save_state(deepcopy(summary)); failure = None
        try:
            with _deadline(end):
                audit = None
                try:
                    with attention.scoped_window_attention(asr.model.thinker.audio_tower) as audit:
                        capture.capture_call(asr, audios[slot['slot_id']], request['capture_request'],
                            artifact_root=folder, save_capture=save_capture, deadline=end)
                finally:
                    receipt['attention'] = deepcopy(audit); _save(path, receipt)
                receipt.update(status='empty' if receipt['low_level_capture']['native_result']['text'] == '' else 'complete',
                               finished_utc=_now(), seconds=time.monotonic()-started)
                _save(path, receipt)
                output = validate_native_receipt(receipt, request, plan, assets=assets, artifact_root=folder, processor=asr.processor)
                if slot['kind'] == 'control': summary['controls'][slot['slot_id']] = control_result(output)
        except BaseException as exc:
            failure = exc; receipt.update(status='failed', error_type=type(exc).__name__)
        finally:
            receipt.update(finished_utc=_now(), seconds=time.monotonic()-started); _save(path, receipt)
            slot['status'] = receipt['status']; save_state(deepcopy(summary))
        if failure is not None:
            summary.update(status='failed', error_type=type(failure).__name__); break
        if slot['kind'] == 'control' and not summary['controls'][slot['slot_id']]['passed']:
            summary['status'] = 'negative_control_output'; break
    else: summary['status'] = 'complete'
    save_state(deepcopy(summary)); return summary


def runtime_identity(asr, plan, assets):
    config = attention.encoder_config(asr.model.thinker.audio_tower)
    require(_same(config, plan['encoder_config']), 'Actual audio backend/config changed')
    text_backend = asr.model.thinker.config.text_config._attn_implementation
    require(type(text_backend) is str and bool(text_backend) and asr.backend == 'transformers', 'Actual thinker/backend identity missing')
    return {'model_dir': str(Path(assets['model']['model_dir']).resolve()), 'model_config': deepcopy(plan['model_config']),
            'asr_backend': asr.backend, 'audio_backend': config['backend'], 'thinker_text_backend': text_backend,
            'audio_encoder_config': config, 'device': str(asr.model.device), 'dtype': str(asr.model.dtype)}


def run_worker(spec_path):
    started = time.monotonic(); spec = _read(spec_path); ec._keys(spec, capture.base._SPEC_KEYS)
    require(spec['version'] == VERSION and type(spec['worker_seconds']) in (int, float)
            and math.isfinite(spec['worker_seconds']) and 0 < spec['worker_seconds'] <= 840, 'Invalid masked worker budget')
    deadline = started+min(spec['worker_seconds'], (_time(spec['work_deadline_utc'])-datetime.now(timezone.utc)).total_seconds())
    folder = Path(spec['output_dir']); require(folder.is_absolute(), 'Absolute worker folder required')
    folder.mkdir(parents=True, exist_ok=True); require(not any(folder.iterdir()), 'Worker already attempted')
    state = {'version': VERSION, 'status': 'prepared', 'started_utc': _now(), 'finished_utc': None, 'seconds': 0.,
             'asr_calls': 0, 'model_loads': 0, 'model_load_attempts': 0, 'error_type': None, 'spec_sha256': _file(spec_path)}
    def save(value): state.update(value); state['seconds'] = time.monotonic()-started; _save(folder/'worker-result.json', state)
    save({})
    try:
        with _deadline(deadline):
            require(_file(__file__) == spec['worker_sha256'] and _file(spec['plan_path']) == spec['plan_file_sha256'], 'Worker/plan changed')
            require(capture.base.late_audio._interpreter_path(sys.executable) == spec['assets']['interpreter'], 'Wrong worker interpreter')
            plan = _read(spec['plan_path']); check_plan(plan, spec['assets'])
            capture.base.validate_assets(spec['assets'], capture.base.execution_identity(spec['assets']), runtime_check=True)
            for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'): os.environ[key] = '1'
            audios = {s['slot_id']: capture._read_audio(build_request(plan, s['slot_id'])['capture_request']['audio']) for s in plan['slots']}
            save({'model_load_attempts': 1, 'model_load_started_utc': _now()})
            asr = capture.base._load_asr(spec['assets']); capture.base._check_processor(spec['assets'], asr.processor)
            save({'model_loads': 1, 'model_load_finished_utc': _now(), 'runtime_before': runtime_identity(asr, plan, spec['assets'])})
            save(run_slots(plan, spec['assets'], folder, asr, audios, deadline=deadline, save_state=save))
            after = runtime_identity(asr, plan, spec['assets']); save({'runtime_after': after})
            require(_same(state['runtime_before'], after), 'Runtime backend changed during acquisition')
    except BaseException as exc: save({'status': 'failed', 'error_type': type(exc).__name__})
    finally: state['finished_utc'] = _now(); save({})
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec', type=Path, required=True)
    value = run_worker(parser.parse_args().spec)
    print({k: value.get(k) for k in ('status', 'asr_calls', 'model_loads', 'seconds', 'error_type')})
    return 1 if value['status'] == 'failed' else 0
if __name__ == '__main__': raise SystemExit(main())
