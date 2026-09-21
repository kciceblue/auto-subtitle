"""One isolated mask-corrected acquisition: two controls, then all66 saved views."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts import auto_source_acquisition as prior
from src import masked_source_native as native
from src.workflow_state import file_hash, fingerprint, write_json
from src.contextual_review_contract import require
VERSION = 'masked-source-acquisition-1'
FOLDER = ROOT/'output/quality-masked-source-20260916'
PARENT = ROOT/'output/quality-auto-source-20260915'
CONTROLS = ROOT/'output/quality-contextual-asr-auto-20260915'
SOURCE = ROOT/'output/quality-contextual-asr-projected-20260915'
PLAN = ROOT/'design/quality-masked-source-acquisition-20260916.md'
base = prior.base
read, immutable, same, locked, state, now, phase = prior.read, prior.immutable, prior.same, prior.locked, prior.state, prior.now, prior.phase


def direct_inputs():
    """Direct geometry/PCM provenance only; no predecessor generation or scores."""
    paths = [PARENT/'registration.json', PARENT/'native-plan.json', PARENT/'assets.json',
             PARENT/'audio-preparation.json', SOURCE/'input-capsule.json', SOURCE/'audio-preparation.json', CONTROLS/'native-plan.json']
    reg = read(paths[0]); require(reg['registration_sha256'] == fingerprint({k:v for k,v in reg.items() if k != 'registration_sha256'}), 'Original input registration changed')
    for path in paths[1:]: require(reg['pins'].get(str(path.resolve())) == file_hash(path), 'Direct original input pin changed')
    parent, assets, capsule = read(paths[1]), read(paths[2]), read(SOURCE/'input-capsule.json')
    native.capture.check_plan(parent, assets)
    require(parent['source_rows_sha256'] == native.ec._hash(capsule['source_rows'])
            and parent['detector_record_sha256'] == native.ec._hash(capsule['detector_record'])
            and parent['original_mono_sha256'] == capsule['detector_provenance']['mono_sha256']
            and same(parent['geometry'], native.capture.owner_geometry(capsule['source_rows'], capsule['detector_record'])), 'Original source geometry changed')
    # This is the frozen gain/crop calculation, not historical inference replay.
    prior.validate_audio(PARENT, capsule, parent['audio_bindings'])
    control_plan = read(CONTROLS/'native-plan.json')
    controls = {key: deepcopy(control_plan['parent_plan']['audio_bindings'][key]) for key in ('silence', 'noise')}
    import numpy as np
    arrays = {'silence': np.zeros(192000, dtype=np.float32),
              'noise': np.random.Generator(np.random.PCG64(20260915)).uniform(-.01,.01,192000).astype(np.float32)}
    for key in controls: require(native.capture._read_audio(controls[key]).tobytes() == arrays[key].tobytes(), 'Fixed control PCM changed')
    paths += [Path(capsule['mono_path']), *(Path(v['path']) for v in parent['audio_bindings'].values()), *(Path(v['path']) for v in controls.values())]
    pins = {str(p.resolve()): file_hash(p) for p in paths}
    proof = {'parent_plan_sha256': file_hash(PARENT/'native-plan.json'), 'audio_preparation_sha256': file_hash(PARENT/'audio-preparation.json'),
             'source_rows_sha256': parent['source_rows_sha256'], 'same_saved_owner_pcm': True, 'fresh_negative_controls': 2,
             'old_native_semantics_replayed': False, 'source_bindings': {k: deepcopy(capsule[k]) for k in ('source_provenance', 'detector_provenance')}}
    return parent, controls, assets, proof, pins


def producers():
    return [PLAN, Path(__file__), Path(native.__file__), Path(native.attention.__file__), *[p for p in prior.producers() if p != prior.PLAN]]


def validate_registration(folder):
    reg = read(folder/'registration.json')
    require(reg['version'] == VERSION and same(reg['policy'], native.POLICY)
            and reg['registration_sha256'] == fingerprint({k:v for k,v in reg.items() if k != 'registration_sha256'}), 'Masked registration changed')
    required = {str(p.resolve()) for p in producers()} | {str(folder/name) for name in ('native-plan.json','assets.json','input-proof.json')}
    require(required <= set(reg['pins']), 'Missing mandatory input/producer pin')
    for path,digest in reg['pins'].items(): require(file_hash(Path(path)) == digest, 'Pinned masked input changed')
    # Direct originals are independently compared; no recursive inference replay.
    parent = read(PARENT/'native-plan.json'); controls_plan = read(CONTROLS/'native-plan.json'); assets = read(PARENT/'assets.json')
    controls = {k: controls_plan['parent_plan']['audio_bindings'][k] for k in ('silence', 'noise')}
    for path in (PARENT/'native-plan.json', PARENT/'assets.json', CONTROLS/'native-plan.json', SOURCE/'input-capsule.json'):
        require(reg['pins'].get(str(path.resolve())) == file_hash(path), 'Original acquisition origin missing')
    require(same(read(folder/'assets.json'), assets), 'Copied assets changed')
    expected = native.build_plan(parent, controls, assets, fingerprint(read(folder/'input-proof.json')))
    require(same(expected, read(folder/'native-plan.json')), 'Masked request differs from original views')
    for path,digest in expected['attention_source']['files'].items(): require(reg['pins'].get(path) == digest, 'Attention source pin missing')
    return reg


def prepare(folder=FOLDER):
    folder = Path(folder).resolve()
    with locked(folder):
        require(set(folder.iterdir()) == {folder/'.lock'}, 'Masked acquisition already attempted')
        write_json(folder/'diagnostic.json', {'version': VERSION, 'status': 'preparing', 'local_seconds': 0., 'work_seconds': 0.,
                   'stages': [], 'passed': False, 'original_backend_restored': True, 'created_utc': now()})
        try:
            with phase(folder, 'direct_input_preparation', 180):
                parent, controls, assets, proof, pins = direct_inputs()
                plan = native.build_plan(parent, controls, assets, fingerprint(proof))
                for name,value in [('native-plan.json',plan), ('assets.json',assets), ('input-proof.json',proof)]: immutable(folder/name,value)
                pins.update({str(p.resolve()):file_hash(p) for p in [*producers(),folder/'native-plan.json',folder/'assets.json',folder/'input-proof.json']})
                pins.update(plan['attention_source']['files']); pins[plan['model_config']['path']] = plan['model_config']['sha256']
                reg = {'version':VERSION,'policy':native.POLICY,'pins':pins};reg['registration_sha256']=fingerprint(reg);immutable(folder/'registration.json',reg)
            value=state(folder);value.update(status='prepared',registration_sha256=reg['registration_sha256']);write_json(folder/'diagnostic.json',value);return value
        except BaseException as exc: base._failed(folder,exc);raise


def receipt_path(folder,number,slot): return folder/'worker/receipts'/f'{number:02d}-{slot["slot_id"]}.json'


def observation_envelope(folder,plan,outputs):
    parent=plan['parent_plan']; rows=[]; attestations=[]; control_receipts=[]
    for number,slot in enumerate(plan['slots'],1):
        value=outputs[slot['slot_id']];path=receipt_path(folder,number,slot);request=native.build_request(plan,slot['slot_id'])
        if slot['kind']=='control':
            control_receipts.append({'slot_id':slot['slot_id'],'path':str(path),'sha256':file_hash(path),'attention_sha256':value['attention_sha256']});continue
        i=slot['owner_id'];observation_id=f'masked-source-owner-{i:03d}';audio=request['capture_request']['audio']
        rows.append({'observation_id':observation_id,**deepcopy(parent['geometry'][i-1]),'raw_text':value['raw_text'],
            'text':value['parsed_text'],'detected_language':value['native_language'],'native_status':'empty' if value['parsed_text']=='' else 'complete',
            'protocol_category':native.capture.protocol_category(value['raw_text'],value['parsed_text']),
            'audio_sha256':audio['sha256'],'pcm_sha256':audio['pcm_sha256'],
            'receipt':{'path':str(path),'sha256':file_hash(path),'request_sha256':native.ec._hash(request)}})
        attestations.append({'observation_id':observation_id,'attention_sha256':value['attention_sha256']})
    return {'version':native.OBSERVATIONS_VERSION,'plan_sha256':plan['plan_sha256'],
        **{k:parent[k] for k in ('source_rows_sha256','original_mono_sha256','original_mono_frames','sample_rate')},
        'execution_identity':deepcopy(plan['execution_identity']),'conditioning':deepcopy(native.CONDITIONING),
        'audio_attention':{'policy':deepcopy(native.ATTENTION_POLICY),'source_identity':deepcopy(plan['attention_source']),
            'encoder_config':deepcopy(plan['encoder_config']),'model_config':deepcopy(plan['model_config']),
            'control_receipts':control_receipts,'observation_attestations':attestations},
        'observations':rows,'observations_sha256':native.ec._hash(rows)}


def verify_worker(folder,plan,assets):
    worker=native._sealed(read(folder/'worker/worker-result.json'));spec=read(folder/'worker-spec.json')
    proc=read(folder/'worker-process.json');life=read(folder/'lifecycle.json');original=read(folder/'original-backend.json')
    require(life['restored'] is True and life['owned_worker_stopped'] is True and life['error_type'] is None
            and same(life['original_backend'],original['model']) and proc['exit_code']==0 and type(proc['pid']) is int and proc['pid']>0, 'Owned lifecycle incomplete')
    require(spec['version']==native.VERSION and spec['plan_path']==str(folder/'native-plan.json')
        and spec['plan_file_sha256']==file_hash(folder/'native-plan.json') and spec['output_dir']==str(folder/'worker')
        and same(spec['assets'],assets) and spec['worker_sha256']==file_hash(Path(native.__file__))
        and type(spec['worker_seconds']) in (int,float) and 0<spec['worker_seconds']<=840, 'Worker specification changed')
    require(worker['version']==native.VERSION and worker['spec_sha256']==file_hash(folder/'worker-spec.json')
        and worker['plan_sha256']==plan['plan_sha256'] and worker['model_loads']==worker['model_load_attempts']==1
        and type(worker['seconds']) in (int,float) and 0<=worker['seconds']<=spec['worker_seconds']
        and worker['error_type'] is None and worker['status'] in ('complete','negative_control_output'), 'Worker identity/count/status changed')
    before=worker['runtime_before'];after=worker['runtime_after']
    require(same(before,after) and set(before)=={'model_dir','model_config','asr_backend','audio_backend','thinker_text_backend','audio_encoder_config','device','dtype'}
        and before['model_dir']==str(Path(assets['model']['model_dir']).resolve()) and same(before['model_config'],plan['model_config'])
        and before['asr_backend']=='transformers' and before['audio_backend']==plan['encoder_config']['backend']
        and same(before['audio_encoder_config'],plan['encoder_config']) and type(before['thinker_text_backend']) is str and bool(before['thinker_text_backend'])
        and before['device'].startswith('cuda') and before['dtype']=='torch.float16','Actual runtime backend/config changed')
    native.check_plan(plan,assets);processor=native.capture.base.load_processor(assets);outputs={};controls={};expected_files=set()
    require(len(worker['slots'])==68,'Fixed68 slots missing');previous=native._time(original['captured_utc']);retired=False
    for number,(slot,expected) in enumerate(zip(worker['slots'],plan['slots']),1):
        path=receipt_path(folder,number,expected)
        require(same({k:slot[k] for k in expected},expected) and slot['number']==number,'Slot order changed')
        if retired:
            require(slot['status']=='undispatched' and slot['receipt'] is None,'Dispatched after negative control');continue
        require(slot['status'] in ('empty','complete') and slot['receipt']==str(path),'Missing complete slot')
        receipt=read(path);request=native.build_request(plan,slot['slot_id']);reservation=read(path.with_suffix('.started.json'))
        require(set(reservation)=={'version','request_sha256','started_utc'} and reservation['version']==native.VERSION
            and reservation['request_sha256']==native.ec._hash(request) and previous<=native._time(reservation['started_utc'])<=native._time(receipt['started_utc'])
            and slot['status']==receipt['status'],'Reservation/order changed')
        previous=native._time(receipt['finished_utc']);expected_files.update({path,path.with_suffix('.started.json')})
        output=native.validate_native_receipt(receipt,request,plan,assets=assets,artifact_root=folder/'worker',processor=processor);outputs[slot['slot_id']]=output
        if slot['kind']=='control': controls[slot['slot_id']]=native.control_result(output);retired=not controls[slot['slot_id']]['passed']
    require(set((folder/'worker/receipts').glob('*.json'))==expected_files and worker['asr_calls']==len(outputs)
        and same(worker['controls'],controls) and worker['real_audio_calls']==sum(k.startswith('owner-') for k in outputs)
        and worker['status']==('negative_control_output' if retired else 'complete'),'Attempt coverage/summary changed')
    result={'passed':not retired,'reason_code':'negative_control_output' if retired else 'complete_masked_observations',
        'asr_calls':len(outputs),'control_calls':len(controls),'real_audio_calls':worker['real_audio_calls'],'model_loads':1,
        'controls':controls,'new_recap_calls':0,'candidate_generated':False,'score':None,'accuracy_verified':False,'independent_recognizer_vote':False}
    envelope=None if retired else observation_envelope(folder,plan,outputs)
    result.update(native_complete=not retired, runtime_backend=deepcopy(before),
        attention_policy=deepcopy(native.ATTENTION_POLICY), attention_source_sha256=native.ec._hash(plan['attention_source']),
        source_rows_sha256=plan['parent_plan']['source_rows_sha256'], geometry_sha256=native.ec._hash(plan['parent_plan']['geometry']),
        observation_envelope_sha256=None if envelope is None else native.ec._hash(envelope))
    return result,envelope


def execute(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(state(folder)['status']=='prepared','Masked acquisition cannot resume or reroll')
        previous=None;captured=False;process=None;failure=None
        try:
            with phase(folder,'input_replay'):
                reg=validate_registration(folder);require(state(folder)['registration_sha256']==reg['registration_sha256'],'State registration changed')
            value=state(folder);value.update(status='running',started_utc=now());write_json(folder/'diagnostic.json',value)
            with phase(folder,'masked_source_worker') as deadline:
                previous=base.late._backend_identity(base.rw._request_json(base.rw.ADMIN+'/status',admin=True));captured=True
                immutable(folder/'original-backend.json',{'model':previous,'captured_utc':now()})
                value=state(folder);value['original_backend_restored']=False;write_json(folder/'diagnostic.json',value)
                base.rw._request_json(base.rw.ADMIN+'/unload',{},admin=True,timeout=60)
                require(base.late._backend_identity(base.rw._request_json(base.rw.ADMIN+'/status',admin=True)) is None,'Backend not unloaded')
                assets=read(folder/'assets.json');seconds=deadline-time.monotonic();require(seconds>0,'No worker time remains')
                spec={'version':native.VERSION,'plan_path':str(folder/'native-plan.json'),'plan_file_sha256':file_hash(folder/'native-plan.json'),
                    'output_dir':str(folder/'worker'),'assets':assets,'worker_seconds':seconds,
                    'work_deadline_utc':(datetime.now(timezone.utc)+timedelta(seconds=seconds)).isoformat(),'worker_sha256':file_hash(Path(native.__file__))}
                immutable(folder/'worker-spec.json',spec)
                with (folder/'worker-process.log').open('xb') as log:
                    process=subprocess.Popen([assets['interpreter'],str(Path(native.__file__).resolve()),'--spec',str(folder/'worker-spec.json')],
                        cwd=ROOT,env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'},stdout=log,stderr=subprocess.STDOUT)
                    code=process.wait(timeout=max(.1,deadline-time.monotonic()))
                immutable(folder/'worker-process.json',{'pid':process.pid,'exit_code':code,'finished_utc':now()});require(code==0,'Masked worker failed')
        except BaseException as exc:failure=exc
        finally:
            if captured:
                try:
                    base.restore(folder,previous,process)
                    if read(folder/'lifecycle.json')['seconds']>360:raise base.rw.LocalBudgetExceeded('Cleanup exceeded360s reserve')
                except BaseException as exc:
                    if failure is None:failure=exc
                    else:failure.add_note('Owned restoration failed: '+type(exc).__name__)
        if state(folder)['local_seconds']>1200 and failure is None:failure=base.rw.LocalBudgetExceeded('Inclusive1200s exceeded')
        if failure is not None:base._failed(folder,failure);raise failure
        try:
            with phase(folder,'final_native_replay'):
                validate_registration(folder);result,envelope=verify_worker(folder,read(folder/'native-plan.json'),read(folder/'assets.json'))
                if envelope is not None:immutable(folder/'observations.json',envelope)
                artifacts={str(p.resolve()):file_hash(p) for p in folder.rglob('*') if p.is_file() and p.name not in ('.lock','diagnostic.json','result.json')}
            value=state(folder);value.update(status='complete' if result['passed'] else 'retired',passed=result['passed'],reason_code=result['reason_code'],finished_utc=now())
            write_json(folder/'diagnostic.json',value);artifacts[str(folder/'diagnostic.json')]=file_hash(folder/'diagnostic.json')
            result.update(version=VERSION,local_seconds=value['local_seconds'],work_seconds=value['work_seconds'],original_backend_restored=True,artifacts=artifacts)
            immutable(folder/'result.json',result);return result
        except BaseException as exc:base._failed(folder,exc);raise


def replay_acquisition(folder,track=None):
    folder=Path(folder).resolve();reg=validate_registration(folder);value=state(folder);result=read(folder/'result.json')
    require(value['version']==VERSION and value['status']=='complete' and value['passed'] is True and value['original_backend_restored'] is True
        and value['registration_sha256']==reg['registration_sha256'] and 0<=value['work_seconds']<=840
        and value['work_seconds']<=value['local_seconds']<=1200,'Masked acquisition not complete/restored/in budget')
    expected,envelope=verify_worker(folder,read(folder/'native-plan.json'),read(folder/'assets.json'))
    require(all(same(result.get(k),v) for k,v in expected.items()) and result['version']==VERSION
        and result['local_seconds']==value['local_seconds'] and result['work_seconds']==value['work_seconds']
        and result['original_backend_restored'] is True and same(envelope,read(folder/'observations.json')),'Neutral observation/result changed')
    artifacts={str(p.resolve()):file_hash(p) for p in folder.rglob('*') if p.is_file() and p.name not in ('.lock','result.json')}
    require(same(artifacts,result['artifacts']),'Acquisition artifact coverage changed')
    pins={**reg['pins'],**artifacts,str(folder/'result.json'):file_hash(folder/'result.json')}
    if track is not None:
        for path in pins:track(Path(path))
    return {'envelope':envelope,'result':{k:v for k,v in result.items() if k!='artifacts'},
        'source_bindings':read(folder/'input-proof.json')['source_bindings'],'local_seconds':value['local_seconds'],
        'work_seconds':value['work_seconds'],'original_backend_restored':True,'pins':pins}


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--execute',action='store_true');args=parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        result=prepare() if args.prepare else execute();logging.info('Masked acquisition status=%s passed=%s',result.get('status'),result.get('passed'));return 0
    except Exception as exc:logging.error('Masked acquisition stopped: %s',type(exc).__name__);return 1
if __name__=='__main__':raise SystemExit(main())
