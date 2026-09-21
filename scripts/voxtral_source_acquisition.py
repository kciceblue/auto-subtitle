"""One isolated Voxtral family acquisition; direct source pins and local lifecycle."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import hashlib
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import long_source_acquisition as mechanics
from src import voxtral_source_native as native
from src import auto_source_native as source_geometry
from src import evidence_context as ec
from src.workflow_state import file_hash,fingerprint,write_json
from src.contextual_review_contract import require
VERSION='voxtral-source-acquisition-1'
FOLDER=ROOT/'output/quality-voxtral-source-20260916'
PARENT=ROOT/'output/quality-auto-source-20260915'
SOURCE=ROOT/'output/quality-contextual-asr-projected-20260915'
SETUP=ROOT/'output/voxtral-setup-20260916'
MODEL=ROOT/'models/voxtral-mini-4b-realtime-2602'
INTERPRETER=ROOT/'.venv-voxtral/bin/python'
PLAN=ROOT/'design/quality-voxtral-source-acquisition-20260916.md'
REVISION='2769294da9567371363522aac9bbcfdd19447add'
TOTAL_SECONDS,WORK_SECONDS,CLEANUP_SECONDS=1800,1380,420
read,same,now,state=mechanics.read,mechanics.same,mechanics.now,mechanics.state
immutable,locked,phase,restore,failed=mechanics.immutable,mechanics.locked,mechanics.phase,mechanics.restore,mechanics.failed
backend,late_audio,ADMIN=mechanics.backend,mechanics.late_audio,mechanics.ADMIN


def direct_inputs():
    paths=[PARENT/'registration.json',PARENT/'native-plan.json',PARENT/'assets.json',PARENT/'audio-preparation.json',
        SOURCE/'input-capsule.json',SOURCE/'audio-preparation.json',SETUP/'hf-resolution.json',SETUP/'assets.json',SETUP/'runtime.json']
    reg=read(paths[0]);require(reg['registration_sha256']==fingerprint({k:v for k,v in reg.items() if k!='registration_sha256'}),'Original source registration changed')
    for path in paths[1:6]:require(reg['pins'].get(str(path.resolve()))==file_hash(path),'Direct original source pin changed')
    parent=read(PARENT/'native-plan.json');capsule=read(SOURCE/'input-capsule.json');gain=read(PARENT/'audio-preparation.json')
    rows=capsule['source_rows'];detector=capsule['detector_record'];geometry=source_geometry.owner_geometry(rows,detector)
    require(parent['source_rows_sha256']==ec._hash(rows) and parent['original_mono_sha256']==capsule['detector_provenance']['mono_sha256']
        and same(parent['geometry'],geometry) and same(gain,read(SOURCE/'audio-preparation.json')),'Original owner/gain binding changed')
    bindings={f'owner-{i:03d}':deepcopy(parent['audio_bindings'][str(i)]) for i in range(1,67)}
    paths.extend(Path(v['path']) for v in bindings.values());paths.append(Path(capsule['mono_path']))
    require(file_hash(Path(capsule['mono_path']))==parent['original_mono_sha256'],'Original mono changed')
    resolution=read(SETUP/'hf-resolution.json');assets=read(SETUP/'assets.json');runtime=read(SETUP/'runtime.json')
    require(resolution['repo_id']==assets['repo_id']=='mistralai/Voxtral-Mini-4B-Realtime-2602' and resolution['revision']==assets['revision']==REVISION
        and resolution['private'] is False and resolution['gated'] is False and assets['status']=='complete'
        and runtime['status']=='imports_verified_no_weights_loaded' and runtime['base_protected_files_unchanged'] is True,'Pinned Voxtral setup incomplete')
    require(Path(runtime['interpreter']).absolute()==INTERPRETER.absolute() and runtime['torch_version']=='2.10.0+cu128'
        and runtime['transformers_version']=='5.3.0' and runtime['mistral_common_version']=='1.11.7','Declared isolated runtime changed')
    for item in assets['files']:
        path=Path(item['path']);require(path.parent==MODEL and path.stat().st_size==item['size'] and file_hash(path)==item['sha256'],'Pinned public model asset changed');paths.append(path)
    proof={'source_rows_sha256':ec._hash(rows),'original_mono_sha256':parent['original_mono_sha256'],
        'source_capsule_sha256':file_hash(SOURCE/'input-capsule.json'),'shared_gain_sha256':file_hash(PARENT/'audio-preparation.json'),
        'geometry_sha256':ec._hash(geometry),'setup_assets_sha256':file_hash(SETUP/'assets.json'),'model_revision':REVISION,
        'same_saved_owner_pcm':True,'fresh_negative_controls':2,'old_native_semantics_replayed':False,
        'source_bindings':{k:deepcopy(capsule[k]) for k in ('source_provenance','detector_provenance')}}
    return capsule,bindings,proof,{str(p.resolve()):file_hash(p) for p in paths}


def control_arrays():
    import numpy as np
    return {'silence':np.zeros(192000,dtype=np.float32),'noise':np.random.Generator(np.random.PCG64(20260915)).uniform(-.01,.01,192000).astype(np.float32)}


def prepare_controls(folder,bindings):
    import soundfile as sf
    directory=folder/'audio';directory.mkdir();result=deepcopy(bindings)
    for key,value in control_arrays().items():
        path=directory/(key+'.wav');sf.write(path,value,16000,subtype='FLOAT',format='WAV')
        result[key]={'path':str(path.resolve()),'sha256':file_hash(path),'pcm_sha256':hashlib.sha256(value.tobytes()).hexdigest(),
            'frames':192000,'sample_rate':16000,'dtype':'float32'}
        require(native._read_audio(result[key]).tobytes()==value.tobytes(),'Fixed control FLOAT roundtrip changed')
    return result


def producers():
    names=('voxtral_source_native','auto_source_native','long_source_grid','long_source_native','masked_source_native','qwen_audio_attention',
        'contextual_asr_native','contextual_asr_auto_native','contextual_asr_requests','evidence_context','contextual_review_contract',
        'late_audio','local_backend','workflow_state')
    return [PLAN,Path(__file__),Path(mechanics.__file__),*[ROOT/'src'/(name+'.py') for name in names]]


def validate_registration(folder):
    reg=read(folder/'registration.json')
    require(reg['version']==VERSION and same(reg['policy'],native.POLICY)
        and reg['registration_sha256']==fingerprint({k:v for k,v in reg.items() if k!='registration_sha256'}),'Voxtral registration changed')
    required={str(p.resolve()) for p in producers()}|{str(folder/name) for name in ('native-plan.json','assets.json','input-proof.json')}
    require(required<=set(reg['pins']),'Required local producer/input pin missing')
    for path,digest in reg['pins'].items():require(file_hash(Path(path))==digest,'Pinned Voxtral input changed')
    for path in (PARENT/'native-plan.json',PARENT/'audio-preparation.json',SOURCE/'input-capsule.json',SETUP/'assets.json',SETUP/'runtime.json'):
        require(reg['pins'].get(str(path.resolve()))==file_hash(path),'Direct source/model origin missing')
    capsule=read(SOURCE/'input-capsule.json');parent=read(PARENT/'native-plan.json');assets=read(folder/'assets.json');plan=read(folder/'native-plan.json')
    proof=read(folder/'input-proof.json')
    require(same(proof['source_bindings'],{k:capsule[k] for k in ('source_provenance','detector_provenance')})
        and proof['source_rows_sha256']==ec._hash(capsule['source_rows'])
        and proof['original_mono_sha256']==capsule['detector_provenance']['mono_sha256']
        and proof['source_capsule_sha256']==file_hash(SOURCE/'input-capsule.json')
        and proof['shared_gain_sha256']==file_hash(PARENT/'audio-preparation.json')
        and proof['geometry_sha256']==ec._hash(parent['geometry'])
        and proof['setup_assets_sha256']==file_hash(SETUP/'assets.json') and proof['model_revision']==REVISION,
        'Direct source/setup proof changed')
    expected=native.build_plan(capsule['source_rows'],capsule['detector_record'],capsule['detector_provenance']['mono_sha256'],
        plan['audio_bindings'],assets,fingerprint(proof))
    require(same(plan,expected),'Voxtral request plan changed')
    for i in range(1,67):require(same(plan['audio_bindings'][f'owner-{i:03d}'],parent['audio_bindings'][str(i)]),'Owner PCM differs from pinned original acquisition')
    for binding in plan['audio_bindings'].values():require(reg['pins'].get(binding['path'])==binding['sha256'],'Prepared FLOAT PCM not pinned')
    for key,value in control_arrays().items():
        require(native._read_audio(plan['audio_bindings'][key]).tobytes()==value.tobytes(),'Fixed negative-control PCM changed')
    native.validate_assets(assets,plan['execution_identity'])
    return reg


def prepare(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(set(folder.iterdir())=={folder/'.lock'},'Voxtral acquisition already attempted')
        write_json(folder/'diagnostic.json',{'version':VERSION,'status':'preparing','local_seconds':0.,'work_seconds':0.,'stages':[],
            'passed':False,'original_backend_restored':True,'created_utc':now()})
        try:
            with phase(folder,'direct_source_model_preparation',240):
                capsule,bindings,proof,pins=direct_inputs();bindings=prepare_controls(folder,bindings)
                assets=native.build_assets(MODEL,INTERPRETER)
                plan=native.build_plan(capsule['source_rows'],capsule['detector_record'],capsule['detector_provenance']['mono_sha256'],bindings,assets,fingerprint(proof))
                for name,value in [('native-plan.json',plan),('assets.json',assets),('input-proof.json',proof)]:immutable(folder/name,value)
                pins.update({str(p.resolve()):file_hash(p) for p in [*producers(),*[p for p in folder.rglob('*') if p.is_file() and p.name not in ('.lock','diagnostic.json')]]})
                for item in assets['model_files']+assets['runtime_files']:pins[item['path']]=item['sha256']
                reg={'version':VERSION,'policy':native.POLICY,'pins':pins};reg['registration_sha256']=fingerprint(reg);immutable(folder/'registration.json',reg)
            value=state(folder);value.update(status='prepared',registration_sha256=reg['registration_sha256']);write_json(folder/'diagnostic.json',value);return value
        except BaseException as exc:failed(folder,exc);raise


def receipt_path(folder,number,slot):return folder/'worker/receipts'/f'{number:02d}-{slot["slot_id"]}.json'


def observation_envelope(folder,plan,outputs):
    rows=[];controls=[]
    for number,slot in enumerate(plan['slots'],1):
        value=outputs[slot['slot_id']];path=receipt_path(folder,number,slot);request=native.build_request(plan,slot['slot_id'])
        if slot['kind']=='control':
            controls.append({'slot_id':slot['slot_id'],'path':str(path),'sha256':file_hash(path)});continue
        owner=slot['owner_id'];view=plan['geometry'][owner-1]
        rows.append({'observation_id':request['observation_id'],**deepcopy(view),'raw_text':value['raw_text'],'text':value['parsed_text'],
            'detected_language':value['native_language'],'language_origin':'not_reported','native_status':'empty' if value['parsed_text']=='' else 'complete',
            'audio_sha256':request['audio']['sha256'],'pcm_sha256':request['audio']['pcm_sha256'],'coverage':deepcopy(value['coverage']),
            'protocol_category':value['protocol_category'],'special_token_insertions':deepcopy(value['special_token_insertions']),
            'receipt':{'path':str(path),'sha256':file_hash(path),'request_sha256':ec._hash(request)}})
    return {'version':'voxtral-source-observations-1','plan_sha256':plan['plan_sha256'],
        **{k:plan[k] for k in ('source_rows_sha256','original_mono_sha256','original_mono_frames','sample_rate')},
        'execution_identity':deepcopy(plan['execution_identity']),'conditioning':{'context':'','language':None,'hotwords':[]},
        'recognizer_family':'Voxtral-Mini-4B-Realtime-2602','language_reported':False,'word_alignment':'unknown',
        'control_receipts':controls,'observations':rows,'observations_sha256':ec._hash(rows)}


sealed,memory_peaks=mechanics.sealed,mechanics.memory_peaks


def verify_worker(folder,plan,assets):
    worker=sealed(read(folder/'worker/worker-result.json'));spec=read(folder/'worker-spec.json');proc=read(folder/'worker-process.json')
    life=read(folder/'lifecycle.json');original=read(folder/'original-backend.json')
    require(life['restored'] is True and life['owned_worker_stopped'] is True and life['error_type'] is None and same(life['original_backend'],original['model'])
        and type(proc['exit_code']) is int and proc['exit_code']==0 and type(proc['pid']) is int and proc['pid']>0,'Owned Voxtral lifecycle incomplete')
    require(spec['version']==native.VERSION and spec['plan_path']==str(folder/'native-plan.json') and spec['plan_file_sha256']==file_hash(folder/'native-plan.json')
        and spec['output_dir']==str(folder/'worker') and same(spec['assets'],assets) and spec['worker_sha256']==file_hash(Path(native.__file__))
        and type(spec['worker_seconds']) in (int,float) and math.isfinite(spec['worker_seconds']) and 0<spec['worker_seconds']<=WORK_SECONDS,'Voxtral worker spec changed')
    require(worker['version']==native.VERSION and worker['spec_sha256']==file_hash(folder/'worker-spec.json') and worker['plan_sha256']==plan['plan_sha256']
        and type(worker['model_loads']) is int and worker['model_loads']==1 and type(worker['model_load_attempts']) is int and worker['model_load_attempts']==1
        and type(worker['seconds']) in (int,float) and math.isfinite(worker['seconds']) and 0<=worker['seconds']<=spec['worker_seconds'] and worker['error_type'] is None
        and worker['status'] in ('complete','negative_control_output'),'Voxtral worker binding/status changed')
    memory=memory_peaks(worker);before=native._validate_runtime(worker['runtime_before'])
    require(same(before,worker['runtime_after']),'Actual Voxtral runtime changed between calls')
    native.check_plan(plan,assets);processor=native.load_processor(assets);outputs={};controls={};expected_files=set();retired=False
    previous=datetime.fromisoformat(original['captured_utc']);require(len(worker['slots'])==68,'Exactly 68 Voxtral slots required')
    for number,(slot,expected) in enumerate(zip(worker['slots'],plan['slots']),1):
        require(same({k:slot[k] for k in expected},expected) and type(slot['number']) is int and slot['number']==number,'Voxtral slot order changed')
        if retired:require(slot['status']=='undispatched' and slot['receipt'] is None,'Dispatch after negative control');continue
        path=receipt_path(folder,number,slot);receipt=read(path);reservation=read(path.with_suffix('.started.json'));request=native.build_request(plan,slot['slot_id'])
        require(slot['status'] in ('empty','complete') and slot['receipt']==str(path) and slot['status']==receipt['status']
            and set(reservation)=={'version','request_sha256','started_utc'} and reservation['version']==native.VERSION
            and reservation['request_sha256']==ec._hash(request) and previous<=datetime.fromisoformat(reservation['started_utc'])<=datetime.fromisoformat(receipt['started_utc']),'Voxtral reservation/order changed')
        previous=datetime.fromisoformat(receipt['finished_utc']);expected_files.update({path,path.with_suffix('.started.json')})
        output=native.validate_native_receipt(receipt,request,plan,assets=assets,artifact_root=folder/'worker',processor=processor);outputs[slot['slot_id']]=output
        require(output['coverage']['physical_audio_complete'] is True,'Physical audio coverage incomplete')
        if slot['kind']=='control':controls[slot['slot_id']]=native.control_result(output);retired=not controls[slot['slot_id']]['passed']
    real=sum(k.startswith('owner-') for k in outputs)
    require(set((folder/'worker/receipts').glob('*.json'))==expected_files and type(worker['asr_calls']) is int and worker['asr_calls']==len(outputs)
        and type(worker['real_audio_calls']) is int and worker['real_audio_calls']==real and same(worker['controls'],controls)
        and worker['status']==('negative_control_output' if retired else 'complete'),'Voxtral attempt coverage changed')
    envelope=None if retired else observation_envelope(folder,plan,outputs)
    return {**memory,'passed':not retired,'native_complete':not retired,'reason_code':'negative_control_output' if retired else 'complete_voxtral_observations',
        'asr_calls':len(outputs),'real_audio_calls':real,'control_calls':len(controls),'model_loads':1,'controls':controls,
        'runtime_backend':before,'runtime_policy':plan['runtime_policy'],'model_config_sha256':ec._hash(plan['model_config']),
        'source_rows_sha256':plan['source_rows_sha256'],'geometry_sha256':ec._hash(plan['geometry']),
        'execution_identity':deepcopy(plan['execution_identity']),
        'observation_envelope_sha256':None if envelope is None else ec._hash(envelope),
        'candidate_generated':False,'new_recap_calls':0,'score':None,'accuracy_verified':False,'independent_recognizer_vote':False},envelope


def execute(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(state(folder)['status']=='prepared','Voxtral acquisition cannot resume or reroll');captured=False;previous=None;process=None;failure=None
        try:
            with phase(folder,'input_replay'):
                reg=validate_registration(folder);require(state(folder)['registration_sha256']==reg['registration_sha256'],'State binding changed')
            value=state(folder);value.update(status='running',started_utc=now());write_json(folder/'diagnostic.json',value)
            with phase(folder,'voxtral_source_worker') as deadline:
                previous=late_audio._backend_identity(backend._request_json(ADMIN+'/status',admin=True));captured=True
                immutable(folder/'original-backend.json',{'model':previous,'captured_utc':now()});value=state(folder);value['original_backend_restored']=False;write_json(folder/'diagnostic.json',value)
                backend._request_json(ADMIN+'/unload',{},admin=True,timeout=60)
                require(late_audio._backend_identity(backend._request_json(ADMIN+'/status',admin=True)) is None,'Backend not unloaded')
                assets=read(folder/'assets.json');seconds=deadline-time.monotonic();require(seconds>0,'No Voxtral worker time remains')
                spec={'version':native.VERSION,'plan_path':str(folder/'native-plan.json'),'plan_file_sha256':file_hash(folder/'native-plan.json'),
                    'output_dir':str(folder/'worker'),'assets':assets,'worker_seconds':seconds,
                    'work_deadline_utc':(datetime.now(timezone.utc)+timedelta(seconds=seconds)).isoformat(),'worker_sha256':file_hash(Path(native.__file__))}
                immutable(folder/'worker-spec.json',spec)
                with (folder/'worker-process.log').open('xb') as log:
                    process=subprocess.Popen([assets['interpreter'],str(Path(native.__file__).resolve()),'--spec',str(folder/'worker-spec.json')],cwd=ROOT,
                        env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'},stdout=log,stderr=subprocess.STDOUT)
                    code=process.wait(timeout=max(.1,deadline-time.monotonic()))
                immutable(folder/'worker-process.json',{'pid':process.pid,'exit_code':code,'finished_utc':now()});require(code==0,'Voxtral worker failed')
        except BaseException as exc:failure=exc
        finally:
            if captured:
                try:restore(folder,previous,process)
                except BaseException as exc:
                    if failure is None:failure=exc
                    else:failure.add_note('Owned restoration failed: '+type(exc).__name__)
        if failure is not None:failed(folder,failure);raise failure
        try:
            with phase(folder,'final_native_replay'):
                validate_registration(folder);result,envelope=verify_worker(folder,read(folder/'native-plan.json'),read(folder/'assets.json'))
                if envelope is not None:immutable(folder/'observations.json',envelope)
                artifacts={str(p.resolve()):file_hash(p) for p in folder.rglob('*') if p.is_file() and p.name not in ('.lock','diagnostic.json','result.json')}
            value=state(folder);value.update(status='complete' if result['passed'] else 'retired',passed=result['passed'],reason_code=result['reason_code'],finished_utc=now());write_json(folder/'diagnostic.json',value)
            artifacts[str(folder/'diagnostic.json')]=file_hash(folder/'diagnostic.json');result.update(version=VERSION,local_seconds=value['local_seconds'],work_seconds=value['work_seconds'],original_backend_restored=True,artifacts=artifacts)
            immutable(folder/'result.json',result);return result
        except BaseException as exc:failed(folder,exc);raise


def replay_acquisition(folder):
    folder=Path(folder).resolve();reg=validate_registration(folder);value=state(folder);result=read(folder/'result.json')
    require(value['status']=='complete' and value['passed'] is True and value['original_backend_restored'] is True
        and value['registration_sha256']==reg['registration_sha256'] and 0<=value['work_seconds']<=WORK_SECONDS
        and value['work_seconds']<=value['local_seconds']<=TOTAL_SECONDS,'Voxtral acquisition not complete/restored/in budget')
    expected,envelope=verify_worker(folder,read(folder/'native-plan.json'),read(folder/'assets.json'))
    require(all(same(result.get(k),v) for k,v in expected.items()) and same(envelope,read(folder/'observations.json'))
        and result['version']==VERSION and result['local_seconds']==value['local_seconds'] and result['work_seconds']==value['work_seconds']
        and result['original_backend_restored'] is True,'Voxtral terminal observations changed')
    artifacts={str(p.resolve()):file_hash(p) for p in folder.rglob('*') if p.is_file() and p.name not in ('.lock','result.json')}
    require(same(artifacts,result['artifacts']),'Voxtral artifact coverage changed')
    return {'envelope':envelope,'result':{k:v for k,v in result.items() if k!='artifacts'},'source_bindings':read(folder/'input-proof.json')['source_bindings'],
        'pins':{**reg['pins'],**artifacts,str(folder/'result.json'):file_hash(folder/'result.json')}}


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--execute',action='store_true');args=parser.parse_args();logging.basicConfig(level=logging.INFO)
    try:
        value=prepare() if args.prepare else execute();logging.info('Voxtral acquisition status=%s passed=%s',value.get('status'),value.get('passed'));return 0
    except Exception as exc:logging.error('Voxtral acquisition stopped: %s',type(exc).__name__);return 1
if __name__=='__main__':raise SystemExit(main())
