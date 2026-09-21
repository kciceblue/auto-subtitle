"""Conditional14-call long-ASR acquisition. No writer, reviewer or score dependency."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src import long_source_grid as geometry
from src import evidence_context as ec
from src import long_source_native as native
from src import local_backend as backend
from src import late_audio
from src.contextual_review_contract import require, strict_json
from src.workflow_state import file_hash, fingerprint, write_json
VERSION='long-source-acquisition-1'
FOLDER=ROOT/'output/quality-long-source-20260916'
PARENT=ROOT/'output/quality-auto-source-20260915'
SOURCE=ROOT/'output/quality-contextual-asr-projected-20260915'
PLAN=ROOT/'design/quality-long-source-acquisition-20260916.md'
ADMIN='http://127.0.0.1:8089/admin'
TOTAL_SECONDS,WORK_SECONDS,CLEANUP_SECONDS=1800,1380,420


def read(path):return strict_json(Path(path).read_bytes())
def same(a,b):return fingerprint(a)==fingerprint(b)
def now():return datetime.now(timezone.utc).isoformat()
def state(folder):return read(folder/'diagnostic.json')
def sealed(value):
    require(value['receipt_sha256']==ec._hash({k:v for k,v in value.items() if k!='receipt_sha256'}),'Receipt seal changed');return value

def immutable(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        import json
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    path.chmod(0o444)


@contextmanager
def locked(folder):
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


class LocalBudgetExceeded(BaseException):pass


@contextmanager
def phase(folder,name,ceiling=None):
    start=time.monotonic();value=state(folder)
    require(all(type(value[k]) in (int,float) and math.isfinite(value[k]) and value[k]>=0 for k in ('local_seconds','work_seconds')),'Invalid time ledger')
    allowance=min(TOTAL_SECONDS-value['local_seconds'],WORK_SECONDS-value['work_seconds'])
    if ceiling is not None:allowance=min(allowance,ceiling)
    require(allowance>0,'Long acquisition budget exhausted')
    old_handler=signal.getsignal(signal.SIGALRM);old_timer=signal.getitimer(signal.ITIMER_REAL)
    if old_timer[0]:allowance=min(allowance,old_timer[0])
    def expired(signum,frame):raise LocalBudgetExceeded('Long acquisition work deadline')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,allowance);error=None
    try:yield start+allowance
    except BaseException as exc:error=type(exc).__name__;raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler);elapsed=time.monotonic()-start
        if old_timer[0]:signal.setitimer(signal.ITIMER_REAL,max(.001,old_timer[0]-elapsed),old_timer[1])
        value=state(folder);value['local_seconds']+=elapsed;value['work_seconds']+=elapsed
        over=elapsed>allowance or value['local_seconds']>TOTAL_SECONDS or value['work_seconds']>WORK_SECONDS
        value['stages'].append({'stage':name,'seconds':elapsed,'error_type':error or ('LocalBudgetExceeded' if over else None)})
        write_json(folder/'diagnostic.json',value)
        if over and error is None:raise LocalBudgetExceeded('Long acquisition phase overrun')


def failed(folder,exc):
    value=state(folder);value.update(status='failed',passed=False,error_type=type(exc).__name__,finished_utc=now());write_json(folder/'diagnostic.json',value)


def direct_inputs():
    paths=[PARENT/'registration.json',PARENT/'native-plan.json',PARENT/'assets.json',PARENT/'audio-preparation.json',
           SOURCE/'input-capsule.json',SOURCE/'audio-preparation.json']
    reg=read(paths[0]);require(reg['registration_sha256']==fingerprint({k:v for k,v in reg.items() if k!='registration_sha256'}),'Original source registration changed')
    for path in paths[1:]:require(reg['pins'].get(str(path.resolve()))==file_hash(path),'Direct original input changed')
    parent=read(paths[1]);assets=read(paths[2]);capsule=read(SOURCE/'input-capsule.json');gain=read(PARENT/'audio-preparation.json')
    grid=geometry.build_grid(capsule['source_rows'],capsule['detector_record']['mono']['frames'])
    require(len(grid['source_owners'])==66 and len(grid['windows'])==12 and grid['source_rows_sha256']==parent['source_rows_sha256']
        and capsule['detector_provenance']['mono_sha256']==parent['original_mono_sha256']
        and same(gain,read(SOURCE/'audio-preparation.json')),'Original source/gain binding changed')
    mono=Path(capsule['mono_path']);require(file_hash(mono)==parent['original_mono_sha256'],'Original mono changed');paths.append(mono)
    proof={'original_mono_sha256':file_hash(mono),'source_rows_sha256':grid['source_rows_sha256'],
        'source_capsule_sha256':file_hash(SOURCE/'input-capsule.json'),'audio_preparation_sha256':file_hash(PARENT/'audio-preparation.json'),
        'grid_sha256':ec._hash(grid),'source_bindings':{k:deepcopy(capsule[k]) for k in ('source_provenance','detector_provenance')},
        'old_native_semantics_replayed':False,'internal_word_alignment':'unknown'}
    return capsule,grid,assets,gain,proof,{str(p.resolve()):file_hash(p) for p in paths}


def prepare_audio(folder,capsule,grid,gain):
    import numpy as np
    import soundfile as sf
    path=Path(capsule['mono_path']);digest=file_hash(path);info=sf.info(path);mono,rate=sf.read(path,dtype='float32',always_2d=False)
    require(info.channels==1 and info.subtype=='FLOAT' and rate==16000 and mono.ndim==1 and len(mono)==grid['original_mono_frames']
        and np.isfinite(mono).all() and digest==capsule['detector_provenance']['mono_sha256'] and file_hash(path)==digest,'Original finite mono domain changed')
    peak=float(np.max(np.abs(mono)));factor=min(1.,.99/peak) if peak>0 else 1.;scalar=np.float32(factor)
    expected={'version':'contextual-asr-audio-preparation-1','policy':'one_global_gain_for_all_real_crops',
        'original_mono_path':str(path.resolve()),'original_mono_sha256':digest,'original_frames':len(mono),'sample_rate':16000,
        'global_peak':peak,'target_peak':.99,'gain':factor,'gain_float32':float(scalar),
        'float32_formula':'np.multiply(original_float32_crop, np.float32(gain), dtype=np.float32)',
        'clipping_applied':False,'crop_selection_changed':False,'synthetic_controls_scaled':False}
    require(same(gain,expected),'Global shared gain changed')
    arrays={'silence':np.zeros(geometry.WINDOW_FRAMES,dtype=np.float32),
        'noise':np.random.Generator(np.random.PCG64(20260915)).uniform(-.01,.01,geometry.WINDOW_FRAMES).astype(np.float32)}
    arrays.update({w['window_id']:np.multiply(mono[w['crop_start_frame']:w['crop_end_frame']],scalar,dtype=np.float32) for w in grid['windows']})
    directory=folder/'audio';directory.mkdir();bindings={}
    for key,audio in arrays.items():
        path=directory/(key+'.wav');sf.write(path,audio,16000,subtype='FLOAT',format='WAV')
        bindings[key]={'path':str(path.resolve()),'sha256':file_hash(path),'pcm_sha256':hashlib.sha256(audio.tobytes()).hexdigest(),
            'frames':len(audio),'sample_rate':16000,'dtype':'float32'}
        require(native._read_audio(bindings[key]).tobytes()==audio.tobytes(),'Long prepared PCM roundtrip changed')
    immutable(folder/'audio-preparation.json',{'version':'long-source-audio-preparation-1','shared_gain':expected,
        'control_frames':geometry.WINDOW_FRAMES,'control_noise':{'generator':'PCG64','seed':20260915,'minimum':-.01,'maximum':.01,'dtype':'float32'},
        'clipping_applied':False,'synthetic_controls_scaled':False})
    return bindings


def producers():
    names=('long_source_native','long_source_grid','masked_source_native','qwen_audio_attention','auto_source_native',
        'contextual_asr_native','contextual_asr_auto_native','contextual_asr_requests','evidence_context','contextual_review_contract',
        'late_audio','local_backend','workflow_state')
    return [PLAN,Path(__file__),*[ROOT/'src'/(name+'.py') for name in names]]


def validate_registration(folder):
    reg=read(folder/'registration.json');require(reg['version']==VERSION and same(reg['policy'],native.POLICY)
        and reg['registration_sha256']==fingerprint({k:v for k,v in reg.items() if k!='registration_sha256'}),'Long registration changed')
    required={str(p.resolve()) for p in producers()}|{str(folder/name) for name in ('native-plan.json','assets.json','input-proof.json','grid.json','audio-preparation.json')}
    require(required<=set(reg['pins']),'Mandatory long input/producer missing')
    for path,digest in reg['pins'].items():require(file_hash(Path(path))==digest,'Pinned long input changed')
    for path in (PARENT/'assets.json',PARENT/'audio-preparation.json',SOURCE/'input-capsule.json'):
        require(reg['pins'].get(str(path.resolve()))==file_hash(path),'Original source origin missing')
    capsule=read(SOURCE/'input-capsule.json');grid=read(folder/'grid.json');assets=read(folder/'assets.json')
    geometry.validate_grid(grid,capsule['source_rows'],capsule['detector_record']['mono']['frames'])
    require(same(assets,read(PARENT/'assets.json')) and same(read(folder/'audio-preparation.json')['shared_gain'],read(PARENT/'audio-preparation.json')),'Copied assets/gain changed')
    plan=read(folder/'native-plan.json');require(same(plan,native.build_plan(grid,plan['audio_bindings'],assets,fingerprint(read(folder/'input-proof.json')))),'Long plan changed')
    for binding in plan['audio_bindings'].values():require(reg['pins'].get(binding['path'])==binding['sha256'],'Long PCM input pin missing')
    for path,digest in plan['attention_source']['files'].items():require(reg['pins'].get(path)==digest,'Attention source pin missing')
    return reg


def prepare(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(set(folder.iterdir())=={folder/'.lock'},'Long acquisition already attempted')
        write_json(folder/'diagnostic.json',{'version':VERSION,'status':'preparing','local_seconds':0.,'work_seconds':0.,'stages':[],
            'passed':False,'original_backend_restored':True,'created_utc':now()})
        try:
            with phase(folder,'direct_grid_audio_preparation',240):
                capsule,grid,assets,gain,proof,pins=direct_inputs();bindings=prepare_audio(folder,capsule,grid,gain)
                plan=native.build_plan(grid,bindings,assets,fingerprint(proof))
                for name,value in [('native-plan.json',plan),('assets.json',assets),('input-proof.json',proof),('grid.json',grid)]:immutable(folder/name,value)
                pins.update({str(p.resolve()):file_hash(p) for p in [*producers(),*[p for p in folder.rglob('*') if p.is_file() and p.name not in ('.lock','diagnostic.json')]]})
                pins.update(plan['attention_source']['files']);pins[plan['model_config']['path']]=plan['model_config']['sha256']
                reg={'version':VERSION,'policy':native.POLICY,'pins':pins};reg['registration_sha256']=fingerprint(reg);immutable(folder/'registration.json',reg)
            value=state(folder);value.update(status='prepared',registration_sha256=reg['registration_sha256']);write_json(folder/'diagnostic.json',value);return value
        except BaseException as exc:failed(folder,exc);raise


def receipt_path(folder,number,slot):return folder/'worker/receipts'/f'{number:02d}-{slot["slot_id"]}.json'


def observation_envelope(folder,plan,outputs):
    grid=plan['grid'];rows=[];attestations=[];controls=[]
    for number,slot in enumerate(plan['slots'],1):
        value=outputs[slot['slot_id']];path=receipt_path(folder,number,slot);request=native.build_request(plan,slot['slot_id'])
        if slot['kind']=='control':
            controls.append({'slot_id':slot['slot_id'],'path':str(path),'sha256':file_hash(path),'attention_sha256':value['attention_sha256']});continue
        view=next(w for w in grid['windows'] if w['window_id']==slot['window_id']);observation_id=request['observation_id']
        rows.append({'observation_id':observation_id,**deepcopy(view),'raw_text':value['raw_text'],'text':value['parsed_text'],
            'detected_language':value['native_language'],'native_status':'empty' if value['parsed_text']=='' else 'complete',
            'protocol_category':native.protocol_category(value['raw_text'],value['parsed_text']),
            'audio_sha256':request['audio']['sha256'],'pcm_sha256':request['audio']['pcm_sha256'],
            'receipt':{'path':str(path),'sha256':file_hash(path),'request_sha256':ec._hash(request)}})
        attestations.append({'observation_id':observation_id,'attention_sha256':value['attention_sha256']})
    proof=read(folder/'input-proof.json')
    return {'version':'long-source-observations-1','plan_sha256':plan['plan_sha256'],'source_rows_sha256':grid['source_rows_sha256'],
        'original_mono_sha256':proof['original_mono_sha256'],'original_mono_frames':grid['original_mono_frames'],'sample_rate':16000,
        'source_owners':deepcopy(grid['source_owners']),'execution_identity':deepcopy(plan['execution_identity']),
        'conditioning':{'context':'','language':None,'hotwords':[]},'internal_word_alignment':'unknown',
        'audio_attention':{'policy':deepcopy(plan['attention_policy']),'source_identity':deepcopy(plan['attention_source']),
            'encoder_config':deepcopy(plan['encoder_config']),'model_config':deepcopy(plan['model_config']),
            'control_receipts':controls,'observation_attestations':attestations},'observations':rows,'observations_sha256':ec._hash(rows)}


def memory_peaks(worker):
    names=('cuda_memory_before_load','cuda_memory_after_load','cuda_memory_final')
    fields={'scope','device_index','device_name','device_total_bytes','allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes'}
    require(worker.get('cuda_memory_error_type') is None,'CUDA memory measurement failed')
    values=[worker[name] for name in names]
    for value in values:
        require(type(value) is dict and set(value)==fields and value['scope']=='worker_cuda_reset_before_model_load'
            and type(value['device_name']) is str and bool(value['device_name'])
            and all(type(value[k]) is int and value[k]>=0 for k in fields-{'scope','device_name'})
            and value['device_total_bytes']>0 and 0<=value['allocated_bytes']<=value['reserved_bytes']<=value['device_total_bytes']
            and value['allocated_bytes']<=value['peak_allocated_bytes']<=value['peak_reserved_bytes']<=value['device_total_bytes']
            and value['reserved_bytes']<=value['peak_reserved_bytes'],'Invalid measured CUDA memory peak')
    require(all(all(value[k]==values[0][k] for k in ('device_index','device_name','device_total_bytes')) for value in values)
        and all([value[k] for value in values]==sorted(value[k] for value in values) for k in ('peak_allocated_bytes','peak_reserved_bytes')),
        'CUDA memory device/reset sequence changed')
    return {**{name:deepcopy(value) for name,value in zip(names,values)},'cuda_memory_error_type':None}


def verify_worker(folder,plan,assets):
    worker=sealed(read(folder/'worker/worker-result.json'));spec=read(folder/'worker-spec.json');proc=read(folder/'worker-process.json')
    life=read(folder/'lifecycle.json');original=read(folder/'original-backend.json')
    require(life['restored'] is True and life['owned_worker_stopped'] is True and life['error_type'] is None and same(life['original_backend'],original['model'])
        and type(proc['exit_code']) is int and proc['exit_code']==0 and type(proc['pid']) is int and proc['pid']>0,'Owned long lifecycle incomplete')
    require(spec['version']==native.VERSION and spec['plan_path']==str(folder/'native-plan.json') and spec['plan_file_sha256']==file_hash(folder/'native-plan.json')
        and spec['output_dir']==str(folder/'worker') and same(spec['assets'],assets) and spec['worker_sha256']==file_hash(Path(native.__file__))
        and type(spec['worker_seconds']) in (int,float) and 0<spec['worker_seconds']<=WORK_SECONDS,'Long worker spec changed')
    require(worker['version']==native.VERSION and worker['spec_sha256']==file_hash(folder/'worker-spec.json') and worker['plan_sha256']==plan['plan_sha256']
        and type(worker['model_loads']) is int and worker['model_loads']==1 and type(worker['model_load_attempts']) is int and worker['model_load_attempts']==1
        and type(worker['seconds']) in (int,float) and 0<=worker['seconds']<=spec['worker_seconds'] and worker['error_type'] is None
        and worker['status'] in ('complete','negative_control_output'),'Long worker binding/status changed')
    memory=memory_peaks(worker)
    before=worker['runtime_before'];require(same(before,worker['runtime_after']) and before['audio_backend']==plan['encoder_config']['backend']
        and same(before['audio_encoder_config'],plan['encoder_config']) and same(before['model_config'],plan['model_config'])
        and before['model_dir']==str(Path(assets['model']['model_dir']).resolve()) and before['asr_backend']=='transformers'
        and type(before['thinker_text_backend']) is str and bool(before['thinker_text_backend']) and before['max_new_tokens']==4096
        and before['dtype']=='torch.float16' and before['device'].startswith('cuda'),'Actual long runtime backend changed')
    native.check_plan(plan,assets);processor=native.load_processor(assets);outputs={};controls={};expected_files=set();retired=False
    previous=datetime.fromisoformat(original['captured_utc']);require(len(worker['slots'])==14,'Exactly14 slots required')
    for number,(slot,expected) in enumerate(zip(worker['slots'],plan['slots']),1):
        require(same({k:slot[k] for k in expected},expected) and slot['number']==number,'Long slot order changed')
        if retired:require(slot['status']=='undispatched' and slot['receipt'] is None,'Dispatch after negative control');continue
        path=receipt_path(folder,number,slot);receipt=read(path);reservation=read(path.with_suffix('.started.json'));request=native.build_request(plan,slot['slot_id'])
        require(slot['status'] in ('empty','complete') and slot['receipt']==str(path) and slot['status']==receipt['status']
            and set(reservation)=={'version','request_sha256','started_utc'} and reservation['version']==native.VERSION
            and reservation['request_sha256']==ec._hash(request) and previous<=datetime.fromisoformat(reservation['started_utc'])<=datetime.fromisoformat(receipt['started_utc']),'Long reservation/order changed')
        previous=datetime.fromisoformat(receipt['finished_utc']);expected_files.update({path,path.with_suffix('.started.json')})
        output=native.validate_native_receipt(receipt,request,plan,assets=assets,artifact_root=folder/'worker',processor=processor);outputs[slot['slot_id']]=output
        if slot['kind']=='control':controls[slot['slot_id']]=native.control_result(output);retired=not controls[slot['slot_id']]['passed']
    real=sum(k.startswith('window-') for k in outputs)
    require(set((folder/'worker/receipts').glob('*.json'))==expected_files and type(worker['asr_calls']) is int and worker['asr_calls']==len(outputs)
        and worker['real_audio_calls']==real and same(worker['controls'],controls) and worker['status']==('negative_control_output' if retired else 'complete'),'Long attempt coverage changed')
    envelope=None if retired else observation_envelope(folder,plan,outputs)
    return {**memory,'passed':not retired,'native_complete':not retired,'reason_code':'negative_control_output' if retired else 'complete_long_observations',
        'asr_calls':len(outputs),'real_audio_calls':real,'control_calls':len(controls),'model_loads':1,'controls':controls,
        'runtime_backend':before,'attention_policy':plan['attention_policy'],'attention_source_sha256':ec._hash(plan['attention_source']),
        'source_rows_sha256':plan['grid']['source_rows_sha256'],'grid_sha256':ec._hash(plan['grid']),
        'observation_envelope_sha256':None if envelope is None else ec._hash(envelope),
        'candidate_generated':False,'new_recap_calls':0,'score':None,'accuracy_verified':False,'independent_recognizer_vote':False},envelope


def restore(folder,previous,process):
    start=time.monotonic();error=None;restored=False;stopped=process is None
    mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM});timer=signal.getitimer(signal.ITIMER_REAL);signal.setitimer(signal.ITIMER_REAL,0)
    try:
        if process is not None:backend._stop_owned_process(process);stopped=process.poll() is not None
        require(stopped,'Owned long worker still alive')
        current=late_audio._backend_identity(backend._request_json(ADMIN+'/status',admin=True))
        if current!=previous:backend._request_json(ADMIN+('/unload' if previous is None else '/load'),{} if previous is None else {'model':previous},admin=True,timeout=420)
        restored=late_audio._backend_identity(backend._request_json(ADMIN+'/status',admin=True))==previous;require(restored,'Backend restoration failed')
    except BaseException as exc:error=type(exc).__name__;raise
    finally:
        elapsed=time.monotonic()-start;value=state(folder);value['local_seconds']+=elapsed;value['original_backend_restored']=restored
        value['stages'].append({'stage':'cleanup','seconds':elapsed,'error_type':error});write_json(folder/'diagnostic.json',value)
        write_json(folder/'lifecycle.json',{'original_backend':previous,'restored':restored,'owned_worker_stopped':stopped,'seconds':elapsed,'error_type':error})
        while signal.SIGALRM in signal.sigpending():signal.sigtimedwait({signal.SIGALRM},0)
        if timer[0]>elapsed:signal.setitimer(signal.ITIMER_REAL,timer[0]-elapsed,timer[1])
        signal.pthread_sigmask(signal.SIG_SETMASK,mask)
        if error is None and (elapsed>CLEANUP_SECONDS or value['local_seconds']>TOTAL_SECONDS):raise LocalBudgetExceeded('Long cleanup/total overrun')


def execute(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(state(folder)['status']=='prepared','Long acquisition cannot resume or reroll');captured=False;previous=None;process=None;failure=None
        try:
            with phase(folder,'input_replay'):
                reg=validate_registration(folder);require(state(folder)['registration_sha256']==reg['registration_sha256'],'State binding changed')
            value=state(folder);value.update(status='running',started_utc=now());write_json(folder/'diagnostic.json',value)
            with phase(folder,'long_source_worker') as deadline:
                previous=late_audio._backend_identity(backend._request_json(ADMIN+'/status',admin=True));captured=True
                immutable(folder/'original-backend.json',{'model':previous,'captured_utc':now()});value=state(folder);value['original_backend_restored']=False;write_json(folder/'diagnostic.json',value)
                backend._request_json(ADMIN+'/unload',{},admin=True,timeout=60)
                require(late_audio._backend_identity(backend._request_json(ADMIN+'/status',admin=True)) is None,'Backend not unloaded')
                assets=read(folder/'assets.json');seconds=deadline-time.monotonic();require(seconds>0,'No long worker time remains')
                spec={'version':native.VERSION,'plan_path':str(folder/'native-plan.json'),'plan_file_sha256':file_hash(folder/'native-plan.json'),
                    'output_dir':str(folder/'worker'),'assets':assets,'worker_seconds':seconds,
                    'work_deadline_utc':(datetime.now(timezone.utc)+timedelta(seconds=seconds)).isoformat(),'worker_sha256':file_hash(Path(native.__file__))}
                immutable(folder/'worker-spec.json',spec)
                with (folder/'worker-process.log').open('xb') as log:
                    process=subprocess.Popen([assets['interpreter'],str(Path(native.__file__).resolve()),'--spec',str(folder/'worker-spec.json')],cwd=ROOT,
                        env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'},stdout=log,stderr=subprocess.STDOUT)
                    code=process.wait(timeout=max(.1,deadline-time.monotonic()))
                immutable(folder/'worker-process.json',{'pid':process.pid,'exit_code':code,'finished_utc':now()});require(code==0,'Long worker failed')
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
        and value['work_seconds']<=value['local_seconds']<=TOTAL_SECONDS,'Long acquisition not complete/restored/in budget')
    expected,envelope=verify_worker(folder,read(folder/'native-plan.json'),read(folder/'assets.json'))
    require(all(same(result.get(k),v) for k,v in expected.items()) and same(envelope,read(folder/'observations.json'))
        and result['version']==VERSION and result['local_seconds']==value['local_seconds'] and result['work_seconds']==value['work_seconds']
        and result['original_backend_restored'] is True,'Long terminal observations changed')
    artifacts={str(p.resolve()):file_hash(p) for p in folder.rglob('*') if p.is_file() and p.name not in ('.lock','result.json')}
    require(same(artifacts,result['artifacts']),'Long artifact coverage changed')
    return {'envelope':envelope,'result':{k:v for k,v in result.items() if k!='artifacts'},'source_bindings':read(folder/'input-proof.json')['source_bindings'],
        'pins':{**reg['pins'],**artifacts,str(folder/'result.json'):file_hash(folder/'result.json')}}


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--execute',action='store_true');args=parser.parse_args();logging.basicConfig(level=logging.INFO)
    try:
        value=prepare() if args.prepare else execute();logging.info('Long acquisition status=%s passed=%s',value.get('status'),value.get('passed'));return 0
    except Exception as exc:logging.error('Long acquisition stopped: %s',type(exc).__name__);return 1
if __name__=='__main__':raise SystemExit(main())
