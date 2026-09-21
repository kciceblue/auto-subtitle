"""One source-only recap and bounded paired local-ASR diagnostic; no translation/scoring."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src import contextual_asr_inputs as inputs
from src import contextual_asr_requests as requests
from src import contextual_asr_recap_native as recap
from src import contextual_asr_native as native
from src import late_audio as late
from src import local_backend as lb
from src import revisit_workflow as rw
from src import evidence_context as ec
from src.workflow_state import file_hash, fingerprint, write_json
from src.contextual_review_contract import require, strict_json
from scripts.full_draft_diagnostic import locked, immutable

VERSION = 'contextual-asr-diagnostic-2'
PRIOR_LOCAL_SECONDS = 18.184413342009066
FOLDER = ROOT/'output/quality-contextual-asr-gain-20260915'
PLAN = ROOT/'design/quality-recap-conditioned-asr-implementation-20260915.md'
LQ = ROOT/'output/quality-large-compact-20260915'
MODEL = Path('/home/kciceblue/HF/asr-models/Qwen3-ASR-1.7B')
KEY = 'source-recap'
LOG = logging.getLogger(__name__)


def read(path): return strict_json(Path(path).read_bytes())
def now(): return datetime.now(timezone.utc).isoformat()
def same(a,b): return fingerprint(a) == fingerprint(b)
def state(folder): return read(folder/'diagnostic.json')


def _failed(folder, exc):
    value=state(folder);value.update(status='failed',error_type=type(exc).__name__,finished_utc=now(),passed=False)
    write_json(folder/'diagnostic.json',value)


def remaining(folder):
    s=state(folder)
    for key in ('local_seconds','work_seconds'):
        require(type(s[key]) in (int,float) and math.isfinite(s[key]) and s[key]>=0,'Invalid time ledger')
    return min(1800-s['local_seconds'],1380-s['work_seconds'])


@contextmanager
def phase(folder,name,ceiling=None):
    allowance=remaining(folder)
    if ceiling is not None: allowance=min(allowance,ceiling)
    require(allowance>0,'Diagnostic work budget exhausted')
    start=time.monotonic();error=None
    handler=signal.getsignal(signal.SIGALRM);timer=signal.getitimer(signal.ITIMER_REAL)
    def expired(signum,frame): raise rw.LocalBudgetExceeded('C-ASR1 work deadline')
    if timer[0]: allowance=min(allowance,timer[0])
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,allowance)
    try: yield start+allowance
    except BaseException as exc: error=type(exc).__name__;raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
        elapsed=time.monotonic()-start
        if timer[0]:signal.setitimer(signal.ITIMER_REAL,max(.001,timer[0]-elapsed),timer[1])
        s=state(folder);s['local_seconds']+=elapsed;s['work_seconds']+=elapsed
        over=elapsed>=allowance or s['local_seconds']>=1800 or s['work_seconds']>=1380
        s['stages'].append({'stage':name,'seconds':elapsed,'error_type':error or ('LocalBudgetExceeded' if over else None)})
        write_json(folder/'diagnostic.json',s);LOG.info('%s %.2fs error=%s',name,elapsed,error)
        if over and error is None: raise rw.LocalBudgetExceeded('C-ASR1 phase exceeded allowance')


def prerequisite():
    s=read(LQ/'screen.json');require(s.get('original_backend_restored') is True,'LQ1 backend not restored')
    pins={str(LQ/'screen.json'):file_hash(LQ/'screen.json')}
    if s.get('status')=='complete':
        score=read(LQ/'score.json');require(type(score.get('score')) is int and score['score']<3,
            'Validate a promising larger draft before this diagnostic')
        pins[str(LQ/'score.json')]=file_hash(LQ/'score.json')
        outcome={'status':'scored_below_three','score':score['score']}
    else:
        require(s.get('status')=='failed' and not (LQ/'score.json').exists(),'LQ1 is still active or has an inconsistent score')
        outcome={'status':'failed_unscored','error_type':s.get('error_type')}
    life=LQ/'writer-backend/lifecycle.json'
    if life.exists():
        value=read(life);require(value.get('original_backend_restored') is True,'LQ1 native restoration unconfirmed')
        require(value.get('owned_process_stopped') in (True,None),'LQ1 process shutdown unconfirmed')
        pins[str(life)]=file_hash(life)
    return {'outcome':outcome,'pins':pins}


def build_assets():
    model=late._model_pins(MODEL,late.ENGINES[1]);require(model.get('available') is True,'Local QwenASR missing')
    distribution=importlib.metadata.distribution('qwen-asr')
    names={'tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','vocab.json','merges.txt','chat_template.json'}
    return {'model':model,'runtime':late._runtime_identity(),
        'parser':late._pin(distribution.locate_file('qwen_asr/inference/utils.py')),
        'api':late._pin(distribution.locate_file('qwen_asr/inference/qwen3_asr.py')),
        'tokenizer_files':[p for p in model['files'] if Path(p['path']).name in names],
        'interpreter':late._interpreter_path(ROOT/'.venv/bin/python3')}


def crop_arrays(preparation,mono_path,*,return_preparation=False):
    import numpy as np
    import soundfile as sf
    mono_path=Path(mono_path).resolve()
    original_sha=file_hash(mono_path)
    require(original_sha==preparation['inputs']['detector_provenance']['mono_sha256'],'Pinned original mono changed')
    info=sf.info(mono_path); require(info.channels==1 and info.samplerate==16000 and info.subtype=='FLOAT', 'Original mono changed')
    audio,rate=sf.read(mono_path,dtype='float32',always_2d=False)
    require(audio.ndim==1 and len(audio)==preparation['inputs']['detector_record']['mono']['frames']
        and len(audio)>0 and rate==16000 and np.isfinite(audio).all(),'Original mono domain or finite samples changed')
    require(file_hash(mono_path)==original_sha,'Original mono changed during shared-gain preparation')
    global_peak=float(np.max(np.abs(audio)))
    gain=min(1.0,0.99/global_peak) if global_peak>0 else 1.0
    gain_float32=np.float32(gain)
    metadata={'version':'contextual-asr-audio-preparation-1','policy':'one_global_gain_for_all_real_crops',
        'original_mono_path':str(mono_path),'original_mono_sha256':original_sha,
        'original_frames':len(audio),'sample_rate':16000,'global_peak':global_peak,'target_peak':0.99,
        'gain':gain,'gain_float32':float(gain_float32),
        'float32_formula':'np.multiply(original_float32_crop, np.float32(gain), dtype=np.float32)',
        'clipping_applied':False,'crop_selection_changed':False,'synthetic_controls_scaled':False}
    result={}
    for pair in preparation['pairs']:
        if pair['kind']=='synthetic_silence': value=np.zeros(192000,dtype=np.float32)
        elif pair['kind']=='synthetic_noise':value=np.random.Generator(np.random.PCG64(20260915)).uniform(-.01,.01,192000).astype(np.float32)
        else:value=np.multiply(audio[pair['crop']['start_frame']:pair['crop']['end_frame']],gain_float32,dtype=np.float32)
        require(value.dtype==np.float32 and value.shape==(192000,) and np.isfinite(value).all() and float(np.abs(value).max())<=1,
            'Prepared shared-gain ASR input would require hidden normalization')
        result[pair['pair_id']]=value
    return (result,metadata) if return_preparation else result


def prepare_audio(folder,preparation,mono_path):
    import soundfile as sf
    values,metadata=crop_arrays(preparation,mono_path,return_preparation=True)
    immutable(folder/'audio-preparation.json',metadata)
    bindings={};directory=folder/'audio';directory.mkdir()
    for pair,value in values.items():
        path=directory/(pair+'.wav');sf.write(path,value,16000,subtype='FLOAT',format='WAV')
        bindings[pair]={'path':str(path.resolve()),'sha256':file_hash(path),'pcm_sha256':hashlib.sha256(value.tobytes()).hexdigest(),
            'frames':192000,'sample_rate':16000,'dtype':'float32'}
        require(native._read_audio(bindings[pair]).tobytes()==value.tobytes(),'Prepared PCM roundtrip differs')
    return bindings


def validate_audio(folder,prep,capsule,bindings):
    expected,metadata=crop_arrays(prep,capsule['mono_path'],return_preparation=True)
    require(ec._hash(read(folder/'audio-preparation.json'))==ec._hash(metadata),'Audio preparation differs from pinned whole-mono gain')
    require(set(bindings)==set(requests.PAIR_IDS),'Incomplete paired audio')
    for pair in requests.PAIR_IDS:
        require(Path(bindings[pair]['path'])==folder/'audio'/(pair+'.wav'),'Audio path escaped diagnostic')
        require(native._read_audio(bindings[pair]).tobytes()==expected[pair].tobytes(),'ASR input differs from fixed original crop/control')


def producers():
    return [PLAN,Path(__file__),Path(inputs.__file__),Path(requests.__file__),Path(native.__file__),Path(recap.__file__),
        Path(late.__file__),Path(lb.__file__),Path(rw.__file__),Path(ec.__file__),ROOT/'src/translate.py',ROOT/'src/config.py',
        ROOT/'src/workflow_state.py',ROOT/'src/contextual_review_contract.py',ROOT/'src/voice_activity_evidence.py',
        ROOT/'src/temporal_evidence.py',ROOT/'src/short_audio.py',ROOT/'scripts/full_draft_diagnostic.py']


def validate_registration(folder):
    reg=read(folder/'registration.json');require(reg['version']==VERSION and reg['registration_sha256']==fingerprint({k:v for k,v in reg.items() if k!='registration_sha256'}),'Registration changed')
    required={str(p.resolve()) for p in producers()}|{str(folder/name) for name in ('input-capsule.json','preparation.json','assets.json','execution-identity.json','audio-bindings.json','audio-preparation.json')}
    require(required<=set(reg['pins']),'Required diagnostic producer/input pin missing')
    for path,digest in reg['pins'].items():require(file_hash(Path(path))==digest,'Diagnostic producer/input changed')
    require(same(reg['prerequisite'],prerequisite()),'Prerequisite changed')
    capsule=inputs.load_inputs();require(same(capsule,read(folder/'input-capsule.json')),'Original input lineage changed')
    require(all(reg['pins'].get(path)==digest for path,digest in {**capsule['input_hashes'],**reg['prerequisite']['pins']}.items()),'Input lineage pin coverage incomplete')
    prep=read(folder/'preparation.json')
    requests.validate_preparation(prep,capsule['source_rows'],capsule['original_context'],
        source_provenance=capsule['source_provenance'],detector_record=capsule['detector_record'],detector_provenance=capsule['detector_provenance'])
    assets=read(folder/'assets.json');identity=native.execution_identity(assets)
    require(same(identity,read(folder/'execution-identity.json')),'Native identity changed')
    native.validate_assets(assets,identity,runtime_check=True)
    validate_audio(folder,prep,capsule,read(folder/'audio-bindings.json'))
    return reg


def prepare(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(set(folder.iterdir())=={folder/'.lock'},'Diagnostic already prepared/attempted; no restart')
        write_json(folder/'diagnostic.json',{'version':VERSION,'status':'preparing','local_seconds':PRIOR_LOCAL_SECONDS,'work_seconds':PRIOR_LOCAL_SECONDS,
            'stages':[{'stage':'prior_preparation_failure_and_numeric_diagnosis','seconds':PRIOR_LOCAL_SECONDS,'error_type':'ValueError'}],'original_backend_restored':True,'passed':False,'created_utc':now()})
        try:
            with phase(folder,'preparation',180-PRIOR_LOCAL_SECONDS):
                prior=prerequisite();capsule=inputs.load_inputs()
                prep=requests.prepare_diagnostic(capsule['source_rows'],capsule['original_context'],
                    source_provenance=capsule['source_provenance'],detector_record=capsule['detector_record'],detector_provenance=capsule['detector_provenance'])
                assets=build_assets();identity=native.execution_identity(assets);native.validate_assets(assets,identity,runtime_check=True)
                audio=prepare_audio(folder,prep,capsule['mono_path']);recap.prepare_request(prep)
                for name,value in [('input-capsule.json',capsule),('preparation.json',prep),('assets.json',assets),('execution-identity.json',identity),('audio-bindings.json',audio)]:immutable(folder/name,value)
                paths=producers()
                paths.extend(p for p in folder.rglob('*') if p.is_file() and p.name not in ('.lock','diagnostic.json'))
                pins={str(p.resolve()):file_hash(p) for p in paths};pins.update(capsule['input_hashes']);pins.update(prior['pins'])
                reg={'version':VERSION,'prerequisite':prior,'pins':pins};reg['registration_sha256']=fingerprint(reg);immutable(folder/'registration.json',reg)
            s=state(folder);s.update(status='prepared',registration_sha256=reg['registration_sha256']);write_json(folder/'diagnostic.json',s)
            return s
        except BaseException as exc:_failed(folder,exc);raise


def restore(folder,previous,process=None):
    start=time.monotonic();failure=None;restored=False;stopped=process is None
    mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM});timer=signal.getitimer(signal.ITIMER_REAL);signal.setitimer(signal.ITIMER_REAL,0)
    try:
        if process is not None:lb._stop_owned_process(process);stopped=process.poll() is not None
        require(stopped,'Owned ASR process still running')
        current=late._backend_identity(rw._request_json(rw.ADMIN+'/status',admin=True))
        if current!=previous:
            rw._request_json(rw.ADMIN+('/unload' if previous is None else '/load'),{} if previous is None else {'model':previous},admin=True,timeout=360)
        restored=late._backend_identity(rw._request_json(rw.ADMIN+'/status',admin=True))==previous
        require(restored,'Original backend restoration failed')
    except BaseException as exc:failure=exc;raise
    finally:
        elapsed=time.monotonic()-start;s=state(folder);s['local_seconds']+=elapsed;s['original_backend_restored']=restored
        s['stages'].append({'stage':'cleanup','seconds':elapsed,'error_type':type(failure).__name__ if failure else None})
        write_json(folder/'diagnostic.json',s);write_json(folder/'lifecycle.json',{'original_backend':previous,'restored':restored,'owned_worker_stopped':stopped,'seconds':elapsed,'error_type':type(failure).__name__ if failure else None})
        while signal.SIGALRM in signal.sigpending():signal.sigtimedwait({signal.SIGALRM},0)
        if timer[0] and timer[0]>elapsed:signal.setitimer(signal.ITIMER_REAL,timer[0]-elapsed,timer[1])
        signal.pthread_sigmask(signal.SIG_SETMASK,mask)
        if failure is None and (elapsed>420 or s['local_seconds']>=1800):raise rw.LocalBudgetExceeded('Cleanup exceeded diagnostic allowance')


def verify_worker(folder,plan,assets):
    result=native._sealed(read(folder/'worker/worker-result.json'));spec=read(folder/'worker-spec.json')
    proc=read(folder/'worker-process.json');life=read(folder/'lifecycle.json');original=read(folder/'original-backend.json')
    require(type(proc.get('pid')) is int and proc['pid']>0 and type(proc.get('exit_code')) is int and proc['exit_code']==0,'Worker process success missing')
    require(life.get('restored') is True and life.get('owned_worker_stopped') is True and life.get('error_type') is None and same(life.get('original_backend'),original['model']),'Owned lifecycle restoration not bound')
    require(spec['version']==native.VERSION and spec['plan_path']==str(folder/'native-plan.json') and spec['plan_file_sha256']==file_hash(folder/'native-plan.json')
        and spec['output_dir']==str(folder/'worker') and same(spec['assets'],assets) and spec['worker_sha256']==file_hash(Path(native.__file__))
        and type(spec['worker_seconds']) in (int,float) and 0<spec['worker_seconds']<=900,'Worker dispatch specification changed')
    require(result.get('version')==native.VERSION and result.get('spec_sha256')==file_hash(folder/'worker-spec.json') and result.get('plan_sha256')==plan['plan_sha256'],'Worker binding changed')
    require(type(result.get('model_load_attempts')) is int and result['model_load_attempts']==1 and type(result.get('model_loads')) is int and result['model_loads']==1 and type(result.get('seconds')) in (int,float) and 0<=result['seconds']<=spec['worker_seconds'],'Worker load/time incomplete')
    slots=result['slots'];require(len(slots)==20 and type(result.get('asr_calls')) is int,'Worker coverage incomplete')
    receipts={};outputs={};processor=native.load_processor(assets);expected_files=set()
    prior_end=datetime.fromisoformat(read(folder/'recap'/(KEY+'.json'))['attempts'][-1]['finished_utc'])
    require(prior_end.utcoffset() is not None,'Recap timestamp lacks timezone')
    for number,(slot,expected) in enumerate(zip(slots,plan['slots']),1):
        require({k:slot.get(k) for k in expected}==expected and slot.get('number')==number,'Worker slot order changed')
        if slot.get('status')=='undispatched':require(slot.get('receipt') is None,'Undispatched receipt exists');continue
        path=folder/'worker/receipts'/f'{number:02d}-{slot["pair_id"]}-{slot["arm"]}.json'
        require(slot['receipt']==str(path) and slot['status'] in ('complete','empty'),'Worker technical failure')
        receipt=read(path);require(receipt['status']==slot['status'],'Slot/receipt status differs')
        started=path.with_suffix('.started.json');reservation=read(started);expected_files.update({path,started})
        stamp=datetime.fromisoformat(receipt['started_utc']);end=datetime.fromisoformat(receipt['finished_utc'])
        require(stamp.utcoffset() is not None and end.utcoffset() is not None and prior_end<=stamp<=end,'Native decode ordering changed');prior_end=end
        request=requests.build_native_request(plan,slot['pair_id'],slot['arm'])
        require(set(reservation)=={'version','request_sha256','started_utc'} and reservation['version']==native.VERSION and reservation['request_sha256']==ec._hash(request)
            and datetime.fromisoformat(reservation['started_utc'])<=stamp,'Native dispatch reservation changed')
        outputs[slot['slot_id']]=native.validate_native_receipt(receipt,request,plan,assets=assets,artifact_root=folder/'worker',processor=processor)
        receipts[slot['slot_id']]=receipt
    require(set((folder/'worker/receipts').glob('*.json'))==expected_files,'Unexpected native dispatch artifacts')
    count=len(receipts);require(result['asr_calls']==count and [x['status']!='undispatched' for x in slots]==[True]*count+[False]*(20-count),'Hidden or reordered native attempt')
    controls={key:requests.control_metrics(plan,value['raw_text'],value['parsed_text']) for key,value in outputs.items() if key.split(':')[0] in ('silence','noise')}
    require(same(controls,result['controls']),'Native negative-control metrics changed')
    if result['status']=='negative_control_output':
        require(count==4 and any(v['raw_nonempty'] or v['parsed_nonempty'] for v in controls.values()),'Negative-control retirement not supported')
        return {'passed':False,'reason_code':'negative_control_output','asr_calls':4,'changed_real_pairs':None,'controls':controls}
    require(result['status'] in ('complete','conditioning_inactive') and count==20,'Incomplete worker')
    changed=[pair for pair in requests.PAIR_IDS[2:] if outputs[pair+':unhinted']['parsed_text']!=outputs[pair+':hinted']['parsed_text']]
    require(same(result['changed_real_pairs'],changed) and result['status']==('complete' if len(changed)>=2 else 'conditioning_inactive'),'Native effect summary changed')
    s=state(folder);rr=read(folder/'recap'/(KEY+'.json'));life={'local_seconds':s['local_seconds'],'work_seconds':s['work_seconds'],'restored':s['original_backend_restored'],'asr_calls':20,'recap_attempts':len(rr['attempts'])}
    answer=requests.validate_diagnostic(plan,receipts,validate_native_receipt=lambda receipt,request,plan:native.validate_native_receipt(receipt,request,plan,assets=assets,artifact_root=folder/'worker',processor=processor),lifecycle=life)
    return {**answer,'asr_calls':20}


def execute(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(state(folder)['status']=='prepared','Diagnostic cannot resume or reroll')
        previous=None;captured=False;process=None;failure=None
        try:
            with phase(folder,'input_replay'):
                reg=validate_registration(folder)
                require(state(folder)['version']==VERSION and state(folder)['registration_sha256']==reg['registration_sha256'],'Diagnostic state identity changed')
            s=state(folder);s.update(status='running',started_utc=now());write_json(folder/'diagnostic.json',s)
            with phase(folder,'source_recap') as deadline:
                previous=late._backend_identity(rw._request_json(rw.ADMIN+'/status',admin=True));captured=True
                s=state(folder);s.update(original_backend_restored=False,original_backend=previous);write_json(folder/'diagnostic.json',s)
                immutable(folder/'original-backend.json',{'model':previous,'captured_utc':now()})
                loaded=rw._request_json(rw.ADMIN+'/load',{'model':rw.QWEN},admin=True,timeout=min(360,remaining(folder)))
                require(loaded.get('loaded')==rw.QWEN,'Recap backend not loaded')
                immutable(folder/'recap-backend.json',rw._request_json(rw.ADMIN+'/status',admin=True))
                prep=read(folder/'preparation.json');response=recap.ask(folder/'recap',KEY,'http://127.0.0.1:8089/v1/chat/completions',prep,deadline=deadline)
                require(same(response,recap.replay_native(folder/'recap'/(KEY+'.json'),prep,lambda p,raw=False:Path(p).read_bytes() if raw else read(p))),'Native recap replay differs')
                plan=requests.bind_recap(prep,response,read(folder/'audio-bindings.json'),read(folder/'execution-identity.json'),recap_native_sha256=file_hash(folder/'recap'/(KEY+'.json')))
                immutable(folder/'native-plan.json',plan)
            with phase(folder,'paired_asr',900) as deadline:
                rw._request_json(rw.ADMIN+'/unload',{},admin=True,timeout=60)
                require(late._backend_identity(rw._request_json(rw.ADMIN+'/status',admin=True)) is None,'Warden not unloaded')
                assets=read(folder/'assets.json');seconds=max(0,deadline-time.monotonic());require(seconds>0,'No worker time remains')
                spec={'version':native.VERSION,'plan_path':str(folder/'native-plan.json'),'plan_file_sha256':file_hash(folder/'native-plan.json'),'output_dir':str(folder/'worker'),'assets':assets,'worker_seconds':seconds,'work_deadline_utc':(datetime.now(timezone.utc)+timedelta(seconds=seconds)).isoformat(),'worker_sha256':file_hash(Path(native.__file__))}
                immutable(folder/'worker-spec.json',spec)
                env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'}
                with (folder/'worker-process.log').open('xb') as log:
                    process=subprocess.Popen([assets['interpreter'],str(Path(native.__file__).resolve()),'--spec',str(folder/'worker-spec.json')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    code=process.wait(timeout=max(.1,deadline-time.monotonic()))
                immutable(folder/'worker-process.json',{'pid':process.pid,'exit_code':code,'finished_utc':now()});require(code==0,'ASR worker failed')
        except BaseException as exc:failure=exc
        finally:
            if captured:
                try:restore(folder,previous,process)
                except BaseException as exc:
                    if failure is None:failure=exc
                    else:failure.add_note('Cleanup failed: '+type(exc).__name__)
        if failure is not None:_failed(folder,failure);raise failure
        try:
            with phase(folder,'final_native_replay'):
                validate_registration(folder);prep=read(folder/'preparation.json')
                response=recap.replay_native(folder/'recap'/(KEY+'.json'),prep,lambda p,raw=False:Path(p).read_bytes() if raw else read(p))
                plan=read(folder/'native-plan.json');requests.validate_plan(plan,prep,response,read(folder/'audio-bindings.json'),read(folder/'execution-identity.json'),recap_native_sha256=file_hash(folder/'recap'/(KEY+'.json')))
                result=verify_worker(folder,plan,read(folder/'assets.json'))
                artifacts={str(p.resolve()):file_hash(p) for p in folder.rglob('*') if p.is_file() and p.name not in ('.lock','result.json','diagnostic.json')}
            s=state(folder);s.update(status='complete' if result['passed'] else 'retired',passed=result['passed'],reason_code=result['reason_code'],finished_utc=now());write_json(folder/'diagnostic.json',s)
            artifacts[str((folder/'diagnostic.json').resolve())]=file_hash(folder/'diagnostic.json')
            immutable(folder/'result.json',{**result,'artifacts':artifacts,'version':VERSION,'local_seconds':s['local_seconds'],'work_seconds':s['work_seconds'],'original_backend_restored':s['original_backend_restored'],'candidate_generated':False,'score':None,'accuracy_verified':False})
            return read(folder/'result.json')
        except BaseException as exc:_failed(folder,exc);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);m=parser.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--execute',action='store_true');args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    try:
        result=prepare() if args.prepare else execute();LOG.info('C-ASR1 status=%s passed=%s score=None',result.get('status'),result.get('passed'));return 0
    except Exception as exc:LOG.error('C-ASR1 stopped: %s; artifacts retained',type(exc).__name__);return 1

if __name__=='__main__':raise SystemExit(main())
