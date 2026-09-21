"""CAUTO1: four automatic-language controls; no real-audio or Chinese dispatch."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone,timedelta
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import contextual_asr_diagnostic as base
from scripts import contextual_asr_projected as parent_controller
from src import contextual_asr_auto_native as native
from src import contextual_asr_recap_native as deadline_helper
from src.workflow_state import file_hash,fingerprint,write_json
from src.contextual_review_contract import require
VERSION='contextual-asr-auto-diagnostic-1'
FOLDER=ROOT/'output/quality-contextual-asr-auto-20260915'
PARENT=ROOT/'output/quality-contextual-asr-projected-20260915'
STAGED=ROOT/'output/quality-staged-source-20260915'
PLAN=ROOT/'design/quality-contextual-asr-auto-20260915.md'
read,immutable,same,locked,state,now=base.read,base.immutable,base.same,base.locked,base.state,base.now
LOG=logging.getLogger(__name__)


def remaining(folder):
    s=state(folder)
    require(all(type(s[k]) in (int,float) and math.isfinite(s[k]) and s[k]>=0 for k in ('local_seconds','work_seconds')),'Invalid time ledger')
    return min(600-s['local_seconds'],240-s['work_seconds'])


@contextmanager
def phase(folder,name,ceiling=None):
    start=time.monotonic();error=None;allowance=remaining(folder)
    if ceiling is not None:allowance=min(allowance,ceiling)
    try:
        with deadline_helper._within(start+allowance):yield start+allowance
    except BaseException as exc:error=type(exc).__name__;raise
    finally:
        elapsed=time.monotonic()-start;s=state(folder);s['local_seconds']+=elapsed;s['work_seconds']+=elapsed
        s['stages'].append({'stage':name,'seconds':elapsed,'error_type':error});write_json(folder/'diagnostic.json',s)


def scheduling():
    s=read(STAGED/'screen.json')
    require(s.get('original_backend_restored') is True and s.get('status') in ('local_complete','review_reserved','complete','failed'),
            'SS1 must release and restore its local backend before CAUTO1')
    return {'screen_sha256':file_hash(STAGED/'screen.json'),'status':s['status'],'restored':True}


def parent_inputs():
    reg=parent_controller.validate_registration(PARENT)
    result=read(PARENT/'result.json');s=state(PARENT)
    require(s['status']=='retired' and s['original_backend_restored'] is True
        and result['reason_code']=='negative_control_output' and result['asr_calls']==4
        and result['candidate_generated'] is False and result['score'] is None,'Required forced-language control failure missing')
    original=read(PARENT/'native-plan.json');assets=read(PARENT/'assets.json')
    replay=base.verify_worker(PARENT,original,assets)
    require(replay['reason_code']=='negative_control_output' and replay['asr_calls']==4,'Forced-language native control replay failed')
    proof={'parent_result_sha256':file_hash(PARENT/'result.json'),'parent_plan_sha256':file_hash(PARENT/'native-plan.json'),
           'projected_recap_sha256':file_hash(PARENT/'projection.json'),'parent_local_seconds':s['local_seconds'],
           'forced_controls':replay['controls']}
    pins={**reg['pins'],**result['artifacts'],str(PARENT/'registration.json'):file_hash(PARENT/'registration.json'),
          str(PARENT/'result.json'):file_hash(PARENT/'result.json')}
    return original,assets,proof,pins


def producers():
    return [PLAN,Path(__file__).resolve(),Path(native.__file__),Path(base.__file__),Path(parent_controller.__file__),
            Path(native.base.__file__),Path(native.ec.__file__),Path(deadline_helper.__file__),
            ROOT/'src/contextual_asr_requests.py',ROOT/'src/workflow_state.py',ROOT/'src/local_backend.py',
            ROOT/'src/late_audio.py',ROOT/'src/revisit_workflow.py']


def validate_registration(folder):
    reg=read(folder/'registration.json')
    require(reg['version']==VERSION and reg['registration_sha256']==fingerprint({k:v for k,v in reg.items() if k!='registration_sha256'}),'Registration changed')
    fixed={str(p.resolve()) for p in producers()}|{str(folder/name) for name in ('native-plan.json','assets.json','parent-proof.json')}
    require(fixed<=set(reg['pins']),'Required CAUTO input/producer missing')
    for path,digest in reg['pins'].items():require(file_hash(Path(path))==digest,'Pinned CAUTO input/producer changed')
    original,assets,proof,pins=parent_inputs()
    require(all(reg['pins'].get(k)==v for k,v in pins.items()),'Parent provenance pin missing')
    require(same(proof,read(folder/'parent-proof.json')) and same(assets,read(folder/'assets.json')),'Parent/asset binding changed')
    require(same(native.build_plan(original,assets),read(folder/'native-plan.json')),'Automatic-language request plan changed')
    return reg


def prepare(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(set(folder.iterdir())=={folder/'.lock'},'CAUTO1 already attempted')
        write_json(folder/'diagnostic.json',{'version':VERSION,'status':'preparing','local_seconds':0.,'work_seconds':0.,
            'stages':[],'passed':False,'original_backend_restored':True,'created_utc':now()})
        try:
            with phase(folder,'preparation',90):
                schedule=scheduling();original,assets,proof,pins=parent_inputs();plan=native.build_plan(original,assets)
                for name,value in [('native-plan.json',plan),('assets.json',assets),('parent-proof.json',proof)]:immutable(folder/name,value)
                paths=[*producers(),*(folder/name for name in ('native-plan.json','assets.json','parent-proof.json'))]
                pins.update({str(p.resolve()):file_hash(p) for p in paths})
                reg={'version':VERSION,'policy':native.POLICY,'scheduling_snapshot':schedule,'pins':pins}
                reg['registration_sha256']=fingerprint(reg);immutable(folder/'registration.json',reg)
            s=state(folder);s.update(status='prepared',registration_sha256=reg['registration_sha256']);write_json(folder/'diagnostic.json',s);return s
        except BaseException as exc:base._failed(folder,exc);raise


def verify_worker(folder,plan,assets):
    worker=native._sealed(read(folder/'worker/worker-result.json'));spec=read(folder/'worker-spec.json')
    proc=read(folder/'worker-process.json');life=read(folder/'lifecycle.json');original=read(folder/'original-backend.json')
    require(life.get('restored') is True and life.get('owned_worker_stopped') is True and life.get('error_type') is None
        and same(life.get('original_backend'),original['model']),'Restored owned lifecycle missing')
    require(type(proc['pid']) is int and proc['pid']>0 and type(proc['exit_code']) is int and proc['exit_code']==0,'Owned worker did not complete')
    require(spec['version']==native.VERSION and spec['plan_path']==str(folder/'native-plan.json')
        and spec['plan_file_sha256']==file_hash(folder/'native-plan.json') and spec['output_dir']==str(folder/'worker')
        and same(spec['assets'],assets) and spec['worker_sha256']==file_hash(Path(native.__file__))
        and type(spec['worker_seconds']) in (int,float) and 0<spec['worker_seconds']<=240,'Worker spec differs')
    require(worker['version']==native.VERSION and worker['spec_sha256']==file_hash(folder/'worker-spec.json')
        and worker['plan_sha256']==plan['plan_sha256'] and type(worker['seconds']) in (int,float)
        and 0<=worker['seconds']<=spec['worker_seconds'] and worker['error_type'] is None
        and type(worker['asr_calls']) is int and worker['asr_calls']==4
        and type(worker['model_loads']) is int and worker['model_loads']==1
        and type(worker['model_load_attempts']) is int and worker['model_load_attempts']==1,'Worker count/time/binding differs')
    native.check_plan(plan,assets);processor=base.native.load_processor(assets);outputs={};expected_files=set()
    require(len(worker['slots'])==4,'Only four automatic-language controls are authorized')
    previous=native._time(original['captured_utc'])
    for number,(slot,expected) in enumerate(zip(worker['slots'],plan['slots']),1):
        path=folder/'worker/receipts'/f'{number:02d}-{expected["pair_id"]}-{expected["arm"]}.json'
        require(same(slot,{**expected,'number':number,'status':slot['status'],'receipt':str(path)})
            and slot['status'] in ('empty','complete'),'Unexpected native control slot')
        receipt=read(path);reservation_path=path.with_suffix('.started.json');reservation=read(reservation_path)
        expected_files.update({path,reservation_path});request=native.build_request(plan,slot['pair_id'],slot['arm'])
        require(set(reservation)=={'version','request_sha256','started_utc'} and reservation['version']==native.VERSION
            and reservation['request_sha256']==native.ec._hash(request)
            and previous<=native._time(reservation['started_utc'])<=native._time(receipt['started_utc'])
            and receipt['status']==slot['status'],'Control reservation/order changed')
        previous=native._time(receipt['finished_utc'])
        outputs[slot['slot_id']]=native.validate_native_receipt(receipt,request,plan,assets=assets,artifact_root=folder/'worker',processor=processor)
    require(set((folder/'worker/receipts').glob('*.json'))==expected_files,'Unexpected native attempt artifacts')
    for pair in ('silence','noise'):
        for key in ('audio_features_sha256','audio_mask_sha256'):
            require(outputs[pair+':unhinted'][key]==outputs[pair+':hinted'][key],'Paired physical audio features differ')
    controls={k:native.control_metrics(plan,v) for k,v in outputs.items()};clean=all(c['no_speech'] for c in controls.values())
    require(same(controls,worker['controls']) and worker['status']==('controls_clear' if clean else 'negative_control_output')
        and worker['eligible_for_separate_real_test'] is clean,'Control classification changed')
    return {'passed':clean,'reason_code':worker['status'],'controls':controls,'asr_calls':4,'model_loads':1,
        'real_audio_calls':0,'new_recap_calls':0,'eligible_for_separate_real_test':clean,'automatic_real_dispatch':False,
        'future_real_paired_calls_if_separately_declared':16,'candidate_generated':False,'score':None,'accuracy_verified':False}


def execute(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(state(folder)['status']=='prepared','CAUTO1 cannot resume or reroll')
        previous=None;captured=False;process=None;failure=None
        try:
            with phase(folder,'input_replay'):
                scheduling();reg=validate_registration(folder)
                require(state(folder)['version']==VERSION and state(folder)['registration_sha256']==reg['registration_sha256'],'State identity changed')
            s=state(folder);s.update(status='running',started_utc=now());write_json(folder/'diagnostic.json',s)
            with phase(folder,'controls_worker') as deadline:
                previous=base.late._backend_identity(base.rw._request_json(base.rw.ADMIN+'/status',admin=True));captured=True
                immutable(folder/'original-backend.json',{'model':previous,'captured_utc':now()})
                s=state(folder);s['original_backend_restored']=False;write_json(folder/'diagnostic.json',s)
                base.rw._request_json(base.rw.ADMIN+'/unload',{},admin=True,timeout=60)
                require(base.late._backend_identity(base.rw._request_json(base.rw.ADMIN+'/status',admin=True)) is None,'Backend not unloaded')
                assets=read(folder/'assets.json');seconds=deadline-time.monotonic();require(seconds>0,'No controls work time remains')
                spec={'version':native.VERSION,'plan_path':str(folder/'native-plan.json'),'plan_file_sha256':file_hash(folder/'native-plan.json'),
                    'output_dir':str(folder/'worker'),'assets':assets,'worker_seconds':seconds,
                    'work_deadline_utc':(datetime.now(timezone.utc)+timedelta(seconds=seconds)).isoformat(),'worker_sha256':file_hash(Path(native.__file__))}
                immutable(folder/'worker-spec.json',spec)
                env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'}
                with (folder/'worker-process.log').open('xb') as log:
                    process=subprocess.Popen([assets['interpreter'],str(Path(native.__file__).resolve()),'--spec',str(folder/'worker-spec.json')],
                        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    code=process.wait(timeout=max(.1,deadline-time.monotonic()))
                immutable(folder/'worker-process.json',{'pid':process.pid,'exit_code':code,'finished_utc':now()});require(code==0,'Automatic-language worker failed')
        except BaseException as exc:failure=exc
        finally:
            if captured:
                try:base.restore(folder,previous,process)
                except BaseException as exc:
                    if failure is None:failure=exc
                    else:failure.add_note('Owned restoration failed: '+type(exc).__name__)
        if state(folder)['local_seconds']>600 and failure is None:failure=base.rw.LocalBudgetExceeded('CAUTO1 inclusive600s exceeded')
        if failure is not None:base._failed(folder,failure);raise failure
        try:
            with phase(folder,'final_native_replay'):
                validate_registration(folder);result=verify_worker(folder,read(folder/'native-plan.json'),read(folder/'assets.json'))
                artifacts={str(p.resolve()):file_hash(p) for p in folder.rglob('*') if p.is_file() and p.name not in ('.lock','diagnostic.json','result.json','results-summary.json')}
            s=state(folder);s.update(status='complete' if result['passed'] else 'retired',passed=result['passed'],reason_code=result['reason_code'],finished_utc=now())
            write_json(folder/'diagnostic.json',s);artifacts[str((folder/'diagnostic.json').resolve())]=file_hash(folder/'diagnostic.json')
            result.update(version=VERSION,local_seconds=s['local_seconds'],work_seconds=s['work_seconds'],original_backend_restored=True,artifacts=artifacts)
            immutable(folder/'result.json',result)
            immutable(folder/'results-summary.json',{k:v for k,v in result.items() if k!='artifacts'})
            return result
        except BaseException as exc:base._failed(folder,exc);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true')
    args=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    try:
        value=prepare() if args.prepare else execute();LOG.info('CAUTO1 status=%s passed=%s',value.get('status'),value.get('passed'));return 0
    except Exception as exc:LOG.error('CAUTO1 stopped: %s',type(exc).__name__);return 1
if __name__=='__main__':raise SystemExit(main())
