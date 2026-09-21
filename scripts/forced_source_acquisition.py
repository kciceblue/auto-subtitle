"""AU-S2: fixed nine-window forced-Japanese source treatment; no writing/scoring."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
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
from scripts import auto_source_acquisition as automatic
from src import forced_source_native as native
from src import contextual_asr_recap_native as deadline_helper
from src.workflow_state import file_hash,fingerprint,write_json
from src.contextual_review_contract import require
VERSION='forced-source-acquisition-1'
FOLDER=ROOT/'output/quality-forced-source-20260916'
PARENT=ROOT/'output/quality-auto-source-20260915'
PLAN=ROOT/'design/quality-forced-source-acquisition-20260916.md'
ELIGIBILITY=ROOT/'design/quality-auto-source-forced-language-eligibility-20260916.json'
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


def producers():
    return [PLAN,ELIGIBILITY,Path(__file__).resolve(),Path(native.__file__),Path(automatic.__file__),Path(base.__file__),
        Path(native.automatic.__file__),Path(native.base.__file__),Path(native.ec.__file__),Path(deadline_helper.__file__),
        ROOT/'src/contextual_asr_auto_native.py',ROOT/'src/contextual_asr_requests.py',ROOT/'src/workflow_state.py',
        ROOT/'src/local_backend.py',ROOT/'src/late_audio.py',ROOT/'src/revisit_workflow.py']


def validate_registration(folder):
    reg=read(folder/'registration.json')
    require(reg['version']==VERSION and same(reg['policy'],native.POLICY)
        and reg['registration_sha256']==fingerprint({k:v for k,v in reg.items() if k!='registration_sha256'}),'Forced registration changed')
    fixed={str(p.resolve()) for p in producers()}|{str(folder/name) for name in
        ('native-plan.json','assets.json','input-proof.json','automatic-observations.json','automatic-plan.json','eligibility.json')}
    require(fixed<=set(reg['pins']),'Required forced input/producer missing')
    for path,digest in reg['pins'].items():require(file_hash(Path(path))==digest,'Pinned forced acquisition input/producer changed')
    proof=read(folder/'input-proof.json');automatic_source=read(folder/'automatic-observations.json');prior_plan=read(folder/'automatic-plan.json')
    require(proof['original_backend_restored'] is True and all(reg['pins'].get(p)==h for p,h in proof['pins'].items())
        and proof['automatic_source_sha256']==native.ec._hash(automatic_source)
        and proof['automatic_plan_sha256']==fingerprint(prior_plan),'Automatic source proof changed')
    require(same(automatic_source,read(PARENT/'observations.json')) and same(prior_plan,read(PARENT/'native-plan.json'))
        and same(read(folder/'assets.json'),read(PARENT/'assets.json')),'Copied automatic source/asset identity changed')
    require(same(read(folder/'eligibility.json'),read(ELIGIBILITY)),'Pre-score eligibility record changed')
    expected=native.build_plan(automatic_source,prior_plan,read(folder/'assets.json'),read(folder/'eligibility.json'))
    require(same(read(folder/'native-plan.json'),expected),'Forced source request plan changed')
    return reg


def prepare(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(set(folder.iterdir())=={folder/'.lock'},'Forced source acquisition already attempted')
        write_json(folder/'diagnostic.json',{'version':VERSION,'status':'preparing','local_seconds':0.,'work_seconds':0.,
            'stages':[],'passed':False,'original_backend_restored':True,'created_utc':now()})
        try:
            with phase(folder,'preparation',120):
                acquired=automatic.replay_acquisition(PARENT)
                source=acquired['envelope'];prior_plan=read(PARENT/'native-plan.json');assets=read(PARENT/'assets.json');eligibility=read(ELIGIBILITY)
                plan=native.build_plan(source,prior_plan,assets,eligibility)
                proof={k:deepcopy(acquired[k]) for k in ('local_seconds','work_seconds','original_backend_restored','pins','source_bindings')}
                proof.update(automatic_source_sha256=native.ec._hash(source),automatic_plan_sha256=fingerprint(prior_plan))
                for name,value in [('native-plan.json',plan),('assets.json',assets),('input-proof.json',proof),
                    ('automatic-observations.json',source),('automatic-plan.json',prior_plan),('eligibility.json',eligibility)]:immutable(folder/name,value)
                paths=[*producers(),*(p for p in folder.rglob('*') if p.is_file() and p.name not in ('.lock','diagnostic.json'))]
                pins={**acquired['pins'],**{str(p.resolve()):file_hash(p) for p in paths}}
                reg={'version':VERSION,'policy':native.POLICY,'pins':pins};reg['registration_sha256']=fingerprint(reg);immutable(folder/'registration.json',reg)
            s=state(folder);s.update(status='prepared',registration_sha256=reg['registration_sha256']);write_json(folder/'diagnostic.json',s);return s
        except BaseException as exc:base._failed(folder,exc);raise


def observation_envelope(folder,plan,outputs):
    rows=[]
    for number,slot in enumerate(plan['slots'],1):
        i=slot['owner_id'];value=outputs[slot['slot_id']];request=native.build_request(plan,i)
        receipt=folder/'worker/receipts'/f'{number:02d}-owner-{i:03d}.json'
        rows.append({'observation_id':request['observation_id'],'parent_observation_id':request['parent_observation_id'],
            **deepcopy(plan['geometry'][number-1]),'raw_text':value['raw_text'],'text':value['parsed_text'],
            'parser_language':value['parser_language'],'forced_language':'Japanese','language_origin':'forced',
            'native_status':'empty' if value['parsed_text']=='' else 'complete',
            'protocol_category':native.protocol_category(value['raw_text'],value['parsed_text']),
            'audio_sha256':request['audio']['sha256'],'pcm_sha256':request['audio']['pcm_sha256'],
            'receipt':{'path':str(receipt),'sha256':file_hash(receipt),'request_sha256':native.ec._hash(request)}})
    return {'version':native.OBSERVATIONS_VERSION,'plan_sha256':plan['plan_sha256'],
        'automatic_source_sha256':plan['automatic_source_sha256'],'parent_observations_sha256':plan['parent_observations_sha256'],
        'eligible_owner_ids':list(native.OWNERS),'source_rows_sha256':plan['source_rows_sha256'],
        'original_mono_sha256':plan['original_mono_sha256'],'original_mono_frames':plan['original_mono_frames'],'sample_rate':16000,
        'execution_identity':deepcopy(plan['execution_identity']),'conditioning':deepcopy(native.CONDITIONING),
        'observations':rows,'observations_sha256':native.ec._hash(rows)}

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
        and type(worker['asr_calls']) is int and worker['asr_calls']==9
        and type(worker['model_loads']) is int and worker['model_loads']==1
        and type(worker['model_load_attempts']) is int and worker['model_load_attempts']==1,'Worker count/time/binding differs')
    native.check_plan(plan,assets);processor=base.native.load_processor(assets);outputs={};expected_files=set()
    require(len(worker['slots'])==9,'Exactly nine fixed source windows are authorized')
    previous=native._time(original['captured_utc'])
    for number,(slot,expected) in enumerate(zip(worker['slots'],plan['slots']),1):
        path=folder/'worker/receipts'/f'{number:02d}-owner-{expected["owner_id"]:03d}.json'
        require(same(slot,{**expected,'number':number,'status':slot['status'],'receipt':str(path)})
            and slot['status'] in ('empty','complete'),'Unexpected native control slot')
        receipt=read(path);reservation_path=path.with_suffix('.started.json');reservation=read(reservation_path)
        expected_files.update({path,reservation_path});request=native.build_request(plan,slot['owner_id'])
        require(set(reservation)=={'version','request_sha256','started_utc'} and reservation['version']==native.VERSION
            and reservation['request_sha256']==native.ec._hash(request)
            and previous<=native._time(reservation['started_utc'])<=native._time(receipt['started_utc'])
            and receipt['status']==slot['status'],'Control reservation/order changed')
        previous=native._time(receipt['finished_utc'])
        outputs[slot['slot_id']]=native.validate_native_receipt(receipt,request,plan,assets=assets,artifact_root=folder/'worker',processor=processor)
    require(set((folder/'worker/receipts').glob('*.json'))==expected_files,'Unexpected native attempt artifacts')
    metrics=native.summarize(plan,outputs)
    require(all(same(worker.get(k),v) for k,v in metrics.items()) and worker['status']=='complete'
        and type(worker['available_slots']) is int and worker['available_slots']==9,'Native summary changed')
    for slot in plan['slots']:
        owner=slot['owner_id'];pin=plan['parent_receipts'][str(owner)];path=Path(pin['path'])
        require(file_hash(path)==pin['sha256'],'Paired automatic receipt changed')
        parent=native._sealed(read(path));capture=parent['capture'];output=outputs[slot['slot_id']]
        require(parent['request_sha256']==pin['request_sha256'],'Paired automatic request changed')
        for key,field in [('audio_features_sha256','input_features'),('audio_mask_sha256','feature_attention_mask')]:
            require(output[key]==native.tensor_identity(capture['outer_inputs'][field]),'Paired automatic/forced audio features differ')
    result={**metrics,'passed':True,'reason_code':'complete_native_observations','asr_calls':9,'model_loads':1,
        'real_audio_calls':9,'new_recap_calls':0,'control_calls':0,'candidate_generated':False,
        'score':None,'accuracy_verified':False,'independent_recognizer_vote':False,'automatic_downstream_dispatch':False}
    return result,observation_envelope(folder,plan,outputs)


def execute(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(state(folder)['status']=='prepared','Forced source acquisition cannot resume or reroll')
        previous=None;captured=False;process=None;failure=None
        try:
            with phase(folder,'input_replay'):
                reg=validate_registration(folder)
                require(state(folder)['version']==VERSION and state(folder)['registration_sha256']==reg['registration_sha256'],'State identity changed')
            s=state(folder);s.update(status='running',started_utc=now());write_json(folder/'diagnostic.json',s)
            with phase(folder,'forced_source_worker') as deadline:
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
                try:
                    base.restore(folder,previous,process)
                    if read(folder/'lifecycle.json')['seconds']>360:raise base.rw.LocalBudgetExceeded('Cleanup exceeded360s reserve')
                except BaseException as exc:
                    if failure is None:failure=exc
                    else:failure.add_note('Owned restoration failed: '+type(exc).__name__)
        if state(folder)['local_seconds']>600 and failure is None:failure=base.rw.LocalBudgetExceeded('Forced source acquisition inclusive600s exceeded')
        if failure is not None:base._failed(folder,failure);raise failure
        try:
            with phase(folder,'final_native_replay'):
                validate_registration(folder);result,envelope=verify_worker(folder,read(folder/'native-plan.json'),read(folder/'assets.json'))
                immutable(folder/'observations.json',envelope)
                artifacts={str(p.resolve()):file_hash(p) for p in folder.rglob('*') if p.is_file() and p.name not in ('.lock','diagnostic.json','result.json','results-summary.json')}
            s=state(folder);s.update(status='complete' if result['passed'] else 'retired',passed=result['passed'],reason_code=result['reason_code'],finished_utc=now())
            write_json(folder/'diagnostic.json',s);artifacts[str((folder/'diagnostic.json').resolve())]=file_hash(folder/'diagnostic.json')
            result.update(version=VERSION,local_seconds=s['local_seconds'],work_seconds=s['work_seconds'],original_backend_restored=True,artifacts=artifacts)
            immutable(folder/'result.json',result)
            immutable(folder/'results-summary.json',{k:v for k,v in result.items() if k!='artifacts'})
            return result
        except BaseException as exc:base._failed(folder,exc);raise


def replay_acquisition(folder,track=None):
    """CPU/native receipt replay for a later writer; never starts acquisition."""
    folder=Path(folder).resolve();reg=validate_registration(folder);s=state(folder);result=read(folder/'result.json')
    require(s['version']==VERSION and s['status']=='complete' and s['original_backend_restored'] is True
        and s['registration_sha256']==reg['registration_sha256'] and s['passed'] is True
        and 0<=s['work_seconds']<=240 and s['work_seconds']<=s['local_seconds']<=600,'Acquisition is not complete/restored/in budget')
    expected,envelope=verify_worker(folder,read(folder/'native-plan.json'),read(folder/'assets.json'))
    require(all(same(result.get(k),v) for k,v in expected.items()) and result['version']==VERSION
        and result['local_seconds']==s['local_seconds'] and result['work_seconds']==s['work_seconds']
        and result['original_backend_restored'] is True and same(envelope,read(folder/'observations.json')),'Saved forced observations/result changed')
    required={str(p.resolve()):file_hash(p) for p in folder.rglob('*') if p.is_file() and p.name not in ('.lock','result.json','results-summary.json')}
    require(same(required,result['artifacts']),'Acquisition artifact coverage changed')
    pins={**reg['pins'],**required,str(folder/'result.json'):file_hash(folder/'result.json')}
    if track is not None:
        for path in pins:track(Path(path))
    return {'envelope':envelope,'source_bindings':read(folder/'input-proof.json')['source_bindings'],
        'local_seconds':s['local_seconds'],'work_seconds':s['work_seconds'],
        'original_backend_restored':True,'pins':pins}


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true')
    args=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    try:
        value=prepare() if args.prepare else execute();LOG.info('Forced source acquisition status=%s passed=%s',value.get('status'),value.get('passed'));return 0
    except Exception as exc:LOG.error('Forced source acquisition stopped: %s',type(exc).__name__);return 1
if __name__=='__main__':raise SystemExit(main())
