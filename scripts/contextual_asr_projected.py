"""C-ASR2: reuse a projected saved local recap; never generate another recap."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import contextual_asr_diagnostic as base
from src import contextual_asr_projected_recap_native as projected
from src.workflow_state import file_hash,fingerprint,write_json
from src.contextual_review_contract import require

VERSION='contextual-asr-projected-diagnostic-1'
FOLDER=ROOT/'output/quality-contextual-asr-projected-20260915'
PARENT=ROOT/'output/quality-contextual-asr-gain-20260915'
PLAN=ROOT/'design/quality-contextual-asr-projection-20260915.md'
PARENT_SECONDS=50.072952143033035
PRIOR=PARENT_SECONDS+0.09006963201682083
KEY=base.KEY
read,immutable,same,locked=base.read,base.immutable,base.same,base.locked
state,phase,remaining,now=base.state,base.phase,base.remaining,base.now
LOG=logging.getLogger(__name__)


def tracker(path,raw=False):
    return Path(path).read_bytes() if raw else read(path)


def projection(prep):
    return projected.replay_native(PARENT/'recap'/(KEY+'.json'),prep,tracker)


def prerequisite():
    prior=base.prerequisite();s=read(PARENT/'diagnostic.json')
    require(s['status']=='failed' and s['original_backend_restored'] is True
        and s['local_seconds']==PARENT_SECONDS,'Parent is not the recorded restored failure')
    require(not (PARENT/'worker').exists() and not (PARENT/'native-plan.json').exists(),'Parent already dispatched or bound ASR')
    return prior


def producers():
    from src import contextual_asr_recap_projection
    return [*base.producers(),Path(__file__).resolve(),PLAN,Path(projected.__file__),
        Path(contextual_asr_recap_projection.__file__)]


def validate_registration(folder):
    reg=read(folder/'registration.json')
    require(reg['version']==VERSION and reg['registration_sha256']==fingerprint({k:v for k,v in reg.items() if k!='registration_sha256'}),'Registration changed')
    fixed=('input-capsule.json','preparation.json','assets.json','execution-identity.json','audio-bindings.json','audio-preparation.json','projection.json','native-plan.json','recap/'+KEY+'.json','recap/metrics.jsonl')
    required={str(p.resolve()) for p in producers()}|{str(folder/name) for name in fixed}
    require(required<=set(reg['pins']),'Required projection input/producer missing')
    for path,digest in reg['pins'].items():require(file_hash(Path(path))==digest,'Projection input/producer changed')
    require(same(prerequisite(),reg['prerequisite']),'Prerequisite changed')
    capsule=base.inputs.load_inputs();require(same(capsule,read(folder/'input-capsule.json')),'Source input changed')
    prep=read(folder/'preparation.json')
    base.requests.validate_preparation(prep,capsule['source_rows'],capsule['original_context'],
        source_provenance=capsule['source_provenance'],detector_record=capsule['detector_record'],detector_provenance=capsule['detector_provenance'])
    require(same(prep,read(PARENT/'preparation.json')),'Original recap preparation changed')
    result=projection(prep);require(same(result,read(folder/'projection.json')),'Derived recap projection changed')
    require((folder/'recap'/(KEY+'.json')).read_bytes()==(PARENT/'recap'/(KEY+'.json')).read_bytes()
        and (folder/'recap/metrics.jsonl').read_bytes()==(PARENT/'recap/metrics.jsonl').read_bytes(),'Copied failed native receipt changed')
    assets=read(folder/'assets.json');identity=base.native.execution_identity(assets)
    require(same(identity,read(folder/'execution-identity.json')),'ASR execution identity changed')
    base.native.validate_assets(assets,identity,runtime_check=True)
    bindings=read(folder/'audio-bindings.json');base.validate_audio(folder,prep,capsule,bindings)
    base.requests.validate_plan(read(folder/'native-plan.json'),prep,result['response'],bindings,identity,
        recap_native_sha256=file_hash(folder/'projection.json'))
    require(all(reg['pins'].get(p)==h for p,h in capsule['input_hashes'].items()),'Source lineage pin missing')
    return reg


def prepare(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(set(folder.iterdir())=={folder/'.lock'},'Projection diagnostic already attempted')
        write_json(folder/'diagnostic.json',{'version':VERSION,'status':'preparing','local_seconds':PRIOR,'work_seconds':PRIOR,
            'stages':[{'stage':'completed_parent_and_failure_validation','seconds':PRIOR,'error_type':'ValueError'}],
            'passed':False,'original_backend_restored':True,'created_utc':now()})
        try:
            with phase(folder,'preparation',180-PRIOR):
                prior=prerequisite();parent_reg=base.validate_registration(PARENT)
                capsule=base.inputs.load_inputs();prep=read(PARENT/'preparation.json')
                result=projection(prep)
                assets=base.build_assets();identity=base.native.execution_identity(assets)
                base.native.validate_assets(assets,identity,runtime_check=True)
                bindings=base.prepare_audio(folder,prep,capsule['mono_path'])
                for name,value in [('input-capsule.json',capsule),('preparation.json',prep),('assets.json',assets),
                    ('execution-identity.json',identity),('audio-bindings.json',bindings),('projection.json',result)]:immutable(folder/name,value)
                for name in (KEY+'.json','metrics.jsonl'):
                    immutable(folder/'recap'/name,(PARENT/'recap'/name).read_bytes(),raw=True)
                plan=base.requests.bind_recap(prep,result['response'],bindings,identity,recap_native_sha256=file_hash(folder/'projection.json'))
                immutable(folder/'native-plan.json',plan)
                paths=[*producers(),PARENT/'registration.json',PARENT/'diagnostic.json',PARENT/'preparation.json',
                       PARENT/'recap'/(KEY+'.json'),PARENT/'recap/metrics.jsonl']
                paths.extend(p for p in folder.rglob('*') if p.is_file() and p.name not in ('.lock','diagnostic.json'))
                pins={str(p.resolve()):file_hash(p) for p in paths};pins.update(capsule['input_hashes']);pins.update(prior['pins'])
                reg={'version':VERSION,'prerequisite':prior,'maximum_asr_calls':20,'new_recap_calls':0,'carried_local_seconds':PRIOR,'pins':pins}
                reg['registration_sha256']=fingerprint(reg);immutable(folder/'registration.json',reg)
            s=state(folder);s.update(status='prepared',registration_sha256=reg['registration_sha256']);write_json(folder/'diagnostic.json',s)
            return s
        except BaseException as exc:base._failed(folder,exc);raise


def execute(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(state(folder)['status']=='prepared','Diagnostic cannot resume or reroll')
        previous=None;captured=False;process=None;failure=None
        try:
            with phase(folder,'input_replay'):
                reg=validate_registration(folder)
                require(state(folder)['version']==VERSION and state(folder)['registration_sha256']==reg['registration_sha256'],'State identity changed')
            s=state(folder);s.update(status='running',started_utc=now());write_json(folder/'diagnostic.json',s)
            with phase(folder,'paired_asr',900) as deadline:
                previous=base.late._backend_identity(base.rw._request_json(base.rw.ADMIN+'/status',admin=True));captured=True
                immutable(folder/'original-backend.json',{'model':previous,'captured_utc':now()})
                s=state(folder);s.update(original_backend_restored=False,original_backend=previous);write_json(folder/'diagnostic.json',s)
                base.rw._request_json(base.rw.ADMIN+'/unload',{},admin=True,timeout=60)
                require(base.late._backend_identity(base.rw._request_json(base.rw.ADMIN+'/status',admin=True)) is None,'Backend not unloaded')
                assets=read(folder/'assets.json');seconds=deadline-time.monotonic();require(seconds>0,'No ASR time remains')
                spec={'version':base.native.VERSION,'plan_path':str(folder/'native-plan.json'),'plan_file_sha256':file_hash(folder/'native-plan.json'),
                    'output_dir':str(folder/'worker'),'assets':assets,'worker_seconds':seconds,
                    'work_deadline_utc':(datetime.now(timezone.utc)+timedelta(seconds=seconds)).isoformat(),'worker_sha256':file_hash(Path(base.native.__file__))}
                immutable(folder/'worker-spec.json',spec)
                env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'}
                with (folder/'worker-process.log').open('xb') as log:
                    process=subprocess.Popen([assets['interpreter'],str(Path(base.native.__file__).resolve()),'--spec',str(folder/'worker-spec.json')],
                        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    code=process.wait(timeout=max(.1,deadline-time.monotonic()))
                immutable(folder/'worker-process.json',{'pid':process.pid,'exit_code':code,'finished_utc':now()});require(code==0,'ASR worker failed')
        except BaseException as exc:failure=exc
        finally:
            if captured:
                try:base.restore(folder,previous,process)
                except BaseException as exc:
                    if failure is None:failure=exc
                    else:failure.add_note('Cleanup failed: '+type(exc).__name__)
        if failure is not None:base._failed(folder,failure);raise failure
        try:
            with phase(folder,'final_native_replay'):
                validate_registration(folder)
                result=base.verify_worker(folder,read(folder/'native-plan.json'),read(folder/'assets.json'))
                artifacts={str(p.resolve()):file_hash(p) for p in folder.rglob('*') if p.is_file() and p.name not in ('.lock','diagnostic.json','result.json')}
            s=state(folder);s.update(status='complete' if result['passed'] else 'retired',passed=result['passed'],reason_code=result['reason_code'],finished_utc=now());write_json(folder/'diagnostic.json',s)
            artifacts[str((folder/'diagnostic.json').resolve())]=file_hash(folder/'diagnostic.json')
            immutable(folder/'result.json',{**result,'version':VERSION,'artifacts':artifacts,'local_seconds':s['local_seconds'],'work_seconds':s['work_seconds'],
                'original_backend_restored':s['original_backend_restored'],'candidate_generated':False,'score':None,'accuracy_verified':False,'new_recap_calls':0})
            return read(folder/'result.json')
        except BaseException as exc:base._failed(folder,exc);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    try:
        result=prepare() if a.prepare else execute();LOG.info('C-ASR2 status=%s passed=%s',result.get('status'),result.get('passed'));return 0
    except Exception as exc:LOG.error('C-ASR2 stopped: %s; artifacts retained',type(exc).__name__);return 1
if __name__=='__main__':raise SystemExit(main())
