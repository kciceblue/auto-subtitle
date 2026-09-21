"""LC-G1: exact B1 raw-evidence draft on owned Gemma31 at163840 context."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import logging
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import staged_source_screen as ss
from src import long_context_gemma_native as native
from src.contextual_review_contract import require
from src.workflow_state import file_hash,fingerprint,write_json

VERSION='long-context-gemma-screen-1'
ROUND='LC-G1'
FOLDER=ROOT/'output/quality-long-context-gemma-20260915'
SS_PARENT=ss.FOLDER
B_PARENT=ss.B_PARENT
B_REQUEST_SHA256=ss.B_REQUEST_SHA256
PLAN=ROOT/'design/quality-long-context-gemma-20260915.md'
PROFILE=ROOT/'profiles/long-context-gemma.json'
LIMIT,WORK_LIMIT,CLEANUP_LIMIT=2400,1980,420
KEY='whole-draft'
read,immutable,same,locked=ss.read,ss.immutable,ss.same,ss.locked
state,_save,_failed,counted=ss.state,ss._save,ss._failed,ss.counted
catalog,er,sr,rw,lb=ss.catalog,ss.er,ss.sr,ss.rw,ss.local_backend
Tracker=ss.Tracker
LOG=logging.getLogger(__name__)


def producers():
    modules=(ss,native,ss.brequests,ss.codec,catalog,er,sr,rw,lb,ss.late_audio)
    return [PLAN,PROFILE,Path(__file__).resolve(),*[Path(m.__file__) for m in modules],
        ROOT/'src/staged_source_native.py',ROOT/'src/staged_source_requests.py',
        ROOT/'src/contextual_asr_recap_native.py',ROOT/'src/evidence_context.py',
        ROOT/'src/raw_evidence_draft_requests.py',ROOT/'src/readable_draft_requests.py',
        ROOT/'src/readable_evidence.py',ROOT/'src/contextual_review_contract.py',
        ROOT/'src/translate.py',ROOT/'src/config.py',ROOT/'src/selected_pipeline.py',
        ROOT/'src/workflow_state.py',Path(sys.modules[Tracker.__module__].__file__)]


def _receipt_pins(receipt,track):
    for name,digest in receipt['receipt_hashes'].items():
        require(Path(name).name==name,'Receipt path escaped its directory')
        path=Path(receipt['output_dir'])/name;track(path,raw=True)
        require(track.pins[str(path.resolve())]==digest,'Native scalar receipt changed')


def prerequisite(track):
    """Actual SS native draft and primary, not a status-only scheduling guess."""
    current=track(SS_PARENT/'screen.json');reg=ss.validate_registration(SS_PARENT)
    require(current['status']=='complete' and current['original_backend_restored'] is True
        and type(current.get('score')) is int and current['score']<4
        and current.get('qualified') is False and current.get('confirmed') is False
        and current.get('eligible_for_validation') is False
        and current['local_seconds']<LIMIT and current['work_seconds']<WORK_LIMIT
        and current['cleanup_seconds']<CLEANUP_LIMIT,'SS1 must be completed, restored and below four')
    track(SS_PARENT/'registration.json')
    for path in reg['pins']:track(Path(path),raw=True)
    native_track=Tracker();source,draft=ss.replay_local(SS_PARENT,reg,native_track)
    arm=ss._bundle_arm(SS_PARENT,reg,source,draft,SS_PARENT/'evaluation/bundle/manifest.json')
    for path in (SS_PARENT/'evaluation/bundle').rglob('*'):
        if path.is_file():native_track(path,raw=True)
    require(same(track(SS_PARENT/'score-inputs.json'),{**arm,'native_artifacts':native_track.pins}),
            'SS1 scored target does not match its exact native draft')
    for path in native_track.pins:track(Path(path),raw=True)
    wrapper=track(SS_PARENT/'score.json');receipt=er.validate_receipt(wrapper['primary'])
    catalog.validate_score(receipt,arm);_receipt_pins(receipt,track)
    require(receipt['purpose'] in ('candidate','baseline') and type(receipt['score']) is int and receipt['score']<4
        and wrapper['score']==current['score']==receipt['score'] and wrapper['status']=='complete'
        and wrapper['version']==ss.VERSION and wrapper['round']==ss.ROUND
        and wrapper['dispatch_id']==receipt['dispatch_id'] and same(wrapper['receipt_hashes'],receipt['receipt_hashes'])
        and all(same(wrapper.get(k),v) for k,v in arm.items()),'SS1 scalar wrapper/native receipt mismatch')
    primary=Path(arm['primary_directory'])
    if wrapper['reused_completed_primary']:
        require(not primary.exists() or not any(primary.iterdir()),'SS1 contains an unregistered own primary')
    else:require(Path(receipt['output_dir']).resolve()==primary.resolve(),'SS1 primary path mismatch')
    return {'folder':str(SS_PARENT),'score':receipt['score'],'target_sha256':arm['target_sha256'],
            'primary':receipt['output_dir'],'dispatch_id':receipt['dispatch_id']}


def _inputs(track):
    prior=prerequisite(track);breg=track(B_PARENT/'registration.json');request=track(B_PARENT/'request.json')
    require(fingerprint(request)==B_REQUEST_SHA256,'Fixed B1 request changed')
    coverage=ss.brequests.validate_writer_request(request);decoded=ss.codec.decode_body(request['body'])
    require(same(ss.codec.encode_body(decoded),request['body']),'Exact B codec changed')
    source=[ss.SrtBlock(**row) for row in track(B_PARENT/'source.json')]
    context_bytes=track(B_PARENT/'original-context.txt',raw=True);context=context_bytes.decode('utf-8',errors='strict')
    require(er.owner_rows_sha256(source)==catalog.SOURCE and hashlib.sha256(context_bytes).hexdigest()==catalog.CONTEXT,
            'Original source/context changed')
    require(same(decoded['source_evidence']['source_rows'],[asdict(row) for row in source])
        and decoded['source_evidence']['original_context']==context,'B source/context does not match originals')
    baseline=ss._baseline(track,breg,source,context)
    return prior,breg,request,source,context_bytes,coverage,baseline


def recipe():
    value=rw.load_writer_recipe(PROFILE)
    require(value['context_size']==163840 and value['full_swa'] is False and value['gpu_layers']==99
        and value['cpu_moe_layers']==0 and value['threads']==4,'Fixed LC-G1 deployment profile changed')
    return value


def validate_registration(folder):
    reg=read(folder/'registration.json')
    fixed={'version':VERSION,'round':ROUND,'maximum_local_seconds':LIMIT,'maximum_work_seconds':WORK_LIMIT,
        'maximum_cleanup_seconds':CLEANUP_LIMIT,'maximum_candidates':1,'maximum_writer_calls':1,
        'maximum_primary_dispatches':1,'maximum_confirmations':1,'context_size':163840,'output_reserve':16384,
        'source_sha256':catalog.SOURCE,'context_sha256':catalog.CONTEXT,'pool_version':catalog.POOL,
        'qualified':False,'source_accuracy_verified':False}
    require(same({k:reg.get(k) for k in fixed},fixed) and reg['registration_sha256']==fingerprint({k:v for k,v in reg.items() if k!='registration_sha256'}),'LC-G1 registration changed')
    required={str(p.resolve()) for p in producers()}|{str(folder/name) for name in
        ('source.json','original-context.txt','B-request.json','native-request.json','score-index-before.json')}
    require(required<=set(reg['pins']),'Mandatory LC-G1 pin missing')
    for path,digest in reg['pins'].items():require(file_hash(Path(path))==digest,'LC-G1 input/producer changed')
    track=Tracker();prior,breg,request,source,context,coverage,baseline=_inputs(track)
    require(all(reg['pins'].get(path)==digest for path,digest in track.pins.items()),'LC-G1 direct pin coverage changed')
    require(same(reg['ss_prerequisite'],prior) and same(reg['coverage'],coverage)
        and same(read(folder/'source.json'),[asdict(row) for row in source])
        and (folder/'original-context.txt').read_bytes()==context and same(read(folder/'B-request.json'),request)
        and same(read(folder/'native-request.json'),native.prepare_request(request))
        and reg['baseline_review']==baseline['output_dir'] and same(reg['evidence_paths'],breg['evidence_paths']),
        'LC-G1 request/source binding changed')
    recipe();return reg


def prepare(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(set(folder.iterdir())=={folder/'.lock'},'LC-G1 already initialized; no restart')
        write_json(folder/'screen.json',{'version':VERSION,'round':ROUND,'status':'preparing','local_seconds':0.,
            'work_seconds':0.,'cleanup_seconds':0.,'stages':[],'original_backend_restored':True,
            'qualified':False,'confirmed':False,'eligible_for_validation':False})
        try:
            with counted(folder,'direct_prerequisite_and_input_preparation'):
                track=Tracker();prior,breg,request,source,context,coverage,baseline=_inputs(track);recipe()
                index=catalog.load_score_index(track)
                for path in [*ss.EXTRA_SCORES,SS_PARENT/'score.json']:
                    if path.exists():
                        value=track(path);index['records'].append({'path':str(path),'metadata_sha256':track.pins[str(path.resolve())],
                            'hashes':{k:value.get(k) for k in ('pool_version','target_sha256','source_sha256')}})
                for name,value in [('source.json',[asdict(row) for row in source]),('B-request.json',request),
                                   ('native-request.json',native.prepare_request(request)),('score-index-before.json',index)]:immutable(folder/name,value)
                immutable(folder/'original-context.txt',context,raw=True)
                for path in [*producers(),*[p for p in folder.rglob('*') if p.is_file() and p.name not in ('.lock','screen.json')]]:track(path,raw=True)
                reg={'version':VERSION,'round':ROUND,'maximum_local_seconds':LIMIT,'maximum_work_seconds':WORK_LIMIT,
                    'maximum_cleanup_seconds':CLEANUP_LIMIT,'maximum_candidates':1,'maximum_writer_calls':1,
                    'maximum_primary_dispatches':1,'maximum_confirmations':1,'context_size':163840,'output_reserve':16384,
                    'source_sha256':catalog.SOURCE,'context_sha256':catalog.CONTEXT,'pool_version':catalog.POOL,
                    'qualified':False,'source_accuracy_verified':False,'coverage':coverage,'ss_prerequisite':prior,
                    'baseline_review':baseline['output_dir'],'evidence_paths':breg['evidence_paths'],'pins':track.pins}
                reg['registration_sha256']=fingerprint(reg);immutable(folder/'registration.json',reg);validate_registration(folder)
            return _save(folder,status='prepared',registration_sha256=reg['registration_sha256'])
        except BaseException as exc:_failed(folder,exc);raise


def _run_writer(folder,request):
    manager=None;entered=False;identity=None;previous=None;failure=None;value=None
    try:
        with counted(folder,'long_context_writer') as deadline:
            settings=recipe();previous=ss._backend_state();_save(folder,original_backend_restored=False)
            backend_folder=folder/'writer-backend';backend_folder.mkdir()
            manager=lb.temporary_local_writer(settings['weights'],settings['server_binary'],alias=settings['model'],
                expected_sha256=settings['sha256'],context_size=163840,full_swa=False,threads=4,gpu_layers=99,
                cpu_moe_layers=0,restore_previous=True,admin_url=settings['admin_url'],
                restore_model=settings['restore_model'],log_path=backend_folder/'server.log')
            endpoint,identity=manager.__enter__();entered=True;immutable(backend_folder/'backend.json',identity)
            value=native.ask(folder/'requests',KEY,endpoint,request,identity,deadline)
    except BaseException as exc:failure=exc
    finally:
        if manager is not None:
            ok=False;error=None;closed=not entered
            try:
                with counted(folder,'owned_backend_cleanup',cleanup=True):
                    notes=tuple(getattr(failure,'__notes__',()))
                    if entered:manager.__exit__(type(failure) if failure else None,failure,failure.__traceback__ if failure else None)
                    require(tuple(getattr(failure,'__notes__',()))==notes,'Owned cleanup reported a failure')
                    closed=True;ok=closed and (identity is None or ss._pid_gone(identity['pid'])) and ss._backend_state()==previous
                    require(ok,'Owned Gemma did not stop/restore exactly')
            except BaseException as exc:
                error=type(exc).__name__
                if failure is None:failure=exc
                else:failure.add_note('LC-G1 cleanup failed: '+error)
            finally:
                immutable(folder/'writer-lifecycle.json',{'original_backend':previous,'original_backend_restored':ok,
                    'context_closed':closed,'error_type':error});_save(folder,original_backend_restored=ok)
    if failure is not None:raise failure
    return value


def _compile(response,source):
    require(type(response) is dict and set(response)=={'owners'} and set(response['owners'])=={str(row.index) for row in source},'Wrong whole draft owners')
    rows=[]
    for row in source:
        value=response['owners'][str(row.index)]
        require(type(value) is dict and set(value)=={'chinese'} and type(value['chinese']) is str and bool(value['chinese'].strip()),'Invalid whole draft text')
        rows.append(ss.SrtBlock(row.index,row.ts_line,value['chinese']))
    return rows


def replay_local(folder,reg,track):
    source=rw.rows(folder/'source.json');request=track(folder/'native-request.json')
    require(same(request,native.prepare_request(read(folder/'B-request.json'))),'Native B request differs')
    value=native.replay_native(folder/'requests'/f'{KEY}.json',request,track(folder/'writer-backend/backend.json'),track)
    draft=_compile(value,source)
    require(same(track(folder/'translation.json'),value),'Saved parsed draft changed')
    for name,rows in (('source',source),('draft',draft)):
        require(same(track(folder/(name+'.json')),[asdict(row) for row in rows])
            and ss.parse_srt(folder/(name+'.utterances.srt'),preserve_text_whitespace=True)==rows,'Exact rows/SRT changed')
        track(folder/(name+'.utterances.srt'),raw=True)
    life=track(folder/'writer-lifecycle.json')
    require(life['original_backend_restored'] is True and life['context_closed'] is True and life['error_type'] is None,'Owned cleanup failed')
    metrics=native.shared._metrics(track(folder/'requests/metrics.jsonl',raw=True))
    require(len(metrics)==1 and metrics[0]['stage']==KEY
        and {p.name for p in (folder/'requests').glob('*.json')}=={f'{KEY}.json'},'Extra native writer attempt')
    return source,draft


def _arm(folder,reg,source,draft):
    arm={'label':ROUND,'source_sha256':reg['source_sha256'],'context_sha256':reg['context_sha256'],
        'pool_version':reg['pool_version'],'target_sha256':er.owner_rows_sha256(draft),
        'manifest':str(folder/'evaluation/bundle/manifest.json'),'primary_directory':str(folder/'evaluation/primary')}
    bundle=er.read_bundle(arm['manifest']);catalog.validate_score(bundle,arm)
    require(same(bundle['rows']['source'],[asdict(row) for row in source]) and same(bundle['rows']['raw'],[asdict(row) for row in source])
        and same(bundle['rows']['target'],[asdict(row) for row in draft]),'External source/target bundle differs')
    return arm


def execute_local(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(state(folder)['status']=='prepared','LC-G1 native attempt cannot resume or reroll')
        try:
            with counted(folder,'execution_input_replay'):
                reg=validate_registration(folder);require(state(folder)['registration_sha256']==reg['registration_sha256'],'State binding changed')
                _save(folder,status='running',started_utc=sr.now());source=rw.rows(folder/'source.json');sr.write_checked(source,folder/'source.utterances.srt')
            value=_run_writer(folder,read(folder/'native-request.json'))
            with counted(folder,'exact_draft_and_unchanged_pool_bundle'):
                immutable(folder/'translation.json',value);ss._save_rows(folder,'draft',_compile(value,source))
                track=Tracker();source,draft=replay_local(folder,reg,track)
                er.prepare_bundle(source,draft,source,list(map(Path,reg['evidence_paths'])),folder/'original-context.txt',folder/'evaluation/bundle')
                arm=_arm(folder,reg,source,draft)
                for path in (folder/'evaluation/bundle').rglob('*'):
                    if path.is_file():track(path,raw=True)
                immutable(folder/'score-inputs.json',{**arm,'native_artifacts':track.pins})
            return _save(folder,status='local_complete',finished_local_utc=sr.now())
        except BaseException as exc:_failed(folder,exc);raise


def _score_inputs(folder):
    reg=validate_registration(folder);require(state(folder)['original_backend_restored'] is True,'Backend not restored')
    track=Tracker();source,draft=replay_local(folder,reg,track);arm=_arm(folder,reg,source,draft)
    for path in (folder/'evaluation/bundle').rglob('*'):
        if path.is_file():track(path,raw=True)
    require(same(read(folder/'score-inputs.json'),{**arm,'native_artifacts':track.pins}),'Scoring artifacts changed')
    return reg,arm


def score(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(state(folder)['status']=='local_complete','LC-G1 primary reservation cannot be retried')
        try:
            with counted(folder,'primary_preflight'):
                reg,arm=_score_inputs(folder);primary=Path(arm['primary_directory'])
                require(not primary.exists() or not any(primary.iterdir()),'Own primary reservation already exists')
                prior=catalog.duplicates(read(folder/'score-index-before.json'),arm,folder)
                _save(folder,status='primary_reserved',review_reserved_utc=sr.now())
            started=time.monotonic()
            receipt=prior or sr.review_once(arm['manifest'],Path(arm['primary_directory']),purpose='candidate',baseline_review=reg['baseline_review'])
            with counted(folder,'primary_validation'):
                receipt=er.validate_receipt(receipt['output_dir']);catalog.validate_score(receipt,arm)
                require(receipt['purpose'] in ('candidate','baseline'),'Not a primary receipt')
                result={**arm,'version':VERSION,'round':ROUND,'status':'complete','score':receipt['score'],
                    'primary':receipt['output_dir'],'dispatch_id':receipt['dispatch_id'],'receipt_hashes':receipt['receipt_hashes'],
                    'reused_completed_primary':prior is not None,'runner_score_seconds':time.monotonic()-started,
                    'qualified':False,'confirmed':False,'eligible_for_validation':receipt['score']>=4}
                immutable(folder/'score.json',result)
            return _save(folder,status='complete',score=result['score'],eligible_for_validation=result['eligible_for_validation'],finished_utc=sr.now())
        except BaseException as exc:_failed(folder,exc);raise



def _primary_receipt(folder,arm):
    wrapper=read(folder/'score.json');receipt=er.validate_receipt(wrapper['primary'])
    catalog.validate_score(receipt,arm)
    require(wrapper['version']==VERSION and wrapper['round']==ROUND and wrapper['status']=='complete'
        and wrapper['score']==receipt['score'] and wrapper['dispatch_id']==receipt['dispatch_id']
        and same(wrapper['receipt_hashes'],receipt['receipt_hashes'])
        and all(same(wrapper.get(k),value) for k,value in arm.items())
        and receipt['purpose'] in ('candidate','baseline'),'Primary wrapper/native inputs differ')
    primary=Path(arm['primary_directory'])
    if wrapper['reused_completed_primary']:
        require(not primary.exists() or not any(primary.iterdir()),'Unregistered own primary after reuse')
    else:require(Path(receipt['output_dir']).resolve()==primary.resolve(),'Wrong owned primary path')
    return receipt


def _confirmation_directories(index):
    return {Path(row['directory']) for row in index['reviews']} | {
        path.parent for name in ('review-run.json','dispatch-reservation.json')
        for path in (ROOT/'output').rglob(name)}


def _prior_confirmation(index,arm,primary):
    completed=[]
    for directory in sorted(_confirmation_directories(index)):
        path=directory/'review-run.json'
        if not path.exists():
            preparation=directory/'preparation.json'
            if preparation.exists():
                record=read(preparation)
                if catalog.matches(record,arm):
                    raise ValueError('Exact-target review reservation is incomplete; no confirmation reroll')
            continue
        record=read(path)
        if not catalog.matches(record,arm):continue
        require(record.get('status')=='completed' and record.get('protocol')==er.VERSION,
                'Exact-target prior review failed or remains incomplete')
        receipt=er.validate_receipt(directory);catalog.validate_score(receipt,arm)
        if receipt['purpose']!='confirmation':continue
        require(receipt['dependency']['dispatch_id']==primary['dispatch_id'],
                'Prior confirmation belongs to another primary; do not select a new attempt')
        completed.append(receipt)
    if not completed:return None
    return min(completed,key=lambda item:(item['started_utc'],item['dispatch_id']))


def confirm(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(state(folder)['status']=='complete' and state(folder).get('eligible_for_validation') is True,
                'Confirmation requires the unchanged passing primary and cannot be retried')
        try:
            with counted(folder,'confirmation_preflight'):
                reg,arm=_score_inputs(folder);primary=_primary_receipt(folder,arm)
                require(type(primary['score']) is int and primary['score']>=4,'Primary must reach four')
                directory=folder/'evaluation/confirmation'
                require(not directory.exists() or not any(directory.iterdir()),'Own confirmation reservation already exists')
                prior=_prior_confirmation(read(folder/'score-index-before.json'),arm,primary)
                _save(folder,status='confirmation_reserved',confirmation_reserved_utc=sr.now(),eligible_for_validation=False)
            started=time.monotonic()
            receipt=prior or sr.review_once(arm['manifest'],directory,purpose='confirmation',primary_review=primary['output_dir'])
            with counted(folder,'confirmation_validation'):
                receipt=er.validate_receipt(receipt['output_dir']);catalog.validate_score(receipt,arm)
                require(receipt['purpose']=='confirmation' and receipt['dispatch_id']!=primary['dispatch_id']
                    and receipt['dependency']['dispatch_id']==primary['dispatch_id']
                    and same(receipt['inputs'],primary['inputs']),'Confirmation is not independent on identical inputs')
                passed=receipt['score']>=4
                result={**arm,'version':VERSION,'round':ROUND,'status':'complete','primary':primary['output_dir'],
                    'primary_score':primary['score'],'confirmation':receipt['output_dir'],'confirmation_score':receipt['score'],
                    'dispatch_id':receipt['dispatch_id'],'receipt_hashes':receipt['receipt_hashes'],
                    'reused_completed_confirmation':prior is not None,'runner_score_seconds':time.monotonic()-started,
                    'benchmark_confirmed':passed,'qualified':False,'release_ready':False,'source_accuracy_verified':False}
                immutable(folder/'confirmation.json',result)
            return _save(folder,status='benchmark_confirmed' if passed else 'confirmation_below_target',
                confirmation_score=receipt['score'],confirmed=passed,benchmark_confirmed=passed,
                eligible_for_validation=passed,qualified=False,finished_utc=sr.now())
        except BaseException as exc:_failed(folder,exc);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true');modes.add_argument('--execute-local',action='store_true');modes.add_argument('--score',action='store_true');modes.add_argument('--confirm',action='store_true')
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    try:
        result=prepare() if args.prepare else execute_local() if args.execute_local else score() if args.score else confirm()
        LOG.info('LC-G1 status=%s score=%s qualified=false',result.get('status'),result.get('score'));return 0
    except Exception as exc:LOG.error('LC-G1 stopped: %s; artifacts retained',type(exc).__name__);return 1
if __name__=='__main__':raise SystemExit(main())
