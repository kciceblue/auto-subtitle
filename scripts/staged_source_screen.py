"""SS1: one local Japanese reconstruction, one Chinese draft, separate primary score."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import logging
import math
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import full_draft_diagnostic as catalog
from scripts.raw_evidence_screen import Tracker
from src import compact_raw_draft_requests as brequests
from src import compact_raw_evidence as codec
from src import contextual_evidence_review as er
from src import late_audio, local_backend
from src import revisit_workflow as rw
from src import sparse_revisit as sr
from src import staged_source_native as native
from src import staged_source_requests as requests
from src.contextual_review_contract import require
from src.translate import SrtBlock, parse_srt
from src.workflow_state import file_hash,fingerprint,write_json

VERSION='staged-source-screen-1'
ROUND='SS1'
FOLDER=ROOT/'output/quality-staged-source-20260915'
B_PARENT=ROOT/'output/quality-compact-raw-20260915'
B_REQUEST_SHA256='e9815cde9b9e20523c9e0d92b643da945081fe4efd8db7ba6413e5578c8e2d29'
PLAN=ROOT/'design/quality-staged-source-20260915.md'
PROFILE=ROOT/'profiles/selected-writer.json'
LIMIT,WORK_LIMIT,CLEANUP_LIMIT=2400,1980,420
STAGES=(('qwen','reconstruction'),('gemma','translation'))
EXTRA_SCORES=[ROOT/'output'/name/'score.json' for name in ('quality-raw-evidence-20260915',
    'quality-compact-raw-20260915','quality-reasoned-raw-20260915','quality-large-compact-20260915')]+[
    ROOT/'output/quality-readable-draft-20260915/R1/score.json',
    *[ROOT/'output/quality-full-draft-diagnostic-20260915'/label/'score.json' for label in ('DF-L1','DF-R1')]]
read,immutable,same,locked=catalog.read,catalog.immutable,catalog.same,catalog.locked
LOG=logging.getLogger(__name__)


def state(folder):return read(folder/'screen.json')
def _save(folder,**changes):
    value=state(folder);value.update(changes);write_json(folder/'screen.json',value);return value

def _failed(folder,error):
    _save(folder,status='failed',error_type=type(error).__name__,qualified=False,confirmed=False,
          eligible_for_validation=False,finished_utc=sr.now())


@contextmanager
def cleanup_counted(folder,name):
    # Closing an owned backend is unconditional, including after budget expiry.
    start=time.monotonic();mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM})
    timer=signal.getitimer(signal.ITIMER_REAL);signal.setitimer(signal.ITIMER_REAL,0);error=None
    try:yield
    except BaseException as exc:error=type(exc).__name__;raise
    finally:
        elapsed=time.monotonic()-start;over=False
        try:
            s=state(folder);s['local_seconds']+=elapsed;s['cleanup_seconds']+=elapsed
            over=s['local_seconds']>=LIMIT or s['cleanup_seconds']>=CLEANUP_LIMIT or bool(timer[0] and elapsed>=timer[0])
            s['stages'].append({'stage':name,'seconds':elapsed,'cleanup':True,
                                'error_type':error or ('LocalBudgetExceeded' if over else None)})
            write_json(folder/'screen.json',s)
        finally:
            while signal.SIGALRM in signal.sigpending():
                signal.sigtimedwait({signal.SIGALRM},0);over=True
            if timer[0]>elapsed:signal.setitimer(signal.ITIMER_REAL,timer[0]-elapsed,timer[1])
            signal.pthread_sigmask(signal.SIG_SETMASK,mask)
        if over and error is None:raise rw.LocalBudgetExceeded('SS1 cleanup exceeded allowance after restoration')


@contextmanager
def counted(folder,name,*,cleanup=False):
    if cleanup:
        with cleanup_counted(folder,name):yield None
        return
    start=time.monotonic();s=state(folder)
    for key in ('local_seconds','work_seconds','cleanup_seconds'):
        require(type(s[key]) in (int,float) and math.isfinite(s[key]) and s[key]>=0,'Invalid local time ledger')
    remaining=min(LIMIT-s['local_seconds'],(CLEANUP_LIMIT-s['cleanup_seconds']) if cleanup else WORK_LIMIT-s['work_seconds'])
    handler=signal.getsignal(signal.SIGALRM);timer=signal.getitimer(signal.ITIMER_REAL)
    if timer[0]:remaining=min(remaining,timer[0])
    armed=False;error=None
    def expired(*_):raise rw.LocalBudgetExceeded('SS1 local deadline')
    try:
        if remaining<=0:raise rw.LocalBudgetExceeded('SS1 allowance exhausted')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining);armed=True
        yield start+remaining
    except BaseException as exc:error=type(exc).__name__;raise
    finally:
        elapsed=time.monotonic()-start
        if armed:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
            if timer[0]:signal.setitimer(signal.ITIMER_REAL,max(.001,timer[0]-elapsed),timer[1])
        s=state(folder);s['local_seconds']+=elapsed;s['cleanup_seconds' if cleanup else 'work_seconds']+=elapsed
        over=elapsed>=remaining or s['local_seconds']>=LIMIT or s['work_seconds']>=WORK_LIMIT or s['cleanup_seconds']>=CLEANUP_LIMIT
        s['stages'].append({'stage':name,'seconds':elapsed,'cleanup':cleanup,'error_type':error or ('LocalBudgetExceeded' if over else None)})
        write_json(folder/'screen.json',s)
        if over and error is None:raise rw.LocalBudgetExceeded('SS1 stage exceeded allowance')


def producers():
    return [PLAN,PROFILE,Path(__file__).resolve(),Path(sys.modules[Tracker.__module__].__file__),
        *[Path(module.__file__) for module in (requests,native,brequests,codec,catalog,er,rw,sr,late_audio,local_backend)],
        ROOT/'src/translate.py',ROOT/'src/config.py',ROOT/'src/selected_pipeline.py',
        ROOT/'src/contextual_review_contract.py',ROOT/'src/workflow_state.py']


def _baseline(track,breg,source,context):
    baseline=er.validate_receipt(breg['baseline_review'])
    require(baseline['purpose']=='baseline' and baseline['score']==2 and baseline['pool_version']==catalog.POOL,
            'Current genuine baseline must be two')
    arm={'pool_version':catalog.POOL,'source_sha256':catalog.SOURCE,'context_sha256':catalog.CONTEXT,
         'target_sha256':baseline['inputs']['target']['sha256']};catalog.validate_score(baseline,arm)
    for name,digest in baseline['receipt_hashes'].items():
        require(Path(name).name==name,'Baseline receipt escaped its directory')
        path=Path(baseline['output_dir'])/name;track(path,raw=True)
        require(track.pins[str(path.resolve())]==digest,'Baseline receipt changed')
    pools=[track(Path(path)) for path in breg['evidence_paths']]
    _,view=er._evidence(pools,[asdict(row) for row in source])
    require(er.sha256(er._bytes(er._pool([asdict(row) for row in source],view,context)))==catalog.POOL,
            'Consumed evidence does not reproduce the unchanged pool')
    return baseline


def _inputs(track):
    breg=track(B_PARENT/'registration.json');brequest=track(B_PARENT/'request.json')
    require(fingerprint(brequest)==B_REQUEST_SHA256,'Declared B request changed')
    coverage=brequests.validate_writer_request(brequest);body=codec.decode_body(brequest['body'])
    require(same(codec.encode_body(body),brequest['body']),'B codec round trip changed')
    source=[SrtBlock(**row) for row in track(B_PARENT/'source.json')]
    context_bytes=track(B_PARENT/'original-context.txt',raw=True);context=context_bytes.decode('utf-8',errors='strict')
    require(er.owner_rows_sha256(source)==catalog.SOURCE and hashlib.sha256(context_bytes).hexdigest()==catalog.CONTEXT,
            'Original source/context identity changed')
    request=requests.build_reconstruction_request(source,body,context,audio_interpretations=[])
    baseline=_baseline(track,breg,source,context)
    return breg,brequest,source,context_bytes,request,coverage,baseline


def validate_registration(folder):
    reg=read(folder/'registration.json')
    fixed={'version':VERSION,'round':ROUND,'maximum_local_seconds':LIMIT,'maximum_work_seconds':WORK_LIMIT,
        'maximum_cleanup_seconds':CLEANUP_LIMIT,'maximum_reconstruction_calls':1,'maximum_translation_calls':1,
        'maximum_primary_dispatches':1,'maximum_confirmations':1,'audio_interpretations':[],
        'source_sha256':catalog.SOURCE,'context_sha256':catalog.CONTEXT,'pool_version':catalog.POOL,'qualified':False}
    require(same({k:reg.get(k) for k in fixed},fixed) and reg['registration_sha256']==fingerprint({k:v for k,v in reg.items() if k!='registration_sha256'}),'SS1 registration changed')
    required={str(p.resolve()) for p in producers()}|{str(folder/name) for name in
        ('source.json','original-context.txt','B-request.json','reconstruction-request.json','reconstruction-native-request.json','score-index-before.json')}
    require(required<=set(reg['pins']),'Mandatory SS1 input/producer pin missing')
    for path,digest in reg['pins'].items():require(file_hash(Path(path))==digest,'SS1 input/producer changed')
    track=Tracker();breg,brequest,source,context,request,coverage,baseline=_inputs(track)
    require(all(reg['pins'].get(path)==digest for path,digest in track.pins.items()),'Direct input pin coverage changed')
    require(same(read(folder/'B-request.json'),brequest) and same(read(folder/'source.json'),[asdict(row) for row in source])
        and (folder/'original-context.txt').read_bytes()==context and same(read(folder/'reconstruction-request.json'),request)
        and same(read(folder/'reconstruction-native-request.json'),native.prepare_request(request,'qwen'))
        and same(reg['coverage'],coverage) and reg['baseline_review']==baseline['output_dir']
        and same(reg['evidence_paths'],breg['evidence_paths']),'Registered direct source request changed')
    return reg


def prepare(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(set(folder.iterdir())=={folder/'.lock'},'SS1 already initialized; no restart')
        write_json(folder/'screen.json',{'version':VERSION,'round':ROUND,'status':'preparing','local_seconds':0.,
            'work_seconds':0.,'cleanup_seconds':0.,'stages':[],'original_backend_restored':True,'qualified':False,'confirmed':False})
        try:
            with counted(folder,'direct_input_preparation'):
                track=Tracker();breg,brequest,source,context,request,coverage,baseline=_inputs(track)
                rw.load_writer_recipe(PROFILE)
                index=catalog.load_score_index(track)
                for path in EXTRA_SCORES:
                    if path.exists():
                        value=track(path);index['records'].append({'path':str(path),'metadata_sha256':track.pins[str(path.resolve())],
                            'hashes':{key:value.get(key) for key in ('pool_version','target_sha256','source_sha256')}})
                for name,value in [('source.json',[asdict(row) for row in source]),('B-request.json',brequest),
                    ('reconstruction-request.json',request),('reconstruction-native-request.json',native.prepare_request(request,'qwen')),
                    ('score-index-before.json',index)]:immutable(folder/name,value)
                immutable(folder/'original-context.txt',context,raw=True)
                for path in [*producers(),*[p for p in folder.rglob('*') if p.is_file() and p.name not in ('.lock','screen.json')]]:track(path,raw=True)
                reg={'version':VERSION,'round':ROUND,'maximum_local_seconds':LIMIT,'maximum_work_seconds':WORK_LIMIT,
                    'maximum_cleanup_seconds':CLEANUP_LIMIT,'maximum_reconstruction_calls':1,'maximum_translation_calls':1,
                    'maximum_primary_dispatches':1,'maximum_confirmations':1,'audio_interpretations':[],
                    'source_sha256':catalog.SOURCE,'context_sha256':catalog.CONTEXT,'pool_version':catalog.POOL,
                    'qualified':False,'coverage':coverage,'baseline_review':baseline['output_dir'],
                    'evidence_paths':breg['evidence_paths'],'pins':track.pins}
                reg['registration_sha256']=fingerprint(reg);immutable(folder/'registration.json',reg)
                validate_registration(folder)
            return _save(folder,status='prepared',registration_sha256=reg['registration_sha256'])
        except BaseException as exc:_failed(folder,exc);raise


def _backend_state():return late_audio._backend_identity(rw._request_json(rw.ADMIN+'/status',admin=True))
def _pid_gone(pid):
    try:os.kill(pid,0)
    except ProcessLookupError:return True
    return False


def _run_stage(folder,family,key,request):
    manager=None;entered=False;identity=None;previous=None;failure=None;value=None
    try:
        with counted(folder,key) as deadline:
            previous=_backend_state();_save(folder,original_backend_restored=False)
            manager=rw.backend(family,folder/(key+'-backend'),PROFILE)
            endpoint,identity=manager.__enter__();entered=True
            value=native.ask(folder/'requests',key,endpoint,request,family,identity,deadline)
    except BaseException as exc:failure=exc
    finally:
        if manager is not None:
            ok=False;error=None;closed=not entered
            try:
                with counted(folder,key+'_cleanup',cleanup=True):
                    notes=tuple(getattr(failure,'__notes__',()))
                    if entered:manager.__exit__(type(failure) if failure else None,failure,failure.__traceback__ if failure else None)
                    require(tuple(getattr(failure,'__notes__',()))==notes,'Owned backend reported cleanup failure')
                    closed=True
                    stopped=family=='qwen' or (entered and identity is not None and _pid_gone(identity['pid']))
                    after=_backend_state();ok=closed and stopped and after==previous
                    require(ok,'Owned backend did not stop/restore exactly')
            except BaseException as exc:
                error=type(exc).__name__
                if failure is None:failure=exc
                else:failure.add_note('SS1 cleanup failed: '+error)
            finally:
                immutable(folder/(key+'-lifecycle.json'),{'original_backend':previous,'original_backend_restored':ok,
                    'context_closed':closed,'error_type':error})
                _save(folder,original_backend_restored=ok)
    if failure is not None:raise failure
    return value


def _compile(response,source,family):
    checked=(requests.validate_reconstruction(response,source) if family=='qwen' else requests.validate_translation(response,source))
    key='japanese' if family=='qwen' else 'chinese'
    return [SrtBlock(row.index,row.ts_line,checked['owners'][str(row.index)][key]) for row in source]


def _save_rows(folder,name,rows):
    immutable(folder/(name+'.json'),[asdict(row) for row in rows]);sr.write_checked(rows,folder/(name+'.utterances.srt'))
    require(parse_srt(folder/(name+'.utterances.srt'),preserve_text_whitespace=True)==rows,'Strict SRT round trip changed')


def replay_local(folder,reg,track):
    source=rw.rows(folder/'source.json');context=(folder/'original-context.txt').read_text(encoding='utf-8')
    first=read(folder/'reconstruction-native-request.json')
    raw=native.replay_native(folder/'requests/reconstruction.json',first,'qwen',track(folder/'reconstruction-backend/backend.json'),track)
    second=native.prepare_request(requests.build_translation_request(source,raw,context),'gemma')
    require(same(second,track(folder/'translation-native-request.json')),'Gemma request changed from exact Japanese intermediate')
    translated=native.replay_native(folder/'requests/translation.json',second,'gemma',track(folder/'translation-backend/backend.json'),track)
    rows_by_name={'source':source,'reconstructed-source':_compile(raw,source,'qwen'),'draft':_compile(translated,source,'gemma')}
    for name,rows in rows_by_name.items():
        require(same([asdict(row) for row in rows],track(folder/(name+'.json')))
            and parse_srt(folder/(name+'.utterances.srt'),preserve_text_whitespace=True)==rows,'Native rows/SRT changed')
        track(folder/(name+'.utterances.srt'),raw=True)
    for key in ('reconstruction','translation'):
        life=track(folder/(key+'-lifecycle.json'));require(life['original_backend_restored'] is True
            and life['context_closed'] is True and life['error_type'] is None,'Stage cleanup was not successful')
    metrics=native.shared._metrics(track(folder/'requests/metrics.jsonl',raw=True))
    require(len(metrics)==2 and [row['stage'] for row in metrics]==['reconstruction','translation']
        and {p.name for p in (folder/'requests').glob('*.json')}=={'reconstruction.json','translation.json'},'Unexpected local generation coverage')
    require(same(track(folder/'reconstruction.json'),raw) and same(track(folder/'translation.json'),translated),'Saved parsed outputs changed')
    return source,rows_by_name['draft']


def _bundle_arm(folder,reg,source,draft,manifest):
    arm={'label':ROUND,'source_sha256':reg['source_sha256'],'context_sha256':reg['context_sha256'],
        'pool_version':reg['pool_version'],'target_sha256':er.owner_rows_sha256(draft),
        'manifest':str(manifest),'primary_directory':str(folder/'evaluation/primary'),
        'compilation':{'source_owners':len(source),'target_owners':len(draft),'local_generation_calls':2,
            'normalization_changes':0,'audio_interpretations':0,'raw_observations':reg['coverage']['observations']}}
    bundle=er.read_bundle(manifest);catalog.validate_score(bundle,arm)
    require(same(bundle['rows']['source'],[asdict(row) for row in source])
        and same(bundle['rows']['raw'],[asdict(row) for row in source])
        and same(bundle['rows']['target'],[asdict(row) for row in draft]),'External bundle changed original source or exact Chinese')
    return arm


def execute_local(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(state(folder)['status']=='prepared','SS1 local attempt cannot resume or reroll')
        try:
            with counted(folder,'execution_input_replay'):
                reg=validate_registration(folder);require(state(folder)['registration_sha256']==reg['registration_sha256'],'State binding changed')
                _save(folder,status='running',started_utc=sr.now());source=rw.rows(folder/'source.json')
                sr.write_checked(source,folder/'source.utterances.srt')
            reconstruction=_run_stage(folder,'qwen','reconstruction',read(folder/'reconstruction-native-request.json'))
            with counted(folder,'intermediate_assembly'):
                immutable(folder/'reconstruction.json',reconstruction);_save_rows(folder,'reconstructed-source',_compile(reconstruction,source,'qwen'))
                second=native.prepare_request(requests.build_translation_request(source,reconstruction,
                    (folder/'original-context.txt').read_text(encoding='utf-8')),'gemma')
                immutable(folder/'translation-native-request.json',second)
            translated=_run_stage(folder,'gemma','translation',second)
            with counted(folder,'native_replay_and_unchanged_pool_bundle'):
                immutable(folder/'translation.json',translated);_save_rows(folder,'draft',_compile(translated,source,'gemma'))
                track=Tracker();source,draft=replay_local(folder,reg,track)
                manifest=er.prepare_bundle(source,draft,source,list(map(Path,reg['evidence_paths'])),folder/'original-context.txt',folder/'evaluation/bundle')
                arm=_bundle_arm(folder,reg,source,draft,manifest)
                for path in (folder/'evaluation/bundle').rglob('*'):
                    if path.is_file():track(path,raw=True)
                immutable(folder/'score-inputs.json',{**arm,'native_artifacts':track.pins})
            return _save(folder,status='local_complete',finished_local_utc=sr.now(),eligible_for_validation=False)
        except BaseException as exc:_failed(folder,exc);raise


def score(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(state(folder)['status']=='local_complete','Reserved, completed or failed SS1 score cannot retry')
        try:
            with counted(folder,'score_preflight'):
                reg=validate_registration(folder);require(state(folder)['original_backend_restored'] is True,'Backend restoration missing')
                track=Tracker();source,draft=replay_local(folder,reg,track);saved=read(folder/'score-inputs.json')
                arm=_bundle_arm(folder,reg,source,draft,folder/'evaluation/bundle/manifest.json')
                for path in (folder/'evaluation/bundle').rglob('*'):
                    if path.is_file():track(path,raw=True)
                require(same(saved,{**arm,'native_artifacts':track.pins}),'Scoring artifacts changed')
                primary=Path(arm['primary_directory']);require(not primary.exists() or not any(primary.iterdir()),'Own primary reservation cannot be retried')
                prior=catalog.duplicates(read(folder/'score-index-before.json'),arm,folder)
                _save(folder,status='review_reserved',review_reserved_utc=sr.now())
            started=time.monotonic()
            receipt=prior or sr.review_once(arm['manifest'],Path(arm['primary_directory']),purpose='candidate',baseline_review=reg['baseline_review'])
            with counted(folder,'primary_receipt_validation'):
                receipt=er.validate_receipt(receipt['output_dir']);catalog.validate_score(receipt,arm)
                require(receipt['purpose'] in ('candidate','baseline'),'Primary receipt required')
                result={**arm,'version':VERSION,'round':ROUND,'status':'complete','score':receipt['score'],
                    'primary':receipt['output_dir'],'dispatch_id':receipt['dispatch_id'],'receipt_hashes':receipt['receipt_hashes'],
                    'reused_completed_primary':prior is not None,'runner_score_seconds':time.monotonic()-started,
                    'qualified':False,'confirmed':False,'eligible_for_validation':receipt['score']>=4}
                immutable(folder/'score.json',result)
            return _save(folder,status='complete',score=result['score'],eligible_for_validation=result['eligible_for_validation'],finished_utc=sr.now())
        except BaseException as exc:_failed(folder,exc);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true');modes.add_argument('--execute-local',action='store_true');modes.add_argument('--score',action='store_true')
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    try:
        result=prepare() if args.prepare else execute_local() if args.execute_local else score()
        LOG.info('SS1 status=%s score=%s qualified=false',result.get('status'),result.get('score'));return 0
    except Exception as exc:LOG.error('SS1 stopped: %s; artifacts retained',type(exc).__name__);return 1
if __name__=='__main__':raise SystemExit(main())
