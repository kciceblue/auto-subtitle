"""One blind Chinese-to-Japanese probe and one full-evidence repair; optional scoring."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts import greedy_source_screen as common
from src import compact_raw_evidence as bcodec
from src import blind_backtranslation_requests as contract
from src import local_backend as lb
from src import evidence_context as ec
from src import local_gemma_native as native
from src import selected_pipeline as selected
from src.translate import SrtBlock, parse_srt
from src.workflow_state import file_hash, fingerprint, write_json

VERSION, ROUND = 'blind-backtranslation-screen-1', 'BT-G1'
FOLDER = ROOT/'output/quality-blind-backtranslation-20260916'
BASELINE = ROOT/'output/quality-long-context-gemma-20260915/draft.json'
BASELINE_FILE_SHA256 = '922c497c0a6f44fccaefb578eabf6ab1e314ea266c100b3a8cc4fdcf0b753fcc'
BASELINE_FINGERPRINT = '13f09b26c6e98bbf0cdb7d52ea2c9c6dd0f1cb9a0dcf894592468e710dee89b4'
INHERITED_SECONDS = 190.883754286
B_PARENT = common.B_PARENT
B_REQUEST_SHA256 = common.B_REQUEST_SHA256
SOURCE, CONTEXT, POOL = common.SOURCE, common.CONTEXT, common.POOL
LIMIT, WORK_LIMIT, CLEANUP_LIMIT = 3600, 3180, 420
PLAN = ROOT/'design/quality-blind-backtranslation-two-call-20260916.md'
PROFILE = ROOT/'profiles/long-context-gemma.json'
KEYS=('blind-probe','evidence-repair')
SETTINGS = native.GemmaSettings(maximum_seconds=WORK_LIMIT)
require, same, now = native.require, native.same, native.now
# Explicit shared utilities; no mutation of the thinking screen's policy or globals.
read, state, save = common.read, common.state, common.save
immutable, locked, counted = common.immutable, common.locked, common.counted
Tracker, rows_hash = common.Tracker, common.rows_hash
backend_state, pid_gone, scoring = common.backend_state, common.pid_gone, common.scoring
COPIES=('source.json','original-context.txt','B-request.json','baseline.json','probe-request.json','probe-native-request.json')


def recipe():
    value = selected.load_writer_recipe(PROFILE)
    require(value['context_size'] == SETTINGS.context_size and value['full_swa'] is False
        and value['gpu_layers'] == 99 and value['cpu_moe_layers'] == 0 and value['threads'] == 4
        and value['sha256'] == SETTINGS.model_sha256, 'Declared Gemma deployment differs')
    return value


def prepared(request,stage):
    require(stage in ('probe','repair'),'Unknown stage')
    validator=contract.validate_probe_request if stage=='probe' else contract.validate_repair_request
    version=contract.PROBE_VERSION if stage=='probe' else contract.REPAIR_VERSION
    return native.prepare_request(request,SETTINGS,contract_id=version,validate_request=validator)


def producers():
    names = ('blind_backtranslation_requests','auto_source_draft_requests','compact_raw_draft_requests','compact_raw_evidence',
        'raw_evidence_draft_requests','readable_draft_requests','readable_evidence','episode_draft_requests','temporal_evidence',
        'evidence_context','contextual_review_contract','coherence','coherence_workflow','quality','local_gemma_native',
        'local_backend','selected_pipeline','late_audio','translate','config','workflow_state')
    return [PLAN,PROFILE,Path(__file__).resolve(),Path(common.__file__).resolve(),
            *[ROOT/'src'/(name+'.py') for name in names]]


def inputs(track):
    request = track(B_PARENT/'request.json')
    require(fingerprint(request) == B_REQUEST_SHA256,'Fixed B request changed')
    source = track(B_PARENT/'source.json'); context = track(B_PARENT/'original-context.txt',raw=True)
    require(rows_hash(source) == SOURCE and hashlib.sha256(context).hexdigest() == CONTEXT,'Original source/context changed')
    decoded = bcodec.decode_body(request['body'])
    require(same(decoded['source_evidence']['source_rows'],source)
        and decoded['source_evidence']['original_context'] == context.decode('utf-8'),'B source/context differs')
    baseline=track(BASELINE)
    require(file_hash(BASELINE)==BASELINE_FILE_SHA256 and fingerprint(baseline)==BASELINE_FINGERPRINT,
        'Exact predeclared LC baseline changed')
    require([(r['index'],r['ts_line']) for r in baseline]==[(r['index'],r['ts_line']) for r in source],
        'Baseline owner geometry differs')
    return request,source,context,baseline


def initial_requests(b,baseline):
    # No source/body/context is passed to the blind probe builder.
    request=contract.build_probe_request(baseline)
    coverage=common.contract.validate_writer_request(b)
    require(coverage['observations']==799 and len(baseline)==66,'All799 observations and66 baseline owners required')
    return request,prepared(request,'probe'),coverage


def validate_registration(folder):
    reg=read(folder/'registration.json')
    require(reg['version']==VERSION and reg['round']==ROUND and same(reg['settings'],asdict(SETTINGS))
        and same(reg['limits'],[LIMIT,WORK_LIMIT,CLEANUP_LIMIT]) and reg['maximum_calls']==2
        and reg['inherited_baseline_seconds']==INHERITED_SECONDS
        and reg['registration_sha256']==fingerprint({k:v for k,v in reg.items() if k!='registration_sha256'}),'Registration changed')
    required={str(p.resolve()) for p in producers()}|{str((folder/name).resolve()) for name in COPIES}|{str(BASELINE.resolve())}
    require(required<=set(reg['pins']),'Required direct input/producer pin missing')
    for path,digest in reg['pins'].items():require(file_hash(Path(path))==digest,'Input/producer changed')
    track=Tracker();b,source,context,baseline=inputs(track);request,envelope,coverage=initial_requests(b,baseline)
    expected={'B-request.json':b,'source.json':source,'baseline.json':baseline,'probe-request.json':request,'probe-native-request.json':envelope}
    require(all(reg['pins'].get(p)==h for p,h in track.pins.items()) and same(reg['coverage'],coverage)
        and all(same(read(folder/name),value) for name,value in expected.items())
        and (folder/'original-context.txt').read_bytes()==context,'Copied direct inputs changed')
    d=recipe();require(same(d,reg['deployment']) and reg['pins'].get(d['server_binary'])==file_hash(Path(d['server_binary'])),'Deployment changed')
    return reg


def prepare(folder=FOLDER):
    """Register local inputs only; no score, reviewer or historical native dependency."""
    folder=Path(folder).resolve()
    with locked(folder):
        require(set(folder.iterdir())=={folder/'.lock'},'Already initialized; no restart')
        write_json(folder/'screen.json',{'version':VERSION,'round':ROUND,'status':'preparing','local_seconds':0.,
            'work_seconds':0.,'cleanup_seconds':0.,'stages':[],'original_backend_restored':True,'qualified':False,'score':None,
            'calls':[],'calls_started':0,'calls_completed':0,'inherited_baseline_seconds':INHERITED_SECONDS})
        try:
            with counted(folder,'local_input_preparation'):
                track=Tracker();b,source,context,baseline=inputs(track);request,envelope,coverage=initial_requests(b,baseline);d=recipe()
                for name,value in [('B-request.json',b),('source.json',source),('baseline.json',baseline),
                                   ('probe-request.json',request),('probe-native-request.json',envelope)]:immutable(folder/name,value)
                immutable(folder/'original-context.txt',context,raw=True)
                for path in [*producers(),Path(d['server_binary']),*[folder/name for name in COPIES]]:track(path,raw=True)
                reg={'version':VERSION,'round':ROUND,'settings':asdict(SETTINGS),'limits':[LIMIT,WORK_LIMIT,CLEANUP_LIMIT],
                    'maximum_calls':2,'inherited_baseline_seconds':INHERITED_SECONDS,'coverage':coverage,'deployment':d,'pins':track.pins}
                reg['registration_sha256']=fingerprint(reg);immutable(folder/'registration.json',reg)
                calls=[{'key':key,'status':'undispatched','native_request_sha256':fingerprint(envelope) if i==0 else None,
                    'started_utc':None,'finished_utc':None,'response_sha256':None,'error_type':None} for i,key in enumerate(KEYS)]
            return save(folder,status='prepared',registration_sha256=reg['registration_sha256'],calls=calls)
        except BaseException as exc:save(folder,status='failed',error_type=type(exc).__name__);raise


start_writer,stop_writer=common.start_writer,common.stop_writer


def run_writer(folder,reg):
    owned=None;identity=failure=None;responses=[]
    try:
        with counted(folder,'blind_probe_and_evidence_repair') as deadline:
            d=reg['deployment'];previous=backend_state(d);owned={'previous':previous};save(folder,original_backend_restored=False)
            endpoint,identity=start_writer(folder,reg,owned);immutable(folder/'writer-backend/backend.json',identity)
            require(endpoint==native.ENDPOINT and identity['model_path']==d['weights']
                and identity['server_binary_sha256']==reg['pins'][d['server_binary']],'Owned Gemma runtime changed')
            calls=state(folder)['calls'];require(len(calls)==2 and all(c['status']=='undispatched' for c in calls),'Calls already attempted')
            b=read(folder/'B-request.json');baseline=read(folder/'baseline.json');probe_request=read(folder/'probe-request.json')
            for index,(stage,key) in enumerate(zip(('probe','repair'),KEYS)):
                if stage=='probe':
                    request=probe_request;envelope=read(folder/'probe-native-request.json')
                    require(calls[index]['native_request_sha256']==fingerprint(envelope),'Probe state binding changed')
                else:
                    request=contract.build_repair_request(b,baseline,responses[0])
                    contract.validate_extension(request,b,baseline,probe_request,responses[0])
                    envelope=prepared(request,'repair')
                    immutable(folder/'repair-request.json',request);immutable(folder/'repair-native-request.json',envelope)
                require(calls[index]['key']==key,'Stage key changed')
                calls[index].update(status='running',started_utc=now(),native_request_sha256=fingerprint(envelope))
                save(folder,calls=calls,calls_started=index+1)
                try:
                    value=native.ask(folder/'requests',key,envelope,identity,deadline)
                    if stage=='probe':contract.validate_probe_response(value,request)
                    else:contract.validate_repair_response(value)
                    immutable(folder/(stage+'-response.json'),value);responses.append(value)
                    calls[index].update(status='complete',finished_utc=now(),response_sha256=fingerprint(value))
                    save(folder,calls=calls,calls_completed=index+1)
                except BaseException as exc:
                    calls[index].update(status='failed',finished_utc=now(),error_type=type(exc).__name__)
                    save(folder,calls=calls);raise
            require(all(lb._file_identity(path)==old for path,old in owned.get('model_stats',{}).items()),'Model changed during generation')
    except BaseException as exc:failure=exc
    finally:
        if owned is not None:
            restored=closed=False;error=None
            try:
                with counted(folder,'writer_owned_cleanup',cleanup=True):
                    stop_writer(reg,owned);closed=restored=True
                    require(identity is None or pid_gone(identity['pid']),'Owned Gemma process remains')
            except BaseException as exc:
                error=type(exc).__name__
                if failure is None:failure=exc
            finally:
                immutable(folder/'writer-lifecycle.json',{'original_backend':owned['previous'],'original_backend_restored':restored,
                    'context_closed':closed,'error_type':error});save(folder,original_backend_restored=restored)
    if failure is not None:raise failure
    return responses[1]


def replay_local(folder,reg,track):
    source=track(folder/'source.json');b=track(folder/'B-request.json');baseline=track(folder/'baseline.json')
    probe_request=track(folder/'probe-request.json');require(same(probe_request,contract.build_probe_request(baseline)),'Blind input changed')
    identity=track(folder/'writer-backend/backend.json')
    require(identity['model_path']==reg['deployment']['weights'] and identity['server_binary_sha256']==reg['pins'][reg['deployment']['server_binary']],
        'Runtime pin changed')
    calls=state(folder)['calls'];require(len(calls)==2 and state(folder)['calls_started']==state(folder)['calls_completed']==2,'Incomplete call coverage')
    responses=[]
    for index,(stage,key) in enumerate(zip(('probe','repair'),KEYS)):
        request=probe_request if index==0 else contract.build_repair_request(b,baseline,responses[0])
        if index:contract.validate_extension(request,b,baseline,probe_request,responses[0])
        require(same(track(folder/(stage+'-request.json')),request),'Dynamic stage request changed')
        envelope=track(folder/(stage+'-native-request.json'));require(same(envelope,prepared(request,stage)),'Native stage request changed')
        value=native.replay_native(folder/'requests'/(key+'.json'),envelope,identity,track)
        if index==0:contract.validate_probe_response(value,request)
        else:contract.validate_repair_response(value)
        require(same(value,track(folder/(stage+'-response.json'))),'Saved native response changed')
        call=calls[index]
        require(call['key']==key and call['status']=='complete' and call['error_type'] is None
            and call['native_request_sha256']==fingerprint(envelope) and call['response_sha256']==fingerprint(value),
            'Stage state/native output changed')
        responses.append(value)
    value=responses[1];require(same(value,track(folder/'translation.json')),'Saved final target changed')
    draft=[{**r,'text':value['owners'][str(r['index'])]['chinese']} for r in source]
    for name,rows in [('source',source),('draft',draft)]:
        require(same(track(folder/(name+'.json')),rows)
            and parse_srt(folder/(name+'.utterances.srt'),preserve_text_whitespace=True)==[SrtBlock(**r) for r in rows],'Exact owner/SRT changed')
        track(folder/(name+'.utterances.srt'),raw=True)
    life=track(folder/'writer-lifecycle.json')
    require(life['original_backend_restored'] is True and life['context_closed'] is True and life['error_type'] is None,'Owned cleanup failed')
    require({p.name for p in (folder/'requests').iterdir()}=={key+suffix for key in KEYS for suffix in ('.json','.sse')},'Unexpected attempt artifacts')
    return source,draft


def execute_local(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        require(state(folder)['status']=='prepared','Two-call candidate cannot resume or reroll')
        try:
            with counted(folder,'execution_input_replay'):
                reg=validate_registration(folder);require(state(folder)['registration_sha256']==reg['registration_sha256'],'State binding changed')
                save(folder,status='running');source=read(folder/'source.json')
                selected.write_checked([SrtBlock(**r) for r in source],folder/'source.utterances.srt')
            value=run_writer(folder,reg)
            with counted(folder,'complete_candidate_and_replay'):
                contract.validate_repair_response(value);immutable(folder/'translation.json',value)
                draft=[{**r,'text':value['owners'][str(r['index'])]['chinese']} for r in source]
                immutable(folder/'draft.json',draft);selected.write_checked([SrtBlock(**r) for r in draft],folder/'draft.utterances.srt')
                track=Tracker();source,draft=replay_local(folder,reg,track)
                outcome='unchanged_baseline' if same(draft,read(folder/'baseline.json')) else 'repaired'
                immutable(folder/'completion.json',{'version':VERSION,'source_sha256':rows_hash(source),'target_sha256':rows_hash(draft),
                    'native_artifacts':track.pins,'probe_calls':1,'repair_calls':1,'outcome':outcome,
                    'inherited_baseline_seconds':INHERITED_SECONDS,'treatment_adherent':True,'source_accuracy_verified':False})
            for name in ('source.utterances.srt','draft.utterances.srt'):(folder/name).chmod(0o444)
            return save(folder,status='local_complete',finished_utc=now(),treatment_adherent=True,outcome=outcome,
                probe_calls=1,repair_calls=1,baseline_plus_candidate_local_seconds=INHERITED_SECONDS+state(folder)['local_seconds'])
        except BaseException as exc:
            save(folder,status='treatment_failed' if isinstance(exc,native.TreatmentAdherenceError) else 'failed',
                error_type=type(exc).__name__,treatment_adherent=False);raise


def review_inputs(folder):
    catalog, helpers, _ = scoring(); reg = validate_registration(folder); track = Tracker()
    source, draft = replay_local(folder, reg, track); completed = read(folder/'completion.json')
    require(completed['treatment_adherent'] is True and completed['target_sha256'] == rows_hash(draft)
        and same(completed['native_artifacts'], track.pins), 'Immutable completion changed')
    breg = track(B_PARENT/'registration.json'); baseline = helpers._baseline(track, breg, [SrtBlock(**r) for r in source], (folder/'original-context.txt').read_bytes().decode('utf-8'))
    manifest = folder/'evaluation/bundle/manifest.json'
    if not manifest.exists(): helpers.er.prepare_bundle(source, draft, source, list(map(Path, breg['evidence_paths'])), folder/'original-context.txt', manifest.parent)
    arm = {'label':ROUND, 'pool_version':POOL, 'source_sha256':SOURCE, 'context_sha256':CONTEXT,
        'target_sha256':rows_hash(draft), 'manifest':str(manifest), 'primary_directory':str(folder/'evaluation/primary')}
    bundle = helpers.er.read_bundle(manifest); catalog.validate_score(bundle, arm)
    require(same(bundle['rows']['source'], source) and same(bundle['rows']['raw'], source) and same(bundle['rows']['target'], draft), 'Scoring bundle differs')
    index = catalog.load_score_index(track)
    additions = [*helpers.EXTRA_SCORES, *[ROOT/'output'/name/'score.json' for name in (
        'quality-staged-source-20260915', 'quality-long-context-gemma-20260915',
        'quality-auto-source-continuation-20260916', 'quality-forced-source-draft-20260916',
        'quality-gemma-thinking-20260916', 'quality-compact-source-20260916',
        'quality-masked-source-draft-continuation-20260916', 'quality-long-source-draft-20260916',
        'quality-voxtral-source-draft-20260916', 'quality-greedy-source-20260916',
        'quality-full-evidence-critic-20260916', 'quality-filtered-local-critic-20260916',
        'quality-vad-ctc-source-draft-recovery-20260916', 'quality-focused-owner-20260916',
        'quality-raw-ctc-admission-20260916')]]
    for path in additions:
        if path.exists():
            value = track(path); index['records'].append({'path':str(path), 'metadata_sha256':file_hash(path),
                'hashes':{k:value.get(k) for k in ('pool_version','target_sha256','source_sha256')}})
    return catalog, helpers, arm, baseline, index, track


def review(folder=FOLDER, *, confirmation=False):
    folder = Path(folder).resolve(); phase = 'confirmation' if confirmation else 'primary'
    with locked(folder):
        require(state(folder)['status'] == 'local_complete', 'An immutable local completion is required')
        ledger = folder/(phase+'-review.json'); require(not ledger.exists(), 'Review cannot resume or reroll')
        write_json(ledger, {'status':'preparing', 'started_utc':now()})
        try:
            with counted(folder, phase+'_preflight'):
                catalog, helpers, arm, baseline, index, track = review_inputs(folder)
                directory = folder/'evaluation'/phase
                require(not directory.exists() or not any(directory.iterdir()), 'Own review reservation already exists')
                primary = None
                if confirmation:
                    wrapper = read(folder/'score.json'); primary = helpers.er.validate_receipt(wrapper['primary']); catalog.validate_score(primary, arm)
                    require(primary['score'] >= 4 and wrapper['score'] == primary['score'] and wrapper['dispatch_id'] == primary['dispatch_id'], 'Passing exact primary required')
                    prior = scoring()[2]._prior_confirmation(index, arm, primary)
                else: prior = catalog.duplicates(index, arm, folder)
                immutable(folder/(phase+'-inputs.json'), {'arm':arm, 'pins':track.pins, 'prior_dispatch_id':None if prior is None else prior['dispatch_id']})
                write_json(ledger, {'status':'reserved', 'started_utc':now()})
            started = time.monotonic()
            receipt = prior or helpers.sr.review_once(arm['manifest'], directory, purpose='confirmation' if confirmation else 'candidate',
                **({'primary_review':primary['output_dir']} if confirmation else {'baseline_review':baseline['output_dir']}))
            with counted(folder, phase+'_validation'):
                receipt = helpers.er.validate_receipt(receipt['output_dir']); catalog.validate_score(receipt, arm)
                require(receipt['purpose'] == 'confirmation' if confirmation else receipt['purpose'] in ('candidate','baseline'), 'Wrong review purpose')
                if confirmation:
                    require(receipt['dispatch_id'] != primary['dispatch_id'] and receipt['dependency']['dispatch_id'] == primary['dispatch_id']
                        and same(receipt['inputs'], primary['inputs']), 'Confirmation is not fresh on identical inputs')
                result = {**arm, 'version':VERSION, 'round':ROUND, 'status':'complete', 'runner_score_seconds':time.monotonic()-started, 'score':receipt['score'], 'dispatch_id':receipt['dispatch_id'],
                    'receipt_hashes':receipt['receipt_hashes'], 'reused_completed_review':prior is not None,
                    'primary':receipt['output_dir'] if not confirmation else primary['output_dir'],
                    'review_directory':receipt['output_dir'], 'qualified':False, 'benchmark_confirmed':confirmation and receipt['score'] >= 4}
                immutable(folder/('confirmation.json' if confirmation else 'score.json'), result)
                write_json(ledger, {'status':'complete', 'finished_utc':now(), 'score':receipt['score']})
            return result
        except BaseException as exc:
            write_json(ledger, {'status':'failed', 'error_type':type(exc).__name__, 'finished_utc':now()}); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_mutually_exclusive_group(required=True)
    for flag in ('prepare','execute-local','score','confirm'): modes.add_argument('--'+flag,action='store_true')
    args = parser.parse_args(); logging.basicConfig(level=logging.INFO)
    result = prepare() if args.prepare else execute_local() if args.execute_local else review(confirmation=args.confirm)
    logging.getLogger(__name__).info('Blind backtranslation status=%s score=%s qualified=false',result.get('status'),result.get('score'))


if __name__ == '__main__': main()
