"""One local draft over exact B plus mask-corrected ASR; optional final scoring."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import logging
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts import gemma_thinking_screen as common
from src import compact_raw_evidence as bcodec
from src import masked_source_draft_requests as contract
from src import local_backend as lb
from src import local_gemma_native as native
from src import selected_pipeline as selected
from src.translate import SrtBlock, parse_srt
from src.workflow_state import file_hash, fingerprint, write_json

VERSION, ROUND = 'masked-source-screen-1', 'AM-G1'
FOLDER = ROOT/'output/quality-masked-source-draft-20260916'
ACQUISITION_PARENT = ROOT/'output/quality-masked-source-20260916'
B_PARENT = common.B_PARENT
B_REQUEST_SHA256 = common.B_REQUEST_SHA256
SOURCE, CONTEXT, POOL = common.SOURCE, common.CONTEXT, common.POOL
LIMIT, WORK_LIMIT, CLEANUP_LIMIT = 3600, 3180, 420
PLAN = ROOT/'design/quality-attention-repair-plan-20260916.md'
PROFILE = ROOT/'profiles/long-context-gemma.json'
KEY = 'masked-source-draft'
SETTINGS = native.GemmaSettings(maximum_seconds=WORK_LIMIT)
require, same, now = native.require, native.same, native.now
# Explicit shared utilities; no mutation of the thinking screen's policy or globals.
read, state, save = common.read, common.state, common.save
immutable, locked, counted = common.immutable, common.locked, common.counted
Tracker, rows_hash = common.Tracker, common.rows_hash
backend_state, pid_gone, scoring = common.backend_state, common.pid_gone, common.scoring
COPIES = ('source.json','original-context.txt','B-request.json','observations.json','acquisition-proof.json','request.json','native-request.json')


def recipe():
    value = selected.load_writer_recipe(PROFILE)
    require(value['context_size'] == SETTINGS.context_size and value['full_swa'] is False
        and value['gpu_layers'] == 99 and value['cpu_moe_layers'] == 0 and value['threads'] == 4
        and value['sha256'] == SETTINGS.model_sha256, 'Declared Gemma deployment differs')
    return value


def prepared(request):
    return native.prepare_request(request, SETTINGS, contract_id=contract.VERSION, validate_request=contract.validate_writer_request)


def producers():
    names = ('masked_source_draft_requests','auto_source_draft_requests','compact_raw_draft_requests','compact_raw_evidence',
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
    return request,source,context


def acquisition_proof(track):
    """Check the root-replayed native completion and exact direct pins, no model imports."""
    folder = ACQUISITION_PARENT.resolve()
    reg = track(folder/'registration.json'); result = track(folder/'result.json'); current = track(folder/'diagnostic.json')
    require(reg['version'] == 'masked-source-acquisition-1'
        and reg['registration_sha256'] == fingerprint({k:v for k,v in reg.items() if k != 'registration_sha256'})
        and current['registration_sha256'] == reg['registration_sha256'] and current['status'] == 'complete'
        and current['passed'] is True and current['original_backend_restored'] is True,'Fresh acquisition is not complete/restored')
    fixed = {'version':'masked-source-acquisition-1','passed':True,'reason_code':'complete_masked_observations',
        'asr_calls':68,'control_calls':2,'real_audio_calls':66,'model_loads':1,'new_recap_calls':0,
        'candidate_generated':False,'score':None,'accuracy_verified':False,'independent_recognizer_vote':False,
        'original_backend_restored':True,'native_complete':True}
    require(all(same(result.get(k),v) for k,v in fixed.items()),'Fresh native source/control coverage changed')
    require(set(result['controls']) == {'silence','noise'} and all(v.get('passed') is True
        and v.get('parsed_nonempty') is False and v.get('protocol_category') in ('empty_raw','native_none_empty_tail')
        for v in result['controls'].values()),'Both fresh negative controls must be clear')
    require(all(type(current[k]) in (int,float) and math.isfinite(current[k]) and current[k] >= 0
        and same(current[k],result[k]) for k in ('local_seconds','work_seconds'))
        and current['work_seconds'] <= 840 and current['work_seconds'] <= current['local_seconds'] <= 1200,'Acquisition budget changed')
    artifacts = {str(p.resolve()):file_hash(p) for p in folder.rglob('*') if p.is_file() and p.name not in ('.lock','result.json')}
    require(same(artifacts,result['artifacts']),'Fresh acquisition artifact coverage changed')
    pins = {**reg['pins'],**artifacts,str(folder/'result.json'):file_hash(folder/'result.json')}
    required = {str((folder/name).resolve()) for name in ('registration.json','native-plan.json','assets.json','input-proof.json',
        'observations.json','diagnostic.json','worker/worker-result.json','lifecycle.json')}
    required |= {str((ROOT/name).resolve()) for name in ('scripts/masked_source_acquisition.py','src/masked_source_native.py','src/qwen_audio_attention.py')}
    require(required <= set(pins),'Mandatory native source proof missing')
    for path,digest in pins.items():
        require(not Path(path).is_symlink() and file_hash(Path(path)) == digest,'Fresh acquisition artifact changed')
        track.pins[str(Path(path).resolve())] = digest
    plan = track(folder/'native-plan.json'); observation = track(folder/'observations.json')
    require(plan['version'] == 'masked-source-plan-1'
        and plan['plan_sha256'] == fingerprint({k:v for k,v in plan.items() if k != 'plan_sha256'})
        and observation['plan_sha256'] == plan['plan_sha256']
        and same(observation['execution_identity'],plan['execution_identity'])
        and same(observation['audio_attention']['policy'],plan['attention_policy'])
        and same(observation['audio_attention']['source_identity'],plan['attention_source'])
        and same(observation['audio_attention']['encoder_config'],plan['encoder_config'])
        and same(observation['audio_attention']['model_config'],plan['model_config'])
        and all(same(observation[k],plan['parent_plan'][k]) for k in
            ('source_rows_sha256','original_mono_sha256','original_mono_frames','sample_rate')),'Native source/attention binding changed')
    runtime = result['runtime_backend']
    require(same(result['attention_policy'],contract.ATTENTION_POLICY)
        and result['attention_source_sha256'] == fingerprint(plan['attention_source'])
        and result['source_rows_sha256'] == observation['source_rows_sha256']
        and result['geometry_sha256'] == fingerprint(plan['parent_plan']['geometry'])
        and result['observation_envelope_sha256'] == fingerprint(observation)
        and runtime['asr_backend'] == 'transformers' and runtime['audio_backend'] == contract.ENCODER_CONFIG['backend']
        and same(runtime['model_config'],plan['model_config']) and same(runtime['audio_encoder_config'],contract.ENCODER_CONFIG)
        and type(runtime['thinker_text_backend']) is str and bool(runtime['thinker_text_backend'])
        and type(runtime['device']) is str and runtime['device'].startswith('cuda') and runtime['dtype'] == 'torch.float16',
        'Native runtime/mask/source completion attestation changed')
    require(len(plan['parent_plan']['geometry']) == len(observation['observations'])
        and all(same({k:row[k] for k in geometry},geometry) for row,geometry in
            zip(observation['observations'],plan['parent_plan']['geometry'])), 'Native source geometry differs')
    attention = observation['audio_attention']
    require(all(pins.get(p) == h for p,h in attention['source_identity']['files'].items())
        and pins.get(attention['model_config']['path']) == attention['model_config']['sha256'],'Native attention code/model configuration unpinned')
    for row in observation['observations']:
        require(pins.get(row['receipt']['path']) == row['receipt']['sha256'],'Native owner receipt missing')
    for row in attention['control_receipts']:
        require(pins.get(row['path']) == row['sha256'],'Native control receipt missing')
    source_bindings = track(folder/'input-proof.json')['source_bindings']
    require(source_bindings['detector_provenance']['mono_sha256'] == observation['original_mono_sha256'],'Original mono binding changed')
    proof = {'version':'masked-source-inheritance-1','folder':str(folder),'envelope_sha256':fingerprint(observation),
        'native_artifacts':pins,'result':{k:v for k,v in result.items() if k != 'artifacts'},'source_bindings':source_bindings,
        'local_seconds':current['local_seconds'],'work_seconds':current['work_seconds'],'original_backend_restored':True}
    return observation,proof


def validate_registration(folder):
    reg = read(folder/'registration.json')
    require(reg['version'] == VERSION and reg['round'] == ROUND and same(reg['settings'],asdict(SETTINGS))
        and same(reg['limits'],[LIMIT,WORK_LIMIT,CLEANUP_LIMIT])
        and reg['registration_sha256'] == fingerprint({k:v for k,v in reg.items() if k != 'registration_sha256'}),'Registration changed')
    required = {str(p.resolve()) for p in producers()} | {str((folder/n).resolve()) for n in COPIES}
    required.add(str((ACQUISITION_PARENT/'observations.json').resolve()))
    require(required <= set(reg['pins']),'Required input/native proof missing')
    for path,digest in reg['pins'].items(): require(file_hash(Path(path)) == digest,'Local input/native producer changed')
    track = Tracker(); b,source,context = inputs(track)
    observations,proof = acquisition_proof(track)
    require(same(observations,read(folder/'observations.json')) and same(proof,read(folder/'acquisition-proof.json')),
            'Copied acquisition differs from direct native origin')
    request = contract.build_writer_request(b,observations); coverage = contract.validate_writer_request(request)
    expected = {'B-request.json':b,'request.json':request,'source.json':source,'native-request.json':prepared(request)}
    require(all(reg['pins'].get(p) == h for p,h in track.pins.items()) and same(reg['coverage'],coverage)
        and all(same(read(folder/name),value) for name,value in expected.items())
        and (folder/'original-context.txt').read_bytes() == context,'Copied local inputs changed')
    deployment = recipe()
    require(same(deployment,reg['deployment']) and reg['pins'].get(deployment['server_binary']) == file_hash(Path(deployment['server_binary'])),'Deployment changed')
    return reg


def prepare(folder=FOLDER):
    folder = Path(folder).resolve()
    with locked(folder):
        require(set(folder.iterdir()) == {folder/'.lock'},'Already initialized; no restart')
        write_json(folder/'screen.json',{'version':VERSION,'round':ROUND,'status':'preparing','local_seconds':0.,
            'work_seconds':0.,'cleanup_seconds':0.,'stages':[],'original_backend_restored':True,'qualified':False,'score':None})
        try:
            with counted(folder,'local_input_preparation'):
                track = Tracker(); b,source,context = inputs(track)
                observations,proof = acquisition_proof(track)
                request = contract.build_writer_request(b,observations)
                coverage = contract.validate_writer_request(request); deployment = recipe()
                copies = {'B-request.json':b,'observations.json':observations,'acquisition-proof.json':proof,
                          'request.json':request,'source.json':source,'native-request.json':prepared(request)}
                for name,value in copies.items(): immutable(folder/name,value)
                immutable(folder/'original-context.txt',context,raw=True)
                for path in [*producers(),Path(deployment['server_binary']),*[folder/n for n in COPIES]]: track(path,raw=True)
                reg = {'version':VERSION,'round':ROUND,'settings':asdict(SETTINGS),'limits':[LIMIT,WORK_LIMIT,CLEANUP_LIMIT],
                       'coverage':coverage,'deployment':deployment,'pins':track.pins}
                reg['registration_sha256'] = fingerprint(reg); immutable(folder/'registration.json',reg)
            return save(folder,status='prepared',registration_sha256=reg['registration_sha256'],
                        acquisition_local_seconds=proof['local_seconds'],acquisition_work_seconds=proof['work_seconds'])
        except BaseException as exc: save(folder,status='failed',error_type=type(exc).__name__); raise


def run_writer(folder,reg):
    manager = None; entered = False; identity = previous = failure = result = None
    try:
        with counted(folder,'masked_source_writer') as deadline:
            d = reg['deployment']; previous = backend_state(d); save(folder,original_backend_restored=False)
            manager = lb.temporary_local_writer(d['weights'],d['server_binary'],alias=d['model'],expected_sha256=d['sha256'],
                context_size=SETTINGS.context_size,full_swa=False,threads=4,gpu_layers=99,cpu_moe_layers=0,
                restore_previous=True,admin_url=d['admin_url'],restore_model=d['restore_model'],log_path=folder/'writer-backend/server.log')
            endpoint,identity = manager.__enter__(); entered = True
            immutable(folder/'writer-backend/backend.json',identity)
            require(endpoint == native.ENDPOINT and identity['model_path'] == d['weights']
                and identity['server_binary_sha256'] == reg['pins'][d['server_binary']], 'Owned runtime differs from pins')
            result = native.ask(folder/'requests',KEY,read(folder/'native-request.json'),identity,deadline)
    except BaseException as exc: failure = exc
    finally:
        if manager is not None:
            restored = False; closed = not entered; error = None
            try:
                with counted(folder,'owned_cleanup',cleanup=True):
                    notes = tuple(getattr(failure,'__notes__',()))
                    if entered: manager.__exit__(type(failure) if failure else None,failure,failure.__traceback__ if failure else None)
                    require(tuple(getattr(failure,'__notes__',())) == notes,'Owned cleanup failed')
                    closed = True; restored = backend_state(reg['deployment']) == previous
                    require(restored and (identity is None or pid_gone(identity['pid'])),'Owned backend not stopped/restored')
            except BaseException as exc:
                error = type(exc).__name__
                if failure is None: failure = exc
            finally:
                immutable(folder/'writer-lifecycle.json',{'original_backend':previous,'original_backend_restored':restored,
                    'context_closed':closed,'error_type':error}); save(folder,original_backend_restored=restored)
    if failure is not None: raise failure
    return result


def replay_local(folder,reg,track):
    source = track(folder/'source.json'); identity = track(folder/'writer-backend/backend.json')
    require(identity['model_path'] == reg['deployment']['weights'] and identity['server_binary_sha256'] == reg['pins'][reg['deployment']['server_binary']], 'Native runtime pin changed')
    value = native.replay_native(folder/'requests'/(KEY+'.json'),track(folder/'native-request.json'),identity,track)
    contract.validate_writer_response(value)
    require(same(value,track(folder/'translation.json')),'Saved native draft changed')
    draft = [{'index':r['index'],'ts_line':r['ts_line'],'text':value['owners'][str(r['index'])]['chinese']} for r in source]
    for name,rows in [('source',source),('draft',draft)]:
        require(same(track(folder/(name+'.json')),rows)
            and parse_srt(folder/(name+'.utterances.srt'),preserve_text_whitespace=True) == [SrtBlock(**r) for r in rows], 'Exact owner/SRT changed')
        track(folder/(name+'.utterances.srt'),raw=True)
    life = track(folder/'writer-lifecycle.json')
    require(life['original_backend_restored'] is True and life['context_closed'] is True and life['error_type'] is None,'Owned cleanup failed')
    require({p.name for p in (folder/'requests').iterdir()} == {KEY+'.json',KEY+'.sse'},'Unexpected native attempt artifacts')
    return source,draft


def execute_local(folder=FOLDER):
    folder = Path(folder).resolve()
    with locked(folder):
        require(state(folder)['status'] == 'prepared','Native attempt cannot resume or reroll')
        try:
            with counted(folder,'execution_input_replay'):
                reg = validate_registration(folder); require(state(folder)['registration_sha256'] == reg['registration_sha256'],'State binding changed')
                save(folder,status='running'); source = read(folder/'source.json')
                selected.write_checked([SrtBlock(**r) for r in source],folder/'source.utterances.srt')
            value = run_writer(folder,reg)
            with counted(folder,'exact_draft_replay'):
                contract.validate_writer_response(value); immutable(folder/'translation.json',value)
                draft = [{**r,'text':value['owners'][str(r['index'])]['chinese']} for r in source]
                immutable(folder/'draft.json',draft); selected.write_checked([SrtBlock(**r) for r in draft],folder/'draft.utterances.srt')
                track = Tracker(); source,draft = replay_local(folder,reg,track)
                immutable(folder/'completion.json',{'version':VERSION,'source_sha256':rows_hash(source),'target_sha256':rows_hash(draft),
                    'native_artifacts':track.pins,'treatment_adherent':True,'source_accuracy_verified':False})
            for name in ('source.utterances.srt','draft.utterances.srt'): (folder/name).chmod(0o444)
            return save(folder,status='local_complete',finished_utc=now(),treatment_adherent=True,
                        acquisition_plus_candidate_local_seconds=state(folder)['local_seconds']+read(folder/'acquisition-proof.json')['local_seconds'])
        except BaseException as exc:
            save(folder,status='treatment_failed' if isinstance(exc,native.TreatmentAdherenceError) else 'failed',
                 error_type=type(exc).__name__,treatment_adherent=False); raise


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
        'quality-gemma-thinking-20260916', 'quality-compact-source-20260916')]]
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
    logging.getLogger(__name__).info('Masked source status=%s score=%s qualified=false',result.get('status'),result.get('score'))


if __name__ == '__main__': main()
