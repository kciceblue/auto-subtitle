"""DG-G1 conditional greedy Gemma draft; local work is independent of scoring."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from copy import deepcopy
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src import compact_raw_draft_requests as contract
from src import compact_raw_evidence as codec
from src import local_backend as lb
from src import local_gemma_native as native
from src import long_context_gemma_native as lc_native
from src import selected_pipeline as selected
from src.late_audio import _backend_identity
from src.translate import SrtBlock, parse_srt
from src.workflow_state import file_hash, fingerprint, write_json

VERSION, ROUND = 'greedy-source-screen-1', 'DG-G1'
FOLDER = ROOT/'output/quality-greedy-source-20260916'
B_PARENT = ROOT/'output/quality-compact-raw-20260915'
B_REQUEST_SHA256 = 'e9815cde9b9e20523c9e0d92b643da945081fe4efd8db7ba6413e5578c8e2d29'
SOURCE = '807e6b35f2099421f68ef326813123d8f4a9c958afe179a0e68540641dbf33a1'
CONTEXT = '7ea692a93e6c2a9bc64bab0d7e8c58ddc47b07a268055de25c11d4c85f3bbf21'
POOL = 'cf71e2d12a6905c8ca728b00a9a548567b4f90fecd63e9d988841ead96c31a13'
PLAN = ROOT/'design/quality-greedy-source-20260916.md'
PROFILE = ROOT/'profiles/long-context-gemma.json'
LIMIT, WORK_LIMIT, CLEANUP_LIMIT = 3600, 3180, 420
KEY = 'greedy-source-draft'
SETTINGS = native.GemmaSettings(temperature=0.0, maximum_seconds=WORK_LIMIT)
require, same, now = native.require, native.same, native.now
LOG = logging.getLogger(__name__)


def read(path): return native.strict_json(Path(path).read_bytes())
def state(folder): return read(folder/'screen.json')
def save(folder, **changes):
    value = state(folder); value.update(changes); write_json(folder/'screen.json', value); return value


def immutable(path, value, *, raw=False):
    data = value if raw else (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle: handle.write(data)
    path.chmod(0o444)


@contextmanager
def locked(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); yield


class Tracker:
    def __init__(self): self.pins = {}
    def __call__(self, path, raw=False):
        path = Path(path); require(not path.is_symlink(), 'Symlink input'); path = path.resolve(strict=True)
        data = path.read_bytes(); digest = hashlib.sha256(data).hexdigest()
        require(str(path) not in self.pins or self.pins[str(path)] == digest, 'Input changed during replay')
        self.pins[str(path)] = digest; return data if raw else native.strict_json(data)


def rows_hash(rows):
    return hashlib.sha256((json.dumps(rows, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False)+'\n').encode('utf-8')).hexdigest()


@contextmanager
def counted(folder, name, *, cleanup=False):
    start = time.monotonic(); s = state(folder); error = None
    require(all(type(s[k]) in (int, float) and math.isfinite(s[k]) and s[k] >= 0
                for k in ('local_seconds', 'work_seconds', 'cleanup_seconds')), 'Invalid time ledger')
    allowance = min(LIMIT-s['local_seconds'], WORK_LIMIT-s['work_seconds'])
    mask = timer = None
    try:
        if cleanup:
            # Owned restoration is unconditional, even after the work/total limit.
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
            timer = signal.getitimer(signal.ITIMER_REAL); signal.setitimer(signal.ITIMER_REAL, 0)
            yield None
        else:
            with native._within(start+allowance): yield start+allowance
    except BaseException as exc: error = type(exc).__name__; raise
    finally:
        elapsed = time.monotonic()-start; s = state(folder)
        s['local_seconds'] += elapsed; s['cleanup_seconds' if cleanup else 'work_seconds'] += elapsed
        over = s['local_seconds'] >= LIMIT or s['work_seconds'] >= WORK_LIMIT or s['cleanup_seconds'] >= CLEANUP_LIMIT
        if cleanup:
            over |= bool(timer[0] and elapsed >= timer[0])
            while signal.SIGALRM in signal.sigpending(): signal.sigtimedwait({signal.SIGALRM}, 0); over = True
            if timer[0] > elapsed: signal.setitimer(signal.ITIMER_REAL, timer[0]-elapsed, timer[1])
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        s['stages'].append({'stage':name, 'seconds':elapsed, 'cleanup':cleanup,
                            'error_type':error or ('TimeoutError' if over else None)})
        write_json(folder/'screen.json', s)
        if over and error is None: raise TimeoutError('Greedy screen local allowance exhausted')


def recipe():
    value = selected.load_writer_recipe(PROFILE)
    require(value['context_size'] == SETTINGS.context_size and value['full_swa'] is False
        and value['gpu_layers'] == 99 and value['cpu_moe_layers'] == 0 and value['threads'] == 4
        and value['sha256'] == SETTINGS.model_sha256, 'Declared Gemma deployment differs')
    return value


def prepared(request):
    """The complete visible payload changes exactly one frozen LC field."""
    value = native.prepare_request(request, SETTINGS, contract_id=contract.VERSION,
                                   validate_request=contract.validate_writer_request)
    original = lc_native.prepare_request(request)
    require(same(original['payload']['temperature'], 1.0), 'Frozen LC temperature changed')
    expected = deepcopy(original['payload']); expected['temperature'] = 0.0
    require(same(value['payload'], expected), 'Greedy payload differs beyond temperature')
    require(same(value['request'], original['request']), 'Greedy source/instruction/schema changed')
    return value


def producers():
    names = ('compact_raw_draft_requests', 'compact_raw_evidence', 'raw_evidence_draft_requests',
        'readable_draft_requests', 'readable_evidence', 'episode_draft_requests', 'temporal_evidence',
        'coherence', 'coherence_workflow', 'quality', 'evidence_context', 'contextual_review_contract',
        'long_context_gemma_native', 'contextual_asr_recap_native', 'revisit_workflow',
        'local_gemma_native', 'local_backend', 'selected_pipeline', 'late_audio', 'translate', 'config', 'workflow_state')
    return [PLAN, PROFILE, Path(__file__).resolve(), *[ROOT/'src'/(name+'.py') for name in names]]


def inputs(track):
    request = track(B_PARENT/'request.json'); require(fingerprint(request) == B_REQUEST_SHA256, 'Fixed B request changed')
    coverage = contract.validate_writer_request(request); require(coverage['observations'] == 799, 'All 799 observations required')
    body = codec.decode_body(request['body']); require(same(codec.encode_body(body), request['body']), 'Codec roundtrip changed')
    source = track(B_PARENT/'source.json'); context = track(B_PARENT/'original-context.txt', raw=True)
    require(rows_hash(source) == SOURCE and hashlib.sha256(context).hexdigest() == CONTEXT, 'Original source/context changed')
    require(same(body['source_evidence']['source_rows'], source)
        and body['source_evidence']['original_context'] == context.decode('utf-8'), 'B source/context binding differs')
    return request, source, context, coverage


def validate_registration(folder):
    reg = read(folder/'registration.json')
    require(reg['version'] == VERSION and reg['round'] == ROUND and same(reg['settings'], asdict(SETTINGS))
        and reg['limits'] == [LIMIT, WORK_LIMIT, CLEANUP_LIMIT]
        and reg['registration_sha256'] == fingerprint({k:v for k,v in reg.items() if k != 'registration_sha256'}), 'Registration changed')
    required = {str(p.resolve()) for p in producers()} | {str((folder/n).resolve()) for n in
        ('source.json', 'original-context.txt', 'request.json', 'native-request.json')}
    require(required <= set(reg['pins']), 'Required local input/producer pin missing')
    for path, digest in reg['pins'].items(): require(file_hash(Path(path)) == digest, 'Local input/producer changed')
    track = Tracker(); request, source, context, coverage = inputs(track)
    require(all(reg['pins'].get(p) == h for p,h in track.pins.items()) and same(reg['coverage'], coverage)
        and same(read(folder/'request.json'), request) and same(read(folder/'source.json'), source)
        and (folder/'original-context.txt').read_bytes() == context
        and same(read(folder/'native-request.json'), prepared(request)), 'Copied local input changed')
    value = recipe(); require(same(value, reg['deployment']) and reg['pins'].get(value['server_binary']) == file_hash(Path(value['server_binary'])), 'Deployment changed')
    return reg


def prepare(folder=FOLDER):
    folder = Path(folder).resolve()
    with locked(folder):
        require(set(folder.iterdir()) == {folder/'.lock'}, 'Already initialized; no restart')
        write_json(folder/'screen.json', {'version':VERSION, 'round':ROUND, 'status':'preparing',
            'local_seconds':0., 'work_seconds':0., 'cleanup_seconds':0., 'stages':[],
            'original_backend_restored':True, 'qualified':False, 'score':None})
        try:
            with counted(folder, 'local_input_preparation'):
                track = Tracker(); request, source, context, coverage = inputs(track); deployment = recipe()
                for name, value in [('source.json', source), ('request.json', request), ('native-request.json', prepared(request))]: immutable(folder/name, value)
                immutable(folder/'original-context.txt', context, raw=True)
                for path in [*producers(), Path(deployment['server_binary']), *[folder/n for n in
                    ('source.json', 'original-context.txt', 'request.json', 'native-request.json')]]: track(path, raw=True)
                reg = {'version':VERSION, 'round':ROUND, 'settings':asdict(SETTINGS), 'limits':[LIMIT, WORK_LIMIT, CLEANUP_LIMIT],
                       'coverage':coverage, 'deployment':deployment, 'pins':track.pins}
                reg['registration_sha256'] = fingerprint(reg); immutable(folder/'registration.json', reg)
            return save(folder, status='prepared', registration_sha256=reg['registration_sha256'])
        except BaseException as exc: save(folder, status='failed', error_type=type(exc).__name__); raise


def backend_state(deployment):
    return _backend_identity(lb._request_json(deployment['admin_url'].rstrip('/')+'/status', admin=True))


def pid_gone(pid):
    try: os.kill(pid, 0)
    except ProcessLookupError: return True
    return False


def start_writer(folder,reg,owned):
    """Start the owned child without an internal finally under the work alarm."""
    d=reg['deployment'];model=Path(d['weights']).resolve(strict=True);binary=Path(d['server_binary']).resolve(strict=True)
    require(lb._ALIAS.fullmatch(d['model']) and (owned['previous'] is None or lb._ALIAS.fullmatch(owned['previous'])),
        'Invalid owned or original backend alias')
    require(binary.is_file() and os.access(binary,os.X_OK),'Writer binary is not executable')
    manifest,stats=lb._weight_bundle(model,d['sha256'],None);owned['model_stats']=stats
    digest=lb._sha256(binary);require(digest==reg['pins'][str(binary)],'Writer executable changed')
    command=[str(binary),'--model',str(model),'--host','127.0.0.1','--port',str(lb.LOCAL_PORT),
        '--alias',d['model'],'--ctx-size',str(SETTINGS.context_size),'--parallel','1','--n-gpu-layers','99',
        '--threads','4','--batch-size','512','--ubatch-size','128','--flash-attn','on',
        '--cache-type-k','q8_0','--cache-type-v','q8_0','--fit','off','--jinja']
    owned['reserved']=lb._reserve_port();log=folder/'writer-backend/server.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as output:
        owned['unload_attempted']=True
        lb._request_json(d['admin_url'].rstrip('/')+'/unload',{},admin=True,timeout=45)
        owned['reserved'].close()
        owned['process']=lb.subprocess.Popen(command,stdout=output,stderr=lb.subprocess.STDOUT,start_new_session=True)
    props=lb._ready(owned['process'],model,d['model'])
    require(all(lb._file_identity(path)==old for path,old in stats.items()),'Writer model changed during startup')
    identity={'model_path':str(model),'model_alias':d['model'],'model_sha256':manifest[0]['sha256'],
        'model_files':manifest,'model_bundle_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        'gpu_layers':99,'cpu_moe_layers':0,'threads':4,'server_binary':str(binary),'server_binary_sha256':digest,
        'pid':owned['process'].pid,'context_size':SETTINGS.context_size,'full_swa':False,'command':command,
        'build_info':props.get('build_info'),'model_ftype':props.get('model_ftype'),
        'reported_model_alias':props.get('model_alias'),
        'chat_template_sha256':hashlib.sha256(str(props.get('chat_template','')).encode()).hexdigest(),
        'default_generation_settings':props.get('default_generation_settings')}
    return native.ENDPOINT,identity


def stop_writer(reg,owned):
    """Called only inside unconditional, alarm-masked controller cleanup."""
    if owned.get('reserved') is not None:owned['reserved'].close()
    if owned.get('process') is not None:lb._stop_owned_process(owned['process'])
    if owned.get('unload_attempted'):
        admin=reg['deployment']['admin_url'].rstrip('/');previous=owned['previous']
        if previous is None:
            lb._request_json(admin+'/unload',{},admin=True,timeout=45)
        else:
            answer=lb._request_json(admin+'/load',{'model':previous},admin=True,timeout=lb.RESTORE_TIMEOUT)
            require(answer.get('loaded')==previous,'Original backend load was not confirmed')
    require(backend_state(reg['deployment'])==owned['previous'],'Original backend was not restored')


def run_writer(folder,reg):
    owned=None;identity=failure=value=None
    try:
        with counted(folder,'local_whole_draft_writer') as deadline:
            d=reg['deployment'];previous=backend_state(d);owned={'previous':previous};save(folder,original_backend_restored=False)
            endpoint,identity=start_writer(folder,reg,owned);immutable(folder/'writer-backend/backend.json',identity)
            require(endpoint==native.ENDPOINT and identity['model_path']==d['weights']
                and identity['server_binary_sha256']==reg['pins'][d['server_binary']],'Owned Gemma runtime changed')
            value=native.ask(folder/'requests',KEY,read(folder/'native-request.json'),identity,deadline)
            require(all(lb._file_identity(path)==old for path,old in owned.get('model_stats',{}).items()),
                'Writer model changed during inference')
    except BaseException as exc:failure=exc
    finally:
        if owned is not None:
            restored=False;closed=False;error=None
            try:
                with counted(folder,'writer_owned_cleanup',cleanup=True):
                    stop_writer(reg,owned);closed=True;restored=True
                    require(identity is None or pid_gone(identity['pid']),'Gemma owned process not stopped')
            except BaseException as exc:
                error=type(exc).__name__
                if failure is None:failure=exc
            finally:
                immutable(folder/'writer-lifecycle.json',{'original_backend':owned['previous'],'original_backend_restored':restored,
                    'context_closed':closed,'error_type':error});save(folder,original_backend_restored=restored)
    if failure is not None:raise failure
    return value



def replay_local(folder, reg, track):
    source = track(folder/'source.json'); identity = track(folder/'writer-backend/backend.json')
    require(identity['model_path'] == reg['deployment']['weights'] and identity['server_binary_sha256'] == reg['pins'][reg['deployment']['server_binary']], 'Native runtime pin changed')
    deployment = reg['deployment']
    require(identity['model_alias'] == SETTINGS.model and identity['model_sha256'] == SETTINGS.model_sha256
        and identity['context_size'] == SETTINGS.context_size and identity['full_swa'] is False
        and identity['cpu_moe_layers'] == 0 and identity['gpu_layers'] == 99 and identity['threads'] == 4,
        'Native deployment configuration changed')
    request = track(folder/'request.json')
    require(same(track(folder/'native-request.json'), prepared(request)), 'Native greedy request changed')
    value = native.replay_native(folder/'requests'/(KEY+'.json'), prepared(request), identity, track)
    require(same(value, track(folder/'translation.json')), 'Saved native draft changed')
    draft = [{'index':r['index'], 'ts_line':r['ts_line'], 'text':value['owners'][str(r['index'])]['chinese']} for r in source]
    require(len(draft) == 66 and all(type(r['text']) is str and r['text'].strip() for r in draft), 'All 66 owner texts required')
    for name, rows in [('source', source), ('draft', draft)]:
        require(same(track(folder/(name+'.json')), rows)
            and parse_srt(folder/(name+'.utterances.srt'), preserve_text_whitespace=True) == [SrtBlock(**r) for r in rows], 'Exact owner/SRT changed')
        track(folder/(name+'.utterances.srt'), raw=True)
    life = track(folder/'writer-lifecycle.json')
    require(life['original_backend_restored'] is True and life['context_closed'] is True and life['error_type'] is None, 'Owned cleanup failed')
    require({p.name for p in (folder/'requests').iterdir()} == {KEY+'.json', KEY+'.sse'}, 'Unexpected native attempt artifacts')
    return source, draft


def execute_local(folder=FOLDER):
    folder = Path(folder).resolve()
    with locked(folder):
        require(state(folder)['status'] == 'prepared', 'Native attempt cannot resume or reroll')
        try:
            with counted(folder, 'execution_input_replay'):
                reg = validate_registration(folder); require(state(folder)['registration_sha256'] == reg['registration_sha256'], 'State binding changed')
                save(folder, status='running'); source = read(folder/'source.json')
                selected.write_checked([SrtBlock(**r) for r in source], folder/'source.utterances.srt')
            value = run_writer(folder, reg)
            with counted(folder, 'exact_draft_replay'):
                immutable(folder/'translation.json', value)
                draft = [{**r, 'text':value['owners'][str(r['index'])]['chinese']} for r in source]
                immutable(folder/'draft.json', draft); selected.write_checked([SrtBlock(**r) for r in draft], folder/'draft.utterances.srt')
                track = Tracker(); source, draft = replay_local(folder, reg, track)
                immutable(folder/'completion.json', {'version':VERSION, 'source_sha256':rows_hash(source),
                    'target_sha256':rows_hash(draft), 'native_artifacts':track.pins, 'treatment_adherent':True, 'source_accuracy_verified':False})
            for name in ('source.utterances.srt', 'draft.utterances.srt'): (folder/name).chmod(0o444)
            return save(folder, status='local_complete', finished_utc=now(), treatment_adherent=True)
        except BaseException as exc:
            treatment = isinstance(exc, getattr(native, 'TreatmentAdherenceError', ()))
            save(folder, status='treatment_failed' if treatment else 'failed', error_type=type(exc).__name__, treatment_adherent=False)
            raise


def scoring():
    # These imports and every evaluation/history read occur only on explicit review commands.
    from scripts import full_draft_diagnostic as catalog
    from scripts import staged_source_screen as helpers
    from scripts import long_context_gemma_screen as confirmations
    return catalog, helpers, confirmations


RECENT_SCORE_FOLDERS = (
        'quality-staged-source-20260915', 'quality-long-context-gemma-20260915',
        'quality-auto-source-continuation-20260916', 'quality-forced-source-draft-20260916',
        'quality-gemma-thinking-20260916', 'quality-compact-source-20260916',
        'quality-masked-source-draft-continuation-20260916', 'quality-long-source-draft-20260916',
        'quality-voxtral-source-draft-20260916', 'quality-full-evidence-critic-20260916')


def recent_score_paths():
    return [ROOT/'output'/name/'score.json' for name in RECENT_SCORE_FOLDERS]


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
    additions = [*helpers.EXTRA_SCORES, *recent_score_paths()]
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
    modes.add_argument('--prepare', action='store_true'); modes.add_argument('--execute-local', action='store_true')
    modes.add_argument('--score', action='store_true'); modes.add_argument('--confirm', action='store_true')
    args = parser.parse_args(); logging.basicConfig(level=logging.INFO)
    result = prepare() if args.prepare else execute_local() if args.execute_local else review(confirmation=args.confirm)
    LOG.info('Greedy source status=%s score=%s qualified=false', result.get('status'), result.get('score'))


if __name__ == '__main__': main()
