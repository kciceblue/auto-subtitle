"""One local raw-evidence draft and one scalar-only screen, conditional on R1."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import logging
import math
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import full_draft_diagnostic as catalog
from src import raw_evidence_draft_requests as requests
from src import raw_evidence_draft_native as native
from src import readable_draft_revisit as parent
from src import readable_draft_release as parent_release
from src import sparse_revisit as sr
from src import revisit_workflow as rw
from src import contextual_evidence_review as er
from src import whole_owner as wo
from src import late_audio
from src.contextual_review_contract import require, strict_json
from src.workflow_state import file_hash, fingerprint, write_json
from src.translate import parse_srt

VERSION = 'raw-evidence-primary-screen-1'
ROUND = 'A1'
FOLDER = ROOT/'output/quality-raw-evidence-20260915'
PARENT = ROOT/'output/quality-readable-draft-20260915'
PLAN = ROOT/'design/quality-raw-evidence-screen-20260915.md'
LIMIT = 1200
KEY = 'raw-evidence-draft'
read, immutable, same, locked = catalog.read, catalog.immutable, catalog.same, catalog.locked
LOG = logging.getLogger(__name__)
EXTRA_SCORES = [ROOT/'output/quality-full-draft-diagnostic-20260915'/label/'score.json'
                for label in ('DF-L1', 'DF-R1')]+[PARENT/'R1/score.json']


class Tracker:
    def __init__(self): self.pins = {}
    def __call__(self, path, raw=False):
        path = Path(path)
        require(not path.is_symlink(), 'Symlink cannot bind a screen input')
        path = path.resolve(strict=True); data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        require(str(path) not in self.pins or self.pins[str(path)] == digest,
                'Input changed during registration')
        self.pins[str(path)] = digest
        return data if raw else strict_json(data)


def compile_draft(raw, source, target, pack):
    require(isinstance(raw, dict) and set(raw) == {'owners'}
        and set(raw['owners']) == set(map(str, range(1, 67))), 'Incomplete full draft')
    proposed = [wo.prepare_owner(i, raw['owners'][str(i)]['chinese'], source, target, pack)
                for i in range(1, 67)]
    require(all(tx['valid'] is True and tx['target_normalized'] is False for tx in proposed),
            'Draft needs normalization or contains invalid owner text')
    for tx in proposed: wo.validate_owner_transaction(tx, pack, source=source, target=target)
    applied = [tx for tx in proposed if not tx['no_op']]
    result_source, draft, ledger = wo.apply_owner_transactions(source, target, applied, pack)
    require(same([asdict(x) for x in source], [asdict(x) for x in result_source])
        and all(row.text == raw['owners'][str(row.index)]['chinese'] for row in draft),
        'Assembly changed source or model-written words')
    return proposed, draft, ledger


def validate_registration(folder):
    folder = Path(folder)
    reg = read(folder/'registration.json')
    fixed = {'version': VERSION, 'round': ROUND, 'maximum_local_seconds': LIMIT,
        'maximum_drafts': 1, 'maximum_primary_dispatches': 1, 'maximum_confirmations': 0,
        'diagnostic_only': True, 'qualified': False, 'source_sha256': catalog.SOURCE,
        'context_sha256': catalog.CONTEXT, 'pool_version': catalog.POOL}
    require(same({key: reg.get(key) for key in fixed}, fixed)
        and reg.get('registration_sha256') == fingerprint({k:v for k,v in reg.items() if k != 'registration_sha256'}),
        'Screen registration changed')
    required = {PLAN, Path(__file__).resolve(), Path(catalog.__file__).resolve(),
        Path(requests.__file__).resolve(), Path(native.__file__).resolve(), *EXTRA_SCORES,
        *(PARENT/name for name in ('source-pack.json', 'source-map.json',
            'readable/identity-map.json', 'readable/projection.json')),
        *(folder/name for name in ('source.json', 'target.json', 'original-context.txt',
            'request.json', 'ablation.json', 'R1-native-replay.json', 'score-index-before.json'))}
    require({str(path.resolve()) for path in required} <= set(reg['pins']), 'Screen required pin missing')
    for name, digest in reg['pins'].items():
        require(file_hash(Path(name)) == digest, 'Screen producer or input changed')
    replay = read(folder/'R1-native-replay.json')
    require(replay['status'] == 'replayed_not_qualified'
        and all(reg['pins'].get(name) == digest for name, digest in replay['artifacts'].items()),
        'R1 native replay ancestry changed')
    bundle = read(folder/'ablation.json')
    require(bundle.get('version') == requests.VERSION, 'Raw ablation version changed')
    pack = read(PARENT/'source-pack.json')
    counts = requests.validate_ablation(bundle['request'], bundle['audit'], pack,
        read(PARENT/'source-map.json'), read(PARENT/'readable/identity-map.json'),
        projection=read(PARENT/'readable/projection.json'))
    require(same(bundle['request'], read(folder/'request.json'))
        and reg['writer_request_sha256'] == fingerprint(bundle['request'])
        and same(reg['coverage'], counts), 'Registered writer request changed')
    source, target = rw.rows(folder/'source.json'), rw.rows(folder/'target.json')
    require(er.owner_rows_sha256(source) == catalog.SOURCE
        and same([asdict(row) for row in source], pack['source_rows'])
        and [(row.index,row.ts_line) for row in source] == [(row.index,row.ts_line) for row in target]
        and file_hash(folder/'original-context.txt') == catalog.CONTEXT,
        'Registered source, context or geometry changed')
    baseline = er.validate_receipt(reg['baseline_review'])
    baseline_arm = {**fixed, 'target_sha256': er.owner_rows_sha256(target)}
    catalog.validate_score(baseline, baseline_arm)
    require(baseline['score'] == 2 and baseline['purpose'] == 'baseline', 'Benchmark baseline changed')
    return reg


def _prepare_inputs(folder):
    conf = parent.check(PARENT)
    parent._terminal_unqualified(conf, 'R1')
    require(conf['status'] == 'candidate_budget_exhausted', 'R1 did not complete its declared round')
    replay = parent_release.validate_campaign(PARENT, require_four=False)
    require(replay['status'] == 'replayed_not_qualified', 'R1 is not a completed nonqualifying control')
    track = Tracker()
    for name, digest in replay['artifacts'].items():
        track(name, raw=True); require(track.pins[str(Path(name).resolve())] == digest, 'R1 native input changed')
    track(PLAN, raw=True); track(Path(__file__), raw=True); track(Path(catalog.__file__), raw=True)
    for module in (requests, native): track(Path(module.__file__), raw=True)
    extras = [(path, track(path)) for path in EXTRA_SCORES]
    index = catalog.load_score_index(track)
    for path, value in extras:
        index['records'].append({'path': str(path), 'metadata_sha256': track.pins[str(path.resolve())],
            'hashes': {key: value.get(key) for key in ('pool_version', 'target_sha256', 'source_sha256')}})
    immutable(folder/'score-index-before.json', index)
    pack, bound = track(PARENT/'source-pack.json'), track(PARENT/'source-map.json')
    ids, projection = parent.prepared_readable(PARENT, pack, bound)
    ablation = requests.build_ablation(pack, bound, ids, projection=projection)
    counts = requests.validate_ablation(ablation['request'], ablation['audit'], pack, bound, ids, projection=projection)
    require(same(requests.reconstruct_request(ablation['request'], ablation['audit']),
                 parent.writer_request(pack, bound, ids, projection)), 'Ablation does not reconstruct R')
    native.prepare_request(ablation['request'])
    for name in ('source.json', 'target.json', 'original-context.txt'):
        immutable(folder/name, track(PARENT/name, raw=True), raw=True)
    immutable(folder/'ablation.json', ablation); immutable(folder/'request.json', ablation['request'])
    immutable(folder/'R1-native-replay.json', replay)
    baseline = er.validate_receipt(conf['baseline_review'])
    require(baseline['score'] == 2 and baseline['pool_version'] == catalog.POOL, 'Benchmark baseline changed')
    for path in folder.rglob('*'):
        if path.is_file() and path.name not in {'.lock', 'screen.json'}: track(path, raw=True)
    reg = {'version': VERSION, 'round': ROUND, 'created_utc': sr.now(),
        'maximum_local_seconds': LIMIT, 'maximum_drafts': 1, 'maximum_primary_dispatches': 1,
        'maximum_confirmations': 0, 'diagnostic_only': True, 'qualified': False,
        'writer_request_sha256': fingerprint(ablation['request']), 'coverage': counts,
        'source_sha256': catalog.SOURCE, 'context_sha256': catalog.CONTEXT, 'pool_version': catalog.POOL,
        'evidence_paths': conf['evidence_paths'], 'baseline_review': conf['baseline_review'],
        'writer_recipe': conf['writer_recipe'], 'pins': track.pins}
    reg['registration_sha256'] = fingerprint(reg); immutable(folder/'registration.json', reg)
    return validate_registration(folder)


def _failed(folder, exc):
    state = read(folder/'screen.json')
    state.update(status='failed', error_type=type(exc).__name__, finished_utc=sr.now(),
                 qualified=False, confirmed=False, eligible_for_validation=False)
    write_json(folder/'screen.json', state)


def prepare(folder=FOLDER):
    folder = Path(folder).resolve()
    with locked(folder):
        if (folder/'screen.json').exists():
            require(read(folder/'screen.json')['status'] == 'prepared',
                    'Incomplete or failed screen preparation cannot restart')
        else:
            require(set(folder.iterdir()) == {folder/'.lock'}, 'Unregistered screen artifacts exist')
            write_json(folder/'screen.json', {'version': VERSION, 'status': 'preparing', 'round': ROUND,
                'registration_sha256': None, 'local_seconds': 0., 'original_backend_restored': True,
                'diagnostic_only': True, 'qualified': False, 'confirmed': False, 'stages': []})
        try:
            with local_stage(folder, 'registration_and_native_source_replay'):
                reg = (validate_registration(folder) if (folder/'registration.json').exists()
                       else _prepare_inputs(folder))
            state = read(folder/'screen.json')
            state.update(status='prepared', registration_sha256=reg['registration_sha256'])
            write_json(folder/'screen.json', state)
            return reg
        except BaseException as exc:
            _failed(folder, exc)
            raise


@contextmanager
def local_stage(folder, name):
    start = time.monotonic(); state = read(folder/'screen.json')
    seconds = state['local_seconds']
    require(type(seconds) in (int,float) and math.isfinite(seconds) and seconds >= 0, 'Invalid time ledger')
    remaining = LIMIT-seconds
    old_deadline = sr._DEADLINE
    old_handler = signal.getsignal(signal.SIGALRM); old_timer = signal.getitimer(signal.ITIMER_REAL)
    def expired(signum, frame): raise rw.LocalBudgetExceeded('Screen local deadline')
    error = None
    try:
        if remaining <= 0: raise rw.LocalBudgetExceeded('Screen local budget exhausted')
        sr._DEADLINE = start+remaining
        signal.signal(signal.SIGALRM, expired); signal.setitimer(signal.ITIMER_REAL, remaining)
        yield
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old_handler)
        sr._DEADLINE = old_deadline
        elapsed = time.monotonic()-start
        if old_timer[0]: signal.setitimer(signal.ITIMER_REAL, max(.001,old_timer[0]-elapsed),old_timer[1])
        state = read(folder/'screen.json')
        state['local_seconds'] += elapsed
        overrun = error is None and state['local_seconds'] >= LIMIT
        if overrun: error = 'LocalBudgetExceeded'
        state['stages'].append({'stage':name,'seconds':elapsed,'error':error,'finished_utc':sr.now()})
        write_json(folder/'screen.json', state); LOG.info('%s %.2fs error=%s',name,elapsed,error)
        if overrun: raise rw.LocalBudgetExceeded('Screen local stage exceeded budget')


def restore(folder, previous):
    start=time.monotonic(); ok=False; error=None
    try:
        active=late_audio._backend_identity(rw._request_json(rw.ADMIN+'/status',admin=True))
        if active != previous:
            rw._request_json(rw.ADMIN+('/unload' if previous is None else '/load'),
                {} if previous is None else {'model':previous},admin=True,timeout=360)
        ok=late_audio._backend_identity(rw._request_json(rw.ADMIN+'/status',admin=True)) == previous
        require(ok,'Original backend restoration failed')
    except BaseException as exc: error=type(exc).__name__; raise
    finally:
        state=read(folder/'screen.json'); elapsed=time.monotonic()-start
        state['local_seconds']+=elapsed;state['original_backend_restored']=ok
        state['stages'].append({'stage':'restoration','seconds':elapsed,'error':error,'finished_utc':sr.now()})
        write_json(folder/'screen.json',state)


def generate(folder, reg):
    previous=None; captured=False; failure=None
    try:
        with local_stage(folder,'local_writer_and_exact_assembly'):
            validate_registration(folder)
            previous=late_audio._backend_identity(rw._request_json(rw.ADMIN+'/status',admin=True));captured=True
            state=read(folder/'screen.json');state.update(original_backend_restored=False,original_backend=previous)
            write_json(folder/'screen.json',state)
            request=read(folder/'request.json')
            with rw.backend('qwen',folder/'writer-backend',Path(reg['writer_recipe'])) as (endpoint,_):
                raw=native.ask(folder/'requests',KEY,endpoint,request)
            source,target=rw.rows(folder/'source.json'),rw.rows(folder/'target.json')
            proposals,draft,ledger=compile_draft(raw,source,target,read(PARENT/'source-pack.json'))
            for name,value in [('proposed-transactions.json',proposals),('draft.json',[asdict(row) for row in draft]),
                ('transaction-ledger.json',ledger)]: immutable(folder/name,value)
            sr.write_checked(source,folder/'source.utterances.srt');sr.write_checked(draft,folder/'draft.utterances.srt')
    except BaseException as exc: failure=exc;raise
    finally:
        if captured:
            try: restore(folder,previous)
            except BaseException as exc:
                if failure is None: raise
                failure.add_note('Restoration failed: '+type(exc).__name__)


def prepare_score(folder,reg):
    with local_stage(folder,'native_output_replay_and_score_bundle'):
        validate_registration(folder)
        track=Tracker();raw=native.replay_native(folder/'requests'/(KEY+'.json'),read(folder/'request.json'),track)
        source,target=rw.rows(folder/'source.json'),rw.rows(folder/'target.json')
        proposals,draft,ledger=compile_draft(raw,source,target,read(PARENT/'source-pack.json'))
        require(same(proposals,read(folder/'proposed-transactions.json'))
            and same(ledger,read(folder/'transaction-ledger.json'))
            and same([asdict(x) for x in draft],read(folder/'draft.json')), 'Native draft or assembly changed')
        manifest=er.prepare_bundle(source,draft,source,list(map(Path,reg['evidence_paths'])),
                                  folder/'original-context.txt',folder/'evaluation/bundle')
        arm={'label':ROUND,'target_sha256':er.owner_rows_sha256(draft),'source_sha256':reg['source_sha256'],
            'context_sha256':reg['context_sha256'],'pool_version':reg['pool_version'],
            'manifest':str(manifest),'primary_directory':str(folder/'evaluation/primary')}
        catalog.validate_score(er.read_bundle(manifest),arm)
        for path in [folder/'draft.json',folder/'proposed-transactions.json',folder/'transaction-ledger.json',
                     folder/'draft.utterances.srt',folder/'source.utterances.srt',folder/'writer-backend/backend.json']:
            track(path,raw=True)
        for path in (folder/'evaluation/bundle').rglob('*'):
            if path.is_file():track(path,raw=True)
        previous=catalog.duplicates(read(folder/'score-index-before.json'),arm,folder)
        arm.update(previous_primary=previous['output_dir'] if previous else None,native_artifacts=track.pins)
        immutable(folder/'score-inputs.json',arm)
    return arm


def validate_score_inputs(folder, reg):
    """Pure replay of the exact saved native draft, assembly and review bundle."""
    arm = read(folder/'score-inputs.json')
    require(arm['label'] == ROUND and arm['source_sha256'] == reg['source_sha256']
        and arm['context_sha256'] == reg['context_sha256'] and arm['pool_version'] == reg['pool_version']
        and arm['manifest'] == str(folder/'evaluation/bundle/manifest.json')
        and arm['primary_directory'] == str(folder/'evaluation/primary'), 'Screen score input identity changed')
    track = Tracker()
    raw = native.replay_native(folder/'requests'/(KEY+'.json'), read(folder/'request.json'), track)
    source,target = rw.rows(folder/'source.json'),rw.rows(folder/'target.json')
    proposals,draft,ledger = compile_draft(raw,source,target,read(PARENT/'source-pack.json'))
    require(same(proposals,read(folder/'proposed-transactions.json'))
        and same(ledger,read(folder/'transaction-ledger.json'))
        and same([asdict(row) for row in draft],read(folder/'draft.json'))
        and arm['target_sha256'] == er.owner_rows_sha256(draft), 'Native scored draft changed')
    require(parse_srt(folder/'source.utterances.srt',preserve_text_whitespace=True) == source
        and parse_srt(folder/'draft.utterances.srt',preserve_text_whitespace=True) == draft,
        'Saved subtitle artifact differs from exact native owner rows')
    catalog.validate_score(er.read_bundle(arm['manifest']),arm)
    for path in [folder/'draft.json',folder/'proposed-transactions.json',folder/'transaction-ledger.json',
                 folder/'draft.utterances.srt',folder/'source.utterances.srt',folder/'writer-backend/backend.json']:
        track(path,raw=True)
    require(read(folder/'writer-backend/backend.json').get('loaded_model') == rw.QWEN,
            'Native writer backend identity changed')
    for path in (folder/'evaluation/bundle').rglob('*'):
        if path.is_file():track(path,raw=True)
    require(same(arm['native_artifacts'],track.pins), 'Scored native artifact coverage changed')
    return arm


def _own_primary_clear(arm):
    primary = Path(arm['primary_directory'])
    require(not any(primary.glob('dispatch-*'))
        and not any((primary/name).exists() for name in ('assessment.json','review-run.json',
            'review-process.log','review-response.json')),
        'Own interrupted or completed primary cannot be bypassed or rerolled')


def _score_fields(receipt, arm, reused):
    return {'version':VERSION,'round':ROUND,'status':'complete','score':receipt['score'],
        'primary':receipt['output_dir'],'dispatch_id':receipt['dispatch_id'],'receipt_hashes':receipt['receipt_hashes'],
        'target_sha256':arm['target_sha256'],'source_sha256':arm['source_sha256'],'pool_version':arm['pool_version'],
        'context_sha256':arm['context_sha256'],'reused_completed_primary':reused,'review_seconds':receipt['seconds'],
        'diagnostic_only':True,'qualified':False,'confirmed':False,
        'eligible_for_validation':receipt['score']>=3,'local_gate_passed':None}


def _primary_receipt(path, arm):
    receipt = er.validate_receipt(path)
    catalog.validate_score(receipt, arm)
    require(receipt['purpose'] in {'candidate','baseline'}, 'Screen receipt is not a primary')
    return receipt


def _cached_result(folder, reg):
    arm = validate_score_inputs(folder, reg)
    result = read(folder/'score.json')
    require(type(result.get('reused_completed_primary')) is bool, 'Cached reuse status changed')
    receipt = _primary_receipt(result['primary'], arm)
    if result['reused_completed_primary']:
        _own_primary_clear(arm)
        previous = catalog.duplicates(read(folder/'score-index-before.json'),arm,folder)
        require(previous is not None and previous['dispatch_id'] == receipt['dispatch_id']
            and previous['output_dir'] == receipt['output_dir'], 'Cached earliest primary changed')
    else:
        require(receipt['output_dir'] == arm['primary_directory'], 'Cached native primary path changed')
    expected = _score_fields(receipt, arm, result['reused_completed_primary'])
    seconds = result.get('runner_score_seconds')
    require(set(result) == {*expected, 'runner_score_seconds'}
        and same({key:result[key] for key in expected},expected)
        and type(seconds) in (int,float) and math.isfinite(seconds) and seconds >= 0,
        'Cached score wrapper changed')
    state = read(folder/'screen.json')
    require(state['original_backend_restored'] is True and state['local_seconds'] < LIMIT
        and same(state.get('score'), result['score'])
        and state.get('eligible_for_validation') is result['eligible_for_validation'],
        'Cached screen state changed')
    return result


def screen(folder,reg,arm):
    with local_stage(folder,'score_dispatch_precheck'):
        state=read(folder/'screen.json')
        require(state['status']=='running' and state['original_backend_restored'] is True,
                'Local execution failed status/restoration')
        require(same(arm,validate_score_inputs(folder,reg)), 'Scoring arm changed')
        _own_primary_clear(arm)
        previous=catalog.duplicates(read(folder/'score-index-before.json'),arm,folder)
    started=time.monotonic()
    if previous:
        receipt=previous;reused=True
    else:
        state=read(folder/'screen.json')
        state.update(status='review_reserved',review_reserved_utc=sr.now());write_json(folder/'screen.json',state)
        receipt=sr.review_once(arm['manifest'],Path(arm['primary_directory']),purpose='candidate',baseline_review=reg['baseline_review'])
        reused=False
    with local_stage(folder,'score_receipt_validation'):
        receipt=_primary_receipt(receipt['output_dir'],arm)
        result={**_score_fields(receipt,arm,reused),'runner_score_seconds':time.monotonic()-started}
    immutable(folder/'score.json',result)
    state=read(folder/'screen.json');state.update(status='complete',score=result['score'],
        eligible_for_validation=result['eligible_for_validation'],finished_utc=sr.now())
    write_json(folder/'screen.json',state);return result


def execute(folder=FOLDER):
    folder=Path(folder).resolve()
    with locked(folder):
        try:
            with local_stage(folder,'execution_input_validation'):
                reg=validate_registration(folder);state=read(folder/'screen.json')
                require(state['version']==VERSION and state['registration_sha256']==reg['registration_sha256']
                    and state['diagnostic_only'] is True and state['qualified'] is False
                    and state['confirmed'] is False,'Screen state drift')
                if state['status']=='complete':
                    result=_cached_result(folder,reg)
                else:
                    require(state['status']=='prepared','Reserved/incomplete/failed screen cannot reroll')
                    result=None
            if result is not None:return result
            state=read(folder/'screen.json');state.update(status='running',started_utc=sr.now())
            write_json(folder/'screen.json',state)
            generate(folder,reg);arm=prepare_score(folder,reg);return screen(folder,reg,arm)
        except BaseException as exc:
            _failed(folder,exc)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true');modes.add_argument('--execute',action='store_true')
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    try:
        result=prepare() if args.prepare else execute()
        LOG.info('A1 status=%s score=%s diagnostic_only=true',result.get('status','prepared'),result.get('score'))
        return 0
    except Exception as exc:
        LOG.error('A1 stopped: %s; artifacts retained',type(exc).__name__);return 1

if __name__=='__main__':raise SystemExit(main())
