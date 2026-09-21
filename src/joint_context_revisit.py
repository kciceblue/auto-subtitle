"""Conditional J1: replay the L1 draft and vary only verifier episode context."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from copy import deepcopy
import json
import logging
import math
from pathlib import Path
import signal
import time
from src import joint_context as joint
from src import whole_owner as wo
from src import whole_owner_requests as wr
from src import episode_draft_requests as draft_requests
from src import episode_draft_native as native
from src import episode_draft_revisit as previous
from src import episode_draft_recovery as recovery
from src import episode_draft_release as inherited_gate
from src import temporal_release as treplay
from src import context_revisit as cr
from src import context_release as replay
from src import sparse_release
from src import sparse_revisit as sr
from src import revisit_workflow as rw
from src import late_audio
from src import evidence_context as ec
from src import contextual_evidence_review as er
from src.contextual_review_contract import require, strict_json
from src.workflow_state import file_hash, fingerprint

ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT/'output/quality-episode-draft-20260915'
PLAN = ROOT/'design/quality-next-directions-20260915.md'
VERSION = 'joint-context-local-revisit-1'
ROUND = 'J1'
LIMIT = ROUND_LIMIT = 7200
OWNER_IDS = list(range(1, 67))
LOG = logging.getLogger(__name__)
read, write, _source_pair = cr.read, cr.write, cr._source_pair
_same = joint._same_json


def _terminal_control(conf):
    require(conf.get('status') == 'candidate_budget_exhausted'
        and conf.get('original_backend_restored') is True and not conf.get('eligible_candidate'),
        'J1 requires terminal restored L1 without an eligible candidate')
    require(type(conf.get('local_seconds')) in (int, float)
        and math.isfinite(conf['local_seconds']) and 0 <= conf['local_seconds'] < LIMIT,
        'L1 control campaign exceeded its declared local budget')


def replay_controls(folder, conf, source, target, pack, bound, track):
    """Replay all 66 fixed controls and their original writer without generation."""
    folder = Path(folder).resolve(); parent = Path(conf['control_predecessor']).resolve(strict=True)
    old = previous.check(parent); _terminal_control(old)
    require(_same(track(parent/'campaign.json'), old)
        and conf['input_hashes'].get(str(parent/'campaign.json')) == file_hash(parent/'campaign.json'),
        'L1 control campaign changed or is not pinned')
    require(conf['predecessor'] == old['predecessor'], 'T1 source ancestry changed')
    for key in ('source_sha256', 'target_sha256', 'context_sha256', 'source_pack_sha256',
                'map_sha256', 'pool_version', 'evidence_paths', 'baseline_review',
                'media', 'media_sha256', 'writer_recipe', 'writer_recipe_sha256',
                'short_evidence_directory', 'starting_candidate'):
        require(_same(conf[key], old[key]), 'J1 changed frozen L1 inputs')
    require(conf['reused_predecessor_local_seconds'] == old['local_seconds'],
        'Reused predecessor time changed')
    for path, digest in {**old['input_hashes'], **old['code_pins']}.items():
        track(path, raw=True)
        require(file_hash(Path(path)) == digest, 'L1 input or producer changed')
    sparse_release._producer_snapshots(parent, old, track)
    required = {str(ROOT/'src'/f'{name}.py') for name in
        ('episode_draft_revisit', 'episode_draft_release', 'episode_draft_requests',
         'episode_draft_native', 'whole_owner', 'whole_owner_requests')}
    if old.get('native_attempt_recovery'):
        required.add(str(ROOT/'src/episode_draft_recovery.py'))
    require(required <= set(old['code_pins']), 'Missing original L1 producer snapshots')
    for name in ('source.json', 'target.json', 'original-context.txt', 'source-pack.json',
                 'source-map.json', 'source-inheritance.json'):
        require(track(parent/name, raw=True) == track(folder/name, raw=True),
            'Copied L1 baseline or evidence differs')
    require(_source_pair(parent) == (source, target) and len(source) == len(target) == 66
        and _same(track(parent/'source-pack.json'), pack)
        and _same(track(parent/'source-map.json'), bound), 'Control replay inputs differ')
    request = draft_requests.build_writer_request(pack, bound)
    require(request['body']['focus_owner_ids'] == OWNER_IDS, 'Joint writer coverage changed')
    if old.get('native_attempt_recovery'):
        raw = recovery.replay_native(parent, request, track)
    else:
        raw = native.replay_native(parent/'L1/requests/episode-draft.json', request, track)
    require(set(raw) == {'owners'} and set(raw['owners']) == {str(i) for i in OWNER_IDS},
        'Original joint draft is incomplete')
    proposals = [wo.prepare_owner(i, raw['owners'][str(i)]['chinese'], source, target, pack)
                 for i in OWNER_IDS]
    require(all(t['valid'] and not t['no_op'] for t in proposals),
        'J1 requires all 66 valid non-noop original proposals')
    require(_same(track(parent/'L1/proposed-transactions.json'), proposals),
        'Saved original proposals differ from the sole native writer')
    full_draft = joint.assemble_draft(source, target, proposals, pack)
    expected = {'verify-' + t['transaction_id'] for t in proposals}
    actual = {path.stem for path in (parent/'L1/requests').glob('verify-*.json')}
    require(actual == expected, 'L1 native verifier stage coverage is not exactly 66')
    decisions = []
    for transaction in proposals:
        request = wr.build_verifier_request(pack, bound, source, target, transaction)
        verdict = treplay._native(parent/'L1/requests'/('verify-'+transaction['transaction_id']+'.json'),
            request['instruction'], request['body'], request['schema'], 'qwen', track)
        decisions.append({'transaction_id': transaction['transaction_id'],
            'accepted': wo.accept_owner_decision(verdict, transaction, pack), 'verdict': verdict})
    require(_same(track(parent/'L1/local-decisions.json'), decisions),
        'L1 control decisions differ from complete native replay')
    _control_native_ledger(parent, expected, track)
    return {'proposals': proposals, 'decisions': decisions, 'full_draft': full_draft}


def _control_native_ledger(parent, controls, track):
    """Every old request/telemetry attempt must belong to the sole L1 run."""
    directory = parent/'L1/requests'
    paths = {path.stem: path for path in directory.glob('*.json')}
    allowed = {'episode-draft', *controls, *(f'{prefix}-{i}'
        for prefix in ('global', 'assembled') for i in range(1, 12))}
    require(set(paths) <= allowed and {'episode-draft', *controls} <= set(paths),
        'Unaccounted L1 native request or extra writer')
    for prefix in ('global', 'assembled'):
        stages = {name for name in paths if name.startswith(prefix+'-')}
        require(not stages or stages == {f'{prefix}-{i}' for i in range(1, 12)},
            'Incomplete L1 global request stage set')
    rows = [strict_json(line) for line in track(directory/'metrics.jsonl', raw=True).decode('utf-8').splitlines()
            if line.strip()]
    require(all(isinstance(row, dict) and row.get('stage') in paths for row in rows),
        'Unaccounted L1 native telemetry stage')
    for name, path in sorted(paths.items()):
        saved = track(path); attempts = saved.get('attempts')
        require(isinstance(attempts, list) and 1 <= len(attempts) <= 2,
            'L1 request lacks its bounded attempt ledger')
        receipts = []
        for attempt in attempts:
            require(isinstance(attempt, dict) and isinstance(attempt.get('native_receipts'), list),
                'L1 request lacks native attempt receipts')
            receipts.extend(attempt['native_receipts'])
        require(_same([row for row in rows if row['stage'] == name], receipts),
            'L1 native telemetry includes hidden or unrecorded attempts')


def init(folder):
    started = time.monotonic(); folder = Path(folder).resolve()
    if (folder/'campaign.json').exists():
        return check(folder)
    old = previous.check(PREDECESSOR); _terminal_control(old)
    source, target = _source_pair(PREDECESSOR)
    baseline = er.validate_receipt(old['baseline_review'])
    require(baseline['purpose'] == 'baseline' and baseline['pool_version'] == old['pool_version']
        and baseline['inputs']['source']['sha256'] == er.owner_rows_sha256(source)
        and baseline['inputs']['target']['sha256'] == er.owner_rows_sha256(target)
        and baseline['inputs']['context']['sha256'] == old['context_sha256'], 'J1 baseline drift')
    require(len(source) == len(target) == 66, 'J1 requires exactly 66 original owners')
    folder.mkdir(parents=True, exist_ok=True)
    require(not any(folder.iterdir()), 'Unregistered J1 output exists')
    pins = {**old['input_hashes'], **old['code_pins'],
        str(PLAN): file_hash(PLAN), str(PREDECESSOR/'campaign.json'): file_hash(PREDECESSOR/'campaign.json')}
    for name in ('source.json', 'target.json', 'original-context.txt', 'source-pack.json',
                 'source-map.json', 'source-inheritance.json'):
        original = PREDECESSOR/name; copied = folder/name
        copied.write_bytes(original.read_bytes())
        pins[str(original)] = file_hash(original); pins[str(copied)] = file_hash(copied)
    prior = list(dict.fromkeys([*old['prior_score_paths'], str(PREDECESSOR/'L1/score.json')]))
    for name in prior:
        pins[name] = file_hash(Path(name))
    keys = ('media', 'media_sha256', 'writer_recipe', 'writer_recipe_sha256', 'evidence_paths',
        'pool_version', 'baseline_review', 'short_evidence_directory', 'starting_candidate',
        'predecessor', 'capacity_failure_predecessor', 'compact_capacity_failure_predecessor',
        'source_sha256', 'target_sha256', 'context_sha256', 'source_pack_sha256', 'map_sha256')
    conf = {key: deepcopy(old[key]) for key in keys}
    seconds = time.monotonic()-started
    conf.update(version=VERSION, status='prepared', created_utc=sr.now(), target=4, maximum_candidates=1,
        maximum_local_seconds=LIMIT, maximum_round_seconds=ROUND_LIMIT, local_seconds=seconds,
        rounds={ROUND: {'local_seconds': seconds, 'stages': [{'name': 'input_registration',
            'seconds': seconds, 'error': None, 'finished_utc': sr.now()}]}},
        plan=str(PLAN), plan_sha256=file_hash(PLAN), input_hashes=pins,
        control_predecessor=str(PREDECESSOR), reused_predecessor_local_seconds=old['local_seconds'],
        prior_score_paths=prior, external_feedback='score_only', source_edits_allowed=False,
        raw_source_fidelity_verified=False, human_reference_used=False, visual_input_used=False,
        new_audio_acquired=False, selected_output_modified=False, writer_sees_old_chinese=False,
        maximum_writer_groups=0, maximum_owner_checks=66, maximum_global_groups=11,
        maximum_rollback_groups=11, original_backend_restored=True)
    write(folder/'campaign.json', conf)
    try:
        prepare_controls(folder)
    except BaseException as exc:
        conf = read(folder/'campaign.json')
        conf.update(status='time_budget_exhausted' if isinstance(exc, rw.LocalBudgetExceeded)
            else 'failed', error_type=type(exc).__name__)
        write(folder/'campaign.json', conf)
        raise
    return check(folder)


def check(folder):
    folder = Path(folder).resolve(); conf = read(folder/'campaign.json')
    require(conf['version'] == VERSION and conf['maximum_candidates'] == 1
        and conf['maximum_local_seconds'] == LIMIT and conf['maximum_round_seconds'] == ROUND_LIMIT
        and conf['maximum_writer_groups'] == 0 and conf['maximum_owner_checks'] == 66
        and conf['maximum_global_groups'] == conf['maximum_rollback_groups'] == 11
        and conf['external_feedback'] == 'score_only' and conf['target'] == 4
        and all(conf[key] is False for key in ('source_edits_allowed', 'raw_source_fidelity_verified',
            'human_reference_used', 'visual_input_used', 'new_audio_acquired', 'selected_output_modified',
            'writer_sees_old_chinese')),
        'J1 scope drift')
    require(not any(key in conf for key in ('native_attempt_recovery', 'code_epochs')),
        'J1 must not adopt the predecessor recovery or runtime epoch')
    require(Path(conf['control_predecessor']).is_absolute()
        and Path(conf['control_predecessor']) != Path(conf['predecessor'])
        and conf['input_hashes'].get(str(Path(conf['control_predecessor'])/'campaign.json')),
        'J1 control ancestry missing')
    for name, digest in {**conf['input_hashes'], **conf.get('code_pins', {})}.items():
        require(file_hash(Path(name)) == digest, 'Pinned J1 input or producer changed')
    return conf


def freeze(folder):
    """Snapshot J1 and every current L1 producer only after all J1 files exist."""
    folder = Path(folder).resolve(); conf = check(folder)
    old = previous.check(Path(conf['control_predecessor']))
    paths = {*old['code_pins'], *(str(ROOT/'src'/f'{name}.py') for name in
        ('joint_context_revisit', 'joint_context_release', 'joint_context', 'episode_draft_recovery'))}
    pins = {path: file_hash(Path(path)) for path in sorted(paths)}
    manifest = {}
    for original, digest in pins.items():
        destination = folder/'producer-code'/Path(original).relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(original).read_bytes()); destination.chmod(0o444)
        manifest[original] = {'snapshot': str(destination), 'sha256': digest}
    write(folder/'producer-code/manifest.json', manifest)
    conf.update(code_pins=pins, producer_snapshot_manifest=str(folder/'producer-code/manifest.json'))
    write(folder/'campaign.json', conf)


@contextmanager
def measured(folder, round_id, stage):
    start = time.monotonic(); prior_deadline = sr._DEADLINE
    prior_handler = signal.getsignal(signal.SIGALRM); prior_timer = signal.getitimer(signal.ITIMER_REAL)
    armed = False; error = None
    def expired(signum, frame):
        raise rw.LocalBudgetExceeded('Context local stage deadline')
    try:
        conf = check(folder); record = conf['rounds'].setdefault(round_id, {'local_seconds': 0, 'stages': []})
        allowance = min(LIMIT-conf['local_seconds'], ROUND_LIMIT-record['local_seconds'])
        deadline = start + allowance; remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise rw.LocalBudgetExceeded('Context local budget exhausted during input checks')
        signal.signal(signal.SIGALRM, expired); armed = True
        sr._DEADLINE = deadline; signal.setitimer(signal.ITIMER_REAL, remaining)
        yield
    except BaseException as exc:
        error = type(exc).__name__; raise
    finally:
        elapsed = time.monotonic()-start
        if armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prior_handler); sr._DEADLINE = prior_deadline
            if prior_timer[0]:
                signal.setitimer(signal.ITIMER_REAL, max(.001, prior_timer[0]-elapsed), prior_timer[1])
        conf = read(folder/'campaign.json'); record = conf['rounds'].setdefault(round_id, {'local_seconds': 0, 'stages': []})
        record['local_seconds'] += elapsed; conf['local_seconds'] += elapsed
        record['stages'].append({'name': stage, 'seconds': elapsed, 'error': error, 'finished_utc': sr.now()})
        write(folder/'campaign.json', conf); LOG.info('%s %s %.2fs error=%s', round_id, stage, elapsed, error)


def prepare_controls(folder):
    """Persist only generation-free replay products; charge this work to J1."""
    folder = Path(folder).resolve(); conf = check(folder)
    with measured(folder, ROUND, 'l1_writer_and_control_native_replay'):
        pins = {}
        def track(path, raw=False):
            path = Path(path)
            require(not path.is_symlink(), 'Symlink cannot bind a control artifact')
            path = path.resolve(strict=True); digest = file_hash(path)
            require(str(path) not in pins or pins[str(path)] == digest, 'Control artifact changed during replay')
            pins[str(path)] = digest
            data = path.read_bytes()
            return data if raw else strict_json(data)
        source, target = _source_pair(folder)
        pack, bound = read(folder/'source-pack.json'), read(folder/'source-map.json')
        pools = [er._validate_pool(Path(path)) for path in conf['evidence_paths']]
        inherited_pack, inherited_bound, _, _ = inherited_gate._source_ancestry(
            folder, conf, source, target, pools, track)
        require(_same(pack, inherited_pack) and _same(bound, inherited_bound),
            'Copied T1 source evidence differs from full native ancestry')
        source_pins = dict(pins); pins.clear()
        controls = replay_controls(folder, conf, source, target, pack, bound, track)
        require(all(file_hash(Path(path)) == digest for path, digest in {**source_pins, **pins}.items()),
            'Source or control artifacts changed during preparation')
        out = folder/ROUND
        write(out/'proposed-transactions.json', controls['proposals'])
        write(out/'control-decisions.json', controls['decisions'])
        draft_rows = [asdict(row) for row in controls['full_draft']]
        write(out/'full-draft.json', draft_rows)
        receipt = {'control_predecessor': conf['control_predecessor'], 'native_artifacts': pins,
            'proposals_sha256': fingerprint(controls['proposals']),
            'decisions_sha256': fingerprint(controls['decisions']), 'full_draft_sha256': fingerprint(draft_rows)}
        write(folder/'control-inheritance.json', receipt)
        conf = check(folder)
        conf['input_hashes'].update(source_pins)
        conf['input_hashes'].update(pins)
        for path in (out/'proposed-transactions.json', out/'control-decisions.json', out/'full-draft.json',
                     folder/'control-inheritance.json'):
            conf['input_hashes'][str(path)] = file_hash(path)
        conf['control_decisions_sha256'] = receipt['decisions_sha256']
        write(folder/'campaign.json', conf)
    return controls


def inherited_source(folder):
    conf = check(folder)
    pack, bound = read(folder/'source-pack.json'), read(folder/'source-map.json')
    ec._check_pack(pack)
    require(pack['pack_sha256'] == conf['source_pack_sha256'] and bound['map_sha256'] == conf['map_sha256'],
        'Copied source evidence changed')
    return pack, bound


def prepared_controls(folder, source, target, pack):
    conf = check(folder); out = Path(folder)/ROUND
    proposals = read(out/'proposed-transactions.json'); decisions = read(out/'control-decisions.json')
    draft = joint.assemble_draft(source, target, proposals, pack)
    receipt = read(Path(folder)/'control-inheritance.json')
    require(all(t['valid'] and not t['no_op'] for t in proposals) and len(decisions) == 66
        and [d['transaction_id'] for d in decisions] == [t['transaction_id'] for t in proposals]
        and receipt['proposals_sha256'] == fingerprint(proposals)
        and receipt['decisions_sha256'] == conf['control_decisions_sha256'] == fingerprint(decisions)
        and receipt['full_draft_sha256'] == fingerprint([asdict(row) for row in draft])
        and _same(read(out/'full-draft.json'), [asdict(row) for row in draft]),
        'Prepared control replay changed')
    return {'proposals': proposals, 'decisions': decisions, 'full_draft': draft}


def comparison(controls, treatments, diffs):
    """Numeric paired outcomes are diagnostic, not an acceptance criterion."""
    require(len(controls) == len(treatments) == len(diffs) == 66
        and [d['transaction_id'] for d in controls] == [d['transaction_id'] for d in treatments],
        'Paired control/treatment coverage changed')
    pairs = [{'transaction_id': old['transaction_id'], 'control_accepted': old['accepted'],
              'treatment_accepted': new['accepted'], 'request_diff': diff}
             for old, new, diff in zip(controls, treatments, diffs)]
    counts = {'control_accepted': sum(p['control_accepted'] for p in pairs),
        'treatment_accepted': sum(p['treatment_accepted'] for p in pairs),
        'accepted_both': sum(p['control_accepted'] and p['treatment_accepted'] for p in pairs),
        'rejected_both': sum(not p['control_accepted'] and not p['treatment_accepted'] for p in pairs),
        'newly_accepted': sum(not p['control_accepted'] and p['treatment_accepted'] for p in pairs),
        'newly_rejected': sum(p['control_accepted'] and not p['treatment_accepted'] for p in pairs)}
    return {'control_decisions_sha256': fingerprint(controls),
        'treatment_decisions_sha256': fingerprint(treatments), 'pairs': pairs, 'counts': counts}


def check_global(folder,source,before,target,transactions,pack,bound,endpoint,prefix):
    result={};observations={o['observation_id']:o for o in pack['observations']}
    for number,ids in enumerate(sr.chunk_ids(source),1):
        request=wr.build_global_request(pack,bound,source,before,target,transactions,ids)
        value=sr.ask(folder/ROUND/'requests',f'{prefix}-{number}','qwen',endpoint,request['instruction'],request['body'],request['schema'],thinking=True)
        for owner,row in value.items():
            for finding in row['regressions']:
                witness=observations[finding['observation_id']]
                finding['quotes_valid']=bool(finding['transaction_ids'] and finding['before_quote'] and finding['after_quote'] and finding['source_quote']
                    and witness['owner_id']==int(owner) and finding['before_quote'] in before[int(owner)-1].text
                    and finding['after_quote'] in target[int(owner)-1].text and finding['source_quote'] in witness['text'])
        result.update(value)
    return result


def build_candidate(folder,pack,bound):
    conf=check(folder);source,target=_source_pair(folder);out=folder/ROUND;out.mkdir(parents=True,exist_ok=True)
    with measured(folder,ROUND,'paired_verifier_input_preparation'):
        controls=prepared_controls(folder,source,target,pack)
    proposals=controls['proposals'];decisions=[];accepted=[];rolled=[];diffs=[]
    with measured(folder,ROUND,'paired_complete_draft_context_verification'):
        with rw.backend('qwen',out/'checker-backend',Path(conf['writer_recipe'])) as (endpoint,_):
            for transaction in proposals:
                control=wr.build_verifier_request(pack,bound,source,target,transaction)
                request=joint.build_treatment_request(pack,bound,source,target,proposals,transaction['owner_id'])
                diffs.append(joint.assert_context_only_change(control,request,controls['full_draft']))
                verdict=sr.ask(out/'requests','verify-'+transaction['transaction_id'],'qwen',endpoint,
                    request['instruction'],request['body'],request['schema'],thinking=True)
                keep=wo.accept_owner_decision(verdict,transaction,pack)
                decisions.append({'transaction_id':transaction['transaction_id'],'accepted':keep,'verdict':verdict})
                if keep:accepted.append(transaction)
    write(out/'local-decisions.json',decisions)
    write(out/'control-comparison.json',comparison(controls['decisions'],decisions,diffs))
    current_source,current_target,ledger=wo.apply_owner_transactions(source,target,accepted,pack)
    if accepted:
        with measured(folder,ROUND,'all_owner_incremental_verification'):
            with rw.backend('qwen',out/'global-backend',Path(conf['writer_recipe'])) as (endpoint,_):
                checks=check_global(folder,source,target,current_target,accepted,pack,bound,endpoint,'global')
                rolled=sorted({identifier for row in checks.values() for f in row['regressions'] if f['quotes_valid'] for identifier in f['transaction_ids']})
                if rolled:
                    accepted=[t for t in accepted if t['transaction_id'] not in rolled]
                    current_source,current_target,ledger=wo.apply_owner_transactions(source,target,accepted,pack)
                    checks=check_global(folder,source,target,current_target,accepted,pack,bound,endpoint,'assembled') if accepted else {}
    else:checks={}
    if not accepted:checks={str(r.index):{'regressions':[],'legacy_or_source_uncertainty':'Unchanged retained baseline; no generated delta'} for r in source}
    gate=set(checks)==set(rw.rowmap(source)) and not any(f['quotes_valid'] for row in checks.values() for f in row['regressions'])
    require(current_source==source,'Whole owner candidate changed Japanese')
    write(out/'source.json',[asdict(r) for r in source]);write(out/'target.json',[asdict(r) for r in current_target])
    sr.write_checked(source,out/'source.utterances.srt');sr.write_checked(current_target,out/'target.utterances.srt')
    result={'version':VERSION,'round':ROUND,'source_sha256':file_hash(out/'source.json'),'target_sha256':file_hash(out/'target.json'),
        'owners_considered':len(proposals),'valid_proposals':sum(t['valid'] and not t['no_op'] for t in proposals),
        'accepted_transactions':accepted,'transaction_ledger':ledger,'rolled_back_transactions':rolled,'global_checks':checks,
        'local_gate_passed':gate,'coverage':len(checks),'changed_target_owners':[a.index for a,b in zip(current_target,target) if a.text!=b.text],
        'writer':'reused_l1_qwen','checker':'qwen','control_predecessor':conf['control_predecessor'],
        'control_decisions_sha256':fingerprint(controls['decisions']),'writer_sees_old_chinese':False,'evidence_paths':conf['evidence_paths'],
        'pack_sha256':pack['pack_sha256'],'map_sha256':bound['map_sha256'],'raw_source_fidelity_verified':False,
        'external_feedback_used':False,'human_reference_used':False,'source_edits_allowed':False}
    write(out/'candidate.json',result);return result


def validate_candidate_artifacts(folder):
    conf=check(folder);source,target=_source_pair(folder);out=folder/ROUND;candidate=read(out/'candidate.json')
    require(candidate['version']==VERSION and candidate['round']==ROUND and candidate['source_edits_allowed'] is False
        and candidate['writer_sees_old_chinese'] is False,'J1 candidate scope changed')
    pack=read(folder/'source-pack.json');ec._check_pack(pack)
    require(candidate['pack_sha256']==conf['source_pack_sha256']==pack['pack_sha256']
        and candidate['map_sha256']==conf['map_sha256'],'J1 candidate evidence changed')
    result_source,result_target,ledger=wo.apply_owner_transactions(source,target,candidate['accepted_transactions'],pack)
    require(ledger==candidate['transaction_ledger'] and result_source==source
        and rw.rows(out/'source.json')==source and rw.rows(out/'target.json')==result_target
        and file_hash(out/'source.json')==candidate['source_sha256'] and file_hash(out/'target.json')==candidate['target_sha256'], 'J1 candidate artifact changed')
    require(candidate['changed_target_owners']==[a.index for a,b in zip(result_target,target) if a.text!=b.text],'J1 change ledger drift')
    if candidate['changed_target_owners']:
        from src import sparse_release
        sparse_release.display_checks._display(out/'display',source,result_target)
    return candidate


def score(folder,baseline):
    conf=check(folder);out=folder/ROUND;candidate=validate_candidate_artifacts(folder);source,target=_source_pair(out)
    target_hash=er.owner_rows_sha256(target);result_path=out/'score.json'
    if target_hash==er.owner_rows_sha256(rw.rows(folder/'target.json')):
        result={'status':'complete','score':baseline['score'],'inherited_unchanged_baseline':True,'confirmed':False,'pool_version':baseline['pool_version'],'target_sha256':target_hash}
        write(result_path,result);return result
    for name in conf['prior_score_paths']:
        prior=read(name)
        if prior.get('status')=='complete' and prior.get('pool_version')==baseline['pool_version'] and prior.get('target_sha256')==target_hash:
            result={'status':'complete','score':min(prior['score'],prior.get('confirmation_score',prior['score'])),
                'inherited_duplicate_target':name,'confirmed':False,'pool_version':baseline['pool_version'],'target_sha256':target_hash}
            write(result_path,result);return result
    manifest=out/'evaluation/initial/bundle/manifest.json'
    if not manifest.exists():er.prepare_bundle(source,target,rw.rows(folder/'source.json'),list(map(Path,conf['evidence_paths'])),folder/'original-context.txt',manifest.parent)
    bundle=er.read_bundle(manifest)
    require(bundle['pool_version']==baseline['pool_version'] and bundle['inputs']['target']['sha256']==target_hash
        and bundle['inputs']['source']['sha256']==er.owner_rows_sha256(source),'J1 score bundle drift')
    primary=sr.review_once(manifest,out/'evaluation/initial/primary',purpose='candidate',baseline_review=baseline['output_dir'])
    result={'status':'primary_complete','score':primary['score'],'primary':primary['output_dir'],'pool_version':primary['pool_version'],
        'target_sha256':target_hash,'source_sha256':primary['source_sha256'],'confirmed':False,'local_gate_passed':candidate['local_gate_passed']}
    write(result_path,result)
    if primary['score']>=4 and candidate['local_gate_passed']:
        confirmation=sr.review_once(manifest,out/'evaluation/initial/confirmation',purpose='confirmation',primary_review=primary['output_dir'])
        result.update(confirmation_score=confirmation['score'],confirmation=confirmation['output_dir'],confirmed=confirmation['score']>=4)
    result['status']='complete';write(result_path,result);LOG.info('J1 independent score=%s confirmed=%s',result['score'],result['confirmed']);return result


def _restore(folder,previous):
    start=time.monotonic();error=None;ok=False
    try:
        state=rw._request_json(rw.ADMIN+'/status',admin=True)
        if late_audio._backend_identity(state)!=previous:
            rw._request_json(rw.ADMIN+('/unload' if previous is None else '/load'),{} if previous is None else {'model':previous},admin=True,timeout=360)
        ok=late_audio._backend_identity(rw._request_json(rw.ADMIN+'/status',admin=True))==previous
        require(ok,'Original backend restoration failed')
    except BaseException as exc:error=type(exc).__name__;raise
    finally:
        conf=read(folder/'campaign.json');seconds=time.monotonic()-start
        conf['local_seconds']+=seconds;conf['rounds'][ROUND]['local_seconds']+=seconds
        conf['rounds'][ROUND]['stages'].append({'name':'original_backend_restoration','seconds':seconds,'error':error,'finished_utc':sr.now()})
        conf.update(original_backend_restored=ok,finished_utc=sr.now())
        if not ok:
            conf.update(status='failed',restoration_error=error);conf.pop('eligible_candidate',None)
        elif conf['local_seconds']>=LIMIT:
            conf['status']='time_budget_exhausted';conf.pop('eligible_candidate',None)
        write(folder/'campaign.json',conf)


def execute(folder):
    folder=Path(folder).resolve();conf=init(folder)
    if conf['status']!='prepared':return conf
    setup_start=time.monotonic()
    freeze(folder);conf=check(folder);initial=rw._request_json(rw.ADMIN+'/status',admin=True);previous=late_audio._backend_identity(initial)
    conf.update(status='running',active_round=ROUND,started_utc=sr.now(),original_backend_initial=initial,original_backend_restored=False)
    seconds=time.monotonic()-setup_start;conf['local_seconds']+=seconds
    record=conf['rounds'].setdefault(ROUND,{'local_seconds':0.,'stages':[]})
    record['status']='reserved';record['local_seconds']+=seconds
    record['stages'].append({'name':'input_and_producer_preparation','seconds':seconds,'error':None,'finished_utc':sr.now()})
    write(folder/'campaign.json',conf);primary_error=None
    try:
        pack,bound=inherited_source(folder);candidate=build_candidate(folder,pack,bound);conf=check(folder)
        if candidate['changed_target_owners']:
            directory=folder/ROUND/'display';directory.mkdir(exist_ok=True)
            with measured(folder,ROUND,'exact_text_display'):
                with rw.backend('qwen',folder/ROUND/'display-backend',Path(conf['writer_recipe'])):
                    sr._display(rw.rows(folder/ROUND/'source.json'),rw.rows(folder/ROUND/'target.json'),directory)
        if check(folder)['local_seconds']>=LIMIT:raise rw.LocalBudgetExceeded('J1 budget exhausted before scoring')
        result=score(folder,er.validate_receipt(conf['baseline_review']));conf=check(folder)
        conf['rounds'][ROUND].update(status='scored',score=result['score'],confirmed=result['confirmed'])
        if result['confirmed']:conf.update(status='confirmed_four',eligible_candidate=ROUND)
        else:conf['status']='candidate_budget_exhausted'
        write(folder/'campaign.json',conf)
    except BaseException as exc:
        primary_error=exc;conf=read(folder/'campaign.json')
        conf.update(status='time_budget_exhausted' if isinstance(exc,rw.LocalBudgetExceeded) else 'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error_type=type(exc).__name__)
        write(folder/'campaign.json',conf);raise
    finally:
        try:_restore(folder,previous)
        except BaseException as exc:
            if primary_error is None:raise
            primary_error.add_note('J1 restoration failed: '+type(exc).__name__)
    return check(folder)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,default=ROOT/'output/quality-joint-context-20260915')
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    if args.execute:
        import fcntl
        args.campaign.parent.mkdir(parents=True,exist_ok=True)
        with args.campaign.with_suffix('.execution.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);conf=execute(args.campaign)
    else:conf=init(args.campaign)
    print(json.dumps({k:conf.get(k) for k in ('status','local_seconds','eligible_candidate','original_backend_restored')}))

if __name__=='__main__':main()
