"""Conditional R1: one readable-evidence Qwen draft, original L1 owner checks."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
import signal
import time
from copy import deepcopy
from src import whole_owner as wo
from src import readable_draft_requests as wr
from src import readable_draft_native as native
from src import readable_evidence as readable
from src import readable_identity as identity
from src import episode_draft_revisit as l1
from src import joint_context_revisit as j1
from src import temporal_revisit as tr
from src import temporal_release as treplay
from src import context_revisit as cr
from src import context_release as replay
from src import sparse_revisit as sr
from src import revisit_workflow as rw
from src import late_audio
from src import evidence_context as ec
from src import contextual_evidence_review as er
from src.contextual_review_contract import require, strict_json
from src.workflow_state import file_hash,fingerprint
ROOT=Path(__file__).resolve().parents[1]
PREDECESSOR=ROOT/'output/quality-temporal-20260915'
PLAN=ROOT/'design/quality-next-directions-20260915.md'
L1_PREDECESSOR=ROOT/'output/quality-episode-draft-20260915'
J1_PREDECESSOR=ROOT/'output/quality-joint-context-20260915'
VERSION='readable-draft-local-revisit-1'
ROUND='R1'
WRITER_REQUEST_KEY='readable-draft'
LIMIT=ROUND_LIMIT=7200
LOG=logging.getLogger(__name__)
read,write,_source_pair,_cached,_save=cr.read,cr.write,cr._source_pair,cr._cached,cr._save


TERMINAL_UNQUALIFIED = {'candidate_budget_exhausted', 'time_budget_exhausted', 'failed', 'interrupted'}


def _terminal_unqualified(conf, label):
    require(conf.get('status') in TERMINAL_UNQUALIFIED
        and conf.get('original_backend_restored') is True and not conf.get('eligible_candidate'),
        label+' must be terminal, restored and unqualified')


def _tracker(pins):
    def track(path, raw=False):
        path = Path(path)
        require(not path.is_symlink(), 'Symlink cannot bind an R1 input')
        path = path.resolve(strict=True); digest = file_hash(path)
        require(str(path) not in pins or pins[str(path)] == digest, 'R1 input changed during preparation')
        pins[str(path)] = digest
        data = path.read_bytes()
        require(file_hash(path) == digest, 'R1 input changed while reading')
        return data if raw else strict_json(data)
    return track


def joint_eligibility(folder, old, source, target, pack, bound, pins):
    """Pin a completed J1 or reproduce its declared generation-free control failure."""
    folder = Path(folder).resolve(); conf = j1.check(folder); _terminal_unqualified(conf, 'J1')
    track = _tracker(pins); require(track(folder/'campaign.json') == conf, 'J1 campaign changed')
    require(conf['control_predecessor'] == str(L1_PREDECESSOR)
        and all(conf[key] == old[key] for key in
            ('source_sha256', 'target_sha256', 'context_sha256', 'pool_version', 'source_pack_sha256', 'map_sha256')),
        'J1 predecessor inputs differ from L1')
    stages = conf.get('rounds', {}).get('J1', {}).get('stages', [])
    failures = [row for row in stages if row.get('name') == 'l1_writer_and_control_native_replay' and row.get('error')]
    if not failures:
        require(any(row.get('name') == 'l1_writer_and_control_native_replay' and row.get('error') is None for row in stages)
            and conf.get('control_decisions_sha256'), 'J1 never completed its control prerequisite')
        for path, digest in {**conf['input_hashes'], **conf.get('code_pins', {})}.items():
            require(file_hash(Path(path)) == digest, 'J1 pinned input changed'); pins[path] = digest
        return {'status': 'terminal_unqualified', 'campaign_sha256': pins[str(folder/'campaign.json')]}
    require(conf['status'] == 'failed' and len(failures) == 1
        and conf.get('error_type') == failures[0]['error'], 'J1 failure is not its sole native control preparation')
    receipt = track(folder/'ineligibility.json')
    require(set(receipt) == {'version', 'control_predecessor', 'predecessor_campaign_sha256', 'reason_code',
        'preparation_error_type', 'native_artifacts', 'local_seconds', 'inference_dispatched', 'generated_at'}
        and receipt['version'] == 'joint-context-ineligibility-1'
        and receipt['control_predecessor'] == str(L1_PREDECESSOR)
        and receipt['predecessor_campaign_sha256'] == file_hash(L1_PREDECESSOR/'campaign.json')
        and receipt['reason_code'] == 'native_control_replay_failed'
        and receipt['preparation_error_type'] == conf['error_type']
        and receipt['inference_dispatched'] is False
        and type(receipt['local_seconds']) in (int, float) and math.isfinite(receipt['local_seconds'])
        and receipt['local_seconds'] == conf['local_seconds']
        and isinstance(receipt['generated_at'], str) and receipt['generated_at']
        and isinstance(receipt['native_artifacts'], dict), 'Invalid J1 ineligibility record')
    require(not list((folder/'J1/requests').glob('*'))
        and not any((folder/'J1'/name).exists() for name in ('candidate.json', 'target.json', 'score.json')),
        'Ineligible J1 already dispatched or generated a candidate')
    native_pins = {}; replay_track = _tracker(native_pins)
    try:
        j1.replay_controls(folder, conf, source, target, pack, bound, replay_track)
    except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
        require(type(exc).__name__ == receipt['preparation_error_type'], 'J1 control failure no longer reproduces')
    else:
        raise ValueError('J1 controls replay; ineligibility claim is unsupported')
    required = {str(folder/'campaign.json'): file_hash(folder/'campaign.json'),
        str(L1_PREDECESSOR/'campaign.json'): file_hash(L1_PREDECESSOR/'campaign.json'), **native_pins}
    require(all(receipt['native_artifacts'].get(path) == digest for path, digest in required.items()),
        'J1 failure ancestry missing')
    for path, digest in receipt['native_artifacts'].items():
        require(file_hash(Path(path)) == digest, 'J1 failure artifact changed'); pins[path] = digest
    return {'status': 'native_control_replay_failed', 'campaign_sha256': required[str(folder/'campaign.json')],
        'ineligibility_sha256': pins[str(folder/'ineligibility.json')]}


def init(folder):
    started = time.monotonic(); folder = Path(folder).resolve()
    if (folder/'campaign.json').exists(): return check(folder)
    old = l1.check(L1_PREDECESSOR); _terminal_unqualified(old, 'L1')
    require(Path(old['predecessor']) == PREDECESSOR, 'R1 must inherit the declared T1 source evidence')
    source, target = _source_pair(L1_PREDECESSOR)
    require(len(source) == len(target) == 66, 'R1 requires exactly 66 original owners')
    baseline = er.validate_receipt(old['baseline_review'])
    require(baseline['purpose'] == 'baseline' and baseline['pool_version'] == old['pool_version']
        and baseline['score'] == 2
        and baseline['inputs']['source']['sha256'] == er.owner_rows_sha256(source)
        and baseline['inputs']['target']['sha256'] == er.owner_rows_sha256(target)
        and baseline['inputs']['context']['sha256'] == old['context_sha256'], 'R1 baseline drift')
    pins = {**old['input_hashes'], **old['code_pins'], str(PLAN): file_hash(PLAN),
        str(L1_PREDECESSOR/'campaign.json'): file_hash(L1_PREDECESSOR/'campaign.json'),
        str(PREDECESSOR/'campaign.json'): file_hash(PREDECESSOR/'campaign.json')}
    pack, bound = read(L1_PREDECESSOR/'source-pack.json'), read(L1_PREDECESSOR/'source-map.json')
    eligibility = joint_eligibility(J1_PREDECESSOR, old, source, target, pack, bound, pins)
    folder.mkdir(parents=True, exist_ok=True); require(not any(folder.iterdir()), 'Unregistered R1 output exists')
    for name in ('source.json', 'target.json', 'original-context.txt'):
        original = L1_PREDECESSOR/name; copied = folder/name
        copied.write_bytes(original.read_bytes()); pins[str(original)] = file_hash(original); pins[str(copied)] = file_hash(copied)
    prior = list(old.get('prior_score_paths', []))
    for parent, rid in ((L1_PREDECESSOR, 'L1'), (J1_PREDECESSOR, 'J1')):
        path = parent/rid/'score.json'
        if path.exists(): prior.append(str(path))
    prior = list(dict.fromkeys(prior))
    for name in prior: pins[name] = file_hash(Path(name))
    conf = {key: deepcopy(old[key]) for key in ('media', 'media_sha256', 'writer_recipe', 'writer_recipe_sha256',
        'evidence_paths', 'pool_version', 'baseline_review', 'short_evidence_directory', 'starting_candidate',
        'capacity_failure_predecessor', 'compact_capacity_failure_predecessor')}
    seconds = time.monotonic()-started
    conf.update(version=VERSION, status='prepared', created_utc=sr.now(), target=4, maximum_candidates=1,
        maximum_local_seconds=LIMIT, maximum_round_seconds=ROUND_LIMIT, local_seconds=seconds,
        preparation_seconds=seconds, rounds={ROUND: {'local_seconds': seconds, 'stages': [
            {'name': 'input_registration', 'seconds': seconds, 'error': None, 'finished_utc': sr.now()}]}},
        plan=str(PLAN), plan_sha256=file_hash(PLAN), input_hashes=pins, predecessor=str(PREDECESSOR),
        l1_predecessor=str(L1_PREDECESSOR), joint_predecessor=str(J1_PREDECESSOR), joint_eligibility=eligibility,
        reused_predecessor_local_seconds=old['local_seconds'],
        source_sha256=file_hash(folder/'source.json'), target_sha256=file_hash(folder/'target.json'),
        context_sha256=file_hash(folder/'original-context.txt'), prior_score_paths=prior, external_feedback='score_only',
        source_edits_allowed=False, raw_source_fidelity_verified=False, human_reference_used=False,
        visual_input_used=False, new_audio_acquired=False, selected_output_modified=False, writer_sees_old_chinese=False,
        maximum_writer_groups=1, maximum_owner_checks=66, maximum_global_groups=11, maximum_rollback_groups=11,
        verifier_context='original_l1_baseline', original_backend_restored=True)
    write(folder/'campaign.json', conf)
    try:
        pack, bound = inherited_source(folder)
        prepare_readable_evidence(folder, pack, bound)
    except BaseException as exc:
        conf = read(folder/'campaign.json')
        conf.update(status='time_budget_exhausted' if isinstance(exc, rw.LocalBudgetExceeded) else
            'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error_type=type(exc).__name__)
        write(folder/'campaign.json', conf); raise
    conf = read(folder/'campaign.json'); conf['preparation_seconds'] = conf['local_seconds']
    write(folder/'campaign.json', conf)
    return check(folder)


def check(folder):
    folder = Path(folder).resolve(); conf = read(folder/'campaign.json')
    require(conf['version'] == VERSION and conf['target'] == 4 and conf['maximum_candidates'] == 1
        and conf['maximum_local_seconds'] == conf['maximum_round_seconds'] == LIMIT
        and conf['maximum_writer_groups'] == 1 and conf['maximum_owner_checks'] == 66
        and conf['maximum_global_groups'] == conf['maximum_rollback_groups'] == 11
        and conf['external_feedback'] == 'score_only' and conf['verifier_context'] == 'original_l1_baseline'
        and all(conf[key] is False for key in ('source_edits_allowed', 'raw_source_fidelity_verified',
            'human_reference_used', 'visual_input_used', 'new_audio_acquired', 'selected_output_modified', 'writer_sees_old_chinese')),
        'R1 scope drift')
    require(not any(key in conf for key in ('native_attempt_recovery', 'code_epochs')), 'R1 cannot adopt a runtime recovery')
    for key in ('predecessor', 'l1_predecessor', 'joint_predecessor'):
        path = Path(conf[key]); require(path.is_absolute() and str(path/'campaign.json') in conf['input_hashes'], 'R1 ancestry missing')
    for name, digest in {**conf['input_hashes'], **conf.get('code_pins', {})}.items():
        require(file_hash(Path(name)) == digest, 'Pinned R1 input or producer changed')
    return conf


def freeze(folder):
    folder = Path(folder).resolve(); conf = check(folder)
    old = l1.check(Path(conf['l1_predecessor']))
    paths = {*old['code_pins'], *(str(ROOT/'src'/f'{name}.py') for name in
        ('readable_draft_revisit', 'readable_draft_release', 'readable_draft_requests', 'readable_draft_native',
         'readable_evidence', 'readable_identity', 'episode_draft_native', 'episode_draft_requests',
         'joint_context_revisit', 'joint_context_release', 'joint_context', 'episode_draft_recovery'))}
    pins = {path: file_hash(Path(path)) for path in sorted(paths)}; manifest = {}
    for original, digest in pins.items():
        destination = folder/'producer-code'/Path(original).relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(Path(original).read_bytes())
        destination.chmod(0o444); manifest[original] = {'snapshot': str(destination), 'sha256': digest}
    write(folder/'producer-code/manifest.json', manifest)
    conf.update(code_pins=pins, producer_snapshot_manifest=str(folder/'producer-code/manifest.json'))
    write(folder/'campaign.json', conf)


def _charge_stage(folder, round_id, stage, elapsed, error):
    conf = read(folder/'campaign.json'); record = conf['rounds'].setdefault(round_id, {'local_seconds': 0, 'stages': []})
    record['local_seconds'] += elapsed; conf['local_seconds'] += elapsed
    record['stages'].append({'name': stage, 'seconds': elapsed, 'error': error, 'finished_utc': sr.now()})
    write(folder/'campaign.json', conf); LOG.info('%s %s %.2fs error=%s', round_id, stage, elapsed, error)


@contextmanager
def measured(folder, round_id, stage):
    start = time.monotonic(); prior_deadline = sr._DEADLINE
    prior_handler = signal.getsignal(signal.SIGALRM); prior_timer = signal.getitimer(signal.ITIMER_REAL)
    armed = False; error = None
    def expired(signum, frame):
        raise rw.LocalBudgetExceeded('R1 local stage deadline')
    try:
        conf = check(folder); record = conf['rounds'].setdefault(round_id, {'local_seconds': 0, 'stages': []})
        allowance = min(LIMIT-conf['local_seconds'], ROUND_LIMIT-record['local_seconds'])
        deadline = start + allowance; remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise rw.LocalBudgetExceeded('R1 local budget exhausted during input verification')
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
        _charge_stage(folder, round_id, stage, elapsed, error)


def inherited_source(folder):
    conf=check(folder);parent=Path(conf['predecessor']);old=tr.check(parent)
    with measured(folder,ROUND,'native_source_frames_and_recap_replay'):
        pins={}
        def track(path,raw=False):
            path=Path(path).resolve();pins[str(path)]=file_hash(path)
            return path.read_bytes() if raw else read(path)
        from src import sparse_release
        sparse_release._producer_snapshots(parent,old,track)
        for name,digest in old['code_pins'].items():require(file_hash(Path(name))==digest,'T1 producer changed before source inheritance')
        source,target=_source_pair(folder);obs=sr.all_observations(list(map(Path,conf['evidence_paths'])),source)
        context=(folder/'original-context.txt').read_text(encoding='utf-8')
        frames=treplay._source_frames(parent,old,source,obs,context,track)['frames']
        pools=[er._validate_pool(Path(p)) for p in conf['evidence_paths']]
        pack=ec.build_source_pack(source,obs,frames,context,pool_versions=[p['pool_version'] for p in pools])
        require(pack==track(parent/'source-pack.json'),'Inherited T1 source pack differs')
        prepared=tr.prepare_recap_request(pack)
        binding={'pack_sha256':pack['pack_sha256'],'request':fingerprint({k:prepared[k] for k in ('instruction','body','schema')})}
        bound=replay._bound(parent/'shared/source-map.json',binding,track)
        raw=treplay._native(parent/'T1/requests'/(tr.RECAP_REQUEST_KEY+'.json'),prepared['instruction'],prepared['body'],prepared['schema'],'qwen',track)
        require(bound==tr.bind_recap(raw,pack),'Inherited T1 recap differs')
        require(all(file_hash(Path(p))==digest for p,digest in pins.items()),'Source inheritance changed while replaying')
        write(folder/'source-pack.json',pack);write(folder/'source-map.json',bound)
        write(folder/'source-inheritance.json',{'predecessor':str(parent),'native_artifacts':pins,'pack_sha256':pack['pack_sha256'],'map_sha256':bound['map_sha256']})
        conf=check(folder);conf.update(source_pack_sha256=pack['pack_sha256'],map_sha256=bound['map_sha256'])
        for path in [folder/'source-pack.json',folder/'source-map.json',folder/'source-inheritance.json']:
            conf['input_hashes'][str(path)]=file_hash(path)
        conf['input_hashes'].update(pins);write(folder/'campaign.json',conf)
    return pack,bound


def identity_inputs(pack, evidence_paths, track, pins):
    """Capture every receipt and file hash used by the native identity validator."""
    def native_track(path, raw=False):
        path = Path(path).resolve(strict=True); digest = file_hash(path)
        require(str(path) not in pins or pins[str(path)] == digest, 'Native identity artifact changed')
        value = track(path, raw=raw)
        require(file_hash(path) == digest, 'Native identity artifact changed while reading')
        pins[str(path)] = digest
        return value
    identities = identity.load_identity_map(pack, evidence_paths, native_track)
    for path in evidence_paths:
        pool = native_track(path); plan = native_track(pool['plan_path']); prep = native_track(plan['preparation']['path'])
        files = [prep['master']['media'], prep['master']['master']]
        for observation in pool['observations']:
            native_track(observation['receipt_path'])
            require(file_hash(Path(observation['receipt_path'])) == observation['receipt_sha256'], 'Native identity receipt changed')
            if observation['status'] in ('ok', 'empty'):
                files.append({'path': observation['provenance']['wav_path'], 'sha256': observation['provenance']['wav_sha256']})
        for item in files:
            path = Path(item['path']).resolve(strict=True)
            require(file_hash(path) == item['sha256'], 'Native identity media changed')
            pins[str(path)] = item['sha256']
    return identities


def prepare_readable_evidence(folder, pack, bound):
    folder = Path(folder).resolve(); conf = check(folder)
    with measured(folder, ROUND, 'readable_identity_projection_preparation'):
        pins = {}; track = _tracker(pins)
        identities = identity_inputs(pack, conf['evidence_paths'], track, pins)
        projection = readable.build_projection(pack, bound, identities)
        counts = readable.validate_projection(projection, pack, bound, identities)
        reconstruction = readable.reconstruct_projection(projection['body'], projection['audit'])
        rendered = readable.render_prompt_body(projection)
        readable.validate_prompt_body(rendered, projection)
        require(counts['source_owners'] == 66 and counts['observations'] == len(pack['observations']), 'R1 readable coverage differs')
        require(all(file_hash(Path(path)) == digest for path, digest in pins.items()), 'Identity ancestry changed during preparation')
        for name, value in (('identity-map.json', identities), ('projection.json', projection),
            ('canonical-reconstruction.json', reconstruction), ('rendered-body.json', rendered)):
            write(folder/'readable'/name, value)
        receipt = {'version': 'readable-draft-preparation-1', 'native_artifacts': pins,
            'pack_sha256': pack['pack_sha256'], 'map_sha256': bound['map_sha256'],
            'identity_map_sha256': fingerprint(identities), 'projection_sha256': fingerprint(projection),
            'canonical_body_sha256': fingerprint(reconstruction), 'rendered_body_sha256': fingerprint(rendered),
            'coverage': counts, 'inference_dispatched': False}
        write(folder/'readable/preparation.json', receipt)
        conf = check(folder); conf['input_hashes'].update(pins)
        for name in ('identity-map.json', 'projection.json', 'canonical-reconstruction.json', 'rendered-body.json', 'preparation.json'):
            path = folder/'readable'/name; conf['input_hashes'][str(path)] = file_hash(path)
        conf.update(readable_projection_sha256=receipt['projection_sha256'], identity_map_sha256=receipt['identity_map_sha256'])
        write(folder/'campaign.json', conf)
    return identities, projection


def prepared_readable(folder, pack, bound):
    folder = Path(folder).resolve(); conf = check(folder)
    identities = read(folder/'readable/identity-map.json'); projection = read(folder/'readable/projection.json')
    receipt = read(folder/'readable/preparation.json')
    counts = readable.validate_projection(projection, pack, bound, identities)
    reconstructed = readable.reconstruct_projection(projection['body'], projection['audit'])
    rendered = readable.render_prompt_body(projection)
    readable.validate_prompt_body(read(folder/'readable/rendered-body.json'), projection)
    require(receipt['version'] == 'readable-draft-preparation-1' and receipt['inference_dispatched'] is False
        and receipt['coverage'] == counts and receipt['pack_sha256'] == pack['pack_sha256']
        and receipt['map_sha256'] == bound['map_sha256']
        and receipt['identity_map_sha256'] == conf['identity_map_sha256'] == fingerprint(identities)
        and receipt['projection_sha256'] == conf['readable_projection_sha256'] == fingerprint(projection)
        and receipt['canonical_body_sha256'] == fingerprint(reconstructed)
        and receipt['rendered_body_sha256'] == fingerprint(rendered)
        and read(folder/'readable/canonical-reconstruction.json') == reconstructed
        and all(conf['input_hashes'].get(path) == digest for path, digest in receipt['native_artifacts'].items()),
        'Prepared readable evidence changed')
    return identities, projection


def writer_request(pack, bound, identity_map, projection):
    return wr.build_writer_request(pack, bound, identity_map, projection=projection)


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
    proposals=[];decisions=[];accepted=[];rolled=[]
    with measured(folder, ROUND, 'readable_writer_request_preparation'):
        identities, projection = prepared_readable(folder, pack, bound)
        request = writer_request(pack, bound, identities, projection)
    with measured(folder,ROUND,'source_only_readable_episode_generation'):
        with rw.backend('qwen',out/'writer-backend',Path(conf['writer_recipe'])) as (endpoint,_):
            raw=native.ask(out/'requests',WRITER_REQUEST_KEY,endpoint,request)
    for owner in request['body']['focus_owner_ids']:
        proposals.append(wo.prepare_owner(owner,raw['owners'][str(owner)]['chinese'],source,target,pack))
    require([t['owner_id'] for t in proposals]==list(range(1,67)),'Whole owner generation coverage changed')
    write(out/'proposed-transactions.json',proposals)
    with measured(folder,ROUND,'whole_owner_material_improvement_verification'):
        with rw.backend('qwen',out/'checker-backend',Path(conf['writer_recipe'])) as (endpoint,_):
            for transaction in proposals:
                if not transaction['valid'] or transaction['no_op']:continue
                request=wr.build_verifier_request(pack,bound,source,target,transaction)
                verdict=sr.ask(out/'requests','verify-'+transaction['transaction_id'],'qwen',endpoint,
                    request['instruction'],request['body'],request['schema'],thinking=True)
                keep=wo.accept_owner_decision(verdict,transaction,pack)
                decisions.append({'transaction_id':transaction['transaction_id'],'accepted':keep,'verdict':verdict})
                if keep:accepted.append(transaction)
    write(out/'local-decisions.json',decisions)
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
        'writer':'qwen','checker':'qwen','writer_sees_old_chinese':False,'evidence_paths':conf['evidence_paths'],
        'pack_sha256':pack['pack_sha256'],'map_sha256':bound['map_sha256'],'readable_projection_sha256':conf['readable_projection_sha256'],
        'identity_map_sha256':conf['identity_map_sha256'],'verifier_context':'original_l1_baseline','raw_source_fidelity_verified':False,
        'external_feedback_used':False,'human_reference_used':False,'source_edits_allowed':False}
    write(out/'candidate.json',result);return result


def validate_candidate_artifacts(folder):
    conf=check(folder);source,target=_source_pair(folder);out=folder/ROUND;candidate=read(out/'candidate.json')
    require(candidate['version']==VERSION and candidate['round']==ROUND and candidate['source_edits_allowed'] is False
        and candidate['writer_sees_old_chinese'] is False
        and candidate['verifier_context']=='original_l1_baseline'
        and candidate['readable_projection_sha256']==conf['readable_projection_sha256']
        and candidate['identity_map_sha256']==conf['identity_map_sha256'],'R1 candidate scope changed')
    pack=read(folder/'source-pack.json');ec._check_pack(pack)
    require(candidate['pack_sha256']==conf['source_pack_sha256']==pack['pack_sha256']
        and candidate['map_sha256']==conf['map_sha256'],'R1 candidate evidence changed')
    result_source,result_target,ledger=wo.apply_owner_transactions(source,target,candidate['accepted_transactions'],pack)
    require(ledger==candidate['transaction_ledger'] and result_source==source
        and rw.rows(out/'source.json')==source and rw.rows(out/'target.json')==result_target
        and file_hash(out/'source.json')==candidate['source_sha256'] and file_hash(out/'target.json')==candidate['target_sha256'], 'R1 candidate artifact changed')
    require(candidate['changed_target_owners']==[a.index for a,b in zip(result_target,target) if a.text!=b.text],'R1 change ledger drift')
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
        and bundle['inputs']['source']['sha256']==er.owner_rows_sha256(source),'R1 score bundle drift')
    primary=sr.review_once(manifest,out/'evaluation/initial/primary',purpose='candidate',baseline_review=baseline['output_dir'])
    result={'status':'primary_complete','score':primary['score'],'primary':primary['output_dir'],'pool_version':primary['pool_version'],
        'target_sha256':target_hash,'source_sha256':primary['source_sha256'],'confirmed':False,'local_gate_passed':candidate['local_gate_passed']}
    write(result_path,result)
    if primary['score']>=4 and candidate['local_gate_passed']:
        confirmation=sr.review_once(manifest,out/'evaluation/initial/confirmation',purpose='confirmation',primary_review=primary['output_dir'])
        result.update(confirmation_score=confirmation['score'],confirmation=confirmation['output_dir'],confirmed=confirmation['score']>=4)
    result['status']='complete';write(result_path,result);LOG.info('R1 independent score=%s confirmed=%s',result['score'],result['confirmed']);return result


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
    folder = Path(folder).resolve(); conf = init(folder)
    if conf['status'] != 'prepared': return conf
    # Registration/source replay/projection were charged by init, even when run separately.
    conf.update(status='running', active_round=ROUND, started_utc=sr.now())
    conf['rounds'][ROUND]['status'] = 'reserved'; write(folder/'campaign.json', conf)
    previous_backend = None; backend_captured = False; primary_error = None
    try:
        with measured(folder, ROUND, 'producer_freeze_and_backend_capture'):
            freeze(folder); conf = check(folder)
            initial = rw._request_json(rw.ADMIN+'/status', admin=True)
            previous_backend = late_audio._backend_identity(initial); backend_captured = True
            conf.update(original_backend_initial=initial, original_backend_restored=False)
            write(folder/'campaign.json', conf)
        pack, bound = read(folder/'source-pack.json'), read(folder/'source-map.json')
        candidate = build_candidate(folder, pack, bound); conf = check(folder)
        if candidate['changed_target_owners']:
            directory = folder/ROUND/'display'; directory.mkdir(exist_ok=True)
            with measured(folder, ROUND, 'exact_text_display'):
                with rw.backend('qwen', folder/ROUND/'display-backend', Path(conf['writer_recipe'])):
                    sr._display(rw.rows(folder/ROUND/'source.json'), rw.rows(folder/ROUND/'target.json'), directory)
        if check(folder)['local_seconds'] >= LIMIT: raise rw.LocalBudgetExceeded('R1 budget exhausted before scoring')
        result = score(folder, er.validate_receipt(conf['baseline_review'])); conf = check(folder)
        conf['rounds'][ROUND].update(status='scored', score=result['score'], confirmed=result['confirmed'])
        conf.update(status='confirmed_four' if result['confirmed'] else 'candidate_budget_exhausted')
        if result['confirmed']: conf['eligible_candidate'] = ROUND
        write(folder/'campaign.json', conf)
    except BaseException as exc:
        primary_error = exc; conf = read(folder/'campaign.json')
        conf.update(status='time_budget_exhausted' if isinstance(exc, rw.LocalBudgetExceeded) else
            'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error_type=type(exc).__name__)
        write(folder/'campaign.json', conf); raise
    finally:
        if backend_captured:
            try: _restore(folder, previous_backend)
            except BaseException as exc:
                if primary_error is None: raise
                primary_error.add_note('R1 restoration failed: '+type(exc).__name__)
    return check(folder)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,default=ROOT/'output/quality-readable-draft-20260915')
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
