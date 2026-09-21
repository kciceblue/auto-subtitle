"""Offline R1 readable-evidence, native generation and release replay.

All voice literals, identity/view scope and original source-map content must
reconstruct before the sole local writer is replayed. No inference, rewriting,
source-truth claim or automatic promotion occurs in this module.
"""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime
import hashlib
import math
from pathlib import Path

from src import readable_draft_revisit as controller
from src import readable_draft_requests as requests
from src import readable_draft_native as writer_native
from src import readable_evidence as readable
from src import episode_draft_release as inherited_gate
from src import episode_draft_requests as original_draft_requests
from src import joint_context_release as joint_gate
from src import whole_owner as whole
from src import whole_owner_requests as original_requests
from src import temporal_release as temporal_gate
from src import context_release as replay
from src import sparse_release as old
from src import contextual_evidence_review as review
from src import revisit_workflow as workflow
from src.contextual_review_contract import require, strict_json
from src.workflow_state import file_hash, fingerprint, write_json
from src.translate import parse_srt

_same=replay._same
_source_ancestry=inherited_gate._source_ancestry
_global=inherited_gate._global


def _prerequisites(folder, conf, source, target, pack, bound, track):
    parent=Path(conf['l1_predecessor']).resolve(strict=True)
    previous=controller.l1.check(parent)
    controller._terminal_unqualified(previous, 'L1')
    require(_same(track(parent/'campaign.json'),previous), 'L1 prerequisite campaign changed')
    for key in ('source_sha256','target_sha256','context_sha256','source_pack_sha256','map_sha256',
                'pool_version','evidence_paths','baseline_review','media','media_sha256','writer_recipe',
                'writer_recipe_sha256','predecessor','capacity_failure_predecessor',
                'compact_capacity_failure_predecessor'):
        require(_same(conf[key],previous[key]), 'R1 changed its inherited L1 source or comparison inputs')
    require(conf['reused_predecessor_local_seconds']==previous['local_seconds'], 'Inherited L1 timing changed')
    old._producer_snapshots(parent,previous,track)
    for path,digest in {**previous['input_hashes'],**previous['code_pins']}.items():
        track(path,raw=True)
        require(conf['input_hashes'].get(path)==digest==file_hash(Path(path)), 'L1 prerequisite pin missing or changed')
    for name in ('source.json','target.json','original-context.txt','source-pack.json','source-map.json'):
        require(track(parent/name,raw=True)==track(folder/name,raw=True), 'R1 changed its inherited L1 evidence')
    pins={}
    eligibility=controller.joint_eligibility(Path(conf['joint_predecessor']),previous,source,target,pack,bound,pins)
    require(_same(eligibility,conf['joint_eligibility']), 'Declared J1 eligibility differs from native prerequisite replay')
    for predecessor,round_id in ((parent,'L1'),(Path(conf['joint_predecessor']),'J1')):
        score_path=predecessor/round_id/'score.json'
        if score_path.exists():
            score=track(score_path)
            require(conf['input_hashes'].get(str(score_path))==file_hash(score_path), 'Prior scalar prerequisite was not pinned')
            require(not (score.get('status')=='complete' and score.get('confirmed') is True
                and score.get('score',-10)>=4 and score.get('confirmation_score',-10)>=4),
                'R1 cannot continue after a previously confirmed four')
    return eligibility,pins


def _readable_inputs(folder,conf,pack,bound,track):
    native_pins={}
    identities=controller.identity_inputs(pack,conf['evidence_paths'],track,native_pins)
    require(all(conf['input_hashes'].get(path)==digest==file_hash(Path(path))
        for path,digest in native_pins.items()), 'Readable native identity ancestry was not pinned')
    projection=readable.build_projection(pack,bound,identities)
    counts=readable.validate_projection(projection,pack,bound,identities)
    reconstruction=readable.reconstruct_projection(projection['body'],projection['audit'])
    require(_same(reconstruction,original_draft_requests.build_writer_request(pack,bound)['body']),
            'Readable projection does not reconstruct exact L1 source evidence and full map')
    rendered=readable.render_prompt_body(projection)
    readable.validate_prompt_body(rendered,projection)
    require(counts['source_owners']==66 and counts['observations']==len(pack['observations']),
            'Readable projection omitted source owners or acoustic observations')
    expected={'version':'readable-draft-preparation-1','native_artifacts':native_pins,
        'pack_sha256':pack['pack_sha256'],'map_sha256':bound['map_sha256'],
        'identity_map_sha256':fingerprint(identities),'projection_sha256':fingerprint(projection),
        'canonical_body_sha256':fingerprint(reconstruction),'rendered_body_sha256':fingerprint(rendered),
        'coverage':counts,'inference_dispatched':False}
    require(conf['identity_map_sha256']==expected['identity_map_sha256']
        and conf['readable_projection_sha256']==expected['projection_sha256'], 'Readable campaign binding changed')
    for name,value in [('identity-map.json',identities),('projection.json',projection),
        ('canonical-reconstruction.json',reconstruction),('rendered-body.json',rendered),('preparation.json',expected)]:
        path=folder/'readable'/name
        require(conf['input_hashes'].get(str(path))==file_hash(path)
            and _same(track(path),value), 'Readable preparation differs from independently reconstructed native inputs')
    request=requests.build_writer_request(pack,bound,identities,projection=projection)
    require(_same(request['body'],{'source_evidence':rendered,'focus_owner_ids':list(range(1,67))}),
            'Readable writer prompt differs from exact grouped evidence')
    return request,counts,native_pins


def _candidate(folder, conf, source, target, pack, bound, track, *, require_local_gate=True):
    out = folder/controller.ROUND
    candidate = track(out/'candidate.json')
    require(candidate.get('version') == controller.VERSION and candidate.get('round') == controller.ROUND
        and candidate.get('writer') == 'qwen' and candidate.get('checker') == 'qwen'
        and all(candidate.get(key) is False for key in ('source_edits_allowed', 'raw_source_fidelity_verified',
            'external_feedback_used', 'human_reference_used', 'writer_sees_old_chinese')),
        'Whole-owner candidate scope or identity changed')
    require(candidate.get('evidence_paths') == conf['evidence_paths']
        and candidate.get('pack_sha256') == pack['pack_sha256']
        and candidate.get('map_sha256') == bound['map_sha256'], 'Candidate source evidence changed')
    dispatched = set()
    def native(path, request, family):
        path = Path(path)
        require(path not in dispatched, 'Duplicate native request stage')
        dispatched.add(path)
        joint_gate._assert_native_request(path, request, family, track)
        return temporal_gate._native(path, request['instruction'], request['body'], request['schema'], family, track)
    request,readable_counts,native_pins=_readable_inputs(folder,conf,pack,bound,track)
    require(candidate.get('readable_projection_sha256')==conf['readable_projection_sha256']
        and candidate.get('identity_map_sha256')==conf['identity_map_sha256']
        and candidate.get('verifier_context')=='original_l1_baseline', 'Readable candidate scope changed')
    writer_path=out/'requests'/(controller.WRITER_REQUEST_KEY+'.json')
    dispatched.add(writer_path)
    value=writer_native.replay_native(writer_path,request,track)
    require(set(value) == {'owners'} and set(value['owners']) == {str(i) for i in range(1, 67)},
            'Joint native answer lacks exact full-owner coverage')
    proposed = [whole.prepare_owner(owner, value['owners'][str(owner)]['chinese'], source, target, pack)
        for owner in range(1, 67)]
    require(_same(track(out/'proposed-transactions.json'), proposed),
            'Whole-owner proposals differ from exact native generation')
    accepted = []; decisions = []
    for transaction in proposed:
        if not transaction['valid'] or transaction['no_op']:
            continue
        request = requests.build_verifier_request(pack, bound, source, target, transaction)
        require(_same(request, original_requests.build_verifier_request(pack, bound, source, target, transaction)),
                'R1 owner verifier differs from unchanged whole-owner verification')
        verdict = native(out/'requests'/('verify-'+transaction['transaction_id']+'.json'), request, 'qwen')
        keep = whole.accept_owner_decision(verdict, transaction, pack)
        decisions.append({'transaction_id': transaction['transaction_id'], 'accepted': keep, 'verdict': verdict})
        if keep:
            accepted.append(transaction)
    require(_same(track(out/'local-decisions.json'), decisions), 'Whole-owner acceptance differs from native checks')
    first_count = len(accepted)
    final_source, final_target, ledger = whole.apply_owner_transactions(source, target, accepted, pack)
    checks = _global(out, source, target, final_target, accepted, pack, bound, 'global', native) if accepted else {}
    rolled = sorted({identifier for row in checks.values() for finding in row['regressions']
        if finding['quotes_valid'] for identifier in finding['transaction_ids']})
    if rolled:
        accepted = [transaction for transaction in accepted if transaction['transaction_id'] not in rolled]
        final_source, final_target, ledger = whole.apply_owner_transactions(source, target, accepted, pack)
        checks = _global(out, source, target, final_target, accepted, pack, bound, 'assembled', native) if accepted else {}
    if not accepted:
        checks = {str(row.index): {'regressions': [],
            'legacy_or_source_uncertainty': 'Unchanged retained baseline; no generated delta'} for row in source}
    require(set((out/'requests').glob('*.json')) == dispatched,
            'Unaccounted native request records or duplicate generation')
    metrics = [strict_json(line) for line in track(out/'requests/metrics.jsonl', raw=True).splitlines() if line.strip()]
    require(all(isinstance(item, dict) and item.get('stage') in {path.stem for path in dispatched}
        for item in metrics), 'Unregistered R1 native telemetry stage or hidden generation')
    for path in dispatched:
        record = track(path)
        receipts = [receipt for attempt in record['attempts'] for receipt in attempt.get('native_receipts', [])]
        matching = [item for item in metrics if item['stage'] == path.stem]
        require(_same(matching, receipts), 'R1 native telemetry contains unregistered stage attempts')

    require(final_source == source
        and [(r.index, r.ts_line) for r in final_target] == [(r.index, r.ts_line) for r in target],
        'Whole-owner source or geometry changed')
    require(_same(candidate['accepted_transactions'], accepted)
        and _same(candidate['transaction_ledger'], ledger)
        and candidate['rolled_back_transactions'] == rolled
        and _same(candidate['global_checks'], checks), 'Whole-owner assembly or rollback ledger changed')
    require(set(checks) == set(workflow.rowmap(source)) and candidate['coverage'] == 66,
            'Native readable-draft global checks do not cover the complete episode')
    local_gate = not any(finding['quotes_valid'] for row in checks.values() for finding in row['regressions'])
    require(candidate['local_gate_passed'] is local_gate,
            'Saved local gate differs from complete native regression checks')
    if require_local_gate:
        require(local_gate, 'Complete native readable-draft local gate did not pass')
    require(candidate['owners_considered'] == 66
        and candidate['valid_proposals'] == sum(item['valid'] and not item['no_op'] for item in proposed),
        'Whole-owner proposal counts changed')
    for name, expected_rows in [('source.json', source), ('target.json', final_target)]:
        track(out/name)
        require(file_hash(out/name) == candidate[name.replace('.json', '_sha256')]
            and old._rows(out/name) == expected_rows, 'Whole-owner result artifact changed')
    changed = [a.index for a, b in zip(target, final_target) if a.text != b.text]
    require(candidate['changed_target_owners'] == changed, 'Changed owner ledger differs')
    if require_local_gate:
        require(bool(changed), 'No distinct readable-draft candidate exists')
    for name, expected_rows in [('source.utterances.srt', source), ('target.utterances.srt', final_target)]:
        raw = track(out/name, raw=True)
        serialized = ''.join(f'{row.index}\n{row.ts_line}\n{row.text}\n\n' for row in expected_rows).encode('utf-8')
        require(raw == serialized and parse_srt(out/name, preserve_text_whitespace=True) == expected_rows,
                'Owner SRT differs from exact source/target rows')
    if changed:
        for name in ('display.json', 'source.srt', 'subtitles.zh.srt'):
            track(out/'display'/name, raw=True)
        display = old.display_checks._display(out/'display', source, final_target)
    else:
        require(final_target == target and not accepted and not ledger
            and not list((out/'display').glob('*')), 'Unchanged baseline has unexpected edits or display artifacts')
        display = {'status': 'not_run_unchanged_baseline', 'cues': 0}
    return final_target, {'owners_considered': 66, 'writer_groups': 1,
        'local_gate_passed': local_gate, 'readable_coverage': readable_counts,
        'native_identity_artifacts': native_pins, 'owner_checks': len(decisions), 'accepted_before_rollback': first_count,
        'accepted_transactions': len(accepted), 'rolled_back_transactions': len(rolled),
        'owner_coverage': 66, 'changed_target_owners': len(changed),
        'distinct_candidate': bool(changed), 'display': display}

def _below_target_score(folder, conf, source, target, pools, baseline, track, local_gate):
    """Replay a fresh or inherited native scalar without turning it into release."""
    target_hash = review.owner_rows_sha256(target)
    allowed = {str(Path(name).resolve(strict=True)) for name in conf.get('prior_score_paths', [])}
    for path in allowed:
        require(conf['input_hashes'].get(path) == file_hash(Path(path)), 'Historical duplicate score was not pinned')
    seen = set()
    def replay_score(path, *, candidate):
        path = Path(path).resolve(strict=True)
        require(path not in seen, 'Cyclic duplicate score ancestry')
        seen.add(path); score = track(path)
        require(score.get('status') == 'complete' and score.get('pool_version') == conf['pool_version']
            and score.get('target_sha256') == target_hash, 'Scalar does not bind readable-draft target')
        if score.get('inherited_unchanged_baseline'):
            require(candidate and set(score) == {'status', 'score', 'inherited_unchanged_baseline',
                'confirmed', 'pool_version', 'target_sha256'}
                and score['inherited_unchanged_baseline'] is True and score['confirmed'] is False
                and type(score['score']) in (int, float) and score['score'] == baseline['score'] == 2
                and target == old._rows(folder/'target.json'), 'Invalid unchanged-baseline scalar inheritance')
            return score['score']
        if score.get('inherited_duplicate_target'):
            parent = str(Path(score['inherited_duplicate_target']).resolve(strict=True))
            require(parent in allowed and score.get('confirmed') is False
                and not score.get('primary') and not score.get('confirmation'), 'Invalid duplicate score inheritance')
            inherited_score = replay_score(Path(parent), candidate=False)
            require(score.get('score') == inherited_score, 'Duplicate target changed its native scalar')
            return inherited_score
        if candidate:
            require(score.get('local_gate_passed') is local_gate, 'Fresh scalar local gate binding changed')
            for prior_path in allowed:
                prior = track(prior_path)
                require(not (prior.get('status') == 'complete' and prior.get('pool_version') == conf['pool_version']
                    and prior.get('target_sha256') == target_hash), 'Previously scored target received a fresh reroll')
        primary = review.validate_receipt(score['primary'])
        dependency = lambda value: {key: value[key] for key in
            ('purpose', 'output_dir', 'dispatch_id', 'finished_utc', 'receipt_hashes')}
        require(primary['purpose'] == 'candidate' and primary['score'] == score['score']
            and primary['pool_version'] == conf['pool_version'] and primary['dependency'] == dependency(baseline)
            and primary['dispatch_id'] != baseline['dispatch_id']
            and datetime.fromisoformat(baseline['finished_utc']) <= datetime.fromisoformat(primary['started_utc']),
            'Primary is not a fresh native dependent review')
        for key, rows in [('source', source), ('raw', source), ('target', target)]:
            require(primary['inputs'][key]['sha256'] == review.owner_rows_sha256(rows), 'Primary owner input changed')
        require(all(primary['inputs'][key]['sha256'] == baseline['inputs'][key]['sha256']
            for key in ('source', 'raw', 'context', 'evidence'))
            and score.get('source_sha256') == review.owner_rows_sha256(source), 'Primary changed the expanded benchmark')
        final_score = primary['score']; receipts = [primary]
        if score.get('confirmation'):
            confirmation = review.validate_receipt(score['confirmation'])
            require(primary['score'] >= 4 and score.get('local_gate_passed') is True
                and confirmation['purpose'] == 'confirmation'
                and confirmation['score'] == score.get('confirmation_score')
                and confirmation['inputs'] == primary['inputs']
                and confirmation['pool_version'] == primary['pool_version']
                and confirmation['dependency'] == dependency(primary)
                and datetime.fromisoformat(primary['finished_utc']) <= datetime.fromisoformat(confirmation['started_utc'])
                and len({baseline['dispatch_id'], primary['dispatch_id'], confirmation['dispatch_id']}) == 3,
                'Confirmation is not a fresh designated native review')
            require(score.get('confirmed') is (confirmation['score'] >= 4), 'Confirmation result differs from native scalar')
            final_score = min(final_score, confirmation['score']); receipts.append(confirmation)
        else:
            require(not (primary['score'] >= 4 and score.get('local_gate_passed') is True)
                and score.get('confirmed') is False and 'confirmation_score' not in score,
                'Required designated confirmation is absent')
        preparation = track(Path(primary['output_dir'])/'preparation.json')
        manifest = Path(preparation['manifest_path']); track(manifest)
        bundle = review.read_bundle(manifest)
        require(bundle['pool_version'] == conf['pool_version']
            and sorted(fingerprint(pool) for pool in strict_json(bundle['raw_inputs']['evidence']))
                == sorted(fingerprint(pool) for pool in pools), 'Native scalar acoustic pools changed')
        for receipt in receipts:
            for name, digest in receipt['receipt_hashes'].items():
                receipt_path = Path(receipt['output_dir'])/name; track(receipt_path, raw=True)
                require(file_hash(receipt_path) == digest, 'Native scalar receipt changed')
        return final_score
    return replay_score(folder/controller.ROUND/'score.json', candidate=True)

def validate_campaign(folder: str | Path, *, require_four: bool = True) -> dict:
    """Replay R1 from immutable native records, with no inference or promotion."""
    folder = Path(folder).resolve(strict=True); conf = controller.check(folder)
    require(conf.get('original_backend_restored') is True
        and conf.get('status') in ('confirmed_four', 'candidate_budget_exhausted'),
        'Whole-owner campaign is not complete and restored')
    require(conf.get('target') == 4, 'Whole-owner score target changed')
    elapsed = conf.get('local_seconds')
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed < controller.LIMIT,
            'Whole-owner local budget was exceeded or is invalid')
    if require_four:
        require(conf['status'] == 'confirmed_four' and conf.get('eligible_candidate') == controller.ROUND,
                'R1 has not reached a confirmed four')
    pins = {}
    def track(path, raw=False):
        path = Path(path); require(not path.is_symlink(), 'Symlink cannot serve as immutable artifact')
        path = path.resolve(strict=True); data = path.read_bytes(); digest = hashlib.sha256(data).hexdigest()
        require(str(path) not in pins or pins[str(path)] == digest, 'Artifact changed during readable-draft replay')
        pins[str(path)] = digest
        return data if raw else strict_json(data)
    require(_same(track(folder/'campaign.json'), conf), 'Whole-owner campaign changed during validation')
    old._producer_snapshots(folder, conf, track)
    required = {str(controller.ROOT/'src'/f'{name}.py') for name in
        ('readable_draft_revisit','readable_draft_release','readable_draft_requests','readable_draft_native',
         'readable_evidence','readable_identity','joint_context_revisit','joint_context_release','joint_context',
         'episode_draft_revisit','episode_draft_release','episode_draft_requests','episode_draft_native',
         'episode_draft_recovery','compact_owner_revisit','compact_owner_release','compact_owner_requests',
         'whole_owner_revisit','whole_owner_release','whole_owner','whole_owner_requests')}
    require(required <= set(conf['code_pins']), 'Missing readable-draft producer snapshots')
    for path, digest in {**conf['input_hashes'], **conf['code_pins']}.items():
        track(path, raw=True); require(file_hash(Path(path)) == digest, 'Pinned readable-draft input or producer changed')
    require(conf['writer_recipe_sha256'] == file_hash(Path(conf['writer_recipe']))
        and conf['plan_sha256'] == file_hash(Path(conf['plan'])), 'Whole-owner recipe or plan changed')
    source, target = controller._source_pair(folder)
    require(len(source) == len(target) == 66
        and [(row.index, row.ts_line) for row in source] == [(row.index, row.ts_line) for row in target],
        'Original readable-draft geometry changed')
    pools = [review._validate_pool(Path(path)) for path in conf['evidence_paths']]
    require(bool(pools) and all(pool['media_sha256'] == conf['media_sha256'] for pool in pools),
            'Whole-owner evidence belongs to another recording')
    require(file_hash(Path(conf['media'])) == conf['media_sha256'], 'Recording identity changed')
    pins[str(Path(conf['media']).resolve())] = conf['media_sha256']
    pack, bound, baseline, fallback_ids = _source_ancestry(folder, conf, source, target, pools, track)
    failures = {
        'W1': inherited_gate.compact_gate._capacity_failure_ancestry(folder, conf, source, target, pools, pack, bound, track),
        'W2': inherited_gate._compact_capacity_failure_ancestry(folder, conf, source, target, pools, pack, bound, track)}
    eligibility,prerequisite_pins=_prerequisites(folder,conf,source,target,pack,bound,track)
    final_target, local = _candidate(folder, conf, source, target, pack, bound, track, require_local_gate=require_four)
    for path,digest in {**prerequisite_pins,**local.pop('native_identity_artifacts')}.items():
        require(conf['input_hashes'].get(path)==digest and (path not in pins or pins[path]==digest),
                'Readable prerequisite or identity pin changed')
        pins[path]=digest
    if require_four:
        primary, confirmation = replay._scores(folder, conf, controller.ROUND, source, final_target, pools, track)
        score = min(primary['score'], confirmation['score'])
    else:
        score = _below_target_score(folder, conf, source, final_target, pools, baseline, track, local['local_gate_passed'])
    require(all(file_hash(Path(path)) == digest for path, digest in pins.items()), 'Artifact changed during final replay')
    return {'version': controller.VERSION, 'round': controller.ROUND,
        'status': 'contextual_ready' if require_four else 'replayed_not_qualified',
        'contextual_ready': require_four, 'local_gate_passed': local['local_gate_passed'], 'score': score,
        'local': local, 'closed_capacity_failures': failures,
        'writer_generation_reused': False, 'joint_eligibility': eligibility, 'fallback_owner_ids': fallback_ids, 'pack_sha256': pack['pack_sha256'],
        'map_sha256': bound['map_sha256'], 'artifacts': pins, 'source_accuracy_verified': False,
        'audio_truth_verified': False, 'playback_verified': False, 'selected_output_modified': False}

def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path); parser.add_argument('--below-target', action='store_true')
    args = parser.parse_args(); result = validate_campaign(args.campaign, require_four=not args.below_target)
    write_json(args.campaign/'release-validation.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'score', 'local')}))

if __name__ == '__main__':
    main()
