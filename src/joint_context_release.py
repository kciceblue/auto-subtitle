"""Offline J1 control/treatment replay and independent source-aware score gates.

No inference, rewriting or promotion occurs during replay. Local gate failures
may be reported without being qualified; release still requires a true local
gate and the designated primary and confirmation scores both at least four.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import math
from pathlib import Path

from src import joint_context_revisit as controller
from src import joint_context as joint
from src import episode_draft_release as inherited_gate
from src import whole_owner as whole
from src import whole_owner_requests as original_requests
from src import temporal_release as temporal_gate
from src import context_release as replay
from src import sparse_release as old
from src import sparse_revisit as sparse
from src import contextual_evidence_review as review
from src import revisit_workflow as workflow
from src.contextual_review_contract import require, strict_json
from src.workflow_state import file_hash, fingerprint, write_json
from src.translate import parse_srt

_same = replay._same
_source_ancestry = inherited_gate._source_ancestry
_global = inherited_gate._global


def _assert_native_request(path, request, family, track):
    """Bind exact JSON types and the complete fixed Qwen request payload.

    The inherited native replay also validates execution/answer receipts, but
    Python equality there can conflate bool/int and int/float. J1 additionally
    requires the exact expected request fingerprint for both control and
    treatment paths, including schema copies and every sampler/template field.
    """
    require(family == 'qwen', 'J1 permits only the declared local Qwen checker')
    extra = {'model': workflow.QWEN, 'temperature': .3, 'seed': 20260915, 'top_p': .95,
        'max_tokens': 12288, 'chat_template_kwargs': {'enable_thinking': True},
        'reasoning_budget_tokens': 6144, 'reasoning_effort': 'high',
        'response_format': {'type': 'json_object', 'schema': request['schema']}}
    expected = {'version': sparse.VERSION, 'family': 'qwen',
        'instruction': request['instruction'], 'body': request['body'], 'schema': request['schema'],
        'extra': extra, 'thinking': True}
    record = track(path)
    require(joint._same_json(record.get('request'), expected)
        and record.get('request_sha256') == fingerprint(expected),
        'Native J1 control or treatment differs from exact expected request')


def _controls(folder, conf, source, target, pack, bound, track):
    pins = {}
    def control_track(path, raw=False):
        value = track(path, raw=raw)
        path = Path(path).resolve(strict=True)
        digest = file_hash(path)
        require(conf['input_hashes'].get(str(path)) == digest,
                'Inherited L1 control artifact was not input-pinned')
        require(str(path) not in pins or pins[str(path)] == digest,
                'Inherited L1 control artifact changed during replay')
        pins[str(path)] = digest
        return value
    controls = controller.replay_controls(folder, conf, source, target, pack, bound, control_track)
    proposals, decisions, full_draft = controls['proposals'], controls['decisions'], controls['full_draft']
    require(len(proposals) == 66 and all(item['valid'] and not item['no_op'] for item in proposals)
        and [item['owner_id'] for item in proposals] == list(range(1, 67)),
        'J1 requires all 66 valid non-noop original L1 proposals')
    require(joint.assemble_draft(source, target, proposals, pack) == full_draft,
            'Complete L1 draft differs from exact canonical proposals')
    require(len(decisions) == 66 and [item['transaction_id'] for item in decisions]
        == [item['transaction_id'] for item in proposals], 'L1 control coverage differs from all 66 proposals')
    for proposal, decision in zip(proposals, decisions):
        request = original_requests.build_verifier_request(pack, bound, source, target, proposal)
        control_path = Path(conf['control_predecessor'])/'L1/requests'/('verify-'+proposal['transaction_id']+'.json')
        _assert_native_request(control_path, request, 'qwen', control_track)
        require(set(decision) == {'transaction_id', 'accepted', 'verdict'}
            and decision['accepted'] is whole.accept_owner_decision(decision['verdict'], proposal, pack),
            'Inherited L1 control acceptance differs from the unchanged predicate')
    expected = {'control_predecessor': conf['control_predecessor'], 'native_artifacts': pins,
        'proposals_sha256': fingerprint(proposals), 'decisions_sha256': fingerprint(decisions),
        'full_draft_sha256': fingerprint([asdict(row) for row in full_draft])}
    path = folder/'control-inheritance.json'
    require(conf['input_hashes'].get(str(path)) == file_hash(path)
        and _same(track(path), expected), 'L1 control inheritance differs from native replay')
    out = folder/controller.ROUND
    for name, value in [('proposed-transactions.json', proposals), ('control-decisions.json', decisions),
                        ('full-draft.json', [asdict(row) for row in full_draft])]:
        saved = out/name
        require(conf['input_hashes'].get(str(saved)) == file_hash(saved)
            and _same(track(saved), value), 'Prepared L1 control products differ from native replay')
    return controls


def _candidate(folder, conf, source, target, pack, bound, track, *, require_local_gate=True):
    out = folder/controller.ROUND
    candidate = track(out/'candidate.json')
    require(candidate.get('version') == controller.VERSION and candidate.get('round') == controller.ROUND
        and candidate.get('writer') == 'reused_l1_qwen' and candidate.get('checker') == 'qwen'
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
        _assert_native_request(path, request, family, track)
        return temporal_gate._native(path, request['instruction'], request['body'], request['schema'], family, track)
    controls = _controls(folder, conf, source, target, pack, bound, track)
    proposed = controls['proposals']; full_draft = controls['full_draft']
    require(candidate.get('control_predecessor') == conf['control_predecessor']
        and candidate.get('control_decisions_sha256') == fingerprint(controls['decisions']),
        'Candidate L1 control identity changed')
    require(_same(track(out/'proposed-transactions.json'), proposed),
            'Joint context proposals differ from immutable native L1 generation')
    pairs = []
    accepted = []; decisions = []
    for transaction in proposed:
        if not transaction['valid'] or transaction['no_op']:
            continue
        request = joint.build_treatment_request(pack, bound, source, target, proposed, transaction['owner_id'])
        control = original_requests.build_verifier_request(pack, bound, source, target, transaction)
        difference = joint.assert_context_only_change(control, request, full_draft)
        verdict = native(out/'requests'/('verify-'+transaction['transaction_id']+'.json'), request, 'qwen')
        keep = whole.accept_owner_decision(verdict, transaction, pack)
        decisions.append({'transaction_id': transaction['transaction_id'], 'accepted': keep, 'verdict': verdict})
        before = controls['decisions'][transaction['owner_id']-1]
        pairs.append({'transaction_id': transaction['transaction_id'], 'control_accepted': before['accepted'],
            'treatment_accepted': keep, 'request_diff': difference})
        if keep:
            accepted.append(transaction)
    require(_same(track(out/'local-decisions.json'), decisions), 'Whole-owner acceptance differs from native checks')
    counts = {'control_accepted': sum(p['control_accepted'] for p in pairs),
        'treatment_accepted': sum(p['treatment_accepted'] for p in pairs),
        'accepted_both': sum(p['control_accepted'] and p['treatment_accepted'] for p in pairs),
        'rejected_both': sum(not p['control_accepted'] and not p['treatment_accepted'] for p in pairs),
        'newly_accepted': sum(not p['control_accepted'] and p['treatment_accepted'] for p in pairs),
        'newly_rejected': sum(p['control_accepted'] and not p['treatment_accepted'] for p in pairs)}
    comparison = {'control_decisions_sha256': fingerprint(controls['decisions']),
        'treatment_decisions_sha256': fingerprint(decisions), 'pairs': pairs, 'counts': counts}
    require(_same(track(out/'control-comparison.json'), comparison),
            'Paired control/treatment comparison differs from exact native decisions')
    require(bool(accepted), 'No accepted non-noop joint-context candidate')
    first_count = len(accepted)
    final_source, final_target, ledger = whole.apply_owner_transactions(source, target, accepted, pack)
    checks = _global(out, source, target, final_target, accepted, pack, bound, 'global', native)
    rolled = sorted({identifier for row in checks.values() for finding in row['regressions']
        if finding['quotes_valid'] for identifier in finding['transaction_ids']})
    if rolled:
        accepted = [transaction for transaction in accepted if transaction['transaction_id'] not in rolled]
        require(bool(accepted), 'All joint-context changes rolled back; no distinct candidate')
        final_source, final_target, ledger = whole.apply_owner_transactions(source, target, accepted, pack)
        checks = _global(out, source, target, final_target, accepted, pack, bound, 'assembled', native)
    require(set((out/'requests').glob('*.json')) == dispatched,
            'Unaccounted native request records or duplicate generation')
    metrics = [strict_json(line) for line in track(out/'requests/metrics.jsonl', raw=True).splitlines() if line.strip()]
    require(all(isinstance(item, dict) and item.get('stage') in {path.stem for path in dispatched}
        for item in metrics), 'Unregistered J1 native telemetry stage or hidden generation')
    for path in dispatched:
        record = track(path)
        receipts = [receipt for attempt in record['attempts'] for receipt in attempt.get('native_receipts', [])]
        matching = [item for item in metrics if item['stage'] == path.stem]
        require(_same(matching, receipts), 'J1 native telemetry contains unregistered stage attempts')
    require(final_source == source
        and [(r.index, r.ts_line) for r in final_target] == [(r.index, r.ts_line) for r in target],
        'Whole-owner source or geometry changed')
    require(_same(candidate['accepted_transactions'], accepted)
        and _same(candidate['transaction_ledger'], ledger)
        and candidate['rolled_back_transactions'] == rolled
        and _same(candidate['global_checks'], checks), 'Whole-owner assembly or rollback ledger changed')
    require(set(checks) == set(workflow.rowmap(source)) and candidate['coverage'] == 66,
            'Native joint-context global checks do not cover the complete episode')
    local_gate = not any(finding['quotes_valid'] for row in checks.values() for finding in row['regressions'])
    require(candidate['local_gate_passed'] is local_gate,
            'Saved local gate differs from complete native regression checks')
    if require_local_gate:
        require(local_gate, 'Complete native joint-context local gate did not pass')
    require(candidate['owners_considered'] == 66
        and candidate['valid_proposals'] == sum(item['valid'] and not item['no_op'] for item in proposed),
        'Whole-owner proposal counts changed')
    for name, expected_rows in [('source.json', source), ('target.json', final_target)]:
        track(out/name)
        require(file_hash(out/name) == candidate[name.replace('.json', '_sha256')]
            and old._rows(out/name) == expected_rows, 'Whole-owner result artifact changed')
    changed = [a.index for a, b in zip(target, final_target) if a.text != b.text]
    require(bool(changed) and candidate['changed_target_owners'] == changed, 'Changed owner ledger differs')
    for name, expected_rows in [('source.utterances.srt', source), ('target.utterances.srt', final_target)]:
        raw = track(out/name, raw=True)
        serialized = ''.join(f'{row.index}\n{row.ts_line}\n{row.text}\n\n' for row in expected_rows).encode('utf-8')
        require(raw == serialized and parse_srt(out/name, preserve_text_whitespace=True) == expected_rows,
                'Owner SRT differs from exact source/target rows')
    for name in ('display.json', 'source.srt', 'subtitles.zh.srt'):
        track(out/'display'/name, raw=True)
    display = old.display_checks._display(out/'display', source, final_target)
    return final_target, {'owners_considered': 66, 'writer_groups': 0, 'reused_writer_groups': 1, 'control_checks': 66,
        'local_gate_passed': local_gate, 'paired_counts': counts,
        'owner_checks': len(decisions), 'accepted_before_rollback': first_count,
        'accepted_transactions': len(accepted), 'rolled_back_transactions': len(rolled),
        'owner_coverage': 66, 'changed_target_owners': len(changed), 'display': display}

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
            and score.get('target_sha256') == target_hash, 'Scalar does not bind joint-context target')
        require(not score.get('inherited_unchanged_baseline'), 'No distinct candidate exists')
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
    """Replay J1 from immutable native records, with no inference or promotion."""
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
                'J1 has not reached a confirmed four')
    pins = {}
    def track(path, raw=False):
        path = Path(path); require(not path.is_symlink(), 'Symlink cannot serve as immutable artifact')
        path = path.resolve(strict=True); data = path.read_bytes(); digest = hashlib.sha256(data).hexdigest()
        require(str(path) not in pins or pins[str(path)] == digest, 'Artifact changed during joint-context replay')
        pins[str(path)] = digest
        return data if raw else strict_json(data)
    require(_same(track(folder/'campaign.json'), conf), 'Whole-owner campaign changed during validation')
    old._producer_snapshots(folder, conf, track)
    required = {str(controller.ROOT/'src'/f'{name}.py') for name in
        ('joint_context_revisit', 'joint_context_release', 'joint_context',
            'episode_draft_revisit', 'episode_draft_release', 'episode_draft_requests', 'episode_draft_native', 'episode_draft_recovery',
            'compact_owner_revisit', 'compact_owner_release', 'compact_owner_requests',
            'whole_owner_revisit', 'whole_owner_release', 'whole_owner', 'whole_owner_requests')}
    require(required <= set(conf['code_pins']), 'Missing joint-context producer snapshots')
    for path, digest in {**conf['input_hashes'], **conf['code_pins']}.items():
        track(path, raw=True); require(file_hash(Path(path)) == digest, 'Pinned joint-context input or producer changed')
    require(conf['writer_recipe_sha256'] == file_hash(Path(conf['writer_recipe']))
        and conf['plan_sha256'] == file_hash(Path(conf['plan'])), 'Whole-owner recipe or plan changed')
    source, target = controller._source_pair(folder)
    require(len(source) == len(target) == 66
        and [(row.index, row.ts_line) for row in source] == [(row.index, row.ts_line) for row in target],
        'Original joint-context geometry changed')
    pools = [review._validate_pool(Path(path)) for path in conf['evidence_paths']]
    require(bool(pools) and all(pool['media_sha256'] == conf['media_sha256'] for pool in pools),
            'Whole-owner evidence belongs to another recording')
    require(file_hash(Path(conf['media'])) == conf['media_sha256'], 'Recording identity changed')
    pins[str(Path(conf['media']).resolve())] = conf['media_sha256']
    pack, bound, baseline, fallback_ids = _source_ancestry(folder, conf, source, target, pools, track)
    failures = {
        'W1': inherited_gate.compact_gate._capacity_failure_ancestry(folder, conf, source, target, pools, pack, bound, track),
        'W2': inherited_gate._compact_capacity_failure_ancestry(folder, conf, source, target, pools, pack, bound, track)}
    final_target, local = _candidate(folder, conf, source, target, pack, bound, track, require_local_gate=require_four)
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
        'writer_generation_reused': True, 'control_predecessor': conf['control_predecessor'], 'fallback_owner_ids': fallback_ids, 'pack_sha256': pack['pack_sha256'],
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
