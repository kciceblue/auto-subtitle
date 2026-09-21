"""Read-only replay of T1 temporal interpretation, local edits and scalar gates.

No models are dispatched. Native source-frame outputs, fallbacks, recap, every
proposal and all owner checks must replay before a candidate can qualify.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from src import temporal_revisit as tr
from src import short_audio
from src import context_release as replay
from src import evidence_context as ec
from src import sparse_revisit as sr
from src import sparse_edits as edits
from src import sparse_release as old
from src import revisit_workflow as rw
from src import contextual_evidence_review as er
from src.contextual_review_contract import require, strict_json
from src.workflow_state import file_hash, fingerprint, write_json

_same, _bound = replay._same, replay._bound


def _native(path, instruction, body, schema, family, track):
    result = replay._native(path, instruction, body, schema, family, track)
    attempts = track(path)['attempts']
    require(all(isinstance(a.get('error'), str) and bool(a['error']) for a in attempts[:-1])
            and 'error' not in attempts[-1], 'Successful semantic request was rerolled')
    import jsonschema
    for attempt in attempts[:-1]:
        raw = attempt.get('raw'); receipts = attempt.get('native_receipts', [])
        normal = (isinstance(raw, str) and bool(receipts)
            and receipts[-1].get('finish_reason') == 'stop'
            and receipts[-1].get('stage') == Path(path).stem
            and receipts[-1].get('answer_chars') == len(raw))
        if normal:
            try:
                jsonschema.validate(strict_json(raw), schema)
            except (ValueError, jsonschema.ValidationError):
                continue
            raise ValueError('Successful semantic request was rerolled despite its error marker')
    return result


def _source_frames(folder, conf, source, observations, context, track):
    requests = tr.source_frame_requests(source, observations, context)
    coverage = [owner for request in requests for owner in request['body']['focus_owner_ids']]
    require(len(requests) == 33 and coverage == list(range(1, 67))
            and all(len(request['body']['focus_owner_ids']) == 2 for request in requests),
            'Temporal source interpretation must cover every owner once in pairs')
    responses = []
    for number, request in enumerate(requests, 1):
        responses.append(_native(folder/'T1/requests'/f'source-frames-{number}.json',
            request['instruction'], request['body'], request['schema'], 'qwen', track))
    rebuilt = tr.compile_source_frames(responses, requests)
    binding = {'observations_sha256': fingerprint(observations),
        'source_sha256': file_hash(folder/'source.json'), 'context_sha256': conf['context_sha256'],
        'requests_sha256': fingerprint(requests)}
    saved = _bound(folder/'shared/source-frames.json', binding, track)
    require(_same(saved, rebuilt), 'Temporal source frames or fallback diagnostics differ from native replay')
    require(set(rebuilt['frames']) == {str(i) for i in range(1, 67)}, 'Source-frame owner coverage changed')
    return rebuilt


def _map_and_analysis(folder, conf, source, target, observations, context, track):
    frame_result = _source_frames(folder, conf, source, observations, context, track)
    pools = [er._validate_pool(Path(p)) for p in conf['evidence_paths']]
    expected = ec.build_source_pack(source, observations, frame_result['frames'], context,
        pool_versions=[p['pool_version'] for p in pools])
    pack = track(folder/'source-pack.json')
    require(_same(pack, expected) and pack['pack_sha256'] == conf['source_pack_sha256'],
            'Fresh temporal source pack differs from native source interpretation')
    prepared = tr.prepare_recap_request(pack)
    binding = {'pack_sha256': pack['pack_sha256'],
        'request': fingerprint({k: prepared[k] for k in ('instruction', 'body', 'schema')})}
    bound_map = _bound(folder/'shared/source-map.json', binding, track)
    native_map = _native(folder/'T1/requests'/(tr.RECAP_REQUEST_KEY+'.json'),
        prepared['instruction'], prepared['body'], prepared['schema'], 'qwen', track)
    require(_same(bound_map, tr.bind_recap(native_map, pack)),
            'Map does not replay native recap with mechanically derived owners')
    tr.focus_context(pack, bound_map, [1])
    binding = {'pack_sha256': pack['pack_sha256'], 'map_sha256': bound_map['map_sha256'],
        'target_sha256': conf['target_sha256'], 'instruction': fingerprint(tr.DIAGNOSE)}
    analysis = _bound(folder/'shared/analysis.json', binding, track)
    cases = []; invalid = []; coverage = []
    case_schema = rw.obj({'target_span': rw.string(1000), 'source_span': rw.string(1000),
        'category': {'type': 'string', 'enum': ['mistranslation', 'omission', 'unsupported_specificity', 'negation', 'intent', 'terminology']},
        'severity': {'type': 'integer', 'minimum': 1, 'maximum': 3}, 'issue': rw.string(1200), 'unresolved': {'type': 'boolean'}})
    for number, ids in enumerate(sr.chunk_ids(source), 1):
        body = {**sr.episode_body(source, target, context), 'focus_ids': ids,
            'source_readings': {str(i): pack['frames'][str(i)] for i in ids},
            'focused_context': tr.focus_context(pack, bound_map, ids)}
        result = _native(folder/'T1/requests'/f'discrepancy-{number}.json', tr.DIAGNOSE, body,
            rw.keyed(ids, rw.obj({'cases': rw.arr(case_schema, 2)})), 'qwen', track)
        coverage.extend(ids)
        for owner, row in result.items():
            for ordinal, case in enumerate(row['cases'], 1):
                case = {**case, 'owner_id': int(owner), 'case_id': f'T1-o{int(owner):03d}-c{ordinal}',
                    'readings': pack['frames'][owner]['readings'],
                    'unresolved': case['unresolved'] or pack['frames'][owner]['unresolved']}
                try:
                    cases.append(edits.validate_case(case, source, target, observations))
                except ValueError as exc:
                    invalid.append({'case_id': case['case_id'], 'error_type': type(exc).__name__})
    require(coverage == list(range(1, 67)), 'Temporal diagnosis omitted an owner')
    expected = {'version': tr.VERSION, 'cases': cases, 'invalid_cases': invalid, 'coverage': coverage,
        'pack_sha256': pack['pack_sha256'], 'map_sha256': bound_map['map_sha256'], 'external_feedback_used': False}
    require(_same(analysis, expected), 'Shared diagnosis differs from native full-coverage replay')
    return pack, bound_map, analysis


def _global(folder, out, source, target, before_target, transactions, pack, bound_map, checker, prefix, track):
    obs = {o['observation_id']: o for o in pack['observations']}; result = {}
    transaction_ids = [t['case_id'] for t in transactions]
    for number, ids in enumerate(sr.chunk_ids(source), 1):
        own = [o for o in pack['observations'] if o['owner_id'] in ids]
        regression = rw.obj({'transaction_ids': rw.arr({'type': 'string', 'enum': transaction_ids}, 6),
            'before_quote': rw.string(1000), 'after_quote': rw.string(1000),
            'observation_id': {'type': 'string', 'enum': [o['observation_id'] for o in own]},
            'source_quote': rw.string(1200), 'reason': rw.string(1200)})
        body = {**sr.episode_body(source, before_target, pack['original_context']), 'candidate_japanese': rw.rowmap(source),
            'candidate_chinese': rw.rowmap(target), 'focus_ids': ids,
            'transactions': [{k: t[k] for k in ('case_id', 'owner_id', 'old_chinese', 'new_chinese', 'old_japanese', 'new_japanese')} for t in transactions],
            'focused_context': tr.focus_context(pack, bound_map, ids)}
        value = _native(out/'requests'/f'{prefix}-{number}.json', tr.GLOBAL, body,
            rw.keyed(ids, rw.obj({'regressions': rw.arr(regression, 6), 'legacy_or_source_uncertainty': rw.string(1200)})), checker, track)
        for owner, row in value.items():
            for finding in row['regressions']:
                witness = obs[finding['observation_id']]
                finding['quotes_valid'] = bool(finding['transaction_ids'] and finding['before_quote'] and finding['after_quote'] and finding['source_quote']
                    and witness['owner_id'] == int(owner) and finding['before_quote'] in before_target[int(owner)-1].text
                    and finding['after_quote'] in target[int(owner)-1].text and finding['source_quote'] in witness['text'])
        result.update(value)
    return result


def _candidate(folder, conf, rid, source, target, pack, bound_map, analysis, track,
               *, candidate_version=None, roles=None, writer_native=None):
    out = folder/rid
    binding = {'round': rid, 'pack_sha256': pack['pack_sha256'], 'map_sha256': bound_map['map_sha256'],
        'analysis_sha256': file_hash(folder/'shared/analysis.json'), 'target_sha256': conf['target_sha256']}
    candidate = _bound(out/'candidate.json', binding, track)
    require(candidate.get('version') == (tr.VERSION if candidate_version is None else candidate_version) and candidate.get('round') == rid
        and candidate.get('source_edits_allowed') is False and candidate.get('raw_source_fidelity_verified') is False
        and candidate.get('external_feedback_used') is False and candidate.get('human_reference_used') is False,
        'Candidate scope or identity changed')
    require(candidate.get('evidence_paths') == conf['evidence_paths'] and candidate.get('pack_sha256') == pack['pack_sha256']
        and candidate.get('map_sha256') == bound_map['map_sha256'], 'Candidate context evidence changed')
    require(_same(track(out/'analysis.json'), analysis), 'Candidate diagnosis differs from shared temporal analysis')
    cases = sr.choose_cases(analysis, 24)
    require(_same(track(out/'selected-cases.json'), cases), 'Selected cases changed')
    proposed = track(out/'proposed-transactions.json')
    require(isinstance(proposed, list) and len(proposed) == len(cases), 'Proposal coverage changed')
    writer, checker = roles if roles is not None else ('gemma', 'qwen')
    require(candidate.get('writer') == writer and candidate.get('checker') == checker, 'Wrong temporal writer/checker allocation')
    decisions = []; accepted = []
    for case, transaction in zip(cases, proposed):
        focused = tr.focus_context(pack, bound_map, [case['owner_id']])
        native = (writer_native or _native)(out/'requests'/('propose-'+case['case_id']+'.json'), tr.PROPOSE,
            {**sr.episode_body(source, target, pack['original_context']), 'case': case, 'focused_context': focused},
            tr.PROPOSAL_SCHEMA, writer, track)
        proposal = {**native, **{k: '' for k in ('old_japanese', 'new_japanese', 'source_observation_id', 'source_support_quote')}}
        expected = edits.prepare_patch(case, proposal, source, target, pack['observations'], allow_source=False)
        require(_same(transaction, expected), 'Two-field proposal transaction does not replay exactly')
        if not transaction['valid'] or transaction['no_op']:
            continue
        own = [o for o in pack['observations'] if o['owner_id'] == case['owner_id']]
        body = {'episode_japanese': rw.rowmap(source), 'episode_chinese': rw.rowmap(target), 'original_context': pack['original_context'],
            'case': case, 'before_japanese': transaction['before_source_text'], 'before_chinese': transaction['before_target_text'],
            'after_japanese': transaction['after_source_text'], 'after_chinese': transaction['after_target_text'],
            'source_readings': case['readings'], 'focused_context': focused}
        verdict = _native(out/'requests'/('verify-'+case['case_id']+'.json'), tr.VERIFY, body,
            tr.decision_schema([o['observation_id'] for o in own]), checker, track)
        keep = sr.accept_decision(verdict, case, transaction, pack['observations'])
        decisions.append({'case_id': case['case_id'], 'accepted': keep, 'verdict': verdict})
        if keep:
            accepted.append(transaction)
    require(_same(track(out/'local-decisions.json'), decisions), 'Acceptance differs from native decisions')
    require(bool(accepted), 'No accepted non-noop candidate')
    first_count = len(accepted)
    final_source, final_target, ledger = edits.apply_transactions(source, target, accepted)
    checks = _global(folder, out, source, final_target, target, accepted, pack, bound_map, checker, 'global', track)
    rolled = sorted({cid for row in checks.values() for f in row['regressions'] if f['quotes_valid'] for cid in f['transaction_ids']})
    if rolled:
        accepted = [t for t in accepted if t['case_id'] not in rolled]
        require(bool(accepted), 'All changes rolled back; no distinct candidate')
        final_source, final_target, ledger = edits.apply_transactions(source, target, accepted)
        checks = _global(folder, out, source, final_target, target, accepted, pack, bound_map, checker, 'assembled', track)
    require(final_source == source, 'Source changed in Chinese-only candidate')
    require(_same(candidate['accepted_transactions'], accepted) and _same(candidate['transaction_ledger'], ledger)
        and candidate['rolled_back_transactions'] == rolled and _same(candidate['global_checks'], checks), 'Assembly or rollback ledger changed')
    require(set(checks) == set(rw.rowmap(source)) and candidate['coverage'] == 66 and candidate['local_gate_passed'] is True
        and not any(f['quotes_valid'] for row in checks.values() for f in row['regressions']), 'Complete native local gate did not pass')
    require(candidate['cases_considered'] == len(cases) and candidate['valid_proposals'] == sum(t['valid'] and not t['no_op'] for t in proposed),
        'Candidate case counts changed')
    for name, expected_rows in [('source.json', final_source), ('target.json', final_target)]:
        track(out/name)
        require(file_hash(out/name) == candidate[name.replace('.json', '_sha256')] and old._rows(out/name) == expected_rows,
                'Candidate owner artifact changed')
    changed = [a.index for a, b in zip(target, final_target) if a.text != b.text]
    require(bool(changed) and candidate['changed_owners'] == candidate['changed_target_owners'] == changed, 'Changed owner ledger differs')
    sr.validate_candidate_artifacts(folder, rid)
    for name in ('display.json', 'source.srt', 'subtitles.zh.srt'):
        track(out/'display'/name, raw=True)
    display = old.display_checks._display(out/'display', source, final_target)
    return final_target, {'cases_considered': len(cases), 'accepted_before_rollback': first_count,
        'accepted_transactions': len(accepted), 'rolled_back_transactions': len(rolled), 'owner_coverage': 66,
        'changed_target_owners': len(changed), 'display': display}


def _ancestry(folder, conf, source, target, pools, track):
    parent = Path(conf['predecessor']).resolve(strict=True)
    path = parent/'campaign.json'
    require(conf['input_hashes'].get(str(path)) == file_hash(path), 'N1 campaign was not input-pinned')
    inherited = track(path)
    require(inherited.get('original_backend_restored') is True
        and inherited.get('status') not in ('prepared', 'running') and not inherited.get('eligible_candidate'),
        'N1 did not finish safely below target')
    for key in ('evidence_paths', 'pool_version', 'baseline_review', 'media', 'media_sha256',
                'writer_recipe', 'writer_recipe_sha256', 'starting_candidate'):
        require(_same(conf[key], inherited[key]), 'T1 changed inherited N1 inputs')
    for name, key in [('source.json', 'source_sha256'), ('target.json', 'target_sha256'),
                      ('original-context.txt', 'context_sha256')]:
        original = parent/name
        track(original, raw=True); track(folder/name, raw=True)
        require(conf['input_hashes'].get(str(original)) == file_hash(original)
            and conf['input_hashes'].get(str(folder/name)) == file_hash(folder/name)
            and file_hash(original) == file_hash(folder/name) == conf[key] == inherited[key],
            'Original N1 source, G1 Chinese or context changed')
    directory = Path(conf['short_evidence_directory']).resolve(strict=True)
    require(directory == parent/'evidence/short'
        and str(directory/'evidence.json') in conf['evidence_paths'], 'Wrong imported N1 short evidence')
    short = short_audio.validate_evidence(directory)
    require(any(_same(short, pool) for pool in pools), 'Imported short evidence differs from local pool')
    baseline = er.validate_receipt(conf['baseline_review'])
    require(baseline['purpose'] == 'baseline' and baseline['pool_version'] == conf['pool_version']
        and baseline['score'] == conf['starting_candidate_score'] == 2,
        'Unchanged N1 expanded-pool baseline must remain score two')
    for key, rows in [('source', source), ('raw', source), ('target', target)]:
        require(baseline['inputs'][key]['sha256'] == er.owner_rows_sha256(rows),
                'Inherited baseline owner input changed')
    require(baseline['inputs']['context']['sha256'] == conf['context_sha256'],
            'Inherited baseline context changed')
    for name, digest in baseline['receipt_hashes'].items():
        path = Path(baseline['output_dir'])/name
        track(path, raw=True)
        require(conf['input_hashes'].get(str(path)) == file_hash(path) == digest,
                'Original N1 baseline receipt was not retained')
    return baseline


def _below_target_score(folder, conf, source, target, baseline, track):
    score = track(folder/'T1/score.json'); target_hash = er.owner_rows_sha256(target)
    require(score['status'] == 'complete' and score['pool_version'] == conf['pool_version']
        and score['target_sha256'] == target_hash, 'Final scalar does not bind the temporal candidate')
    if score.get('inherited_duplicate_target'):
        path = Path(score['inherited_duplicate_target']).resolve(strict=True)
        require(str(path) in conf.get('prior_score_paths', [])
            and conf['input_hashes'].get(str(path)) == file_hash(path), 'Duplicate score ancestry was not pinned')
        inherited = track(path)
        inherited_score = min(inherited['score'], inherited.get('confirmation_score', inherited['score']))
        require(inherited['status'] == 'complete' and inherited['pool_version'] == score['pool_version']
            and inherited['target_sha256'] == target_hash and inherited_score == score['score']
            and score.get('confirmed') is False and not score.get('primary') and not score.get('confirmation'),
            'Duplicate target must retain its existing scalar without reroll')
        return score['score']
    require(not score.get('inherited_unchanged_baseline'), 'No distinct candidate exists')
    for name in conf.get('prior_score_paths', []):
        prior = track(name)
        require(not (prior.get('status') == 'complete' and prior.get('pool_version') == conf['pool_version']
            and prior.get('target_sha256') == target_hash), 'Previously scored target received a fresh reroll')
    primary = er.validate_receipt(score['primary'])
    require(primary['purpose'] == 'candidate' and primary['score'] == score['score']
        and primary['pool_version'] == conf['pool_version'], 'Native temporal primary differs')
    dependency = lambda value: {key: value[key] for key in (
        'purpose', 'output_dir', 'dispatch_id', 'finished_utc', 'receipt_hashes')}
    require(primary['dependency'] == dependency(baseline)
        and primary['dispatch_id'] != baseline['dispatch_id'], 'Primary review is not a fresh dependent dispatch')
    for key, rows in [('source', source), ('raw', source), ('target', target)]:
        require(primary['inputs'][key]['sha256'] == er.owner_rows_sha256(rows), 'Primary owner input changed')
    require(all(primary['inputs'][key]['sha256'] == baseline['inputs'][key]['sha256']
        for key in ('source', 'raw', 'context', 'evidence')), 'Primary changed the N1 benchmark')
    final_score = primary['score']; receipts = [primary]
    if score.get('confirmation'):
        confirmation = er.validate_receipt(score['confirmation'])
        require(confirmation['purpose'] == 'confirmation' and confirmation['score'] == score['confirmation_score']
            and confirmation['inputs'] == primary['inputs']
            and confirmation['pool_version'] == primary['pool_version']
            and confirmation['dependency'] == dependency(primary)
            and len({baseline['dispatch_id'], primary['dispatch_id'], confirmation['dispatch_id']}) == 3,
            'Native confirmation is not an independent dependent review')
        receipts.append(confirmation); final_score = min(final_score, confirmation['score'])
    for receipt in receipts:
        for name, digest in receipt['receipt_hashes'].items():
            path = Path(receipt['output_dir'])/name
            track(path, raw=True); require(file_hash(path) == digest, 'Native scalar receipt changed')
    return final_score


def validate_campaign(folder: str | Path, *, require_four: bool = True) -> dict:
    """Replay T1 without inference; eligibility always requires two fresh fours."""
    folder = Path(folder).resolve(strict=True); conf = tr.check(folder)
    require(conf['original_backend_restored'] is True
        and conf['status'] in ('confirmed_four', 'candidate_budget_exhausted'),
        'Temporal campaign is not complete and restored')
    elapsed = conf.get('local_seconds')
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed < tr.LIMIT,
            'Temporal local budget was exceeded or is invalid')
    if require_four:
        require(conf['status'] == 'confirmed_four' and conf.get('eligible_candidate') == 'T1',
                'T1 has not reached a confirmed four')
    pins = {}
    def track(path, raw=False):
        path = Path(path)
        require(not path.is_symlink(), 'Symlink cannot serve as immutable temporal artifact')
        path = path.resolve(strict=True); data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        require(str(path) not in pins or pins[str(path)] == digest, 'Temporal artifact changed during replay')
        pins[str(path)] = digest
        return data if raw else strict_json(data)
    require(_same(track(folder/'campaign.json'), conf), 'Temporal campaign changed during validation')
    old._producer_snapshots(folder, conf, track)
    require({str(tr.ROOT/'src'/f'{name}.py') for name in (
        'temporal_revisit', 'temporal_release', 'temporal_evidence', 'temporal_source_frames')}
        <= set(conf['code_pins']), 'Missing temporal producer snapshots')
    for path, digest in conf['input_hashes'].items():
        track(path, raw=True); require(file_hash(Path(path)) == digest, 'Pinned temporal input changed')
    for path, digest in conf['code_pins'].items():
        track(path, raw=True)
        require(file_hash(Path(path)) == digest, 'Temporal replay code differs from measured producer')
    source, target = tr._source_pair(folder)
    require(len(source) == len(target) == 66
        and [(row.index, row.ts_line) for row in source] == [(row.index, row.ts_line) for row in target],
        'Original temporal owner geometry changed')
    pools = [er._validate_pool(Path(path)) for path in conf['evidence_paths']]
    require(bool(pools) and all(pool['media_sha256'] == conf['media_sha256'] for pool in pools),
            'Temporal evidence belongs to another recording')
    require(file_hash(Path(conf['media'])) == conf['media_sha256'], 'Recording identity changed')
    pins[str(Path(conf['media']).resolve())] = conf['media_sha256']
    baseline = _ancestry(folder, conf, source, target, pools, track)
    observations = sr.all_observations(list(map(Path, conf['evidence_paths'])), source)
    context = (folder/'original-context.txt').read_text(encoding='utf-8')
    pack, bound, analysis = _map_and_analysis(folder, conf, source, target, observations, context, track)
    rebuilt, local = _candidate(folder, conf, 'T1', source, target, pack, bound, analysis, track,
        candidate_version=tr.VERSION, roles=('gemma', 'qwen'))
    if require_four:
        primary, confirmation = replay._scores(folder, conf, 'T1', source, rebuilt, pools, track)
        final_score = min(primary['score'], confirmation['score'])
    else:
        final_score = _below_target_score(folder, conf, source, rebuilt, baseline, track)
    for name in ('source.json', 'target.json', 'source.utterances.srt', 'target.utterances.srt'):
        track(folder/'T1'/name, raw=True)
    require(rw.rows(folder/'T1/source.json') == source, 'Japanese source changed')
    frames = track(folder/'shared/source-frames.json')
    require(all(file_hash(Path(path)) == digest for path, digest in pins.items()),
            'Temporal artifact changed during replay')
    return {'version': tr.VERSION, 'round': 'T1',
        'status': 'contextual_ready' if require_four else 'replayed_below_target',
        'score': final_score, 'local': local, 'fallback_owner_ids': frames['fallback_owner_ids'],
        'artifacts': pins, 'audio_truth_verified': False, 'source_accuracy_verified': False,
        'playback_verified': False, 'selected_output_modified': False}


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
