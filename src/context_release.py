"""Read-only C1/C2 replay and explicit reference manifest; never model dispatch.

A contextual four is not verified audio/source fidelity or playback approval.
Native scalar receipts, source-frame ancestry, the fallible map, every local
proposal/check and complete assembly are replayed before a reference can qualify.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from src import context_revisit as cr
from src import context_recap as recap
from src import evidence_context as ec
from src import sparse_release as old
from src import sparse_revisit as sr
from src import sparse_edits as edits
from src import revisit_workflow as rw
from src import contextual_evidence_review as er
from src.contextual_review_contract import require, strict_json
from src.workflow_state import file_hash, fingerprint

VERSION = 'context-recap-reference-1'


def _same(a, b):
    return fingerprint(a) == fingerprint(b)


def _bound(path, binding, track):
    value = track(path); receipt = track(path.with_suffix('.binding.json'))
    require(set(receipt) == {'binding', 'sha256'} and _same(receipt['binding'], binding)
            and receipt['sha256'] == file_hash(path), 'Stage input/output binding changed')
    return value


def _native(path, instruction, body, schema, family, track, *, pins=None):
    """Use original native stage/metrics for S-frame aliases, never fake C stages."""
    parsed = old._native(path, instruction, body, schema, family, track)
    receipt = track(path)
    paths = [Path(path)]
    if receipt.get('status') == 'reused':
        path = Path(receipt['inherited_from']); paths.append(path); receipt = track(path)
    paths.append(Path(path).parent/'metrics.jsonl')
    if pins is not None:
        require(all(pins.get(str(p.resolve())) == file_hash(p) for p in paths),
                'Inherited native source receipt was not input-pinned')
    cap = receipt['attempts'][-1]['capacity']
    maximum = 12288 if family == 'qwen' else 6144
    require(all(type(cap.get(k)) is int for k in ('prompt_tokens', 'reserved_output', 'context'))
        and cap['prompt_tokens'] > 0 and cap['reserved_output'] == maximum
        and cap['context'] == (196608 if family == 'qwen' else 32768)
        and cap['prompt_tokens'] + maximum + 64 <= cap['context'], 'Native capacity arithmetic changed')
    return parsed


def _inherited_producers(folder, conf, track):
    """Verify retained S-frame code, including earlier cache-recovery epochs."""
    parent = Path(conf['predecessor']).resolve()
    path = Path(conf.get('inherited_producer_metadata', '')).resolve()
    require(path == folder/'inherited-producer-metadata.json', 'Missing inherited producer metadata')
    pins = conf['input_hashes']
    require(pins.get(str(path)) == file_hash(path), 'Inherited producer metadata was not input-pinned')
    metadata = track(path)
    require(isinstance(metadata, dict) and set(metadata) == {'predecessor', 'code_pins', 'code_epochs', 'producer_snapshot_manifest'}
        and metadata['predecessor'] == str(parent), 'Inherited producer identity changed')
    count = old._producer_snapshots(parent, metadata, track)
    for epoch in [*metadata['code_epochs'], metadata]:
        manifest_path = Path(epoch['producer_snapshot_manifest']).resolve()
        require(pins.get(str(manifest_path)) == file_hash(manifest_path), 'Inherited producer manifest was not input-pinned')
        manifest = track(manifest_path)
        for spec in manifest.values():
            snapshot = Path(spec['snapshot']).resolve()
            require(pins.get(str(snapshot)) == file_hash(snapshot), 'Inherited producer snapshot was not input-pinned')
        runtime = Path(manifest[str(cr.ROOT/'src/sparse_revisit.py')]['snapshot'])
        require(old._syntax(runtime, {'SOURCE', 'reading_schema', 'analyze'})
            == old._syntax(Path(sr.__file__), {'SOURCE', 'reading_schema', 'analyze'}),
            'Inherited source-frame replay semantics changed')
        require(manifest[str(cr.ROOT/'src/revisit_workflow.py')]['sha256'] == file_hash(Path(rw.__file__)),
            'Inherited source observation/schema helpers changed')
    return count


def _source_frames(folder, conf, source, target, observations, context, track):
    parent = Path(conf['predecessor']).resolve()
    path = Path(conf['inherited_analysis']).resolve()
    require(path == parent/'S2/analysis-expanded.json', 'Wrong source-frame ancestry')
    inherited = track(path); copied = track(folder/'inherited-source-analysis.json')
    require(file_hash(path) == conf['inherited_analysis_sha256']
        and file_hash(folder/'inherited-source-analysis.json') == conf['inherited_analysis_sha256']
        and _same(inherited, copied), 'Inherited source-frame artifact changed')
    require(inherited.get('version') == sr.VERSION and inherited.get('source_hash') == conf['source_sha256']
        and inherited.get('target_hash') == conf['target_sha256'] and inherited.get('evidence_paths') == conf['evidence_paths']
        and inherited.get('coverage') == list(range(1, 67)) and inherited.get('external_feedback_used') is False,
        'Inherited source analysis has different inputs/coverage')
    rebuilt = {}
    for number, ids in enumerate(sr.chunk_ids(source), 1):
        relevant = [o for o in observations if o['owner_id'] in ids]; allowed = [o['observation_id'] for o in relevant]
        schema = rw.keyed(ids, rw.obj({'readings': rw.arr(sr.reading_schema(allowed), 10), 'unresolved': {'type': 'boolean'},
            'summary_ja': rw.string(1000), 'unknowns': rw.arr(rw.string(500), 8)}))
        body = {'japanese': rw.rowmap(source), 'original_context': context, 'focus_ids': ids, 'observations': relevant,
                'ownership_warning': 'padded_crops_can_contain_neighboring_speech'}
        value = _native(parent/'S2/requests'/f'expanded-source-{number}.json', sr.SOURCE, body, schema,
                        'qwen', track, pins=conf['input_hashes'])
        by_id = {o['observation_id']: o for o in relevant}
        for owner, frame in value.items():
            expected = {o['observation_id'] for o in relevant if o['owner_id'] == int(owner)}
            valid = (len(frame['readings']) == len(expected)
                and {r['observation_id'] for r in frame['readings']} == expected
                and all(r['quote'] and r['quote'] in by_id[r['observation_id']]['text'] for r in frame['readings']))
            if not valid:
                frame = {'readings': [{'observation_id': o['observation_id'], 'quote': o['text'], 'interpretation_ja': '未確定',
                    'speech_act': '未確定', 'viable': True} for o in relevant if o['owner_id'] == int(owner)],
                    'unresolved': True, 'summary_ja': '未確定', 'unknowns': ['local_source_analysis_failed_literal_coverage']}
            rebuilt[owner] = {**frame, 'source_accuracy_verified': False, 'literal_observation_coverage_valid': valid}
    require(_same(rebuilt, inherited['frames']), 'Inherited source frames do not replay native answers/fallback')
    return rebuilt


def _recap_history(folder, conf, pack, track):
    """Retain failed length-limited attempts as cost/provenance, never a valid map."""
    if not conf.get('code_epochs'):
        return
    directory = folder/'recovery-compact-recap1'
    record_path = directory/'recovery.json'
    paths = [record_path, directory/'campaign-before.json', directory/'source-recap.json', directory/'metrics.jsonl',
             folder/'C1/requests/source-recap.json']
    require(all(conf['input_hashes'].get(str(p)) == file_hash(p) for p in paths), 'Compact recovery provenance was not pinned')
    record = track(record_path); before = track(directory/'campaign-before.json')
    require(record['old_request_key'] == 'source-recap' and record['new_request_key'] == cr.RECAP_REQUEST_KEY
        and record['new_maximum_claims'] == cr.RECAP_MAX_CLAIMS == 8 and record['full_source_input_unchanged'] is True
        and record['candidate_limit_unchanged'] == 2 and record['maximum_local_seconds_unchanged'] == cr.LIMIT
        and record['maximum_round_seconds_unchanged'] == cr.ROUND_LIMIT and record['maximum_new_attempts'] == 2
        and record['additional_recovery_after_this'] is False, 'Compact recovery exceeded its declared contract')
    preserved = record['preserved_files']
    require(set(preserved) == {str(p) for p in paths[1:4]}
        and all(preserved[str(p)] == file_hash(p) for p in paths[1:4]), 'Preserved recap files changed')
    elapsed = record['local_seconds_preserved']
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0
        and elapsed == before['local_seconds'] and conf['local_seconds'] >= elapsed,
        'Failed recap time was not retained')
    require(before['status'] == 'failed' and before['original_backend_restored'] is True
        and not before.get('eligible_candidate') and all(_same(before[k], conf[k]) for k in
            ('source_sha256', 'target_sha256', 'context_sha256', 'source_pack_sha256', 'pool_version', 'evidence_paths')),
        'Compact recovery changed source/evidence or restarted a qualified candidate')
    require(any(_same(epoch['code_pins'], before['code_pins'])
        and epoch['producer_snapshot_manifest'] == before['producer_snapshot_manifest'] for epoch in conf['code_epochs']),
        'Failed recap producer epoch was not retained')
    failed = track(directory/'source-recap.json')
    require(file_hash(directory/'source-recap.json') == file_hash(paths[-1]) and failed['status'] == 'failed'
        and isinstance(failed['attempts'], list) and len(failed['attempts']) == 2, 'Original failed recap attempts changed')
    track(paths[-1])
    prepared = recap.prepare_recap(pack['source_rows'], pack['observations'], pack['frames'], pack['original_context'], pool_versions=pack['pool_versions'])
    current = track(folder/'C1/requests'/(cr.RECAP_REQUEST_KEY+'.json'))
    expected = deepcopy(current['request'])
    expected.update({k: prepared[k] for k in ('instruction', 'body', 'schema')})
    expected['extra']['response_format']['schema'] = prepared['schema']
    require(_same(failed['request'], expected) and failed['request_sha256'] == fingerprint(expected),
        'Failed recap did not use the same complete source pack/native policy')
    metrics = [strict_json(line) for line in track(directory/'metrics.jsonl', raw=True).decode('utf-8').splitlines() if line.strip()]
    require(len(metrics) == 2 and all(m.get('stage') == 'source-recap' and m.get('finish_reason') == 'length'
        and type(a.get('raw')) is str and m.get('answer_chars') == len(a['raw'])
        for m, a in zip(metrics, failed['attempts'])), 'Failed recap native length receipts changed')


def _map_and_analysis(folder, conf, source, target, observations, context, track):
    frames = _source_frames(folder, conf, source, target, observations, context, track)
    prepared = recap.prepare_recap(source, observations, frames, context, pool_versions=[conf['pool_version']])
    pack = track(folder/'source-pack.json')
    require(_same(pack, prepared['pack']) and pack['pack_sha256'] == conf['source_pack_sha256'], 'Source pack differs from pinned ancestry')
    _recap_history(folder, conf, pack, track)
    prepared = cr.prepare_recap_request(pack)
    binding = {'pack_sha256': pack['pack_sha256'], 'request': fingerprint({k: prepared[k] for k in ('instruction', 'body', 'schema')})}
    bound_map = _bound(folder/'shared/source-map.json', binding, track)
    native_map = _native(folder/'C1/requests'/(cr.RECAP_REQUEST_KEY+'.json'), prepared['instruction'], prepared['body'], prepared['schema'], 'qwen', track)
    require(_same(bound_map, ec.validate_context_map(native_map, pack)), 'Map does not replay the native source-only recap')
    recap.focus_context(pack, bound_map, [1])
    binding = {'pack_sha256': pack['pack_sha256'], 'map_sha256': bound_map['map_sha256'],
               'target_sha256': conf['target_sha256'], 'instruction': fingerprint(cr.DIAGNOSE)}
    analysis = _bound(folder/'shared/analysis.json', binding, track)
    cases = []; invalid = []; coverage = []
    case_schema = rw.obj({'target_span': rw.string(1000), 'source_span': rw.string(1000),
        'category': {'type': 'string', 'enum': ['mistranslation', 'omission', 'unsupported_specificity', 'negation', 'intent', 'terminology']},
        'severity': {'type': 'integer', 'minimum': 1, 'maximum': 3}, 'issue': rw.string(1200), 'unresolved': {'type': 'boolean'}})
    for number, ids in enumerate(sr.chunk_ids(source), 1):
        body = {**sr.episode_body(source, target, context), 'focus_ids': ids,
            'source_readings': {str(i): pack['frames'][str(i)] for i in ids},
            'focused_context': recap.focus_context(pack, bound_map, ids)}
        result = _native(folder/'C1/requests'/f'discrepancy-{number}.json', cr.DIAGNOSE, body,
            rw.keyed(ids, rw.obj({'cases': rw.arr(case_schema, 2)})), 'qwen', track)
        coverage.extend(ids)
        for owner, row in result.items():
            for ordinal, case in enumerate(row['cases'], 1):
                case = {**case, 'owner_id': int(owner), 'case_id': f'C1-o{int(owner):03d}-c{ordinal}',
                    'readings': pack['frames'][owner]['readings'], 'unresolved': case['unresolved'] or pack['frames'][owner]['unresolved']}
                try:
                    cases.append(edits.validate_case(case, source, target, observations))
                except ValueError as exc:
                    invalid.append({'case_id': case['case_id'], 'error_type': type(exc).__name__})
    expected = {'version': cr.VERSION, 'cases': cases, 'invalid_cases': invalid, 'coverage': coverage,
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
            'focused_context': recap.focus_context(pack, bound_map, ids)}
        value = _native(out/'requests'/f'{prefix}-{number}.json', cr.GLOBAL, body,
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
    require(candidate.get('version') == (cr.VERSION if candidate_version is None else candidate_version) and candidate.get('round') == rid
        and candidate.get('source_edits_allowed') is False and candidate.get('raw_source_fidelity_verified') is False
        and candidate.get('external_feedback_used') is False and candidate.get('human_reference_used') is False,
        'Candidate scope or identity changed')
    require(candidate.get('evidence_paths') == conf['evidence_paths'] and candidate.get('pack_sha256') == pack['pack_sha256']
        and candidate.get('map_sha256') == bound_map['map_sha256'], 'Candidate context evidence changed')
    require(_same(track(out/'analysis.json'), analysis), 'C2 or candidate diagnosis differs from shared C1 analysis')
    cases = sr.choose_cases(analysis, 24)
    require(_same(track(out/'selected-cases.json'), cases), 'Selected cases changed')
    proposed = track(out/'proposed-transactions.json')
    require(isinstance(proposed, list) and len(proposed) == len(cases), 'Proposal coverage changed')
    writer, checker = roles if roles is not None else (('gemma', 'qwen') if rid == 'C1' else ('qwen', 'gemma'))
    require(candidate.get('writer') == writer and candidate.get('checker') == checker, 'Wrong C writer/checker allocation')
    decisions = []; accepted = []
    for case, transaction in zip(cases, proposed):
        focused = recap.focus_context(pack, bound_map, [case['owner_id']])
        native = (writer_native or _native)(out/'requests'/('propose-'+case['case_id']+'.json'), cr.PROPOSE,
            {**sr.episode_body(source, target, pack['original_context']), 'case': case, 'focused_context': focused},
            cr.PROPOSAL_SCHEMA, writer, track)
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
        verdict = _native(out/'requests'/('verify-'+case['case_id']+'.json'), cr.VERIFY, body,
            sr.decision_schema([o['observation_id'] for o in own]), checker, track)
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


def _scores(folder, conf, rid, source, target, local_pools, track):
    score_path = Path(conf.get('eligible_score_receipt', folder/rid/'score.json')).resolve()
    require(score_path == folder/rid/'score.json', 'Wrong eligible score receipt')
    score = track(score_path)
    require(score.get('status') == 'complete' and score.get('confirmed') is True and score.get('local_gate_passed') is True
        and not score.get('inherited_duplicate_target') and not score.get('inherited_unchanged_baseline'), 'Score is inherited/incomplete or local gate failed')
    prior_paths = [Path(p).resolve() for p in conf.get('prior_score_paths', [])]
    require(all(conf['input_hashes'].get(str(p)) == file_hash(p) for p in prior_paths), 'Historical duplicate scores were not pinned')
    if rid == 'C2' and (folder/'C1/score.json').exists():
        prior_paths.append(folder/'C1/score.json')
    for path in prior_paths:
        prior = track(path)
        require(not (prior.get('status') == 'complete' and prior.get('pool_version') == conf['pool_version']
            and prior.get('target_sha256') == er.owner_rows_sha256(target)), 'Previously scored target cannot receive a fresh reroll')
    primary = er.validate_receipt(score['primary']); confirmation = er.validate_receipt(score['confirmation'])
    baseline = er.validate_receipt(conf['baseline_review'])
    require(primary['purpose'] == 'candidate' and confirmation['purpose'] == 'confirmation' and baseline['purpose'] == 'baseline', 'Wrong native score roles')
    dependency = lambda r: {k: r[k] for k in ('purpose', 'output_dir', 'dispatch_id', 'finished_utc', 'receipt_hashes')}
    require(primary['dependency'] == dependency(baseline) and confirmation['dependency'] == dependency(primary)
        and len({r['dispatch_id'] for r in (baseline, primary, confirmation)}) == 3, 'Reviews are not fresh designated dependent dispatches')
    require(datetime.fromisoformat(baseline['finished_utc']) <= datetime.fromisoformat(primary['started_utc'])
        and datetime.fromisoformat(primary['finished_utc']) <= datetime.fromisoformat(confirmation['started_utc']), 'Reviews are not sequential')
    require(all(type(r['score']) is int and 4 <= r['score'] <= 5 for r in (primary, confirmation))
        and score.get('score') == primary['score'] and score.get('confirmation_score') == confirmation['score'], 'Both genuine reviews must score four')
    require(primary['inputs'] == confirmation['inputs']
        and primary['pool_version'] == confirmation['pool_version'] == baseline['pool_version'] == conf['pool_version'] == score['pool_version'], 'Review inputs/pool differ')
    for key, rows in [('source', source), ('raw', source), ('target', target)]:
        require(primary['inputs'][key]['sha256'] == er.owner_rows_sha256(rows), 'Reviewed owner wording differs')
    require(primary['inputs']['context']['sha256'] == conf['context_sha256']
        and score.get('source_sha256') == er.owner_rows_sha256(source) and score.get('target_sha256') == er.owner_rows_sha256(target), 'Score artifact input binding changed')
    base_target = old._rows(folder/'target.json')
    require(baseline['inputs']['target']['sha256'] == er.owner_rows_sha256(base_target)
        and all(baseline['inputs'][key]['sha256'] == primary['inputs'][key]['sha256'] for key in ('source', 'raw', 'context', 'evidence')), 'Baseline has different inputs')
    prep = track(Path(primary['output_dir'])/'preparation.json'); track(Path(prep['manifest_path']))
    bundle = er.read_bundle(prep['manifest_path'])
    require(bundle['pool_version'] == conf['pool_version'], 'Native scoring bundle pool changed')
    reviewed = strict_json(bundle['raw_inputs']['evidence'])
    require(sorted(fingerprint(p) for p in reviewed) == sorted(fingerprint(p) for p in local_pools), 'Reviewed acoustic pools differ from local inputs')
    for receipt in (baseline, primary, confirmation):
        for name, digest in receipt['receipt_hashes'].items():
            path = Path(receipt['output_dir'])/name; track(path, raw=True)
            require(file_hash(path) == digest, 'Native review receipt changed')
    return primary, confirmation


def validate_campaign(folder: str | Path) -> dict:
    """Return a context-only eligibility summary after full native artifact replay."""
    folder = Path(folder).expanduser().resolve(strict=True); artifacts = {}
    def track(path, *, raw=False):
        path = Path(path); require(not path.is_symlink(), 'Symlink cannot serve as immutable artifact')
        path = path.resolve(strict=True); data = path.read_bytes(); artifacts[str(path)] = file_hash(path)
        return data if raw else strict_json(data)
    conf = track(folder/'campaign.json')
    require(conf.get('version') == cr.VERSION and conf.get('status') == 'confirmed_four'
        and conf.get('original_backend_restored') is True, 'Campaign is not confirmed/restored')
    require(conf.get('maximum_candidates') == 2 and conf.get('maximum_local_seconds') == cr.LIMIT
        and conf.get('maximum_round_seconds') == cr.ROUND_LIMIT and conf.get('external_feedback') == 'score_only'
        and all(conf.get(k) is False for k in ('source_edits_allowed', 'raw_source_fidelity_verified', 'human_reference_used',
            'visual_input_used', 'new_audio_acquired', 'selected_output_modified')), 'Campaign scope or budget changed')
    rid = conf.get('eligible_candidate'); require(rid in cr.ROUNDS, 'Unknown eligible C round')
    pins = conf.get('input_hashes'); require(isinstance(pins, dict) and bool(pins), 'Input pins missing')
    for path, digest in pins.items():
        track(Path(path), raw=True); require(file_hash(Path(path)) == digest, 'Pinned input changed')
    for name, key in [('source.json', 'source_sha256'), ('target.json', 'target_sha256'), ('original-context.txt', 'context_sha256')]:
        path = folder/name; track(path, raw=True)
        require(pins.get(str(path)) == conf.get(key) == file_hash(path), 'Immutable base/context binding changed')
    require(conf['writer_recipe_sha256'] == file_hash(Path(conf['writer_recipe'])) and conf['plan_sha256'] == file_hash(Path(conf['plan'])), 'Recipe/plan binding changed')
    require(file_hash(Path(conf['media'])) == conf['media_sha256'], 'Recording identity changed')
    artifacts[str(Path(conf['media']).resolve())] = conf['media_sha256']
    count = old._producer_snapshots(folder, conf, track)
    require({str(cr.ROOT/'src'/f'{name}.py') for name in ('context_revisit', 'context_recap', 'evidence_context')} <= set(conf['code_pins']), 'Missing context producer snapshots')
    for path, digest in conf['code_pins'].items():
        require(file_hash(Path(path)) == digest, 'Current replay helper differs from measured producer')
    inherited_count = _inherited_producers(folder, conf, track)
    source, target = old._rows(folder/'source.json'), old._rows(folder/'target.json')
    require([(r.index, r.ts_line) for r in source] == [(r.index, r.ts_line) for r in target], 'Base owner geometry differs')
    context = (folder/'original-context.txt').read_text(encoding='utf-8')
    pools = [er._validate_pool(Path(path)) for path in conf['evidence_paths']]
    require(bool(pools) and all(p['media_sha256'] == conf['media_sha256'] for p in pools), 'Evidence belongs to another recording')
    for path in conf['evidence_paths']:
        track(Path(path)); require(str(Path(path).resolve()) in pins, 'Evidence pool was not pinned')
    observations = rw.observations(pools, source)
    pack, bound_map, analysis = _map_and_analysis(folder, conf, source, target, observations, context, track)
    final_target, local = _candidate(folder, conf, rid, source, target, pack, bound_map, analysis, track)
    primary, confirmation = _scores(folder, conf, rid, source, final_target, pools, track)
    require(all(file_hash(Path(path)) == digest for path, digest in artifacts.items()), 'Artifact changed during validation')
    return {'version': VERSION, 'status': 'contextual_ready', 'campaign': str(folder), 'round': rid,
        'score': min(primary['score'], confirmation['score']), 'pool_version': conf['pool_version'], 'owner_count': 66,
        'pack_sha256': pack['pack_sha256'], 'map_sha256': bound_map['map_sha256'], 'context_claim_count': len(bound_map['claims']),
        'local': local, 'producer_snapshot_count': count, 'inherited_producer_snapshot_count': inherited_count,
        'artifacts': artifacts, 'reference_only': True,
        'source_accuracy_verified': False, 'audio_truth_verified': False, 'playback_verified': False,
        'coverage_is_read_attestation': False, 'snapshot_predispatch_attested': False, 'selected_output_modified': False,
        'reviews': {name: {k: r[k] for k in ('score', 'output_dir', 'dispatch_id', 'receipt_hashes')}
                    for name, r in [('primary', primary), ('confirmation', confirmation)]}}


def prepare_reference(folder: str | Path, dest: str | Path) -> Path:
    """Explicitly write a small reference manifest only; never promote artifacts."""
    dest = Path(dest).expanduser().absolute(); require(not dest.exists(), 'Reference destination already exists')
    result = validate_campaign(folder); dest.mkdir(parents=True, exist_ok=False)
    path = dest/'reference-ready.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump({**result, 'created_utc': datetime.now(timezone.utc).isoformat()}, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('campaign', type=Path)
    parser.add_argument('--prepare-reference', type=Path); args = parser.parse_args()
    if args.prepare_reference:
        print(prepare_reference(args.campaign, args.prepare_reference))
    else:
        result = validate_campaign(args.campaign)
        print(json.dumps({key: result[key] for key in ('status', 'round', 'score', 'owner_count', 'context_claim_count')}))


if __name__ == '__main__':
    main()
