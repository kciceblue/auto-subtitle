"""Read-only W2 native replay, including source-only T1 ancestry and scalar gates.

This never dispatches models, modifies subtitles, or certifies audio truth. A
reference requires a complete local regression gate and two fresh scalar fours.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import math
import re
from pathlib import Path

from src import compact_owner_revisit as controller
from src import whole_owner as whole
from src import whole_owner_revisit as previous_controller
from src import whole_owner_release as previous_gate
from src import whole_owner_requests as previous_requests
from src import compact_owner_requests as requests
from src import temporal_revisit as temporal
from src import temporal_release as temporal_gate
from src import context_release as replay
from src import sparse_release as old
from src import sparse_revisit as sparse
from src import evidence_context as evidence
from src import revisit_workflow as workflow
from src import contextual_evidence_review as review
from src.contextual_review_contract import require, strict_json
from src.workflow_state import file_hash, fingerprint, write_json
from src.translate import parse_srt

_same = replay._same


def _source_ancestry(folder, conf, source, target, pools, track):
    """Replay only inherited source frames/recap; no prior Chinese diagnosis."""
    parent = Path(conf['predecessor']).resolve(strict=True)
    parent_path = parent/'campaign.json'
    require(conf['input_hashes'].get(str(parent_path)) == file_hash(parent_path),
            'T1 campaign was not input-pinned')
    inherited = temporal.check(parent)
    require(_same(track(parent_path), inherited), 'T1 campaign changed during replay')
    require(inherited.get('original_backend_restored') is True
        and inherited.get('status') not in ('prepared', 'running')
        and not inherited.get('eligible_candidate'), 'T1 did not finish safely below target')
    for key in ('evidence_paths', 'pool_version', 'baseline_review', 'media', 'media_sha256',
                'writer_recipe', 'writer_recipe_sha256', 'short_evidence_directory', 'starting_candidate'):
        require(_same(conf[key], inherited[key]), 'W2 changed inherited T1 inputs')
    for name, key in [('source.json', 'source_sha256'), ('target.json', 'target_sha256'),
                      ('original-context.txt', 'context_sha256')]:
        original, copied = parent/name, folder/name
        track(original, raw=True); track(copied, raw=True)
        require(conf['input_hashes'].get(str(original)) == file_hash(original)
            and conf['input_hashes'].get(str(copied)) == file_hash(copied)
            and file_hash(original) == file_hash(copied) == conf[key] == inherited[key],
            'Original source, G1 Chinese or context changed')
    baseline = temporal_gate._ancestry(parent, inherited, source, target, pools, track)
    require(baseline['score'] == 2, 'Expanded-pool baseline must remain score two')
    for path, digest in inherited['input_hashes'].items():
        track(path, raw=True)
        require(file_hash(Path(path)) == digest, 'Inherited T1 input changed')

    native_pins = {}
    def native_track(path, raw=False):
        value = track(path, raw=raw); path = Path(path).resolve(strict=True)
        digest = file_hash(path)
        require(str(path) not in native_pins or native_pins[str(path)] == digest,
                'Inherited source artifact changed during replay')
        native_pins[str(path)] = digest
        require(conf['input_hashes'].get(str(path)) == digest,
                'Inherited source native artifact was not input-pinned')
        return value

    old._producer_snapshots(parent, inherited, native_track)
    required = {str(temporal.ROOT/'src'/f'{name}.py') for name in
        ('temporal_revisit', 'temporal_release', 'temporal_evidence', 'temporal_source_frames')}
    require(required <= set(inherited['code_pins']), 'Missing inherited temporal producer snapshots')
    for path, digest in inherited['code_pins'].items():
        track(path, raw=True)
        require(file_hash(Path(path)) == digest, 'Inherited source replay producer changed')
    observations = sparse.all_observations(list(map(Path, conf['evidence_paths'])), source)
    context = (folder/'original-context.txt').read_text(encoding='utf-8')
    frames = temporal_gate._source_frames(parent, inherited, source, observations, context, native_track)
    pack = evidence.build_source_pack(source, observations, frames['frames'], context,
        pool_versions=[pool['pool_version'] for pool in pools])
    require(_same(pack, native_track(parent/'source-pack.json')), 'Inherited source pack differs from native replay')
    prepared = temporal.prepare_recap_request(pack)
    binding = {'pack_sha256': pack['pack_sha256'],
        'request': fingerprint({key: prepared[key] for key in ('instruction', 'body', 'schema')})}
    bound = replay._bound(parent/'shared/source-map.json', binding, native_track)
    raw = temporal_gate._native(parent/'T1/requests'/(temporal.RECAP_REQUEST_KEY+'.json'),
        prepared['instruction'], prepared['body'], prepared['schema'], 'qwen', native_track)
    require(_same(bound, temporal.bind_recap(raw, pack)), 'Inherited recap differs from native replay')
    require(_same(track(folder/'source-pack.json'), pack)
        and _same(track(folder/'source-map.json'), bound)
        and conf['source_pack_sha256'] == pack['pack_sha256']
        and conf['map_sha256'] == bound['map_sha256'], 'Copied source frames or recap changed')
    for name in ('source-pack.json', 'source-map.json', 'source-inheritance.json'):
        require(conf['input_hashes'].get(str(folder/name)) == file_hash(folder/name),
                'Inherited source artifact was not pinned')
    expected = {'predecessor': str(parent), 'native_artifacts': native_pins,
        'pack_sha256': pack['pack_sha256'], 'map_sha256': bound['map_sha256']}
    require(_same(track(folder/'source-inheritance.json'), expected), 'Source inheritance ledger differs from native replay')
    return pack, bound, baseline, frames['fallback_owner_ids']



def _capacity_failure_ancestry(folder, conf, source, target, pools, pack, bound, track):
    """Replay the immutable W1 failure; it must have made no generation calls."""
    name = conf.get('capacity_failure_predecessor')
    require(isinstance(name, str) and Path(name).is_absolute(), 'W1 capacity-failure predecessor is missing')
    parent = Path(name).resolve(strict=True)
    conf_path = parent/'campaign.json'; preflight_path = parent/'W1/writer-capacity-preflight.json'
    for path in (conf_path, preflight_path):
        require(conf['input_hashes'].get(str(path)) == file_hash(path),
                'W1 capacity-failure prerequisite was not input-pinned')
    prior = previous_controller.check(parent)
    require(_same(track(conf_path), prior) and prior.get('version') == previous_controller.VERSION
        and prior.get('status') == 'failed' and prior.get('error_type') == 'RuntimeError'
        and prior.get('original_backend_restored') is True and not prior.get('eligible_candidate'),
        'W1 must be a restored preflight failure without an eligible candidate')
    for key in ('source_sha256', 'target_sha256', 'context_sha256', 'evidence_paths', 'pool_version',
                'media', 'media_sha256', 'writer_recipe', 'writer_recipe_sha256', 'baseline_review', 'predecessor'):
        require(_same(conf[key], prior[key]), 'W2 changed the failed W1 comparison inputs')
    old._producer_snapshots(parent, prior, track)
    required = {str(previous_controller.ROOT/'src'/f'{name}.py') for name in
        ('whole_owner_revisit', 'whole_owner_release', 'whole_owner', 'whole_owner_requests')}
    require(required <= set(prior['code_pins']), 'Failed W1 producer snapshots are incomplete')
    for path, digest in {**prior['input_hashes'], **prior['code_pins']}.items():
        track(path, raw=True)
        require(file_hash(Path(path)) == digest, 'Failed W1 producer or input changed')
    prior_pack, prior_bound, _, _ = previous_gate._source_ancestry(parent, prior, source, target, pools, track)
    require(_same(pack, prior_pack) and _same(bound, prior_bound), 'W2 changed W1 source-only frames or recap')
    out = parent/'W1'; request_dir = out/'requests'
    require(not any(request_dir.glob('*.json')) and not (request_dir/'metrics.jsonl').exists(),
            'W1 generated native requests before its failed preflight')
    require(not any((out/name).exists() for name in ('candidate.json', 'target.json', 'source.json',
        'proposed-transactions.json', 'local-decisions.json', 'score.json', 'display')),
        'Failed W1 contains generated candidate artifacts')
    value = track(preflight_path)
    keys = {'version', 'round', 'family', 'status', 'expected_requests', 'requests_sha256',
        'coverage_owner_ids', 'started_utc', 'finished_utc', 'elapsed_seconds', 'records',
        'error_type', 'preflight_sha256'}
    original_requests = previous_controller.writer_requests(pack, bound)
    require(isinstance(value, dict) and set(value) == keys
        and value['version'] == previous_controller.WRITER_PREFLIGHT_VERSION
        and value['round'] == 'W1' and value['family'] == 'gemma' and value['status'] == 'failed'
        and value['error_type'] == 'RuntimeError' and value['expected_requests'] == 33
        and value['coverage_owner_ids'] == list(range(1, 67))
        and value['requests_sha256'] == fingerprint(original_requests)
        and value['preflight_sha256'] == fingerprint({key: item for key, item in value.items()
            if key != 'preflight_sha256'}), 'Failed W1 preflight envelope changed')
    records = value['records']
    require(isinstance(records, list) and 1 <= len(records) <= 33,
            'Failed W1 preflight must preserve its attempted prefix')
    started = _timestamp(value['started_utc'], 'W1 preflight start')
    finished = _timestamp(value['finished_utc'], 'W1 preflight finish')
    require(started <= finished and type(value['elapsed_seconds']) in (int, float)
        and math.isfinite(value['elapsed_seconds']) and value['elapsed_seconds'] >= 0,
        'W1 preflight timing changed')
    previous_finish = started
    record_keys = {'request_key', 'owner_ids', 'request_sha256', 'instruction_sha256', 'body_sha256',
        'schema_sha256', 'native_request_sha256', 'started_utc', 'finished_utc', 'elapsed_seconds',
        'capacity', 'error_type'}
    for ordinal, (record, request) in enumerate(zip(records, original_requests), 1):
        require(isinstance(record, dict) and set(record) == record_keys
            and record['request_key'] == f'writer-group-{ordinal}'
            and record['owner_ids'] == request['body']['focus_owner_ids']
            and record['request_sha256'] == fingerprint(request)
            and record['native_request_sha256'] == fingerprint(previous_controller.writer_native_request(request))
            and all(record[name+'_sha256'] == fingerprint(request[name]) for name in ('instruction', 'body', 'schema')),
            'Failed W1 preflight request prefix changed')
        start = _timestamp(record['started_utc'], 'W1 group start')
        end = _timestamp(record['finished_utc'], 'W1 group finish')
        require(previous_finish <= start <= end <= finished
            and type(record['elapsed_seconds']) in (int, float)
            and math.isfinite(record['elapsed_seconds']) and record['elapsed_seconds'] >= 0,
            'Failed W1 preflight group timing changed')
        previous_finish = end
        if ordinal < len(records):
            require(record['error_type'] is None, 'W1 preflight retried after an earlier error')
            _capacity(record['capacity'])
        else:
            require(record['error_type'] == 'RuntimeError' and record['capacity'] is None,
                    'W1 final preflight record does not preserve its native capacity failure')
    return {'campaign': str(parent), 'attempted_preflight_groups': len(records),
        'successful_preflight_groups': len(records)-1, 'generated_requests': 0,
        'original_backend_restored': True, 'local_seconds_recorded': prior.get('local_seconds')}


def _lossless_writer_requests(pack, bound, writer_requests):
    require(len(writer_requests) == 33, 'W2 lossless writer coverage changed')
    for number, request in enumerate(writer_requests, 1):
        owner_ids = [number*2-1, number*2]
        original = previous_requests.build_writer_request(pack, bound, owner_ids)
        expected = requests.build_writer_request(pack, bound, owner_ids)
        decoded = requests.decode_body(request['body'])
        require(_same(request, expected) and _same(decoded, original['body'])
            and request['schema'] == original['schema']
            and request['instruction'].startswith(original['instruction'])
            and request['body']['focus_owner_ids'] == owner_ids,
            'Compact writer does not losslessly preserve the entire original W1 request')
    return len(writer_requests)


def _global(out, source, before, after, transactions, pack, bound, prefix, native):
    observations = {item['observation_id']: item for item in pack['observations']}
    checks = {}; groups = sparse.chunk_ids(source)
    require(len(groups) == 11 and [owner for ids in groups for owner in ids] == list(range(1, 67)),
            'Global owner coverage changed')
    for number, ids in enumerate(groups, 1):
        request = requests.build_global_request(pack, bound, source, before, after, transactions, ids)
        require(_same(request, previous_requests.build_global_request(pack, bound, source, before, after, transactions, ids)),
                'W2 global verification differs from W1')
        value = native(out/'requests'/f'{prefix}-{number}.json', request, 'qwen')
        for owner, row in value.items():
            for finding in row['regressions']:
                witness = observations[finding['observation_id']]
                finding['quotes_valid'] = bool(finding['transaction_ids'] and finding['before_quote']
                    and finding['after_quote'] and finding['source_quote']
                    and witness['owner_id'] == int(owner)
                    and finding['before_quote'] in before[int(owner)-1].text
                    and finding['after_quote'] in after[int(owner)-1].text
                    and finding['source_quote'] in witness['text'])
        checks.update(value)
    return checks



_CAPACITY_FIELDS = {'rendered_prompt_sha256', 'context', 'prompt_tokens', 'reserved_output', 'fits'}


def _timestamp(value, label):
    require(isinstance(value, str), label + ' timestamp is missing')
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(label + ' timestamp is invalid') from exc
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
            label + ' timestamp must have a timezone')
    return parsed


def _capacity(value):
    require(isinstance(value, dict) and set(value) == _CAPACITY_FIELDS,
            'Writer preflight capacity fields changed')
    require(all(type(value.get(key)) is int for key in ('context', 'prompt_tokens', 'reserved_output'))
        and value['context'] == 32768 and value['reserved_output'] == 6144
        and value['prompt_tokens'] > 0 and value['fits'] is True
        and value['prompt_tokens'] + value['reserved_output'] + 64 <= value['context']
        and isinstance(value['rendered_prompt_sha256'], str)
        and re.fullmatch(r'[0-9a-f]{64}', value['rendered_prompt_sha256']) is not None,
        'Writer preflight capacity arithmetic or rendered prompt hash changed')
    return value


def _writer_preflight(out, writer_requests, track):
    value = track(out/'writer-capacity-preflight.json')
    keys = {'version', 'round', 'family', 'status', 'expected_requests', 'requests_sha256',
        'coverage_owner_ids', 'started_utc', 'finished_utc', 'elapsed_seconds', 'records',
        'error_type', 'preflight_sha256'}
    require(isinstance(value, dict) and set(value) == keys
        and value['version'] == controller.WRITER_PREFLIGHT_VERSION
        and value['round'] == 'W2' and value['family'] == 'gemma' and value['status'] == 'complete'
        and value['error_type'] is None and value['expected_requests'] == 33
        and value['requests_sha256'] == fingerprint(writer_requests)
        and value['coverage_owner_ids'] == list(range(1, 67))
        and value['preflight_sha256'] == fingerprint({key: item for key, item in value.items()
            if key != 'preflight_sha256'}), 'Complete all-group writer capacity preflight is missing or changed')
    require(type(value['elapsed_seconds']) in (int, float) and math.isfinite(value['elapsed_seconds'])
        and value['elapsed_seconds'] >= 0, 'Writer preflight elapsed time is invalid')
    started = _timestamp(value['started_utc'], 'Preflight start')
    finished = _timestamp(value['finished_utc'], 'Preflight finish')
    require(started <= finished, 'Writer preflight timestamps are reversed')
    records = value['records']
    require(isinstance(records, list) and len(records) == len(writer_requests) == 33,
            'Writer preflight must include all 33 groups exactly once')
    record_keys = {'request_key', 'owner_ids', 'request_sha256', 'instruction_sha256', 'body_sha256',
        'schema_sha256', 'native_request_sha256', 'started_utc', 'finished_utc', 'elapsed_seconds',
        'capacity', 'error_type'}
    previous_finish = started; by_key = {}
    for number, (record, request) in enumerate(zip(records, writer_requests), 1):
        key = f'writer-group-{number}'
        require(isinstance(record, dict) and set(record) == record_keys
            and record['request_key'] == key and record['owner_ids'] == request['body']['focus_owner_ids']
            and record['request_sha256'] == fingerprint(request)
            and record['native_request_sha256'] == fingerprint(controller.writer_native_request(request))
            and all(record[name+'_sha256'] == fingerprint(request[name]) for name in ('instruction', 'body', 'schema'))
            and record['error_type'] is None, 'Writer preflight request binding or group coverage changed')
        start = _timestamp(record['started_utc'], key + ' preflight start')
        end = _timestamp(record['finished_utc'], key + ' preflight finish')
        require(previous_finish <= start <= end <= finished,
                'Writer preflight group timestamps are out of order')
        require(type(record['elapsed_seconds']) in (int, float) and math.isfinite(record['elapsed_seconds'])
            and record['elapsed_seconds'] >= 0, 'Writer group preflight elapsed time is invalid')
        _capacity(record['capacity']); by_key[key] = record; previous_finish = end
    return by_key, finished


def _match_writer_preflight(path, record, preflight_finished, track):
    saved = track(path)
    require(saved.get('request_sha256') == record['native_request_sha256'],
            'Native writer differs from its frozen all-group preflight request')
    for attempt in saved['attempts']:
        started = _timestamp(attempt.get('started_utc'), Path(path).stem + ' generation start')
        require(preflight_finished <= started, 'Writer generation began before all 33 groups passed preflight')
        if attempt.get('capacity') is not None:
            require(_same(_capacity(attempt['capacity']), record['capacity']),
                    'Native writer capacity differs from its all-group preflight')
    require(_same(_capacity(saved['attempts'][-1]['capacity']), record['capacity']),
            'Native writer lacks its matching successful preflight capacity')


def _candidate(folder, conf, source, target, pack, bound, track):
    out = folder/controller.ROUND
    candidate = track(out/'candidate.json')
    require(candidate.get('version') == controller.VERSION and candidate.get('round') == controller.ROUND
        and candidate.get('writer') == 'gemma' and candidate.get('checker') == 'qwen'
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
        return temporal_gate._native(path, request['instruction'], request['body'], request['schema'], family, track)
    writer_requests = controller.writer_requests(pack, bound)
    require(len(writer_requests) == 33
        and [owner for request in writer_requests for owner in request['body']['focus_owner_ids']]
            == list(range(1, 67)), 'Writer must cover all owners once in 33 pairs')
    lossless_groups = _lossless_writer_requests(pack, bound, writer_requests)
    preflight, preflight_finished = _writer_preflight(out, writer_requests, track)
    proposed = []
    for number, request in enumerate(writer_requests, 1):
        path = out/'requests'/f'writer-group-{number}.json'
        value = native(path, request, 'gemma')
        _match_writer_preflight(path, preflight[f'writer-group-{number}'], preflight_finished, track)
        for owner in request['body']['focus_owner_ids']:
            proposed.append(whole.prepare_owner(owner, value['owners'][str(owner)]['chinese'], source, target, pack))
    require(_same(track(out/'proposed-transactions.json'), proposed),
            'Whole-owner proposals differ from exact native generation')
    accepted = []; decisions = []
    for transaction in proposed:
        if not transaction['valid'] or transaction['no_op']:
            continue
        request = requests.build_verifier_request(pack, bound, source, target, transaction)
        require(_same(request, previous_requests.build_verifier_request(pack, bound, source, target, transaction)),
                'W2 owner verification differs from W1')
        verdict = native(out/'requests'/('verify-'+transaction['transaction_id']+'.json'), request, 'qwen')
        keep = whole.accept_owner_decision(verdict, transaction, pack)
        decisions.append({'transaction_id': transaction['transaction_id'], 'accepted': keep, 'verdict': verdict})
        if keep:
            accepted.append(transaction)
    require(_same(track(out/'local-decisions.json'), decisions), 'Whole-owner acceptance differs from native checks')
    require(bool(accepted), 'No accepted non-noop compact-owner candidate')
    first_count = len(accepted)
    final_source, final_target, ledger = whole.apply_owner_transactions(source, target, accepted, pack)
    checks = _global(out, source, target, final_target, accepted, pack, bound, 'global', native)
    rolled = sorted({identifier for row in checks.values() for finding in row['regressions']
        if finding['quotes_valid'] for identifier in finding['transaction_ids']})
    if rolled:
        accepted = [transaction for transaction in accepted if transaction['transaction_id'] not in rolled]
        require(bool(accepted), 'All compact-owner changes rolled back; no distinct candidate')
        final_source, final_target, ledger = whole.apply_owner_transactions(source, target, accepted, pack)
        checks = _global(out, source, target, final_target, accepted, pack, bound, 'assembled', native)
    require(set((out/'requests').glob('*.json')) == dispatched,
            'Unaccounted native request records or duplicate generation')
    require(final_source == source
        and [(r.index, r.ts_line) for r in final_target] == [(r.index, r.ts_line) for r in target],
        'Whole-owner source or geometry changed')
    require(_same(candidate['accepted_transactions'], accepted)
        and _same(candidate['transaction_ledger'], ledger)
        and candidate['rolled_back_transactions'] == rolled
        and _same(candidate['global_checks'], checks), 'Whole-owner assembly or rollback ledger changed')
    require(set(checks) == set(workflow.rowmap(source)) and candidate['coverage'] == 66
        and candidate['local_gate_passed'] is True
        and not any(finding['quotes_valid'] for row in checks.values() for finding in row['regressions']),
        'Complete native compact-owner local gate did not pass')
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
    return final_target, {'owners_considered': 66, 'writer_groups': 33, 'writer_preflight_groups': len(preflight), 'lossless_writer_groups': lossless_groups,
        'owner_checks': len(decisions), 'accepted_before_rollback': first_count,
        'accepted_transactions': len(accepted), 'rolled_back_transactions': len(rolled),
        'owner_coverage': 66, 'changed_target_owners': len(changed), 'display': display}


def _below_target_score(folder, conf, source, target, pools, baseline, track):
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
            and score.get('target_sha256') == target_hash, 'Scalar does not bind compact-owner target')
        require(not score.get('inherited_unchanged_baseline'), 'No distinct candidate exists')
        if score.get('inherited_duplicate_target'):
            parent = str(Path(score['inherited_duplicate_target']).resolve(strict=True))
            require(parent in allowed and score.get('confirmed') is False
                and not score.get('primary') and not score.get('confirmation'), 'Invalid duplicate score inheritance')
            inherited_score = replay_score(Path(parent), candidate=False)
            require(score.get('score') == inherited_score, 'Duplicate target changed its native scalar')
            return inherited_score
        if candidate:
            require(score.get('local_gate_passed') is True, 'Fresh scalar local gate binding changed')
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
    """Replay W2 from immutable native records, with no inference or promotion."""
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
                'W2 has not reached a confirmed four')
    pins = {}
    def track(path, raw=False):
        path = Path(path); require(not path.is_symlink(), 'Symlink cannot serve as immutable artifact')
        path = path.resolve(strict=True); data = path.read_bytes(); digest = hashlib.sha256(data).hexdigest()
        require(str(path) not in pins or pins[str(path)] == digest, 'Artifact changed during compact-owner replay')
        pins[str(path)] = digest
        return data if raw else strict_json(data)
    require(_same(track(folder/'campaign.json'), conf), 'Whole-owner campaign changed during validation')
    old._producer_snapshots(folder, conf, track)
    required = {str(controller.ROOT/'src'/f'{name}.py') for name in
        ('compact_owner_revisit', 'compact_owner_release', 'compact_owner_requests',
            'whole_owner_revisit', 'whole_owner_release', 'whole_owner', 'whole_owner_requests')}
    require(required <= set(conf['code_pins']), 'Missing compact-owner producer snapshots')
    for path, digest in {**conf['input_hashes'], **conf['code_pins']}.items():
        track(path, raw=True); require(file_hash(Path(path)) == digest, 'Pinned compact-owner input or producer changed')
    require(conf['writer_recipe_sha256'] == file_hash(Path(conf['writer_recipe']))
        and conf['plan_sha256'] == file_hash(Path(conf['plan'])), 'Whole-owner recipe or plan changed')
    source, target = controller._source_pair(folder)
    require(len(source) == len(target) == 66
        and [(row.index, row.ts_line) for row in source] == [(row.index, row.ts_line) for row in target],
        'Original compact-owner geometry changed')
    pools = [review._validate_pool(Path(path)) for path in conf['evidence_paths']]
    require(bool(pools) and all(pool['media_sha256'] == conf['media_sha256'] for pool in pools),
            'Whole-owner evidence belongs to another recording')
    require(file_hash(Path(conf['media'])) == conf['media_sha256'], 'Recording identity changed')
    pins[str(Path(conf['media']).resolve())] = conf['media_sha256']
    pack, bound, baseline, fallback_ids = _source_ancestry(folder, conf, source, target, pools, track)
    capacity_failure = _capacity_failure_ancestry(folder, conf, source, target, pools, pack, bound, track)
    final_target, local = _candidate(folder, conf, source, target, pack, bound, track)
    if require_four:
        primary, confirmation = replay._scores(folder, conf, controller.ROUND, source, final_target, pools, track)
        score = min(primary['score'], confirmation['score'])
    else:
        score = _below_target_score(folder, conf, source, final_target, pools, baseline, track)
    require(all(file_hash(Path(path)) == digest for path, digest in pins.items()), 'Artifact changed during final replay')
    return {'version': controller.VERSION, 'round': controller.ROUND,
        'status': 'contextual_ready' if require_four else 'replayed_below_target', 'score': score,
        'local': local, 'capacity_failure_prerequisite': capacity_failure, 'fallback_owner_ids': fallback_ids, 'pack_sha256': pack['pack_sha256'],
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
