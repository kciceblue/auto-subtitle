"""Read-only release validation for sparse-local-revisit-1 campaigns.

This verifies a contextual score milestone and offers an explicit, small reference
manifest. It never dispatches models, promotes output, copies media, or certifies
source fidelity/playback. Local decisions are replayed from native saved answers;
producer snapshots are checked without executing snapshot code.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from src import contextual_evidence_review as er
from src import contextual_evidence_release as display_checks
from src import sparse_edits as edits
from src import sparse_revisit as sr
from src import revisit_workflow as rw
from src.contextual_review_contract import require, strict_json
from src.workflow_state import file_hash, fingerprint

VERSION = 'sparse-contextual-ready-reference-1'
ROOT = Path(__file__).resolve().parents[1]
CORE_PRODUCERS = ('src/sparse_revisit.py', 'src/sparse_edits.py', 'src/revisit_workflow.py',
                  'src/contextual_evidence_review.py', 'src/late_audio.py')
REPLAY_NAMES = {'VERSION', 'FOCUS', 'RULES', 'SOURCE', 'DIAGNOSE', 'PROPOSE', 'VERIFY', 'GLOBAL',
                'chunk_ids', 'episode_body', 'choose_cases', 'decision_schema', 'accept_decision',
                'check_global', 'ask'}


def _read(path: Path):
    return strict_json(path.read_bytes())


def _rows(path: Path):
    values = er._rows(_read(path))
    require(len(values) == 66, 'Expected exactly 66 owners')
    return [rw.SrtBlock(**row) for row in values]


def _syntax(path: Path, names: set[str]) -> dict:
    result = {}
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        selected = getattr(node, 'name', None)
        if isinstance(node, ast.Assign):
            selected = next((n.id for n in node.targets if isinstance(n, ast.Name) and n.id in names), None)
        if selected in names:
            result[selected] = ast.dump(node, include_attributes=False)
    require(set(result) == names, 'Producer replay definitions missing')
    return result


def _producer_snapshots(folder, conf, track):
    """Retain all code epochs, allowing only historical request-cache plumbing drift."""
    epochs = conf.get('code_epochs', [])
    require(isinstance(epochs, list), 'Producer code epochs must be a list')
    seen = set(); count = 0
    for epoch in epochs:
        require(isinstance(epoch, dict) and set(epoch) == {'name', 'code_pins', 'producer_snapshot_manifest',
                'producer_snapshot_manifest_sha256', 'reason', 'ended_utc'}, 'Invalid historical producer epoch')
        require(all(isinstance(epoch[k], str) and bool(epoch[k]) for k in ('name', 'reason', 'ended_utc')),
                'Historical producer epoch metadata missing')
        path = Path(epoch['producer_snapshot_manifest']).resolve()
        require(path not in seen, 'Duplicate historical producer epoch')
        count += _producer_epoch(folder, epoch, track, historical=True)
        seen.add(path)
    require(Path(conf.get('producer_snapshot_manifest', '')).resolve() not in seen,
            'Current producer epoch also appears as historical')
    return count + _producer_epoch(folder, conf, track, historical=False)


def _producer_epoch(folder, conf, track, *, historical):
    manifest_path = Path(conf.get('producer_snapshot_manifest', '')).resolve()
    require(manifest_path.is_relative_to(folder), 'Missing retained producer snapshot manifest')
    manifest = track(manifest_path)
    if historical:
        require(file_hash(manifest_path) == conf.get('producer_snapshot_manifest_sha256'),
                'Historical producer manifest changed')
    pins = conf.get('code_pins')
    require(isinstance(pins, dict) and set(pins) == set(manifest), 'Producer pin/snapshot set mismatch')
    require({str(ROOT / name) for name in CORE_PRODUCERS} <= set(pins), 'Missing core producer snapshot')
    for original, spec in manifest.items():
        require(isinstance(spec, dict) and set(spec) == {'sha256', 'snapshot'}
                and spec['sha256'] == pins[original], 'Producer manifest pin mismatch')
        snapshot = Path(spec['snapshot']).resolve()
        require(snapshot.is_relative_to(manifest_path.parent), 'Snapshot escapes retained directory')
        track(snapshot, raw=True)
        require(file_hash(snapshot) == spec['sha256'], 'Producer snapshot changed')
    runtime = Path(manifest[str(ROOT / 'src/sparse_revisit.py')]['snapshot'])
    names = REPLAY_NAMES - {'ask'} if historical else REPLAY_NAMES
    require(_syntax(runtime, names) == _syntax(Path(sr.__file__), names),
            'Local replay semantics differ from measured producer')
    require(file_hash(Path(edits.__file__)) == pins[str(ROOT / 'src/sparse_edits.py')],
            'Sparse patch replay helper differs from measured producer')
    return len(pins)


def _native(path, instruction, body, schema, family, track):
    """Replay a native receipt or one-hop alias at its original telemetry stage."""
    path = Path(path)
    value = track(path)
    request = value.get('request')
    require(isinstance(request, dict) and value.get('request_sha256') == fingerprint(request), 'Native request binding changed')
    thinking = family == 'qwen'
    maximum = 12288 if thinking else 6144
    extra = {'model': rw.QWEN if family == 'qwen' else 'gemma4-31b-qat-q4',
             'temperature': .3 if family == 'qwen' else .2, 'seed': 20260915, 'top_p': .95,
             'max_tokens': maximum, 'chat_template_kwargs': {'enable_thinking': thinking},
             'reasoning_budget_tokens': 6144 if thinking else 0,
             'reasoning_effort': 'high' if thinking else 'none',
             'response_format': {'type': 'json_object', 'schema': schema}}
    if family == 'gemma':
        extra.update(top_k=64, repeat_penalty=1)
    expected = {'version': sr.VERSION, 'family': family, 'instruction': instruction,
                'body': body, 'schema': schema, 'extra': extra, 'thinking': thinking}
    require(request == expected, 'Native local request differs from expected candidate inputs')
    if value.get('status') == 'reused':
        require(set(value) == {'status', 'request', 'request_sha256', 'attempts', 'inherited_from', 'inherited_sha256'}
                and value['attempts'] == [], 'Reused request contains execution data or invalid fields')
        require(path.parent.name == 'requests' and path.parent.parent.name in ('S1', 'S2', 'S3'),
                'Alias is outside a campaign request directory')
        inherited = value['inherited_from']
        require(isinstance(inherited, str) and Path(inherited).is_absolute(), 'Native alias path must be absolute')
        original = Path(inherited)
        campaign = path.resolve().parents[2]
        resolved = original.resolve(strict=True)
        require(resolved != path.resolve() and resolved.parent.name == 'requests'
                and resolved.parent.parent.name in ('S1', 'S2', 'S3') and resolved.parents[2] == campaign
                and resolved.suffix == '.json', 'Native alias leaves its campaign or points to itself')
        original_value = track(original)
        require(file_hash(original) == value['inherited_sha256'], 'Inherited native receipt changed')
        require(original_value.get('status') == 'complete', 'Alias must point directly to a complete native request')
        require(original_value.get('request') == expected
                and original_value.get('request_sha256') == fingerprint(expected), 'Inherited request differs from alias inputs')
        path, value = original, original_value
    attempts = value.get('attempts')
    require(value.get('status') == 'complete' and isinstance(attempts, list) and 1 <= len(attempts) <= 2,
            'Local request lacks bounded completed execution')
    attempt = attempts[-1]; raw = attempt.get('raw')
    require(isinstance(raw, str) and attempt.get('answer_sha256') == fingerprint(raw), 'Native answer hash missing or changed')
    parsed = strict_json(raw)
    require(parsed == value.get('parsed'), 'Parsed local answer differs from native raw JSON')
    import jsonschema
    jsonschema.validate(parsed, schema)
    capacity = attempt.get('capacity', {})
    require(capacity.get('fits') is True, 'Native capacity check did not pass')
    receipts = attempt.get('native_receipts')
    require(isinstance(receipts, list) and bool(receipts), 'Missing native local metrics')
    last = receipts[-1]
    require(last.get('stage') == path.stem and last.get('finish_reason') == 'stop'
            and type(last.get('answer_chars')) is int and last['answer_chars'] == len(raw), 'Local response did not stop normally')
    metrics = path.parent / 'metrics.jsonl'
    raw_metrics = track(metrics, raw=True).decode('utf-8', errors='strict')
    matching = [strict_json(line) for line in raw_metrics.splitlines() if line.strip()]
    matching = [row for row in matching if row.get('stage') == path.stem]
    require(len(matching) >= len(receipts) and matching[-len(receipts):] == receipts,
            'Saved local metrics differ from native telemetry')
    return deepcopy(parsed)


def _global(directory, base_source, base_target, source, target, transactions, obs, context, checker, prefix, track):
    result = {}; byid = {row['observation_id']: row for row in obs}
    transaction_ids = [tx['case_id'] for tx in transactions]
    for n, ids in enumerate(sr.chunk_ids(source), 1):
        relevant = [o for o in obs if o['owner_id'] in ids]
        regression = rw.obj({'transaction_ids': rw.arr({'type': 'string', 'enum': transaction_ids}, 6),
            'before_quote': rw.string(1000), 'after_quote': rw.string(1000),
            'observation_id': {'type': 'string', 'enum': [o['observation_id'] for o in relevant]},
            'source_quote': rw.string(1200), 'reason': rw.string(1200)})
        schema = rw.keyed(ids, rw.obj({'regressions': rw.arr(regression, 6), 'legacy_or_source_uncertainty': rw.string(1200)}))
        body = {**sr.episode_body(base_source, base_target, context),
                'candidate_japanese': rw.rowmap(source), 'candidate_chinese': rw.rowmap(target),
                'focus_ids': ids, 'transactions': [{k: tx[k] for k in
                    ('case_id', 'owner_id', 'old_chinese', 'new_chinese', 'old_japanese', 'new_japanese')} for tx in transactions],
                'observations': relevant}
        value = _native(directory / 'requests' / f'{prefix}-{n}.json', sr.GLOBAL, body, schema, checker, track)
        for owner, row in value.items():
            for finding in row['regressions']:
                finding['quotes_valid'] = bool(finding['transaction_ids'] and finding['before_quote']
                    and finding['after_quote'] and finding['source_quote']
                    and finding['before_quote'] in base_target[int(owner)-1].text
                    and finding['after_quote'] in target[int(owner)-1].text
                    and byid[finding['observation_id']]['owner_id'] == int(owner)
                    and finding['source_quote'] in byid[finding['observation_id']]['text'])
        result.update(value)
    return result


def _local(folder, directory, round_id, candidate, source, target, obs, track):
    """Rebuild exact proposals, native decisions and rollback/final global checks."""
    paths = candidate['evidence_paths']
    analysis_path = directory / 'analysis-expanded.json'
    if not analysis_path.exists() or _read(analysis_path).get('evidence_paths') != paths:
        analysis_path = directory / 'analysis.json'
    analysis = track(analysis_path)
    require(analysis.get('version') == sr.VERSION and analysis.get('source_hash') == file_hash(folder/'source.json')
            and analysis.get('target_hash') == file_hash(folder/'target.json')
            and analysis.get('evidence_paths') == paths and analysis.get('external_feedback_used') is False,
            'Analysis inputs changed')
    analysis_origin = round_id
    inheritance_keys = {'inherited_analysis', 'inherited_analysis_sha256'}
    if inheritance_keys & set(analysis):
        require(round_id == 'S2' and inheritance_keys <= set(analysis), 'Invalid inherited analysis declaration')
        inherited_path = Path(analysis['inherited_analysis']).resolve(strict=True)
        require(inherited_path == folder/'S1'/'analysis.json', 'Inherited analysis is not this campaign S1 analysis')
        inherited = track(inherited_path)
        require(file_hash(inherited_path) == analysis['inherited_analysis_sha256'], 'Inherited analysis hash changed')
        require({k: v for k, v in analysis.items() if k not in inheritance_keys} == inherited,
                'Inherited analysis differs from its frozen original')
        analysis_origin = 'S1'
    require(analysis.get('coverage') == list(range(1, 67)), 'Analysis omitted or duplicated owners')
    cases = sr.choose_cases(analysis, 12 if round_id == 'S1' else 24)
    require(track(directory/'selected-cases.json') == cases, 'Selected cases differ from fixed local ordering')
    proposed = track(directory/'proposed-transactions.json')
    require(isinstance(proposed, list) and len(proposed) == len(cases), 'Proposal coverage mismatch')
    context = (folder/'original-context.txt').read_text(encoding='utf-8')
    writer, checker = ('qwen', 'gemma') if round_id == 'S3' else ('gemma', 'qwen')
    require(candidate.get('writer') == writer and candidate.get('checker') == checker, 'Writer/checker identity mismatch')
    decisions = []; accepted = []; native_cases = 0
    proposal_schema = rw.obj({k: rw.string(1800) for k in
        ('old_chinese', 'new_chinese', 'old_japanese', 'new_japanese', 'source_observation_id', 'source_support_quote')})
    for case, transaction in zip(cases, proposed):
        edits.validate_case(case, source, target, obs)
        require(re.fullmatch(rf'{analysis_origin}-o{case["owner_id"]:03d}-c[12]', case['case_id']) is not None, 'Invalid generated case identifier')
        own = [o for o in obs if o['owner_id'] == case['owner_id']]
        require(len(case['readings']) == len(own) and {r['observation_id'] for r in case['readings']}
                == {o['observation_id'] for o in own}, 'Case dropped a source observation')
        proposal = _native(directory/'requests'/('propose-'+case['case_id']+'.json'), sr.PROPOSE,
            {**sr.episode_body(source, target, context), 'case': case, 'allow_source': round_id != 'S1', 'observations': own},
            proposal_schema, writer, track)
        rebuilt = edits.prepare_patch(case, proposal, source, target, obs, allow_source=round_id != 'S1')
        require(transaction == rebuilt, 'Saved proposal transaction does not replay exactly')
        if not transaction['valid'] or transaction['no_op']:
            continue
        verdict = _native(directory/'requests'/('verify-'+case['case_id']+'.json'), sr.VERIFY,
            {'episode_japanese': rw.rowmap(source), 'episode_chinese': rw.rowmap(target), 'original_context': context,
             'case': case, 'before_japanese': transaction['before_source_text'], 'before_chinese': transaction['before_target_text'],
             'after_japanese': transaction['after_source_text'], 'after_chinese': transaction['after_target_text'],
             'source_readings': case['readings'], 'observations': own},
            sr.decision_schema([o['observation_id'] for o in own]), checker, track)
        keep = sr.accept_decision(verdict, case, transaction, obs); native_cases += 1
        decisions.append({'case_id': case['case_id'], 'accepted': keep, 'verdict': verdict})
        if keep:
            accepted.append(transaction)
    require(track(directory/'local-decisions.json') == decisions, 'Local acceptance differs from native answer replay')
    require(candidate.get('cases_considered') == len(cases)
            and candidate.get('valid_proposals') == sum(t['valid'] and not t['no_op'] for t in proposed), 'Candidate counts changed')
    require(bool(accepted), 'No accepted non-noop transactions')
    before_rollback = list(accepted)
    candidate_source, candidate_target, ledger = edits.apply_transactions(source, target, accepted)
    checks = _global(directory, source, target, candidate_source, candidate_target, accepted, obs, context, checker, 'global', track)
    rolled = sorted({cid for row in checks.values() for f in row['regressions'] if f['quotes_valid'] for cid in f['transaction_ids']})
    if rolled:
        accepted = [t for t in accepted if t['case_id'] not in rolled]
        require(bool(accepted), 'Every transaction rolled back; unchanged baseline cannot qualify')
        candidate_source, candidate_target, ledger = edits.apply_transactions(source, target, accepted)
        checks = _global(directory, source, target, candidate_source, candidate_target, accepted, obs, context, checker, 'assembled', track)
    require(candidate.get('accepted_transactions') == accepted and candidate.get('rolled_back_transactions') == rolled
            and candidate.get('transaction_ledger') == ledger, 'Candidate transaction/rollback ledger changed')
    require(set(checks) == set(rw.rowmap(source)) and candidate.get('coverage') == 66, 'Final local check coverage incomplete')
    require(candidate.get('global_checks') == checks, 'Final check differs from native local answers')
    require(not any(f['quotes_valid'] for row in checks.values() for f in row['regressions'])
            and candidate.get('local_gate_passed') is True, 'Supported local regression remains')
    return candidate_source, candidate_target, {'native_case_decisions': native_cases,
        'transactions_before_rollback': len(before_rollback), 'accepted_transactions': len(accepted),
        'rolled_back_transactions': len(rolled), 'local_owner_coverage': len(checks),
        'unsupported_global_flags': sum(not f['quotes_valid'] for row in checks.values() for f in row['regressions'])}


def validate_campaign(folder: str | Path) -> dict:
    folder = Path(folder).expanduser().resolve(strict=True)
    artifacts = {}
    def track(path, *, raw=False):
        path = Path(path)
        require(not path.is_symlink(), 'Artifact symlink is not an immutable file')
        path = path.resolve(strict=True); data = path.read_bytes()
        artifacts[str(path)] = file_hash(path)
        return data if raw else strict_json(data)
    conf = track(folder/'campaign.json')
    require(conf.get('version') == sr.VERSION and conf.get('status') == 'confirmed_four', 'Campaign has no confirmed candidate')
    require(conf.get('original_backend_restored') is True, 'Original local backend restoration is unconfirmed')
    require(conf.get('external_feedback') == 'score_only' and all(conf.get(k) is False for k in
        ('raw_source_fidelity_verified', 'human_reference_used', 'visual_input_used', 'selected_output_modified')),
        'Campaign scope or authority claims changed')
    round_id = conf.get('eligible_candidate')
    require(round_id in ('S1', 'S2', 'S3'), 'Unknown eligible sparse round')
    directory = folder/round_id
    score_path = Path(conf.get('eligible_score_receipt', directory/'score.json')).resolve()
    require(score_path.parent == directory and score_path.name in ('score.json', 'score-expanded.json'), 'Invalid eligible score path')
    score = track(score_path); candidate = track(directory/'candidate.json')
    require(score.get('status') == 'complete' and score.get('confirmed') is True
            and score.get('local_gate_passed') is True and not score.get('inherited_unchanged_baseline')
            and not score.get('inherited_duplicate_target'), 'Candidate score is incomplete, inherited, or unconfirmed')
    require(candidate.get('version') == sr.VERSION and candidate.get('round') == round_id
            and all(candidate.get(k) is False for k in ('raw_source_fidelity_verified', 'external_feedback_used', 'human_reference_used')),
            'Candidate scope changed')
    for name, key in [('source.json', 'source_sha256'), ('target.json', 'target_sha256'), ('original-context.txt', 'context_sha256')]:
        track(folder/name, raw=True)
        require(file_hash(folder/name) == conf.get(key), 'Immutable baseline/context changed')
    for name, key in [('source.json', 'source_sha256'), ('target.json', 'target_sha256')]:
        track(directory/name, raw=True)
        require(file_hash(directory/name) == candidate.get(key), 'Candidate owner artifact changed')
    for path_key, hash_key in [('media', 'media_sha256'), ('writer_recipe', 'writer_recipe_sha256')]:
        path = Path(conf[path_key]).resolve(strict=True)
        digest = file_hash(path)
        require(digest == conf[hash_key], 'Media or writer recipe changed')
        artifacts[str(path)] = digest
    source, target = _rows(folder/'source.json'), _rows(folder/'target.json')
    final_source, final_target = _rows(directory/'source.json'), _rows(directory/'target.json')
    geometry = lambda rows: [(r.index, r.ts_line) for r in rows]
    require(geometry(source) == geometry(target) == geometry(final_source) == geometry(final_target), 'Owner geometry changed')
    changed_target = [a.index for a, b in zip(target, final_target) if a.text != b.text]
    changed = [a.index for a, b, c, d in zip(source, final_source, target, final_target) if a.text != b.text or c.text != d.text]
    require(bool(changed_target) and candidate.get('changed_target_owners') == changed_target
            and candidate.get('changed_owners') == changed, 'Changed-owner ledger is inaccurate or target unchanged')
    producer_count = _producer_snapshots(folder, conf, track)
    primary = er.validate_receipt(score['primary']); confirmation = er.validate_receipt(score['confirmation'])
    require(primary['purpose'] == 'candidate' and confirmation['purpose'] == 'confirmation'
            and primary['inputs'] == confirmation['inputs'] and primary['pool_version'] == confirmation['pool_version'] == score['pool_version'],
            'Primary and confirmation do not use identical candidate inputs/pool')
    require(all(type(r['score']) is int and 4 <= r['score'] <= 5 for r in (primary, confirmation))
            and score.get('score') == primary['score'] and score.get('confirmation_score') == confirmation['score'], 'Both native scalar scores must reach four')
    require(confirmation['dependency'] == {k: primary[k] for k in ('purpose', 'output_dir', 'dispatch_id', 'finished_utc', 'receipt_hashes')}
            and primary['dispatch_id'] != confirmation['dispatch_id']
            and datetime.fromisoformat(primary['finished_utc']) <= datetime.fromisoformat(confirmation['started_utc']), 'Confirmation is not a fresh sequential dispatch')
    for role, rows in [('source', final_source), ('target', final_target), ('raw', source)]:
        require(primary['inputs'][role]['sha256'] == er.owner_rows_sha256(rows), 'Reviewed owner wording differs')
    require(score.get('source_sha256') == er.owner_rows_sha256(final_source)
            and score.get('target_sha256') == er.owner_rows_sha256(final_target), 'Score owner hash changed')
    require(primary['inputs']['context']['sha256'] == conf['context_sha256'], 'Reviewed original context differs')
    preparation = track(Path(primary['output_dir'])/'preparation.json')
    bundle = er.read_bundle(preparation['manifest_path']); track(Path(preparation['manifest_path']))
    require(bundle['pool_version'] == primary['pool_version'], 'Reviewed bundle pool differs')
    reviewed_pools = strict_json(bundle['raw_inputs']['evidence'])
    require(all(p['media_sha256'] == conf['media_sha256'] for p in reviewed_pools), 'Evidence belongs to different media')
    local_pools = [track(Path(path)) for path in candidate['evidence_paths']]
    require(bool(local_pools) and all(any(p == reviewed for reviewed in reviewed_pools) for p in local_pools), 'Local evidence is not retained in reviewed pool')
    obs = rw.observations(local_pools, source)
    rebuilt_source, rebuilt_target, local = _local(folder, directory, round_id, candidate, source, target, obs, track)
    require(rebuilt_source == final_source and rebuilt_target == final_target, 'Sparse transactions do not reproduce candidate')
    baseline = er.validate_receipt(primary['dependency']['output_dir'])
    require(baseline['purpose'] == 'baseline' and baseline['pool_version'] == primary['pool_version']
            and baseline['inputs']['source']['sha256'] == er.owner_rows_sha256(source)
            and baseline['inputs']['target']['sha256'] == er.owner_rows_sha256(target), 'Same-pool baseline differs from immutable round93 pair')
    for name in ('display.json', 'source.srt', 'subtitles.zh.srt'):
        track(directory/'display'/name, raw=True)
    display = display_checks._display(directory/'display', final_source, final_target)
    for receipt in (baseline, primary, confirmation):
        for name, digest in receipt['receipt_hashes'].items():
            path = Path(receipt['output_dir'])/name; track(path, raw=True)
            require(file_hash(path) == digest, 'Native review receipt changed during validation')
    require(all(file_hash(Path(path)) == digest for path, digest in artifacts.items()), 'Artifact changed during validation')
    return {'version': VERSION, 'status': 'contextual_ready', 'campaign': str(folder), 'round': round_id,
        'score': min(primary['score'], confirmation['score']), 'pool_version': primary['pool_version'],
        'owner_count': 66, 'changed_target_owners': len(changed_target), 'local': local, 'display': display,
        'local_gate_scope': 'all_owners_checked_and_no_supported_new_regression',
        'producer_snapshot_count': producer_count, 'producer_code_binding': 'retained_execution_snapshots',
        'snapshot_predispatch_attested': False, 'local_evidence_pool_versions': [p['pool_version'] for p in local_pools],
        'reviewed_evidence_pool_versions': [p['pool_version'] for p in reviewed_pools],
        'reviews': {name: {k: r[k] for k in ('output_dir', 'dispatch_id', 'score', 'receipt_hashes')}
                    for name, r in [('baseline', baseline), ('primary', primary), ('confirmation', confirmation)]},
        'artifacts': artifacts, 'source_accuracy_verified': False, 'audio_truth_verified': False,
        'playback_verified': False, 'whole_bundle_read_attested': False, 'source_verified_release_authorized': False,
        'selected_output_modified': False, 'reference_only': True}


def prepare_reference(folder: str | Path, dest: str | Path) -> Path:
    """Explicitly write only a new reference manifest after complete validation."""
    dest = Path(dest).expanduser().absolute()
    require(not dest.exists(), 'Reference destination already exists')
    result = validate_campaign(folder)
    dest.mkdir(parents=True, exist_ok=False)
    path = dest/'reference-ready.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump({**result, 'created_utc': datetime.now(timezone.utc).isoformat(),
                   'retention_policy': 'Keep referenced artifacts and native evidence immutable and available'},
                  stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--prepare-reference', type=Path)
    args = parser.parse_args()
    if args.prepare_reference:
        print(prepare_reference(args.campaign, args.prepare_reference))
    else:
        value = validate_campaign(args.campaign)
        print(json.dumps({k: value[k] for k in ('status', 'round', 'score', 'owner_count', 'changed_target_owners')}))


if __name__ == '__main__':
    main()
