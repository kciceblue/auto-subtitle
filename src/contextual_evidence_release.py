"""Validate a confirmed evidence-view milestone and freeze a reference package.

This is not a source-verified release or a promotion operation. The small package
references immutable campaign artifacts; retain those paths for revalidation.
No media is copied, no selected output changes, and no reviewer is dispatched.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from src import contextual_evidence_review as review
from src import contextual_review as native
from src.contextual_review_contract import ASSESSMENT_SCOPE, BENCHMARK, require, strict_json
from src.translate import SrtBlock
from src.workflow_state import file_hash, fingerprint

VERSION = 'contextual-evidence-ready-reference-v1'
ROOT = Path(__file__).resolve().parents[1]
OWNER_COUNT = 66
LOCAL_SCOPE = 'complete_owner_coverage_and_no_detected_new_material_error'
CODE_FILES = ('scripts/run_revisit_loop.py', 'src/revisit_workflow.py', 'src/late_audio.py',
              'src/contextual_evidence_review.py', 'src/contextual_review.py', 'src/contextual_review_contract.py')


def _pin(path) -> dict:
    path = native._file(path)
    return {'path': str(path), 'sha256': file_hash(path)}


def _read(path):
    return strict_json(native._file(path).read_bytes())


def _rows(path) -> list[SrtBlock]:
    values = review._rows(_read(path))
    require(len(values) == OWNER_COUNT, 'Expected all 66 original owners')
    return [SrtBlock(**x) for x in values]


def _exact_srt(path) -> list[SrtBlock]:
    """Accept exactly the pipeline's lossless SRT encoding, without skipped chunks."""
    raw = native._file(path).read_bytes()
    text = raw.decode('utf-8', errors='strict')
    require(text.endswith('\n\n'), 'Display SRT has an incomplete final cue')
    rows = []
    for chunk in text[:-2].split('\n\n'):
        fields = chunk.split('\n', 2)
        require(len(fields) == 3 and re.fullmatch('[1-9][0-9]*', fields[0]) is not None,
                'Malformed display cue')
        rows.append(SrtBlock(int(fields[0]), fields[1], fields[2]))
    review.owner_rows_sha256(rows)  # positive ordered timing, IDs, bounded exact UTF-8
    require(''.join(f'{r.index}\n{r.ts_line}\n{r.text}\n\n' for r in rows).encode('utf-8') == raw,
            'Display serialization changed text')
    return rows


def _display(path: Path, source, target) -> dict:
    ledger = _read(path / 'display.json'); mapping = ledger.get('cues')
    ds, dt = _exact_srt(path / 'source.srt'), _exact_srt(path / 'subtitles.zh.srt')
    require(ledger.get('semantic_text_changed') is False and isinstance(mapping, list)
            and len(mapping) == len(ds) == len(dt), 'Incomplete exact-display receipt')
    require([(r.index, r.ts_line) for r in ds] == [(r.index, r.ts_line) for r in dt],
            'Paired display geometry differs')
    lineage = [x.get('utterance') for x in mapping]
    require(all(type(i) is int for i in lineage) and lineage == sorted(lineage)
            and set(lineage) == set(range(1, 67)), 'Display owner coverage changed')
    for i, (s, t, item) in enumerate(zip(ds, dt, mapping), 1):
        require(item.get('line') == i and item.get('source') == s.text and item.get('target') == t.text,
                'Rendered display differs from its owner ledger')
    for s, t in zip(source, target):
        pieces = [x for x in mapping if x['utterance'] == s.index]
        require(''.join(x['source'] for x in pieces) == s.text and ''.join(x['target'] for x in pieces) == t.text,
                'Display lost, added, or transferred owner text')
    return {'display_cues': len(dt), 'layout_warning_count': len(ledger.get('warnings', [])),
            'exact_owner_text': True, 'source_srt_sha256': file_hash(path / 'source.srt'),
            'target_srt_sha256': file_hash(path / 'subtitles.zh.srt')}


def _completed_step(state, name, path):
    step = state.get('steps', {}).get(name, {})
    require(step.get('status') == 'complete' and step.get('artifact_sha256') == file_hash(native._file(path)),
            'Completed campaign stage artifact changed')


def _frozen_verify(path: Path) -> str:
    """Read only constant string expressions; never execute archived producer code."""
    assignments = {}
    for node in ast.parse(native._file(path).read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name): assignments[target.id] = node.value
    def value(node, depth=0):
        require(depth < 10, 'Unsupported producer instruction expression')
        if isinstance(node, ast.Constant) and isinstance(node.value, str): return node.value
        if isinstance(node, ast.Name): return value(assignments[node.id], depth + 1)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return value(node.left, depth + 1) + value(node.right, depth + 1)
        raise ValueError('Producer validation instruction is not a constant string')
    return value(assignments['VERIFY'])


def _rowmap(rows):
    return {str(row.index): row.text for row in rows}


def _local_validation(candidate, directory, source, target, raw, previous_target, artifacts, verify_instruction):
    checked = candidate.get('final_validation')
    require(isinstance(checked, dict) and set(checked) == {str(i) for i in range(1, 67)}
            and candidate.get('coverage') == OWNER_COUNT and candidate.get('local_gate_passed') is True
            and candidate.get('local_gate_scope') == LOCAL_SCOPE, 'Incomplete local no-new-error gate')
    require(all(type(x.get('pass')) is bool and x.get('new_material_error') is False for x in checked.values()),
            'A new material error remains')
    require(candidate.get('legacy_or_unresolved_issue_owners') == [int(k) for k, v in checked.items() if not v['pass']],
            'Legacy issue inventory changed')
    combined = {}
    prefix = 'assembled-verify-' if candidate.get('rolled_back_owners') else 'verify-'
    groups = [[x.index for x in source[start:start+8]] for start in range(0, len(source), 8)]
    for n, group in enumerate(groups, 1):
        key = prefix + str(n); path = directory / 'requests' / (key + '.json')
        artifacts['local-validation-' + str(n)] = _pin(path)
        value = _read(path); request = value.get('request', {}); body = request.get('body', {})
        require(value.get('status') == 'complete' and value.get('request_sha256') == fingerprint(request)
                and request.get('family') == candidate['checker'] and request.get('instruction') == verify_instruction,
                'Local validation is not the frozen completed request')
        require(body.get('focus_ids') == group and body.get('candidate_japanese') == _rowmap(source)
                and body.get('candidate_chinese') == _rowmap(target)
                and body.get('original_japanese') == _rowmap(raw)
                and body.get('previous_chinese') == _rowmap(previous_target),
                'Local validation reviewed another assembled candidate')
        attempts = value.get('attempts', [])
        require(isinstance(attempts, list) and 1 <= len(attempts) <= 2, 'Local request budget invalid')
        last = attempts[-1]; receipts = last.get('native_receipts', [])
        require(isinstance(last.get('raw'), str) and receipts and receipts[-1].get('stage') == key
                and receipts[-1].get('finish_reason') == 'stop'
                and receipts[-1].get('answer_chars') == len(last['raw'])
                and strict_json(last['raw']) == value.get('parsed'), 'Local native completion/answer changed')
        require(last.get('capacity', {}).get('fits') is True and set(value['parsed']) == {str(i) for i in group},
                'Local validation coverage/capacity failed')
        combined.update(value['parsed'])
    require(combined == checked, 'Candidate local flags differ from genuine completed local outputs')


def validate_candidate(campaign: str | Path) -> dict:
    """Read-only validation; returned metadata contains no subtitle or finding text."""
    folder = native._directory(campaign)
    artifacts = {}
    def read(role, path):
        artifacts[role] = _pin(path)
        return _read(path)
    state = read('loop-state', folder / 'loop-state.json')
    conf = read('campaign', folder / 'campaign.json')
    require(state.get('version') == 'bounded-revisit-controller-1' and state.get('status') == 'confirmed_candidate'
            and state.get('original_backend_restored') is True and state.get('selected_output_modified') is False,
            'Campaign has no finished, restored confirmed candidate')
    eligible = state.get('eligible_candidate'); require(isinstance(eligible, dict), 'Missing eligible candidate')
    round_id = eligible.get('round')
    require(round_id in {'Q1', 'Q2', 'Q3'} and eligible.get('promoted') is False, 'Invalid candidate slot')
    require(conf.get('maximum_candidates') == 3 and conf.get('maximum_local_seconds') == 5400
            and conf.get('maximum_round_seconds') == 1800 and set(state.get('rounds', {})) <= {'Q1', 'Q2', 'Q3'},
            'Campaign bounds changed')
    expected_code = {str(ROOT / p) for p in CODE_FILES}
    require(set(state.get('code_pins', {})) == expected_code, 'Incomplete producing-code identity')
    explicit_snapshot = state.get('producer_snapshot_manifest')
    snapshot_path = folder / 'producer-code/manifest.json'
    if explicit_snapshot is not None:
        require(isinstance(explicit_snapshot, str), 'Invalid producer snapshot manifest path')
        snapshot_path = native._file(explicit_snapshot)
        require(snapshot_path.is_relative_to(folder), 'Producer snapshot manifest must remain inside its campaign')
    snapshots = read('producer-snapshot-manifest', snapshot_path) if snapshot_path.exists() else None
    if snapshots is not None:
        require(set(snapshots) == expected_code, 'Incomplete producer snapshots')
    workflow_code = None
    for i, (path, digest) in enumerate(state['code_pins'].items()):
        code_path = path
        if snapshots is not None:
            entry = snapshots[path]
            require(isinstance(entry, dict) and set(entry) == {'sha256', 'snapshot'}
                    and entry['sha256'] == digest, 'Snapshot differs from execution code pin')
            code_path = entry['snapshot']
            require(Path(code_path).is_relative_to(snapshot_path.parent), 'Producer snapshot outside its retained directory')
        artifacts['code-' + str(i)] = pin = _pin(code_path)
        require(pin['sha256'] == digest, 'Producing code changed')
        if path == str(ROOT / 'src/revisit_workflow.py'): workflow_code = Path(code_path)
    verify_instruction = _frozen_verify(workflow_code)
    directory = folder / round_id
    candidate = read('candidate', directory / 'candidate.json')
    _completed_step(state, round_id + ':repair', directory / 'candidate.json')
    require(candidate.get('status') == 'candidate_complete' and candidate.get('external_feedback_used') is False
            and candidate.get('human_reference_used') is False and candidate.get('source_fidelity_verified') is False,
            'Wrong candidate completion or evidence scope')
    require(candidate.get('writer') == ('qwen' if round_id == 'Q3' else 'gemma')
            and candidate.get('checker') == ('gemma' if round_id == 'Q3' else 'qwen'), 'Local writer/checker identity changed')
    paths = {'source-owners': directory / 'source.json', 'target-owners': directory / 'target.json',
             'raw-owners': folder / 'source.json', 'baseline-target-owners': folder / 'target.json',
             'original-context': folder / 'original-context.txt', 'writer-recipe': Path(conf['writer_recipe']),
             'media': Path(conf['media']), 'display-ledger': directory / 'display/display.json',
             'source-srt': directory / 'display/source.srt', 'target-srt': directory / 'display/subtitles.zh.srt'}
    for role, path in paths.items(): artifacts[role] = _pin(path)
    for role, expected in [('raw-owners', conf['source_sha256']), ('baseline-target-owners', conf['target_sha256']),
            ('original-context', conf['context_sha256']), ('writer-recipe', conf['writer_recipe_sha256']),
            ('media', conf['media_sha256']), ('source-owners', candidate['source_sha256']),
            ('target-owners', candidate['target_sha256'])]:
        require(artifacts[role]['sha256'] == expected, 'Campaign/candidate input hash changed')
    source, target, raw = (_rows(paths[role]) for role in ('source-owners', 'target-owners', 'raw-owners'))
    require([(x.index, x.ts_line) for x in source] == [(x.index, x.ts_line) for x in target]
            == [(x.index, x.ts_line) for x in raw], 'Original owner geometry changed')
    previous_path = folder / ('target.json' if round_id == 'Q1' else f'Q{int(round_id[1])-1}/target.json')
    artifacts['previous-target-owners'] = _pin(previous_path)
    _local_validation(candidate, directory, source, target, raw, _rows(previous_path), artifacts, verify_instruction)
    if round_id == 'Q3':
        previous_candidate = read('previous-candidate', folder / 'Q2/candidate.json')
        artifacts['previous-source-owners'] = _pin(folder / 'Q2/source.json')
        require(source == _rows(folder / 'Q2/source.json')
                and candidate.get('grounded_recap') == previous_candidate.get('grounded_recap'), 'Q3 changed frozen source/recap')
    _completed_step(state, round_id + ':display', paths['display-ledger'])
    structural = _display(directory / 'display', source, target)
    require(eligible.get('structural') == structural and eligible.get('local_gate_scope') == LOCAL_SCOPE,
            'Recorded eligibility differs from exact current structure')
    primary = review.validate_receipt(eligible['primary'])
    confirmation = review.validate_receipt(eligible['confirmation'])
    require(primary['purpose'] == 'candidate' and confirmation['purpose'] == 'confirmation'
            and primary['inputs'] == confirmation['inputs']
            and primary['pool_version'] == confirmation['pool_version'] == eligible['pool_version'],
            'Primary/confirmation inputs or common pool differ')
    require(confirmation['dependency'] == {key: primary[key] for key in
            ('purpose', 'output_dir', 'dispatch_id', 'finished_utc', 'receipt_hashes')},
            'Confirmation does not confirm this exact primary dispatch')
    require(primary['dispatch_id'] != confirmation['dispatch_id']
            and datetime.fromisoformat(primary['finished_utc']) <= datetime.fromisoformat(confirmation['started_utc']),
            'Confirmation was not fresh and sequential')
    require(all(type(x['score']) is int and 4 <= x['score'] <= 5 for x in (primary, confirmation)),
            'Both genuine scalar scores must be at least four')
    score = min(primary['score'], confirmation['score'])
    require(eligible['minimum_score'] == score, 'Eligible score differs from genuine receipts')
    for role, rows in [('source', source), ('target', target), ('raw', raw)]:
        digest = review.owner_rows_sha256(rows)
        require(primary['inputs'][role]['sha256'] == digest, 'Reviewed owner wording differs from candidate')
        if role != 'raw': require(eligible[role + '_sha256'] == digest, 'Eligible owner hash changed')
    require(primary['inputs']['context']['sha256'] == conf['context_sha256'], 'Original context changed')
    preparation = read('review-preparation', Path(primary['output_dir']) / 'preparation.json')
    bundle = review.read_bundle(preparation['manifest_path'])
    artifacts['bundle-manifest'] = _pin(Path(preparation['manifest_path']))
    pools = strict_json(bundle['raw_inputs']['evidence'])
    require(all(p['media_sha256'] == conf['media_sha256'] for p in pools), 'Reviewed audio evidence belongs to another media')
    baseline = review.validate_receipt(primary['dependency']['output_dir'])
    require(baseline['purpose'] == 'baseline' and baseline['pool_version'] == primary['pool_version']
            and baseline['inputs']['source']['sha256'] == review.owner_rows_sha256(raw)
            and baseline['inputs']['target']['sha256'] == review.owner_rows_sha256(_rows(paths['baseline-target-owners'])),
            'Same-pool baseline is not the immutable original comparison')
    for receipt in (primary, confirmation):
        matches = [x for x in state['reviews'].values() if x.get('dispatch_id') == receipt['dispatch_id']]
        require(len(matches) == 1 and matches[0]['receipt_hashes'] == receipt['receipt_hashes'], 'Controller receipt set changed')
    for role, pin in artifacts.items():
        require(_pin(pin['path']) == pin, 'Artifact changed during gate validation')
    return {'version': VERSION, 'status': 'contextual_ready', 'benchmark': BENCHMARK,
            'assessment_scope': ASSESSMENT_SCOPE, 'review_protocol': review.VERSION, 'campaign': str(folder),
            'round': round_id, 'score': score, 'pool_version': primary['pool_version'],
            'local_writer_family': candidate['writer'], 'writer_execution': 'local', 'owner_count': OWNER_COUNT,
            'local_gate_scope': LOCAL_SCOPE, 'legacy_or_unresolved_owner_count': len(candidate['legacy_or_unresolved_issue_owners']),
            'structural': structural, 'artifacts': artifacts,
            'reviews': {role: {'output_dir': value['output_dir'], 'dispatch_id': value['dispatch_id'],
                        'score': value['score'], 'receipt_hashes': value['receipt_hashes']}
                        for role, value in [('baseline', baseline), ('primary', primary), ('confirmation', confirmation)]},
            'source_accuracy_verified': False, 'audio_truth_verified': False, 'playback_verified': False,
            'whole_bundle_read_attested': False, 'source_verified_release_authorized': False,
            'selected_output_modified': False, 'reference_only': True,
            'producer_code_binding': 'immutable_execution_snapshots' if snapshots is not None else 'current_files_match_execution_pins',
            'snapshot_predispatch_attested': False,
            'retention_policy': 'Keep all referenced campaign and native evidence paths immutable and available'}


def prepare_package(campaign: str | Path, output_dir: str | Path) -> Path:
    """Create only a new versioned reference manifest after all gates pass."""
    directory = native._directory(output_dir)
    require(not directory.exists() and not directory.is_relative_to(ROOT / 'output/selected'),
            'Reference package must be new and must preserve selected output')
    value = validate_candidate(campaign)
    value['created_utc'] = datetime.now(timezone.utc).isoformat()
    directory.mkdir(parents=True, exist_ok=False)
    manifest = directory / 'manifest.json'; native._write(manifest, value)
    return manifest


def validate_package(manifest: str | Path) -> dict:
    """Revalidate real original receipts and artifact bytes, never trust score labels."""
    path = native._file(manifest); value = _read(path)
    require(isinstance(value, dict) and value.get('version') == VERSION, 'Unknown reference package')
    created = value.pop('created_utc', None)
    require(isinstance(created, str) and datetime.fromisoformat(created).tzinfo is not None, 'Missing actual package date')
    require(value == validate_candidate(value['campaign']), 'Reference package differs from validated campaign artifacts')
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--campaign', type=Path)
    source.add_argument('--manifest', type=Path)
    parser.add_argument('--prepare-dir', type=Path)
    args = parser.parse_args()
    try:
        if args.manifest:
            require(args.prepare_dir is None, 'Package validation is read-only')
            result = validate_package(args.manifest)
        elif args.prepare_dir:
            path = prepare_package(args.campaign, args.prepare_dir)
            print(json.dumps({'status': 'contextual_ready', 'reference_manifest': str(path), 'selected_output_modified': False}))
            return 0
        else:
            result = validate_candidate(args.campaign)
        print(json.dumps({k: result[k] for k in ('status', 'score', 'round', 'pool_version', 'owner_count',
                          'selected_output_modified', 'source_verified_release_authorized')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'blocked', 'error_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
