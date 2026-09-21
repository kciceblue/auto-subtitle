"""Bounded C1/C2 recap-conditioned Chinese patches; CPU preparation by default."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import logging
from pathlib import Path
import signal
import time

from src import contextual_evidence_review as er
from src import context_recap as recap
from src import evidence_context as ec
from src import revisit_workflow as rw
from src import sparse_edits as edits
from src import sparse_revisit as sr
from src.workflow_state import file_hash, fingerprint, write_json

ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT / 'output/quality-local-20260915'
PLAN = ROOT / 'design/quality-context-20260915.md'
VERSION = 'context-local-revisit-1'
LIMIT = 7200
ROUND_LIMIT = 4500
ROUNDS = ('C1', 'C2')
RECAP_REQUEST_KEY = 'source-recap-compact'
RECAP_MAX_CLAIMS = 8
LOG = logging.getLogger(__name__)
CONTEXT_RULE = '\nfocused_context中的全片地図与解釈只是可错假设。核对其全部支持、冲突和分支观察，不可把摘要当作声音真值或排除仍可行的读法。引用裁剪观察不等于证明邻句归属。'
DIAGNOSE = sr.DIAGNOSE + CONTEXT_RULE
VERIFY = sr.VERIFY + CONTEXT_RULE
GLOBAL = sr.GLOBAL + CONTEXT_RULE
PROPOSE = sr.RULES + CONTEXT_RULE + '''\n只修case指定的中文target_span，不生成或修改日文。返回且仅返回old_chinese和new_chinese两个字符串字段。old_chinese逐字等于target_span；new_chinese仅为该连续片段的替换文本。综合全部可行读法和给出的全片资料，保留人物、动作、否定、指代及不确定性，不重译整段，不改变邻句。没有可靠改进时返回原片段。'''
PROPOSAL_SCHEMA = rw.obj({k: rw.string(1800) for k in ('old_chinese', 'new_chinese')})


def read(path):
    return rw.read(Path(path))


def write(path, value):
    write_json(Path(path), value)


def _source_pair(folder):
    source, target = rw.rows(folder/'source.json'), rw.rows(folder/'target.json')
    er.owner_rows_sha256(source); er.owner_rows_sha256(target)
    if len(source) != 66 or [(r.index, r.ts_line) for r in source] != [(r.index, r.ts_line) for r in target]:
        raise ValueError('Expected exact paired 66-owner geometry')
    return source, target


def init(folder: Path):
    folder = Path(folder).resolve(); folder.mkdir(parents=True, exist_ok=True)
    if (folder/'campaign.json').exists():
        return finalize_preparation(folder)
    old = read(PREDECESSOR/'campaign.json')
    evidence = [str(Path(p).resolve()) for p in old['active_evidence_paths']]
    inherited = PREDECESSOR/'S2/analysis-expanded.json'
    analysis = read(inherited)
    if analysis['evidence_paths'] != evidence or analysis['source_hash'] != old['source_sha256'] or analysis['target_hash'] != old['target_sha256']:
        raise ValueError('Inherited source analysis differs from final evidence/base')
    baseline = er.validate_receipt(PREDECESSOR/'evaluation/expanded-baseline/primary')
    source, target = _source_pair(PREDECESSOR)
    if (baseline['purpose'] != 'baseline' or baseline['inputs']['source']['sha256'] != er.owner_rows_sha256(source)
        or baseline['inputs']['target']['sha256'] != er.owner_rows_sha256(target)
        or baseline['inputs']['context']['sha256'] != old['context_sha256']):
        raise ValueError('Inherited baseline receipt differs from immutable pair/context')
    input_hashes = {}; prior_score_paths = []
    for relative in ('S1/score-expanded.json', 'S2/score.json', 'S3/score.json'):
        prior_path = PREDECESSOR/relative
        if prior_path.exists():
            prior = read(prior_path)
            if prior.get('status') == 'complete' and prior.get('pool_version') == baseline['pool_version']:
                prior_score_paths.append(str(prior_path)); input_hashes[str(prior_path)] = file_hash(prior_path)
    for name, expected in [('source.json', old['source_sha256']), ('target.json', old['target_sha256']), ('original-context.txt', old['context_sha256'])]:
        original = PREDECESSOR/name
        if file_hash(original) != expected:
            raise ValueError('Predecessor immutable input changed')
        input_hashes[str(original)] = expected
        destination = folder/name
        if destination.exists():
            raise FileExistsError('Unregistered preparation output exists')
        destination.write_bytes(original.read_bytes()); input_hashes[str(destination)] = file_hash(destination)
    (folder/'inherited-source-analysis.json').write_bytes(inherited.read_bytes())
    for path in [inherited, folder/'inherited-source-analysis.json', PLAN, Path(old['writer_recipe']), *map(Path, evidence)]:
        input_hashes[str(path.resolve())] = file_hash(path)
    for name, digest in baseline['receipt_hashes'].items():
        path = Path(baseline['output_dir'])/name
        if file_hash(path) != digest:
            raise ValueError('Baseline receipt changed during preparation')
        input_hashes[str(path)] = digest
    context = (folder/'original-context.txt').read_text(encoding='utf-8')
    obs = sr.all_observations(list(map(Path, evidence)), source)
    prepared = recap.prepare_recap(source, obs, analysis['frames'], context, pool_versions=[baseline['pool_version']])
    write(folder/'source-pack.json', prepared['pack'])
    input_hashes[str(folder/'source-pack.json')] = file_hash(folder/'source-pack.json')
    conf = {'version': VERSION, 'created_utc': sr.now(), 'status': 'prepared', 'target': 4,
        'maximum_candidates': 2, 'maximum_local_seconds': LIMIT, 'maximum_round_seconds': ROUND_LIMIT,
        'local_seconds': 0, 'rounds': {}, 'predecessor': str(PREDECESSOR), 'plan': str(PLAN),
        'plan_sha256': input_hashes[str(PLAN)], 'input_hashes': input_hashes,
        'source_sha256': file_hash(folder/'source.json'), 'target_sha256': file_hash(folder/'target.json'),
        'context_sha256': file_hash(folder/'original-context.txt'), 'media': old['media'], 'media_sha256': old['media_sha256'],
        'writer_recipe': old['writer_recipe'], 'writer_recipe_sha256': old['writer_recipe_sha256'],
        'evidence_paths': evidence, 'pool_version': baseline['pool_version'], 'baseline_review': baseline['output_dir'],
        'inherited_analysis': str(inherited), 'inherited_analysis_sha256': file_hash(inherited), 'prior_score_paths': prior_score_paths,
        'source_pack_sha256': prepared['pack']['pack_sha256'], 'external_feedback': 'score_only',
        'source_edits_allowed': False, 'raw_source_fidelity_verified': False, 'human_reference_used': False,
        'visual_input_used': False, 'new_audio_acquired': False, 'selected_output_modified': False}
    write(folder/'campaign.json', conf)
    return finalize_preparation(folder)


def finalize_preparation(folder):
    """Pin inherited native provenance before any new dispatch; never refresh an epoch."""
    conf = check(folder)
    if conf['status'] != 'prepared':
        return conf
    predecessor = Path(conf['predecessor']).resolve(); old = read(predecessor/'campaign.json')
    pins = dict(conf['input_hashes'])
    def pin(path):
        path = Path(path).resolve()
        if not path.is_relative_to(predecessor):
            raise ValueError('Inherited receipt escapes predecessor')
        digest = file_hash(path)
        if str(path) in pins and pins[str(path)] != digest:
            raise ValueError('Inherited receipt changed')
        pins[str(path)] = digest
        return path
    for n in range(1, 12):
        path = pin(predecessor/f'S2/requests/expanded-source-{n}.json'); value = read(path)
        pin(path.parent/'metrics.jsonl')
        if value['status'] == 'reused':
            original = pin(value['inherited_from'])
            if value['inherited_sha256'] != file_hash(original) or value.get('attempts') != []:
                raise ValueError('Inherited native alias changed')
            inherited = read(original)
            if value['request_sha256'] != inherited.get('request_sha256') or fingerprint(value['request']) != value['request_sha256']:
                raise ValueError('Inherited native alias request differs')
            value = inherited; path = original
        if value['status'] != 'complete':
            raise ValueError('Inherited source request did not complete')
        pin(path.parent/'metrics.jsonl')
        request = value['request']
        # A confirmed complete path only: this replays validation, never dispatches.
        sr.ask(path.parent, path.stem, request['family'], '', request['instruction'], request['body'], request['schema'], thinking=request['thinking'])
    metadata = {k: old[k] for k in ('code_pins', 'producer_snapshot_manifest')}
    metadata.update(predecessor=str(predecessor), code_epochs=old.get('code_epochs', []))
    for epoch in [*metadata['code_epochs'], metadata]:
        path = pin(epoch['producer_snapshot_manifest']); manifest = read(path)
        if set(manifest) != set(epoch['code_pins']):
            raise ValueError('Inherited producer pin set changed')
        if epoch.get('producer_snapshot_manifest_sha256', file_hash(path)) != file_hash(path):
            raise ValueError('Inherited producer manifest changed')
        for original, spec in manifest.items():
            snapshot = pin(spec['snapshot'])
            if not snapshot.is_relative_to(path.parent) or spec['sha256'] != epoch['code_pins'][original] or file_hash(snapshot) != spec['sha256']:
                raise ValueError('Inherited producer snapshot changed')
    metadata_path = folder/'inherited-producer-metadata.json'
    if metadata_path.exists() and read(metadata_path) != metadata:
        raise ValueError('Inherited producer metadata changed')
    if not metadata_path.exists():
        write(metadata_path, metadata)
    pins[str(metadata_path)] = file_hash(metadata_path)
    prior_paths = []
    for relative in ('S1/score-expanded.json', 'S2/score.json', 'S3/score.json'):
        path = predecessor/relative
        if path.exists():
            prior = read(path)
            if prior.get('status') == 'complete' and prior.get('pool_version') == conf['pool_version']:
                prior_paths.append(str(pin(path)))
    conf.update(input_hashes=pins, prior_score_paths=prior_paths, inherited_producer_metadata=str(metadata_path))
    write(folder/'campaign.json', conf)
    return check(folder)


def check(folder):
    conf = read(folder/'campaign.json')
    if (conf['version'] != VERSION or conf['maximum_candidates'] != 2 or conf['maximum_local_seconds'] != LIMIT
        or conf['maximum_round_seconds'] != ROUND_LIMIT or conf['source_edits_allowed'] is not False):
        raise ValueError('Context campaign contract changed')
    for path, digest in {**conf['input_hashes'], **conf.get('code_pins', {})}.items():
        if file_hash(Path(path)) != digest:
            raise ValueError('Pinned input or producer changed')
    if conf['plan_sha256'] != file_hash(PLAN) or conf['writer_recipe_sha256'] != file_hash(Path(conf['writer_recipe'])):
        raise ValueError('Plan or writer recipe changed')
    return conf


@contextmanager
def measured(folder, round_id, stage):
    conf = check(folder); record = conf['rounds'].setdefault(round_id, {'local_seconds': 0, 'stages': []})
    allowance = min(LIMIT-conf['local_seconds'], ROUND_LIMIT-record['local_seconds'])
    if allowance <= 0:
        raise rw.LocalBudgetExceeded('Context local budget exhausted')
    start = time.monotonic(); prior_deadline = sr._DEADLINE; sr._DEADLINE = start + allowance
    prior_handler = signal.getsignal(signal.SIGALRM); prior_timer = signal.getitimer(signal.ITIMER_REAL)
    def expired(signum, frame):
        raise rw.LocalBudgetExceeded('Context local stage deadline')
    signal.signal(signal.SIGALRM, expired); signal.setitimer(signal.ITIMER_REAL, allowance)
    error = None
    try:
        yield
    except BaseException as exc:
        error = type(exc).__name__; raise
    finally:
        elapsed = time.monotonic()-start; signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler); sr._DEADLINE = prior_deadline
        if prior_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, max(.001, prior_timer[0]-elapsed), prior_timer[1])
        conf = read(folder/'campaign.json'); record = conf['rounds'].setdefault(round_id, {'local_seconds': 0, 'stages': []})
        record['local_seconds'] += elapsed; conf['local_seconds'] += elapsed
        record['stages'].append({'name': stage, 'seconds': elapsed, 'error': error, 'finished_utc': sr.now()})
        write(folder/'campaign.json', conf); LOG.info('%s %s %.2fs error=%s', round_id, stage, elapsed, error)


def _cached(path, binding):
    receipt = path.with_suffix('.binding.json')
    if not path.exists() and not receipt.exists():
        return None
    if not path.exists() or not receipt.exists():
        raise ValueError('Incomplete cached stage binding')
    saved = read(receipt)
    if saved['binding'] != binding or saved['sha256'] != file_hash(path):
        raise ValueError('Cached stage input/output changed')
    return read(path)


def _save(path, result, binding):
    write(path, result); write(path.with_suffix('.binding.json'), {'binding': binding, 'sha256': file_hash(path)})
    return result


def prepare_recap_request(pack):
    prepared = recap.prepare_recap(pack['source_rows'], pack['observations'], pack['frames'], pack['original_context'], pool_versions=pack['pool_versions'])
    prepared['schema']['properties']['claims']['maxItems'] = RECAP_MAX_CLAIMS
    prepared['instruction'] += '\n今回の出力は最大8件の短いclaimに限定してください。入力全体と競合証拠は省略せず検討し、少数の重要な関係を選んでください。件数を満たす必要はありません。各claimは簡潔に、必要な反証と分岐は保持してください。'
    return prepared


def source_recap(folder):
    conf = check(folder); pack = read(folder/'source-pack.json'); ec._check_pack(pack)
    prepared = prepare_recap_request(pack)
    binding = {'pack_sha256': pack['pack_sha256'], 'request': fingerprint({k: prepared[k] for k in ('instruction', 'body', 'schema')})}
    path = folder/'shared/source-map.json'; cached = _cached(path, binding)
    if cached is not None:
        recap.focus_context(pack, cached, [1]); return pack, cached
    with measured(folder, 'C1', 'source_evidence_recap'):
        with rw.backend('qwen', folder/'C1/recap-backend', Path(conf['writer_recipe'])) as (endpoint, _):
            raw = sr.ask(folder/'C1/requests', RECAP_REQUEST_KEY, 'qwen', endpoint, prepared['instruction'], prepared['body'], prepared['schema'], thinking=True)
        bound = ec.validate_context_map(raw, pack)
    return pack, _save(path, bound, binding)


def analyze(folder, pack, bound_map):
    conf = check(folder); source, target = _source_pair(folder); context = pack['original_context']; frames = pack['frames']
    binding = {'pack_sha256': pack['pack_sha256'], 'map_sha256': bound_map['map_sha256'], 'target_sha256': conf['target_sha256'], 'instruction': fingerprint(DIAGNOSE)}
    path = folder/'shared/analysis.json'; cached = _cached(path, binding)
    if cached is not None:
        return cached
    cases = []; invalid = []; coverage = []
    case_schema = rw.obj({'target_span': rw.string(1000), 'source_span': rw.string(1000),
        'category': {'type': 'string', 'enum': ['mistranslation', 'omission', 'unsupported_specificity', 'negation', 'intent', 'terminology']},
        'severity': {'type': 'integer', 'minimum': 1, 'maximum': 3}, 'issue': rw.string(1200), 'unresolved': {'type': 'boolean'}})
    with measured(folder, 'C1', 'contextual_material_discrepancies'):
        with rw.backend('qwen', folder/'C1/analysis-backend', Path(conf['writer_recipe'])) as (endpoint, _):
            for number, ids in enumerate(sr.chunk_ids(source), 1):
                focused = recap.focus_context(pack, bound_map, ids)
                body = {**sr.episode_body(source, target, context), 'focus_ids': ids,
                    'source_readings': {str(i): frames[str(i)] for i in ids}, 'focused_context': focused}
                value = sr.ask(folder/'C1/requests', f'discrepancy-{number}', 'qwen', endpoint, DIAGNOSE, body,
                    rw.keyed(ids, rw.obj({'cases': rw.arr(case_schema, 2)})), thinking=True)
                coverage.extend(ids)
                for owner, row in value.items():
                    for number, case in enumerate(row['cases'], 1):
                        case = {**case, 'owner_id': int(owner), 'case_id': f'C1-o{int(owner):03d}-c{number}', 'readings': frames[owner]['readings'],
                            'unresolved': case['unresolved'] or frames[owner]['unresolved']}
                        try:
                            cases.append(edits.validate_case(case, source, target, pack['observations']))
                        except ValueError as exc:
                            invalid.append({'case_id': case['case_id'], 'error_type': type(exc).__name__})
    if coverage != list(range(1, 67)):
        raise ValueError('Incomplete diagnosis coverage')
    return _save(path, {'version': VERSION, 'cases': cases, 'invalid_cases': invalid, 'coverage': coverage,
        'pack_sha256': pack['pack_sha256'], 'map_sha256': bound_map['map_sha256'], 'external_feedback_used': False}, binding)


def check_global(folder, round_id, source, target, transactions, pack, bound_map, checker, endpoint, prefix):
    previous_source, previous_target = _source_pair(folder); result = {}; obs = {o['observation_id']: o for o in pack['observations']}
    ids_all = [t['case_id'] for t in transactions]
    for number, ids in enumerate(sr.chunk_ids(source), 1):
        own = [o for o in pack['observations'] if o['owner_id'] in ids]
        regression = rw.obj({'transaction_ids': rw.arr({'type': 'string', 'enum': ids_all}, 6),
            'before_quote': rw.string(1000), 'after_quote': rw.string(1000),
            'observation_id': {'type': 'string', 'enum': [o['observation_id'] for o in own]}, 'source_quote': rw.string(1200), 'reason': rw.string(1200)})
        body = {**sr.episode_body(previous_source, previous_target, pack['original_context']), 'candidate_japanese': rw.rowmap(source),
            'candidate_chinese': rw.rowmap(target), 'focus_ids': ids,
            'transactions': [{k: t[k] for k in ('case_id', 'owner_id', 'old_chinese', 'new_chinese', 'old_japanese', 'new_japanese')} for t in transactions],
            'focused_context': recap.focus_context(pack, bound_map, ids)}
        value = sr.ask(folder/round_id/'requests', f'{prefix}-{number}', checker, endpoint, GLOBAL, body,
            rw.keyed(ids, rw.obj({'regressions': rw.arr(regression, 6), 'legacy_or_source_uncertainty': rw.string(1200)})), thinking=checker == 'qwen')
        for owner, row in value.items():
            for finding in row['regressions']:
                witness = obs[finding['observation_id']]
                finding['quotes_valid'] = bool(finding['transaction_ids'] and finding['before_quote'] and finding['after_quote'] and finding['source_quote']
                    and witness['owner_id'] == int(owner) and finding['before_quote'] in previous_target[int(owner)-1].text
                    and finding['after_quote'] in target[int(owner)-1].text and finding['source_quote'] in witness['text'])
        result.update(value)
    return result


def candidate_binding(folder, round_id, pack, bound_map):
    conf = check(folder)
    return {'round': round_id, 'pack_sha256': pack['pack_sha256'], 'map_sha256': bound_map['map_sha256'], 'analysis_sha256': file_hash(folder/'shared/analysis.json'), 'target_sha256': conf['target_sha256']}


def build_candidate(folder, round_id, pack, bound_map, analysis):
    conf = check(folder); out = folder/round_id; out.mkdir(parents=True, exist_ok=True)
    binding = candidate_binding(folder, round_id, pack, bound_map)
    cached = _cached(out/'candidate.json', binding)
    if cached is not None:
        if cached['source_sha256'] != file_hash(out/'source.json') or cached['target_sha256'] != file_hash(out/'target.json'):
            raise ValueError('Cached candidate owner files changed')
        return cached
    source, target = _source_pair(folder); obs = pack['observations']; cases = sr.choose_cases(analysis, 24)
    write(out/'selected-cases.json', cases); write(out/'analysis.json', analysis)
    writer, checker = ('gemma', 'qwen') if round_id == 'C1' else ('qwen', 'gemma')
    proposed = []; decisions = []; accepted = []; rolled = []
    if cases:
        with measured(folder, round_id, 'chinese_only_sparse_proposals'):
            with rw.backend(writer, out/'writer-backend', Path(conf['writer_recipe'])) as (endpoint, _):
                for case in cases:
                    body = {**sr.episode_body(source, target, pack['original_context']), 'case': case,
                        'focused_context': recap.focus_context(pack, bound_map, [case['owner_id']])}
                    native = sr.ask(out/'requests', 'propose-'+case['case_id'], writer, endpoint, PROPOSE, body, PROPOSAL_SCHEMA, thinking=writer == 'qwen')
                    proposal = {**native, **{k: '' for k in ('old_japanese', 'new_japanese', 'source_observation_id', 'source_support_quote')}}
                    proposed.append(edits.prepare_patch(case, proposal, source, target, obs, allow_source=False))
        write(out/'proposed-transactions.json', proposed)
        with measured(folder, round_id, 'contrastive_context_verification'):
            with rw.backend(checker, out/'checker-backend', Path(conf['writer_recipe'])) as (endpoint, _):
                for case, transaction in zip(cases, proposed):
                    if not transaction['valid'] or transaction['no_op']:
                        continue
                    own = [o for o in obs if o['owner_id'] == case['owner_id']]
                    body = {'episode_japanese': rw.rowmap(source), 'episode_chinese': rw.rowmap(target), 'original_context': pack['original_context'],
                        'case': case, 'before_japanese': transaction['before_source_text'], 'before_chinese': transaction['before_target_text'],
                        'after_japanese': transaction['after_source_text'], 'after_chinese': transaction['after_target_text'], 'source_readings': case['readings'],
                        'focused_context': recap.focus_context(pack, bound_map, [case['owner_id']])}
                    verdict = sr.ask(out/'requests', 'verify-'+case['case_id'], checker, endpoint, VERIFY, body,
                        sr.decision_schema([o['observation_id'] for o in own]), thinking=checker == 'qwen')
                    keep = sr.accept_decision(verdict, case, transaction, obs)
                    decisions.append({'case_id': case['case_id'], 'accepted': keep, 'verdict': verdict})
                    if keep:
                        accepted.append(transaction)
    write(out/'local-decisions.json', decisions)
    current_source, current_target, ledger = edits.apply_transactions(source, target, accepted)
    if accepted:
        with measured(folder, round_id, 'all_owner_incremental_context_verification'):
            with rw.backend(checker, out/'global-backend', Path(conf['writer_recipe'])) as (endpoint, _):
                checks = check_global(folder, round_id, current_source, current_target, accepted, pack, bound_map, checker, endpoint, 'global')
                rolled = sorted({cid for row in checks.values() for f in row['regressions'] if f['quotes_valid'] for cid in f['transaction_ids']})
                if rolled:
                    accepted = [t for t in accepted if t['case_id'] not in rolled]
                    current_source, current_target, ledger = edits.apply_transactions(source, target, accepted)
                    checks = check_global(folder, round_id, current_source, current_target, accepted, pack, bound_map, checker, endpoint, 'assembled') if accepted else {}
    else:
        checks = {}
    if not accepted:
        checks = {str(r.index): {'regressions': [], 'legacy_or_source_uncertainty': 'Unchanged retained baseline; no generated delta'} for r in source}
    complete = set(checks) == set(rw.rowmap(source)) and analysis['coverage'] == list(range(1, 67))
    gate = complete and not any(f['quotes_valid'] for row in checks.values() for f in row['regressions'])
    if current_source != source:
        raise ValueError('Chinese-only campaign changed Japanese')
    write(out/'source.json', [asdict(r) for r in current_source]); write(out/'target.json', [asdict(r) for r in current_target])
    sr.write_checked(current_source, out/'source.utterances.srt'); sr.write_checked(current_target, out/'target.utterances.srt')
    result = {'version': VERSION, 'round': round_id, 'source_sha256': file_hash(out/'source.json'), 'target_sha256': file_hash(out/'target.json'),
        'cases_considered': len(cases), 'valid_proposals': sum(t['valid'] and not t['no_op'] for t in proposed),
        'accepted_transactions': accepted, 'transaction_ledger': ledger, 'rolled_back_transactions': rolled, 'global_checks': checks,
        'local_gate_passed': gate, 'coverage': len(checks), 'changed_owners': [r.index for r, old in zip(current_target, target) if r.text != old.text],
        'changed_target_owners': [r.index for r, old in zip(current_target, target) if r.text != old.text], 'writer': writer, 'checker': checker,
        'evidence_paths': conf['evidence_paths'], 'pack_sha256': pack['pack_sha256'], 'map_sha256': bound_map['map_sha256'],
        'raw_source_fidelity_verified': False, 'external_feedback_used': False, 'human_reference_used': False, 'source_edits_allowed': False}
    return _save(out/'candidate.json', result, binding)


def score(folder, round_id, baseline):
    conf = check(folder); candidate = sr.validate_candidate_artifacts(folder, round_id)
    target_hash = er.owner_rows_sha256(rw.rows(folder/round_id/'target.json'))
    prior_paths = [Path(p) for p in conf.get('prior_score_paths', [])]
    if round_id == 'C2':
        prior_paths.append(folder/'C1/score.json')
    for prior_path in prior_paths:
        if not prior_path.exists():
            continue
        prior = read(prior_path)
        if prior.get('status') == 'complete' and prior.get('pool_version') == baseline['pool_version'] and prior.get('target_sha256') == target_hash:
            result = {'status': 'complete', 'score': prior['score'], 'confirmed': False, 'pool_version': baseline['pool_version'],
                'target_sha256': target_hash, 'inherited_duplicate_target': str(prior_path), 'inherited_score_sha256': file_hash(prior_path)}
            write(folder/round_id/'score.json', result); return result
    return sr.score(folder, round_id, list(map(Path, conf['evidence_paths'])), baseline)


def validate_score(folder, round_id, pack, bound_map, baseline):
    """A cached scalar record cannot bypass candidate or genuine dispatch validation."""
    conf = check(folder); out = folder/round_id
    cached = _cached(out/'candidate.json', candidate_binding(folder, round_id, pack, bound_map))
    if cached is None:
        raise ValueError('Cached score lacks its bound candidate')
    candidate = sr.validate_candidate_artifacts(folder, round_id)
    source, target = _source_pair(out); base_source, base_target = _source_pair(folder)
    if source != base_source or candidate['pack_sha256'] != pack['pack_sha256'] or candidate['map_sha256'] != bound_map['map_sha256']:
        raise ValueError('Scored candidate source or context changed')
    target_hash = er.owner_rows_sha256(target); source_hash = er.owner_rows_sha256(source)
    result = read(out/'score.json')
    if (result.get('status') != 'complete' or type(result.get('score')) is not int or not -10 <= result['score'] <= 5
        or type(result.get('confirmed')) is not bool or result.get('pool_version') != baseline['pool_version']
        or result.get('target_sha256') != target_hash):
        raise ValueError('Invalid cached score identity')
    if result.get('inherited_unchanged_baseline'):
        if result['confirmed'] or target_hash != er.owner_rows_sha256(base_target) or result['score'] != baseline['score']:
            raise ValueError('Invalid unchanged baseline inheritance')
        return result
    if result.get('inherited_duplicate_target'):
        path = Path(result['inherited_duplicate_target'])
        allowed = list(map(Path, conf.get('prior_score_paths', []))) + ([folder/'C1/score.json'] if round_id == 'C2' else [])
        if path not in allowed or file_hash(path) != result.get('inherited_score_sha256') or result['confirmed']:
            raise ValueError('Invalid duplicate score provenance')
        prior = read(path)
        if any(prior.get(k) != result[k] for k in ('status', 'pool_version', 'target_sha256', 'score')):
            raise ValueError('Inherited scalar changed')
        if path == folder/'C1/score.json':
            validate_score(folder, 'C1', pack, bound_map, baseline)
        return result
    manifest = out/'evaluation/initial/bundle/manifest.json'
    bundle = er.read_bundle(manifest)
    expected = {'source': source_hash, 'target': target_hash, 'raw': er.owner_rows_sha256(base_source), 'context': conf['context_sha256']}
    if bundle['pool_version'] != conf['pool_version'] or any(bundle['inputs'][k]['sha256'] != v for k, v in expected.items()):
        raise ValueError('Cached review bundle differs from candidate')
    primary = er.validate_receipt(result['primary'], manifest)
    dependency_keys = ('purpose', 'output_dir', 'dispatch_id', 'finished_utc', 'receipt_hashes')
    if (primary['purpose'] != 'candidate' or primary['pool_version'] != conf['pool_version']
        or primary['dependency'] != {k: baseline[k] for k in dependency_keys}
        or primary['score'] != result['score'] or result.get('source_sha256') != source_hash
        or result.get('local_gate_passed') is not candidate['local_gate_passed']):
        raise ValueError('Cached primary differs from actual receipt')
    needs_confirmation = primary['score'] >= 4 and candidate['local_gate_passed']
    if needs_confirmation:
        confirm = er.validate_receipt(result['confirmation'], manifest)
        if (confirm['purpose'] != 'confirmation' or confirm['dependency'] != {k: primary[k] for k in dependency_keys}
            or confirm['dispatch_id'] == primary['dispatch_id'] or result.get('confirmation_score') != confirm['score']
            or result['confirmed'] is not (confirm['score'] >= 4)):
            raise ValueError('Cached confirmation differs from actual receipt')
    elif result['confirmed'] or 'confirmation' in result or 'confirmation_score' in result:
        raise ValueError('Cached score asserts an ineligible confirmation')
    return result


def freeze_producers(folder):
    conf = check(folder)
    names = ['context_revisit', 'context_recap', 'evidence_context', 'sparse_revisit', 'sparse_edits', 'revisit_workflow',
        'contextual_evidence_review', 'late_audio', 'selected_pipeline', 'display', 'translate', 'config', 'coherence',
        'workflow_state', 'local_backend', 'contextual_review_contract', 'contextual_review', 'pause_layout', 'contextual_evidence_release']
    pins = {str(ROOT/'src'/f'{name}.py'): file_hash(ROOT/'src'/f'{name}.py') for name in names}
    if 'code_pins' in conf and conf['code_pins'] != pins:
        raise ValueError('Producer epoch changed')
    manifest = {}
    for original, digest in pins.items():
        dst = folder/conf.get('producer_snapshot_dir', 'producer-code')/Path(original).relative_to(ROOT); dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and file_hash(dst) != digest:
            raise ValueError('Producer snapshot changed')
        if not dst.exists():
            dst.write_bytes(Path(original).read_bytes()); dst.chmod(0o444)
        manifest[original] = {'snapshot': str(dst), 'sha256': digest}
    path = folder/conf.get('producer_snapshot_dir', 'producer-code')/'manifest.json'
    if path.exists() and read(path) != manifest:
        raise ValueError('Producer manifest changed')
    if not path.exists():
        write(path, manifest)
    conf.update(code_pins=pins, producer_snapshot_manifest=str(path)); write(folder/'campaign.json', conf)


def execute(folder):
    folder = Path(folder).resolve(); conf = init(folder)
    if conf['status'] in {'confirmed_four', 'candidate_budget_exhausted', 'time_budget_exhausted', 'failed', 'interrupted'}:
        return conf
    previous_campaign = read(Path(conf['predecessor'])/'campaign.json')
    if previous_campaign['status'] not in {'candidate_budget_exhausted', 'time_budget_exhausted'} or any(r.get('confirmed') for r in previous_campaign.get('rounds', {}).values()):
        raise ValueError('Predecessor must finish without confirmed four')
    freeze_producers(folder); baseline = er.validate_receipt(conf['baseline_review'])
    if baseline['pool_version'] != conf['pool_version']:
        raise ValueError('Baseline pool changed')
    initial = rw._request_json(rw.ADMIN+'/status', admin=True); previous = initial.get('loaded_model')
    if previous is None and initial.get('state') != 'idle':
        raise ValueError('Unstable initial backend')
    conf = check(folder); conf.update(status='running', started_utc=sr.now(), active_round='C1')
    conf['rounds'].setdefault('C1', {'local_seconds': 0, 'stages': []})['status'] = 'reserved'
    write(folder/'campaign.json', conf)
    try:
        pack, bound_map = source_recap(folder); analysis = analyze(folder, pack, bound_map)
        for round_id in ROUNDS:
            conf = check(folder); conf['active_round'] = round_id
            record = conf['rounds'].setdefault(round_id, {'local_seconds': 0, 'stages': []}); record['status'] = 'reserved'; write(folder/'campaign.json', conf)
            if conf['local_seconds'] >= LIMIT:
                raise rw.LocalBudgetExceeded('Campaign budget exhausted')
            cached_score = folder/round_id/'score.json'
            if cached_score.exists() and read(cached_score).get('status') == 'complete':
                result = validate_score(folder, round_id, pack, bound_map, baseline)
            else:
                candidate = build_candidate(folder, round_id, pack, bound_map, analysis)
                display = folder/round_id/'display'
                if candidate['changed_target_owners'] and not (display/'display.json').exists():
                    display.mkdir(exist_ok=True)
                    with measured(folder, round_id, 'exact_text_display'):
                        with rw.backend('qwen', folder/round_id/'display-backend', Path(conf['writer_recipe'])):
                            sr._display(rw.rows(folder/round_id/'source.json'), rw.rows(folder/round_id/'target.json'), display)
                result = score(folder, round_id, baseline)
            conf = check(folder); conf['rounds'][round_id].update(status='scored', score=result['score'], confirmed=result.get('confirmed', False)); write(folder/'campaign.json', conf)
            if result.get('confirmed'):
                conf.update(status='confirmed_four', eligible_candidate=round_id, eligible_score_receipt=str(cached_score)); write(folder/'campaign.json', conf); break
        else:
            conf = check(folder); conf['status'] = 'candidate_budget_exhausted'; write(folder/'campaign.json', conf)
    except BaseException as exc:
        conf = read(folder/'campaign.json'); conf['status'] = 'time_budget_exhausted' if isinstance(exc, rw.LocalBudgetExceeded) else 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        conf['error_type'] = type(exc).__name__; write(folder/'campaign.json', conf); raise
    finally:
        state = rw._request_json(rw.ADMIN+'/status', admin=True)
        if state.get('loaded_model') != previous or previous is None and state.get('state') != 'idle':
            rw._request_json(rw.ADMIN+('/unload' if previous is None else '/load'), {} if previous is None else {'model': previous}, admin=True, timeout=360)
        restored = rw._request_json(rw.ADMIN+'/status', admin=True)
        ok = restored.get('loaded_model') == previous and (previous is not None or restored.get('state') == 'idle')
        conf = read(folder/'campaign.json'); conf['original_backend_restored'] = ok; write(folder/'campaign.json', conf)
        if not ok:
            raise RuntimeError('Original backend restoration failed')
    return check(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT/'output/quality-context-20260915')
    parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.execute:
        import fcntl
        args.campaign.mkdir(parents=True, exist_ok=True)
        with (args.campaign/'execution.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); conf = execute(args.campaign)
    else:
        conf = init(args.campaign)
    print(json.dumps({k: conf.get(k) for k in ('status', 'active_round', 'local_seconds', 'eligible_candidate', 'original_backend_restored')}))


if __name__ == '__main__':
    main()
