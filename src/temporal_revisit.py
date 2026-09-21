"""One bounded temporal source reinterpretation revisit; all writing stays local."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import json
import logging
from pathlib import Path
import signal
import time
from src import context_revisit as cr
from src import context_release as replay
from src import context_recap as recap
from src import evidence_context as ec
from src import short_audio
from src import recap_index
from src import temporal_evidence as te
from src import temporal_source_frames as tf
from src import late_audio
from src import sparse_revisit as sr
from src import sparse_edits as edits
from src import revisit_workflow as rw
from src import contextual_evidence_review as er
from src.contextual_review_contract import require
from src.workflow_state import file_hash,fingerprint,write_json
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'design/quality-temporal-20260915.md'
PREDECESSOR=ROOT/'output/quality-short-20260915'
VERSION='temporal-source-local-revisit-1'
ROUND='T1'
LIMIT=ROUND_LIMIT=7200
LOG=logging.getLogger(__name__)
RECAP_REQUEST_KEY='temporal-source-recap'
TEMPORAL_RULE = '\n音声观察按实际裁剪时间分组：不同或不重合短片段通常互补，不是整个owner的竞争译法。某短片段未包含词语不等于反证；只对该片段实际表达的内容判断支持/冲突。不得据时间邻近推定词语或说话人归属。保持严格修复条件，不能用此规则忽视片段中的真实反证。temporal_groups是无损列式表：rows每个元素按columns解读，owner_interval_ms及owner_intersection按各自列名解读，分数是[分子,分母]。'
DIAGNOSE,VERIFY,GLOBAL,PROPOSE=(p+TEMPORAL_RULE for p in (cr.DIAGNOSE,cr.VERIFY,cr.GLOBAL,cr.PROPOSE))
PROPOSAL_SCHEMA=cr.PROPOSAL_SCHEMA
read,write,_source_pair,_cached,_save=cr.read,cr.write,cr._source_pair,cr._cached,cr._save



def decision_schema(observation_ids):
    # Every viable short and long observation must fit; acceptance stays strict.
    require(bool(observation_ids) and len(observation_ids)==len(set(observation_ids)), 'Invalid verifier observations')
    schema=sr.decision_schema(observation_ids)
    schema['properties']['reading_checks']['maxItems']=len(observation_ids)
    return schema


def init(folder):
    folder=Path(folder).resolve()
    if (folder/'campaign.json').exists():return check(folder)
    old=read(PREDECESSOR/'campaign.json')
    require(old.get('original_backend_restored') is True and old['status'] not in ('prepared','running') and not old.get('eligible_candidate'), 'N1 must finish safely without a confirmed four')
    source,target=_source_pair(PREDECESSOR)
    baseline=er.validate_receipt(old['baseline_review'])
    require(baseline['purpose']=='baseline' and baseline['pool_version']==old['pool_version']
        and baseline['inputs']['source']['sha256']==er.owner_rows_sha256(source)
        and baseline['inputs']['target']['sha256']==er.owner_rows_sha256(target)
        and baseline['inputs']['context']['sha256']==old['context_sha256'],'Inherited baseline differs')
    short_audio.validate_evidence(PREDECESSOR/'evidence/short')
    folder.mkdir(parents=True,exist_ok=True);require(not any(folder.iterdir()),'Unregistered T1 output exists')
    pins={str(PLAN):file_hash(PLAN),str(PREDECESSOR/'campaign.json'):file_hash(PREDECESSOR/'campaign.json')}
    for name in ('source.json','target.json','original-context.txt'):
        original=PREDECESSOR/name;destination=folder/name;destination.write_bytes(original.read_bytes())
        pins[str(original)]=file_hash(original);pins[str(destination)]=file_hash(destination)
    for p in [Path(old['writer_recipe']),*map(Path,old['evidence_paths'])]:pins[str(p)]=file_hash(p)
    for name,digest in baseline['receipt_hashes'].items():
        p=Path(baseline['output_dir'])/name;require(file_hash(p)==digest,'Inherited baseline changed');pins[str(p)]=digest
    prior=[]
    if (PREDECESSOR/'N1/score.json').exists():
        score=read(PREDECESSOR/'N1/score.json')
        if score.get('status')=='complete' and score['pool_version']==baseline['pool_version']:
            p=PREDECESSOR/'N1/score.json';prior.append(str(p));pins[str(p)]=file_hash(p)
    conf={k:deepcopy(old[k]) for k in ('media','media_sha256','writer_recipe','writer_recipe_sha256','evidence_paths','pool_version','baseline_review')}
    conf.update(version=VERSION,status='prepared',created_utc=sr.now(),target=4,maximum_candidates=1,
        maximum_local_seconds=LIMIT,maximum_round_seconds=LIMIT,local_seconds=0.,rounds={},plan=str(PLAN),plan_sha256=file_hash(PLAN),
        source_sha256=file_hash(folder/'source.json'),target_sha256=file_hash(folder/'target.json'),context_sha256=file_hash(folder/'original-context.txt'),
        input_hashes=pins,starting_candidate=old['starting_candidate'],starting_candidate_score=baseline['score'],
        predecessor=str(PREDECESSOR),prior_score_paths=prior,short_evidence_directory=str(PREDECESSOR/'evidence/short'),
        external_feedback='score_only',source_edits_allowed=False,raw_source_fidelity_verified=False,
        human_reference_used=False,visual_input_used=False,new_audio_acquired=False,selected_output_modified=False)
    write(folder/'campaign.json',conf);return check(folder)


def check(folder):
    folder=Path(folder).resolve();conf=read(folder/'campaign.json')
    require(conf['version']==VERSION and conf['maximum_candidates']==1 and conf['maximum_local_seconds']==LIMIT
        and conf['maximum_round_seconds']==LIMIT and conf['external_feedback']=='score_only'
        and all(conf[k] is False for k in ('source_edits_allowed','raw_source_fidelity_verified','human_reference_used','visual_input_used','new_audio_acquired','selected_output_modified')),'T1 experiment contract changed')
    for path,digest in {**conf['input_hashes'],**conf.get('code_pins',{})}.items():require(file_hash(Path(path))==digest,'Pinned T1 input changed')
    return conf


def freeze(folder):
    conf=check(folder)
    names=['temporal_revisit','temporal_release','temporal_evidence','temporal_source_frames','short_audio','recap_index',
        'context_revisit','context_release','context_recap','evidence_context','sparse_revisit','sparse_edits','sparse_release',
        'revisit_workflow','late_audio','adjudicate','local_backend','contextual_evidence_review','contextual_evidence_release',
        'contextual_review_contract','contextual_review','selected_pipeline','display','translate','config','coherence','workflow_state','pause_layout']
    pins={str(ROOT/'src'/f'{name}.py'):file_hash(ROOT/'src'/f'{name}.py') for name in names};manifest={}
    for original,digest in pins.items():
        destination=folder/'producer-code'/Path(original).relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(Path(original).read_bytes());destination.chmod(0o444)
        manifest[original]={'snapshot':str(destination),'sha256':digest}
    write(folder/'producer-code/manifest.json',manifest)
    conf.update(code_pins=pins,producer_snapshot_manifest=str(folder/'producer-code/manifest.json'));write(folder/'campaign.json',conf)


def source_frame_requests(source,observations,context):
    ids=[row.index for row in source]
    return [tf.build_temporal_source_request(source,observations,context,ids[i:i+2]) for i in range(0,len(ids),2)]


def compile_source_frames(responses,requests):
    require(len(responses)==len(requests),'Source frame response coverage changed')
    result={'frames':{},'fallback_owner_ids':[],'owner_diagnostics':{}}
    for raw,request in zip(responses,requests):
        compiled=tf.compile_temporal_source_frames(raw,request)
        require(not(set(result['frames'])&set(compiled['frames'])),'Duplicate compiled frame owner')
        result['frames'].update(compiled['frames']);result['fallback_owner_ids'].extend(compiled['fallback_owner_ids'])
        result['owner_diagnostics'].update(compiled['owner_diagnostics'])
    return result


def frame_binding(folder,conf,observations,requests):
    return {'observations_sha256':fingerprint(observations),'source_sha256':file_hash(folder/'source.json'),
        'context_sha256':conf['context_sha256'],'requests_sha256':fingerprint(requests)}


def evidence_pack(folder):
    conf=check(folder);source,target=_source_pair(folder)
    with measured(folder,ROUND,'fresh_temporal_source_interpretation'):
        observations=sr.all_observations(list(map(Path,conf['evidence_paths'])),source)
        context=(folder/'original-context.txt').read_text(encoding='utf-8')
        requests=source_frame_requests(source,observations,context)
        binding=frame_binding(folder,conf,observations,requests);path=folder/'shared/source-frames.json'
        compiled=_cached(path,binding)
        if compiled is None:
            responses=[]
            with rw.backend('qwen',folder/ROUND/'source-frame-backend',Path(conf['writer_recipe'])) as (endpoint,_):
                for number,request in enumerate(requests,1):
                    responses.append(sr.ask(folder/ROUND/'requests',f'source-frames-{number}','qwen',endpoint,
                        request['instruction'],request['body'],request['schema'],thinking=True))
            compiled=_save(path,compile_source_frames(responses,requests),binding)
        pools=[er._validate_pool(Path(p)) for p in conf['evidence_paths']]
        pack=ec.build_source_pack(source,observations,compiled['frames'],context,pool_versions=[p['pool_version'] for p in pools])
        write(folder/'source-pack.json',pack)
        conf=check(folder);conf['source_pack_sha256']=pack['pack_sha256'];conf['input_hashes'][str(folder/'source-pack.json')]=file_hash(folder/'source-pack.json')
        write(folder/'campaign.json',conf)
    return pack,er.validate_receipt(conf['baseline_review'])


def temporal_table(source,observations):
    """Lossless column encoding of repeated geometry; observation text stays by ID."""
    grouped=te.build_temporal_groups(source,observations)
    intersection_columns=['start_seconds','end_seconds','duration_seconds','start_frame','end_frame']
    columns=['owner_id','sample_rate','crop_start_frame','crop_end_frame','owner_interval_ms','owner_intersection','observation_ids']
    rows=[]
    for group in grouped['groups']:
        crossing=group['owner_intersection']
        intersection=None if crossing is None else [[crossing[k]['numerator'],crossing[k]['denominator']] for k in intersection_columns]
        rows.append([group['owner_id'],group['sample_rate'],group['crop_start_frame'],group['crop_end_frame'],
            [group['owner_interval_ms']['start'],group['owner_interval_ms']['end']],intersection,
            [o['observation_id'] for o in group['observations']]])
    return {'columns':columns,'rows':rows,'scope':'partial_interval','full_owner_competing_readings':False,
        'owner_interval_ms_columns':['start','end'],'intersection_columns':intersection_columns,
        'rational_columns':['numerator','denominator']},grouped['authority']


def focus_context(pack,bound_map,owner_ids):
    focused=recap.focus_context(pack,bound_map,owner_ids)
    focused['temporal_groups'],focused['temporal_authority']=temporal_table(pack['source_rows'],focused['observations'])
    return focused


def prepare_recap_request(pack):
    prepared=cr.prepare_recap_request(pack)
    for frame in prepared['body']['frames'].values():frame.pop('readings')
    prepared['body']['temporal_groups'],prepared['body']['temporal_authority']=temporal_table(pack['source_rows'],pack['observations'])
    claim=prepared['schema']['properties']['claims']['items'];claim['properties'].pop('owner_ids');claim['required'].remove('owner_ids')
    prepared['instruction']+='\n今回はframesに要約とunknownsのみを示します。個別readingの補助解釈は省略しましたが、全ての原文観測はobservations/textsに残っています。全観測を確認してください。owner_idsは出力せず、引用観測IDからプログラムが導出します。'+TEMPORAL_RULE
    return prepared


def bind_recap(raw,pack):
    import jsonschema
    jsonschema.validate(raw,prepare_recap_request(pack)['schema'])
    value=deepcopy(raw);by_id={o['observation_id']:o['owner_id'] for o in pack['observations']}
    for claim in value['claims']:
        cited=set(claim['supporting_ids'])|set(claim['conflicting_ids'])
        for branch in claim['alternatives']:cited.update(branch['observation_ids'])
        require(cited<=set(by_id),'Unknown recap citation')
        claim['owner_ids']=sorted({by_id[i] for i in cited})
    return ec.validate_context_map(value,pack)


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


def source_recap(folder):
    conf = check(folder); pack = read(folder/'source-pack.json'); ec._check_pack(pack)
    prepared = prepare_recap_request(pack)
    binding = {'pack_sha256': pack['pack_sha256'], 'request': fingerprint({k: prepared[k] for k in ('instruction', 'body', 'schema')})}
    path = folder/'shared/source-map.json'; cached = _cached(path, binding)
    if cached is not None:
        focus_context(pack, cached, [1]); return pack, cached
    with measured(folder, 'T1', 'source_evidence_recap'):
        with rw.backend('qwen', folder/'T1/recap-backend', Path(conf['writer_recipe'])) as (endpoint, _):
            raw = sr.ask(folder/'T1/requests', RECAP_REQUEST_KEY, 'qwen', endpoint, prepared['instruction'], prepared['body'], prepared['schema'], thinking=True)
        bound = bind_recap(raw,pack)
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
    with measured(folder, 'T1', 'contextual_material_discrepancies'):
        with rw.backend('qwen', folder/'T1/analysis-backend', Path(conf['writer_recipe'])) as (endpoint, _):
            for number, ids in enumerate(sr.chunk_ids(source), 1):
                focused = focus_context(pack, bound_map, ids)
                body = {**sr.episode_body(source, target, context), 'focus_ids': ids,
                    'source_readings': {str(i): frames[str(i)] for i in ids}, 'focused_context': focused}
                value = sr.ask(folder/'T1/requests', f'discrepancy-{number}', 'qwen', endpoint, DIAGNOSE, body,
                    rw.keyed(ids, rw.obj({'cases': rw.arr(case_schema, 2)})), thinking=True)
                coverage.extend(ids)
                for owner, row in value.items():
                    for number, case in enumerate(row['cases'], 1):
                        case = {**case, 'owner_id': int(owner), 'case_id': f'T1-o{int(owner):03d}-c{number}', 'readings': frames[owner]['readings'],
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
            'focused_context': focus_context(pack, bound_map, ids)}
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
    writer, checker = ('gemma', 'qwen') if round_id == 'T1' else ('qwen', 'gemma')
    proposed = []; decisions = []; accepted = []; rolled = []
    if cases:
        with measured(folder, round_id, 'chinese_only_sparse_proposals'):
            with rw.backend(writer, out/'writer-backend', Path(conf['writer_recipe'])) as (endpoint, _):
                for case in cases:
                    body = {**sr.episode_body(source, target, pack['original_context']), 'case': case,
                        'focused_context': focus_context(pack, bound_map, [case['owner_id']])}
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
                        'focused_context': focus_context(pack, bound_map, [case['owner_id']])}
                    verdict = sr.ask(out/'requests', 'verify-'+case['case_id'], checker, endpoint, VERIFY, body,
                        decision_schema([o['observation_id'] for o in own]), thinking=checker == 'qwen')
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


def score(folder,round_id,evidence,baseline):
    conf=check(folder);candidate=sr.validate_candidate_artifacts(folder,round_id)
    target_hash=er.owner_rows_sha256(rw.rows(folder/round_id/'target.json'))
    for name in conf['prior_score_paths']:
        path=Path(name);prior=read(path)
        if prior.get('status')=='complete' and prior.get('pool_version')==baseline['pool_version'] and prior.get('target_sha256')==target_hash:
            result={'status':'complete','score':min(prior['score'],prior.get('confirmation_score',prior['score'])),
                'inherited_duplicate_target':str(path),'confirmed':False,'pool_version':baseline['pool_version'],
                'target_sha256':target_hash,'local_gate_passed':candidate['local_gate_passed']}
            write(folder/round_id/'score.json',result);return result
    return sr.score(folder,round_id,evidence,baseline)


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
        if not ok:conf.update(status='failed',restoration_error=error)
        elif conf['local_seconds']>=LIMIT:conf['status']='time_budget_exhausted'
        write(folder/'campaign.json',conf)


def execute(folder):
    setup_start=time.monotonic();folder=Path(folder).resolve();conf=init(folder)
    if conf['status']!='prepared':return conf
    freeze(folder);conf=check(folder)
    initial=rw._request_json(rw.ADMIN+'/status',admin=True);previous=late_audio._backend_identity(initial)
    conf.update(status='running',active_round=ROUND,started_utc=sr.now(),original_backend_initial=initial,original_backend_restored=False)
    setup_seconds=time.monotonic()-setup_start;conf['local_seconds']+=setup_seconds
    record=conf['rounds'].setdefault(ROUND,{'local_seconds':0.,'stages':[]});record['status']='reserved';record['local_seconds']+=setup_seconds
    record['stages'].append({'name':'input_and_producer_preparation','seconds':setup_seconds,'error':None,'finished_utc':sr.now()});write(folder/'campaign.json',conf)
    primary_error=None
    try:
        pack,baseline=evidence_pack(folder)
        pack,bound=source_recap(folder);analysis=analyze(folder,pack,bound)
        conf=check(folder);conf.update(map_sha256=bound['map_sha256']);write(folder/'campaign.json',conf)
        candidate=build_candidate(folder,ROUND,pack,bound,analysis)
        if candidate['changed_target_owners']:
            display=folder/ROUND/'display';display.mkdir(exist_ok=True)
            with measured(folder,ROUND,'exact_text_display'):
                with rw.backend('qwen',folder/ROUND/'display-backend',Path(conf['writer_recipe'])):
                    sr._display(rw.rows(folder/ROUND/'source.json'),rw.rows(folder/ROUND/'target.json'),display)
        if check(folder)['local_seconds']>=LIMIT:raise rw.LocalBudgetExceeded('T1 local budget exhausted before scoring')
        result=score(folder,ROUND,list(map(Path,conf['evidence_paths'])),baseline)
        conf=check(folder);conf['rounds'][ROUND].update(status='scored',score=result['score'],confirmed=result['confirmed'])
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
            primary_error.add_note('T1 restoration failed: '+type(exc).__name__)
    return check(folder)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,default=ROOT/'output/quality-temporal-20260915')
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
