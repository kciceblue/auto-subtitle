"""One conditional source-first whole-owner Chinese candidate, entirely local."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import logging
import math
import re
from pathlib import Path
import signal
import time
from copy import deepcopy
from src import whole_owner as wo
from src import whole_owner_requests as wr
from src import temporal_revisit as tr
from src import temporal_release as treplay
from src import context_revisit as cr
from src import context_release as replay
from src import sparse_revisit as sr
from src import revisit_workflow as rw
from src import late_audio
from src import evidence_context as ec
from src import contextual_evidence_review as er
from src.contextual_review_contract import require
from src.config import TranslateConfig
from src.workflow_state import file_hash,fingerprint,write_json
ROOT=Path(__file__).resolve().parents[1]
PREDECESSOR=ROOT/'output/quality-temporal-20260915'
PLAN=ROOT/'design/quality-whole-owner-20260915.md'
VERSION='whole-owner-local-revisit-1'
ROUND='W1'
LIMIT=ROUND_LIMIT=7200
WRITER_PREFLIGHT_VERSION='whole-owner-writer-capacity-preflight-1'
LOG=logging.getLogger(__name__)
read,write,_source_pair,_cached,_save=cr.read,cr.write,cr._source_pair,cr._cached,cr._save


def init(folder):
    folder=Path(folder).resolve()
    if (folder/'campaign.json').exists():return check(folder)
    old=tr.check(PREDECESSOR)
    require(old.get('original_backend_restored') is True and old['status'] not in ('prepared','running')
        and not old.get('eligible_candidate'),'T1 must finish safely without a confirmed four')
    source,target=_source_pair(PREDECESSOR);baseline=er.validate_receipt(old['baseline_review'])
    require(baseline['purpose']=='baseline' and baseline['pool_version']==old['pool_version']
        and baseline['inputs']['source']['sha256']==er.owner_rows_sha256(source)
        and baseline['inputs']['target']['sha256']==er.owner_rows_sha256(target)
        and baseline['inputs']['context']['sha256']==old['context_sha256'],'W1 baseline drift')
    folder.mkdir(parents=True,exist_ok=True);require(not any(folder.iterdir()),'Unregistered W1 output exists')
    pins={str(PLAN):file_hash(PLAN),str(PREDECESSOR/'campaign.json'):file_hash(PREDECESSOR/'campaign.json')}
    for name in ('source.json','target.json','original-context.txt'):
        a=PREDECESSOR/name;b=folder/name;b.write_bytes(a.read_bytes());pins[str(a)]=file_hash(a);pins[str(b)]=file_hash(b)
    for path in [Path(old['writer_recipe']),*map(Path,old['evidence_paths'])]:pins[str(path)]=file_hash(path)
    for name,digest in baseline['receipt_hashes'].items():
        path=Path(baseline['output_dir'])/name;require(file_hash(path)==digest,'Baseline changed');pins[str(path)]=digest
    prior=[*old.get('prior_score_paths',[])]
    if (PREDECESSOR/'T1/score.json').exists():
        prior.append(str(PREDECESSOR/'T1/score.json'))
    for name in prior:pins[name]=file_hash(Path(name))
    conf={k:deepcopy(old[k]) for k in ('media','media_sha256','writer_recipe','writer_recipe_sha256','evidence_paths','pool_version','baseline_review','short_evidence_directory','starting_candidate')}
    conf.update(version=VERSION,status='prepared',created_utc=sr.now(),target=4,maximum_candidates=1,
        maximum_local_seconds=LIMIT,maximum_round_seconds=LIMIT,local_seconds=0.,rounds={},plan=str(PLAN),plan_sha256=file_hash(PLAN),
        source_sha256=file_hash(folder/'source.json'),target_sha256=file_hash(folder/'target.json'),context_sha256=file_hash(folder/'original-context.txt'),
        input_hashes=pins,predecessor=str(PREDECESSOR),prior_score_paths=prior,external_feedback='score_only',
        source_edits_allowed=False,raw_source_fidelity_verified=False,human_reference_used=False,visual_input_used=False,
        new_audio_acquired=False,selected_output_modified=False,writer_sees_old_chinese=False,maximum_writer_groups=33,maximum_owner_checks=66)
    write(folder/'campaign.json',conf);return check(folder)


def check(folder):
    conf=read(Path(folder)/'campaign.json')
    require(conf['version']==VERSION and conf['maximum_candidates']==1 and conf['maximum_local_seconds']==LIMIT
        and conf['maximum_round_seconds']==LIMIT and conf['maximum_writer_groups']==33 and conf['maximum_owner_checks']==66
        and conf['external_feedback']=='score_only' and all(conf[k] is False for k in
        ('source_edits_allowed','raw_source_fidelity_verified','human_reference_used','visual_input_used','new_audio_acquired','selected_output_modified','writer_sees_old_chinese')),'W1 scope drift')
    for name,digest in {**conf['input_hashes'],**conf.get('code_pins',{})}.items():require(file_hash(Path(name))==digest,'Pinned W1 input changed')
    return conf


def freeze(folder):
    conf=check(folder)
    names=['whole_owner_revisit','whole_owner_release','whole_owner','whole_owner_requests','temporal_revisit','temporal_release','temporal_evidence','temporal_source_frames','short_audio','recap_index',
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


def writer_requests(pack,bound):
    ids=[r['index'] for r in pack['source_rows']]
    return [wr.build_writer_request(pack,bound,ids[i:i+2]) for i in range(0,len(ids),2)]



def writer_native_request(request: dict) -> dict:
    """Pure exact Gemma request identity used by the unchanged sr.ask contract."""
    require(isinstance(request,dict) and set(request)=={'instruction','body','schema'}
        and isinstance(request['instruction'],str) and isinstance(request['body'],dict)
        and isinstance(request['schema'],dict),'Invalid whole-owner writer request')
    extra={'model':'gemma4-31b-qat-q4','temperature':.2,'seed':20260915,'top_p':.95,
        'max_tokens':6144,'chat_template_kwargs':{'enable_thinking':False},
        'reasoning_budget_tokens':0,'reasoning_effort':'none',
        'response_format':{'type':'json_object','schema':deepcopy(request['schema'])},
        'top_k':64,'repeat_penalty':1}
    return {'version':sr.VERSION,'family':'gemma','instruction':request['instruction'],
        'body':deepcopy(request['body']),'schema':deepcopy(request['schema']),
        'extra':extra,'thinking':False}


def writer_capacity_config(endpoint: str, directory: Path, key: str, request: dict,
                           remaining: float = 600) -> TranslateConfig:
    """Pure sr.ask-equivalent config; timeout uses the caller's existing budget."""
    require(type(remaining) in (int,float) and math.isfinite(remaining),'Invalid writer capacity budget')
    if remaining<1:
        raise rw.LocalBudgetExceeded('No remaining writer preflight budget')
    native=writer_native_request(request)
    return TranslateConfig(endpoint=endpoint,max_tokens=6144,timeout=max(1,int(min(600,remaining))),retries=0,
        separate_instruction=True,response_guard_floor=65536,stage=key,
        telemetry_path=Path(directory)/'metrics.jsonl',extra_payload=native['extra'])


def preflight_writer_capacity(out: Path, endpoint: str, requests: list[dict]) -> dict:
    """Check every fixed writer prompt before generation; one attempt per prompt."""
    out=Path(out);path=out/'writer-capacity-preflight.json'
    require(not path.exists(),'Writer capacity preflight already exists; no implicit retry')
    started=time.monotonic()
    receipt={'version':WRITER_PREFLIGHT_VERSION,'round':ROUND,'family':'gemma','status':'running',
        'expected_requests':33,'requests_sha256':fingerprint(requests),'coverage_owner_ids':[],
        'started_utc':sr.now(),'finished_utc':None,'elapsed_seconds':0.,'records':[],'error_type':None}
    def save():
        receipt['elapsed_seconds']=time.monotonic()-started
        receipt['preflight_sha256']=fingerprint({key:value for key,value in receipt.items() if key!='preflight_sha256'})
        write(path,receipt)
    save()
    try:
        require(isinstance(requests,list) and len(requests)==33,'Writer preflight requires exactly 33 requests')
        coverage=[]
        for number,request in enumerate(requests,1):
            writer_native_request(request)
            owner_ids=request['body'].get('focus_owner_ids')
            require(isinstance(owner_ids,list) and owner_ids==[number*2-1,number*2]
                and all(type(owner) is int for owner in owner_ids),'Writer preflight owner pairs changed')
            coverage.extend(owner_ids)
        receipt['coverage_owner_ids']=coverage;save()
        for number,request in enumerate(requests,1):
            row_started=time.monotonic();key=f'writer-group-{number}'
            record={'request_key':key,'owner_ids':deepcopy(request['body']['focus_owner_ids']),
                'request_sha256':fingerprint(request),'instruction_sha256':fingerprint(request['instruction']),
                'body_sha256':fingerprint(request['body']),'schema_sha256':fingerprint(request['schema']),
                'native_request_sha256':fingerprint(writer_native_request(request)),
                'started_utc':sr.now(),'finished_utc':None,'elapsed_seconds':0.,'capacity':None,'error_type':None}
            receipt['records'].append(record);save()
            try:
                remaining=600 if sr._DEADLINE is None else sr._DEADLINE-time.monotonic()
                cfg=writer_capacity_config(endpoint,out/'requests',key,request,remaining)
                capacity=rw.native_capacity(json.dumps(request['body'],ensure_ascii=False),request['instruction'],cfg,'gemma')
                record['capacity']=capacity
                require(isinstance(capacity,dict) and capacity.get('fits') is True
                    and all(type(capacity.get(field)) is int for field in ('prompt_tokens','reserved_output','context'))
                    and capacity['prompt_tokens']>0 and capacity['reserved_output']==6144 and capacity['context']==32768
                    and capacity['prompt_tokens']+6144+64<=capacity['context']
                    and isinstance(capacity.get('rendered_prompt_sha256'),str)
                    and re.fullmatch(r'[0-9a-f]{64}',capacity['rendered_prompt_sha256']) is not None,
                    'Writer preflight native capacity identity or arithmetic failed')
            except BaseException as exc:
                record['error_type']=type(exc).__name__
                raise
            finally:
                record['finished_utc']=sr.now();record['elapsed_seconds']=time.monotonic()-row_started;save()
        receipt['status']='complete'
    except BaseException as exc:
        receipt['status']='failed';receipt['error_type']=type(exc).__name__
        raise
    finally:
        receipt['finished_utc']=sr.now();save()
    return deepcopy(receipt)


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
    with measured(folder,ROUND,'source_only_whole_owner_generation'):
        with rw.backend('gemma',out/'writer-backend',Path(conf['writer_recipe'])) as (endpoint,_):
            requests=writer_requests(pack,bound)
            preflight_writer_capacity(out,endpoint,requests)
            for number,request in enumerate(requests,1):
                raw=sr.ask(out/'requests',f'writer-group-{number}','gemma',endpoint,request['instruction'],request['body'],request['schema'],thinking=False)
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
        'writer':'gemma','checker':'qwen','writer_sees_old_chinese':False,'evidence_paths':conf['evidence_paths'],
        'pack_sha256':pack['pack_sha256'],'map_sha256':bound['map_sha256'],'raw_source_fidelity_verified':False,
        'external_feedback_used':False,'human_reference_used':False,'source_edits_allowed':False}
    write(out/'candidate.json',result);return result


def validate_candidate_artifacts(folder):
    conf=check(folder);source,target=_source_pair(folder);out=folder/ROUND;candidate=read(out/'candidate.json')
    require(candidate['version']==VERSION and candidate['round']==ROUND and candidate['source_edits_allowed'] is False
        and candidate['writer_sees_old_chinese'] is False,'W1 candidate scope changed')
    pack=read(folder/'source-pack.json');ec._check_pack(pack)
    require(candidate['pack_sha256']==conf['source_pack_sha256']==pack['pack_sha256']
        and candidate['map_sha256']==conf['map_sha256'],'W1 candidate evidence changed')
    result_source,result_target,ledger=wo.apply_owner_transactions(source,target,candidate['accepted_transactions'],pack)
    require(ledger==candidate['transaction_ledger'] and result_source==source
        and rw.rows(out/'source.json')==source and rw.rows(out/'target.json')==result_target
        and file_hash(out/'source.json')==candidate['source_sha256'] and file_hash(out/'target.json')==candidate['target_sha256'], 'W1 candidate artifact changed')
    require(candidate['changed_target_owners']==[a.index for a,b in zip(result_target,target) if a.text!=b.text],'W1 change ledger drift')
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
        and bundle['inputs']['source']['sha256']==er.owner_rows_sha256(source),'W1 score bundle drift')
    primary=sr.review_once(manifest,out/'evaluation/initial/primary',purpose='candidate',baseline_review=baseline['output_dir'])
    result={'status':'primary_complete','score':primary['score'],'primary':primary['output_dir'],'pool_version':primary['pool_version'],
        'target_sha256':target_hash,'source_sha256':primary['source_sha256'],'confirmed':False,'local_gate_passed':candidate['local_gate_passed']}
    write(result_path,result)
    if primary['score']>=4 and candidate['local_gate_passed']:
        confirmation=sr.review_once(manifest,out/'evaluation/initial/confirmation',purpose='confirmation',primary_review=primary['output_dir'])
        result.update(confirmation_score=confirmation['score'],confirmation=confirmation['output_dir'],confirmed=confirmation['score']>=4)
    result['status']='complete';write(result_path,result);LOG.info('W1 independent score=%s confirmed=%s',result['score'],result['confirmed']);return result


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
    freeze(folder);conf=check(folder);initial=rw._request_json(rw.ADMIN+'/status',admin=True);previous=late_audio._backend_identity(initial)
    conf.update(status='running',active_round=ROUND,started_utc=sr.now(),original_backend_initial=initial,original_backend_restored=False)
    seconds=time.monotonic()-setup_start;conf['local_seconds']+=seconds
    conf['rounds'][ROUND]={'status':'reserved','local_seconds':seconds,'stages':[{'name':'input_and_producer_preparation','seconds':seconds,'error':None,'finished_utc':sr.now()}]}
    write(folder/'campaign.json',conf);primary_error=None
    try:
        pack,bound=inherited_source(folder);candidate=build_candidate(folder,pack,bound);conf=check(folder)
        if candidate['changed_target_owners']:
            directory=folder/ROUND/'display';directory.mkdir(exist_ok=True)
            with measured(folder,ROUND,'exact_text_display'):
                with rw.backend('qwen',folder/ROUND/'display-backend',Path(conf['writer_recipe'])):
                    sr._display(rw.rows(folder/ROUND/'source.json'),rw.rows(folder/ROUND/'target.json'),directory)
        if check(folder)['local_seconds']>=LIMIT:raise rw.LocalBudgetExceeded('W1 budget exhausted before scoring')
        result=score(folder,er.validate_receipt(conf['baseline_review']));conf=check(folder)
        conf['rounds'][ROUND].update(status='scored',score=result['score'],confirmed=result['confirmed'])
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
            primary_error.add_note('W1 restoration failed: '+type(exc).__name__)
    return check(folder)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,default=ROOT/'output/quality-whole-owner-20260915')
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
