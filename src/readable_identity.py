"""Native provenance joins for the unexecuted readable-evidence prototype.

This module does not recognize, interpret or edit speech. The pure join accepts
validated native bundles; the optional loader revalidates their saved receipts.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from src import evidence_context as ec
from src.contextual_review_contract import require, strict_json
from src.workflow_state import fingerprint, file_hash
from src.late_audio import _hash as native_hash

VERSION='readable-native-identity-1'
GEOMETRY=('owner_id','crop_start_frame','crop_end_frame','sample_rate')
KINDS={'raw','raw_early','raw_late','raw_short','bandit_dialogue','bandit_residual'}
ENGINES={'zipformer-ja','qwen3-asr-1.7b'}


def join_identity_map(pack, bundles):
    """Join exact source observations to already-validated native pool bundles.

    Each bundle is {pool, plan, preparation}. The final acquisition plan owns
    actual crops; preparation crops may precede a VAD-driven replan.
    """
    ec._check_pack(pack)
    require(isinstance(bundles,list) and len(bundles)==len(pack['pool_versions']), 'Native pool coverage changed')
    by_id={};versions=[]
    for bundle in bundles:
        require(set(bundle)=={'pool','plan','preparation'}, 'Invalid native identity bundle')
        pool,plan,prep=(bundle[k] for k in ('pool','plan','preparation'))
        versions.append(pool['pool_version'])
        crops={c['view_id']:c for c in plan['crops']}
        require(len(crops)==len(plan['crops']), 'Duplicate final native view')
        scheduled={o['observation_id']:o for o in plan['observations']}
        require(len(scheduled)==len(plan['observations'])==len(pool['observations'])
            and set(scheduled)=={o['observation_id'] for o in pool['observations']}, 'Native schedule coverage changed')
        for obs in pool['observations']:
            identifier=obs['observation_id'];base=scheduled[identifier]
            require(all(obs.get(k)==v for k,v in base.items()), 'Native planned observation changed')
            require(identifier not in by_id, 'Duplicate observation across native pools')
            require(obs['status'] in {'ok','empty','unavailable'} and isinstance(obs['text'],str), 'Invalid native status')
            engine=obs['engine'];provenance=obs['provenance']
            require(engine in ENGINES and engine in prep['models'], 'Unknown native observer')
            require(provenance['view_id'] in crops, 'Unknown final native view')
            crop=crops[provenance['view_id']]
            require(all(obs[k]==crop[k] for k in GEOMETRY)
                and obs['duplicate_geometry']==crop['duplicate_geometry'], 'Final native crop differs')
            kind=crop['kind'];require(kind in KINDS,'Unknown final native crop kind')
            expected_transform='raw' if kind.startswith('raw') else kind
            require(provenance['transform']==expected_transform, 'Native transform differs from final crop')
            core=(crop.get('core_start_frame'),crop.get('core_end_frame'))
            if kind=='raw_short':
                require(all(type(x) is int for x in core)
                    and crop['crop_start_frame']<=core[0]<core[1]<=crop['crop_end_frame'], 'Invalid short core scope')
            else:require(core==(None,None),'Unexpected core for non-short view')
            entry={k:obs[k] for k in GEOMETRY}
            entry.update(family=engine,model_identity_sha256=native_hash(prep['models'][engine]),
                view_id=provenance['view_id'],view_kind=kind,transform=expected_transform,
                core_start_frame=core[0],core_end_frame=core[1],duplicate_geometry=obs['duplicate_geometry'],
                provenance_sha256=fingerprint({'pool_version':pool['pool_version'],
                    'observation_id':identifier,'provenance':provenance,'crop':crop,
                    'model':prep['models'][engine]}))
            by_id[identifier]=(obs,entry)
    require(len(set(versions))==len(versions) and set(versions)==set(pack['pool_versions']), 'Native pool identity changed')
    result={};used=set()
    for obs in pack['observations']:
        identifier=obs['observation_id']
        if obs.get('origin')=='immutable_first_asr':
            require(identifier not in by_id and obs.get('owner_exact') is True, 'Original source origin collision')
            result[identifier]={**{k:obs[k] for k in GEOMETRY},'family':None,'model_identity_sha256':None,
                'view_id':None,'view_kind':'original_transcript','transform':None,
                'core_start_frame':None,'core_end_frame':None,'duplicate_geometry':False,
                'provenance_sha256':pack['source_sha256']}
        else:
            require(identifier in by_id, 'Missing native observation provenance')
            raw,identity=by_id[identifier]
            require(raw['status']=='ok' and raw['text'] and obs['text']==raw['text']
                and all(obs[k]==raw[k] for k in GEOMETRY), 'Pack observation differs from native transcript or scope')
            result[identifier]=deepcopy(identity);used.add(identifier)
    require(used=={key for key,(obs,_) in by_id.items() if obs['status']=='ok' and obs['text']},
        'Usable native observations were omitted from the source pack')
    return result


def load_identity_map(pack, evidence_paths, track):
    """Read and validate local native receipts. No model calls or output writes."""
    from src import late_audio
    bundles=[]
    def pinned(pin):
        path=Path(pin['path']);value=track(path)
        require(file_hash(path)==pin['sha256'], 'Pinned native identity artifact changed')
        return value
    for value in evidence_paths:
        path=Path(value);pool=track(path)
        require(late_audio.validate_evidence(pool)==pool, 'Native pool changed during validation')
        plan=pinned({'path':pool['plan_path'],'sha256':pool['plan_sha256']})
        prep=pinned(plan['preparation'])
        bundles.append({'pool':pool,'plan':plan,'preparation':prep})
    return join_identity_map(pack,bundles)
