"""Derive a redundant owner index from existing, unchanged local citations."""
from copy import deepcopy
from src import evidence_context as ec
from src.contextual_review_contract import require

VERSION='citation-derived-recap-owner-index-1'

def derive(raw:dict,pack:dict)->tuple[dict,list[dict]]:
    ec._check_pack(pack)
    import jsonschema
    jsonschema.validate(raw,ec.context_schema(pack))
    observations={o['observation_id']:o['owner_id'] for o in pack['observations']}
    value=deepcopy(raw);audit=[]
    for claim in value['claims']:
        declared=claim['owner_ids']
        require(len(declared)==len(set(declared)),'Duplicate declared recap owner')
        identifiers=set(claim['supporting_ids'])|set(claim['conflicting_ids'])
        for alternative in claim['alternatives']:identifiers.update(alternative['observation_ids'])
        require(identifiers<=set(observations),'Unknown cited observation')
        derived=sorted({observations[i] for i in identifiers})
        if declared!=derived:
            audit.append({'claim_id':claim['claim_id'],'declared_owner_ids':declared,'derived_owner_ids':derived})
        claim['owner_ids']=derived
    return ec.validate_context_map(value,pack),audit
