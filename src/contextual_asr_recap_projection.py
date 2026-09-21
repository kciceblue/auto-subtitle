"""Pure whole-hypothesis projection of one independently authenticated raw recap.

The caller must authenticate the native receipt and original preparation before
calling this module. Hashes bind inputs; they do not establish native execution
or acoustic truth. No source, citation, hypothesis, alternative or term is
rewritten. Only complete hypotheses and, when necessary, the entire unknowns
list are omitted. The original native raw remains a separate immutable artifact.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib

import jsonschema

from src import contextual_asr_requests as contract
from src import evidence_context as ec
from src.contextual_review_contract import strict_json

VERSION = 'contextual-asr-recap-projection-1'
_BOUND_FAILURE = 'Recap hint exceeds fixed bound; truncation is forbidden'


def compile_recap(preparation: dict, raw: str, *, native_sha256: str) -> dict:
    """Project saved raw JSON deterministically, retaining whole claims in order.

    Full native schema errors are fatal before filtering. A hypothesis that
    fails the existing singleton format/citation check is discarded whole. A
    valid singleton is kept only when it fits after previously retained claims;
    a skipped claim never prevents a later smaller claim from being considered.
    Original unknowns are retained all together if they fit, otherwise none.
    No surviving hypothesis is a terminal failure, not an empty usable hint.
    """
    contract._check_preparation(preparation)
    contract._hash(native_sha256)
    ec._require(type(raw) is str, 'Recap projection requires original raw text')
    ec._text(raw)
    response = strict_json(raw)
    try:
        jsonschema.validate(response, preparation['recap_request']['schema'])
    except jsonschema.ValidationError:
        raise ValueError('Original recap fails full native schema') from None
    # Unknowns may be removed for the aggregate bound only, never to repair
    # malformed prose that the existing formatter would reject.
    contract._texts(response['unknowns_ja'], 4, 80)
    retained = []
    rows = []
    for ordinal, hypothesis in enumerate(response['hypotheses'], 1):
        singleton = {'hypotheses': [hypothesis], 'unknowns_ja': []}
        try:
            contract.format_hint(preparation, singleton)
        except ValueError as error:
            category = ('dropped_singleton_bound' if str(error) == _BOUND_FAILURE
                        else 'dropped_singleton_validation')
        else:
            candidate = {'hypotheses': [*retained, hypothesis], 'unknowns_ja': []}
            try:
                contract.format_hint(preparation, candidate)
            except ValueError as error:
                if str(error) != _BOUND_FAILURE:
                    raise
                category = 'dropped_accumulated_bound'
            else:
                retained.append(deepcopy(hypothesis))
                category = 'retained'
        rows.append({'ordinal': ordinal, 'category': category})
    ec._require(bool(retained), 'No valid whole recap hypothesis survives projection')
    derived = {'hypotheses': retained, 'unknowns_ja': deepcopy(response['unknowns_ja'])}
    unknowns_category = 'retained'
    try:
        hint = contract.format_hint(preparation, derived)
    except ValueError as error:
        if str(error) != _BOUND_FAILURE:
            raise
        derived['unknowns_ja'] = []
        unknowns_category = 'dropped_bound'
        hint = contract.format_hint(preparation, derived)
    value = {'version': VERSION,
             'preparation_sha256': preparation['preparation_sha256'],
             'native_sha256': native_sha256,
             'raw_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
             'response': derived,
             'audit': {'hypotheses': rows,
                       'unknowns': {'category': unknowns_category, 'count': len(response['unknowns_ja'])},
                       'hint_characters': len(hint), 'hint_utf8_bytes': len(hint.encode('utf-8'))}}
    value['projection_sha256'] = ec._hash(value)
    return value


def validate_projection(projection: dict, preparation: dict, raw: str, *, native_sha256: str) -> None:
    """Rebuild from independent native raw/preparation, never a retained backup."""
    expected = compile_recap(preparation, raw, native_sha256=native_sha256)
    ec._require(ec._hash(projection) == ec._hash(expected),
                'Recap projection differs from independently supplied native raw')
