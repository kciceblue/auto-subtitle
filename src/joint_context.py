"""Pure complete-candidate-context diagnostic requests; no inference or I/O.

All 66 original whole-owner proposals, including no-ops, are required. Their
canonical bindings are checked against the supplied immutable bases. Native
writer/control provenance, model settings, budgets and release decisions remain
the caller's responsibility. No acceptance criterion is changed here.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Sequence

from src import evidence_context as ec
from src import whole_owner as whole
from src import whole_owner_requests as requests
from src.translate import SrtBlock

VERSION = 'joint-context-diagnostic-1'
OWNER_COUNT = 66


def _same_json(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int or int/float coercion."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return (all(type(key) is str for key in left)
                and all(type(key) is str for key in right)
                and left.keys() == right.keys()
                and all(_same_json(left[key], right[key]) for key in left))
    if type(left) is list:
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right))
    if type(left) is float:
        return math.isfinite(left) and math.isfinite(right) and left == right
    return type(left) in (str, int, bool, type(None)) and left == right


def assemble_draft(source: list[SrtBlock], target: list[SrtBlock],
                   proposals: Sequence[dict], pack: dict) -> list[SrtBlock]:
    """Replay all original proposals, never an accepted/rejected subset.

    The result contains canonical transaction text, including established
    whitespace normalization and exact original whitespace for no-ops. This
    mechanical assembly does not certify semantic quality or native authorship.
    """
    ec._require(isinstance(source, list) and isinstance(target, list)
                and len(source) == len(target) == OWNER_COUNT,
                'Joint context requires the complete 66-owner source and target')
    ec._require(isinstance(proposals, (list, tuple)) and len(proposals) == OWNER_COUNT,
                'Joint context requires all 66 original proposals, including no-ops')
    owners = []
    for proposal in proposals:
        ec._require(isinstance(proposal, dict) and type(proposal.get('owner_id')) is int,
                    'Invalid original proposal owner')
        owners.append(proposal['owner_id'])
    ec._require(len(set(owners)) == OWNER_COUNT and set(owners) == set(range(1, OWNER_COUNT + 1)),
                'Original proposals have duplicate, missing or foreign owners')
    replayed_source, draft, _ = whole.apply_owner_transactions(source, target, list(proposals), pack)
    ec._require(replayed_source == source, 'Draft assembly changed the source')
    return draft


def _episode_map(value: Any) -> dict[str, str]:
    ec._keys(value, {str(owner) for owner in range(1, OWNER_COUNT + 1)})
    for text in value.values():
        ec._text(text)
    return value


def assert_context_only_change(control: dict, treatment: dict,
                               expected_draft: Sequence[SrtBlock]) -> dict:
    """Assert the sole request difference is the complete Chinese episode map.

    The map also repeats the focus owner's text. Replacing it therefore tests
    complete-candidate-context sensitivity, not isolated neighbor causality.
    The control's native identity and the draft's provenance must be checked by
    the caller; this helper checks the exact supplied request pair only.
    """
    for request in (control, treatment):
        ec._keys(request, {'instruction', 'body', 'schema'})
        ec._require(type(request['body']) is dict and 'episode_chinese' in request['body'],
                    'Verifier episode context missing')
    rows = requests._rows(expected_draft)
    ec._require(len(rows) == OWNER_COUNT, 'Expected draft is incomplete')
    expected = requests._rowmap(rows)
    before = _episode_map(control['body']['episode_chinese'])
    after = _episode_map(treatment['body']['episode_chinese'])
    ec._require(_same_json(after, expected), 'Treatment context differs from complete draft')
    restored = deepcopy(treatment)
    restored['body']['episode_chinese'] = deepcopy(before)
    ec._require(_same_json(control, restored), 'Verifier request changed outside body.episode_chinese')
    owner = control['body'].get('owner_id')
    ec._require(type(owner) is int and 1 <= owner <= OWNER_COUNT, 'Verifier focus owner invalid')
    key = str(owner)
    ec._require(before[key] == control['body'].get('before_chinese')
                and after[key] == control['body'].get('after_chinese')
                and before[key] != after[key],
                'Context must repeat the exact non-noop focus before/after text')
    changed = [owner for owner in range(1, OWNER_COUNT + 1) if before[str(owner)] != after[str(owner)]]
    return {'owner_count': OWNER_COUNT, 'changed_owner_count': len(changed),
            'changed_owner_ids': changed, 'focus_row_changed': True}


def build_treatment_request(pack: dict, bound_map: dict, source: list[SrtBlock],
                            target: list[SrtBlock], proposals: Sequence[dict], owner_id: int) -> dict:
    """Build the unchanged owner verifier with only its episode context replaced."""
    ec._require(type(owner_id) is int and 1 <= owner_id <= OWNER_COUNT,
                'Unknown diagnostic focus owner')
    draft = assemble_draft(source, target, proposals, pack)
    transaction = next(item for item in proposals if item['owner_id'] == owner_id)
    control = requests.build_verifier_request(pack, bound_map, source, target, transaction)
    treatment = deepcopy(control)
    treatment['body']['episode_chinese'] = {str(row.index): row.text for row in draft}
    assert_context_only_change(control, treatment, draft)
    return treatment
