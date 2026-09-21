"""Pure, unused readable L1 evidence prototype; no I/O, inference or registration.

Caller-supplied identity metadata must already be authenticated against native
receipts, model identities and finalized acquisition plans. This module checks
closed schemas and exact joins; hashes alone cannot authenticate a caller.
Model-visible literals live in ``body``. ``audit`` holds only inverse handles,
operational metadata and hashes, never fallback copies of those literals.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import re
from typing import Any

from src import evidence_context as ec
from src import temporal_evidence as temporal
from src import temporal_revisit as tr
from src.workflow_state import fingerprint

VERSION = 'readable-evidence-prototype-1'
_IDENTITY_KEYS = {'owner_id', 'crop_start_frame', 'crop_end_frame', 'sample_rate', 'family',
    'model_identity_sha256', 'view_id', 'view_kind', 'transform', 'core_start_frame',
    'core_end_frame', 'duplicate_geometry', 'provenance_sha256'}
_GEOMETRY_KEYS = ('owner_id', 'crop_start_frame', 'crop_end_frame', 'sample_rate')
_KINDS = {'original_transcript', 'raw', 'raw_early', 'raw_late', 'raw_short',
          'bandit_dialogue', 'bandit_residual'}
_SOURCE_META = {'version', 'pool_versions', 'source_sha256', 'observations_sha256',
                'frames_sha256', 'context_sha256', 'pack_sha256', 'text_encoding'}
_MAP_META = {'version', 'pool_versions', 'pack_sha256', 'map_sha256'}
_SOURCE_KEYS = _SOURCE_META | {'source_rows', 'observations', 'frames', 'original_context',
                             'authority', 'texts', 'temporal_groups', 'temporal_authority'}
_BODY_KEYS = {'source_rows', 'original_context', 'observations', 'observers', 'views', 'frames',
              'source_map', 'authority', 'temporal_authority', 'scope_rules', 'additional_literals'}
_AUDIT_KEYS = {'version', 'source_metadata', 'map_metadata', 'owner_text_ids', 'observation_links',
              'observer_models', 'view_original_ids', 'text_order', 'bindings', 'audit_sha256'}
_BINDINGS = {'canonical_body_sha256', 'identity_map_sha256', 'body_sha256'}
_VIEW_KEYS = {'id', 'owner_id', 'kind', 'transform', 'sample_rate', 'crop_seconds',
              'core_seconds', 'owner_intersection_seconds', 'duplicate_geometry'}
_RULES = {'interval_convention': 'absolute_half_open', 'groups_are_partial_intervals': True,
          'same_family_crops_and_views_are_independent_votes': False,
          'different_model_instances_establish_independence': False,
          'core_window_limits_transcript_word_scope': False,
          'owner_boundary_association_proves_word_ownership': False,
          'original_transcript_recognizer_is_identified': False,
          'additional_literals_are_new_observation_votes': False}


def _same(left: Any, right: Any) -> bool:
    return fingerprint(left) == fingerprint(right)


def _keys(value: Any, required: set[str], optional: set[str] = frozenset()) -> None:
    ec._keys(value, required, optional)


def _hash(value: Any) -> None:
    ec._require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
                'Invalid provenance hash')


def _time(value: Fraction) -> str:
    """Canonical exact seconds: terminating decimal when possible, else fraction."""
    denominator = value.denominator; twos = fives = 0
    while denominator % 2 == 0:
        twos += 1; denominator //= 2
    while denominator % 5 == 0:
        fives += 1; denominator //= 5
    if denominator != 1:
        return f'{value.numerator}/{value.denominator}'
    places = max(twos, fives)
    if not places:
        return str(value.numerator)
    scaled = value.numerator * 10**places // value.denominator
    digits = str(abs(scaled)).zfill(places + 1)
    return ('-' if scaled < 0 else '') + digits[:-places] + '.' + digits[-places:]


def _fraction(value: Any) -> Fraction:
    ec._require(isinstance(value, str) and bool(value), 'Exact time string missing')
    try:
        number = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError('Invalid exact time') from exc
    ec._require(number >= 0 and _time(number) == value, 'Time is rounded or noncanonical')
    return number


def _frames(value: Any, rate: int) -> int:
    frame = _fraction(value) * rate
    ec._require(frame.denominator == 1, 'Time does not lie on its exact sample grid')
    return frame.numerator


def _intersection(start: int, end: int, rate: int, row: dict) -> list[str] | None:
    a, b = map(temporal._milliseconds, row['ts_line'].split(' --> '))
    left, right = max(Fraction(start, rate), Fraction(a, 1000)), min(Fraction(end, rate), Fraction(b, 1000))
    return [_time(left), _time(right)] if left < right else None


def _identities(observations: list[dict], identities: dict) -> None:
    ec._require(isinstance(identities, dict) and set(identities) == {item['observation_id'] for item in observations},
                'Identity map must cover every observation exactly')
    views = {}
    for observation in observations:
        item = identities[observation['observation_id']]; _keys(item, _IDENTITY_KEYS)
        ec._require(all(type(item[key]) is int and item[key] == observation[key] for key in _GEOMETRY_KEYS),
                    'Identity owner or sample geometry differs')
        ec._require(item['view_kind'] in _KINDS and type(item['duplicate_geometry']) is bool, 'Invalid view identity')
        _hash(item['provenance_sha256'])
        original = observation.get('origin') == 'immutable_first_asr'
        if original:
            ec._require(item['view_kind'] == 'original_transcript' and item['duplicate_geometry'] is False
                and all(item[key] is None for key in ('family', 'model_identity_sha256', 'view_id', 'transform',
                                                      'core_start_frame', 'core_end_frame')),
                'Original transcript recognizer must remain unattributed')
        else:
            ec._require(item['family'] in {'zipformer-ja', 'qwen3-asr-1.7b'}
                        and item['view_kind'] != 'original_transcript', 'Unknown acoustic recognizer')
            _hash(item['model_identity_sha256']); ec._text(item['view_id'])
            expected_transform = 'raw' if item['view_kind'].startswith('raw') else item['view_kind']
            ec._require(item['transform'] == expected_transform, 'View kind and audio transform differ')
            if item['view_kind'] == 'raw_short':
                a, b = item['core_start_frame'], item['core_end_frame']
                ec._require(type(a) is int and type(b) is int
                            and item['crop_start_frame'] <= a < b <= item['crop_end_frame'], 'Invalid short core')
            else:
                ec._require(item['core_start_frame'] is None and item['core_end_frame'] is None, 'Unexpected view core')
            view = {key: item[key] for key in _GEOMETRY_KEYS + ('view_kind', 'transform', 'core_start_frame',
                                                              'core_end_frame', 'duplicate_geometry')}
            if item['view_id'] in views:
                ec._require(_same(views[item['view_id']], view), 'Shared view ID has conflicting geometry or kind')
            views[item['view_id']] = view


def _map_refs(value: dict, identifiers: dict[str, str]) -> dict:
    """Relabel only schema-defined reference arrays, never text substrings."""
    result = deepcopy(value)
    def refs(values):
        ec._require(isinstance(values, list) and all(type(item) is str and item in identifiers for item in values),
                    'Unknown recap observation reference')
        return [identifiers[item] for item in values]
    for claim in result['claims']:
        _keys(claim, {'claim_id', 'claim_ja', 'owner_ids', 'supporting_ids', 'conflicting_ids', 'uncertain', 'alternatives'})
        claim['supporting_ids'] = refs(claim['supporting_ids']); claim['conflicting_ids'] = refs(claim['conflicting_ids'])
        for alternative in claim['alternatives']:
            _keys(alternative, {'interpretation_ja', 'observation_ids'})
            alternative['observation_ids'] = refs(alternative['observation_ids'])
    return result


def _map_shape(value: dict, owners: set[int], observations: dict) -> None:
    _keys(value, {'coverage_owner_ids', 'claims', 'authority'})
    ec._require(_same(value['authority'], ec.AUTHORITY), 'Recap authority changed')
    covered = ec._ids(value['coverage_owner_ids'], owners, integer=True, minimum=len(owners))
    ec._require(set(covered) == owners, 'Recap owner coverage changed')
    ec._require(isinstance(value['claims'], list) and len(value['claims']) <= ec.MAX_CLAIMS, 'Invalid recap claims')
    seen = set()
    for claim in value['claims']:
        _keys(claim, {'claim_id', 'claim_ja', 'owner_ids', 'supporting_ids', 'conflicting_ids', 'uncertain', 'alternatives'})
        identifier = claim['claim_id']
        ec._require(isinstance(identifier, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', identifier)
                    and identifier not in seen, 'Invalid recap claim identity')
        seen.add(identifier); ec._text(claim['claim_ja'], maximum=600)
        declared = ec._ids(claim['owner_ids'], owners, integer=True, minimum=1)
        supporting = ec._ids(claim['supporting_ids'], set(observations), minimum=1)
        conflicting = ec._ids(claim['conflicting_ids'], set(observations))
        ec._require(not set(supporting) & set(conflicting) and type(claim['uncertain']) is bool,
                    'Invalid recap support or uncertainty')
        ec._require(isinstance(claim['alternatives'], list) and len(claim['alternatives']) <= ec.MAX_ALTERNATIVES,
                    'Invalid recap alternatives')
        cited = set(supporting) | set(conflicting)
        for alternative in claim['alternatives']:
            _keys(alternative, {'interpretation_ja', 'observation_ids'})
            ec._text(alternative['interpretation_ja'], maximum=400)
            cited.update(ec._ids(alternative['observation_ids'], set(observations), minimum=1))
        ec._require(set(declared) == {observations[key]['owner_id'] for key in cited}, 'Recap citation owner changed')


def _canonical_input(pack: dict, bound: dict) -> dict:
    ec._check_pack(pack)
    ec._require(isinstance(bound, dict) and ec.RAW_MAP_KEYS <= set(bound), 'Bound source map missing')
    rebuilt = ec.validate_context_map({key: bound[key] for key in ec.RAW_MAP_KEYS}, pack)
    ec._require(_same(rebuilt, bound), 'Bound source map changed')
    evidence = tr.prepare_recap_request(pack)['body']; _keys(evidence, _SOURCE_KEYS)
    return {'source_evidence': evidence, 'source_map': deepcopy(bound),
            'focus_owner_ids': [row['index'] for row in pack['source_rows']]}


def build_projection(pack: dict, bound: dict, identity_map: dict) -> dict:
    """Return a readable body and its operational audit; leave all inputs untouched."""
    canonical = _canonical_input(pack, bound); evidence = canonical['source_evidence']
    _identities(pack['observations'], identity_map)
    ordered = sorted(pack['observations'], key=lambda row: (Fraction(row['crop_start_frame'], row['sample_rate']),
        Fraction(row['crop_end_frame'], row['sample_rate']), row['owner_id'], row['observation_id']))
    aliases = {row['observation_id']: f'E{number:04d}' for number, row in enumerate(ordered, 1)}
    observers = []; observer_models = {}; observer_aliases = {}; views = []; view_original_ids = {}; view_aliases = {}
    rows = {row['index']: row for row in pack['source_rows']}; observations = []
    for observation in ordered:
        item = identity_map[observation['observation_id']]
        observer_key = (item['family'], item['model_identity_sha256'])
        if observer_key not in observer_aliases:
            observer = f'R{len(observers)+1:03d}'; observer_aliases[observer_key] = observer
            observers.append({'id': observer, 'family': item['family']}); observer_models[observer] = item['model_identity_sha256']
        observer = observer_aliases[observer_key]
        view_key = ('original', item['owner_id']) if item['view_id'] is None else ('native', item['view_id'])
        if view_key not in view_aliases:
            view = f'V{len(views)+1:04d}'; view_aliases[view_key] = view; view_original_ids[view] = item['view_id']
            start, end, rate = (item[key] for key in ('crop_start_frame', 'crop_end_frame', 'sample_rate'))
            core = None if item['core_start_frame'] is None else [_time(Fraction(item[key], rate))
                                                                for key in ('core_start_frame', 'core_end_frame')]
            views.append({'id': view, 'owner_id': item['owner_id'], 'kind': item['view_kind'], 'transform': item['transform'],
                'sample_rate': rate, 'crop_seconds': [_time(Fraction(start, rate)), _time(Fraction(end, rate))],
                'core_seconds': core, 'owner_intersection_seconds': _intersection(start, end, rate, rows[item['owner_id']]),
                'duplicate_geometry': item['duplicate_geometry']})
        observations.append({'id': aliases[observation['observation_id']], 'owner_id': observation['owner_id'],
            'text': observation['text'], 'observer': observer, 'view': view_aliases[view_key],
            **{key: observation[key] for key in ('origin', 'owner_exact') if key in observation}})
    used_text_ids = {row['text_id'] for row in evidence['source_rows'] + evidence['observations']}
    body = {'source_rows': deepcopy(pack['source_rows']), 'original_context': pack['original_context'],
        'observations': observations, 'observers': observers, 'views': views, 'frames': deepcopy(evidence['frames']),
        'source_map': _map_refs({key: value for key, value in bound.items() if key not in _MAP_META}, aliases),
        'authority': deepcopy(evidence['authority']), 'temporal_authority': deepcopy(evidence['temporal_authority']),
        'scope_rules': deepcopy(_RULES),
        'additional_literals': {key: value for key, value in evidence['texts'].items() if key not in used_text_ids}}
    audit = {'version': VERSION, 'source_metadata': {key: deepcopy(evidence[key]) for key in sorted(_SOURCE_META)},
        'map_metadata': {key: deepcopy(bound[key]) for key in sorted(_MAP_META)},
        'owner_text_ids': [{'index': row['index'], 'text_id': row['text_id']} for row in evidence['source_rows']],
        'observation_links': [{'id': aliases[row['observation_id']], 'original_id': row['observation_id'],
            'text_id': row['text_id'], 'provenance_sha256': identity_map[row['observation_id']]['provenance_sha256']}
            for row in evidence['observations']],
        'observer_models': observer_models, 'view_original_ids': view_original_ids, 'text_order': list(evidence['texts']),
        'bindings': {'canonical_body_sha256': fingerprint(canonical), 'identity_map_sha256': fingerprint(identity_map),
                     'body_sha256': fingerprint(body)}}
    audit['audit_sha256'] = fingerprint(audit)
    result = {'version': VERSION, 'body': body, 'audit': audit}
    ec._require(_same(reconstruct_projection(body, audit), canonical), 'Projection round trip differs')
    return result


def reconstruct_projection(body: dict, audit: dict) -> dict:
    """Recover the exact L1 body from visible values plus nonsemantic provenance."""
    _keys(body, _BODY_KEYS); _keys(audit, _AUDIT_KEYS); _keys(audit['bindings'], _BINDINGS)
    ec._require(audit['version'] == VERSION and audit['audit_sha256'] == fingerprint(
        {key: value for key, value in audit.items() if key != 'audit_sha256'}), 'Audit changed')
    ec._require(audit['bindings']['body_sha256'] == fingerprint(body), 'Model-visible body changed')
    _keys(audit['source_metadata'], _SOURCE_META); _keys(audit['map_metadata'], _MAP_META)
    ec._require(_same(body['scope_rules'], _RULES) and _same(body['authority'], ec.AUTHORITY), 'Evidence authority changed')
    ec._require(_same(body['temporal_authority'], {'word_ownership_inferred': False, 'acoustic_truth_verified': False}),
                'Temporal authority changed')
    for value in audit['bindings'].values(): _hash(value)
    for key in ('source_rows', 'observations', 'observers', 'views'):
        ec._require(isinstance(body[key], list) and bool(body[key]), 'Required readable collection missing')
    source = []; owners = {}
    for row in body['source_rows']:
        _keys(row, {'index', 'ts_line', 'text'})
        ec._require(type(row['index']) is int and row['index'] == len(source)+1, 'Source owner sequence changed')
        ec._text(row['text']); ec._text(row['ts_line']); source.append(deepcopy(row)); owners[row['index']] = row
    ec._text(body['original_context'], nonempty=False)
    ec._require(isinstance(body['frames'], dict) and set(body['frames']) == {str(key) for key in owners}, 'Frame coverage changed')
    for frame in body['frames'].values():
        _keys(frame, {'summary_ja', 'unknowns', 'unresolved'}, {'source_accuracy_verified', 'literal_observation_coverage_valid'})
        ec._text(frame['summary_ja'], nonempty=False)
        ec._require(isinstance(frame['unknowns'], list) and type(frame['unresolved']) is bool, 'Invalid frame uncertainty')
        for value in frame['unknowns']: ec._text(value)
        for key in ('source_accuracy_verified', 'literal_observation_coverage_valid'):
            if key in frame: ec._require(type(frame[key]) is bool, 'Invalid frame coverage flag')
        if 'source_accuracy_verified' in frame: ec._require(frame['source_accuracy_verified'] is False, 'Frame claims authority')
    observers = {}; views = {}
    ec._require(isinstance(audit['observer_models'], dict) and isinstance(audit['view_original_ids'], dict), 'Inverse registry missing')
    for item in body['observers']:
        _keys(item, {'id', 'family'}); identifier = item['id']
        ec._require(isinstance(identifier, str) and identifier not in observers and identifier in audit['observer_models'],
                    'Observer identity duplicated or unknown')
        observers[identifier] = item
    ec._require(set(observers) == set(audit['observer_models']), 'Unused observer identity')
    for item in body['views']:
        _keys(item, _VIEW_KEYS); identifier = item['id']
        ec._require(isinstance(identifier, str) and identifier not in views and identifier in audit['view_original_ids']
            and type(item['owner_id']) is int and item['owner_id'] in owners
            and type(item['sample_rate']) is int and item['sample_rate'] > 0, 'Invalid view registry')
        for name in ('crop_seconds', 'core_seconds', 'owner_intersection_seconds'):
            interval = item[name]
            ec._require((interval is None and name != 'crop_seconds') or
                        isinstance(interval, list) and len(interval) == 2, 'Invalid exact interval')
            if interval is not None: ec._require(_fraction(interval[0]) < _fraction(interval[1]), 'Nonpositive view interval')
        views[identifier] = item
    ec._require(set(views) == set(audit['view_original_ids']), 'Unused view identity')
    shown = {}
    for item in body['observations']:
        _keys(item, {'id', 'owner_id', 'text', 'observer', 'view'}, {'origin', 'owner_exact'})
        ec._require(isinstance(item['id'], str) and item['id'] not in shown and item['observer'] in observers
                    and item['view'] in views and type(item['owner_id']) is int and item['owner_id'] in owners,
                    'Invalid readable observation reference')
        ec._text(item['text']); shown[item['id']] = item
        if 'origin' in item: ec._text(item['origin'])
        if 'owner_exact' in item: ec._require(type(item['owner_exact']) is bool, 'Invalid owner association flag')
    ec._require({item['observer'] for item in shown.values()} == set(observers)
                and {item['view'] for item in shown.values()} == set(views), 'Unreferenced observer or view')
    texts = {}; encoded_source = []; encoded_observations = []; observations = []; identities = {}; inverse = {}
    def add_text(identifier: Any, literal: str) -> None:
        ec._require(isinstance(identifier, str) and re.fullmatch(r't[0-9]{4,}', identifier) is not None, 'Invalid original text ID')
        ec._require(identifier not in texts or texts[identifier] == literal, 'Shared text references disagree')
        texts[identifier] = literal
    ec._require(isinstance(audit['owner_text_ids'], list) and len(audit['owner_text_ids']) == len(source), 'Source text coverage changed')
    for row, link in zip(source, audit['owner_text_ids']):
        _keys(link, {'index', 'text_id'}); ec._require(type(link['index']) is int and link['index'] == row['index'], 'Source text owner differs')
        add_text(link['text_id'], row['text'])
        encoded_source.append({'index': row['index'], 'ts_line': row['ts_line'], 'text_id': link['text_id']})
    ec._require(isinstance(audit['observation_links'], list) and len(audit['observation_links']) == len(shown), 'Observation coverage changed')
    for link in audit['observation_links']:
        _keys(link, {'id', 'original_id', 'text_id', 'provenance_sha256'}); _hash(link['provenance_sha256'])
        ec._require(link['id'] in shown and link['id'] not in inverse and isinstance(link['original_id'], str)
                    and link['original_id'] not in identities, 'Observation inverse mapping is not bijective')
        item = shown[link['id']]; view = views[item['view']]; observer = observers[item['observer']]
        ec._require(item['owner_id'] == view['owner_id'], 'Observation changed view owner')
        rate = view['sample_rate']; start, end = [_frames(value, rate) for value in view['crop_seconds']]
        ec._require(_same(view['owner_intersection_seconds'], _intersection(start, end, rate, owners[item['owner_id']])),
                    'View intersection differs from exact owner and crop')
        core = [None, None] if view['core_seconds'] is None else [_frames(value, rate) for value in view['core_seconds']]
        observation = {'observation_id': link['original_id'], 'owner_id': item['owner_id'], 'text': item['text'],
            'crop_start_frame': start, 'crop_end_frame': end, 'sample_rate': rate,
            **{key: item[key] for key in ('origin', 'owner_exact') if key in item}}
        identity = {key: observation[key] for key in _GEOMETRY_KEYS}
        identity.update(family=observer['family'], model_identity_sha256=audit['observer_models'][observer['id']],
            view_id=audit['view_original_ids'][view['id']], view_kind=view['kind'], transform=view['transform'],
            core_start_frame=core[0], core_end_frame=core[1], duplicate_geometry=view['duplicate_geometry'],
            provenance_sha256=link['provenance_sha256'])
        identities[link['original_id']] = identity; observations.append(observation); inverse[link['id']] = link['original_id']
        add_text(link['text_id'], item['text'])
        encoded_observations.append({**{key: value for key, value in observation.items() if key != 'text'}, 'text_id': link['text_id']})
    _identities(observations, identities)
    ec._require(fingerprint(identities) == audit['bindings']['identity_map_sha256'], 'Observer/view provenance differs')
    ec._require(isinstance(body['additional_literals'], dict), 'Additional literal table missing')
    for identifier, literal in body['additional_literals'].items():
        ec._require(identifier not in texts, 'Additional literal is already referenced'); ec._text(literal)
        add_text(identifier, literal)
    order = audit['text_order']
    ec._require(isinstance(order, list) and all(isinstance(key, str) for key in order)
                and len(order) == len(set(order)) and set(order) == set(texts), 'Original text-table coverage changed')
    map_body = body['source_map']; _map_shape(map_body, set(owners), shown)
    restored_map = {**deepcopy(audit['map_metadata']), **_map_refs(map_body, inverse)}
    groups, authority = tr.temporal_table(source, observations)
    evidence = {**deepcopy(audit['source_metadata']), 'source_rows': encoded_source, 'observations': encoded_observations,
        'original_context': body['original_context'], 'frames': deepcopy(body['frames']), 'authority': deepcopy(body['authority']),
        'texts': {key: texts[key] for key in order}, 'temporal_groups': groups, 'temporal_authority': authority}
    canonical = {'source_evidence': evidence, 'source_map': restored_map, 'focus_owner_ids': list(owners)}
    ec._require(fingerprint(canonical) == audit['bindings']['canonical_body_sha256'], 'Reconstructed literals or metadata changed')
    return canonical


def validate_projection(projection: dict, pack: dict, bound: dict, identity_map: dict) -> dict[str, int]:
    """Compare against separately supplied pinned inputs; return numeric coverage."""
    _keys(projection, {'version', 'body', 'audit'})
    ec._require(projection['version'] == VERSION, 'Unknown readable projection version')
    restored = reconstruct_projection(projection['body'], projection['audit'])
    ec._require(_same(restored, _canonical_input(pack, bound)), 'Projection belongs to different source evidence')
    ec._require(_same(projection, build_projection(pack, bound, identity_map)), 'Projection differs from its pinned inputs')
    return {'source_owners': len(pack['source_rows']), 'observations': len(pack['observations']),
            'observers': len(projection['body']['observers']), 'views': len(projection['body']['views']),
            'recap_claims': len(bound['claims']), 'additional_literals': len(projection['body']['additional_literals'])}


_PROMPT_KEYS = _BODY_KEYS - {'observations', 'observers'}
_READING_KEYS = {'observation_id', 'owner_id', 'text', 'observer_instance', 'family'}


def _checked_projection(projection: dict) -> dict:
    _keys(projection, {'version', 'body', 'audit'})
    ec._require(projection['version'] == VERSION, 'Unknown readable projection version')
    reconstruct_projection(projection['body'], projection['audit'])
    return projection['body']


def _view_order(view: dict) -> tuple:
    return (_fraction(view['crop_seconds'][0]), _fraction(view['crop_seconds'][1]),
            view['owner_id'], view['sample_rate'], view['id'])


def _literal_observation_aliases(original: dict, audit: dict) -> dict[str, str]:
    """Expose exact known IDs occurring in free text, without interpreting it."""
    literals = [original['original_context'], *original['additional_literals'].values()]
    literals.extend(row['text'] for row in original['source_rows'] + original['observations'])
    for frame in original['frames'].values():
        literals.extend([frame['summary_ja'], *frame['unknowns']])
    for claim in original['source_map']['claims']:
        literals.append(claim['claim_ja'])
        literals.extend(branch['interpretation_ja'] for branch in claim['alternatives'])
    return {link['id']: link['original_id'] for link in audit['observation_links']
            if any(link['original_id'] in literal for literal in literals)}


def _inline_reading(observation: dict, observer: dict, original_id: str | None = None) -> dict:
    return {'observation_id': observation['id'], 'owner_id': observation['owner_id'],
            'text': observation['text'], 'observer_instance': observer['id'], 'family': observer['family'],
            **({'original_observation_id': original_id} if original_id is not None else {}),
            **{key: observation[key] for key in ('origin', 'owner_exact') if key in observation}}


def render_prompt_body(projection: dict) -> dict:
    """Place literal readings and observer labels inside each exact view header.

    This is an additive presentation layer. It neither changes the canonical
    projection/audit nor introduces a new acquisition, interpretation or vote.
    """
    original = _checked_projection(projection)
    aliases = _literal_observation_aliases(original, projection['audit'])
    observers = {item['id']: item for item in original['observers']}
    grouped = {view['id']: [] for view in original['views']}
    for observation in original['observations']:
        grouped[observation['view']].append(_inline_reading(
            observation, observers[observation['observer']], aliases.get(observation['id'])))
    result = {key: deepcopy(original[key]) for key in original if key in _PROMPT_KEYS and key != 'views'}
    result['views'] = [{**deepcopy(view), 'readings': sorted(grouped[view['id']], key=lambda item: item['observation_id'])}
                       for view in sorted(original['views'], key=_view_order)]
    return result


def validate_prompt_body(body: dict, projection: dict) -> dict[str, int]:
    """Verify adjacent literal/family/scope associations and complete once-only coverage."""
    original = _checked_projection(projection); _keys(body, _PROMPT_KEYS)
    aliases = _literal_observation_aliases(original, projection['audit'])
    for key in _PROMPT_KEYS - {'views'}:
        ec._require(_same(body[key], original[key]), 'Rendered source, context, recap or uncertainty changed')
    views = {item['id']: item for item in original['views']}
    observations = {item['id']: item for item in original['observations']}
    observers = {item['id']: item for item in original['observers']}
    ec._require(isinstance(body['views'], list) and len(body['views']) == len(views), 'Rendered view coverage changed')
    seen_views = set(); seen_observations = set()
    for view in body['views']:
        _keys(view, _VIEW_KEYS | {'readings'}); identifier = view['id']
        ec._require(isinstance(identifier, str) and identifier in views and identifier not in seen_views,
                    'Unknown or repeated rendered view')
        seen_views.add(identifier)
        ec._require(_same({key: view[key] for key in _VIEW_KEYS}, views[identifier]), 'Rendered view scope changed')
        ec._require(isinstance(view['readings'], list) and bool(view['readings']), 'Rendered view has no readings')
        local_ids = []
        for reading in view['readings']:
            _keys(reading, _READING_KEYS, {'origin', 'owner_exact', 'original_observation_id'})
            observation_id = reading['observation_id']
            ec._require(isinstance(observation_id, str) and observation_id in observations
                        and observation_id not in seen_observations, 'Unknown or repeated rendered observation')
            expected = observations[observation_id]
            ec._require(expected['view'] == identifier, 'Reading moved to another acoustic scope')
            ec._require(_same(reading, _inline_reading(
                expected, observers[expected['observer']], aliases.get(observation_id))),
                        'Rendered literal, observer family or source association changed')
            local_ids.append(observation_id); seen_observations.add(observation_id)
        ec._require(local_ids == sorted(item['id'] for item in observations.values() if item['view'] == identifier),
                    'Rendered readings are missing or out of order')
    ec._require(seen_views == set(views) and seen_observations == set(observations), 'Rendered evidence coverage changed')
    ec._require([view['id'] for view in body['views']] == [view['id'] for view in sorted(views.values(), key=_view_order)],
                'Rendered view chronology changed')
    return {'source_owners': len(original['source_rows']), 'observations': len(seen_observations),
            'observers': len(observers), 'views': len(seen_views), 'recap_claims': len(original['source_map']['claims'])}
