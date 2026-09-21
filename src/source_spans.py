"""Fixed raw-ASR span choices, with exact ownership projection and no model I/O.

W and N corroborate a bounded alternative on the raw Q backbone. Agreement,
dictionary readings and a selector decision never establish source correctness.
Returned text is provisional and must be realigned before display use. This
module never changes the supplied source, raw ASR, tokens or timestamps.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from difflib import SequenceMatcher
from functools import lru_cache
import hashlib
import json
import math
import re

from src.asr_consensus import reading_map

VERSION = 1
_PUNCTUATION = frozenset('。！？!?、，,「」『』“”‘’()（）;；:：\n')


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@lru_cache(maxsize=1)
def _tokenizer():
    from janome.tokenizer import Tokenizer
    return Tokenizer()


def _tokens(text: str) -> list[tuple]:
    result, cursor = [], 0
    for token in _tokenizer().tokenize(text):
        # Janome can omit whitespace; never invent offsets in that case.
        if not text.startswith(token.surface, cursor):
            return []
        end = cursor + len(token.surface)
        result.append((cursor, end, token.part_of_speech.split(',')[:2]))
        cursor = end
    return result if cursor == len(text) else []


def _distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, char in enumerate(a, 1):
        next_row = [i]
        for j, other in enumerate(b, 1):
            next_row.append(min(row[j] + 1, next_row[-1] + 1, row[j-1] + (char != other)))
        row = next_row
    return row[-1]


def _reading_interval(mapping, start: int, end: int):
    indices = [i for i, (a, b) in enumerate(mapping) if b > start and a < end]
    if not indices:
        return None
    lo, hi = indices[0], indices[-1] + 1
    spans = mapping[lo:hi]
    if min(a for a, _ in spans) != start or max(b for _, b in spans) != end:
        return None
    return (lo, hi) if all(start <= a < b <= end for a, b in spans) else None


def _anchor_map(opcodes, lo: int, hi: int):
    for tag, a0, a1, b0, _ in opcodes:
        if tag == 'equal' and a0 <= lo and hi <= a1:
            return b0 + lo - a0, b0 + hi - a0
    return None


def _unique_pair(text: str, left: str, right: str, lo: int, hi: int) -> bool:
    pairs = [(i + len(left), j)
             for i in range(len(text) - len(left) + 1) if text.startswith(left, i)
             for j in range(i + len(left), min(len(text)-len(right), i+len(left)+24)+1)
             if text.startswith(right, j)]
    return pairs == [(lo, hi)]


def _surfaces(text: str, protected: tuple[str, ...]) -> list[str]:
    pattern = '|'.join(re.escape(term) for term in sorted(protected, key=lambda x: (-len(x), x)))
    return re.findall(pattern, text) if pattern else []


def _span_allowed(text: str) -> bool:
    return 0 < len(text) <= 12 and not any(c in _PUNCTUATION or c.isnumeric() for c in text)


def _window_rows(metadata: dict) -> dict[int, tuple[dict, dict]]:
    windows, raw = metadata['windows'], metadata['raw_transcripts']
    if len(windows) != len(raw):
        raise ValueError('Window and raw transcript counts differ')
    result = {}
    for window, transcripts in zip(windows, raw):
        index = window['index']
        if type(index) is not int or index in result:
            raise ValueError('Window IDs must be unique integers')
        if any(not isinstance(transcripts.get(k, ''), str) for k in ('w', 'n', 'q')):
            raise ValueError('Raw W/N/Q hypotheses must be strings')
        result[index] = (window, transcripts)
    return result


def _window_candidates(window: dict, raw: dict, protected: tuple[str, ...]) -> list[dict]:
    q = raw.get('q', '')
    if not q or not raw.get('w') or not raw.get('n'):
        return []
    qr, qm = reading_map(q)
    qt = _tokens(q)
    others = {k: (raw[k], *reading_map(raw[k]), _tokens(raw[k])) for k in ('w', 'n')}
    maps = {k: SequenceMatcher(None, qr, row[1], autojunk=False).get_opcodes()
            for k, row in others.items()}
    local = {}
    for i in range(len(qt)):
        for size in range(1, 7):
            owned = qt[i:i+size]
            if len(owned) != size:
                break
            a, end = owned[0][0], owned[-1][1]
            old = q[a:end]
            if len(old) > 12:
                break
            interval = _reading_interval(qm, a, end)
            if not _span_allowed(old) or interval is None:
                continue
            lo, hi = interval
            if lo < 4 or hi + 4 > len(qr):
                continue
            left, right = qr[lo-4:lo], qr[hi:hi+4]
            if not _unique_pair(qr, left, right, lo, hi):
                continue
            support = []
            for family, (alt, ar, am, at) in others.items():
                lm = _anchor_map(maps[family], lo-4, lo)
                rm = _anchor_map(maps[family], hi, hi+4)
                if lm is None or rm is None:
                    continue
                al, ah = lm[1], rm[0]
                if al >= ah or not _unique_pair(ar, left, right, al, ah):
                    continue
                aa, ae = am[al][0], am[ah-1][1]
                ats = [t for t in at if t[1] > aa and t[0] < ae]
                new = alt[aa:ae]
                if (_reading_interval(am, aa, ae) != (al, ah) or not ats
                        or ats[0][0] != aa or ats[-1][1] != ae or not _span_allowed(new)
                        or _surfaces(q, protected) != _surfaces(q[:a]+new+q[end:], protected)):
                    continue
                distance = _distance(qr[lo:hi], ar[al:ah])
                normalized = distance / max(1, hi-lo, ah-al)
                if distance > 2 or normalized > .5:
                    continue
                support.append(dict(family=family, start=aa, end=ae, exact_text=new,
                                    raw_text_sha256=text_sha256(alt), reading=ar[al:ah],
                                    reading_edit_distance=distance,
                                    normalized_reading_distance=normalized,
                                    pos=[t[2] for t in ats]))
            if len(support) != 2 or support[0]['exact_text'] != support[1]['exact_text']:
                continue
            new = support[0]['exact_text']
            edits = [list(op) for op in SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
                     if op[0] != 'equal']
            # Replacement opcodes may have unequal lengths. This is not a claim
            # that facts, phonemes, predicates or grammatical force are invariant.
            if not edits or any(op[0] != 'replace' for op in edits):
                continue
            prefix = 0
            while prefix < min(len(old), len(new)) and old[prefix] == new[prefix]:
                prefix += 1
            suffix = 0
            while (suffix < min(len(old)-prefix, len(new)-prefix)
                   and old[-1-suffix] == new[-1-suffix]):
                suffix += 1
            signature = (a+prefix, end-suffix, new[prefix:len(new)-suffix if suffix else len(new)])
            rank = (len(old)+len(new), len(old), a)
            row = dict(window=window['index'], window_seconds=[window['start'], window['end']],
                       start=a, end=end, before=old, replacement=new,
                       q_text_sha256=text_sha256(q), q_reading=qr[lo:hi],
                       q_pos=[t[2] for t in owned], left_reading_anchor=left,
                       right_reading_anchor=right, support=support,
                       character_edit_opcodes=edits, source_status='unresolved',
                       applied_to_pipeline=False)
            if signature not in local or rank < local[signature][0]:
                local[signature] = (rank, row)
    return [row for _, row in local.values()]


def build_span_candidates(asr_metadata: dict, *, protected_surfaces=()) -> list[dict]:
    """Enumerate exact all-POS W+N alternatives; names are exclusions only."""
    protected = tuple(protected_surfaces)
    if any(not isinstance(term, str) or not term for term in protected):
        raise ValueError('Protected surfaces must be nonempty text')
    protected = tuple(sorted(set(protected)))
    rows = [row for window, raw in _window_rows(asr_metadata).values()
            for row in _window_candidates(window, raw, protected)]
    rows.sort(key=lambda row: (row['window'], row['start'], row['end'], row['replacement']))
    return [dict(row, id=f'A{i:03d}') for i, row in enumerate(rows, 1)]


def _tasks(candidates: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    ids = set()
    for row in candidates:
        if (not isinstance(row.get('id'), str) or row['id'] in ids
                or row['id'] in {'KEEP', 'UNSURE'} or type(row.get('window')) is not int):
            raise ValueError('Candidate IDs must be unique and have integer windows')
        ids.add(row['id'])
        groups[row['window']].append(row)
    return [dict(window=window, candidate_spans=rows,
                 allowed=['KEEP', 'UNSURE']+[row['id'] for row in rows], max_selected_spans=1)
            for window, rows in sorted(groups.items())]


def selection_schema(candidates: list[dict]) -> dict:
    properties = {str(i): dict(type='object', properties={
        'decision': dict(type='string', enum=task['allowed']),
        'reason': dict(type='string', minLength=1, maxLength=180)},
        required=['decision', 'reason'], additionalProperties=False)
        for i, task in enumerate(_tasks(candidates), 1)}
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def source_selection_instruction() -> str:
    return '''Select a bounded Japanese source correction from a fixed lattice.
The complete raw Q transcript is the coverage spine. Every allowed replacement is
an exact complete span present at the same uniquely anchored position in BOTH raw
Whisper W and raw RNNT N. This is corroboration, not proof of acoustic truth.
Choose KEEP, UNSURE or exactly one listed candidate ID per window. Never write a
sentence, translation, new word, new candidate, offset or combined patch. Text and
clauses outside that one span are immutable. These candidates can repair corrupted
content words, predicates, adjectives or inflection, including a wrong action or
speech act, when both positional raw alternatives and the local dialogue support
that interpretation. Do not choose an alternative merely for fluency or cosmetics.
If the edit would arbitrarily change actor/patient, negation, request, quotation,
character identity, or omitted content rather than repair evidenced ASR corruption,
choose UNSURE. Only the fixed candidate can be selected; no clauses or other words
may be removed or completed. Return the exact requested local JSON IDs, each with
decision and a brief source-grounded reason. All choices remain unresolved for
separate source/audio evaluation; no source confidence is upgraded.'''


def source_selection_body(asr_metadata: dict, candidates: list[dict]) -> str:
    """Only raw local audio hypotheses enter the selector body, never targets."""
    windows = _window_rows(asr_metadata)
    order = list(windows)
    payload = {}
    for local, task in enumerate(_tasks(candidates), 1):
        index = order.index(task['window'])
        window, raw = windows[task['window']]
        payload[str(local)] = dict(task=task, original_window=window,
            raw_W_N_Q={k: raw.get(k, '') for k in ('w', 'n', 'q')},
            neighbors_read_only=[dict(relative_window=k-index, raw_Q=windows[order[k]][1].get('q', ''))
                for k in range(max(0, index-1), min(len(order), index+2)) if k != index])
    return json.dumps(payload, ensure_ascii=False)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f'Invalid JSON constant: {value}')


def parse_span_selections(answer: str, candidates: list[dict]) -> list[dict]:
    """Strict whole-response parsing. Invalid/missing IDs never mean KEEP."""
    tasks = _tasks(candidates)
    parsed = json.loads(answer, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(parsed, dict) or set(parsed) != {str(i) for i in range(1, len(tasks)+1)}:
        raise ValueError('Exact local selection IDs required')
    result = []
    for i, task in enumerate(tasks, 1):
        row = parsed[str(i)]
        if (not isinstance(row, dict) or set(row) != {'decision', 'reason'}
                or row['decision'] not in task['allowed'] or not isinstance(row['reason'], str)
                or not row['reason'].strip() or len(row['reason']) > 180):
            raise ValueError('Invalid fixed choice or bounded reason')
        result.append(dict(window=task['window'], **row, source_status='unresolved'))
    return result


def _token_ownership(metadata: dict):
    windows = _window_rows(metadata)
    segments, tokens = metadata['segments'], metadata['utterance_tokens']
    if len(segments) != len(tokens) or len({s['index'] for s in segments}) != len(segments):
        raise ValueError('Unique segment IDs and matching token rows required')
    selection = {row['window']: row['selected'] for row in metadata['source_selection']}
    if len(selection) != len(metadata['source_selection']):
        raise ValueError('Duplicate source selection window')
    used = metadata.get('used_transcripts')
    if used is not None and len(used) != len(windows):
        raise ValueError('Used transcript count mismatch')
    used_by_window = dict(zip(windows, used)) if used is not None else {}
    coverage, problems, originals = defaultdict(list), defaultdict(set), {}
    for segment, words in zip(segments, tokens):
        uid = segment['index']
        if type(uid) is not int:
            raise ValueError('Utterance IDs must be integers')
        joined = ''.join(token['text'] for token in words)
        originals[uid] = segment['text']
        trim = len(joined) - len(joined.lstrip())
        offset = -trim
        for token in words:
            start, end = token['start'], token['end']
            if (isinstance(start, bool) or isinstance(end, bool)
                    or not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (start, end))
                    or end < start):
                raise ValueError('Invalid token time; ownership cannot be established')
            middle = (start+end)/2
            owners = [wid for wid, (window, _) in windows.items()
                      if window['core_start'] <= middle < window['core_end']]
            if len(owners) != 1:
                # Even an unowned token might cover a requested span. Do not
                # silently discard it and mistake the remainder for exact cover.
                raise ValueError('Token does not have exactly one core owner')
            wid = owners[0]
            window, raw = windows[wid]
            q, begin, finish = raw.get('q', ''), token['begin'], token['finish']
            reason = None
            if selection.get(wid) != 'q' or (used is not None and used_by_window[wid].get('q') != q):
                reason = 'Selected window is not the unchanged raw Q backbone'
            elif joined.strip() != segment['text']:
                reason = 'Utterance text differs from its complete token ledger'
            elif start < window['start'] or end > window['end']:
                reason = 'Token times extend outside the originating window'
            elif type(begin) is not int or type(finish) is not int or not 0 <= begin < finish <= len(q):
                reason = 'Invalid raw token character bounds'
            else:
                fragment = q[begin:finish]
                left_trim = len(fragment)-len(fragment.lstrip())
                # The existing cross-window merge removes trailing full stops.
                # It may remove only these literal characters, never word text.
                if token['text'] not in (fragment.strip(), fragment.strip().rstrip('。')):
                    reason = 'Raw Q/token text mismatch'
                else:
                    for j in range(len(token['text'])):
                        coverage[(wid, begin+left_trim+j)].append((uid, offset+j))
            if reason:
                problems[wid].add(reason)
            offset += len(token['text'])
    return coverage, problems, originals


def project_span_selections(source: dict[int, str], asr_metadata: dict,
                            selections: list[dict], candidates: list[dict], *,
                            protected_surfaces=()) -> dict:
    """Return provisional source, changed IDs and exact original-span provenance.

    Character offsets are zero-based, end-exclusive. Projected offsets refer to
    the unchanged input utterance; multiple disjoint edits are applied backwards.
    Rejected, KEEP and UNSURE decisions all retain unresolved source status.
    The caller must realign changed IDs and preserve existing source blockers.
    """
    tasks = _tasks(candidates)
    task_map = {task['window']: task for task in tasks}
    if (len({row['window'] for row in selections}) != len(selections)
            or {row['window'] for row in selections} != set(task_map)):
        raise ValueError('Exactly one decision per candidate window required')
    fresh = build_span_candidates(asr_metadata, protected_surfaces=protected_surfaces)
    canonical = {json.dumps({k: v for k, v in row.items() if k != 'id'}, sort_keys=True,
                            ensure_ascii=False) for row in fresh}
    by_id = {row['id']: row for row in candidates}
    result = dict(source=dict(source), changed_ids=[], decisions=[])
    try:
        coverage, problems, originals = _token_ownership(asr_metadata)
        ownership_error = None
    except (ValueError, KeyError, TypeError) as exc:
        coverage, problems, originals = {}, {}, {}
        ownership_error = str(exc)
    pending = defaultdict(list)
    for choice in selections:
        wid, decision = choice['window'], choice['decision']
        reason = choice.get('reason')
        if (decision not in task_map[wid]['allowed'] or not isinstance(reason, str)
                or not reason.strip() or len(reason) > 180):
            raise ValueError('Invalid fixed choice or bounded reason')
        record = dict(window=wid, decision=decision, reason=reason,
                      source_status='unresolved', status=decision.lower(), applied=False)
        result['decisions'].append(record)
        if decision in {'KEEP', 'UNSURE'}:
            continue
        patch = by_id[decision]
        record['candidate'] = deepcopy(patch)
        signature = json.dumps({k: v for k, v in patch.items() if k != 'id'}, sort_keys=True,
                               ensure_ascii=False)
        failure = ownership_error
        if signature not in canonical:
            failure = 'Stale, altered or unsupported raw W/N/Q candidate'
        elif problems.get(wid):
            failure = '; '.join(sorted(problems[wid]))
        if not failure:
            positions = [coverage.get((wid, i), []) for i in range(patch['start'], patch['end'])]
            if any(len(owners) != 1 for owners in positions):
                failure = 'Raw span has missing or ambiguous token coverage'
            elif len({owners[0][0] for owners in positions}) != 1:
                failure = 'Raw span crosses utterance ownership'
            else:
                uid, begin = positions[0][0]
                end = begin + len(patch['before'])
                if [owners[0][1] for owners in positions] != list(range(begin, end)):
                    failure = 'Raw span is not contiguous in the selected utterance'
                elif source.get(uid) != originals.get(uid):
                    failure = 'Stale selected source differs from ASR utterance'
                elif source[uid][begin:end] != patch['before']:
                    failure = 'Projected source span differs from raw Q'
                else:
                    record.update(status='projected_unresolved', utterance_id=uid,
                        projected_span=dict(start=begin, end=end, before=patch['before'],
                                            replacement=patch['replacement']),
                        original_source_sha256=text_sha256(source[uid]),
                        timing_status='requires_realignment')
                    pending[uid].append(record)
        if failure:
            record.update(status='rejected', rejection=failure)
    for uid, records in pending.items():
        records.sort(key=lambda row: row['projected_span']['start'])
        if any(a['projected_span']['end'] > b['projected_span']['start'] for a, b in zip(records, records[1:])):
            for row in records:
                row.update(status='rejected', rejection='Projected spans overlap')
            continue
        text = source[uid]
        for row in reversed(records):
            patch = row['projected_span']
            text = text[:patch['start']] + patch['replacement'] + text[patch['end']:]
        result['source'][uid] = text
        result['changed_ids'].append(uid)
        for row in records:
            row.update(applied=True, proposed_source_sha256=text_sha256(text))
    result['changed_ids'].sort()
    return result
