"""Create readable paired SRTs after utterance translation and semantic QA."""
from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from dataclasses import replace
from pathlib import Path

from src.asr_consensus import reading_map
from src.config import TranslateConfig
from src.quality import mode, seconds
from src.translate import LLMRetryExhaustedError, SrtBlock, call_llm
from src.workflow_state import fingerprint, read_json, write_json

logger = logging.getLogger(__name__)

LAYOUT = '''Partition these already translated subtitles into corresponding readable clauses.
Return JSON only: {"parts":[{"source":"exact source substring","target":"exact target substring"}]}.
Do not translate, correct, add, remove, reorder, or paraphrase text. Concatenating source parts
must exactly reproduce SOURCE; concatenating target parts must exactly reproduce TARGET.
Keep each pair aligned in meaning. Keep negations, names and verb auxiliaries together.
Aim for at most 32 Chinese characters per part. Use complete phrases, not isolated particles.
Every source and target part must contain meaningful text, never punctuation alone.
'''


def timestamp(start: float, end: float) -> str:
    def stamp(value):
        ms = max(0, round(value*1000))
        h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
        return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'
    return stamp(start) + ' --> ' + stamp(end)


def validate_parts(parts, source: str, target: str) -> bool:
    return (isinstance(parts, list) and 1 <= len(parts) <= 12
            and all(isinstance(p, dict) and isinstance(p.get('source'), str) and p['source'].strip()
                    and isinstance(p.get('target'), str) and p['target'].strip() for p in parts)
            and ''.join(p['source'] for p in parts) == source
            and ''.join(p['target'] for p in parts) == target)


def merge_nonlexical_parts(parts: list[dict]) -> list[dict]:
    """Attach punctuation-only layout fragments without changing either text.

    A target full stop can correspond to a source verb phrase after the model
    moved the whole Chinese predicate into the preceding part. Joining the pair
    retains its speech interval instead of showing an empty-looking subtitle.
    This preserves text; it does not certify semantic alignment or translation.
    """
    result = [dict(part) for part in parts]
    index = 0
    while len(result) > 1 and index < len(result):
        part = result[index]
        if any(not any(char.isalnum() for char in part[field]) for field in ('source', 'target')):
            if index:
                for field in ('source', 'target'):
                    result[index - 1][field] += part[field]
                result.pop(index)
            else:
                for field in ('source', 'target'):
                    result[1][field] = part[field] + result[1][field]
                result.pop(0)
        else:
            index += 1
    return result


def restore_exact_parts(parts, source: str, target: str):
    """Ignore a layout model's punctuation edits while retaining every word."""
    if not isinstance(parts,list) or not 1 <= len(parts) <= 12:
        return None
    result = [dict() for _ in parts]
    def letters(text):
        return [(i,c) for i,c in enumerate(text) if not c.isspace() and not unicodedata.category(c).startswith('P')]
    for field, text in (('source',source),('target',target)):
        if any(not isinstance(p,dict) or not isinstance(p.get(field),str) for p in parts):
            return None
        units = [letters(p[field]) for p in parts]
        original = letters(text)
        if any(not u for u in units) or ''.join(c for u in units for _,c in u) != ''.join(c for _,c in original):
            return None
        cursor = 0; count = 0
        for i,u in enumerate(units):
            count += len(u)
            end = original[count][0] if count < len(original) else len(text)
            result[i][field] = text[cursor:end]
            cursor = end
    return result if validate_parts(result,source,target) else None


def plan_parts(source: str, target: str, config: TranslateConfig, cache: Path) -> list[dict]:
    key = fingerprint(['display-v2', LAYOUT, source, target, config.endpoint, config.extra_payload])
    saved = read_json(cache, {})
    if not isinstance(saved, dict): saved = {}
    if validate_parts(saved.get(key), source, target): return merge_nonlexical_parts(saved[key])
    cfg = replace(mode(config, 'display-layout'), max_tokens=min(config.max_tokens, 2048),
                  response_guard_floor=max(config.response_guard_floor, 512))
    response = call_llm(f'SOURCE: {source}\nTARGET: {target}', LAYOUT, cfg)
    response = re.sub(r'^```(?:json)?\s*|\s*```$', '', response.strip())
    try: parts = json.loads(response).get('parts')
    except (ValueError, AttributeError): parts = None
    if not validate_parts(parts, source, target):
        parts = restore_exact_parts(parts,source,target)
    if not validate_parts(parts, source, target):
        return [dict(source=source, target=target)]  # Keep text intact; report the long cue.
    parts = merge_nonlexical_parts(parts)
    saved[key] = parts; write_json(cache, saved)
    return parts



def project_spelling_tokens(tokens: list[dict], source: str,
                            patches: list[dict] | None = None
                            ) -> tuple[list[dict] | None, dict]:
    """Project already-applied spelling spans onto copies of aligned tokens.

    Patches use zero-based, end-exclusive Unicode offsets into the stripped raw
    token text and contain start/end/original/replacement. This function neither
    applies source repairs nor validates their meaning. An identical dictionary
    reading only permits reuse of a token's existing time interval; it cannot
    approve a spelling, identity or source proposition. Cross-token changes need
    separate alignment evidence and deliberately retain utterance timing here.
    """
    def fallback(reason):
        return None, dict(status='fallback', reason=reason)

    if not tokens:
        return fallback('missing_tokens')
    joined = ''.join(token['text'] for token in tokens)
    original = joined.strip()
    if not patches:
        if original == source:
            return [dict(token) for token in tokens], dict(status='original')
        return fallback('source_changed_without_applied_spelling_patches')
    if not isinstance(patches, list):
        return fallback('invalid_spelling_patch_ledger')
    offset = len(joined) - len(joined.lstrip())
    spans = []
    cursor = -offset
    for token in tokens:
        spans.append((cursor, cursor + len(token['text'])))
        cursor += len(token['text'])
    checked = []
    for patch in patches:
        if not isinstance(patch, dict):
            return fallback('invalid_spelling_patch')
        left, right = patch.get('start'), patch.get('end')
        before, after = patch.get('original'), patch.get('replacement')
        if (type(left) is not int or type(right) is not int
                or not 0 <= left < right <= len(original)
                or not isinstance(before, str) or not isinstance(after, str)
                or not before.strip() or not after.strip() or before == after):
            return fallback('invalid_spelling_patch')
        if original[left:right] != before:
            return fallback('spelling_patch_original_mismatch')
        owned = [i for i, (begin, end) in enumerate(spans)
                 if begin <= left and right <= end]
        if len(owned) != 1:
            return fallback('spelling_patch_crosses_token_boundary')
        before_reading = reading_map(before)[0]
        if not before_reading or before_reading != reading_map(after)[0]:
            return fallback('spelling_patch_changes_reading')
        checked.append((left, right, before, after, owned[0]))
    checked.sort(key=lambda item: item[0])
    if any(a[1] > b[0] for a, b in zip(checked, checked[1:])):
        return fallback('overlapping_spelling_patches')
    rebuilt = original
    projected = [dict(token) for token in tokens]
    for left, right, before, after, index in reversed(checked):
        rebuilt = rebuilt[:left] + after + rebuilt[right:]
        begin, _ = spans[index]
        text = projected[index]['text']
        projected[index]['text'] = text[:left-begin] + after + text[right-begin:]
    if rebuilt != source or ''.join(token['text'] for token in projected).strip() != source:
        return fallback('spelling_patches_do_not_cover_selected_source')
    # A spelling can change a neighboring kanji's dictionary reading. Do not
    # reuse alignment when the complete original/source reading then differs.
    original_reading = reading_map(original)[0]
    if not original_reading or original_reading != reading_map(source)[0]:
        return fallback('spelling_patches_change_contextual_reading')
    return projected, dict(status='spelling_projection',
                           patch_count=len(checked),
                           token_indices=sorted({item[4] for item in checked}),
                           source_validated=False)


def build_display(source: list[SrtBlock], translated: list[SrtBlock], metadata: dict,
                  config: TranslateConfig, cache: Path, *,
                  spelling_patches: dict[int, list[dict]] | None = None,
                  model_failure_fallback: bool = False
                  ) -> tuple[list[SrtBlock], list[SrtBlock], list[dict]]:
    if len(source) != len(translated): raise ValueError('Utterance source/target count differs')
    rows = []
    for ordinal, (a, b) in enumerate(zip(source, translated), 1):
        start, end = map(seconds, a.ts_line.split('-->'))
        tokens = (metadata.get('utterance_tokens') or [[]]*len(source))[a.index-1]
        projected, projection = project_spelling_tokens(
            tokens, a.text, (spelling_patches or {}).get(a.index))
        parts = None
        pause_plan = {"status": "disabled"}
        layout_stage = "display-layout"
        try:
            if config.pause_layout and projected is not None:
                from src.pause_layout import find_pause_cuts, plan_pause_parts
                cuts = find_pause_cuts(a.text, projected, metadata.get('speech_regions', []),
                                       metadata.get('sample_rate', 0))
                pause_plan = {"status": "no_supported_cuts", "cuts": cuts}
                if cuts:
                    layout_stage = "pause-display-layout"
                    parts = plan_pause_parts(a.text, b.text, cuts, config, cache.with_suffix('.pauses.json'))
                    pause_plan["status"] = "partitioned" if parts is not None else "fallback_invalid_partition"
            elif config.pause_layout:
                pause_plan = {"status": "no_exact_source_token_coverage"}
            if parts is None:
                layout_stage = "display-layout"
                parts = plan_parts(a.text, b.text, config, cache) if len(b.text) > 38 or (end-start > 8 and len(b.text)>12) else [dict(source=a.text,target=b.text)]
        except LLMRetryExhaustedError:
            if not model_failure_fallback:
                raise
            # Layout is optional. Its failed response is never accepted or
            # cached; preserve the existing whole cue, including its geometry.
            if (a.index != ordinal or b.index != ordinal or a.ts_line != b.ts_line
                    or not all(math.isfinite(value) for value in (start, end))
                    or not 0 <= start < end or not a.text.strip() or not b.text.strip()):
                raise ValueError('Cannot preserve an invalid or mismatched layout input')
            if projected is not None:
                for token in projected:
                    left, right = token['start'], token['end']
                    if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value) for value in (left, right))
                            or not 0 <= left <= right):
                        raise ValueError('Cannot preserve layout with corrupt token geometry')
            if layout_stage == 'pause-display-layout':
                pause_plan = dict(pause_plan, status='fallback_model_failure')
            warning = dict(code='layout_model_retry_exhausted', stage=layout_stage,
                           text_preserved=True, timing_preserved=True, review_required=True)
            logger.warning('Layout retries exhausted for utterance %d (%s); '
                           'retaining original whole cue and timing', a.index, layout_stage)
            rows.append(dict(source=a.text, target=b.text, start=start, end=end,
                             utterance=a.index, alignment='utterance', token_projection=projection,
                             pause_layout=pause_plan, layout_warning=warning,
                             preserved_timestamp=a.ts_line))
            continue
        if projected is not None:
            tokens = projected
            joined_tokens = ''.join(t['text'] for t in tokens)
            # A clause owns a character span, and therefore its first and last
            # intersecting tokens. The preceding clause's end is not its start:
            # there may be several seconds of silence between the two.
            cursor = -(len(joined_tokens) - len(joined_tokens.lstrip()))
            token_spans = []
            for token in tokens:
                token_spans.append((cursor, cursor + len(token['text'])))
                cursor += len(token['text'])
            consumed = 0
            timed_parts = []
            previous_last = -1
            for part in parts:
                finish = consumed + len(part['source'])
                owned = [i for i, (left, right) in enumerate(token_spans)
                         if left < finish and right > consumed]
                first, last = owned[0], owned[-1]
                left = min(end, max(start, tokens[first]['start']))
                right = min(end, max(left, tokens[last]['end']))
                row = dict(source=part['source'], target=part['target'], start=left, end=right,
                           utterance=a.index, alignment='word', token_projection=projection, pause_layout=pause_plan)
                # A layout boundary inside one token has no independent time.
                # Keep both corresponding texts together, also when distinct
                # token intervals overlap, instead of trimming either clause.
                if timed_parts and (first <= previous_last or left < timed_parts[-1]['end']):
                    previous_part = timed_parts[-1]
                    previous_part['source'] += row['source']
                    previous_part['target'] += row['target']
                    previous_part['end'] = max(previous_part['end'], right)
                else:
                    timed_parts.append(row)
                previous_last = last
                consumed = finish
            rows.extend(timed_parts)
        else:
            boundaries = [start]; consumed = 0
            for part in parts[:-1]:
                consumed += len(part['source'])
                at = start + (end-start)*consumed/max(1,len(a.text))
                boundaries.append(min(end,max(boundaries[-1],at)))
            boundaries.append(end)
            for part, left, right in zip(parts,boundaries,boundaries[1:]):
                rows.append(dict(source=part['source'],target=part['target'],start=left,end=right,
                                 utterance=a.index,alignment='utterance',token_projection=projection,pause_layout=pause_plan))
    # Several clause boundaries can land on the same aligned word. Keep that
    # speech together instead of inventing a zero-length or flashing subtitle.
    i = 0
    while i < len(rows):
        row = rows[i]
        if row['end']-row['start'] < .4:
            if i and rows[i-1]['utterance'] == row['utterance']:
                previous_row = rows[i-1]
                previous_row['source'] += row['source']
                previous_row['target'] += row['target']
                previous_row['end'] = row['end']
                rows.pop(i)
                continue
            if i+1 < len(rows) and rows[i+1]['utterance'] == row['utterance']:
                following = rows[i+1]
                following['source'] = row['source'] + following['source']
                following['target'] = row['target'] + following['target']
                following['start'] = row['start']
                rows.pop(i)
                continue
        i += 1
    # A preserved cue must not hide an existing timing conflict. Check before
    # neighboring display holds could trim away evidence of that conflict.
    for i, row in enumerate(rows):
        if row.get('preserved_timestamp') and (
                (i and rows[i-1]['end'] > row['start'])
                or (i+1 < len(rows) and row['end'] > rows[i+1]['start'])):
            raise ValueError('Cannot preserve a fallback cue with overlapping display geometry')
    # Display hold uses free time around speech, never another cue's interval.
    previous = 0.0
    for i,row in enumerate(rows):
        if row.get('preserved_timestamp'):
            previous = row['end']
            continue
        next_start = rows[i+1]['start'] if i+1<len(rows) else row['end']+1
        desired = min(6.0,max(1.0,len(row['target'])/10))
        start = max(previous,row['start']-.08)
        end = min(next_start,max(row['end'],start+desired))
        if end-start < .4:
            start = max(previous,min(start,end-.8))
        if end <= start: raise ValueError(f'Display timing cannot preserve a positive interval: {row}')
        row.update(start=start,end=end)
        previous=end
    raw, final = [], []
    for i,row in enumerate(rows,1):
        ts = row.get('preserved_timestamp') or timestamp(row['start'],row['end'])
        raw.append(SrtBlock(i,ts,row['source'])); final.append(SrtBlock(i,ts,row['target']))
        row['line']=i
    return raw,final,rows
