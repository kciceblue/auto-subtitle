"""Optional pause-driven display partition; never changes source/target wording."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

from src.config import TranslateConfig
from src.quality import mode, _unique_json_object, _invalid_json_constant
from src.translate import call_llm
from src.workflow_state import fingerprint, read_json, write_json

VERSION = 'pause-layout-1'
INSTRUCTION = '''Partition the existing Chinese subtitle for the supplied fixed Japanese source pieces.
This is layout only. Never translate, repair, paraphrase, add punctuation, drop text, change
word order, assign speakers or fill missing content. The concatenated targets must equal
EXACTLY the given existing target, including every punctuation/space. Give one nonempty
readable target piece per source piece, in order, corresponding to that source meaning.
Return only JSON with targets: an array of strings. If exact meaningful partition is not
possible, return targets: []; the client keeps the previous layout. A source gap is only a
local timing hint, not proof of silence, a sentence boundary or a different speaker.'''


def lexical_count(text: str) -> int:
    return sum(c.isalnum() for c in text)


def find_pause_cuts(source: str, tokens: list[dict], speech_regions: list[dict],
                    sample_rate: int, *, minimum_gap: float = 2.0) -> list[dict]:
    """Return conservative exact character cuts using aligned gaps and saved VAD."""
    if (not tokens or not speech_regions or type(sample_rate) is not int or sample_rate <= 0
            or not isinstance(minimum_gap, (int, float)) or not math.isfinite(minimum_gap)
            or minimum_gap < 1.5):
        return []
    try:
        joined = ''.join(t['text'] for t in tokens)
        if joined.strip() != source:
            return []
        for token in tokens:
            a, b = token['start'], token['end']
            if (any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t)
                    for t in (a,b)) or not 0 <= a <= b):
                return []
        if any(a['end'] > b['start'] + .001 for a,b in zip(tokens,tokens[1:])):
            return []
        speech = []
        for region in speech_regions:
            a,b = region['start']/sample_rate, region['end']/sample_rate
            if not all(math.isfinite(x) for x in (a,b)) or not 0 <= a < b:
                return []
            speech.append((a,b))
        merged = []
        for a,b in sorted(speech):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(b,merged[-1][1])
            else:
                merged.append([a,b])
        cuts, cursor, previous = [], -(len(joined)-len(joined.lstrip())), 0
        for i,token in enumerate(tokens[:-1]):
            cursor += len(token['text'])
            following = tokens[i+1]
            start,end = token['end'],following['start']
            gap = end-start
            if (gap < minimum_gap or token['end'] <= token['start']
                    or following['end'] <= following['start'] or not previous < cursor < len(source)):
                continue
            fraction = sum(max(0.,min(end,b)-max(start,a)) for a,b in merged)/gap
            if fraction > .2 or min(lexical_count(source[previous:cursor]),lexical_count(source[cursor:])) < 2:
                continue
            cuts.append({'offset':cursor,'gap_start':start,'gap_end':end,
                         'gap_seconds':gap,'vad_speech_fraction':fraction})
            previous = cursor
            if len(cuts) == 11:
                break
        return cuts
    except (KeyError,TypeError,ValueError):
        return []


def _parse_parts(answer: str, source: str, target: str, offsets: list[int]) -> list[dict] | None:
    try:
        row = json.loads(answer,object_pairs_hook=_unique_json_object,parse_constant=_invalid_json_constant)
        values = row['targets']
        if (not isinstance(row,dict) or set(row) != {'targets'} or not isinstance(values,list)
                or len(values) != len(offsets)-1 or any(not isinstance(v,str) or not lexical_count(v) for v in values)
                or ''.join(values) != target or any(c in target for c in '⟦⟧')):
            return None
        return [{'source':source[a:b],'target':value} for a,b,value in zip(offsets,offsets[1:],values)]
    except (ValueError,TypeError,KeyError):
        return None


def plan_pause_parts(source: str, target: str, cuts: list[dict], config: TranslateConfig,
                     cache: Path) -> list[dict] | None:
    """Ask the local writer to split target text only; invalid plans fall back."""
    offsets = [0]+[r['offset'] for r in cuts]+[len(source)]
    if (not cuts or len(cuts)>11 or any(type(i) is not int for i in offsets)
            or any(a >= b for a,b in zip(offsets,offsets[1:]))
            or any(lexical_count(source[a:b])<2 for a,b in zip(offsets,offsets[1:]))):
        return None
    body = json.dumps({'fixed_source_parts':[source[a:b] for a,b in zip(offsets,offsets[1:])],
                       'existing_target':target},ensure_ascii=False)
    cfg = mode(config,'pause-display-layout')
    extra = dict(cfg.extra_payload or {})
    extra['max_tokens'] = min(config.max_tokens,2048)
    cfg = replace(cfg,max_tokens=min(config.max_tokens,2048),extra_payload=extra,
                  response_guard_floor=max(config.response_guard_floor,512))
    key = fingerprint([VERSION,INSTRUCTION,body,cfg.endpoint,cfg.extra_payload])
    saved = read_json(cache,{})
    saved = saved if isinstance(saved,dict) else {}
    if isinstance(saved.get(key),str):
        parts = _parse_parts(saved[key],source,target,offsets)
        if parts is not None:
            return parts
    answer = call_llm(body,INSTRUCTION,cfg)
    parts = _parse_parts(answer,source,target,offsets)
    if parts is not None:
        saved[key] = answer
        write_json(cache,saved)
    return parts
