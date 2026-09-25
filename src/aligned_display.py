"""Word-aligned subtitle presentation for window-level Japanese and Chinese.

Qwen3-ForcedAligner timings split each window's Japanese into display units; a
window's Chinese is partitioned onto those units without changing its text, and
cues are timed from the units. Only cue boundaries, cue timing and the Chinese
subtitle punctuation convention change: ``。，；：`` become spaces and every
other target character is kept in order.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import logging
from pathlib import Path
import re
import sys
import time
import wave

from src.asr_consensus import AudioWindow, aligned_tokens
from src.config import TranslateConfig
from src.pause_layout import INSTRUCTION as PARTITION_INSTRUCTION, lexical_count
from src.quality import _invalid_json_constant, _unique_json_object, mode
from src.translate import call_llm
from src.workflow_state import fingerprint, read_json, write_json

logger = logging.getLogger(__name__)
VERSION = 'aligned-display-2'
SENTENCE_END = '。？！?!'
CLOSERS = '」』）)”’'
CONVENTION_DROPPED = '。，；：'
CONTINUATIONS = ('て', 'で', 'に', 'を', 'が', 'は', 'と', 'も', 'の', 'だ', 'です', 'って', 'よ', 'ね',
                 'な', 'か', 'さ', 'わ', 'ん', 'ー', 'っ', 'ゃ', 'ゅ', 'ょ', 'ぁ', 'ぃ', 'ぅ', 'ぇ', 'ぉ')
SUB_PAUSE_SECONDS = 0.5
MIN_PAUSE_PHRASE = 3
ELLIPSIS_PAUSE_SECONDS = 0.3
MAX_UNIT_SECONDS = 7.0
SECONDS_PER_CHAR_FLOOR = 0.06      # faster than ~16.7 lexical chars/s is not plausible speech
QUANTIZATION_SECONDS = 0.1         # aligner boundaries are 80 ms bins
REPAIR_SECONDS_PER_CHAR = 0.15
MAX_IMPLAUSIBLE_SHARE = 0.5
MAX_MERGE = 4
GAP_SPLIT_SECONDS = 1.0
GAP_SPLIT_TOLERANCE = 0.25
MERGE_PENALTY = 0.05
PARTITION_ATTEMPTS = 2
MAX_CUE_CHARS = 20
MIN_SPLIT_CHARS = 4
READING_CPS = 9.0
LEAD_OUT_SECONDS = 0.3
MIN_CUE_SECONDS = 0.8
MAX_CUE_SECONDS = 7.0
MIN_ROOM_SECONDS = 0.2
CUE_GAP_SECONDS = 0.04


# ---------------------------------------------------------------- helpers

def _seconds(value: str) -> float:
    h, m, rest = value.strip().split(':'); s, ms = rest.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


# ------------------------------------------------------------ Japanese units

def owner_tokens(text: str, items: list[dict], start: float, end: float) -> list[dict]:
    """Map aligner items onto exact character spans with absolute times."""
    window = AudioWindow(index=0, start=start, end=end, core_start=start, core_end=end + 1.0)
    tokens = aligned_tokens(text, items, window)
    if not tokens or tokens[0]['begin'] != 0:
        raise ValueError('Alignment does not start at the first character')
    for left, right in zip(tokens, tokens[1:]):
        if left['finish'] != right['begin']:
            raise ValueError('Alignment is not contiguous')
    return tokens


def _implausible(item: dict) -> bool:
    chars = lexical_count(item['text'])
    return chars > 0 and item['end'] - item['start'] < SECONDS_PER_CHAR_FLOOR * chars - QUANTIZATION_SECONDS


def repair_timing(tokens: list[dict], start: float, end: float) -> tuple[list[dict], int]:
    """Re-time runs of physically implausible items (collapsed or zero-length).

    Applied to sentence units: the aligner's word tokens are 80 ms-quantized and
    often zero-length, so single-token durations are not a plausibility signal.

    A run is spread by character count over the gap between its reliable
    neighbours. When that gap is much longer than the run needs, a span of
    ``REPAIR_SECONDS_PER_CHAR`` per character is centred on where the aligner
    placed the run. Raises when most characters are implausible, so the caller
    falls back to proportional timing.
    """
    tokens = [dict(t) for t in tokens]
    bad = [_implausible(t) for t in tokens]
    total = sum(lexical_count(t['text']) for t in tokens) or 1
    if sum(lexical_count(t['text']) for t, b in zip(tokens, bad) if b) / total > MAX_IMPLAUSIBLE_SHARE:
        raise ValueError('Most aligned characters have implausible timing')
    i, repaired = 0, 0
    while i < len(tokens):
        if not bad[i]:
            i += 1; continue
        j = i
        while j + 1 < len(tokens) and bad[j + 1]:
            j += 1
        chars = sum(max(1, lexical_count(t['text'])) for t in tokens[i:j + 1])
        need = REPAIR_SECONDS_PER_CHAR * chars
        left = tokens[i - 1]['end'] if i else max(start, tokens[j + 1]['start'] - need if j + 1 < len(tokens) else start)
        right = tokens[j + 1]['start'] if j + 1 < len(tokens) else min(end, left + need)
        if right - left > 2 * need:
            centre = min(max((tokens[i]['start'] + tokens[j]['end']) / 2, left + need / 2), right - need / 2)
            left, right = centre - need / 2, centre + need / 2
        cursor = left
        for token in tokens[i:j + 1]:
            share = (right - left) * max(1, lexical_count(token['text'])) / chars
            token['start'], token['end'] = cursor, cursor + share
            cursor += share
        repaired += j - i + 1
        i = j + 1
    return tokens, repaired


def _unit(tokens: list[dict], first: int, last: int) -> dict:
    return {'first': first, 'last': last, 'begin': tokens[first]['begin'],
            'finish': tokens[last]['finish'], 'start': tokens[first]['start'],
            'end': max(t['end'] for t in tokens[first:last + 1])}


def _ends_sentence(token: dict, following: dict | None) -> bool:
    body = token['text'].rstrip(CLOSERS + ' ')
    if body and body[-1] in SENTENCE_END:
        return True
    return bool(body.endswith('…') and following is not None
                and following['start'] - token['end'] >= ELLIPSIS_PAUSE_SECONDS)


def _continues(token: dict) -> bool:
    """A token that begins with a particle/auxiliary continues the phrase before it."""
    return token['text'].lstrip('…、 ').startswith(CONTINUATIONS)


def _split_long(text: str, tokens: list[dict], first: int, last: int) -> list[tuple[int, int]]:
    unit = _unit(tokens, first, last)
    if unit['end'] - unit['start'] <= MAX_UNIT_SECONDS or first == last:
        return [(first, last)]
    candidates = [i for i in range(first, last)
                  if (tokens[i]['text'].rstrip(CLOSERS + ' ').endswith(('、', ',', '…'))
                      or tokens[i + 1]['start'] - tokens[i]['end'] >= SUB_PAUSE_SECONDS)
                  and not _continues(tokens[i + 1])
                  and lexical_count(text[tokens[first]['begin']:tokens[i]['finish']]) >= MIN_PAUSE_PHRASE
                  and lexical_count(text[tokens[i + 1]['begin']:tokens[last]['finish']]) >= MIN_PAUSE_PHRASE]
    if not candidates:
        return [(first, last)]
    middle = (unit['start'] + unit['end']) / 2
    cut = min(candidates, key=lambda i: abs(tokens[i]['end'] - middle))
    return _split_long(text, tokens, first, cut) + _split_long(text, tokens, cut + 1, last)


def japanese_units(text: str, tokens: list[dict]) -> list[dict]:
    """Sentence units (long ones split at 、/pauses, never before a particle);
    each keeps an exact character span of ``text``."""
    groups, first = [], 0
    for i, token in enumerate(tokens):
        following = tokens[i + 1] if i + 1 < len(tokens) else None
        if following is None or _ends_sentence(token, following):
            groups.extend(_split_long(text, tokens, first, i))
            first = i + 1
    merged = []
    for a, b in groups:
        unit = _unit(tokens, a, b)
        body = text[unit['begin']:unit['finish']]
        complete = body.rstrip(CLOSERS + ' ').endswith(tuple(SENTENCE_END + '…'))
        if merged and (not lexical_count(body) or (lexical_count(body) < 2 and not complete)):
            merged[-1].update(finish=unit['finish'], last=unit['last'], end=max(merged[-1]['end'], unit['end']))
        else:
            merged.append(unit)
    if len(merged) > 1 and not lexical_count(text[merged[0]['begin']:merged[0]['finish']]):
        head = merged.pop(0)
        merged[0].update(begin=head['begin'], first=head['first'], start=head['start'])
    merged[-1]['finish'] = len(text)  # unaligned trailing characters stay with the last unit
    for unit in merged:
        unit['text'] = text[unit['begin']:unit['finish']]
    return merged


def proportional_units(text: str, start: float, end: float) -> list[dict]:
    """Fallback when alignment fails: sentence spans timed by character share."""
    spans, begin = [], 0
    for match in re.finditer(rf'[{re.escape(SENTENCE_END)}]+[{re.escape(CLOSERS)}]*', text):
        if lexical_count(text[begin:match.end()]):
            spans.append((begin, match.end())); begin = match.end()
    if begin < len(text):
        if spans and not lexical_count(text[begin:]):
            spans[-1] = (spans[-1][0], len(text))
        else:
            spans.append((begin, len(text)))
    weights = [max(1, lexical_count(text[a:b])) for a, b in spans]
    total, cursor, units = sum(weights), start, []
    for (a, b), weight in zip(spans, weights):
        finish = cursor + (end - start) * weight / total
        units.append({'begin': a, 'finish': b, 'start': cursor, 'end': finish, 'text': text[a:b]})
        cursor = finish
    return units


# ---------------------------------------------------------- Chinese pieces

_TERMINATOR = re.compile(rf'(?:…*[{re.escape(SENTENCE_END)}]+|…+(?![{re.escape(SENTENCE_END)}]))'
                         rf'[{re.escape(CLOSERS)}]*')


def _sentences(target: str) -> list[str]:
    """Split after sentence terminators; an ellipsis after 。 opens the next one."""
    parts, begin = [], 0
    for match in _TERMINATOR.finditer(target):
        if lexical_count(target[begin:match.end()]):
            parts.append(target[begin:match.end()]); begin = match.end()
    if begin < len(target):
        if parts and not lexical_count(target[begin:]):
            parts[-1] += target[begin:]
        else:
            parts.append(target[begin:])
    return parts


def monotone_partition(units: list[dict], target: str) -> list[dict]:
    """Deterministic fallback: monotone DP alignment of Japanese units to Chinese
    sentences by length share; 1:1 when the counts match. Every piece covers at
    least one unit and one sentence; pieces prefer fewer merged items."""
    sentences = _sentences(target) or [target]
    n, m = len(units), len(sentences)
    if n == m:
        return [{'units': [i, i], 'target': s} for i, s in enumerate(sentences)]
    jw = [max(1, lexical_count(u['text'])) for u in units]
    zw = [max(1, lexical_count(s)) for s in sentences]
    jt, zt = sum(jw), sum(zw)
    inf = float('inf')
    best = [[inf] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            for a in range(max(0, i - MAX_MERGE), i):
                for b in range(max(0, j - MAX_MERGE), j):
                    if best[a][b] == inf:
                        continue
                    cost = (best[a][b] + abs(sum(jw[a:i]) / jt - sum(zw[b:j]) / zt)
                            + MERGE_PENALTY * ((i - a - 1) + (j - b - 1)))
                    if cost < best[i][j]:
                        best[i][j], back[i][j] = cost, (a, b)
    if best[n][m] == inf:
        return [{'units': [0, n - 1], 'target': target}]
    pieces, i, j = [], n, m
    while i:
        a, b = back[i][j]
        pieces.append({'units': [a, i - 1], 'target': ''.join(sentences[b:j])})
        i, j = a, b
    return pieces[::-1]


def _parse_targets(answer: str, count: int, target: str) -> list[str] | None:
    try:
        row = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', answer.strip()),
                         object_pairs_hook=_unique_json_object, parse_constant=_invalid_json_constant)
        values = row['targets']
        if (not isinstance(row, dict) or set(row) != {'targets'} or not isinstance(values, list)
                or len(values) != count or any(not isinstance(v, str) or not lexical_count(v) for v in values)
                or ''.join(values) != target):
            return None
        return values
    except (ValueError, TypeError, KeyError):
        return None


def partition_target(units: list[dict], target: str, config: TranslateConfig,
                     cache: Path) -> tuple[list[dict], str]:
    """Exact-text Chinese partition matching the Japanese units (layout only).

    Model answers are cached per request; transport failures are not cached."""
    if len(units) == 1:
        return [{'units': [0, 0], 'target': target}], 'single_unit'
    body = json.dumps({'fixed_source_parts': [u['text'] for u in units], 'existing_target': target},
                      ensure_ascii=False)
    cfg = mode(config, 'aligned-display-partition')
    extra = dict(cfg.extra_payload or {}); extra['max_tokens'] = min(config.max_tokens, 2048)
    cfg = replace(cfg, max_tokens=min(config.max_tokens, 2048), extra_payload=extra,
                  response_guard_floor=max(config.response_guard_floor, 512))
    key = fingerprint([VERSION, PARTITION_INSTRUCTION, body, cfg.endpoint, cfg.extra_payload])
    saved = read_json(cache, {})
    saved = saved if isinstance(saved, dict) else {}
    answers = list(saved.get(key, [])) if isinstance(saved.get(key), list) else []
    errors = []
    for attempt in range(PARTITION_ATTEMPTS):
        if attempt >= len(answers):
            try:
                answers.append(call_llm(body, PARTITION_INSTRUCTION, cfg))
            except Exception as error:  # noqa: BLE001 - layout is optional; record and fall back
                errors.append(f'{type(error).__name__}: {error}')
                break
            saved[key] = answers; write_json(cache, saved)
        values = _parse_targets(answers[attempt], len(units), target)
        if values is not None:
            return [{'units': [i, i], 'target': v} for i, v in enumerate(values)], f'model_attempt_{attempt + 1}'
    return monotone_partition(units, target), 'fallback_transport_error' if errors else 'fallback_dp'


_SPLIT_AT = re.compile(rf'(?:…*[，。；：！？、]+|…+(?![，。；：！？、…]))[{re.escape(CLOSERS)}]*')


def split_piece(text: str, start: float, end: float) -> list[dict]:
    """Split an over-long Chinese piece at internal punctuation, timing by share."""
    if len(convention(text)) <= MAX_CUE_CHARS:
        return [{'text': text, 'start': start, 'end': end}]
    candidates = [m.end() for m in _SPLIT_AT.finditer(text)
                  if lexical_count(text[:m.end()]) >= MIN_SPLIT_CHARS
                  and lexical_count(text[m.end():]) >= MIN_SPLIT_CHARS]
    if not candidates:
        return [{'text': text, 'start': start, 'end': end}]
    total = lexical_count(text)
    cut = min(candidates, key=lambda i: abs(lexical_count(text[:i]) - total / 2))
    middle = start + (end - start) * lexical_count(text[:cut]) / total
    return split_piece(text[:cut], start, middle) + split_piece(text[cut:], middle, end)


def _speech_time(spans: list[tuple[float, float]], share: float, opening: bool) -> float:
    """Map a character share onto speech time only (silent gaps get no time).
    An opening boundary that lands exactly on a gap snaps to the next span."""
    total = sum(b - a for a, b in spans) or 1e-9
    target, run = share * total, 0.0
    for index, (a, b) in enumerate(spans):
        length = b - a
        if target < run + length - 1e-9 or (not opening and target <= run + length + 1e-9) \
                or index == len(spans) - 1:
            return min(b, a + max(0.0, target - run))
        run += length
    return spans[-1][1]


def place_piece(text: str, units: list[dict]) -> list[dict]:
    """Time a Chinese piece covering one or more Japanese units.

    At each gap of ``GAP_SPLIT_SECONDS`` or more between units the text is cut
    at the punctuation nearest that gap's speech-time share, so no cue bridges
    a long silence when the text allows it. Each part is split further when
    long, and every boundary is timed on speech time rather than wall time."""
    spans = [(u['start'], u['end']) for u in units]
    total = sum(b - a for a, b in spans) or 1e-9
    positions = sorted({m.end() for m in _SPLIT_AT.finditer(text)
                        if lexical_count(text[:m.end()]) and lexical_count(text[m.end():])})
    chars = lexical_count(text) or 1
    cuts, run, last = [], 0.0, 0
    for k in range(len(units) - 1):
        run += spans[k][1] - spans[k][0]
        if spans[k + 1][0] - spans[k][1] < GAP_SPLIT_SECONDS:
            continue
        options = [p for p in positions if p > last]
        if not options:
            break
        best = min(options, key=lambda p: abs(lexical_count(text[:p]) / chars - run / total))
        if abs(lexical_count(text[:best]) / chars - run / total) <= GAP_SPLIT_TOLERANCE:
            cuts.append((best, k)); last = best
    pieces, begin, first = [], 0, 0
    for position, k in cuts + [(len(text), len(units) - 1)]:
        part_spans = spans[first:k + 1]
        part_text = text[begin:position]
        if part_text:
            for sub in split_piece(part_text, 0.0, 1.0):  # shares of this part's speech time
                pieces.append({'text': sub['text'],
                               'start': _speech_time(part_spans, sub['start'], True),
                               'end': _speech_time(part_spans, sub['end'], False)})
        begin, first = position, k + 1
    return pieces


def convention(text: str) -> str:
    """Chinese subtitle punctuation: ``。，；：`` next to an ellipsis are dropped,
    the rest become spaces; whitespace is collapsed and edges stripped."""
    text = re.sub(rf'[{CONVENTION_DROPPED}]+(?=…)|(?<=…)[{CONVENTION_DROPPED}]+', '', text)
    text = re.sub(f'[{CONVENTION_DROPPED}]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def minimum_seconds(text: str) -> float:
    return max(MIN_CUE_SECONDS, len(convention(text)) / READING_CPS)


def finalize_timing(cues: list[dict], media_end: float = float('inf')) -> list[dict]:
    """Strict non-overlap, reading-speed minimum, lead-out, then a display cap.

    Minimums are applied before lead-outs so a lead-out never starves the next
    cue. Where a minimum cannot fit between neighbours the cue keeps the room
    available (recorded by callers as ``below_minimum``)."""
    cues = sorted((dict(c) for c in cues), key=lambda c: (c['start'], c['end']))
    for i in range(1, len(cues)):  # resolve raw overlaps
        previous, cue = cues[i - 1], cues[i]
        if cue['start'] < previous['end'] + CUE_GAP_SECONDS:
            previous['end'] = max(previous['start'] + MIN_ROOM_SECONDS, cue['start'] - CUE_GAP_SECONDS)
            cue['start'] = max(cue['start'], previous['end'] + CUE_GAP_SECONDS)
            cue['end'] = max(cue['end'], cue['start'] + MIN_ROOM_SECONDS)

    def ceiling(i: int) -> float:
        return cues[i + 1]['start'] - CUE_GAP_SECONDS if i + 1 < len(cues) else media_end

    for i, cue in enumerate(cues):  # reading-speed minimum
        need = minimum_seconds(cue['text'])
        cue['end'] = max(cue['end'], min(cue['start'] + need, ceiling(i)))
        if cue['end'] - cue['start'] < need:
            floor = cues[i - 1]['end'] + CUE_GAP_SECONDS if i else 0.0
            cue['start'] = max(floor, min(cue['start'], cue['end'] - need))
    for i, cue in enumerate(cues):  # lead-out only into free time; no cue lingers
        cue['end'] = max(cue['end'], min(cue['end'] + LEAD_OUT_SECONDS, ceiling(i)))
        cue['capped'] = cue['end'] - cue['start'] > max(MAX_CUE_SECONDS, minimum_seconds(cue['text']))
        if cue['capped']:
            cue['end'] = cue['start'] + max(MAX_CUE_SECONDS, minimum_seconds(cue['text']))
    for i, cue in enumerate(cues):
        if not cue['end'] > cue['start'] or (i and cue['start'] < cues[i - 1]['end']):
            raise ValueError(f'Cannot place cue without overlap: {cue}')
        cue['below_minimum'] = cue['end'] - cue['start'] + 1e-6 < minimum_seconds(cue['text'])
    return cues


def preserved(original: str, final: str) -> bool:
    """Every character except the dropped punctuation and whitespace, in order."""
    keep = lambda s: re.sub(rf'[{CONVENTION_DROPPED}\s]', '', s)  # noqa: E731
    return keep(original) == keep(final)


# ------------------------------------------------------------ aligner worker

def _read_wav(path: Path):
    import numpy as np
    with wave.open(str(path), 'rb') as stream:
        if stream.getframerate() != 16000 or stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError(f'Expected 16 kHz mono PCM16: {path}')
        audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)
    return audio.astype(np.float32) / 32768.0


def _worker(spec: dict) -> dict:
    import torch
    from qwen_asr import Qwen3ForcedAligner
    from src.warden import ensure_gpu_headroom
    started = time.monotonic()
    result = {'version': VERSION, 'aligner_model': spec['aligner_model'], 'rows': {}}
    free = ensure_gpu_headroom(required_gb=3.0, hard_floor_gb=2.0, admin_url=spec['warden_admin_url'],
                               enabled=True, caller='aligned display')
    result['free_gb_before_load'] = free
    torch.cuda.reset_peak_memory_stats()
    model = Qwen3ForcedAligner.from_pretrained(spec['aligner_model'], dtype=torch.bfloat16,
                                              device_map='cuda', local_files_only=True)
    for job in spec['jobs']:
        row = {'text': job['text'], 'wav': job['wav']}
        before = time.monotonic()
        try:
            audio = _read_wav(Path(job['wav']))
            with torch.inference_mode():
                aligned = model.align(audio=[(audio, 16000)], text=[job['text']], language='Japanese')
            row['items'] = [asdict(item) for item in aligned[0].items]
        except Exception as error:  # noqa: BLE001 - per-window failure is recorded, not fatal
            row['error'] = f'{type(error).__name__}: {error}'
        row['seconds'] = time.monotonic() - before
        result['rows'][str(job['owner'])] = row
    result['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
    result['seconds'] = time.monotonic() - started
    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    if len(sys.argv) != 4 or sys.argv[1] != '--align':
        raise SystemExit('Usage: python -m src.aligned_display --align spec.json result.json')
    write_json(Path(sys.argv[3]), _worker(json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))))

