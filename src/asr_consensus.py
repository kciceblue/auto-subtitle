"""Audio-window consensus and subtitle construction; no model loading here."""
from __future__ import annotations

import re
import math
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache


@dataclass(frozen=True)
class AudioWindow:
    index: int
    start: float
    end: float
    core_start: float
    core_end: float


def audio_windows(duration: float, speech: list[dict], sr: int = 16000,
                  maximum: float = 20.0, overlap: float = 0.8) -> list[AudioWindow]:
    """Cover the original timeline, using silence as a boundary hint, never a mask."""
    if duration <= 0 or maximum <= 2 * overlap or overlap < 0:
        raise ValueError("Invalid audio/window duration")
    gaps = [(a['end'] + b['start']) / (2 * sr) for a, b in zip(speech, speech[1:])
            if b['start'] > a['end']]
    boundaries = [0.0]
    while duration - boundaries[-1] > maximum:
        start = boundaries[-1]
        possible = [p for p in gaps if start + maximum * .65 <= p <= start + maximum]
        boundaries.append(max(possible) if possible else start + maximum)
    boundaries.append(duration)
    return [AudioWindow(i, max(0, a - overlap), min(duration, b + overlap), a, b)
            for i, (a, b) in enumerate(zip(boundaries, boundaries[1:]))]


def compact(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKC', text).lower()
                   if c.isalnum() or c == 'ー')


@lru_cache(maxsize=4096)
def reading_map(text: str) -> tuple[str, tuple[tuple[int, int], ...]]:
    from src.adjudicate import _get_kakasi
    result, positions, cursor = [], [], 0
    for item in _get_kakasi().convert(text):
        original = item['orig']
        start = text.find(original, cursor)
        if start < 0:
            start = cursor
        end = start + len(original)
        reading = compact(item['hira'])
        result.extend(reading)
        # Kakasi groups long kana runs. Map matching kana one character at a
        # time, otherwise punctuation at one end moves into the next sentence.
        original_reading = ''.join(chr(ord(c)-0x60) if 'ァ' <= c <= 'ヶ' else c
                                   for c in unicodedata.normalize('NFKC', original).lower())
        original_chars = [(i,c) for i,c in enumerate(original_reading) if c.isalnum() or c == 'ー']
        compact_original = ''.join(c for _,c in original_chars)
        mapped = [(start,end)] * len(reading)
        for tag,a0,a1,b0,b1 in SequenceMatcher(None,compact_original,reading,autojunk=False).get_opcodes():
            if tag == 'equal':
                for k in range(b1-b0):
                    pos = start + original_chars[a0+k][0]
                    mapped[b0+k] = (pos,pos+1)
            elif a1 > a0:
                span = (start+original_chars[a0][0],start+original_chars[a1-1][0]+1)
                mapped[b0:b1] = [span] * (b1-b0)
        positions.extend(mapped)
        cursor = end
    return ''.join(result), tuple(positions)


def similarity(a: str, b: str) -> float:
    x, y = reading_map(a)[0], reading_map(b)[0]
    return SequenceMatcher(None, x, y, autojunk=False).ratio() if x and y else 0.0


def suspicious_text(text: str) -> bool:
    return bool(re.search(r'(.{2,12})\1{3,}|作詞.{0,15}作曲|ご視聴.{0,8}ありがとう', text))


def choose_transcript(candidates: dict[str, str]) -> tuple[str, float]:
    """One vote per recognizer family; alternate audio is never an extra voter."""
    available = {k: v for k, v in candidates.items() if v.strip()}
    if not available:
        return 'q', 0.0
    scores = {}
    for name, text in available.items():
        others = [similarity(text, other) for key, other in available.items() if key != name]
        scores[name] = (.7 * max(others) + .3 * sum(others) / len(others)) if others else 0.0
        if suspicious_text(text):
            scores[name] -= .4
    best = max(scores, key=scores.get)
    # Two recognizers can both drop speech. A shared short tail is not evidence
    # that a third recognizer's preceding dialogue should be deleted.
    short = reading_map(available[best])[0]
    for name in sorted(available, key=lambda k: len(reading_map(available[k])[0]), reverse=True):
        longer = reading_map(available[name])[0]
        if (len(short) >= .8 * len(longer) or len(longer)-len(short) < 12
                or suspicious_text(available[name])):
            continue
        matched = sum(m.size for m in SequenceMatcher(None,short,longer,autojunk=False).get_matching_blocks())
        if short and matched/len(short) >= .8:
            return name, min(.6,max(0.0,scores[name]))  # Retained speech still needs review.
    return best, max(0.0, scores[best])


def project_phrase(full: str, other: str, begin: int, end: int) -> str:
    """Map a phrase through phonetic alignment to the other recognizer's spelling."""
    a, amap = reading_map(full)
    b, bmap = reading_map(other)
    indices = [i for i, (s, e) in enumerate(amap) if e > begin and s < end]
    if not indices or not bmap:
        return ''
    lo, hi = indices[0], indices[-1] + 1
    projected = []
    for tag, a0, a1, b0, b1 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if a1 <= lo or a0 >= hi or b1 == b0:
            continue
        if tag == 'equal':
            projected.extend(range(b0 + max(lo, a0) - a0, b0 + min(hi, a1) - a0))
        elif tag == 'replace':
            # Keep the complete disputed phrase; do not invent partial words.
            projected.extend(range(b0, b1))
    if not projected:
        return ''
    return other[bmap[min(projected)][0]:bmap[max(projected)][1]].strip()


def aligned_tokens(text: str, items: list[dict], window: AudioWindow, *,
                   require_complete: bool = False) -> list[dict]:
    """Restore punctuation and original offsets; retain only each window's owned words."""
    if require_complete and compact(text) != compact(''.join(item['text'] for item in items)):
        raise ValueError("Forced alignment does not cover the complete selected transcript")
    tokens, cursor = [], 0
    previous_start = -1.0
    for item in items:
        word = item['text'].strip()
        if not word:
            continue
        at = text.find(word, cursor)
        if at < 0:
            # Aligner tokenization ignores punctuation. A word such as ちゃん
            # may span a recognizer's misplaced ちゃ。ん boundary; preserve the
            # original characters instead of falling back to a shorter transcript.
            normalized, original_positions = [], []
            for offset,char in enumerate(text[cursor:],cursor):
                letters = compact(char)
                normalized.extend(letters)
                original_positions.extend([offset]*len(letters))
            needle = compact(word)
            match = ''.join(normalized).find(needle) if needle else -1
            if match < 0:
                raise ValueError(f"Aligner token absent from transcript: {word!r}")
            at = original_positions[match]
            end = original_positions[match+len(needle)-1]+1
        else:
            end = at + len(word)
        while end < len(text) and text[end] in ' 。、，,.！？!?…」』）)':
            end += 1
        raw_start = window.start + float(item['start_time'])
        raw_end = window.start + float(item['end_time'])
        # The aligner quantizes to 80 ms. A final boundary can land one bin
        # beyond the clipped waveform; clip only this bounded rounding error.
        if raw_start > window.end + .081 or raw_end > window.end + .081:
            raise ValueError(f"Alignment outside waveform: {item}; window={window.index}")
        start_time = min(window.end, max(window.start, raw_start))
        end_time = min(window.end, max(window.start, raw_end))
        if (not math.isfinite(start_time) or not math.isfinite(end_time)
                or end_time < start_time or start_time < previous_start):
            raise ValueError(f"Invalid alignment for {word!r}: {item}; window={window.index}")
        previous_start = start_time
        middle = (start_time + end_time) / 2
        if window.core_start <= middle < window.core_end:
            tokens.append(dict(text=text[cursor:end].strip(), start=start_time,
                               end=end_time, begin=cursor, finish=end))
        cursor = end
    return tokens


def subtitle_groups(tokens: list[dict], max_seconds: float = 6.0,
                    max_chars: int = 38) -> list[list[dict]]:
    groups, current = [], []
    for token in tokens:
        trailing_particle = compact(token['text']) in {'って', 'よ', 'ね', 'の', 'か', 'さ', 'と', 'が', 'を', 'に', 'で'}
        if current and ((token['start'] - current[-1]['end'] >= .55 and not trailing_particle)
                        or token['end'] - current[0]['start'] > max_seconds
                        or sum(len(t['text']) for t in current) + len(token['text']) > max_chars):
            groups.append(current)
            current = []
        current.append(token)
        if re.search(r'[。！？!?][」』）)]?$', token['text']):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def restore_sentence_boundaries(text: str, template: str) -> str:
    """Transfer only sentence punctuation through phonetic alignment, never words."""
    if text == template or not template:
        return text
    a, amap = reading_map(text)
    b, bmap = reading_map(template)
    if not a or not b:
        return text
    boundaries = set()
    opcodes = SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    for match in re.finditer(r'[。！？!?]', template):
        previous = [j for j, (_, end) in enumerate(bmap) if end <= match.start()]
        if not previous:
            continue
        j = previous[-1]
        for tag, a0, a1, b0, b1 in opcodes:
            if b0 <= j < b1:
                if tag == 'equal':
                    boundaries.add(amap[a0+j-b0][1])
                elif tag == 'replace' and a1 > a0 and j == b1-1:
                    boundaries.add(amap[a1-1][1])
                break
    result = []
    for i, char in enumerate(text, 1):
        result.append(char)
        if i in boundaries and char not in '。！？!?' and (i == len(text) or text[i] not in '。！？!?'):
            result.append('。')
    return re.sub(r'([。！？!?])[、，,]+', r'\1', ''.join(result))


def utterance_groups(tokens: list[dict]) -> list[list[dict]]:
    """Keep grammatical speech units intact; display timing is a later concern."""
    groups, current = [], []
    for token in tokens:
        current.append(token)
        if re.search(r'[。！？!?][」』）)]?$', token['text']):
            text = ''.join(t['text'] for t in current)
            # A recognizer's punctuation is not permission to strand a suffix.
            unfinished = re.search(r'(?:くらいなら|ことを|ことが|じゃ|では|来ん|意味|見たこと)[。！？!?]+$', text)
            if not unfinished:
                groups.append(current)
                current = []
    if current:
        groups.append(current)
    return groups


def semantic_markers(text: str) -> tuple:
    normalized = unicodedata.normalize('NFKC', text)
    return (tuple(re.findall(r'\d+(?:\.\d+)?', normalized)),
            tuple(re.findall(r'ません|ない|無い|なかった|なければ|なく|ずに', normalized)))


def phrase_grade(chosen: str, candidates: dict[str, str], selected: str) -> str:
    supporters = [k for k, v in candidates.items() if v and similarity(chosen, v) >= .92
                  and semantic_markers(chosen) == semantic_markers(v)]
    if len(supporters) < 2:
        return 'C'
    if all(compact(v) == compact(chosen) for v in candidates.values() if v) and len(supporters) == 3:
        return 'A'
    # Kanji-only disagreements can be real homophones, not harmless punctuation.
    kanji = lambda x: ''.join(re.findall(r'[\u4e00-\u9fff]', x))
    if any(kanji(v) and kanji(chosen) and kanji(v) != kanji(chosen)
           for k, v in candidates.items() if k in supporters):
        return 'E-音'
    return 'E'
