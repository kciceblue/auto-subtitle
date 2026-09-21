"""Conservative pause boundaries before translation, using local timing evidence."""
from __future__ import annotations

from copy import deepcopy

from src.pause_layout import find_pause_cuts

VERSION = 'source-pause-units-1'


def pause_token_groups(tokens: list[dict], speech_regions: list[dict],
                       sample_rate: int) -> tuple[list[list[dict]], list[dict]]:
    """Split only at verified token boundaries; preserve every token and its time.

    A pause is a translation-unit hint, not a claim of punctuation, speaker
    identity, or acoustic truth. Reuse the conservative VAD/alignment guard,
    then reject boundaries that create a sub-400ms unit or lose whitespace when
    the ASR writer strips a unit. Invalid evidence leaves the unit intact.
    """
    original = deepcopy(tokens)
    if not tokens:
        return [original], []
    source = ''.join(token['text'] for token in tokens).strip()
    cuts = find_pause_cuts(source, tokens, speech_regions, sample_rate)
    if not cuts:
        return [original], []
    joined = ''.join(token['text'] for token in tokens)
    cursor = -(len(joined) - len(joined.lstrip()))
    boundaries = {}
    for index, token in enumerate(tokens[:-1], 1):
        cursor += len(token['text'])
        boundaries[cursor] = index
    # Accept each boundary only if the preceding and remaining groups can
    # preserve their exact text and positive, readable source interval.
    selected = []
    left = 0
    for cut in cuts:
        right = boundaries.get(cut['offset'])
        if right is None:
            continue
        before, after = tokens[left:right], tokens[right:]
        if (not before or not after or before[-1]['end'] - before[0]['start'] < .4
                or after[-1]['end'] - after[0]['start'] < .4):
            continue
        prefix = ''.join(t['text'] for t in tokens[:right])
        suffix = ''.join(t['text'] for t in after)
        if prefix.strip() + suffix.strip() != source:
            continue
        selected.append(cut)
        left = right
    if not selected:
        return [original], []
    indices = [0] + [boundaries[cut['offset']] for cut in selected] + [len(tokens)]
    groups = [deepcopy(tokens[a:b]) for a, b in zip(indices, indices[1:])]
    if ''.join(''.join(t['text'] for t in group).strip() for group in groups) != source:
        return [original], []
    return groups, deepcopy(selected)
