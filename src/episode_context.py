"""Bounded, complete chosen-source context for opt-in draft translation only."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.translate import SrtBlock

VERSION = 'episode-source-context-1'
MAX_SOURCE_JSON_BYTES = 32768
RULE = """The following is read-only chosen-source context for the entire episode.
Use it for consistent names, references and discourse, without inventing speaker
identities or repairing the printed target sources. These episode_row values are
NOT output IDs. Translate ONLY the numbered target rows in the task below, with
its original output IDs. Never output or summarize this context document."""


def episode_source_prefix(source: list[SrtBlock]) -> str:
    """Serialize every original chosen row literally, without targets or evidence.

    A byte ceiling bounds resource use; it is not a token-count estimate. Smaller
    server contexts can still reject a request and must not receive sliced text.
    """
    if not source or [row.index for row in source] != list(range(1, len(source)+1)):
        raise ValueError('Episode context requires all sequential chosen-source rows')
    if any(not isinstance(row.text, str) or not row.text.strip() for row in source):
        raise ValueError('Episode context requires nonempty source text for every row')
    document = json.dumps([{'episode_row': row.index, 'source': row.text} for row in source],
                          ensure_ascii=False)
    size = len(document.encode('utf-8'))
    if size > MAX_SOURCE_JSON_BYTES:
        raise ValueError(f'Complete episode source context is {size} bytes, exceeding the '
                         f'{MAX_SOURCE_JSON_BYTES}-byte limit; no rows were truncated. '
                         'Disable --translation-episode-context for this input.')
    return RULE+'\nREAD-ONLY EPISODE SOURCE:\n'+document+'\nEND READ-ONLY CONTEXT. TARGET TASK:\n'
