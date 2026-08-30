"""Align a 台词台本 (dialogue script) with source SRT text — 0 LLM tokens.

The script is ground truth: it *is* what was written, so its confidence
outranks any ASR/audio arbitration. This module turns a raw script file
into per-SRT-line anchors the review pass can consume.

Pipeline:
  clean_script_text(): strip speaker prefixes, drop stage-direction lines,
      split into a sentence stream
  align_script(): monotonic greedy matching of source SRT lines against the
      script stream using character-bigram Dice similarity

Output per source line:
  status "high"     — script sentence matches closely (source text可信)
  status "mismatch" — a script counterpart exists but wording differs
                      (strong correction signal: ASR likely garbled it)
  status "none"     — no script counterpart (line not in script, e.g. ASR
                      hallucination or script omission; leave to audio path)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.translate import read_context_file

logger = logging.getLogger(__name__)

# Speaker prefixes: "A：" / "りり：" / "ナレーション:" / "声1:" — up to 15 chars
# before the colon, no whitespace inside the name (Japanese names don't break).
_SPEAKER_RE = re.compile(r"^\s*([^\s：:]{1,15})\s*[：:]\s*")
# Stage-direction-only lines: （…）, 【…】, 〔…〕, （声） etc. as the whole line.
_STAGE_ONLY_RE = re.compile(r"^\s*[（(【〔].{0,40}[）)】〕]\s*$")

# Similarity threshold (character-bigram Dice).
_TH_MATCH = 0.60     # best-match score required to consider a counterpart
# Above this score a matched line is treated as 表記等价 (particles/尾音/
# punctuation differences) → "high"; only real wording differences below
# this are "mismatch" (correction signal). Keep テニス/デニス (1 kana slip
# on a short line ≈ 0.75) well below it.
_TH_EQUIV = 0.92


def _norm(s: str) -> str:
    """Normalize for exact-equality check (strip whitespace/punctuation)."""
    return re.sub(r"[\s。．、，,！？!?…～〜「」『』()（）]", "", s)


def _kana_norm(s: str) -> str:
    """kana_normalize + strip speech-decoration characters (♪〜ｗ♡…).

    Matching must compare readings, not decorations: trailing 〜♪ｗ are
    emphasis/playfulness, never a wording difference.
    """
    from src.adjudicate import kana_normalize

    out = kana_normalize(s)
    return re.sub(r"[♪♡♠★☆ｗＷwW・＝=]", "", out)


@dataclass
class ScriptAnchor:
    script_text: str          # matched script sentence(s), cleaned
    similarity: float         # best Dice score
    status: str               # "high" | "mismatch" | "none"


# ── cleaning ────────────────────────────────────────────────────────────────

def clean_script_text(text: str) -> list[str]:
    """Raw script → list of clean spoken-line sentences."""
    sentences: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _STAGE_ONLY_RE.match(line):
            continue  # （舞台指示）/【SE】整行
        line = _SPEAKER_RE.sub("", line)
        if not line.strip():
            continue
        # Split on sentence terminators, keep terminator attached to its
        # sentence ('.+?' lazy up to a run of terminators; no-terminator
        # fragments via the alternation tail).
        for part in re.findall(r".+?[。！？!?…]+|.+", line):
            part = part.strip()
            if not part:
                continue
            if _STAGE_ONLY_RE.match(part):
                continue
            sentences.append(part)
    return sentences


# ── similarity ──────────────────────────────────────────────────────────────

def _bigrams(s: str) -> list[str]:
    s = re.sub(r"\s+", "", s)
    return [s[i:i + 2] for i in range(max(0, len(s) - 1))]


def _dice(a: str, b: str) -> float:
    ba, bb = _bigrams(a), _bigrams(b)
    if not ba or not bb:
        return 0.0
    from collections import Counter

    ca, cb = Counter(ba), Counter(bb)
    inter = sum((ca & cb).values())
    return 2.0 * inter / (len(ba) + len(bb))


# ── alignment ───────────────────────────────────────────────────────────────

def align_script(
    source_lines: list[str],
    script_sentences: list[str],
) -> list[ScriptAnchor]:
    """Match each source SRT line to a script sentence.

    Two-phase, monotonicity-tolerant:
      1. global best per line (the SRT covers only part of the work's full
         script, so a pointer that starts at 0 would never reach the track's
         segment — global search finds every line's best sentence directly);
      2. keep only matches that lie on a (nearly) increasing index flow —
         SRT lines merge/split script sentences, so ±5 jitter is allowed;
         anything else is "none".

    Exact equality after normalization (single or 1-4 joined sentences) is
    "high"; a matched-but-different line is "mismatch" (correction signal).
    """
    if not script_sentences:
        return [ScriptAnchor("", 0.0, "none") for _ in source_lines]

    from collections import Counter

    # Matching happens in kana space: 擦り寄って/すり寄って, 気筒/祈祷 and
    # trailing 〜♪ｗ decorations all normalize to the same reading, so only
    # REAL wording differences survive as mismatch. Script text shown to the
    # review LLM stays the original.
    kana_sents = [_kana_norm(s) for s in script_sentences]
    gram_cache = [Counter(_bigrams(k)) for k in kana_sents]

    # Dialogue is contiguous in the script: after the first line anchors the
    # track's segment, each subsequent line searches a LOCAL window around
    # the previous accepted position (absorbing split/merge jitter), never
    # the whole script — distant repeats (identical lines elsewhere in the
    # work) would otherwise hijack the best match and break the flow.
    _WINDOW = 80
    _JITTER = 5
    _MAX_SPAN = 4

    def best_in(ba, ca, lo, hi):
        """Best (score, idx, span) over 1-4 joined sentences in [lo, hi).

        Scoring joined runs matters: an SRT line often equals 2-3 script
        sentences, which scores poorly against any single sentence.
        """
        best_score, best_idx, best_span = 0.0, -1, 1
        n = len(script_sentences)
        for j in range(lo, min(hi, n)):
            acc = Counter()
            for span in range(1, _MAX_SPAN + 1):
                if j + span - 1 >= n:
                    break
                acc += gram_cache[j + span - 1]
                inter = sum((ca & acc).values())
                score = 2.0 * inter / (len(ba) + sum(acc.values()))
                if score > best_score:
                    best_score, best_idx, best_span = score, j, span
        return best_score, best_idx, best_span

    anchors: list[ScriptAnchor] = []
    prev = -1
    for line in source_lines:
        if not line.strip():
            anchors.append(ScriptAnchor("", 0.0, "none"))
            continue
        kana_line = _kana_norm(line)
        ba = _bigrams(kana_line)
        if not ba:
            anchors.append(ScriptAnchor("", 0.0, "none"))
            continue
        ca = Counter(ba)
        if prev < 0:
            # First non-empty line: global anchor.
            score, idx, span = best_in(ba, ca, 0, len(script_sentences))
        else:
            lo = max(0, prev - _JITTER)
            score, idx, span = best_in(ba, ca, lo, prev + _WINDOW)
        if score < _TH_MATCH or idx < 0:
            anchors.append(ScriptAnchor("", score, "none"))
            continue
        sent = "".join(script_sentences[idx:idx + span])
        kana_sent = "".join(kana_sents[idx:idx + span])
        # Exact equality with the same joined run (kana space).
        exact = kana_line == kana_sent
        if not (exact or score >= _TH_EQUIV):
            # Subset/superset check: an SRT line often carries only part of a
            # script sentence (or the line adds a short interjection). If
            # ≥93% of one side's bigrams are contained in the other, treat as
            # 表記等价 — NOT a correction signal.
            cb = Counter(_bigrams(kana_sent))
            inter1 = sum((ca & cb).values())
            cover_st = inter1 / max(1, len(ba))          # SRT → script
            cover_ts = inter1 / max(1, sum(cb.values()))  # script → SRT
            if cover_st >= 0.93 or cover_ts >= 0.93:
                status = "high"
            else:
                status = "mismatch"
        else:
            status = "high"
        anchors.append(ScriptAnchor(sent, score, status))
        prev = idx + span - 1
    return anchors


def build_script_anchors(
    script_path,
    source_lines: list[str],
) -> list[ScriptAnchor] | None:
    """Load + clean + align a script file to source lines. None on failure."""
    from pathlib import Path

    path = Path(script_path)
    if not path.is_file():
        logger.warning("Script file not found: %s", path)
        return None
    text = read_context_file(path)
    if not text:
        logger.warning("Script file yielded no text: %s", path)
        return None
    sentences = clean_script_text(text)
    if not sentences:
        logger.warning("Script file cleaned to zero sentences: %s", path)
        return None
    anchors = align_script(source_lines, sentences)
    high = sum(1 for a in anchors if a.status == "high")
    mismatch = sum(1 for a in anchors if a.status == "mismatch")
    none = sum(1 for a in anchors if a.status == "none")
    logger.info(
        "Script alignment [%s]: %d sentences, %d lines → high %d / mismatch %d / none %d",
        path.name, len(sentences), len(source_lines), high, mismatch, none,
    )
    return anchors
