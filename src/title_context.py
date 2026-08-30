"""Media title → context extraction for the subtitle pipeline.

A meaningful media filename (title) is itself content: e.g.
`田舎動物病院の去勢手術記録.wav` tells you the audio is about
動物病院/去勢/手術. Feeding those terms into ASR hotwords biases
faster-whisper toward the correct reading (去勢 stays 去勢 instead of a
same-reading mishearing), and including the title in the translation/review
context helps the LLM pick the right register and terminology. 8/30 lesson:
a work run without any context produced 31% 存疑 — the title alone would
have anchored the domain terms that the plain ASR garbled.

Batch-named, meaningless titles (track01.wav, audio_001.wav, 12345.wav)
must NOT be used — they carry no content signal.
"""

from __future__ import annotations

import re
from pathlib import Path

logger = __import__("logging").getLogger(__name__)

# Batch-prefix patterns to strip before judging the title: leading numeric
# index ("01."), track/file/audio/chapter/part/配信/録音 labels, 【PV】-style
# tag brackets. Applied repeatedly so "01.【PV】　foo" still resolves.
_PREFIX_STRIP_RES = (
    re.compile(r"^\s*(?:\[[^\]]*\]|【[^】]*】|（[^）]*）|\([^)]*\))\s*"),
    re.compile(r"^\d{1,4}\s*[._\-—:\s]*"),
    re.compile(r"^(?:track|audio|file|chap(?:ter)?|part|clip|movie|video|ep(?:isode)?|配信|録音|音声)\s*[\d._\-:\s]*", re.IGNORECASE),
    re.compile(r"^\s+"),
)
# Generic batch-named remainder that means "no content signal": pure digits /
# symbols / one letter, or a bare index like "001" or "14".
_MEANINGLESS_RE = re.compile(
    r"^[\d\s._\-—:()\[\]【】（）]+$|^[a-z0-9]{1,3}$|^(?:track|audio|file)$",
    re.IGNORECASE,
)
# Generic words that are never a content-bearing title (RJ main/sample files,
# test clips, etc.).
_BLACKLIST = frozenset({
    "main", "sample", "test", "demo", "preview", "clip", "temp", "test1",
    "rec", "recording", "new", "untitled", "audio", "sound", "voice",
})
# A meaningful Japanese title has kana or 2+ kanji.
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_KANJI_RE = re.compile(r"[\u4e00-\u9fff]")


def clean_title(stem: str) -> str:
    """Strip batch naming from a media stem; '' when nothing meaningful."""
    title = stem.strip()
    # Loop until stable: "01.【PV】foo" needs the digit strip before the
    # bracket strip can match, so one pass over the patterns is not enough.
    changed = True
    while changed and title:
        changed = False
        for pat in _PREFIX_STRIP_RES:
            new = pat.sub("", title).strip()
            if new != title:
                title = new
                changed = True
    return title


def is_meaningful_title(title: str) -> bool:
    """True when the title carries content signal (not a batch index)."""
    title = title.strip()
    if not title or len(title) < 2:
        return False
    if _MEANINGLESS_RE.match(title):
        return False
    if title.lower() in _BLACKLIST:
        return False
    kana = len(_KANA_RE.findall(title))
    kanji = len(_KANJI_RE.findall(title))
    if kana >= 1 or kanji >= 2:
        return True
    # Latin/katakana words of substance (e.g. "BDSM", "ネトラレ") are useful.
    if re.search(r"[A-Za-z]{2,}", title) or re.search(r"[\u30a0-\u30ff]{2,}", title):
        return True
    return False


def extract_title(media_path: str | Path) -> str | None:
    """Return the cleaned, meaningful title of a media file, else None."""
    path = Path(media_path)
    title = clean_title(path.stem)
    if not is_meaningful_title(title):
        logger.debug("Media title not meaningful, skipped: %r", path.stem)
        return None
    return title


# ── Title homophone anchoring ─────────────────────────────────────────
# 8/30 lesson: ASR transcribed 虚勢 for what the title calls 去勢 — same
# reading きょせい, different kanji. The title context alone did NOT catch
# it: the LLM free-associated 虚勢 into an unrelated slang meaning instead
# of linking the homophone to the title's 去勢. These helpers make the
# homophone link EXPLICIT evidence for the review pass.

_TITLE_TOKEN_RE = re.compile(
    r"[\u4e00-\u9fff]+|[ぁ-んァ-ン]+|[a-zA-Z]+"
)


def tokenize_title(title: str) -> list[tuple[str, str]]:
    """Title → (word, kana) tokens via kakasi word segmentation.

    「田舎動物病院の去勢手術記録」 → (田舎,いなか) (動物,どうぶつ)
    (病院,びょういん) (去勢,きょせい) (手術,しゅじゅつ) (記録,きろく).
    """
    from src.adjudicate import _get_kakasi

    try:
        items = _get_kakasi().convert(title)
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for it in items:
            orig = it.get("orig", "")
            hira = it.get("hira", "")
            if orig and hira and hira not in seen:
                seen.add(hira)
                out.append((orig, hira))
        return out
    except Exception:
        # Degrade: kanji/kana/latin runs as whole tokens.
        return [(t, t) for t in _TITLE_TOKEN_RE.findall(title)]


def build_title_notes(
    title: str,
    source_lines: list[str],
    min_kana_len: int = 2,
) -> dict[int, list[str]]:
    """Line → homophone notes: source word vs title word, same reading.

    For every source line whose kana reading CONTAINS a title token's kana
    while the literal text does NOT contain the token, emit a note telling
    the review LLM the ASR likely mis-heard it (e.g. 虚勢 きょせい ⊃ 去勢
    きょせい). These are *suggestions* — a homophone can be a deliberate
    slang/wordplay, so the note says 疑似 and the DeepFix gate still applies.
    """
    from src.adjudicate import kana_normalize

    # Normalize BOTH sides through kana_normalize so 拗音 collapsing (きょ→きよ)
    # is applied identically — kakasi's raw hira (きょせい) would never match
    # the normalized line reading (きよせい).
    token_kana = [
        (orig, kana_normalize(orig)) for orig, _ in tokenize_title(title)
        if len(kana_normalize(orig)) >= min_kana_len and orig
    ]
    notes: dict[int, list[str]] = {}
    for i, line in enumerate(source_lines, start=1):
        lk = kana_normalize(line)
        if not lk:
            continue
        hits: list[str] = []
        for tok, k in token_kana:
            if k in lk and tok not in line:
                hits.append(f"{tok}({k})")
        if hits:
            notes[i] = [
                "标题同音提示: 源文读音含标题词 "
                + "/".join(hits)
                + " 但字面不同——疑似 ASR 误识（同音不同字），若语义不通请按标题词修正"
            ]
    return notes
