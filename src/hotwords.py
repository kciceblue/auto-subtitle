"""Generate ASR hotwords from context files (README/synopsis) + a vocab file.

Pipeline:
    context files (README.txt, ...) + vocab candidates
        → LLM extracts a short list of terms the audio is likely to contain
        → faster-whisper hotwords (biased decoding)

The LLM is called through the same OpenAI-compatible endpoint used for
translation (default http://127.0.0.1:8089/v1/chat/completions).

Results are cached in ``.hotword_cache.json`` keyed by a hash of
(context content + vocab + source_lang), so repeated runs over the same
folder are instant.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import requests

from src.translate import _THINKING_OFF_VARIANTS, read_context_file

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "http://127.0.0.1:8089/v1/chat/completions"

# Maximum characters of context content sent to the LLM.
MAX_CONTEXT_CHARS = 6000
# Maximum hotwords passed to faster-whisper (it concatenates them into a
# prompt; very long lists dilute the bias).
MAX_HOTWORDS = 40

SYSTEM_PROMPT = """\
You are preparing an ASR (speech recognition) hotword list.
Given the synopsis / README of a work and optional vocabulary candidates, \
extract the words, names and terms that are most likely to be SPOKEN in the \
audio of this work.

Rules:
- Output ONLY the hotwords, one per line, no numbering, no explanation.
- At most {max_hotwords} items.
- Prefer: character names, place names, recurring objects/acts, domain \
technical terms (medical, military, religious, industry jargon), and \
proper nouns that a generic ASR model would likely mishear.
- The context may include a dialogue/script (台词台本) of the work: prefer \
high-frequency proper nouns, character names and terms from it.
- Keep each hotword short (1-5 words). Use the form most likely to be \
spoken (for Japanese: the spoken form, i.e. kanji or katakana as read).
- Do NOT output common everyday words (e.g. "good morning", "hello").
- If the source language is Japanese, output Japanese hotwords.
"""

USER_TEMPLATE = """\
Source language: {source_lang}

=== SYNOPSIS / CONTEXT ===
{context}

=== VOCABULARY CANDIDATES (may contain corrections like 误->正) ===
{vocab}

Generate the ASR hotword list now (one per line, at most {max_hotwords}).
"""


@dataclass
class HotwordResult:
    hotwords: list[str]
    source: str  # "cache" | "llm" | "vocab-only" | "none"
    cache_key: str | None = None


def _read_lines(path: Path, limit: int = 200) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = [ln.strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln][:limit]


def _read_context(paths: list[Path]) -> str:
    parts: list[str] = []
    for p in paths:
        if not p.is_file():
            logger.debug("Context file not found, skipping: %s", p)
            continue
        text = read_context_file(p)
        if text:
            parts.append(f"--- {p.name} ---\n{text}")
    joined = "\n\n".join(parts)
    if len(joined) > MAX_CONTEXT_CHARS:
        joined = joined[:MAX_CONTEXT_CHARS] + "\n... [truncated]"
    return joined


def _call_llm(
    endpoint: str,
    source_lang: str,
    context: str,
    vocab: list[str],
    extra_payload: dict | None = None,
    timeout: int = 120,
) -> list[str] | None:
    """Call the LLM endpoint for hotword extraction. Returns None on failure."""
    system = SYSTEM_PROMPT.format(max_hotwords=MAX_HOTWORDS)
    user = USER_TEMPLATE.format(
        source_lang=source_lang,
        context=context or "(no context provided)",
        vocab="\n".join(vocab) if vocab else "(none)",
        max_hotwords=MAX_HOTWORDS,
    )
    payload: dict = {
        "model": "qwen3.8-27b-dflash",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
        "stream": False,
    }
    if extra_payload:
        payload.update(extra_payload)

    # Reasoning models (qwen3.8) spend the whole token budget on
    # reasoning_content and return content=""; disable thinking for this short
    # extraction task so the answer always lands in `content`. Escalate through
    # the spellings different servers understand if the answer is still empty.
    content = ""
    for thinking_off in _THINKING_OFF_VARIANTS:
        attempt = dict(payload)
        for key, value in thinking_off.items():
            attempt.setdefault(key, value)
        try:
            resp = requests.post(endpoint, json=attempt, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            logger.warning("Hotword LLM call failed (%s): %s", endpoint, e)
            return None
        if content:
            break
        logger.warning(
            "Hotword LLM returned empty content (finish_reason=%s); retrying "
            "with thinking disabled another way",
            data["choices"][0].get("finish_reason"),
        )
    if not content:
        logger.warning("Hotword LLM returned no content after %d attempt(s)",
                       len(_THINKING_OFF_VARIANTS))
        return None

    # Parse: one hotword per line; strip numbering/markdown bullets.
    words: list[str] = []
    for line in content.splitlines():
        line = re.sub(r"^\s*(\d+[\.\)、]|[-*•]|\|)\s*", "", line).strip()
        line = line.strip("`\"'").strip()
        if not line or len(line) > 40:
            continue
        if line.lower() in {"(none)", "none", "无"}:
            continue
        words.append(line)
    # Dedupe, preserve order, cap.
    seen: set[str] = set()
    unique: list[str] = []
    for w in words:
        key = w.lower()
        if key not in seen:
            seen.add(key)
            unique.append(w)
    return unique[:MAX_HOTWORDS]


def _merge_vocab(vocab: list[str]) -> list[str]:
    """Parse vocab lines like '虚勢->去勢' or plain '去勢'."""
    result: list[str] = []
    seen: set[str] = set()
    for line in vocab:
        if "->" in line:
            # correction form: prefer the target (right side)
            parts = [p.strip() for p in line.split("->")]
            candidates = [parts[-1]] + [parts[0]]
        else:
            candidates = [line]
        for c in candidates:
            key = c.lower()
            if c and key not in seen:
                seen.add(key)
                result.append(c)
    return result


def generate_hotwords(
    context_files: list[Path] | None = None,
    vocab_file: Path | None = None,
    source_lang: str = "Japanese",
    endpoint: str | None = None,
    extra_payload: dict | None = None,
    cache_file: Path | None = None,
    force: bool = False,
    title: str | None = None,
) -> HotwordResult:
    """Generate the final ASR hotword list.

    Args:
        context_files: README/synopsis files to ground hotword extraction in.
        vocab_file: optional vocab file (one term per line, 'A->B' = correction).
        source_lang: e.g. "Japanese" — controls output language of hotwords.
        endpoint: OpenAI-compatible chat completions endpoint.
        extra_payload: merged into the request JSON (e.g. model override).
        cache_file: JSON cache file. Defaults to ./.hotword_cache.json.
        force: bypass cache.
        title: optional meaningful media title (src/title_context). It is
            injected into the context so domain terms in the filename
            (e.g. 去勢/麻酔) bias ASR decoding — 8/30 lesson: without
            any context, garbled terms stay garbled and cannot be recovered.
    """
    context_files = [p for p in (context_files or []) if p.is_file()]
    vocab: list[str] = []
    if vocab_file is not None and vocab_file.is_file():
        vocab = _read_lines(vocab_file)

    context_text = _read_context(context_files)
    if title:
        title_block = f"=== MEDIA TITLE (filename — contains content terms) ===\n{title}"
        context_text = (title_block + "\n\n" + context_text).strip()
    vocab_terms = _merge_vocab(vocab)

    # Cache key: content of context + vocab + language (+ title).
    h = hashlib.sha256()
    for p in context_files:
        try:
            h.update(p.read_bytes())
        except OSError:
            pass
    if title:
        h.update(title.encode("utf-8"))
    for v in vocab_terms:
        h.update(v.encode("utf-8"))
    h.update(source_lang.encode("utf-8"))
    cache_key = h.hexdigest()[:16]

    if cache_file is None:
        cache_file = Path(".hotword_cache.json")
    cache: dict = {}
    if cache_file.is_file() and not force:
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cache = {}

    if not force and cache_key in cache:
        cached = cache[cache_key]
        logger.info("Hotwords from cache (%d words): %s", len(cached), cached[:8])
        return HotwordResult(hotwords=cached, source="cache", cache_key=cache_key)

    llm_words: list[str] | None = None
    if (context_text or vocab_terms) and (endpoint or DEFAULT_ENDPOINT):
        llm_words = _call_llm(
            endpoint or DEFAULT_ENDPOINT,
            source_lang,
            context_text,
            vocab_terms,
            extra_payload=extra_payload,
        )
        if llm_words is None:
            logger.warning("LLM hotword generation failed; falling back to vocab only")

    # Merge: LLM words first (grounded in context), then vocab-only terms
    # the LLM missed (they are explicit user-provided corrections).
    merged: list[str] = []
    seen: set[str] = set()
    for w in (llm_words or []) + vocab_terms:
        key = w.lower()
        if key not in seen:
            seen.add(key)
            merged.append(w)
    merged = merged[:MAX_HOTWORDS]

    # Cache ONLY a result the LLM actually contributed to. On a transient LLM
    # failure `merged` degrades to the vocab terms alone; caching that would
    # pin the degraded list forever, since every later run short-circuits on
    # the cache and never retries the LLM.
    if merged and llm_words is not None:
        cache[cache_key] = merged
        try:
            cache_file.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("Failed to write hotword cache %s: %s", cache_file, e)
    elif merged:
        logger.warning(
            "Hotwords not cached: the LLM did not contribute (vocab-only "
            "fallback) — the next run will retry the LLM",
        )

    source = "llm" if (llm_words and merged) else ("vocab-only" if merged else "none")
    logger.info("Hotwords generated (%s, %d words): %s", source, len(merged), merged)
    return HotwordResult(hotwords=merged, source=source, cache_key=cache_key)
