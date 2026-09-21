"""Compact, provenance-checked context prepared once per work before GPU ASR."""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

from src.config import TranslateConfig
from src.quality import mode
from src.translate import call_llm, read_context_file
from src.workflow_state import fingerprint, read_json, write_json

logger = logging.getLogger(__name__)
WORK_CONTEXT_VERSION = "work-context-2"
MAX_VERBATIM_CHARS = 3000
INSTRUCTION = """Prepare compact background for {source_lang}→{target_lang} subtitles.
Use only supplied material. Write the summary in {target_lang}.
Do not invent characters, relationships or spoken lines. When Japanese omits a subject,
leave the actor unspecified; never assign an action to a named person by proximity.
Return a JSON object, no markdown, with:
{"summary":"brief factual background, <=1000 characters",
 "terms":[{"source":"exact source term","translation":"suggested target rendering",
           "evidence":"exact quote from supplied material"}]}
At most 20 terms. The glossary is a contextual suggestion, never authority to change audio.
"""


def prepare_work_context(paths: list[Path], titles: list[str], config: TranslateConfig,
                         cache: Path) -> str:
    # Preserve an already compact reference instead of asking the writer model
    # to paraphrase facts or a supplied glossary before it translates anything.
    # Keep the long-source excerpts bounded and distributed across input files.
    title = "媒体标题: " + "；".join(sorted(set(titles))) if titles else ""
    parts = [title[:2000]] if title else []
    originals = [title] if title else []
    original_size = len(title)
    complete = original_size <= MAX_VERBATIM_CHARS
    sources = []
    allowance = min(3000, max(1, 10000 // max(1, len(set(paths)))))
    for path in sorted(set(paths)):
        text = read_context_file(path)
        sources.append({"name": path.name, "content_hash": fingerprint(text)})
        if text:
            heading = f"{path.name}:\n"
            parts.append(heading + text[:allowance])
            original_size += len(heading) + len(text) + (2 if originals else 0)
            if complete and original_size <= MAX_VERBATIM_CHARS:
                originals.append(heading + text)
            else:
                complete = False
                originals = []
    verbatim = complete
    corpus = "\n\n".join(originals) if verbatim else "\n\n".join(parts)[:12000]
    if not corpus:
        return "No external work context. Do not infer a story or glossary."
    key = fingerprint([WORK_CONTEXT_VERSION, "verbatim" if verbatim else "compressed",
                       config.endpoint, corpus, sources, config.source_lang,
                       config.target_lang, config.extra_payload])
    saved = read_json(cache, {})
    if (isinstance(saved, dict) and saved.get("key") == key
            and isinstance(saved.get("context"), str)):
        return saved["context"]
    if verbatim:
        context = "作品原始资料（逐字保留；仅背景，不能覆盖录音或补写台词）:\n" + corpus
        write_json(cache, {"key": key, "context": context, "mode": "verbatim",
                           "sources": sources})
        return context
    cfg = replace(mode(config, "work-context"), max_tokens=2048, response_guard_floor=512)
    try:
        response = call_llm(corpus, INSTRUCTION.replace("{source_lang}", config.source_lang)
                            .replace("{target_lang}", config.target_lang), cfg)
        result = json.loads(response.strip().removeprefix("```json").removesuffix("```").strip())
        if not isinstance(result, dict) or not isinstance(result.get("summary"), str):
            raise ValueError("Invalid context summary")
        summary = result["summary"][:1000]
        terms = []
        for term in result.get("terms", [])[:20]:
            if not isinstance(term, dict):
                continue
            word, target, quote = (term.get(k, "") for k in ("source", "translation", "evidence"))
            if (all(isinstance(v, str) and v.strip() for v in (word, target, quote))
                    and word in quote and quote in corpus and len(word) <= 50 and len(target) <= 80):
                terms.append(f"{word} → {target}（建议；出处: {quote[:160]}）")
        context = "作品背景（仅背景，不能覆盖录音）:\n" + summary
        if titles:
            context += "\n媒体标题: " + "；".join(sorted(set(titles)))[:1000]
        if terms:
            context += "\n术语建议（用户词表优先，不能据此改写听到的台词）:\n" + "\n".join(terms)
        write_json(cache, {"key": key, "context": context, "mode": "compressed",
                           "sources": sources})
        return context
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
        logger.warning("Work context compression unavailable; retaining source excerpts: %s", exc)
        # A degraded answer is deliberately not cached, so it can recover next run.
        return "作品原始资料摘录（不能覆盖录音）:\n" + corpus[:6000]
