"""Whole-file proofread pass over a translated SRT.

Runs after chunked translation to fix problems chunking causes:
  - line-content shifts / misalignment
  - untranslated leftover lines
  - mistranslations and awkward phrasing

Terminology consistency is NOT this pass's job: it is judged whole-file by
the review pass (src/review.py steps 2/4), which reads the entire text at
once and can therefore see a drifting rendering that a single chunk cannot.

The pass is strictly alignment-verified: the translated SRT must contain
exactly the same number of blocks as the source SRT, otherwise the pass
aborts (no silent misalignment). Chunks are sent with a few original
context lines (marked [前N]/[后N], never re-output — the same scheme the
review pass uses) so the LLM can see what the neighbouring lines actually
say and recognise content that belongs to a different line.

Because this pass OVERWRITES existing translations, a response carrying no
[N] marker at all is treated as a failed chunk and the chunk is left alone.
The positional fallback that rescues the translation pass would otherwise
assign prose or a refusal to line 1 verbatim. The file is snapshotted as
<stem>.pre-proofread.srt before it is replaced.

Usage (via main.py): pipeline --proofread
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import TranslateConfig
from src.translate import (
    AllChunksFailedError,
    SrtBlock,
    _count_valid,
    build_instruction,
    call_llm,
    has_numbered_markers,
    make_snapshot,
    parse_response,
    parse_srt,
    write_translated_srt,
)

logger = logging.getLogger(__name__)

PROOFREAD_INSTRUCTION = (
    "You are proofreading a {target_lang} translation of {source_lang} "
    "subtitles. Each line to proofread is numbered [N], starting at [1] for "
    "every chunk.\n"
    "Lines marked [前N] / [后N] are CONTEXT ONLY (original {source_lang} "
    "lines immediately before/after this chunk, N = distance from it) — read "
    "them for context, but NEVER output them and never reuse their "
    "numbering.\n\n"
    "For every [N] line, output exactly:\n"
    "  [N] corrected_translation\n\n"
    "Rules:\n"
    "- Keep [N] in the same order and count, numbered from 1 within this "
    "chunk. Do not output [前N] / [后N] lines.\n"
    "- Fix: mistranslations, content shifted from the wrong line, leftover "
    "un-translated {source_lang} text, awkward phrasing. Use the "
    "context/vocabulary sections (if provided) to resolve domain terms and "
    "story-specific wording.\n"
    "- Output language MUST be {target_lang} only — NEVER English and NEVER "
    "a mix. If a current line is in English or mixed, re-translate it fully "
    "into {target_lang}.\n"
    "- Do NOT change the meaning. Do not add, drop, or merge lines.\n"
    "- If a line is already correct, output it unchanged (identical text).\n"
    "- If a line is empty in the translation but has original text, "
    "translate it now.\n"
    "- If the original line itself looks like an ASR error (nonsense words), "
    "do NOT invent content — keep the current translation as-is.\n"
    "- Output ONLY the [N] lines, nothing else.\n"
)


def _format_proofread_chunk(
    source: list[SrtBlock],
    translated: list[SrtBlock],
    start: int,
    end: int,
    context_lines: int,
) -> str:
    """Format one proofread chunk: context originals + [N] orig→trans pairs.

    Context lines carry [前N]/[后N] markers (N = distance from the chunk) and
    are never numbered in the core index space, so the ONLY numbers the model
    sees are the chunk-relative indices it is asked to answer in. Mixing a
    global [cN] index in here invited answers in the wrong index space.
    """
    parts: list[str] = []
    # Context before (originals only, marked by distance from the chunk)
    for i in range(max(0, start - context_lines), start):
        parts.append(f"[前{start - i}] {source[i].text}")
    # Core: source + current translation for each line, numbered from 1
    for i in range(start, end):
        trans = translated[i].text or "(empty — translate this line)"
        parts.append(f"[{i + 1 - start}] {source[i].text}\n    -> {trans}")
    # Context after
    for i in range(end, min(len(source), end + context_lines)):
        parts.append(f"[后{i - end + 1}] {source[i].text}")
    return "\n".join(parts)


def proofread_srt(source_path: Path, translated_path: Path, config: TranslateConfig) -> Path:
    """Run a proofread pass over translated_path, comparing against source_path.

    The translated file is updated in place. Returns the path.
    """
    source = parse_srt(source_path)
    translated = parse_srt(translated_path)

    if len(source) != len(translated):
        raise ValueError(
            f"Alignment check failed: source has {len(source)} blocks, "
            f"translated has {len(translated)} — refusing to proofread "
            "(would risk silent line shifts)"
        )
    logger.info(
        "Proofread: %d blocks aligned (%s ↔ %s)",
        len(source), source_path.name, translated_path.name,
    )

    # Build the proofread instruction with the SAME evidence the translation
    # used (context files + vocab), so domain terms and story-specific
    # wording can be resolved against the README instead of guesswork.
    instruction = build_instruction(config, base_template=PROOFREAD_INSTRUCTION)

    # Proofread in larger chunks than translation (alignment/shift detection
    # needs scope), but bounded for context limits.
    chunk_size = max(20, config.chunk_size * 2)
    total = (len(source) + chunk_size - 1) // chunk_size
    changed = 0
    failed_chunks = 0

    for chunk_idx in range(total):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(source))
        label = f"Proofread {chunk_idx + 1}/{total}"

        numbered_input = _format_proofread_chunk(
            source, translated, start, end, config.proofread_context_lines
        )

        core_count = end - start

        try:
            response = call_llm(numbered_input, instruction, config)
        except RuntimeError as e:
            # call_llm has already retried (incl. with thinking disabled and a
            # larger budget). Keep this chunk's existing translation rather
            # than blanking it, but report it as an error, not a quiet 0/N.
            logger.error("%s: LLM call failed, chunk left unproofread: %s", label, e)
            failed_chunks += 1
            continue

        if not has_numbered_markers(response):
            # Prose, a refusal, or a collapsed format. This pass overwrites
            # existing text, so an unnumbered answer is never usable: assigning
            # it by line position would commit that prose to the subtitle.
            logger.error(
                "%s: response contains no [N] marker — chunk left unproofread "
                "(NOT applied): %r",
                label, response.strip()[:120],
            )
            failed_chunks += 1
            continue

        translations = parse_response(
            response, core_count, allow_positional_fallback=False
        )
        valid = _count_valid(translations)
        if valid == 0:
            # [N] markers were present but none landed in 1..core_count — the
            # model answered in some other index space. Nothing to apply.
            logger.error(
                "%s: no [N] index fell inside 1..%d — chunk left unproofread "
                "(NOT applied)", label, core_count,
            )
            failed_chunks += 1
            continue
        if valid < core_count:
            logger.warning(
                "%s: only %d/%d lines returned; keeping originals for the rest",
                label, valid, core_count,
            )
        for i, t in enumerate(translations):
            if not t:
                continue
            old = translated[start + i].text
            if t != old:
                translated[start + i].text = t
                changed += 1
                if config.verbose:
                    logger.debug(
                        "[#%d] %s → %s",
                        translated[start + i].index, old, t,
                    )

    # No chunk produced a usable answer: the pass carries no information, so
    # refuse rather than rewriting the file and reporting success.
    if total and failed_chunks == total:
        raise AllChunksFailedError(
            f"Proofread failed: all {total} chunk(s) failed — "
            f"{translated_path} left untouched"
        )

    logger.info("Proofread complete: %d line(s) adjusted out of %d", changed, len(source))
    if failed_chunks:
        logger.warning(
            "Proofread: %d/%d chunk(s) failed and were left unproofread — "
            "those lines keep their pre-proofread translation",
            failed_chunks, total,
        )

    # This pass overwrites its own input, which is the only copy of the
    # translation — snapshot first, and abort if the snapshot cannot be taken.
    try:
        make_snapshot(translated_path, "pre-proofread")
    except OSError as e:
        raise RuntimeError(
            f"Failed to write pre-proofread snapshot for {translated_path}: {e} "
            "— refusing to overwrite the translation"
        ) from e
    write_translated_srt(translated, translated_path)
    return translated_path
