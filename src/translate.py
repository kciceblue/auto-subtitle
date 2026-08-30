"""SRT translation via LLM endpoint.

Parses an SRT file, sends subtitle text in numbered chunks to a local LLM,
and reassembles translated text with original timestamps.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from src.config import TranslateConfig

logger = logging.getLogger(__name__)

_NUMBERED_LINE_RE = re.compile(r"^\[(\d+)\][.:)\s]*\s*(.*)", re.MULTILINE)

INSTRUCTION_TEMPLATE = (
    "Translate the following {source_lang} text to {target_lang}.\n"
    "Rules:\n"
    "- Each input line is numbered as [N] text.\n"
    "- Output exactly [N] translated_text for every input line, in order.\n"
    "- Do not skip, merge, or add lines. Do not output anything else.\n"
    "- Line [N] must contain ONLY the translation of line [N]. Never shift or copy "
    "content from a neighbouring line into a different index.\n"
    "- Use consistent terminology for the same recurring word/character across "
    "the whole file.\n"
    "- If a line is only a sound effect or interjection (e.g. 'うーん', 'はっ', '…'), "
    "render it naturally in {target_lang} or keep it; do not invent words."
)

VOCAB_SECTION = "\nVocabulary:\n{vocab}"

# If output exceeds this multiple of input length, it's runaway
_RUNAWAY_MULTIPLIER = 3.0
# Retries at each chunk size level before splitting
_RETRIES_BEFORE_SPLIT = 2

# Ways to ask an OpenAI-compatible server to skip the thinking block, tried in
# order. Translation and proofreading are extraction-shaped tasks: thinking
# costs 10-20x the tokens without improving the numbered output, and if it
# overruns the token budget the answer never arrives at all. Different
# llama.cpp / vLLM builds honour different spellings, so escalate through them.
_THINKING_OFF_VARIANTS: tuple[dict, ...] = (
    {"reasoning_effort": "none"},
    {"chat_template_kwargs": {"enable_thinking": False}},
)
# Upper bound when growing the budget after an empty response
_MAX_TOKENS_GROWTH = 4


def load_vocab(path: Path) -> str:
    """Load vocabulary file and return its content, stripped of blank lines."""
    raw = path.read_text(encoding="utf-8-sig").strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return "\n".join(lines)


def _read_pdf_text(path: Path) -> str:
    """Extract text layer from a PDF via pypdf (scanned/image-only PDFs yield "")."""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning(
            "PDF context file %s skipped: pypdf not installed "
            "(pip install pypdf in the project venv)", path
        )
        return ""
    try:
        parts: list[str] = []
        with path.open("rb") as fh:
            reader = PdfReader(fh)
            for i, page in enumerate(reader.pages):
                try:
                    txt = (page.extract_text() or "").strip()
                except Exception as e:
                    logger.warning("PDF %s page %d extract failed: %s", path.name, i + 1, e)
                    continue
                if txt:
                    parts.append(txt)
        text = "\n".join(parts).strip()
    except Exception as e:
        logger.warning("Failed to parse PDF %s: %s", path, e)
        return ""
    if not text:
        logger.warning("PDF %s yielded no extractable text (scanned/image-only?)", path)
    return text


def read_context_file(path: Path) -> str:
    """Read a context/README file. Supports plain text (.txt/.md) and PDF (.pdf).

    Text encoding: UTF-8 first (with mojibake detection via U+FFFD ratio),
    then Shift-JIS (cp932) / EUC-JP — Japanese context files are often SJIS.
    """
    if path.suffix.lower() == ".pdf":
        return _read_pdf_text(path)
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "cp932", "euc-jp"):
        try:
            raw = path.read_text(encoding=enc).strip()
        except (UnicodeDecodeError, OSError) as e:
            last_err = e
            continue
        if enc == "utf-8-sig" and raw.count("\ufffd") / max(1, len(raw)) > 0.01:
            continue  # 疑似乱码（UTF-8 硬解 SJIS 文本）→ 试下一种编码
        return raw
    logger.warning("Failed to read context file %s: %s", path, last_err)
    return ""


def _sample_script(text: str, budget: int) -> str:
    """Script sampling: head + middle + tail (audio may correspond to any part)."""
    if len(text) <= budget:
        return text
    third = budget // 3
    head = text[:third]
    mid_start = (len(text) - third) // 2
    mid = text[mid_start:mid_start + third]
    tail = text[-third:]
    return f"{head}\n...[中略]...\n{mid}\n...[后略]...\n{tail}"


def _load_context(
    paths: list[Path],
    max_chars: int = 4000,
    kinds: dict[str, str] | None = None,
    title: str | None = None,
) -> str:
    """Load context files (README/synopsis/script), each truncated to max_chars.

    `kinds` maps path-string → "synopsis" | "script" (from src/context_scan).
    Script files get double budget + head/mid/tail sampling so the audio's
    corresponding section is actually visible; synopsis truncates from the
    head as before. A meaningful media `title` (src/title_context) is
    prepended as its own block — filename terms anchor domain vocabulary the
    ASR may have garbled.
    """
    parts: list[str] = []
    if title:
        parts.append(f"--- 媒体标题（文件名，含内容关键词）---\n{title}")
    kinds = kinds or {}
    for p in paths:
        if not p.is_file():
            continue
        kind = kinds.get(str(p)) or kinds.get(p.name) or "synopsis"
        raw = read_context_file(p)
        if not raw:
            continue
        if kind == "script":
            raw = _sample_script(raw, max_chars * 2)
        elif len(raw) > max_chars:
            raw = raw[:max_chars] + "\n... [truncated]"
        parts.append(f"--- {p.name} ({kind}) ---\n{raw}")
    return "\n\n".join(parts)


def build_instruction(
    config: TranslateConfig,
    base_template: str = INSTRUCTION_TEMPLATE,
) -> str:
    """Build the instruction string: base prompt + context files + vocab.

    `base_template` lets other passes (proofread/review) reuse the same
    context/vocab injection with their own base prompt.
    """
    instruction = base_template.format(
        source_lang=config.source_lang,
        target_lang=config.target_lang,
    )
    if config.context_files or config.title:
        context_text = _load_context(config.context_files, kinds=config.context_kinds,
                                     title=config.title)
        if context_text:
            instruction += (
                "\n=== WORK CONTEXT (synopsis/script reference material — use "
                "ONLY to pick correct character names, terms and tone; do NOT "
                "translate or output this section. If it contains dialogue/台词, "
                "it is reference only: subtitles must match the AUDIO, never "
                "copy the script verbatim) ===\n"
                f"{context_text}\n"
                "=== END CONTEXT ===\n"
            )
            for p in config.context_files:
                if p.is_file():
                    logger.info("Loaded context from %s", p)
    if config.vocab_file is not None:
        vocab_text = load_vocab(config.vocab_file)
        if vocab_text:
            instruction += VOCAB_SECTION.format(vocab=vocab_text)
            logger.info("Loaded vocab (%d entries) from %s",
                        vocab_text.count("\n") + 1, config.vocab_file)
    return instruction


@dataclass
class SrtBlock:
    """A single SRT subtitle entry."""

    index: int
    ts_line: str
    text: str


def parse_srt(path: Path) -> list[SrtBlock]:
    """Parse an SRT file into a list of SrtBlock entries."""
    raw = path.read_text(encoding="utf-8-sig")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    chunks = re.split(r"\n\s*\n", raw.strip())

    blocks: list[SrtBlock] = []
    for chunk in chunks:
        lines = chunk.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            logger.warning("Skipping malformed SRT block: %s", lines[0])
            continue
        ts_line = lines[1].strip()
        text = "\n".join(line.strip() for line in lines[2:])
        blocks.append(SrtBlock(index=index, ts_line=ts_line, text=text))

    logger.info("Parsed %d SRT blocks from %s", len(blocks), path)
    return blocks


def format_chunk(blocks: list[SrtBlock]) -> str:
    """Format blocks as numbered lines [1] text, [2] text, ... (1-based per chunk)."""
    return "\n".join(f"[{i}] {b.text}" for i, b in enumerate(blocks, start=1))


# ── Streaming SSE + length guard ─────────────────────────────────────


class RunawayError(RuntimeError):
    """Raised when model output exceeds expected length."""
    pass


class TransportError(RuntimeError):
    """Transport/JSON failure on the non-streaming path.

    `requests` exceptions derive from OSError and `resp.json()` raises
    ValueError — neither is caught by the callers' `except RuntimeError`
    guards, so the non-streaming path normalises them to this type. Handled
    like any other retryable failure inside `call_llm`.
    """
    pass


class EmptyContentError(RuntimeError):
    """Raised when the server produced no assistant content.

    Typically the token budget was spent entirely inside the reasoning block
    (finish_reason="length" with a large reasoning_content and empty content).
    Retryable: retry with thinking disabled and/or a larger budget.
    """
    pass


class AllChunksFailedError(RuntimeError):
    """Raised when every chunk of a whole-file pass failed.

    A proofread/review run that could not process a single chunk has produced
    no information at all. Committing its output and exiting 0 would report a
    clean bill of health for a pass that never actually ran, so the pass
    raises instead and `main.py` turns it into a non-zero exit.
    """
    pass


@dataclass
class StreamResult:
    """Outcome of one streamed chat-completion request."""

    content: str
    reasoning_chars: int = 0
    finish_reason: str | None = None


def _stream_response(
    endpoint: str,
    payload: dict,
    timeout: int,
    expected_len: int,
) -> StreamResult:
    """Stream SSE response with length guard.

    If output exceeds _RUNAWAY_MULTIPLIER * expected_len, the connection
    is closed immediately and RunawayError is raised.

    Reasoning models stream their thinking as `reasoning_content` deltas;
    those are never part of the answer but they do consume the token budget,
    so their size and the final finish_reason are reported back for diagnosis.
    """
    resp = requests.post(
        endpoint, json=payload,
        timeout=(10, timeout),
        stream=True,
    )
    resp.raise_for_status()
    resp.encoding = "utf-8"

    collected: list[str] = []
    total_len = 0
    reasoning_chars = 0
    finish_reason: str | None = None

    try:
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            data_str = raw_line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            try:
                choice = chunk["choices"][0]
                delta_obj = choice.get("delta") or {}
            except (KeyError, IndexError, TypeError, AttributeError):
                continue

            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

            reasoning_chars += len(delta_obj.get("reasoning_content") or "")

            # Extract content delta from SSE chunk
            delta = delta_obj.get("content") or choice.get("text") or ""
            if not delta:
                continue

            collected.append(delta)
            total_len += len(delta)

            # Length guard: abort if output is way too long
            if expected_len > 0 and total_len > expected_len * _RUNAWAY_MULTIPLIER:
                logger.warning(
                    "Runaway: output %d chars exceeds %.0fx input (%d chars), aborting",
                    total_len, _RUNAWAY_MULTIPLIER, expected_len,
                )
                raise RunawayError(
                    f"Output length {total_len} exceeded "
                    f"{_RUNAWAY_MULTIPLIER}x expected ({expected_len})"
                )
    finally:
        resp.close()

    return StreamResult(
        content="".join(collected),
        reasoning_chars=reasoning_chars,
        finish_reason=finish_reason,
    )


def _extract_output(data: dict) -> str:
    """Extract generated text from non-streaming LLM response.

    A reasoning model that never left the thinking block reports
    content=null, so normalise that to "" rather than returning None.
    """
    if "output" in data and isinstance(data["output"], str):
        return data["output"]
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        pass
    try:
        return data["choices"][0]["text"] or ""
    except (KeyError, IndexError, TypeError):
        pass
    raise ValueError(f"Cannot extract output from response: {list(data.keys())}")


def _finish_reason(data: dict) -> str | None:
    """Best-effort finish_reason from a non-streaming response."""
    try:
        return data["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


def _build_payload(
    content: str,
    instruction: str,
    config: TranslateConfig,
    thinking_off: dict,
    max_tokens: int,
    *,
    stream: bool,
    with_thinking: bool = False,
) -> dict:
    """Build a chat-completions payload.

    `max_tokens` is always sent explicitly so behaviour does not depend on
    the server's own default. `thinking_off` keys are applied with setdefault,
    so an explicit setting in extra_payload always wins over ours.

    When `with_thinking` is True the thinking-off defaults are NOT applied,
    so the model may use its reasoning budget on the task (used by the
    semantic review pass, which needs genuine analysis rather than a
    surface-consistency check).
    """
    payload: dict = {
        "messages": [
            {"role": "user", "content": f"{instruction}\n{content}"},
        ],
        "max_tokens": max_tokens,
    }
    if stream:
        payload["stream"] = True
    if config.extra_payload:
        payload.update(config.extra_payload)
    if not with_thinking:
        for key, value in thinking_off.items():
            payload.setdefault(key, value)
    return payload


def call_llm(
    content: str,
    instruction: str,
    config: TranslateConfig,
    with_thinking: bool = False,
) -> str:
    """Send a translation request to the LLM endpoint.

    Uses streaming with length guard. On runaway, transport error, or an
    empty response, retries with exponential backoff — escalating through
    the thinking-off variants and growing the token budget, since an empty
    response usually means the budget was consumed by the reasoning block.

    With `with_thinking=True` (semantic review pass) the thinking-off
    defaults are not applied and the thinking-escalation is skipped:
    the model is allowed to think, and only the budget is grown when
    the response comes back empty.
    """
    expected_len = len(content)
    max_tokens = config.max_tokens
    variant = 0
    use_streaming = True

    last_err: Exception | None = None
    for attempt in range(1 + config.retries):
        if attempt > 0:
            wait = 2 ** attempt
            logger.info("Retry %d/%d in %ds...", attempt, config.retries, wait)
            time.sleep(wait)
        payload = _build_payload(
            content, instruction, config,
            _THINKING_OFF_VARIANTS[variant], max_tokens, stream=True,
            with_thinking=with_thinking,
        )
        try:
            if use_streaming:
                result = _stream_response(
                    config.endpoint, payload, config.timeout, expected_len,
                )
            else:
                # Same attempt budget and escalation as the streaming path —
                # the fallback is just another way to issue this attempt.
                result = _call_llm_non_streaming(
                    content, instruction, config,
                    _THINKING_OFF_VARIANTS[variant], max_tokens,
                    with_thinking=with_thinking,
                )
            if result.content.strip():
                return result.content
            # No answer: almost always the budget went into reasoning_content.
            last_err = EmptyContentError(
                f"empty content (finish_reason={result.finish_reason}, "
                f"reasoning={result.reasoning_chars} chars, "
                f"max_tokens={max_tokens})"
            )
            if with_thinking:
                # Review pass: thinking SHOULD stay on — only grow the budget.
                # Reasoning can swallow the whole budget (observed: 24k chars
                # of reasoning inside an 8192-token cap), so jump straight to
                # the max cap on the first retry.
                logger.warning(
                    "Empty response (thinking-on): finish_reason=%s, "
                    "reasoning_content=%d chars, max_tokens=%d — retrying "
                    "with max token budget",
                    result.finish_reason, result.reasoning_chars, max_tokens,
                )
                max_tokens = config.max_tokens * _MAX_TOKENS_GROWTH
                # 8/30: thinking-on 连续空响应（推理吃光预算）曾导致整个
                # DeepFix chunk 三次尝试全失败、行留未复核。最后一次尝试
                # 降级 thinking-off 兜底：宁可无思考审查，也不留未复核行
                # （未复核行会被静默保留原译文，等于零审查）。降级结果仍
                # 走调用方的 _reject_fix 门，不会直接写入字幕。
                if attempt >= config.retries - 1:
                    logger.warning(
                        "Thinking produced no content on %d attempt(s) — "
                        "falling back to thinking-off retry",
                        attempt + 1,
                    )
                    with_thinking = False
                    variant = 0
                continue
            logger.warning(
                "Empty response: finish_reason=%s, reasoning_content=%d chars, "
                "max_tokens=%d — retrying with thinking disabled",
                result.finish_reason, result.reasoning_chars, max_tokens,
            )
            variant = min(variant + 1, len(_THINKING_OFF_VARIANTS) - 1)
            max_tokens = min(
                max(max_tokens * 2, 16384),
                config.max_tokens * _MAX_TOKENS_GROWTH,
            )
            continue
        except RunawayError as e:
            last_err = e
            logger.warning("Runaway detected, will retry: %s", e)
            continue
        except TransportError as e:
            last_err = e
            logger.warning("Non-streaming request failed, will retry: %s", e)
            continue
        except requests.exceptions.HTTPError as e:
            if use_streaming and e.response is not None and e.response.status_code == 400:
                # Retry this content without streaming instead of calling the
                # fallback inline: it then gets the same retry / variant /
                # budget escalation as every other attempt.
                logger.info("Streaming not supported, falling back to non-streaming")
                last_err = e
                use_streaming = False
                continue
            if e.response is not None and e.response.status_code >= 500:
                last_err = e
                logger.warning("Server error %d, will retry", e.response.status_code)
                continue
            raise
        except requests.exceptions.Timeout as e:
            last_err = e
            logger.warning("Request timed out, will retry")
            continue
        except (requests.exceptions.ConnectionError, requests.exceptions.RequestException) as e:
            last_err = e
            logger.warning("Request failed: %s", e)
            continue

    raise RuntimeError(f"LLM call failed after {1 + config.retries} attempts: {last_err}")


def _call_llm_non_streaming(
    content: str,
    instruction: str,
    config: TranslateConfig,
    thinking_off: dict,
    max_tokens: int,
    *,
    with_thinking: bool = False,
) -> StreamResult:
    """Non-streaming fallback when endpoint doesn't support streaming.

    Returns the same StreamResult shape as `_stream_response` (empty content
    included) so `call_llm` applies one escalation policy to both paths.
    HTTPError is left alone for `call_llm`'s status-code handling; every other
    transport/JSON failure is normalised to TransportError.
    """
    payload = _build_payload(
        content, instruction, config, thinking_off, max_tokens, stream=False,
        with_thinking=with_thinking,
    )

    try:
        resp = requests.post(
            config.endpoint,
            json=payload,
            timeout=(10, config.timeout),
        )
        resp.raise_for_status()
        data = resp.json()
        output = _extract_output(data)
    except requests.exceptions.HTTPError:
        raise
    except (requests.exceptions.RequestException, ValueError) as e:
        raise TransportError(f"non-streaming request failed: {e}") from e
    return StreamResult(content=output, finish_reason=_finish_reason(data))


# ── Response parsing ─────────────────────────────────────────────────


def has_numbered_markers(text: str) -> bool:
    """True if the response contains at least one [N] marker.

    Passes that *overwrite* existing subtitle text (proofread) use this to
    reject a markerless answer outright instead of falling back to line
    positions: prose or a refusal would otherwise be assigned to line 1
    verbatim and committed to the file.
    """
    return _NUMBERED_LINE_RE.search(text) is not None


def parse_response(
    text: str,
    expected: int,
    allow_positional_fallback: bool = True,
) -> list[str]:
    """Parse numbered [N] lines from LLM response.

    Returns a list of translated strings, one per expected index.
    Missing indices fall back to empty string.

    `allow_positional_fallback` keeps the "treat the Nth line as index N"
    rescue for the translation pass, where a chunk is split down to a single
    line and an unnumbered answer is still the translation of that line. It
    MUST be disabled by passes that overwrite existing text: there a
    markerless response is far more likely to be prose or a refusal, and
    positionally assigning it would write that prose into the subtitle.
    """
    found: dict[int, str] = {}
    for m in _NUMBERED_LINE_RE.finditer(text):
        num = int(m.group(1))
        found[num] = m.group(2).strip()

    # Positional fallback: if no [N] markers found, use line positions
    if not found and allow_positional_fallback:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if lines:
            logger.info(
                "No [N] markers in response, using positional fallback "
                "(%d lines for %d expected)",
                len(lines), expected,
            )
            for i, line in enumerate(lines[:expected], start=1):
                found[i] = line

    results: list[str] = []
    missing: list[int] = []
    for i in range(1, expected + 1):
        if i in found:
            results.append(found[i])
        else:
            missing.append(i)
            results.append("")

    if missing:
        logger.warning(
            "Missing %d translation(s) in response: indices %s",
            len(missing), missing,
        )

    return results


def _count_valid(translations: list[str]) -> int:
    """Count non-empty translations."""
    return sum(1 for t in translations if t)


# ── Chunk translation with recursive splitting ───────────────────────


def _translate_blocks(
    blocks: list[SrtBlock],
    instruction: str,
    config: TranslateConfig,
    label: str,
) -> list[str]:
    """Translate a list of blocks, splitting on failure.

    Strategy:
    1. Try the full chunk, retry up to _RETRIES_BEFORE_SPLIT times.
    2. If still incomplete, split in half and recurse each half.
    3. Keep splitting until single lines, which always succeed
       (positional fallback guarantees a 1-line response parses).
    """
    expected = len(blocks)

    # Try at current chunk size with retries
    best: list[str] | None = None
    best_count = 0

    for attempt in range(1 + _RETRIES_BEFORE_SPLIT):
        if attempt > 0:
            logger.info(
                "%s: incomplete (%d/%d), retry %d/%d at size %d...",
                label, best_count, expected, attempt, _RETRIES_BEFORE_SPLIT, expected,
            )

        try:
            numbered_input = format_chunk(blocks)
            response_text = call_llm(numbered_input, instruction, config)
        except RuntimeError as e:
            logger.warning("%s: LLM call failed: %s", label, e)
            continue

        if config.verbose:
            logger.debug("LLM response:\n%s", response_text)

        translations = parse_response(response_text, expected)
        valid = _count_valid(translations)

        if valid > best_count:
            best = translations
            best_count = valid

        if valid == expected:
            return translations

    # All retries exhausted at this size — split and recurse
    if expected > 1:
        mid = expected // 2
        logger.info(
            "%s: splitting %d → %d + %d",
            label, expected, mid, expected - mid,
        )
        left = _translate_blocks(blocks[:mid], instruction, config, f"{label}a")
        right = _translate_blocks(blocks[mid:], instruction, config, f"{label}b")
        merged = left + right
        # The full-size attempt may have translated lines the split retries
        # then failed on — keep its best answers as a per-line fallback.
        if best is not None:
            merged = [m or b for m, b in zip(merged, best)]
        return merged

    # Single line: return best attempt or empty
    if best is not None:
        return best
    return [""] * expected


def make_snapshot(target: Path, tag: str) -> Path:
    """Snapshot `target` as <stem>.<tag><suffix> before a pass overwrites it.

    The FIRST snapshot wins: proofread and review overwrite their own input,
    so a second run over the same file must keep the original copy rather
    than replacing it with already-modified text.

    Raises OSError if the snapshot cannot be taken — callers abort instead of
    overwriting what is then the only copy of the translation.
    """
    snapshot = target.with_name(f"{target.stem}.{tag}{target.suffix}")
    if snapshot.exists():
        logger.info("Snapshot already exists, keeping the original: %s", snapshot)
        return snapshot
    shutil.copy2(target, snapshot)
    logger.info("Snapshot: %s", snapshot)
    return snapshot


def write_translated_srt(blocks: list[SrtBlock], output_path: Path) -> Path:
    """Write SrtBlocks to an SRT file.

    Written to a sibling .tmp file and moved into place with os.replace, so an
    interrupted or failed write can never truncate an existing SRT — the
    proofread and review passes overwrite their own input in place.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for block in blocks:
                f.write(f"{block.index}\n{block.ts_line}\n{block.text}\n\n")
        os.replace(tmp_path, output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    logger.info("Translated SRT written: %s (%d entries)", output_path, len(blocks))
    return output_path


def translate_srt(input_path: Path, config: TranslateConfig) -> Path:
    """Translate an SRT file via LLM endpoint.

    Parses the source SRT, sends text in chunks to the LLM, and writes
    a new SRT with translated text and original timestamps.

    Returns:
        Path to the translated SRT file.
    """
    blocks = parse_srt(input_path)
    if not blocks:
        raise ValueError(f"No subtitle blocks found in {input_path}")

    output_path = config.resolve_translate_output(input_path)
    instruction = build_instruction(config)

    total_chunks = (len(blocks) + config.chunk_size - 1) // config.chunk_size
    untranslated = 0

    for chunk_idx in range(total_chunks):
        start = chunk_idx * config.chunk_size
        end = min(start + config.chunk_size, len(blocks))
        chunk = blocks[start:end]
        chunk_label = f"Chunk {chunk_idx + 1}/{total_chunks}"

        logger.info("Translating %s (%d blocks)...", chunk_label, len(chunk))

        translations = _translate_blocks(chunk, instruction, config, chunk_label)

        for i, translated in enumerate(translations):
            if translated:
                blocks[start + i].text = translated
            else:
                untranslated += 1
                logger.warning(
                    "Keeping original text for block %d (no translation received)",
                    blocks[start + i].index,
                )

    if untranslated:
        logger.warning("Total untranslated blocks: %d / %d", untranslated, len(blocks))

    return write_translated_srt(blocks, output_path)
