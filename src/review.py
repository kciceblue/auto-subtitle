"""Semantic review pass over a translated SRT — 5-step, self-converging.

Runs after proofread. Where proofread is a fast, thinking-off consistency
check, review is a deliberate semantic gate: a line is only "released"
when it can be judged correct given all available evidence — source line,
story context (README), vocab, a whole-file understanding of the content,
and (if provided) a reference SRT's translations at the same timestamps.

The workflow (steps 1-5; step 1 = the upstream translation, nothing to do
here):

  2. global_pass — ONE thinking-OFF call over the FULL source + FULL
     translation. Produces a SHORT context/narrative summary only (subject,
     scene, characters, what is happening, register/tones) — purely
     descriptive, NO term suggestions, NO "suspicious line" lists, NO
     opinions: the summary is context, not legislation. Parsed leniently: an
     unusable answer degrades to "no global summary" with a WARNING, it
     never fails the review.
  3. chunked fine-tuning — the thinking-ON chunk loop. This is the
     semantic gate: it is the only step allowed to change text, and every
     change goes through `_reject_fix`. Each chunk prompt carries the
     step-2 summary/style profile as a 全局背景 section, plus any per-line
     review comments for lines in that chunk.
  4. critic_pass — ONE thinking-OFF call over the full text + context
     profile. It only *describes* problems ("line: problem"), never
     rewrites lines. Failure degrades to "no critic round" with a WARNING.
  5. targeted re-review + convergence — re-run ONLY the chunks holding
     commented lines, carrying those comments. Repeat steps 4-5 for at
     most `_MAX_CRITIC_ROUNDS` critic rounds, stopping early when a critic
     round raises nothing actionable or a targeted re-review changes no
     text. A line commented on in two consecutive rounds whose text still
     did not change is downgraded to 存疑 and excluded from later critic
     rounds (resolved-unfixable) so rounds cannot ping-pong on it.

Key differences from proofread:
  - the LLM is allowed to think in step 3/5 (call_llm(with_thinking=True))
  - the instruction demands evidence for every change and forbids
    speculative fixes of ASR-garbled source (garbled lines are flagged
    as 存疑 for human review instead of being silently "corrected")
  - only lines that actually change/flag are emitted, but "quiet" counts
    as verified only when the response actually parsed: a chunk whose
    response yields no usable [N] result is reported as a failed chunk,
    never as verified

Writes:
  - updated translated SRT (text lines only; timing preserved)
  - a REVIEW report (REVIEW-<translated-stem>.md next to the output):
    the global analysis, every round's diff, every critic round's
    comments, and the release verdict — so a human only has to review the
    flagged subset.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.config import TranslateConfig
from src.translate import (
    AllChunksFailedError,
    SrtBlock,
    build_instruction,
    call_llm,
    make_snapshot,
    parse_srt,
    write_translated_srt,
)
from src.adjudicate import kana_normalize

logger = logging.getLogger(__name__)


# ── Reference SRT anchoring ───────────────────────────────────────────

_TS_RE = re.compile(r"^(\d+):(\d+):(\d+)[,.](\d+)")


def _ts_seconds(ts: str) -> float | None:
    m = _TS_RE.match(ts)
    if not m:
        return None
    h, mi, s, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms) / 1000.0


def _block_times(blk: SrtBlock) -> tuple[float, float] | None:
    parts = blk.ts_line.split("-->")
    if len(parts) != 2:
        return None
    a = _ts_seconds(parts[0].strip())
    b = _ts_seconds(parts[1].strip())
    if a is None or b is None:
        return None
    return a, b


def reference_anchors(
    ref: list[SrtBlock],
    source: list[SrtBlock],
    max_ref_lines: int = 800,
) -> list[tuple[int, int, list[SrtBlock]]]:
    """For each source line, find reference lines whose time ranges overlap.

    Returns [(src_1based, ref_1based, ref_blocks), ...] for source lines
    that have at least one overlapping reference block. Reference blocks
    are matched purely by timestamp overlap, so the two files do not need
    identical segmentation.
    """
    ref = ref[:max_ref_lines]
    ref_times = [(_block_times(b), b) for b in ref]
    out: list[tuple[int, int, list[SrtBlock]]] = []
    for i, src in enumerate(source):
        st = _block_times(src)
        if st is None:
            continue
        hits: list[SrtBlock] = []
        for rt, rb in ref_times:
            if rt is None:
                continue
            if rt[0] < st[1] and rt[1] > st[0]:
                hits.append(rb)
        if hits:
            out.append((i + 1, i + 1, hits))
    return out


# ── Step 3/5: chunked fine-tuning instruction ─────────────────────────

REVIEW_INSTRUCTION = (
    "你是字幕语义终审员。逐条验证 {source_lang}→{target_lang} 字幕的每条译文，"
    "对每条给出明确判定【只输出需要标记的行】。\n"
    "【放行标准】一行只有在「基于现有全部信息（源文+下方上下文+全局背景+参考字幕）」"
    "已确认无法再优化时才允许静默放行；拿不准、证据不足、或能给出更有把握的译法，"
    "都必须输出标记，绝不默认放过。\n"
    "每条编号 [N] 给出：源文、当前译文，可能附带参考字幕（同时间戳的人工/历史"
    "译文，分段可能不同——源文是ASR生成的可能有错字，参考可能比源文更接近真实"
    "台词，冲突时参考字幕优先）。\n"
    "【证据优先级】若提供了「台本参照」（作品的台词原稿），它是最高可信度证据："
    "台本 > 参考字幕 > 音频复听候选。台本与音频候选冲突时以台本为准——音频只是"
    "听感，台本是白纸黑字的原稿。\n"
    "【编号规则】待审行编号 [1]、[2]… 每组从 1 重新开始（1 到本组行数），"
    "输出时必须使用这个组内编号，不要使用全文行号。\n"
    "【上下文行】标记为 [前N]/[后N] 的行是本组前后的相邻行，只供理解语境，"
    "不在待审范围内：绝不要输出 [前N]/[后N] 行，也绝不要引用它们的编号。\n"
    "逐条检查（结合下方全局语境与参考字幕）：\n"
    "1. 语义错误：主语/角色反转、否定丢失、内容来自相邻行、漏译、未翻译的源语言残留\n"
    "2. 术语/一致性：译文里的专业词与源词不符、同一源词全文译法漂移\n"
    "3. 风格：礼貌层级/角色腔调与全局语境不符、书面语与口语混杂\n"
    "4. 过度翻译：源文只是语气词/断句片段却被扩写成完整句子、添加源文没有的内容\n"
    "5. 源文疑似ASR识别错误（词语不通、与上下文/参考字幕语义冲突）\n"
    "【同音词brainstorm（检查第5项前必做，在思考中完成）】对任何读起来不通顺、\n"
    "或译得蹩脚的源文行，先在思考中列出最多3个同音/近音候选词（源文是ASR生成，\n"
    "很可能把A词误识别为B词），再结合上下文和参考字幕判断哪个最可能是真实\n"
    "台词；只有参考字幕或上下文能明确支持某个候选时才按该候选修正译文，\n"
    "候选之间证据相当或都不够充分时按存疑处理，绝不在候选之间瞎猜。源文读\n"
    "得通则跳过此步。\n"
    "输出格式（每行一条，只输出需要处理的 [N] 行）：\n"
    "[N] 修改|<修正译文>\n"
    "   —— 仅限你高置信度判定译文确实错了，且修正只依据源文+上下文+参考字幕，"
    "不臆造剧情\n"
    "[N] 存疑|<一句话说明>\n"
    "   —— 源文疑似ASR错误、或证据不足以定夺。此时【不要改动译文】，只标记供人工"
    "复核\n"
    "确认无法再优化的行：不输出（即放行）。绝不输出没有 [N] 编号的前缀行。\n"
)

# ── Step 2: whole-file global pass instruction ────────────────────────

GLOBAL_INSTRUCTION = (
    "你是字幕语境分析员。下面是 {source_lang}→{target_lang} 字幕的全文，"
    "每行格式为「[行号] 源文 → 当前译文」。\n"
    "通读全文后输出两节：\n"
    "【语境总结】要求：\n"
    "1. 简短：6-10 句以内，只陈述事实性语境，不评价、不建议、不打分。\n"
    "2. 只描述：题材、场景、出场人物及其关系、正在发生的事、"
    "整体的语言风格与礼貌层级（如敬语/口语/粗俗）、各角色的大致腔调。\n"
    "【可疑行】要求：\n"
    "1. 列出你怀疑「源文疑似ASR听错」或「译文明显有问题」的行。\n"
    "2. 每行一条，格式：行号: 理由。只描述现象，例如「源文词语不通」"
    "「与上下文语义冲突」「疑似听错（音相近）」，绝不给出改法、"
    "绝不给统一译法建议。\n"
    "3. 没有可疑行就只输出「无」。\n"
    "严禁输出：术语统一译法建议（如「某词应译为某词」）、具体修正译文、"
    "质量评分。你的输出只会被当作给其他审校员的背景资料，"
    "任何评判性内容都会误导后续审校。\n"
    "只输出【语境总结】和【可疑行】两节，不要输出任何其他内容。\n"
)

# ── Step 4: critic instruction ────────────────────────────────────────

CRITIC_INSTRUCTION = (
    "你是字幕复核批评员（第二读者）。下面是 {source_lang}→{target_lang} 字幕的"
    "全文，每行格式为「[行号] 源文 → 当前译文」，并附全片语境总结。\n"
    "你的唯一任务是【指出问题】，不是改写。重点看：\n"
    "1. 语义：主语/角色反转、否定丢失、内容串到相邻行、漏译、未翻译残留、"
    "把语气词扩写成整句\n"
    "2. 风格：礼貌层级或角色腔调与语境总结不符、书面语与口语混杂、语气突变\n"
    "3. 术语一致性：同一源词在不同行译法不一致（指出应统一成哪个说法）\n"
    "输出格式（每行一条，只输出有问题的行）：\n"
    "行号: 问题描述\n"
    "【严禁】输出整行改写后的译文，只描述问题；也不要复述原文。\n"
    "没有发现问题就只输出「无」。除这些条目外不要输出任何内容。\n"
)


_REVIEW_LINE_RE = re.compile(r"^\[(\d+)\]\s*(修改|存疑)\s*[|｜:：]?\s*(.*)$")
_INDEXED_LINE_RE = re.compile(r"^\[(\d+)\]")

# A "line-number: reason" comment from the global pass / critic pass.
# Tolerates the shapes models actually emit: "12: …", "[12]: …", "#12: …",
# "第12行: …", "行12: …", optionally behind a bullet. A colon is REQUIRED —
# without it an ordinary prose line starting with a digit would be read as
# a comment. (8/30: "行N:" — 数字前直接带"行"字 — was a real model output
# shape the regex missed; 6/8 tracks' critic rounds silently degraded.)
_COMMENT_LINE_RE = re.compile(
    r"^(?:(?:行号\s*[:：])|行|第)?\s*\[?#?\s*(\d+)\s*\]?\s*(?:行)?\s*[:：]\s*(.+)$"
)
# 8/30: some critic answers emit "[N] 描述" with NO colon (06 track). The
# bracketed [N] prefix is distinctive enough to accept without a colon —
# ordinary prose does not start with "[<line number>]".
_BRACKET_ONLY_RE = re.compile(r"^\[\s*(\d+)\s*\]\s*(.+)$")
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•·]|\d+\.)\s+")
# Section headers of the step-2 answer. Matched loosely (the 【】 brackets are
# optional) because that is the part models most often drop.
_SECTION_RES: tuple[tuple[str, re.Pattern], ...] = (
    ("summary", re.compile(r"^\s*[【\[]?\s*(?:语境总结|内容概要|概要|内容摘要|总结|小结)\s*[】\]]?\s*[:：]?\s*$")),
    ("suspects", re.compile(r"^\s*[【\[]?\s*(?:可疑行|可疑|疑似问题行|问题行)\s*[】\]]?\s*[:：]?\s*$")),
)
# "nothing to report" answers for the comment-list sections.
_NO_ITEMS = frozenset({"无", "无。", "（无）", "(无)", "none", "None", "无可疑行", "没有"})
# An answer at most this long with no parsed item is read as an explicit
# "nothing to report"; anything longer is prose the parser failed on.
_EXPLICIT_NONE_CHARS = 12

# Values the model emits when it means "nothing to change" — never subtitle
# text. Matched ANCHORED (whole value, or the start of it), never as a
# substring: 正确 / 保持 / 存疑 are ordinary Chinese words that legitimately
# occur inside a translated line ("你说得完全正确。", "请保持安静。").
_META_EXACT = frozenset({
    "无需修改", "无须修改", "不需要修改", "无需更改", "无改动", "不修改",
    "正确", "无误", "没有问题", "无问题",
    "保持原样", "保持不变", "保持原译", "维持原样",
    "存疑", "同上", "无",
})
# Trailing punctuation to ignore when comparing against _META_EXACT.
_META_PUNCT = " \t。．.!！?？，,、；;：:\"'“”‘’()（）"
# Meta-commentary the model prefixes to a non-answer, e.g. "该行译文正确" or
# "本行无需修改". Anchored at the start so ordinary sentences are unaffected.
_META_PREFIX_RE = re.compile(
    r"^\s*(?:该行|此行|本行|这一行|这行|原译|原文)?\s*(?:译文|翻译)?\s*"
    r"(?:已经|已|完全)?\s*"
    r"(?:无需修改|无须修改|不需要修改|无需更改|不修改|正确|无误|没有问题|"
    r"无问题|保持原样|保持不变|保持原译|维持原样)"
)
# The model re-emitting the output grammar instead of a translation.
_META_GRAMMAR_RE = re.compile(r"^\s*\[?\s*\d*\s*\]?\s*(?:修改|存疑)\s*[|｜:：]")

# Length sanity for a 修改 value, keyed off the SOURCE line, not the current
# translation. The translation is the thing under suspicion — an
# over-translated line (instruction rule 4) is long precisely because it is
# wrong, so measuring the fix against it would reject exactly the correction
# being asked for. The source line is the stable reference: it says how much
# was actually said, so it can tell "a long line collapsing to 嗯 because the
# source is あっ" (legitimate) apart from "a long line collapsing to 嗯 with a
# full sentence as source" (a dropped translation).
_MAX_FIX_RATIO = 3.0
_MIN_FIX_RATIO = 0.2
# Absolute escape hatches, needed because CJK is dense: any fix up to
# _MAX_FIX_ABSOLUTE chars is allowed however short the source, and a fix may
# always shrink to within _MIN_FIX_SHRINK_FLOOR chars of nothing.
_MAX_FIX_ABSOLUTE = 20
_MIN_FIX_SHRINK_FLOOR = 8
# A source this short is an interjection/sound effect. It has no meaningful
# lower bound at all — あっ may render as 啊 or as 啊，等一下 — so only the
# absolute ceiling applies.
_SHORT_SOURCE_CHARS = 3

# Convergence control (step 5).
_MAX_CRITIC_ROUNDS = 3
# Caps on text injected back into per-chunk prompts. The global sections and
# critic comments are model output: bounded here so one verbose answer cannot
# crowd the actual subtitle lines out of every later chunk request.
_MAX_SUMMARY_CHARS = 1200
_MAX_NOTE_CHARS = 160


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _flat(text: str) -> str:
    """One subtitle line as a single physical line (global numbering needs it)."""
    return " ".join(text.split())


# ── Global pass (step 2) / critic pass (step 4) ───────────────────────


@dataclass
class GlobalProfile:
    """Whole-file understanding produced by step 2.

    All fields are best-effort: an unparseable or failed global pass yields
    an empty profile, and every consumer treats that as "no extra evidence"
    rather than an error.

    v1.3 (FTDC): `suspects` is a line-number → reason list for lines the
    global pass flags as suspicious (ASR-garbled source, semantic conflict,
    etc.). It only *describes* the phenomenon — never a fix or a unified
    term — so it cannot hijack terminology (8/17 lesson).
    """

    summary: str = ""
    suspects: dict[int, list[str]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.summary and not self.suspects

    def background_section(self) -> str:
        """The 全局语境 block appended to every step-3/5 chunk instruction."""
        if not self.summary:
            return ""
        return (
            "\n=== 全局语境（通读全文得出，仅供理解剧情与说话人；"
            "不要翻译、也不要输出本节）===\n"
            "【语境总结】\n"
            f"{_clip(self.summary, _MAX_SUMMARY_CHARS)}\n"
            "=== 全局语境结束 ===\n"
        )


@dataclass
class CriticRecord:
    """One critic round (step 4): what it said and what was usable."""

    label: str
    comments: dict[int, list[str]] = field(default_factory=dict)
    out_of_range: list[int] = field(default_factory=list)
    excluded: list[int] = field(default_factory=list)
    failed: bool = False
    # The call returned prose that yielded no "line: problem" item. Kept
    # apart from an explicit "无": no comments parsed is NOT evidence that
    # the critic found nothing, so it must never be reported as convergence.
    unparsed: bool = False


@dataclass
class ReviewRound:
    """One chunked fine-tuning pass (step 3 for R1, step 5 afterwards)."""

    label: str
    chunk_indices: list[int] = field(default_factory=list)
    notes: dict[int, list[str]] = field(default_factory=dict)
    flags: list[tuple[int, str, str]] = field(default_factory=list)
    changed_lines: list[int] = field(default_factory=list)
    downgraded: list[int] = field(default_factory=list)
    applied: int = 0
    rejected: int = 0
    failed_chunks: int = 0
    # Global 1-based line numbers covered by each failed chunk (empty when
    # the round had none). Kept so the report can name the unverified lines
    # instead of just counting them (8/30: 04 track's 1 failed chunk had no
    # way to locate which lines were left unreviewed).
    failed_blocks: list[list[int]] = field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        return len(self.chunk_indices)


def _parse_comment_items(
    text: str,
    max_line: int,
    label: str,
) -> tuple[dict[int, list[str]], list[int]]:
    """Parse "line-number: problem" items into {1-based line: [comments]}.

    Returns (comments, out_of_range). Line numbers outside 1..max_line are
    collected instead of applied — the caller warns and skips them, because
    a comment mapped onto the wrong line would send the next round to
    re-review a line nobody complained about.
    """
    comments: dict[int, list[str]] = {}
    out_of_range: list[int] = []
    for raw in text.splitlines():
        line = _BULLET_PREFIX_RE.sub("", raw).strip()
        if not line or line in _NO_ITEMS:
            continue
        m = _COMMENT_LINE_RE.match(line)
        if m is None:
            # "[N] 描述" without a colon (seen from the critic, 06 track).
            m = _BRACKET_ONLY_RE.match(line)
            if m is None:
                continue
            ln = int(m.group(1))
            note = _clip(m.group(2), _MAX_NOTE_CHARS)
        else:
            ln = int(m.group(1))
            note = _clip(m.group(2), _MAX_NOTE_CHARS)
        if not note:
            continue
        if not 1 <= ln <= max_line:
            out_of_range.append(ln)
            continue
        comments.setdefault(ln, []).append(note)
    if out_of_range:
        logger.warning(
            "%s: skipped %d comment(s) with out-of-range line number "
            "(file has %d lines): %s",
            label, len(out_of_range), max_line, out_of_range[:10],
        )
    return comments, out_of_range


def _parse_global_response(text: str, max_line: int) -> GlobalProfile:
    """Extract the 【语境总结】 and 【可疑行】 sections from the step-2 answer.

    Deliberately forgiving. If the section header is missing entirely the
    whole answer is kept as the summary, so a model that ignored the output
    shape still contributes context instead of nothing. Suspect lines are
    parsed with the shared _COMMENT_LINE_RE (handles "12: …", "行号: 12: …",
    "#12: …", bullets) and clipped to the file's line count.
    """
    if not text or not text.strip():
        return GlobalProfile()

    buckets: dict[str, list[str]] = {"summary": [], "suspects": []}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        matched = False
        for name, pattern in _SECTION_RES:
            if pattern.match(line):
                current = name
                matched = True
                break
        if matched:
            continue
        if current is not None:
            buckets[current].append(line)

    summary = "\n".join(buckets["summary"]).strip()
    if not summary:
        # No recognisable header: treat the whole answer as the summary.
        logger.warning(
            "Global pass: no 【语境总结】 section header found — keeping the "
            "whole answer as the summary"
        )
        summary = _clip(text.strip(), _MAX_SUMMARY_CHARS)

    suspects: dict[int, list[str]] = {}
    if buckets["suspects"]:
        comments, out_of_range = _parse_comment_items(
            "\n".join(buckets["suspects"]), max_line, "Global-suspects")
        suspects = comments
        if out_of_range:
            logger.warning(
                "Global pass: %d suspect line number(s) out of range: %s",
                len(out_of_range), out_of_range[:10])

    return GlobalProfile(summary=summary, suspects=suspects)


def _format_full_text(
    source: list[SrtBlock],
    translated: list[SrtBlock],
    focus: list[int] | None = None,
) -> str:
    """The whole file as "[global line] source → translation" lines.

    FTDC (focus mode): when `focus` is given, only those global 1-based
    lines plus their ±context neighbours are included (global numbering
    preserved), so the critic re-checks a small neighbourhood instead of the
    whole file.
    """
    if focus is not None:
        focus_set = set(focus)
        idxs: list[int] = []
        for ln in sorted(focus_set):
            for j in range(max(1, ln - 2), min(len(source), ln + 2) + 1):
                if j not in idxs:
                    idxs.append(j)
        idxs.sort()
        return "\n".join(
            f"[{i}] {_flat(source[i - 1].text)} → {_flat(translated[i - 1].text)}"
            for i in idxs
        )
    return "\n".join(
        f"[{i + 1}] {_flat(src.text)} → {_flat(translated[i].text)}"
        for i, src in enumerate(source)
    )


def global_pass(
    source: list[SrtBlock],
    translated: list[SrtBlock],
    config: TranslateConfig,
) -> GlobalProfile:
    """Step 2: one thinking-OFF whole-file read.

    Never raises: a transport/LLM failure or an unusable answer degrades to
    an empty profile with a WARNING, because the review's semantic gate is
    step 3 and it must still run.
    """
    if not source:
        return GlobalProfile()
    instruction = build_instruction(config, base_template=GLOBAL_INSTRUCTION)
    content = _format_full_text(source, translated)
    logger.info(
        "Global pass (step 2): reading %d lines (%d chars) in one call",
        len(source), len(content),
    )
    try:
        response = call_llm(content, instruction, config, with_thinking=False)
    except RuntimeError as e:
        logger.warning(
            "Global pass failed, continuing without a global summary: %s", e
        )
        return GlobalProfile()

    profile = _parse_global_response(response, len(source))
    if profile.is_empty:
        logger.warning(
            "Global pass: nothing usable parsed from the response (%r) — "
            "continuing without a global summary",
            response.strip()[:120],
        )
        return profile
    logger.info(
        "Global pass: context summary %d chars",
        len(profile.summary),
    )
    return profile


def critic_pass(
    source: list[SrtBlock],
    translated: list[SrtBlock],
    profile: GlobalProfile,
    config: TranslateConfig,
    label: str,
    focus: list[int] | None = None,
    script_notes: dict[int, list[str]] | None = None,
) -> CriticRecord:
    """Step 4: one thinking-OFF whole-file critique, comments only.

    Never raises: a failed call is recorded as `failed=True` and the caller
    degrades to "no critic round". The critic may not rewrite lines — only
    step 3/5 changes text, so its output is parsed as comments only.

    FTDC (focus mode): `focus` restricts the critique to those global 1-based
    lines ± a small neighbourhood (incremental re-check), not the whole file.
    `script_notes` (line → 台本参照 evidence) is injected so the critic
    cannot overrule script ground truth using the global summary alone.
    """
    if not source:
        return CriticRecord(label=label)
    instruction = build_instruction(config, base_template=CRITIC_INSTRUCTION)
    if profile.summary:
        instruction += (
            "\n=== 全局语境（仅供理解剧情与说话人；不要输出本节）===\n"
            f"{_clip(profile.summary, _MAX_SUMMARY_CHARS)}\n"
            "=== 全局语境结束 ===\n"
        )
    if script_notes:
        parts = []
        for ln in sorted(script_notes):
            for note in script_notes[ln]:
                parts.append(f"#行{ln}: {note}")
        instruction += (
            "\n=== 台本参照（台词原稿，最高可信度，优先于全局语境与音频候选；"
            "只用于理解该行真实台词，不要输出本节）===\n"
            + "\n".join(parts)
            + "\n=== 台本参照结束 ===\n"
        )
    content = _format_full_text(source, translated, focus=focus)
    logger.info("%s: critiquing %d line(s) (%d chars) in one call%s",
                label, len(source) if focus is None else len(content.splitlines()),
                len(content),
                " (focused)" if focus is not None else "")
    try:
        response = call_llm(content, instruction, config, with_thinking=False)
    except RuntimeError as e:
        logger.warning("%s: LLM call failed — no critic round: %s", label, e)
        return CriticRecord(label=label, failed=True)

    comments, out_of_range = _parse_comment_items(response, len(source), label)
    stripped = response.strip()
    # "No items parsed" is only an all-clear when the model actually said so.
    # Prose or a collapsed format means the critique is unknown, not empty.
    unparsed = bool(
        not comments and not out_of_range
        and stripped not in _NO_ITEMS and len(stripped) > _EXPLICIT_NONE_CHARS
    )
    if unparsed:
        logger.warning(
            "%s: no 行号: 问题 item parsed from the response — this round "
            "yields NO usable critique (not an all-clear): %r",
            label, stripped[:120],
        )
    else:
        logger.info("%s: %d line(s) commented on", label, len(comments))
    return CriticRecord(
        label=label, comments=comments, out_of_range=out_of_range,
        unparsed=unparsed,
    )


# ── Chunked fine-tuning (steps 3 and 5) ───────────────────────────────


def _parse_review_response(
    text: str,
    core_count: int,
    label: str = "Review",
) -> dict[int, tuple[str, str]]:
    """Parse [N] 修改|... / [N] 存疑|... lines from a review response.

    Returns {1-based chunk index: (kind, value)} where kind is 'fix' or
    'flag'. Indices are chunk-relative — the same space the model was shown
    (context lines carry [前N]/[后N] markers and are never numbered in it).

    Anything [N]-shaped that falls outside that space or does not match the
    grammar is counted and warned about rather than dropped: the caller
    reports every unflagged line as "verified OK", so a silent drop would
    turn a mis-formatted answer into a false all-clear.
    """
    out: dict[int, tuple[str, str]] = {}
    out_of_range: list[int] = []
    unrecognized: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        m = _REVIEW_LINE_RE.match(line)
        if m is None:
            if _INDEXED_LINE_RE.match(line):
                unrecognized.append(line)
            continue
        idx = int(m.group(1))
        value = m.group(3).strip()
        if not 1 <= idx <= core_count:
            out_of_range.append(idx)
        elif value:
            out[idx] = ("fix" if m.group(2) == "修改" else "flag", value)
        else:
            unrecognized.append(line)
    if out_of_range:
        logger.warning(
            "%s: dropped %d result(s) with out-of-range index (chunk has %d "
            "lines): %s",
            label, len(out_of_range), core_count, out_of_range[:10],
        )
    if unrecognized:
        logger.warning(
            "%s: dropped %d [N]-shaped line(s) not matching 修改|/存疑|: %s",
            label, len(unrecognized), unrecognized[:3],
        )
    return out


def _is_meta_value(value: str) -> str | None:
    """The meta phrase `value` consists of, or None if it is real text.

    Anchored on purpose — see _META_EXACT. A line that merely *contains*
    正确 / 保持 / 存疑 somewhere in the middle is ordinary subtitle text.
    """
    stripped = value.strip().strip(_META_PUNCT)
    if stripped in _META_EXACT:
        return stripped
    m = _META_PREFIX_RE.match(value)
    if m is not None:
        return m.group(0).strip()
    m = _META_GRAMMAR_RE.match(value)
    if m is not None:
        return m.group(0).strip()
    return None


def _visible_len(text: str) -> int:
    """Character count of a subtitle line, ignoring line breaks/padding."""
    return len(text.replace("\n", "").strip())


def _reject_fix(value: str, old: str, src: str = "") -> str | None:
    """Reason to refuse a 修改 value, or None when it is safe to apply.

    The value goes straight into the subtitle, so meta-commentary the model
    emitted instead of a translation ("无需修改", "该行正确，保持原样") and
    fragments/runaways wildly off the source line's length are rejected.

    `src` is the ORIGINAL-language line this translation belongs to and is
    what the length bounds are measured against — see the constants above.
    It falls back to `old` only when a caller has no source line to offer.

    Bounds, with n = length of the source line:
        n <= _SHORT_SOURCE_CHARS   accept anything up to _MAX_FIX_ABSOLUTE
        otherwise                  reject len > max(3n, _MAX_FIX_ABSOLUTE)
                                   reject len < max(1, min(0.2n,
                                                   n - _MIN_FIX_SHRINK_FLOOR))
    """
    if not value:
        return "空值"
    meta = _is_meta_value(value)
    if meta is not None:
        return f"元标记“{meta}”"

    ref = _visible_len(src) or _visible_len(old)
    if not ref:
        return None
    new_len = _visible_len(value)

    max_len = max(ref * _MAX_FIX_RATIO, _MAX_FIX_ABSOLUTE)
    if new_len > max_len:
        return f"过长（{new_len} 字符 vs 源文 {ref}，上限 {max_len:.0f}）"
    if ref <= _SHORT_SOURCE_CHARS:
        # Interjection/sound-effect source: no lower bound is meaningful.
        return None
    min_len = max(1, min(ref * _MIN_FIX_RATIO, ref - _MIN_FIX_SHRINK_FLOOR))
    if new_len < min_len:
        return f"过短（{new_len} 字符 vs 源文 {ref}，下限 {min_len:.0f}）"
    return None


def _format_chunk_input(
    source: list[SrtBlock],
    translated: list[SrtBlock],
    start: int,
    end: int,
    context_lines: int,
    anchors: dict[int, list[SrtBlock]],
    notes: dict[int, list[str]],
    focus: list[int] | None = None,
) -> str:
    """Format one review chunk: context pairs + [N] source→translation core.

    Context lines are marked [前N]/[后N] (N = distance from the chunk) and
    are never numbered: the ONLY numbers the model sees are the
    chunk-relative core indices it is asked to answer in.

    `notes` carries earlier-round comments (step 2 candidates for R1, critic
    comments afterwards) for lines in this chunk. They are attached to their
    own line so the model does not have to map global line numbers itself.

    FTDC (focus mode): when `focus` is given, the core is NOT a contiguous
    window — it is exactly the suspicious lines in `focus` (global 0-based
    indices), each carrying its own [前N]/[后N] neighbourhood. This packs all
    problem lines into one chunk without re-scanning verified lines, and the
    model numbers them [1..len(focus)] in the order given.
    """
    notes = notes or {}
    if focus is not None:
        return _format_chunk_focused(
            focus, source, translated, context_lines, anchors, notes)
    parts: list[str] = []
    for i in range(max(0, start - context_lines), start):
        parts.append(f"[前{start - i}] {source[i].text} → {translated[i].text}")
    flagged = 0
    for i in range(start, end):
        line_no = i + 1
        entry = f"[{i + 1 - start}] {source[i].text} → {translated[i].text}"
        ref_bks = anchors.get(line_no)
        if ref_bks:
            entry += f"\n    参考: {' / '.join(b.text for b in ref_bks)}"
        note = notes.get(line_no)
        if note:
            entry += f"\n    审阅意见: {'；'.join(note)}"
            flagged += 1
        parts.append(entry)
    for i in range(end, min(len(source), end + context_lines)):
        parts.append(f"[后{i - end + 1}] {source[i].text} → {translated[i].text}")
    body = "\n".join(parts)
    if flagged:
        body = (
            "【重点复核】带「审阅意见」的行已被此前的审阅指出问题，请结合源文、"
            "上下文与全局背景重新判断：确实有误就输出 [N] 修改|…；确认当前译文"
            "已是最佳（或证据不足）就输出 [N] 存疑|… 或不输出该行。其余行照常审阅。\n"
            + body
        )
    return body


def _format_chunk_focused(
    focus: list[int],
    source: list[SrtBlock],
    translated: list[SrtBlock],
    context_lines: int,
    anchors: dict[int, list[SrtBlock]],
    notes: dict[int, list[str]],
) -> str:
    """FTDC focus 模式：core 只含可疑行，每行带自己的上下文邻域。"""
    parts: list[str] = []
    flagged = 0
    for pos, i in enumerate(focus, start=1):
        line_no = i + 1
        for d in range(context_lines, 0, -1):
            j = i - d
            if j >= 0:
                parts.append(f"[前{d}] {source[j].text} → {translated[j].text}")
        entry = f"[{pos}] {source[i].text} → {translated[i].text}"
        ref_bks = anchors.get(line_no)
        if ref_bks:
            entry += f"\n    参考: {' / '.join(b.text for b in ref_bks)}"
        note = notes.get(line_no)
        if note:
            entry += f"\n    审阅意见: {'；'.join(note)}"
            flagged += 1
        parts.append(entry)
        for d in range(1, context_lines + 1):
            j = i + d
            if j < len(source):
                parts.append(f"[后{d}] {source[j].text} → {translated[j].text}")
    body = "\n".join(parts)
    if flagged:
        body = (
            "【重点复核】带「审阅意见」的行已被此前的审阅指出问题（含音频复听证据），"
            "请结合源文、上下文、全局背景与审阅意见重新判断：确实有误就输出 "
            "[N] 修改|…；确认当前译文已是最佳（或证据不足）就输出 [N] 存疑|… "
            "或不输出该行。\n"
            + body
        )
    return body


def _review_chunk(
    ci: int,
    source: list[SrtBlock],
    translated: list[SrtBlock],
    instruction: str,
    config: TranslateConfig,
    chunk_size: int,
    anchors: dict[int, list[SrtBlock]],
    notes: dict[int, list[str]],
    label: str,
    round_rec: ReviewRound,
    focus: list[int] | None = None,
) -> bool:
    """Review chunk `ci` (thinking-ON) and apply its accepted fixes in place.

    Returns True when the chunk produced a usable answer. Counters and flags
    are accumulated into `round_rec`. This is the only place that mutates
    subtitle text, and every 修改 passes `_reject_fix` first.

    FTDC (focus mode): `focus` is a list of global 0-based line indices for
    this chunk's suspicious lines. When given, chunk-relative [N] results
    map back through `focus`; `ci`/`chunk_size` only select which block of
    the packed focus list this chunk owns.
    """
    if focus is not None:
        core_count = len(focus)
        numbered_input = _format_chunk_input(
            source, translated, 0, 0,
            config.review_context_lines, anchors, notes, focus=focus,
        )
    else:
        start = ci * chunk_size
        end = min(start + chunk_size, len(source))
        core_count = end - start
        numbered_input = _format_chunk_input(
            source, translated, start, end,
            config.review_context_lines, anchors, notes,
        )

    try:
        response = call_llm(numbered_input, instruction, config, with_thinking=True)
    except RuntimeError as e:
        logger.error("%s: LLM call failed, chunk left unreviewed: %s", label, e)
        return False

    results = _parse_review_response(response, core_count, label)
    if not results:
        # "Silence means verified" only holds for a response that parsed.
        # Prose, a refusal or an empty answer must fail the gate instead of
        # releasing the whole chunk as verified.
        logger.error(
            "%s: no usable [N] result parsed from the response — chunk "
            "left unreviewed (NOT counted as verified)", label,
        )
        return False

    chunk_rejected = 0
    for rel, (kind, value) in sorted(results.items()):
        i = focus[rel - 1] if focus is not None else ci * chunk_size + rel - 1
        line_no = i + 1
        if kind == "fix":
            old = translated[i].text
            # Length sanity is measured against the source line, not the
            # current translation: an over-translated line is long because
            # it is wrong, so it cannot be the reference for its own fix.
            reason = _reject_fix(value, old, source[i].text)
            if reason is not None:
                # A refused 修改 must never be a silent release: the model
                # asserted this line is wrong and the gate blocked its
                # replacement, so the line goes to the human as 存疑.
                logger.warning(
                    "%s: #%d 修改被拒绝（%s）: %r", label, line_no, reason, value,
                )
                round_rec.rejected += 1
                chunk_rejected += 1
                round_rec.flags.append((
                    line_no, "存疑",
                    f"模型提出修改但被校验拒绝（{reason}）；"
                    f"被拒译文: {value}；当前译文保持: {old}",
                ))
                continue
            if value != old:
                translated[i].text = value
                round_rec.applied += 1
                round_rec.flags.append((line_no, "修改", f"{old} → {value}"))
                if config.verbose:
                    logger.debug("#%d [修改] %s → %s", line_no, old, value)
        else:
            round_rec.flags.append((line_no, "存疑", value))
            if config.verbose:
                logger.debug("#%d [存疑] %s", line_no, value)
    logger.info(
        "%s: %d flagged/changed (%d 修改被拒绝), %d verified OK",
        label, len(results), chunk_rejected, core_count - len(results),
    )
    return True


def _run_round(
    round_label: str,
    chunk_indices: list[int],
    source: list[SrtBlock],
    translated: list[SrtBlock],
    instruction: str,
    config: TranslateConfig,
    chunk_size: int,
    anchors: dict[int, list[SrtBlock]],
    notes: dict[int, list[str]] | None = None,
    focus_lines: list[int] | None = None,
) -> ReviewRound:
    """Run one chunked fine-tuning round over `chunk_indices`.

    FTDC (focus mode): when `focus_lines` is given, the round re-packs those
    global 1-based suspicious lines into chunks of `chunk_size` and reviews
    only them (thinking-ON). `chunk_indices` is then informational only.
    """
    if focus_lines is not None:
        focus_0based = sorted({ln - 1 for ln in focus_lines})
        blocks = [
            focus_0based[k:k + chunk_size]
            for k in range(0, len(focus_0based), chunk_size)
        ]
        rec = ReviewRound(label=round_label, chunk_indices=list(blocks),
                          notes=dict(notes or {}))
        total = len(blocks)
        for n, block in enumerate(blocks, start=1):
            label = f"{round_label} {n}/{total}"
            if not _review_chunk(
                n - 1, source, translated, instruction, config, chunk_size,
                anchors, notes, label, rec, focus=block,
            ):
                rec.failed_chunks += 1
                rec.failed_blocks.append([ln + 1 for ln in block])
        return rec

    rec = ReviewRound(label=round_label, chunk_indices=list(chunk_indices), notes=dict(notes or {}))
    total = len(chunk_indices)
    for n, ci in enumerate(chunk_indices, start=1):
        label = f"{round_label} {n}/{total}"
        if not _review_chunk(
            ci, source, translated, instruction, config, chunk_size,
            anchors, notes, label, rec,
        ):
            rec.failed_chunks += 1
            start = ci * chunk_size
            rec.failed_blocks.append(
                [i + 1 for i in range(start, min(start + chunk_size, len(source)))]
            )
    return rec


# ── Review pass ───────────────────────────────────────────────────────


def _triage(
    source: list[SrtBlock],
    translated: list[SrtBlock],
    profile: GlobalProfile,
    adj_by_line: dict[int, dict],
    script_anchors: list | None = None,
) -> tuple[set[int], list[tuple[int, str, str]]]:
    """FTDC Phase 2: 合并信号 → 可疑行集合（1-based）。

    信号源：
     1. global pass 的【可疑行】（只描述现象）
     2. 规则层：英文残留 / 日文假名残留（未翻译）
     3. 音频仲裁：**B+ 全进**（双独立反对，语义最强）；
        **B- 全进**（2:1 多数可能重合错——实测 #395 的 テニス→デニス 只差
        1 个 kana 字符，整句 ratio 稀释，不能靠阈值筛）；
        **C 级不进**（三向读音分歧 = 音证据不足，DeepFix 无可依据，避免乱猜；
        由报告列出供人工）；
        A-音 不进（同音不同字，由报告列出，README 术语人工裁决）。
     4. 台本对齐（script_anchors，最高可信度）：mismatch（有台本对应但
        字面不符）→ 全进；**硬冲突保护**：mismatch 但音频三模型 A 级一致
        → 不进 DeepFix（台本可能与音频版本不一致），进 conflicts 交人工。
    """
    sus: set[int] = set(profile.suspects.keys())
    conflicts: list[tuple[int, str, str]] = []
    for i, tr in enumerate(translated):
        ln = i + 1
        t = tr.text
        if re.search(r"[A-Za-z]{3,}", t) and not re.search(r"[\u3040-\u30ff]", t):
            sus.add(ln)  # 英文残留
        if re.search(r"[\u3040-\u30ff]", t):
            sus.add(ln)  # 假名残留（未翻译）
    for ln, a in adj_by_line.items():
        if a.get("grade") in ("B+", "B-"):
            sus.add(ln)
    if script_anchors:
        for ln, a in enumerate(script_anchors, start=1):
            if a.status != "mismatch":
                continue
            adj = adj_by_line.get(ln)
            if adj and adj.get("grade") == "A":
                # 台本与音频三模型一致冲突：台本可能是别的版本/含未说出口的
                # 台词，不自动改 —— 报告交人工。从任何来源的可疑集合中移除
                # （global suspects / 规则 / 音频信号），防止 DeepFix 借台本
                # 证据改掉音频一致的行。
                conflicts.append((
                    ln, a.script_text,
                    f"台本与音频冲突（音频三模型一致={adj.get('n', '')!r}；"
                    f"相似度 {a.similarity:.2f}）",
                ))
                sus.discard(ln)
                continue
            sus.add(ln)
    return sus, conflicts


def review_srt(
    source_path: Path,
    translated_path: Path,
    config: TranslateConfig,
    ref_srt: Path | None = None,
    adjudication: Path | None = None,
) -> Path:
    """Run the FTDC semantic review (global → triage → deep-fix → critic).

    Updates translated_path in place (text lines only, timing preserved),
    keeping the pre-review file as <stem>.pre-review.srt beside it, and
    writes a REVIEW report listing every round, comment and flagged line.
    Returns the path to the report.

    FTDC (v1.3): when `adjudication` (a main.adjudication.json from
    src/adjudicate.py) is provided, the pass runs the fast path:
      - triage suspicious lines (rules + global suspects + audio evidence)
      - DeepFix runs thinking-ON ONLY on packed suspicious chunks
      - critic re-checks incrementally (≤2 rounds)
    Without it, the legacy full-file thinking-ON path runs unchanged.
    """
    source = parse_srt(source_path)
    translated = parse_srt(translated_path)

    if len(source) != len(translated):
        raise ValueError(
            f"Alignment check failed: source has {len(source)} blocks, "
            f"translated has {len(translated)} — refusing to review "
            "(would risk silent line shifts)"
        )

    # 台本穿透（script_anchors / script_conflicts 供 FTDC 分支与报告使用）
    script_anchors = None
    script_conflicts: list[tuple[int, str, str]] = []
    if config.script_file is not None:
        from src.script_align import build_script_anchors
        script_anchors = build_script_anchors(
            config.script_file, [b.text for b in source])

    # Evidence corpus: context files + vocab (+ reference anchors below).
    instruction = build_instruction(config, base_template=REVIEW_INSTRUCTION)

    ref_blocks = parse_srt(ref_srt) if ref_srt is not None else None
    anchors = {ln: bks for ln, _, bks in reference_anchors(ref_blocks, source)} if ref_blocks else {}
    if anchors:
        logger.info("Review: %d source lines anchored to reference SRT", len(anchors))

    chunk_size = max(8, config.review_chunk_size)
    total_chunks = (len(source) + chunk_size - 1) // chunk_size

    # ── Step 2: whole-file global pass (never fatal) ──────────────────
    profile = global_pass(source, translated, config)
    chunk_instruction = instruction + profile.background_section()

    logger.info(
        "Review: %d blocks in %d chunk(s) of %d — running thinking-enabled "
        "semantic review (global summary: %s)",
        len(source), total_chunks, chunk_size,
        "yes" if not profile.is_empty else "none",
    )

    # ── Step 3-5: DeepFix + critic（FTDC 分支）──────────────────────
    timeline: list[ReviewRound | CriticRecord] = []

    if adjudication is not None:
        # ── FTDC 快速路径（带音频仲裁证据）────────────────────────
        import json as _json
        adj_data = _json.loads(Path(adjudication).read_text(encoding="utf-8"))
        adj_by_line: dict[int, dict] = {}
        for a in adj_data:
            if isinstance(a, dict) and a.get("line"):
                adj_by_line[int(a["line"])] = a

        # 台本穿透：config.script_file（--script 或 script 类 context）→ 对齐
        if config.script_file is not None:
            logger.info("Review script anchors: %s", config.script_file)

        # 仲裁证据注入 chunk notes（B+/B- 带「音频复听」意见；
        # A-音 同音不同字由 adjudication.json 报告列出，不塞进 DeepFix）
        notes: dict[int, list[str]] = {}
        for ln, a in adj_by_line.items():
            g = a.get("grade")
            if g in ("B+", "B-") and a.get("note"):
                cand = a.get("n") or a.get("q") or ""
                notes[ln] = [f"音频复听[{g}]: {a['note']}" +
                             (f"（候选: {cand[:40]}）" if cand else "")]
        # 台本证据注入（mismatch 行：台本是原稿，优先级高于音频候选）
        # script_notes 同时供 critic_pass 与 R2+ 重校使用，防止台本证据
        # 在后续轮次丢失（global 语境可能基于 ASR 错字，把错词当角色名）。
        script_notes: dict[int, list[str]] = {}
        if script_anchors:
            for ln, a in enumerate(script_anchors, start=1):
                if a.status == "mismatch":
                    note = (f"台本参照（最高可信度，优先于音频候选）: {a.script_text[:60]}"
                            f"（相似度 {a.similarity:.2f}，与源文不符——按台本修正）")
                    script_notes[ln] = [note]
                    notes.setdefault(ln, []).append(note)

        # 标题同音锚定（8/30 教训：虚勢/去勢 同音不同字——标题 context 里的
        # 「去勢」从未被 LLM 关联到源文「虚勢」，它反而自由联想成不相干的俗语义。
        # 这里用 kana 读音把标题词与源文显式挂钩，作为「疑似 ASR 误识」证据）。
        # 优先级低于台本（台本=原稿，标题=提示），但高于无依据的自由联想。
        title_notes: dict[int, list[str]] = {}
        if config.title:
            from src.title_context import build_title_notes
            title_notes = build_title_notes(config.title, [b.text for b in source])
            for ln, tnotes in title_notes.items():
                notes.setdefault(ln, []).extend(tnotes)
            if title_notes:
                logger.info(
                    "Title homophone notes: %d line(s) — %s",
                    len(title_notes),
                    ", ".join(f"#{ln}" for ln in sorted(title_notes)[:12]),
                )

        suspicious, script_conflicts = _triage(
            source, translated, profile, adj_by_line, script_anchors)
        conflict_lns = {ln for ln, _, _ in script_conflicts}
        # 硬冲突行必须失去一切自动修复证据：triage 只把它挡在 R1 之外，但
        # 台本 mismatch 的「按台本修正」note 仍会经 critic 的台本参照区 →
        # R2 定向重校，把刚保护下来的行按台本改掉（smoke 场景 B 实测）。
        # 冲突行只进报告交人工，任何轮次不得携带其修复证据。
        for ln in conflict_lns:
            script_notes.pop(ln, None)
            notes.pop(ln, None)
        # 标题同音命中行必须进 DeepFix：正是三模型都写出同一个错别字
        # （虚勢/去勢 读音相同 → 音频仲裁 A 级"一致"）的行，只有标题证据
        # 能触发复核。台本硬冲突行仍交人工，不因标题证据重新进入。
        if title_notes:
            extra_lns = set(title_notes) - conflict_lns - suspicious
            if extra_lns:
                logger.info(
                    "Title homophone triage: +%d line(s) not flagged by any "
                    "other signal: %s",
                    len(extra_lns), sorted(extra_lns)[:12],
                )
            suspicious |= set(title_notes) - conflict_lns
        logger.info(
            "FTDC: %d/%d line(s) suspicious (global %d + rules + audio B+/B-/A-音"
            + (" + 台本 mismatch" if script_anchors else "")
            + ")",
            len(suspicious), len(source), len(profile.suspects))
        if script_conflicts:
            logger.warning(
                "台本/音频硬冲突 %d 行（不进 DeepFix，交人工）：%s",
                len(script_conflicts),
                ", ".join(f"#{ln}" for ln, _, _ in script_conflicts))

        r1 = _run_round(
            "R1", [], source, translated, chunk_instruction,
            config, chunk_size, anchors, notes or None,
            focus_lines=sorted(suspicious),
        )
        timeline.append(r1)

        n_focus_chunks = (len(suspicious) + chunk_size - 1) // chunk_size
        if n_focus_chunks and r1.failed_chunks == n_focus_chunks:
            raise AllChunksFailedError(
                f"Review failed: all {n_focus_chunks} DeepFix chunk(s) failed — "
                f"{translated_path} left untouched"
            )
        if r1.failed_chunks:
            logger.warning(
                "R1: %d/%d DeepFix chunk(s) failed and were left unreviewed — "
                "those lines are NOT verified",
                r1.failed_chunks, n_focus_chunks,
            )

        # Critic 增量（≤2 轮，只复查可疑行 ∪ 已改动行 ± 邻域）
        release_round = "R1"
        max_critic = min(_MAX_CRITIC_ROUNDS, 2)
        release_reason = f"未启用 critic 轮次（上限 {max_critic}）"
        # 硬冲突行从一开始就是 unfixable：critic 若仍对它们发表意见（邻域
        # 可见），意见被记为 excluded，绝不进入定向重校。
        unfixable: set[int] = set(conflict_lns)
        prev_commented: set[int] = set()
        critic_focus: set[int] = set(suspicious)

        for critic_no in range(1, max_critic + 1):
            round_label = f"R{critic_no + 1}"
            critic = critic_pass(
                source, translated, profile, config, f"Critic-{round_label}",
                focus=sorted(critic_focus),
                script_notes=script_notes,
            )
            critic.excluded = sorted(ln for ln in critic.comments if ln in unfixable)
            timeline.append(critic)

            if critic.failed:
                release_reason = (
                    f"critic 第 {critic_no} 轮调用失败（已降级为无 critic 轮次）"
                )
                break

            actionable = {
                ln: notes2 for ln, notes2 in critic.comments.items()
                if ln not in unfixable
            }
            if not actionable:
                if critic.unparsed:
                    release_reason = (
                        f"critic 第 {critic_no} 轮回答无法解析（未获得可用意见，"
                        "并非已收敛）"
                    )
                else:
                    release_reason = (
                        f"critic 第 {critic_no} 轮未提出新问题"
                        + (f"（{len(critic.excluded)} 条意见针对已判定不可修的行）"
                           if critic.excluded else "")
                    )
                break

            before = [b.text for b in translated]
            logger.info(
                "%s: focused re-review of %d commented line(s)",
                round_label, len(actionable),
            )
            # 台本证据常驻重校轮次：critic 意见 + 台本参照（防止 ping-pong）
            merged_notes = {
                ln: list(actionable.get(ln, [])) + script_notes.get(ln, [])
                for ln in actionable
            }
            rnd = _run_round(
                round_label, [], source, translated, chunk_instruction,
                config, chunk_size, anchors, merged_notes,
                focus_lines=sorted(actionable),
            )
            rnd.changed_lines = [
                i + 1 for i in range(len(translated)) if translated[i].text != before[i]
            ]
            timeline.append(rnd)
            if rnd.failed_chunks:
                logger.warning(
                    "%s: %d/%d targeted chunk(s) failed and were left unreviewed",
                    round_label, rnd.failed_chunks, (len(actionable) + chunk_size - 1) // chunk_size,
                )

            changed = set(rnd.changed_lines)
            for ln in sorted(actionable):
                if ln in prev_commented and ln not in changed:
                    unfixable.add(ln)
                    rnd.downgraded.append(ln)
                    rnd.flags.append((
                        ln, "存疑",
                        "反复被审阅指出且重校未改变，需人工确认"
                        f"（审阅意见：{'；'.join(actionable[ln])}）",
                    ))
            if rnd.downgraded:
                logger.info(
                    "%s: %d line(s) downgraded to 存疑 (repeatedly flagged, never "
                    "changed): %s", round_label, len(rnd.downgraded),
                    rnd.downgraded[:10],
                )
            prev_commented = set(actionable)

            release_round = round_label
            if not rnd.changed_lines:
                release_reason = f"{round_label} 定向重校未改变任何文本"
                break
            if critic_no == max_critic:
                release_reason = (
                    f"{round_label} 仍有改动，但已达 critic 轮次上限（{max_critic}）"
                )
            # 下一轮 critic 焦点扩展：可疑行 ∪ 已改动行
            critic_focus |= set(rnd.changed_lines) | set(actionable)

    else:
        # ── 旧路径（无仲裁证据，全量 thinking-ON，向后兼容）────────
        r1 = _run_round(
            "R1", list(range(total_chunks)), source, translated, chunk_instruction,
            config, chunk_size, anchors, None,
        )
        timeline.append(r1)

        # A run in which no chunk produced a usable answer carries no information.
        # Committing its output (and exiting 0) would report a clean bill of health
        # for a review that never happened, so refuse before touching the file.
        if total_chunks and r1.failed_chunks == total_chunks:
            raise AllChunksFailedError(
                f"Review failed: all {total_chunks} chunk(s) failed — "
                f"{translated_path} left untouched"
            )
        if r1.failed_chunks:
            logger.warning(
                "R1: %d/%d chunk(s) failed and were left unreviewed — those "
                "lines are NOT verified",
                r1.failed_chunks, total_chunks,
            )

        # ── Steps 4-5: critic rounds + targeted re-review ──
        release_round = "R1"
        release_reason = f"未启用 critic 轮次（上限 {_MAX_CRITIC_ROUNDS}）"
        unfixable = set()
        prev_commented = set()

        for critic_no in range(1, _MAX_CRITIC_ROUNDS + 1):
            round_label = f"R{critic_no + 1}"
            critic = critic_pass(
                source, translated, profile, config, f"Critic-{round_label}",
            )
            critic.excluded = sorted(ln for ln in critic.comments if ln in unfixable)
            timeline.append(critic)

            if critic.failed:
                release_reason = (
                    f"critic 第 {critic_no} 轮调用失败（已降级为无 critic 轮次）"
                )
                break

            actionable = {
                ln: notes2 for ln, notes2 in critic.comments.items()
                if ln not in unfixable
            }
            if not actionable:
                if critic.unparsed:
                    release_reason = (
                        f"critic 第 {critic_no} 轮回答无法解析（未获得可用意见，"
                        "并非已收敛）"
                    )
                else:
                    release_reason = (
                        f"critic 第 {critic_no} 轮未提出新问题"
                        + (f"（{len(critic.excluded)} 条意见针对已判定不可修的行）"
                           if critic.excluded else "")
                    )
                break

            before = [b.text for b in translated]
            affected = sorted({(ln - 1) // chunk_size for ln in actionable})
            logger.info(
                "%s: targeted re-review of %d/%d chunk(s) for %d commented line(s)",
                round_label, len(affected), total_chunks, len(actionable),
            )
            rnd = _run_round(
                round_label, affected, source, translated, chunk_instruction,
                config, chunk_size, anchors, actionable,
            )
            rnd.changed_lines = [
                i + 1 for i in range(len(translated)) if translated[i].text != before[i]
            ]
            timeline.append(rnd)
            if rnd.failed_chunks:
                logger.warning(
                    "%s: %d/%d targeted chunk(s) failed and were left unreviewed",
                    round_label, rnd.failed_chunks, len(affected),
                )

            # A line the critic keeps flagging that the re-review keeps leaving
            # alone is not converging: hand it to a human and take it out of the
            # loop so later rounds cannot ping-pong on it.
            changed = set(rnd.changed_lines)
            for ln in sorted(actionable):
                if ln in prev_commented and ln not in changed:
                    unfixable.add(ln)
                    rnd.downgraded.append(ln)
                    rnd.flags.append((
                        ln, "存疑",
                        "反复被审阅指出且重校未改变，需人工确认"
                        f"（审阅意见：{'；'.join(actionable[ln])}）",
                    ))
            if rnd.downgraded:
                logger.info(
                    "%s: %d line(s) downgraded to 存疑 (repeatedly flagged, never "
                    "changed): %s", round_label, len(rnd.downgraded),
                    rnd.downgraded[:10],
                )
            prev_commented = set(actionable)

            release_round = round_label
            if affected and rnd.failed_chunks == len(affected):
                # Every targeted chunk failed: "no text changed" here means the
                # re-review never ran, not that it converged. Say so.
                release_reason = (
                    f"{round_label} 定向重校的 {len(affected)} 个 chunk 全部调用失败"
                    "（文本未改变，但并非已收敛）"
                )
                break
            if not rnd.changed_lines:
                release_reason = f"{round_label} 定向重校未改变任何文本"
                break
            if critic_no == _MAX_CRITIC_ROUNDS:
                release_reason = (
                    f"{round_label} 仍有改动，但已达 critic 轮次上限"
                    f"（{_MAX_CRITIC_ROUNDS}）"
                )

    logger.info("Review released at %s: %s", release_round, release_reason)

    # main.py expects translated_path to hold the final reviewed text, and it
    # is the only copy of the translation — snapshot it before replacing it so
    # a bad review run stays recoverable (the write itself is atomic). A failed
    # snapshot aborts the pass rather than overwriting the only copy.
    try:
        make_snapshot(translated_path, "pre-review")
    except OSError as e:
        raise RuntimeError(
            f"Failed to write pre-review snapshot for {translated_path}: {e} — "
            "refusing to overwrite the translation"
        ) from e
    write_translated_srt(translated, translated_path)

    rounds = [r for r in timeline if isinstance(r, ReviewRound)]
    applied = sum(r.applied for r in rounds)
    rejected = sum(r.rejected for r in rounds)
    flags = [f for r in rounds for f in r.flags]
    report_path = _write_report(
        translated_path, source, timeline, profile,
        release_round, release_reason, len(source), total_chunks,
        script_anchors=script_anchors,
        script_conflicts=script_conflicts,
    )
    logger.info(
        "Review complete: %d line(s) changed, %d 修改 rejected (flagged 存疑), "
        "%d 存疑 flag(s) total, %d/%d R1 chunk(s) failed; report: %s",
        applied, rejected, sum(1 for _, k, _ in flags if k == "存疑"),
        r1.failed_chunks, total_chunks, report_path,
    )
    return report_path


def _write_report(
    translated_path: Path,
    source: list[SrtBlock],
    timeline: list,
    profile: GlobalProfile,
    release_round: str,
    release_reason: str,
    total_lines: int,
    total_chunks: int,
    script_anchors: list | None = None,
    script_conflicts: list[tuple[int, str, str]] | None = None,
) -> Path:
    """Write REVIEW-<stem>.md: global analysis, per-round diffs, verdict."""
    rounds = [r for r in timeline if isinstance(r, ReviewRound)]
    critics = [c for c in timeline if isinstance(c, CriticRecord)]
    flags = [f for r in rounds for f in r.flags]
    applied = sum(r.applied for r in rounds)
    rejected = sum(r.rejected for r in rounds)
    failed_chunks = sum(r.failed_chunks for r in rounds)
    attempted_chunks = sum(r.total_chunks for r in rounds)
    suspect = sum(1 for _, k, _ in flags if k == "存疑")
    script_conflicts = script_conflicts or []

    lines = [
        f"# 语义复核报告 — {translated_path.name}",
        "",
        f"- 总行数：{total_lines}；全量 chunk {total_chunks}（每轮 chunk 数见下）",
        f"- 修改 {applied} 行（各轮合计）；失败 chunk {failed_chunks}/{attempted_chunks}",
        f"- 修改被拒绝 {rejected} 处（已降级为存疑，见下）",
        f"- 存疑（需人工/音频复核）{suspect} 行",
        f"- 精修轮次 {len(rounds)} 轮；critic 轮次 {len(critics)} 轮"
        f"（上限 {_MAX_CRITIC_ROUNDS}）",
        "",
        "## 全局语境（步骤 2）",
        "",
    ]
    if profile.is_empty:
        lines.append("- （全局语境不可用，本次复核未使用全局背景）")
    else:
        lines += ["### 语境总结", "", profile.summary or "（无）", ""]
    lines.append("")

    # ── 台本对齐小节（ground truth 证据）────────────────────────
    if script_anchors:
        from collections import Counter
        stats = Counter(a.status for a in script_anchors)
        lines += [
            "## 台本对齐（ground truth）",
            "",
            f"- 对齐状态：high {stats['high']} / mismatch {stats['mismatch']} / "
            f"none {stats['none']}（共 {len(script_anchors)} 行）",
            "- mismatch 行已注入 DeepFix 证据（台本为权威核对源文本/译文）",
        ]
        if script_conflicts:
            lines.append(f"- **硬冲突 {len(script_conflicts)} 行**"
                         "（台本 vs 音频三模型一致，未自动改，需人工）：")
            for ln, stext, why in script_conflicts:
                lines.append(f"  - #{ln}: 台本「{stext[:40]}」— {why}")
        lines.append("")

    lines += ["## 轮次记录", ""]
    for item in timeline:
        if isinstance(item, ReviewRound):
            kind = "初校（全量精修）" if item.label == "R1" else "定向重校"
            lines += [
                f"### {item.label} {kind}",
                "",
                f"- chunk {item.total_chunks} 个"
                + (f"（索引 {item.chunk_indices}）" if item.label != "R1" else "")
                + f"；失败 {item.failed_chunks}",
                f"- 修改 {item.applied} 行；修改被拒绝 {item.rejected} 处；"
                f"标记 {len(item.flags)} 条",
            ]
            if item.failed_blocks:
                spans = []
                for blk in item.failed_blocks:
                    if len(blk) > 1:
                        spans.append(f"#{blk[0]}-#{blk[-1]}")
                    else:
                        spans.append(f"#{blk[0]}")
                lines.append(
                    "- 失败 chunk 覆盖的行（未复核，需补跑或人工）："
                    + ", ".join(spans)
                )
            if item.label != "R1":
                lines.append(
                    "- 实际改变文本的行："
                    + (", ".join(f"#{ln}" for ln in item.changed_lines) or "（无）")
                )
            if item.downgraded:
                lines.append(
                    "- 降级为存疑（反复被指出且未改变）："
                    + ", ".join(f"#{ln}" for ln in item.downgraded)
                )
            lines.append("")
            if item.flags:
                for ln, k, value in sorted(item.flags):
                    lines.append(f"- #{ln} [{k}] {value}")
            else:
                lines.append("- （本轮无修改/标记）")
            lines.append("")
        else:
            lines += [f"### {item.label} 意见（步骤 4）", ""]
            if item.failed:
                lines.append("- （critic 调用失败，本轮无意见）")
            elif item.unparsed:
                lines.append(
                    "- （critic 回答无法解析：本轮没有可用意见，不代表没有问题）"
                )
            elif not item.comments:
                lines.append("- （无意见）")
            else:
                for ln, notes in sorted(item.comments.items()):
                    mark = "（已判定不可修，跳过）" if ln in item.excluded else ""
                    lines.append(f"- #{ln}{mark}: {'；'.join(notes)}")
            if item.out_of_range:
                lines.append(
                    f"- 越界行号已跳过：{item.out_of_range[:10]}"
                )
            lines.append("")

    lines += ["## 修改记录", ""]
    if flags:
        for ln, kind, value in sorted(flags):
            src = source[ln - 1].text if ln - 1 < len(source) else "?"
            lines.append(f"- #{ln} [{kind}] {value}\n  源文: {src}")
    else:
        lines.append("- （无）")
    lines += [
        "",
        f"## 放行结论：{release_round}（{release_reason}）",
        "",
    ]

    report_path = translated_path.with_name(
        f"REVIEW-{translated_path.stem}.md"
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
