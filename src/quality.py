"""Evidence-first source resolution, scene translation and selective subtitle QA.

Every requested ID needs an explicit result. Transport retries belong to call_llm;
format repair retries only missing IDs once, without retranslating valid results.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.entities import EntityPlan

from src.config import TranslateConfig
from src.evidence import CueEvidence
from src.episode_context import VERSION as EPISODE_CONTEXT_VERSION, episode_source_prefix
from src.review import _reject_fix
from src.translate import SrtBlock, build_instruction, call_llm, make_snapshot, write_translated_srt
from src.workflow_state import file_hash, fingerprint, read_json, write_json

logger = logging.getLogger(__name__)
VERSION = "evidence-workflow-10"
STRUCTURED_TRANSLATION_VERSION = "structured-translation-1"

RESOLVE = """核对源语言录音转写。仅从该行列出的候选中选择，不生成新台词。
三模型多数不保证正确；台本可能是不同录音版本；标题只提供词义线索。
结合邻域和证据决定，不能确定就 UNSURE。每个待处理编号必须输出一条：
[N] KEEP|简短依据
[N] SELECT|候选ID|简短依据
[N] UNSURE|简短原因
KEEP 表示确认当前源文（候选 W）；SELECT 表示选用一个完整候选。禁止省略任何待处理行。
"""
TRANSLATE = """Translate the chosen {source_lang} source to {target_lang}.
Output [N] translation for EVERY target ID. IDs are global and may be nonconsecutive.
Do not output context IDs, explanations or extra lines. Never merge, split or shift cues.
Translate the source printed on the target row; it may still be unresolved. Raw ASR
candidates are evidence, not replacement dialogue. Do not silently switch candidates.
Use neighboring dialogue for subjects, tone and terms, but do not add unheard content.
When a character is identified, follow explicit WORK CONTEXT name renderings and roles,
including supported ASR name variants; do not treat ordinary words as names without context.
Preserve who acts on whom, negation, requests, and quoted or reported statements.
For uncertain source text, translate conservatively without inventing a resolution.
"""
DIAGNOSE = """检查 {source_lang}→{target_lang} 的每个待审字幕。只诊断，不改写。
逐句核对“谁对谁做什么”：动作主体、对象、受益者、请求对象以及否定是否一致。
特别区分自主动作、使役、被动与请求；例如「殺す」是杀死某人，「死ぬ」是死亡，
「殺される」是被杀，不能把杀人的威胁改成让对方自行去死。省略的人物只能由邻域支持。
保留引语和转述层次；宣布、假设或引用一件事，不等于该事已经发生。
核对数字、漏译、扩写、相邻行串位、术语、语气和未翻译残留。
已识别人物按 WORK CONTEXT 的明确译名、性别和关系核对，包括有依据的 ASR 名字变体；
不能因转写用了同音常用字而放弃已明确的人名，也不能把普通词一律替换成人名。
以待审行的已选源文为翻译依据；原始 ASR 候选只供核对，不能直接替换它。
源文存疑与译文错误分开判断：即使录音转写待确认，选定源文能直接证明的误译仍报 ISSUE。
仅因台本或另一 ASR 候选不同、或需要猜测源文才成立的问题，报 UNSURE，不能报确定误译。
每个待审 ID 必须输出且只输出一条（全局编号，可不连续）：
[N] OK
[N] ISSUE|问题类型|一句话依据
[N] UNSURE|缺少什么证据
语气差异本身不是错误。没有问题也必须逐行输出 OK，不可省略或只写“无”。
[context:N] 是只读邻域，绝不输出该编号。只输出最后列出的待处理ID，不得添加其他ID。
"""
PATCH = """修正已指出的 {source_lang}→{target_lang} 翻译错误，只输出需要的局部结果。
只修选定源文及明确语境可以直接证明的译文错误，禁止编造或变更源文。
源文可能仍存疑：局部译文修正不代表源文疑问已解决，不能用另一个 ASR 候选替换源文。
保持动作主体、对象、否定和引语关系，已识别人物遵循 WORK CONTEXT 明确译名。
若所提修正必须先猜定源文或人物身份，输出 UNSURE。每个待处理 ID 必须输出一条：
[N] FIX|修正译文
[N] UNSURE|证据不足的原因
不得输出其他行，不得把邻行内容移入该行，不得输出评语作为字幕。
"""


def mode(config: TranslateConfig, stage: str, thinking: bool = False) -> TranslateConfig:
    extra = dict(config.extra_payload or {})
    extra.pop("reasoning_effort", None)
    kwargs = dict(extra.get("chat_template_kwargs") or {})
    kwargs["enable_thinking"] = thinking
    extra["chat_template_kwargs"] = kwargs
    if not thinking:
        extra["reasoning_effort"] = "none"
    return replace(config, extra_payload=extra, stage=stage)


def draft_translation_config(config: TranslateConfig) -> TranslateConfig:
    """Apply optional enforced thinking only to first-draft translation calls.

    The legacy default and other stages retain their existing request/retry
    behavior. Capped requests never silently retry with thinking off or a larger
    total allowance; query's bounded missing-ID retry keeps this same config.
    """
    budget = config.translation_reasoning_budget
    cfg = mode(config, "translation", thinking=bool(budget))
    if not budget:
        return cfg
    extra = dict(cfg.extra_payload or {})
    for key in ("thinking_budget_tokens", "reasoning_budget", "reasoning_budget_message"):
        extra.pop(key, None)
    extra.update(reasoning_budget_tokens=budget, max_tokens=cfg.max_tokens)
    return replace(cfg, extra_payload=extra, retries=0)


def draft_translation_settings(config: TranslateConfig) -> dict:
    """Stage provenance/cache policy; unrelated source/entity keys exclude it."""
    cfg = draft_translation_config(config)
    return {"policy": "draft-thinking-budget-1", "thinking_budget": config.translation_reasoning_budget,
            "request_payload": cfg.extra_payload, "max_tokens": cfg.max_tokens,
            "transport_retries": cfg.retries,
            "structured_translation": {"enabled": config.structured_translation,
                "version": STRUCTURED_TRANSLATION_VERSION if config.structured_translation else None},
            "episode_context": {"enabled": config.translation_episode_context,
                                "policy": EPISODE_CONTEXT_VERSION if config.translation_episode_context else None,
                                "code_hash": file_hash(Path(__file__).with_name("episode_context.py")) if config.translation_episode_context else None}}


def validate_blocks(blocks: list[SrtBlock]) -> None:
    if not blocks:
        raise ValueError("No subtitle cues")
    for i, block in enumerate(blocks, 1):
        if block.index != i or not block.text.strip():
            raise ValueError(f"Invalid or discontinuous subtitle cue {i}")
        if not re.fullmatch(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", block.ts_line):
            raise ValueError(f"Invalid timestamp for cue {i}")


def seconds(ts: str) -> float:
    h, m, tail = ts.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(tail.replace(",", "."))


def scene_groups(source: list[SrtBlock], config: TranslateConfig,
                 ids: list[int] | None = None) -> list[list[int]]:
    groups, group, size = [], [], 0
    for ln in ids if ids is not None else range(1, len(source) + 1):
        block = source[ln - 1]
        gap = 0.0
        if group:
            previous = source[group[-1] - 1]
            gap = seconds(block.ts_line.split("-->")[0]) - seconds(previous.ts_line.split("-->")[1])
        full = len(group) >= config.chunk_size or size + len(block.text) > config.scene_max_chars
        if group and (full or (ids is None and gap >= 3.0 and not config.pack_scenes)):
            groups.append(group)
            group, size = [], 0
        group.append(ln)
        size += len(block.text)
    if group:
        groups.append(group)
    return groups


def format_scene(ids: list[int], source: list[SrtBlock], evidence: list[CueEvidence],
                 config: TranslateConfig, translated: list[SrtBlock] | None = None,
                 issues: dict[int, str] | None = None, candidates: bool = False,
                 local_ids: bool = False) -> str:
    targets = set(ids)
    displayed_ids = {ln: i if local_ids else ln for i, ln in enumerate(ids, 1)}
    visible = set(targets)
    for ln in targets:
        visible.update(range(max(1, ln - config.scene_context_lines),
                             min(len(source), ln + config.scene_context_lines) + 1))
    parts = []
    for ln in sorted(visible):
        if ln in targets:
            tag = f"[{displayed_ids[ln]}]"
        else:
            tag = "[neighbor]" if local_ids else f"[context:{ln}]"
        row = f"{tag} {'当前源文' if candidates else '已选源文'}: {source[ln - 1].text}"
        if translated is not None:
            row += f" → 译文: {translated[ln - 1].text}"
        parts.append(row)
        if ln in targets:
            if candidates:
                parts.extend("    " + note for note in evidence[ln - 1].notes())
                parts.extend("    " + note for note in config.line_notes.get(ln, []))
                parts.extend(f"    候选 {name}: {text}" for name, text in evidence[ln - 1].candidates().items())
            else:
                # Candidate adjudication belongs to the source stage. Repeating
                # rival transcripts here makes the small writer switch source.
                # Full evidence and unresolved-source gates stay in the ledger.
                e = evidence[ln - 1]
                if e.blocked:
                    parts.append("    源文待核实；只检查本行选定源文是否被准确翻译。")
                if e.hard_conflict:
                    parts.append("    台本/音频硬冲突：不得自动修改。")
                parts.extend("    " + note for note in config.entity_notes.get(ln, []))
            if issues and ln in issues:
                parts.append("    问题: " + issues[ln])
    whitelist = ", ".join(str(displayed_ids[ln]) for ln in ids)
    neighbor_tag = "[neighbor]" if local_ids else "[context:N]"
    return (f"本次仅处理这些ID: {whitelist}。标记 {neighbor_tag} 的行只读，不得输出。\n"
            + "\n".join(parts)
            + f"\n本次输出ID必须恰好为: {whitelist}。每个ID一条；禁止输出邻域ID。")


def parse_records(text: str, ids: list[int], kind: str,
                  evidence: list[CueEvidence],
                  translation_transform: Callable[[int, str], str] | None = None) -> dict[int, list[str]]:
    result, duplicates, seen = {}, set(), set()
    for raw in text.splitlines():
        m = re.fullmatch(r"\[(\d+)\]\s*(.+)", raw.strip())
        if not m:
            continue
        ln = int(m[1])
        if ln not in ids:
            # Output in the wrong ID space must never be applied.
            return {}
        if ln in seen:
            duplicates.add(ln)
        seen.add(ln)
        fields = m[2].split("|", 2)
        valid = False
        if kind == "translation":
            fields = [m[2].strip()]
            if translation_transform is not None:
                try:
                    fields[0] = translation_transform(ln, fields[0])
                except ValueError:
                    continue
            valid = (bool(fields[0]) and not any(c in fields[0] for c in "⟦⟧")
                     and not re.search(r"\[\d+\]", fields[0])
                     and not _reject_fix(fields[0], "", evidence[ln - 1].raw))
        elif kind == "resolve":
            valid = ((fields[0] in {"KEEP", "UNSURE"} and len(fields) == 2 and bool(fields[1].strip()))
                     or (fields[0] == "SELECT" and len(fields) == 3 and bool(fields[2].strip())
                         and fields[1] in evidence[ln - 1].candidates()))
        elif kind == "qa":
            valid = (fields == ["OK"] or (fields[0] == "ISSUE" and len(fields) == 3 and all(fields))
                     or (fields[0] == "UNSURE" and len(fields) == 2 and bool(fields[1].strip())))
        elif kind == "patch":
            valid = (fields[0] in {"FIX", "UNSURE"} and len(fields) == 2 and bool(fields[1].strip())
                     and (fields[0] != "FIX" or not any(c in fields[1] for c in "⟦⟧")))
        if valid:
            result[ln] = fields
    return {ln: fields for ln, fields in result.items() if ln not in duplicates}


def qa_schema(ids: list[int]) -> dict:
    """Request-local QA schema supported by the local llama.cpp server."""
    record = {"type": "object", "properties": {
        "status": {"type": "string", "enum": ["OK", "ISSUE", "UNSURE"]},
        "reason": {"type": "string", "maxLength": 120}},
        "required": ["status", "reason"], "additionalProperties": False}
    return {"type": "object", "properties": {str(ln): record for ln in ids},
            "required": [str(ln) for ln in ids], "additionalProperties": False}


def translation_schema(ids: list[int]) -> dict:
    """Bind every request-local ID to one complete, nonempty translation."""
    return {"type": "object", "properties": {
        str(ln): {"type": "string", "minLength": 1} for ln in ids},
        "required": [str(ln) for ln in ids], "additionalProperties": False}


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def parse_structured_qa(text: str, ids: list[int]) -> dict[int, list[str]]:
    """Accept the complete schema only; no markdown, extras or positional fallback."""
    try:
        rows = json.loads(text, object_pairs_hook=_unique_json_object,
                          parse_constant=_invalid_json_constant)
    except (ValueError, TypeError, RecursionError):
        return {}
    if not isinstance(rows, dict) or set(rows) != {str(ln) for ln in ids}:
        return {}
    result = {}
    for ln in ids:
        row = rows[str(ln)]
        if not isinstance(row, dict) or set(row) != {"status", "reason"}:
            return {}
        status, reason = row["status"], row["reason"]
        if (not isinstance(status, str) or status not in {"OK", "ISSUE", "UNSURE"}
                or not isinstance(reason, str) or len(reason) > 120
                or "\n" in reason or "\r" in reason
                or (status != "OK" and not reason.strip())):
            return {}
        result[ln] = (["OK"] if status == "OK" else
                      ["ISSUE", "translation", reason] if status == "ISSUE" else
                      ["UNSURE", reason])
    return result


def parse_structured_translation(
    text: str, ids: list[int], evidence: list[CueEvidence],
    translation_transform: Callable[[int, str], str] | None = None,
) -> dict[int, list[str]]:
    """Validate the entire JSON shape before applying existing per-row guards."""
    try:
        rows = json.loads(text, object_pairs_hook=_unique_json_object,
                          parse_constant=_invalid_json_constant)
    except (ValueError, TypeError, RecursionError):
        return {}
    if not isinstance(rows, dict) or set(rows) != {str(ln) for ln in ids}:
        return {}
    for value in rows.values():
        if (not isinstance(value, str) or not value.strip()
                or any(ord(char) < 32 or char in "\x85\u2028\u2029" for char in value)):
            return {}
    # Newlines/control characters are rejected above, so JSON values cannot
    # inject additional records into the existing translation/entity guards.
    numbered = "\n".join(f"[{ln}] {rows[str(ln)]}" for ln in ids)
    return parse_records(numbered, ids, "translation", evidence, translation_transform)


def cached_structured_translation(
    cached: dict, ids: list[int], evidence: list[CueEvidence],
    translation_transform: Callable[[int, str], str] | None = None,
) -> tuple[dict[int, list[str]], list[dict]]:
    """Replay exact responses and their pending-ID scopes, never trusted results."""
    if (set(cached) != {"format", "responses"}
            or cached.get("format") != STRUCTURED_TRANSLATION_VERSION
            or not isinstance(cached.get("responses"), list)):
        return {}, []
    result = {}
    responses = []
    for record in cached["responses"]:
        pending = [ln for ln in ids if ln not in result]
        if (not isinstance(record, dict) or set(record) != {"global_ids", "raw_answer"}
                or not isinstance(record["global_ids"], list)
                or any(type(ln) is not int for ln in record["global_ids"])
                or not pending or record["global_ids"] != pending
                or not isinstance(record["raw_answer"], str)):
            return {}, []
        local_ids = list(range(1, len(pending) + 1))
        transform = ((lambda ln, value: translation_transform(pending[ln - 1], value))
                     if translation_transform is not None else None)
        parsed = parse_structured_translation(record["raw_answer"], local_ids,
            [evidence[ln - 1] for ln in pending], transform)
        result.update({pending[ln - 1]: fields for ln, fields in parsed.items()})
        responses.append(record)
    return result, responses


def cached_structured_qa(cached: dict, ids: list[int]) -> dict[int, list[str]]:
    """Validate stored global IDs and bounded fields without parsing them as lines."""
    rows = {}
    for key, fields in cached.items():
        if (key not in {str(ln) for ln in ids} or not isinstance(fields, list)
                or not all(isinstance(field, str) for field in fields)):
            return {}
        if fields == ["OK"]:
            rows[key] = {"status": "OK", "reason": ""}
        elif len(fields) == 3 and fields[:2] == ["ISSUE", "translation"]:
            rows[key] = {"status": "ISSUE", "reason": fields[2]}
        elif len(fields) == 2 and fields[0] == "UNSURE":
            rows[key] = {"status": "UNSURE", "reason": fields[1]}
        else:
            return {}
    return parse_structured_qa(json.dumps(rows), [int(key) for key in rows])


def audit_rejection(cache: Path, stage: str, ids: list[int], response: str,
                    valid_ids: list[int], structured: bool, *,
                    response_format: str | None = None) -> None:
    path = cache.with_suffix(".rejections.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "stage": stage,
              "global_ids": ids, "valid_global_ids": valid_ids,
              "format": response_format or ("structured-qa" if structured else "numbered-lines"),
              "raw_answer": response}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def query(ids: list[int], source: list[SrtBlock], evidence: list[CueEvidence],
          config: TranslateConfig, template: str, kind: str, cache: Path,
          translated: list[SrtBlock] | None = None,
          issues: dict[int, str] | None = None,
          entity_plan: EntityPlan | None = None) -> dict[int, list[str]]:
    if entity_plan is not None:
        from src.entities import restore_entity_target
        if kind != "translation" or any(
                entity_plan.source_hashes.get(b.index) != fingerprint(b.text) for b in source):
            raise ValueError("Entity plan requires identical chosen translation source")
    writer_source = entity_plan.marked_source if entity_plan is not None else source
    # The 27B model only sees contiguous request-local IDs. Stable global IDs
    # remain internal to caches, source decisions, and the QA ledger.
    local_template = (template.replace("IDs are global and may be nonconsecutive.",
                                       "IDs are local to this request and consecutive from 1.")
                     .replace("全局编号，可不连续", "本请求局部编号，从1连续递增")
                     .replace("[context:N]", "[neighbor]"))
    structured = kind == "qa" and config.structured_qa
    structured_translation = kind == "translation" and config.structured_translation
    if structured_translation:
        local_template = local_template.replace("Output [N] translation for EVERY target ID.",
                                               "Output a JSON object for EVERY target ID.")
    instruction = build_instruction(config, local_template)
    if kind == "translation" and config.translation_episode_context:
        # Use original chosen rows here. Per-target marker transformations stay
        # local to writer_source and its existing restoration contract.
        instruction = episode_source_prefix(source) + instruction
    instruction += "\n待处理编号仅是本次请求的局部编号，从1开始；[neighbor] 行只读，没有可输出编号。\n"
    if structured:
        instruction += ("\n输出格式覆盖：只输出JSON对象，键为本次局部ID，每个值只含status"
                        "（OK/ISSUE/UNSURE）及reason（120字以内单行依据，OK为空）。"
                        "所有ID必须齐全；禁止额外字段、解释过程、Markdown和方括号行格式。\n")
    elif structured_translation:
        instruction += ("\n只输出JSON对象：键为本次从1开始的局部ID，值为该行完整的单行译文字符串。"
                        "所有ID必须齐全，不得重复、添加或省略键，不得输出空白译文。"
                        "禁止说明、Markdown、方括号行格式及字符串内换行。\n")
    entity_instruction = entity_plan.translation_instruction(ids) if entity_plan is not None else ""
    body = format_scene(ids, writer_source, evidence, config, translated, issues, kind == "resolve", local_ids=True)
    key_parts = [VERSION, ids, config.endpoint, instruction, body, config.extra_payload,
                 config.max_tokens, kind, structured, entity_instruction,
                 entity_plan.key if entity_plan is not None else None]
    if structured_translation:
        key_parts.extend([STRUCTURED_TRANSLATION_VERSION, translation_schema(list(range(1, len(ids) + 1)))])
    key = fingerprint(key_parts)
    saved = read_json(cache, {})
    saved = saved if isinstance(saved, dict) else {}
    cached = saved.get(key, {})
    cached = cached if isinstance(cached, dict) else {}
    # Re-validate cached model output too; a damaged cache is never an approval.
    cached_text = "\n".join(f"[{ln}] {'|'.join(fields)}" for ln, fields in cached.items()
                            if isinstance(fields, list) and all(isinstance(f, str) for f in fields))
    raw_responses = []
    if structured_translation:
        cached_transform = ((lambda ln, text: restore_entity_target(text, entity_plan.rows[ln]))
                            if entity_plan is not None else None)
        result, raw_responses = cached_structured_translation(cached, ids, evidence, cached_transform)
    else:
        result = (cached_structured_qa(cached, ids) if structured else
                  parse_records(cached_text, ids, kind, evidence))
    for _ in range(2):
        pending = [ln for ln in ids if ln not in result]
        if not pending:
            break
        body = format_scene(pending, writer_source, evidence, config, translated, issues, kind == "resolve", local_ids=True)
        request_instruction = instruction + (entity_plan.translation_instruction(pending) if entity_plan is not None else "")
        request_config = config
        if structured:
            request_config = mode(config, config.stage)
            budget = min(config.max_tokens, 2048)
            extra = dict(request_config.extra_payload or {})
            extra["response_format"] = {"type": "json_object", "schema": qa_schema(list(range(1, len(pending) + 1)))}
            # Keep the cap even if the generic HTTP helper grows its retry budget.
            extra["max_tokens"] = budget
            request_config = replace(request_config, extra_payload=extra, max_tokens=budget,
                                     response_guard_floor=max(config.response_guard_floor, 512))
        elif structured_translation:
            extra = dict(config.extra_payload or {})
            extra["response_format"] = {"type": "json_object", "schema":
                                        translation_schema(list(range(1, len(pending) + 1)))}
            request_config = replace(config, extra_payload=extra,
                                     response_guard_floor=max(config.response_guard_floor, 512))
        try:
            capped_draft = (kind == "translation" and request_config.stage == "translation"
                            and request_config.translation_reasoning_budget > 0)
            response = call_llm(body, request_instruction, request_config,
                with_thinking=capped_draft or (kind == "resolve" and bool(
                    (config.extra_payload or {}).get("chat_template_kwargs", {}).get("enable_thinking", True))))
        except RuntimeError as exc:
            # The HTTP retry budget is exhausted. Stop this track's stage instead
            # of issuing the same failing request for every remaining scene.
            raise RuntimeError(f"{config.stage} failed for cue IDs {pending}: {exc}") from exc
        local_evidence = [evidence[ln - 1] for ln in pending]
        local_ids = list(range(1, len(pending) + 1))
        transform = (lambda local_id, text: restore_entity_target(
            text, entity_plan.rows[pending[local_id - 1]])) if entity_plan is not None else None
        local_result = (parse_structured_qa(response, local_ids) if structured else
                        parse_structured_translation(response, local_ids, local_evidence, transform)
                        if structured_translation else
                        parse_records(response, local_ids, kind, local_evidence, transform))
        parsed = {pending[local_id - 1]: fields for local_id, fields in local_result.items()}
        if len(parsed) < len(pending):
            logger.warning("%s: missing/invalid output for cue IDs %s", config.stage,
                           sorted(set(pending) - set(parsed)))
            audit_rejection(cache, config.stage, pending, response, sorted(parsed), structured,
                            response_format="structured-translation" if structured_translation else None)
        result.update(parsed)
        if structured_translation:
            raw_responses.append({"global_ids": pending, "raw_answer": response})
            saved[key] = {"format": STRUCTURED_TRANSLATION_VERSION, "responses": raw_responses}
        else:
            saved[key] = result
        write_json(cache, saved)
    return result


def resolve_source(source: list[SrtBlock], evidence: list[CueEvidence],
                   config: TranslateConfig, cache: Path) -> tuple[list[SrtBlock], list[dict]]:
    resolved = [replace(b) for b in source]
    decisions = [{"line": e.line, "status": "accepted", "candidate": "W", "text": e.raw,
                  "reason": "No source-conflict signal", "evidence": asdict(e)} for e in evidence]
    pending = []
    for e in evidence:
        if e.blocked:
            decisions[e.line - 1].update(status="unresolved", reason="台本/音频硬冲突" if e.hard_conflict else f"音频证据不足 ({e.grade})")
        elif e.needs_resolution:
            pending.append(e.line)
    ensemble = any(e.ensemble for e in evidence)
    cfg = mode(config, "source-resolution", not ensemble)
    cfg = replace(cfg, chunk_size=max(1, config.review_chunk_size),
                  max_tokens=min(config.max_tokens, 2048) if ensemble else config.max_tokens)
    for ids in scene_groups(source, cfg, pending):
        rows = query(ids, source, evidence, cfg, RESOLVE, "resolve", cache)
        for ln in ids:
            decision = decisions[ln - 1]
            fields = rows.get(ln)
            if fields is None:
                decision.update(status="unchecked", reason="Source resolution response missing/invalid")
            elif fields[0] == "UNSURE":
                decision.update(status="unresolved", reason=fields[1])
            else:
                selected = "W" if fields[0] == "KEEP" else fields[1]
                text = evidence[ln - 1].candidates()[selected]
                if _reject_fix(text, source[ln - 1].text, source[ln - 1].text):
                    decision.update(status="unresolved", reason="Candidate failed source length/format validation")
                    continue
                resolved[ln - 1].text = text
                decision.update(status="corrected" if text != source[ln - 1].text else "accepted",
                                text=text, candidate=selected, reason=fields[-1])
    return resolved, decisions


def translate_scenes(source: list[SrtBlock], evidence: list[CueEvidence],
                     config: TranslateConfig, cache: Path,
                     entity_plan: EntityPlan | None = None,
                     entity_render: dict | None = None) -> list[SrtBlock]:
    validate_blocks(source)
    translated = [replace(b) for b in source]
    cfg = draft_translation_config(config)
    missing = []
    for ids in scene_groups(source, cfg):
        rows = query(ids, source, evidence, cfg, TRANSLATE, "translation", cache, entity_plan=entity_plan)
        if entity_render is not None:
            for ln in rows:
                entity_render[str(ln)] = "restored" if entity_plan and entity_plan.rows[ln].replacements else "unmarked"
        if entity_plan is not None:
            pending = [ln for ln in ids if ln not in rows]
            if pending:
                # Retry exhausted marker rows through ordinary local translation.
                # The failure remains an unresolved identity check in the ledger.
                fallback_cfg = replace(cfg, entity_notes={})
                fallback = query(pending, source, evidence, fallback_cfg, TRANSLATE, "translation", cache)
                rows.update(fallback)
                if entity_render is not None:
                    for ln in pending:
                        entity_render[str(ln)] = "fallback_identity_unchecked" if ln in fallback else "translation_missing"
        for ln in ids:
            if ln in rows:
                translated[ln - 1].text = rows[ln][0]
            else:
                missing.append(ln)
    if missing:
        raise RuntimeError(f"Translation incomplete; missing cue IDs {missing}. Valid chunks cached for resume.")
    return translated


def selective_qa(source: list[SrtBlock], draft: list[SrtBlock], evidence: list[CueEvidence],
                 decisions: list[dict], config: TranslateConfig, cache: Path
                 ) -> tuple[list[SrtBlock], list[dict]]:
    validate_blocks(source)
    validate_blocks(draft)
    if len(source) != len(draft) or any(a.ts_line != b.ts_line for a, b in zip(source, draft)):
        raise ValueError("Source/translation cue alignment differs")
    final = [replace(b) for b in draft]
    ledger = [{"line": e.line, "timestamp": source[e.line - 1].ts_line,
               "status": "unchecked", "reason": "QA not completed", "evidence": asdict(e),
               "source_decision": decisions[e.line - 1]} for e in evidence]
    issues = {}
    cfg = mode(config, "diagnosis")
    for ids in scene_groups(source, cfg):
        rows = query(ids, source, evidence, cfg, DIAGNOSE, "qa", cache, draft)
        for ln in ids:
            row = ledger[ln - 1]
            fields = rows.get(ln)
            if fields is None:
                continue
            if fields[0] == "OK":
                row.update(status="accepted", reason="Explicit QA OK")
            elif fields[0] == "UNSURE":
                row.update(status="unresolved", reason=fields[1])
            else:
                row.update(status="unresolved", reason="|".join(fields[1:]))
                # Source uncertainty must not suppress a demonstrable target
                # mistranslation. A script/audio hard conflict remains no-edit.
                if not evidence[ln - 1].hard_conflict:
                    issues[ln] = row["reason"]
    changed = []
    cfg = mode(config, "targeted-repair")
    for ids in scene_groups(source, cfg, sorted(issues)):
        rows = query(ids, source, evidence, cfg, PATCH, "patch", cache, draft, issues)
        for ln in ids:
            fields = rows.get(ln)
            if fields is None:
                ledger[ln - 1].update(status="unchecked", reason="Repair response missing/invalid")
                continue
            if fields[0] == "UNSURE":
                ledger[ln - 1].update(reason=fields[1])
                continue
            reason = _reject_fix(fields[1], draft[ln - 1].text, source[ln - 1].text)
            if reason:
                ledger[ln - 1].update(reason=f"Rejected repair: {reason}")
            elif fields[1] != draft[ln - 1].text:
                final[ln - 1].text = fields[1]
                changed.append(ln)
    if config.delta_verification:
        from src.repair_verification import verify_repairs
        final, ledger = verify_repairs(source, draft, final, ledger, changed, config,
                                       cache.with_suffix(".repair-delta.json"))
    else:
        # Verify changes and their immediate neighbors once. New issues become visible
        # human work instead of another speculative writer/critic loop.
        focus = sorted({j for ln in changed for j in range(max(1, ln - 1), min(len(source), ln + 1) + 1)})
        cfg = mode(config, "verification")
        verification = {}
        for ids in scene_groups(source, cfg, focus):
            verification.update(query(ids, source, evidence, cfg, DIAGNOSE, "qa", cache, final))
        for ln in focus:
            fields = verification.get(ln)
            row = ledger[ln - 1]
            if ln in changed:
                neighborhood = range(max(1, ln - 1), min(len(source), ln + 1) + 1)
                if fields == ["OK"] and all(verification.get(j) == ["OK"] for j in neighborhood):
                    row.update(status="corrected", reason="Targeted repair verified",
                               before=draft[ln - 1].text, after=final[ln - 1].text)
                else:
                    final[ln - 1].text = draft[ln - 1].text
                    row.update(status="unresolved" if fields else "unchecked",
                               reason="Repair reverted: " + ("changed cue or neighbor did not pass verification" if fields else "verification missing"))
            elif fields != ["OK"] and row["status"] not in {"unchecked", "unresolved"}:
                row.update(status="unresolved" if fields else "unchecked",
                           reason="Neighbor verification: " + ("|".join(fields) if fields else "missing"))
    # Target repair and source confidence are independent. An explicit target
    # OK (including after a verified repair) cannot clear unresolved source
    # evidence. Keep both outcomes so useful corrections remain auditable.
    for row in ledger:
        row["target_status"] = row["status"]
        row["target_reason"] = row["reason"]
        decision = row["source_decision"]
        if decision["status"] in {"unresolved", "unchecked"}:
            if row["status"] != "unchecked":
                row["status"] = decision["status"]
            row["reason"] = (f"Source {decision['status']}: {decision['reason']}; "
                             f"target {row['target_status']}: {row['target_reason']}")
    return final, ledger


def write_quality_result(path: Path, blocks: list[SrtBlock], ledger: list[dict],
                         report: Path, ledger_path: Path) -> bool:
    if path.exists():
        make_snapshot(path, "pre-review")
    write_translated_srt(blocks, path)
    counts = {status: sum(row["status"] == status for row in ledger)
              for status in ("accepted", "corrected", "unresolved", "unchecked")}
    complete = counts["unchecked"] == 0
    write_json(ledger_path, {"version": VERSION, "complete": complete,
                            "counts": counts, "cues": ledger})
    lines = [f"# 语义复核报告 — {path.name}", "", f"- 状态: {'完成' if complete else '未完成，需续跑'}",
             f"- 已确认 {counts['accepted']}；修正 {counts['corrected']}；存疑 {counts['unresolved']}；未检查 {counts['unchecked']}",
             "- 音频、台本、源文决定及每行状态见同名 quality.json。", "", "## 修改和待人工复核", ""]
    for row in ledger:
        decision = row["source_decision"]
        if row["status"] != "accepted" or decision["status"] == "corrected":
            lines.append(f"- #{row['line']} [{row['status']}] {row['timestamp']} — {row['reason']}")
            if decision["status"] == "corrected":
                lines.append(f"  源文裁决: {row['evidence']['raw']} → {decision['text']}（{decision['reason']}）")
            if "before" in row:
                lines.append(f"  {row['before']} → {row['after']}")
            if row["status"] in {"unresolved", "unchecked"}:
                e = row["evidence"]
                if e.get("ensemble"):
                    lines.append(f"  源文[{e.get('source_model', 'w').upper()}]: {e['raw']}")
                    lines.append(f"  Whisper: {e.get('whisper') or '(无)'} / N: {e['n'] or '(无)'} / Q: {e['q'] or '(无)'}")
                    if e.get("r"):
                        lines.append(f"  NeMo R（与N同属Reazon系，不作额外独立票）: {e['r']}")
                else:
                    lines.append(f"  W: {e['raw']} / N: {e['n'] or '(无)'} / Q: {e['q'] or '(无)'}")
                if e["script"]:
                    lines.append(f"  台本: {e['script']}")
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(report)
    return complete
