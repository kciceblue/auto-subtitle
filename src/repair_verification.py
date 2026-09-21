"""Verify local repairs as transactions against their exact before/after context.

This is local writer QA, never the independent release assessment. A pre-existing
neighbor issue remains in the ledger but does not itself invalidate an unrelated
repair. Mutually visible proposed edits commit or revert together.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from src.config import TranslateConfig
from src.translate import SrtBlock, build_instruction, call_llm
from src.workflow_state import fingerprint, read_json, write_json

VERSION = "repair-delta-1"
CHECKS = ("issue_fixed", "faithful_to_selected_source", "no_new_error", "no_cross_row_regression")
INSTRUCTION = """核验局部字幕修改，不生成或改写字幕。比较同一已选源文对应的修改前/后译文。
逐个待核验ID判断：原问题是否已修复；修改后是否忠实于选定源文；是否未新增错误；
是否未新增与邻行的语义矛盾、漏译、串行、指代或术语问题。逐项回答 yes/no/uncertain。
核对谁对谁做什么、使役/被动/请求、否定范围、引语与实际发生事件。已有邻行问题
若没有因本次修改变得更坏，仍是已有问题，不等于修改引入了新问题。
本批修改将一起提交或一起撤回；按完整“修改后”邻域核验。未修改邻行只读。
已选源文可能存疑，禁止猜定录音或替换源文来批准修正。能由选定源文直接证明的
局部修正可以通过，但这不消除源文疑问，也不代表字幕可以发布。
只输出指定JSON对象，所有待核验ID齐全，每项给出120字内单行依据，不输出字幕修正。
"""


def components(changed: list[int], radius: int) -> list[list[int]]:
    """Join edits whose visible neighborhoods overlap, so rollback is atomic."""
    groups: list[list[int]] = []
    for ln in sorted(set(changed)):
        if not groups or ln - groups[-1][-1] > 2 * radius:
            groups.append([ln])
        else:
            groups[-1].append(ln)
    return groups


def schema(ids: list[int]) -> dict:
    fields = {key: {"type": "string", "enum": ["yes", "no", "uncertain"]} for key in CHECKS}
    fields["reason"] = {"type": "string", "maxLength": 120}
    row = {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}
    return {"type": "object", "properties": {str(i): row for i in ids},
            "required": [str(i) for i in ids], "additionalProperties": False}


def parse(text: str, ids: list[int]) -> dict[int, dict]:
    from src.quality import _unique_json_object, _invalid_json_constant
    try:
        rows = json.loads(text, object_pairs_hook=_unique_json_object, parse_constant=_invalid_json_constant)
    except (ValueError, TypeError, RecursionError):
        return {}
    if not isinstance(rows, dict) or set(rows) != {str(i) for i in ids}:
        return {}
    for row in rows.values():
        if not isinstance(row, dict) or set(row) != {*CHECKS, "reason"}:
            return {}
        if any(not isinstance(row[key], str) or row[key] not in {"yes", "no", "uncertain"} for key in CHECKS):
            return {}
        reason = row["reason"]
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 120 or "\n" in reason or "\r" in reason:
            return {}
    return {int(i): row for i, row in rows.items()}


def verify_repairs(source: list[SrtBlock], draft: list[SrtBlock], proposed: list[SrtBlock],
                   ledger: list[dict], changed: list[int], config: TranslateConfig, cache: Path
                   ) -> tuple[list[SrtBlock], list[dict]]:
    from src.quality import mode
    final = [replace(b) for b in draft]
    if not changed:
        return final, ledger
    radius = max(1, config.scene_context_lines)
    saved = read_json(cache, {})
    saved = saved if isinstance(saved, dict) else {}
    cfg = mode(config, "repair-delta-verification")
    instruction = build_instruction(cfg, INSTRUCTION)
    for group in components(changed, radius):
        visible = list(range(max(1, group[0] - radius), min(len(source), group[-1] + radius) + 1))
        local = {ln: i for i, ln in enumerate(group, 1)}
        records = []
        for ln in visible:
            records.append({"id": str(local[ln]) if ln in local else f"neighbor-{ln}",
                            "selected_source": source[ln - 1].text,
                            "before": draft[ln - 1].text,
                            "after": proposed[ln - 1].text if ln in local else draft[ln - 1].text,
                            "existing_finding": ledger[ln - 1]["reason"],
                            "source_status": ledger[ln - 1]["source_decision"]["status"]})
        local_ids = list(range(1, len(group) + 1))
        body = json.dumps({"target_ids": local_ids, "rows": records}, ensure_ascii=False)
        key = fingerprint([VERSION, instruction, body, config.endpoint, cfg.extra_payload, cfg.max_tokens])
        cached = saved.get(key, {})
        raw = cached.get("raw_answer") if isinstance(cached, dict) else None
        verdicts = parse(raw, local_ids) if isinstance(raw, str) else {}
        # Large dependency groups are checked in one transaction. Do not split
        # them into independently committed edits or silently truncate context.
        if not verdicts and len(body) <= 24000 and len(group) <= 40:
            extra = dict(cfg.extra_payload or {})
            budget = min(config.max_tokens, max(2048, len(group) * 200))
            extra.update(response_format={"type": "json_object", "schema": schema(local_ids)}, max_tokens=budget)
            request_cfg = replace(cfg, extra_payload=extra, max_tokens=budget,
                                  response_guard_floor=max(cfg.response_guard_floor, 4096))
            raw = call_llm(body, instruction, request_cfg)
            verdicts = parse(raw, local_ids)
            saved[key] = {"raw_answer": raw, "rows": records, "target_ids": local_ids,
                          "global_ids": group, "instruction_sha256": fingerprint(instruction)}
            write_json(cache, saved)
        accepted = bool(verdicts) and all(verdicts[i][check] == "yes" for i in local_ids for check in CHECKS)
        for ln in group:
            row = ledger[ln - 1]
            row["repair_verification"] = {"version": VERSION, "component": group, "visible_ids": visible,
                                          "input_sha256": key, "verdict": verdicts.get(local[ln]),
                                          "committed": accepted}
            row["repair_candidate"] = proposed[ln - 1].text
            if accepted:
                final[ln - 1].text = proposed[ln - 1].text
                row.update(status="corrected", reason="Local repair delta verified; component committed",
                           before=draft[ln - 1].text, after=final[ln - 1].text)
            else:
                row.update(status="unresolved" if verdicts else "unchecked",
                           reason="Repair reverted: component did not pass delta verification" if verdicts else
                           "Repair reverted: missing/invalid or oversized delta verification")
    return final, ledger
