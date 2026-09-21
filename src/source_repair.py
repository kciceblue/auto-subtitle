"""Local fixed-choice source repair, committed only with exact realignment.

The returned source is provisional. ``complete`` describes execution, never
source correctness or release readiness. Raw inputs and source files are immutable;
only the request cache and the separate realignment sidecar are written here.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import logging
from pathlib import Path
import time
from urllib.parse import urlsplit

from src.config import TranscribeConfig, TranslateConfig
from src.quality import mode
from src.source_realign import realign_changed
from src import source_spans
from src.translate import SrtBlock, build_instruction, call_llm
from src.workflow_state import file_hash, fingerprint, read_json, write_json

logger = logging.getLogger(__name__)
VERSION = "source-repair-1"
MAX_WINDOWS = 3
MAX_TOKENS = 2048
DEFAULT_WRITER = "qwen3.8-27b-dflash"


@dataclass
class SourceRepairResult:
    source: list[SrtBlock]
    decisions: list[dict]
    metadata: dict
    report: dict
    complete: bool


def _writer_config(config: TranslateConfig, candidates: list[dict]) -> TranslateConfig:
    if urlsplit(config.endpoint).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Source selection requires the local Qwen writer endpoint")
    configured = mode(config, "source_span_selection", thinking=False)
    extra = deepcopy(configured.extra_payload or {})
    model = extra.setdefault("model", DEFAULT_WRITER)
    if not isinstance(model, str) or not model.startswith("qwen3.8-27b"):
        raise ValueError("Source selection requires the local Qwen 27B writer")
    # Neither messages nor a separate system prompt may replace our bound input.
    if any(key in extra for key in ("messages", "prompt", "system")):
        raise ValueError("Writer payload cannot override fixed source selection inputs")
    extra.update(max_tokens=MAX_TOKENS, response_format={
        "type": "json_object", "schema": source_spans.selection_schema(candidates)})
    return replace(configured, max_tokens=MAX_TOKENS, retries=0, extra_payload=extra,
                   response_guard_floor=max(1024, configured.response_guard_floor))


def _affected_ids(metadata: dict, windows: list[int], ids: set[int]) -> list[int]:
    """Conservative review scope only; never use temporal overlap to edit text."""
    try:
        wanted = [row for row in metadata["windows"] if row["index"] in windows]
        if len(wanted) != len(set(windows)):
            return sorted(ids)
        affected = {segment["index"] for segment in metadata["segments"]
                    if segment["index"] in ids and any(
                        segment["start"] < window["core_end"]
                        and segment["end"] > window["core_start"] for window in wanted)}
        return sorted(affected or ids)
    except (KeyError, TypeError):
        return sorted(ids)


def _pending(row: dict, *, unchecked: bool = False) -> None:
    # A successful mechanical stage cannot upgrade an earlier unchecked stage.
    row["status"] = "unchecked" if unchecked or row.get("status") == "unchecked" else "unresolved"


def repair_source_spans(media: Path, source: list[SrtBlock], decisions: list[dict],
                        metadata: dict, translate_config: TranslateConfig,
                        cache_path: Path, realignment_path: Path, *,
                        protected_surfaces=(),
                        transcribe_config: TranscribeConfig | None = None) -> SourceRepairResult:
    """Select existing raw W+N spans on Q, realign, and return copied results.

    Callers must persist the report and preserve unresolved source blockers. The
    aligner's successful token offsets refer to the repaired utterance, not raw
    ASR windows. Its sidecar retains the original metadata and all timing evidence.
    """
    started = time.monotonic()
    media, cache_path, realignment_path = map(Path, (media, cache_path, realignment_path))
    if not media.is_file():
        raise FileNotFoundError(media)
    paths = [media.resolve(), cache_path.resolve(), realignment_path.resolve()]
    if len(set(paths)) != 3 or any(p.suffix != ".json" or p.name.endswith(".asr.json")
                                  for p in (cache_path, realignment_path)):
        raise ValueError("Use distinct source-selection and realignment JSON sidecars")
    protected = tuple(protected_surfaces)
    if any(not isinstance(value, str) or not value for value in protected):
        raise ValueError("Protected source surfaces must be nonempty strings")
    protected = tuple(sorted(set(protected)))
    ids = {row.index for row in source}
    if (len(ids) != len(source) or any(type(i) is not int or i < 1 for i in ids)
            or not isinstance(metadata, dict)):
        raise ValueError("Unique positive selected-source IDs and ASR metadata required")
    original_decisions = {row.get("line"): row for row in decisions}
    if (len(original_decisions) != len(decisions) or set(original_decisions) != ids
            or any(row.get("text") != block.text
                   for block in source for row in [original_decisions[block.index]])):
        raise ValueError("Source decisions must exactly match the current selected source")
    originals = {block.index: block.text for block in source}
    copied_source, copied_decisions, copied_metadata = deepcopy(source), deepcopy(decisions), deepcopy(metadata)
    rows = {row["line"]: row for row in copied_decisions}
    candidates = source_spans.build_span_candidates(metadata, protected_surfaces=protected)
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["window"]].append(candidate)
    instruction = build_instruction(translate_config, source_spans.source_selection_instruction())
    # Bind the actual legitimate context text, not its file paths. No draft,
    # reference translations or evaluator output are added to the selector body.
    binding = {"version": VERSION, "span_policy": source_spans.VERSION,
        "media_hash": file_hash(media), "source_hash": fingerprint([asdict(b) for b in source]),
        "metadata_hash": fingerprint(metadata), "decisions_hash": fingerprint(decisions),
        "protected_surfaces": protected, "instruction": instruction}
    run_key = fingerprint(binding)
    cached = read_json(cache_path, {})
    if not isinstance(cached, dict) or cached.get("key") != run_key:
        cached = {"version": VERSION, "key": run_key, "binding": binding, "batches": {}}
    if not isinstance(cached.get("batches"), dict):
        cached["batches"] = {}
    report = {"version": VERSION, "key": run_key, "input": binding,
        "candidate_count": len(candidates), "candidate_window_count": len(grouped),
        "candidates": candidates, "batches": [], "patches": [],
        "changed_ids": [], "failed_alignment_ids": {}, "unchecked_ids": [],
        "source_status": "unresolved", "release_approved": False}
    checked_choices, checked_candidates, unchecked = [], [], set()
    complete = True
    windows = sorted(grouped)
    for start in range(0, len(windows), MAX_WINDOWS):
        batch_windows = windows[start:start+MAX_WINDOWS]
        batch = [candidate for window in batch_windows for candidate in grouped[window]]
        config = _writer_config(translate_config, batch)
        body = source_spans.source_selection_body(metadata, batch)
        request = {"run_key": run_key, "instruction": instruction, "body": body,
                   "endpoint": config.endpoint, "payload": config.extra_payload,
                   "max_tokens": config.max_tokens, "thinking": False}
        key = fingerprint(request)
        entry = cached["batches"].get(key, {})
        choices, hit = None, False
        if isinstance(entry, dict) and entry.get("request") == request:
            answer = entry.get("answer")
            if isinstance(answer, str) and entry.get("answer_hash") == fingerprint(answer):
                try:
                    choices = source_spans.parse_span_selections(answer, batch)
                    hit = True
                except (ValueError, TypeError):
                    pass
        attempts = deepcopy(entry.get("attempts", [])) if isinstance(entry, dict) else []
        if not isinstance(attempts, list):
            attempts = []
        if choices is None:
            for attempt in range(2):
                attempt_instruction = instruction
                if attempt:
                    attempt_instruction += ("\nThe previous response failed the fixed JSON contract. "
                        "Return every requested local ID exactly once, only decision and a nonempty "
                        "reason of at most 180 characters; use only the listed choices.")
                call_started, answer = time.monotonic(), None
                error = None
                try:
                    answer = call_llm(body, attempt_instruction, config)
                    choices = source_spans.parse_span_selections(answer, batch)
                except (RuntimeError, OSError, ValueError, TypeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    logger.warning("Source span selection batch %s failed: %s", batch_windows, error)
                attempts.append({"instruction": attempt_instruction, "body_hash": fingerprint(body),
                    "response": answer, "error": error, "wall_seconds": time.monotonic()-call_started})
                if choices is not None:
                    break
            entry = {"request": request, "attempts": attempts,
                     "answer": answer if choices is not None else None,
                     "answer_hash": fingerprint(answer) if choices is not None else None,
                     "status": "checked" if choices is not None else "unchecked"}
            cached["batches"][key] = entry
            write_json(cache_path, cached)
        batch_record = {"key": key, "windows": batch_windows, "cache_hit": hit,
                        "status": "checked" if choices is not None else "unchecked",
                        "attempts": attempts, "choices": choices}
        report["batches"].append(batch_record)
        if choices is None:
            complete = False
            affected = _affected_ids(metadata, batch_windows, ids)
            unchecked.update(affected)
            for uid in affected:
                _pending(rows[uid], unchecked=True)
            batch_record["affected_source_ids"] = affected
        else:
            checked_choices.extend(choices)
            checked_candidates.extend(batch)
    projection = {"source": originals, "changed_ids": [], "decisions": []}
    try:
        media_unchanged = file_hash(media) == binding["media_hash"]
    except OSError:
        media_unchanged = False
    if not media_unchanged:
        complete = False
        unchecked.update(ids)
        for row in rows.values():
            _pending(row, unchecked=True)
        report["input_error"] = "Original media changed during source selection"
    elif checked_candidates:
        projection = source_spans.project_span_selections(originals, metadata, checked_choices,
            checked_candidates, protected_surfaces=protected)
    patches = deepcopy(projection["decisions"])
    # Script/audio hard conflicts are an existing no-auto-edit boundary. A
    # lattice vote cannot override it, even when the new span is well anchored.
    blocked = {uid for uid in projection["changed_ids"]
               if (original_decisions[uid].get("evidence") or {}).get("hard_conflict")}
    for uid in blocked:
        projection["source"][uid] = originals[uid]
    projection["changed_ids"] = [uid for uid in projection["changed_ids"] if uid not in blocked]
    for row in patches:
        if row.get("utterance_id") in blocked and row.get("applied"):
            row.update(applied=False, projected=True, status="rejected",
                       timing_status="not_attempted_hard_conflict",
                       rejection="Existing hard script/audio conflict forbids automatic source editing")
    report["hard_conflict_blocked_ids"] = sorted(blocked)
    proposed = [replace(block, text=projection["source"][block.index]) for block in source]
    changed = set(projection["changed_ids"])
    succeeded, failed, alignment_report = set(), {}, None
    if changed:
        try:
            alignment = realign_changed(media, proposed, sorted(changed), metadata,
                transcribe_config, realignment_path)
            succeeded = set(alignment.successful_ids)
            failed = dict(alignment.failed_ids)
            if (succeeded & set(failed) or succeeded | set(failed) != changed):
                raise ValueError("Realignment result must cover exact changed source IDs")
            alignment_report = alignment.report
            # Import only successful token rows; never copy altered raw metadata
            # or accept tokens for an ID whose source patch will be rolled back.
            for index, block in enumerate(source):
                if block.index in succeeded:
                    tokens = alignment.metadata["utterance_tokens"][index]
                    if (not tokens or "".join(t["text"] for t in tokens) != proposed[index].text
                            or any(t.get("source_id") != block.index
                                or t.get("source_hash") != fingerprint(proposed[index].text)
                                or t.get("offset_scope") != "utterance" for t in tokens)):
                        failed[block.index] = "Realignment tokens do not bind the exact repaired source"
                        succeeded.remove(block.index)
                    else:
                        copied_metadata["utterance_tokens"][index] = deepcopy(tokens)
        except (RuntimeError, OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            succeeded, failed = set(), {uid: str(exc) for uid in changed}
            copied_metadata = deepcopy(metadata)
        if failed:
            complete = False
        for index, block in enumerate(source):
            if block.index in succeeded:
                copied_source[index].text = proposed[index].text
    for patch in patches:
        uid = patch.get("utterance_id")
        if patch.get("applied"):
            patch["projected"] = True
            patch["applied"] = uid in succeeded
            if uid in succeeded:
                patch.update(status="realigned_unresolved", timing_status="mechanically_realigned",
                             realignment_path=str(realignment_path),
                             realignment_key=(alignment_report or {}).get("request_key"))
            else:
                patch.update(status="reverted_unresolved", timing_status="failed",
                             rejection=failed.get(uid, "Realignment did not succeed"))
        affected = [uid] if uid in ids else _affected_ids(metadata, [patch["window"]], ids)
        if patch["status"] != "keep":
            for source_id in affected:
                _pending(rows[source_id])
    # Keep prior evidence, confidence and reasons inspectable for every affected
    # row. The selector's KEEP never clears an existing doubt.
    for index, block in enumerate(source):
        uid = block.index
        relevant = [patch for patch in patches if patch.get("utterance_id") == uid
                    or (patch.get("utterance_id") is None
                        and uid in _affected_ids(metadata, [patch["window"]], ids))]
        if relevant or uid in unchecked:
            rows[uid]["source_span_repair"] = {"version": VERSION,
                "prior_decision": deepcopy(original_decisions[uid]), "patches": deepcopy(relevant),
                "selection_checked": uid not in unchecked, "source_status": "unresolved",
                "original_text": block.text, "final_text": copied_source[index].text,
                "realignment_path": str(realignment_path) if uid in changed else None,
                "realignment_failure": failed.get(uid)}
        rows[uid]["text"] = copied_source[index].text
    report.update(patches=patches, changed_ids=sorted(succeeded),
        projected_changed_ids=sorted(changed), failed_alignment_ids={str(k): v for k, v in failed.items()},
        unchecked_ids=sorted(unchecked), realignment=alignment_report,
        complete=complete, output_source_hash=fingerprint([asdict(b) for b in copied_source]),
        output_metadata_hash=fingerprint(copied_metadata), wall_seconds=time.monotonic()-started)
    return SourceRepairResult(copied_source, copied_decisions, copied_metadata, report, complete)
