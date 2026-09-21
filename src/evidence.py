"""Immutable audio/script evidence shared by resolution, translation and QA."""
from __future__ import annotations

from dataclasses import dataclass

from src.config import TranslateConfig
from src.translate import SrtBlock, parse_srt
from src.workflow_state import read_json


@dataclass(frozen=True)
class CueEvidence:
    line: int
    raw: str
    grade: str
    n: str = ""
    q: str = ""
    script: str = ""
    script_status: str = "none"
    orthographic_conflict: bool = False
    title_notes: tuple[str, ...] = ()
    title_candidates: tuple[str, ...] = ()
    reference: tuple[str, ...] = ()
    audio_note: str = ""
    ensemble: bool = False
    whisper: str = ""
    source_model: str = "w"
    r: str = ""

    @property
    def hard_conflict(self) -> bool:
        return self.grade == "A" and self.script_status == "mismatch"

    @property
    def blocked(self) -> bool:
        return self.hard_conflict or self.grade in {"C", "D"}

    @property
    def needs_resolution(self) -> bool:
        return (self.grade in {"B+", "B-", "A-音", "E-音", "C", "D"}
                or self.script_status == "mismatch" or self.orthographic_conflict
                or bool(self.title_notes))

    def candidates(self) -> dict[str, str]:
        values = {"W": self.raw, "N": self.n, "Q": self.q, "SCRIPT": self.script}
        if self.ensemble and self.whisper:
            values["WHISPER"] = self.whisper
        if self.ensemble and self.r:
            values["R"] = self.r
        values.update({f"TITLE{i}": text for i, text in enumerate(self.title_candidates, 1)})
        return {key: value for key, value in values.items() if value}

    def notes(self) -> list[str]:
        lines = [f"音频[{self.grade}] W={self.raw} | N={self.n or '(无输出)'} | Q={self.q or '(无输出)'}"]
        if self.ensemble:
            lines = [f"源文[{self.source_model.upper()}]={self.raw}；音频[{self.grade}] "
                     f"Whisper={self.whisper or '(无输出)'} | N={self.n or '(无输出)'} | Q={self.q or '(无输出)'}"]
        if self.ensemble and self.r:
            lines.append(f"NeMo R（与N同属Reazon系，不作额外独立票）={self.r}")
        if self.audio_note:
            lines.append("复听说明: " + self.audio_note)
        if self.script:
            lines.append(f"台本[{self.script_status}; 字面冲突={self.orthographic_conflict}]: {self.script}")
        lines.extend(self.title_notes)
        lines.extend(f"参考译文: {text}" for text in self.reference)
        if self.hard_conflict:
            lines.append("台本/三模型硬冲突：禁止自动修改，交人工核对录音版本。")
        return lines


def build_evidence(source: list[SrtBlock], config: TranslateConfig) -> list[CueEvidence]:
    from src.script_align import build_script_anchors
    from src.title_context import build_title_notes, tokenize_title
    from src.adjudicate import kana_normalize
    from src.review import reference_anchors

    records = read_json(config.adjudication, []) if config.adjudication else []
    if not isinstance(records, list):
        raise ValueError("Arbitration must be a list")
    by_line = {}
    for row in records:
        if not isinstance(row, dict) or type(row.get("line")) is not int:
            raise ValueError("Invalid arbitration cue ID")
        ln = row["line"]
        if ln in by_line or not 1 <= ln <= len(source):
            raise ValueError("Duplicate/out-of-range arbitration cue ID")
        if row.get("w") and row["w"] != source[ln - 1].text.strip():
            raise ValueError(f"Stale arbitration text for cue {ln}")
        by_line[ln] = row
    anchors = build_script_anchors(config.script_file, [b.text for b in source]) if config.script_file else None
    title = build_title_notes(config.title, [b.text for b in source]) if config.title else {}
    terms = tokenize_title(config.title) if config.title else []
    refs = {}
    if config.reference_srt:
        refs = {ln: tuple(b.text for b in blocks) for ln, _, blocks in
                reference_anchors(parse_srt(config.reference_srt), source)}
    result = []
    for ln, block in enumerate(source, 1):
        row = by_line.get(ln, {})
        anchor = anchors[ln - 1] if anchors else None
        candidates = []
        if ln in title:
            for original, _ in tokenize_title(block.text):
                for term, _ in terms:
                    if (term != original and len(kana_normalize(term)) >= 2
                            and kana_normalize(term) == kana_normalize(original)):
                        candidate = block.text.replace(original, term)
                        if candidate not in candidates:
                            candidates.append(candidate)
        result.append(CueEvidence(
            ln, block.text, row.get("grade", "D" if config.adjudication else "not_checked"),
            row.get("n", ""), row.get("q", ""),
            anchor.script_text if anchor else "", anchor.status if anchor else "none",
            anchor.orthographic_conflict if anchor else False,
            tuple(title.get(ln, [])), tuple(candidates[:6]), refs.get(ln, ()), row.get("note", ""),
            bool(row.get("ensemble")), row.get("whisper", ""), row.get("source_model", "w"), row.get("r", ""),
        ))
    return result
