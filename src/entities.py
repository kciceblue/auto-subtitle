"""Grounded local entity hints and lossless temporary translation placeholders.

This module never edits source subtitles or timing. Model classifications remain
unresolved identity claims. Call ``restore_entity_target`` BEFORE target length or
fidelity checks: an opaque marker can be much longer than the actual source name.
A marker failure must be retried locally or fall back to direct translation with
identity unchecked, never repaired by appending or guessing a missing name.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from difflib import SequenceMatcher
from functools import lru_cache
import json
import logging
from pathlib import Path
import re
from typing import Any

from src.asr_consensus import reading_map
from src.config import TranslateConfig
from src.quality import _invalid_json_constant, _unique_json_object, mode
from src.translate import SrtBlock, call_llm
from src.workflow_state import fingerprint, read_json, write_json

logger = logging.getLogger(__name__)
VERSION = "entities-5"
_MARKER = re.compile(r"⟦E\d+⟧")
_NEGATIVE = re.compile(r"不是|并非|不得|不要|错误|误写|誤写|誤り|間違|ではない|ではなく|(?<![A-Za-z])(?:not|never|incorrect|wrong)(?![A-Za-z])", re.I)
_EXTRACT = """Extract named entities ONLY from supplied numbered original paragraphs.
Never invent a source form, target translation, alias, reading or story fact.
Include an entity only when its canonical source AND target rendering appear
explicitly in one supplied paragraph. Return its paragraph_id; the client copies
that exact original paragraph as evidence, so do not generate any quotation.
ALL returned target/short_target/aliases must occur verbatim in that SAME paragraph.
Negated examples and wrong spellings are not aliases. A surface explicitly described
as a possible ASR name is a conditional alias (kind=conditional); an unambiguous
declared alias uses kind=explicit. Occurrence-level disambiguation happens later.
Do not extract common subject pronouns as aliases. A short target form must also be
explicitly supplied; otherwise short_target is empty. No external knowledge.
Return JSON only with entities, each containing source, target, short_target,
paragraph_id and aliases (text, kind). At most 30 entities and 12 aliases each.
"""
_CLASSIFY = """Classify each marked source occurrence using ONLY chosen source,
read-only local neighbors and the supplied original entity definitions/quotes.
Return one exact local JSON ID per task, with decision and a short reason.
Decision is an allowed entity ID, ORDINARY_WORD or UNKNOWN. Literal alias matches
and agreement among ASR models do not prove identity. Conditional ASR aliases can
be ordinary words. Background relationships cannot identify a local speaker or
addressee without dialogue evidence. Do not exchange identities by proximity,
complete a clipped name, rewrite source, translate, or invent information.
If the occurrence is part of a longer name or its identity is unclear, UNKNOWN.
All results are provisional; ORDINARY_WORD/UNKNOWN never force later translation.
"""


@dataclass(frozen=True)
class EntityAlias:
    text: str
    quote: str
    kind: str = "explicit"


@dataclass(frozen=True)
class EntityDefinition:
    id: str
    source: str
    target: str
    short_target: str
    source_quote: str
    target_quote: str
    short_quote: str
    aliases: tuple[EntityAlias, ...] = ()

    def surfaces(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([self.source, *(a.text for a in self.aliases)]))


@dataclass
class EntityLexicon:
    context: str
    context_hash: str
    entities: list[EntityDefinition] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    degraded: bool = False

    @property
    def key(self) -> str:
        return fingerprint([VERSION, self.context_hash, [asdict(e) for e in self.entities]])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EntityOccurrence:
    id: str
    source_id: int
    start: int
    end: int
    surface: str
    source_hash: str
    allowed_entities: tuple[str, ...]


@dataclass(frozen=True)
class EntityDecision:
    occurrence_id: str
    decision: str
    reason: str
    checked: bool = True


@dataclass
class EntityClassifications:
    source_hashes: dict[int, str]
    lexicon_hash: str
    occurrences: list[EntityOccurrence]
    decisions: list[EntityDecision]
    issues: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EntityReplacement:
    occurrence_id: str
    start: int
    end: int
    source: str
    marker: str
    entity_id: str
    target: str
    evidence_quote: str


@dataclass(frozen=True)
class EntityRowPlan:
    source_id: int
    timestamp: str
    original_source: str
    marked_source: str
    replacements: tuple[EntityReplacement, ...] = ()

    @property
    def replacement_map(self) -> dict[str, str]:
        return {r.marker: r.target for r in self.replacements}

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EntityPlan:
    marked_source: list[SrtBlock]
    rows: dict[int, EntityRowPlan]
    ledger: list[dict]
    notes: dict[int, list[str]]
    context_hash: str
    lexicon_hash: str
    source_hashes: dict[int, str]
    metadata_hash: str

    @property
    def key(self) -> str:
        return fingerprint(self.to_dict())

    def translation_instruction(self, ids: list[int] | None = None) -> str:
        """Return marker/name notes for writable rows and a strict output rule."""
        selected = set(self.rows) if ids is None else set(ids)
        replacements = [r for source_id, row in self.rows.items() if source_id in selected
                        for r in row.replacements]
        if not replacements:
            return ""
        values = "\n".join(f"{r.marker}: supplied name {r.target}" for r in replacements)
        return ("Protected name occurrences (provisional identity hints):\n" + values
                + "\nKeep each exact marker in its own target row exactly once; preserve "
                "its grammatical role. Do not spell out, remove, duplicate or move it to "
                "another row. The client restores the supplied name afterward. Names do "
                "not establish a speaker/addressee or permit missing dialogue. Ordinary "
                "and unknown words outside markers remain unconstrained.\n")

    def to_dict(self) -> dict:
        return {"version": VERSION, "marked_source": [asdict(r) for r in self.marked_source],
                "rows": {str(k): r.to_dict() for k, r in self.rows.items()},
                "ledger": self.ledger, "notes": {str(k): v for k, v in self.notes.items()},
                "context_hash": self.context_hash, "lexicon_hash": self.lexicon_hash,
                "source_hashes": {str(k): v for k, v in self.source_hashes.items()},
                "metadata_hash": self.metadata_hash}

    @classmethod
    def from_dict(cls, data: dict) -> EntityPlan:
        if data.get("version") != VERSION:
            raise ValueError("Unsupported entity plan version")
        rows = {}
        for key, value in data["rows"].items():
            row = dict(value)
            row["replacements"] = tuple(EntityReplacement(**r) for r in row["replacements"])
            rows[int(key)] = EntityRowPlan(**row)
        plan = cls([SrtBlock(**r) for r in data["marked_source"]], rows,
                   data["ledger"], {int(k): v for k, v in data["notes"].items()},
                   data["context_hash"], data["lexicon_hash"],
                   {int(k): v for k, v in data["source_hashes"].items()}, data["metadata_hash"])
        if (len(plan.marked_source) != len(rows)
                or {r.index for r in plan.marked_source} != set(rows)):
            raise ValueError("Entity plan row count mismatch")
        for block in plan.marked_source:
            row = rows[block.index]
            if (row.source_id != block.index or row.timestamp != block.ts_line
                    or row.marked_source != block.text
                    or fingerprint(row.original_source) != plan.source_hashes[block.index]
                    or restore_entity_source(row) != row.original_source):
                raise ValueError("Invalid or stale serialized entity plan")
        return plan


class EntityMarkerError(ValueError):
    """A target lost/duplicated a protected occurrence; caller must not commit it."""


def _json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_json_object,
                      parse_constant=_invalid_json_constant)


def _schema_config(config: TranslateConfig, stage: str, schema: dict,
                   tokens: int) -> TranslateConfig:
    tokens = min(tokens, config.max_tokens)
    cfg = mode(config, stage)
    extra = dict(cfg.extra_payload or {})
    extra["response_format"] = {"type": "json_object", "schema": schema}
    extra["max_tokens"] = tokens
    return replace(cfg, extra_payload=extra, max_tokens=tokens,
                   response_guard_floor=max(1024, cfg.response_guard_floor))


def _grounded(context: str, value: str, quote: str, source: str = "") -> bool:
    if (not isinstance(value, str) or not value.strip() or len(value) > 80
            or value != value.strip() or "\n" in value or "⟦" in value or "⟧" in value
            or not isinstance(quote, str) or not quote or len(quote) > 1500
            or quote not in context or value not in quote or (source and source not in quote)):
        return False
    # Inspect the clause before this occurrence, not a later negative example.
    # This is a conservative rejection rule; exact quotes do not prove semantics.
    for found in re.finditer(re.escape(value), quote):
        prefix = re.split(r"[。！？!?;；\n]", quote[:found.start()])[-1]
        if not _NEGATIVE.search(prefix):
            return True
    return False


def _parse_lexicon(text: str, context: str) -> EntityLexicon:
    value = _json(text)
    if not isinstance(value, dict) or set(value) != {"entities"} or not isinstance(value["entities"], list) or len(value["entities"]) > 30:
        raise ValueError("Invalid entity lexicon JSON")
    result = EntityLexicon(context, fingerprint(context))
    required = {"source", "target", "short_target", "source_quote", "target_quote", "short_quote", "aliases"}
    seen = set()
    canonical_counts = Counter(r.get("source") for r in value["entities"]
                               if isinstance(r, dict) and isinstance(r.get("source"), str))
    for number, row in enumerate(value["entities"], 1):
        reason = None
        if not isinstance(row, dict) or set(row) != required:
            reason = "Invalid definition fields"
        elif not all(isinstance(row[k], str) for k in required - {"aliases"}):
            reason = "Definition text fields must be strings"
        elif not _grounded(context, row["source"], row["source_quote"]):
            reason = "Source form lacks positive original quotation"
        elif not _grounded(context, row["target"], row["target_quote"], row["source"]):
            reason = "Target form lacks a quoted source association"
        elif row["short_target"] and not _grounded(context, row["short_target"], row["short_quote"], row["source"]):
            reason = "Short target form is not explicitly supplied"
        elif not isinstance(row["aliases"], list) or len(row["aliases"]) > 12:
            reason = "Invalid alias list"
        elif canonical_counts[row["source"]] > 1:
            reason = "Ambiguous duplicate canonical source definition"
        if reason:
            result.issues.append({"definition": number, "reason": reason})
            continue
        aliases = []
        for alias in row["aliases"]:
            if (not isinstance(alias, dict) or set(alias) != {"text", "quote", "kind"}
                    or alias["kind"] not in {"explicit", "conditional"}
                    or not _grounded(context, alias["text"], alias["quote"], row["source"])):
                result.issues.append({"definition": number, "reason": "Unsupported/negated alias"})
                continue
            if alias["text"] not in {a.text for a in aliases} and alias["text"] != row["source"]:
                aliases.append(EntityAlias(**alias))
        seen.add(row["source"])
        result.entities.append(EntityDefinition(f"g{len(result.entities)+1}", row["source"], row["target"],
            row["short_target"] or row["target"], row["source_quote"], row["target_quote"],
            row["short_quote"] or row["target_quote"], tuple(aliases)))
    result.degraded = bool(result.issues)
    return result


def _parse_extraction(text: str, context: str) -> EntityLexicon:
    """Expand a shared exact paragraph mechanically; no generated associations."""
    data = _json(text)
    if isinstance(data, dict) and isinstance(data.get("entities"), list):
        for row in data["entities"]:
            if isinstance(row, dict) and set(row) == {"source", "target", "short_target", "paragraph_id", "aliases"}:
                paragraphs = [line for line in context.splitlines() if line.strip()]
                index = row.pop("paragraph_id")
                if type(index) is not int or not 1 <= index <= len(paragraphs):
                    raise ValueError("Invalid original paragraph ID")
                row["quote"] = paragraphs[index - 1]
            if isinstance(row, dict) and set(row) == {"source", "target", "short_target", "quote", "aliases"}:
                quote = row.pop("quote")
                if not isinstance(row["aliases"], list):
                    raise ValueError("Invalid shared-quote aliases")
                for alias in row["aliases"]:
                    if not isinstance(alias, dict) or set(alias) != {"text", "kind"}:
                        raise ValueError("Invalid shared-quote alias fields")
                    alias["quote"] = quote
                row.update(source_quote=quote, target_quote=quote, short_quote=quote if row["short_target"] else "")
    return _parse_lexicon(json.dumps(data, ensure_ascii=False), context)


def extract_entity_lexicon(context: str, config: TranslateConfig,
                           cache_path: Path) -> EntityLexicon:
    """Ask only the configured local writer for exact supplied entity declarations."""
    if not context.strip():
        return EntityLexicon(context, fingerprint(context))
    key = fingerprint([VERSION, _EXTRACT, context, config.endpoint, config.extra_payload, config.max_tokens,
                       config.source_lang, config.target_lang])
    cached = read_json(cache_path, {})
    if isinstance(cached, dict) and cached.get("key") == key and isinstance(cached.get("response"), str):
        try:
            parsed = _parse_extraction(cached["response"], context)
            if not parsed.degraded:
                return parsed
        except (ValueError, TypeError, KeyError):
            pass
    string = {"type": "string", "maxLength": 1500}
    alias = {"type": "object", "properties": {"text": string,
             "kind": {"type": "string", "enum": ["explicit", "conditional"]}},
             "required": ["text", "kind"], "additionalProperties": False}
    props = {k: string for k in ("source", "target", "short_target")}
    paragraphs = [line for line in context.splitlines() if line.strip()]
    props["paragraph_id"] = {"type": "integer", "minimum": 1, "maximum": len(paragraphs)}
    body = json.dumps({"original_paragraphs": {str(i): line for i, line in enumerate(paragraphs, 1)}}, ensure_ascii=False)
    props["aliases"] = {"type": "array", "items": alias, "maxItems": 12}
    schema = {"type": "object", "properties": {"entities": {"type": "array", "maxItems": 30,
              "items": {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}}},
              "required": ["entities"], "additionalProperties": False}
    instruction = _EXTRACT
    result = EntityLexicon(context, fingerprint(context), degraded=True)
    for attempt in range(2):
        response = ""
        try:
            response = call_llm(body, instruction, _schema_config(config, "entity-lexicon", schema, 4096))
            result = _parse_extraction(response, context)
            if not result.degraded:
                write_json(cache_path, {"key": key, "response": response, "lexicon": result.to_dict()})
                return result
        except RuntimeError as exc:
            # HTTP transport already exhausted its own retries. Do not restart it.
            logger.warning("Entity extraction unavailable: %s", exc)
            result = EntityLexicon(context, fingerprint(context), issues=[{"reason": str(exc)}], degraded=True)
            break
        except (ValueError, TypeError, KeyError) as exc:
            result = EntityLexicon(context, fingerprint(context), issues=[{"reason": str(exc)}], degraded=True)
        rejection_path = cache_path.with_suffix(".rejections.json")
        rejections = read_json(rejection_path, [])
        if not isinstance(rejections, list):
            rejections = []
        rejections.append({"request_key": key, "attempt": attempt + 1, "instruction": instruction,
                           "raw_answer": response, "issues": result.issues})
        write_json(rejection_path, rejections)
        instruction = _EXTRACT + "\nPrevious JSON failed these grounding checks: " + json.dumps(result.issues, ensure_ascii=False) + "\nRegenerate the complete supported entity list from the original context. Choose a paragraph_id whose exact original paragraph contains EVERY returned form and associates it with the canonical source. No new facts."
    return result


@lru_cache(maxsize=4096)
def _boundaries(text: str) -> tuple[frozenset[int], frozenset[int]]:
    tokenizer = _tokenizer()
    starts, ends, cursor = set(), set(), 0
    for token in tokenizer.tokenize(text):
        starts.add(cursor)
        cursor += len(token.surface)
        ends.add(cursor)
    return frozenset(starts), frozenset(ends)


@lru_cache(maxsize=1)
def _tokenizer():
    from janome.tokenizer import Tokenizer
    return Tokenizer()


def _identities(lexicon: EntityLexicon) -> dict[str, set[str]]:
    result = defaultdict(set)
    for entity in lexicon.entities:
        for surface in entity.surfaces():
            result[surface].add(entity.id)
    return dict(result)


def _occurrences(source: list[SrtBlock], lexicon: EntityLexicon) -> list[EntityOccurrence]:
    identities = _identities(lexicon)
    result = []
    for block in source:
        starts, ends = _boundaries(block.text)
        spans = defaultdict(set)
        for surface, entities in identities.items():
            for match in re.finditer(re.escape(surface), block.text):
                if match.start() in starts and match.end() in ends:
                    spans[(match.start(), match.end(), surface)].update(entities)
        # Prefer only longest complete declared aliases. Tied identities remain
        # alternatives for the classifier; overlapping disjoint spans stay audit-only.
        for (start, end, surface), entities in sorted(spans.items()):
            if any(a <= start and end <= b and b-a > end-start for a, b, _ in spans):
                continue
            result.append(EntityOccurrence(f"E{len(result)+1:04d}", block.index, start, end,
                surface, fingerprint(block.text), tuple(sorted(entities))))
    return result


def classify_entity_occurrences(source: list[SrtBlock], lexicon: EntityLexicon,
                                config: TranslateConfig, cache_path: Path, *,
                                batch_size: int = 12, context_lines: int = 2) -> EntityClassifications:
    if batch_size < 1 or batch_size > 24 or context_lines < 0:
        raise ValueError("Invalid entity classification batch/context bounds")
    by_id = {r.index: r for r in source}
    if len(by_id) != len(source):
        raise ValueError("Duplicate selected source IDs")
    occurrences = _occurrences(source, lexicon)
    result = EntityClassifications({r.index: fingerprint(r.text) for r in source}, lexicon.key, occurrences, [])
    saved = read_json(cache_path, {})
    saved = saved if isinstance(saved, dict) else {}
    positions = {row.index: i for i, row in enumerate(source)}
    for offset in range(0, len(occurrences), batch_size):
        batch = occurrences[offset:offset+batch_size]
        visible = set()
        for occ in batch:
            pos = positions[occ.source_id]
            visible.update(range(max(0, pos-context_lines), min(len(source), pos+context_lines+1)))
        payload = {"source_neighborhood": [{"source_id": source[i].index, "text": source[i].text} for i in sorted(visible)],
            "entities": [asdict(e) for e in lexicon.entities],
            "occurrences": {str(i): {"source_id": o.source_id, "surface": o.surface,
                "source_before": by_id[o.source_id].text[:o.start], "source_after": by_id[o.source_id].text[o.end:],
                "allowed_entities": list(o.allowed_entities)} for i, o in enumerate(batch, 1)}}
        body = json.dumps(payload, ensure_ascii=False)
        key = fingerprint([VERSION, _CLASSIFY, body, lexicon.key, config.endpoint, config.extra_payload, config.max_tokens,
                           config.source_lang, config.target_lang])
        props = {str(i): {"type": "object", "properties": {
                 "decision": {"type": "string", "enum": ["ORDINARY_WORD", "UNKNOWN", *o.allowed_entities]},
                 "reason": {"type": "string", "maxLength": 120}},
                 "required": ["decision", "reason"], "additionalProperties": False} for i, o in enumerate(batch, 1)}
        schema = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
        response = saved.get(key)
        try:
            if not isinstance(response, str):
                response = call_llm(body, _CLASSIFY, _schema_config(config, "entity-classify", schema, 2048))
            parsed = _json(response)
            if not isinstance(parsed, dict) or set(parsed) != set(props):
                raise ValueError("Entity decisions require every exact local ID")
            decisions = []
            for i, occ in enumerate(batch, 1):
                row = parsed[str(i)]
                if (not isinstance(row, dict) or set(row) != {"decision", "reason"}
                        or row["decision"] not in ["ORDINARY_WORD", "UNKNOWN", *occ.allowed_entities]
                        or not isinstance(row["reason"], str) or not 1 <= len(row["reason"]) <= 120):
                    raise ValueError("Invalid entity decision/identity/reason")
                decisions.append(EntityDecision(occ.id, row["decision"], row["reason"]))
            result.decisions.extend(decisions)
            saved[key] = response
            write_json(cache_path, saved)
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            if key in saved:
                saved.pop(key, None)
                write_json(cache_path, saved)
            result.issues.append({"occurrence_ids": [o.id for o in batch], "reason": str(exc)})
            result.decisions.extend(EntityDecision(o.id, "UNKNOWN", str(exc), checked=False) for o in batch)
    return result


def _reading_interval(mapping, start: int, end: int):
    ids = [i for i, (a, b) in enumerate(mapping) if b > start and a < end]
    if not ids:
        return None
    lo, hi = ids[0], ids[-1]+1
    if (min(a for a, b in mapping[lo:hi]) != start or max(b for a, b in mapping[lo:hi]) != end
            or any(a < start or b > end for a, b in mapping[lo:hi])):
        return None
    return lo, hi


def _competing_name(base: str, start: int, end: int, entity_id: str,
                    raw: dict[str, str], lexicon: EntityLexicon) -> tuple[bool, str, list[dict]]:
    identities = _identities(lexicon)
    starts, ends = _boundaries(base)
    for alias in identities:
        for match in re.finditer(re.escape(alias), base):
            if (match.start() in starts and match.end() in ends and match.start() <= start
                    and end <= match.end() and len(alias) > end-start):
                return True, "Part of a longer complete declared alias", []
    reading, mapping = reading_map(base)
    interval = _reading_interval(mapping, start, end)
    if interval is None:
        return True, "Cannot bound full name reading", []
    lo, hi = interval
    left, right = reading[max(0, lo-4):lo], reading[hi:hi+4]
    if len(left) < 2 or len(right) < 2:
        return True, "Insufficient two-sided raw name context", []
    evidence = []
    for family in ("w", "n"):
        text = raw.get(family, "")
        alt, amap = reading_map(text)
        opcodes = SequenceMatcher(None, reading, alt, autojunk=False).get_opcodes()
        def mapped(a, b):
            for tag, a0, a1, b0, b1 in opcodes:
                if tag == "equal" and a0 <= a and b <= a1:
                    return b0+a-a0, b0+b-a0
        lm, rm = mapped(lo-len(left), lo), mapped(hi, hi+len(right))
        if lm is None or rm is None or lm[1] >= rm[0]:
            continue
        a, b = lm[1], rm[0]
        pairs = []
        for match in re.finditer("(?=" + re.escape(left) + ")", alt):
            first = match.start()+len(left)
            for last in range(first, min(len(alt)-len(right), first+24)+1):
                if alt.startswith(right, last):
                    pairs.append((first, last))
        if pairs != [(a, b)]:
            continue
        x, y = amap[a][0], amap[b-1][1]
        if _reading_interval(amap, x, y) != (a, b):
            continue
        boundaries, endings = _boundaries(text)
        matches = []
        for alias, ids in identities.items():
            for match in re.finditer(re.escape(alias), text[x:y]):
                if x+match.start() in boundaries and x+match.end() in endings:
                    matches.append({"surface": alias, "entity_ids": sorted(ids),
                                    "start": x+match.start(), "end": x+match.end()})
        evidence.append({"family": family, "start": x, "end": y, "text": text[x:y],
                         "raw_hash": fingerprint(text), "names": matches})
        if any(any(name != entity_id for name in match["entity_ids"]) for match in matches):
            return True, "Same-position raw evidence supports a different/longer known name", evidence
    return not evidence, "No unique raw name context" if not evidence else "", evidence


def _token_positions(source: list[SrtBlock], metadata: dict) -> dict[tuple[int, int], list[dict]]:
    """Map chosen-source character positions to exact owner-window positions."""
    mapped = defaultdict(list)
    groups = metadata.get("utterance_tokens", [])
    if len(groups) != len(source):
        return mapped
    windows = metadata.get("windows", [])
    selections = {r["window"]: r["selected"] for r in metadata.get("source_selection", [])}
    used = metadata.get("used_transcripts", [])
    for block, tokens in zip(source, groups):
        if any(t.get("offset_scope") == "utterance" for t in tokens):
            continue  # Repaired alignments are not raw-window identity evidence.
        if block.text != "".join(t["text"] for t in tokens):
            continue
        offset = 0
        for number, token in enumerate(tokens):
            middle = (token["start"]+token["end"])/2
            owners = [w for w in windows if w["core_start"] <= middle < w["core_end"]]
            if len(owners) == 1:
                w = owners[0]["index"]
                base = used[w].get(selections.get(w, ""), "") if w < len(used) else ""
                piece = base[token["begin"]:token["finish"]]
                if piece.strip() == token["text"]:
                    leading = len(piece)-len(piece.lstrip())
                    for i, char in enumerate(token["text"]):
                        mapped[(block.index, offset+i)].append({"window": w,
                            "raw_offset": token["begin"]+leading+i, "base": base,
                            "token_start": offset, "token_end": offset+len(token["text"]),
                            "token": number, "character": char})
            offset += len(token["text"])
    return mapped


def build_entity_plan(source: list[SrtBlock], classifications: EntityClassifications,
                      lexicon: EntityLexicon, raw_metadata: dict | None = None) -> EntityPlan:
    metadata = raw_metadata or {}
    by_id = {r.index: r for r in source}
    if len(by_id) != len(source):
        raise ValueError("Duplicate source IDs")
    if classifications.lexicon_hash != lexicon.key:
        raise ValueError("Entity classifications refer to a different lexicon")
    try:
        positions = _token_positions(source, metadata)
    except (KeyError, TypeError, IndexError, ValueError):
        positions = {}  # Optional identity hints cannot rely on malformed timing metadata.
    entities = {e.id: e for e in lexicon.entities}
    decisions = defaultdict(list)
    for decision in classifications.decisions:
        decisions[decision.occurrence_id].append(decision)
    ledger, eligible = [], []
    for occ in classifications.occurrences:
        record = {"occurrence": asdict(occ), "status": "unforced", "source_status": "unresolved"}
        ledger.append(record)
        answers = decisions[occ.id]
        if len(answers) != 1 or not answers[0].checked:
            record["reason"] = "Missing, duplicate or unchecked entity label"
            continue
        answer = answers[0]
        record["classification"] = asdict(answer)
        if answer.decision not in entities:
            record["reason"] = "Ordinary/unknown remains unforced"
            continue
        entity = entities[answer.decision]
        block = by_id.get(occ.source_id)
        if (block is None or fingerprint(block.text) != occ.source_hash
                or classifications.source_hashes.get(occ.source_id) != occ.source_hash
                or not 0 <= occ.start < occ.end <= len(block.text)
                or block.text[occ.start:occ.end] != occ.surface):
            record["reason"] = "Stale chosen-source hash/span"
            continue
        if entity.id not in occ.allowed_entities or occ.surface not in entity.surfaces():
            record["reason"] = "No literal declared alias for chosen entity"
            continue
        chars = [positions.get((occ.source_id, i), []) for i in range(occ.start, occ.end)]
        if any(len(c) != 1 for c in chars):
            record["reason"] = "No unique exact aligned-token ownership"
            continue
        owned = [c[0] for c in chars]
        windows = {c["window"] for c in owned}
        starts, ends = _boundaries(block.text)
        if occ.start not in starts or occ.end not in ends or len(windows) != 1:
            record["reason"] = "Incomplete alias token or cross-window name"
            continue
        first, last = owned[0]["raw_offset"], owned[-1]["raw_offset"]+1
        if [c["raw_offset"] for c in owned] != list(range(first, last)):
            record["reason"] = "Noncontiguous raw name positions"
            continue
        partial = False
        for token in {c["token"] for c in owned}:
            item = next(c for c in owned if c["token"] == token)
            a, b = item["token_start"], item["token_end"]
            outside = block.text[a:min(b, occ.start)]+block.text[max(a, occ.end):b]
            partial |= any(c.isalnum() for c in outside)
        if partial:
            record["reason"] = "Partial aligned lexical token"
            continue
        window = next(iter(windows))
        raw = metadata.get("raw_transcripts", [])
        if window >= len(raw):
            record["reason"] = "Missing original raw recognizer evidence"
            continue
        blocked, reason, evidence = _competing_name(owned[0]["base"], first, last, entity.id, raw[window], lexicon)
        record["raw_name_evidence"] = evidence
        if blocked:
            record["reason"] = reason
            continue
        target = entity.target if occ.surface == entity.source else entity.short_target
        quote = entity.target_quote if occ.surface == entity.source else entity.short_quote
        replacement = EntityReplacement(occ.id, occ.start, occ.end, occ.surface,
            f"⟦{occ.id}⟧", entity.id, target, quote)
        eligible.append((occ.source_id, replacement, record))
    conflicts = set()
    for i, (source_id, a, _) in enumerate(eligible):
        for other_id, b, _ in eligible[i+1:]:
            if source_id == other_id and a.start < b.end and b.start < a.end:
                conflicts.update([a.occurrence_id, b.occurrence_id])
    replacements = defaultdict(list)
    for source_id, replacement, record in eligible:
        if replacement.occurrence_id in conflicts:
            record["reason"] = "Overlapping/multiple entity labels"
        else:
            replacements[source_id].append(replacement)
            record.update(status="placeholder_unresolved", replacement=asdict(replacement))
    marked, rows, notes = [], {}, {}
    for block in source:
        if "⟦" in block.text or "⟧" in block.text:
            raise EntityMarkerError("Original source collides with reserved markers")
        repl = tuple(sorted(replacements[block.index], key=lambda r: r.start))
        text = block.text
        for item in reversed(repl):
            text = text[:item.start]+item.marker+text[item.end:]
        row = EntityRowPlan(block.index, block.ts_line, block.text, text, repl)
        if restore_entity_source(row) != block.text:
            raise EntityMarkerError("Temporary source is not lossless")
        rows[block.index] = row
        marked.append(replace(block, text=text))
        if repl:
            notes[block.index] = [f"Preserve {r.marker} exactly once for supplied name {r.target}; identity remains unresolved." for r in repl]
    return EntityPlan(marked, rows, ledger, notes, lexicon.context_hash, lexicon.key,
                      {r.index: fingerprint(r.text) for r in source}, fingerprint(metadata))


def _marker_counts(text: str) -> Counter:
    if not isinstance(text, str):
        raise EntityMarkerError("Target must be text")
    rest = _MARKER.sub("", text)
    if "⟦" in rest or "⟧" in rest:
        raise EntityMarkerError("Malformed or foreign entity marker")
    return Counter(_MARKER.findall(text))


def _validate_row(row_plan: EntityRowPlan) -> None:
    prior = 0
    ordered = sorted(row_plan.replacements, key=lambda r: r.start)
    for item in ordered:
        if (not 0 <= item.start < item.end <= len(row_plan.original_source)
                or item.start < prior or row_plan.original_source[item.start:item.end] != item.source
                or not _MARKER.fullmatch(item.marker) or not item.target
                or item.target not in item.evidence_quote
                or any(c in item.target + item.source for c in "⟦⟧")):
            raise EntityMarkerError("Invalid or stale entity row replacement")
        prior = item.end
    marked = row_plan.original_source
    for item in reversed(ordered):
        marked = marked[:item.start] + item.marker + marked[item.end:]
    if marked != row_plan.marked_source:
        raise EntityMarkerError("Entity row text differs from its exact source spans")


def _restore(text: str, row_plan: EntityRowPlan, source: bool) -> str:
    _validate_row(row_plan)
    expected = Counter(r.marker for r in row_plan.replacements)
    if any(count != 1 for count in expected.values()):
        raise EntityMarkerError("Entity occurrence markers must be unique")
    if _marker_counts(text) != expected:
        raise EntityMarkerError("Entity marker counts differ from this source row")
    values = {r.marker: r.source if source else r.target for r in row_plan.replacements}
    return _MARKER.sub(lambda match: values[match[0]], text)


def restore_entity_source(row_plan: EntityRowPlan) -> str:
    return _restore(row_plan.marked_source, row_plan, source=True)


def restore_entity_target(marked_target: str, row_plan: EntityRowPlan) -> str:
    """Restore supplied names before target length/fidelity validation.

    Failure is a blocked target, not permission to append names or silently drop
    markers. Unmarked rows still reject a foreign marker from another row.
    """
    return _restore(marked_target, row_plan, source=False)
