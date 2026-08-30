"""Content-aware context discovery for auto-subtitle.

Replaces the name-based `_discover_readmes` filter with a content
intelligence layer:

  1. Candidate collection (cheap heuristics): text-ish files near the media
     file's directory chain — any filename, not just README*.
  2. LLM classification (one request per directory): each candidate is
     tagged ``synopsis`` (introduction/background), ``script`` (actual
     dialogue/台词台本), or ``skip`` (unrelated).
  3. Caching (.context_scan_cache.json, keyed by dir + file signature) and
     degradation to the legacy README* filter when the LLM is unavailable.

The ``kind`` flows downstream: ``script`` files get script-aware sampling
in translation context and penetrate into the review pass as ground-truth
anchors (src/script_align.py, src/review.py).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Candidate collection -------------------------------------------------------

# Extensions considered for context candidates (PDF handled by pypdf).
_TEXT_SUFFIXES = {".txt", ".md", ".pdf"}
# Files we must never treat as context (pipeline artifacts / project files).
_EXCLUDE_PATTERNS = (
    re.compile(r"^vocab", re.IGNORECASE),
    re.compile(r"^\.hotword_cache\.json$"),
    re.compile(r"^\.context_scan_cache\.json$"),
    re.compile(r"^REVIEW-", re.IGNORECASE),
    re.compile(r"^(CLAUDE|AGENTS)\.md$"),
    re.compile(r"^(requirements|pyproject|package|setup|Pipfile|poetry)\.", re.IGNORECASE),
)
_EXCLUDE_SUFFIXES = {
    ".srt", ".ass", ".vtt", ".json", ".py", ".sh",
    ".toml", ".lock", ".cfg", ".yaml", ".yml", ".ini", ".log",
}
_MAX_CANDIDATES = 12
_MAX_BYTES = 5 * 1024 * 1024  # scripts/synopses are never bigger

# Sampling windows for the classifier (keep token cost flat).
_SAMPLE_HEAD = 2000
_SAMPLE_MID = 400
_SAMPLE_TAIL = 400


def is_project_root_file(path: Path) -> bool:
    """True for files sitting in the pipeline's own repo root.

    The upward walk from input/<work>/ reaches the project root, where the
    pipeline's own README.md/docs live — those are documentation about THIS
    tool, never work context, and the legacy README* fallback would
    otherwise inject them into translation prompts.
    """
    d = path.parent
    return (d / "run.sh").is_file() and (d / "src" / "config.py").is_file()


def _is_excluded(path: Path) -> bool:
    if path.suffix.lower() in _EXCLUDE_SUFFIXES:
        return True
    if is_project_root_file(path):
        return True
    for pat in _EXCLUDE_PATTERNS:
        if pat.search(path.name):
            return True
    return False


def _collect_candidates(base_dir: Path, max_files: int = _MAX_CANDIDATES) -> list[Path]:
    """Text-ish files near base_dir (up 3 levels, nearest first), deduped.

    Heuristic only — it *collects*, never decides. The classifier decides.
    """
    found: list[Path] = []
    d = base_dir
    for _ in range(3):
        for p in sorted(d.iterdir()):
            if not p.is_file() or p.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            if _is_excluded(p):
                continue
            try:
                if p.stat().st_size > _MAX_BYTES:
                    continue
            except OSError:
                continue
            found.append(p)
        if d.parent == d:
            break
        d = d.parent
    seen: set[Path] = set()
    unique = [p for p in found if not (p in seen or seen.add(p))]
    return unique[:max_files]


def sample_text(path: Path) -> str:
    """Sample a context file for classification: head + middle + tail."""
    from src.translate import read_context_file

    text = read_context_file(path)
    if not text:
        return ""
    if len(text) <= _SAMPLE_HEAD + _SAMPLE_TAIL:
        return text
    head = text[:_SAMPLE_HEAD]
    mid_start = (len(text) - _SAMPLE_MID) // 2
    mid = text[mid_start:mid_start + _SAMPLE_MID]
    tail = text[-_SAMPLE_TAIL:]
    return f"{head}\n\n...[中略]...\n\n{mid}\n\n...[后略]...\n\n{tail}"


# Classification -------------------------------------------------------------

KINDS = ("synopsis", "script", "skip")

SYSTEM_PROMPT = """\
You classify auxiliary text files found next to a media file (audio/video) \
for a subtitle pipeline. Classify each file into exactly one kind:

- synopsis: introduction / synopsis / world-building / character descriptions \
/ glossary of the work. Background material that helps understand terms, \
names and tone.
- script: the actual dialogue script of the work (台词台本原文), typically \
short lines of spoken lines, possibly with speaker name prefixes and stage \
directions.
- skip: anything else (invoices, logs, notes, code, unrelated documents).

Rules:
- Output ONLY a JSON array. No markdown fences, no commentary.
- One object per input file, in the same order.
- confidence: a number 0-1.
- reason: short (under 30 chars)."""

USER_TEMPLATE = """\
Classify these files found near the media (source language: {source_lang}):
{files}

Return exactly:
[{{"file": "<name>", "kind": "synopsis|script|skip", "confidence": 0.9, "reason": "..."}}]"""


@dataclass
class ScannedContext:
    """A context candidate that survived classification."""

    path: Path
    kind: str  # synopsis | script
    confidence: float = 1.0
    reason: str = ""


def _strip_json_fence(content: str) -> str:
    content = content.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", content, re.S)
    if m:
        return m.group(1).strip()
    # Tolerate stray prose around the array.
    start, end = content.find("["), content.rfind("]")
    if start != -1 and end != -1 and end > start:
        return content[start:end + 1]
    return content


def _parse_classification(content: str) -> list[dict] | None:
    try:
        data = json.loads(_strip_json_fence(content))
    except json.JSONDecodeError as e:
        logger.warning("Context classification JSON parse failed: %s", e)
        return None
    if not isinstance(data, list):
        logger.warning("Context classification: expected JSON array, got %r", type(data).__name__)
        return None
    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict) or not item.get("file"):
            continue
        kind = str(item.get("kind", "")).strip().lower()
        if kind not in KINDS:
            kind = "synopsis"  # conservative: unknown → keep
        conf = float(item.get("confidence", 1.0) or 1.0)
        if kind == "skip" and conf < 0.7:
            # 宁多勿漏: low-confidence skip is kept as synopsis
            kind, conf = "synopsis", conf
        out.append({"file": item["file"], "kind": kind,
                    "confidence": conf, "reason": str(item.get("reason", ""))[:40]})
    return out


def _classify(
    candidates: list[Path],
    endpoint: str,
    extra_payload: dict | None,
    source_lang: str,
    timeout: int = 120,
) -> list[dict] | None:
    """One batched LLM request classifying all candidates. None on failure."""
    from src.translate import _THINKING_OFF_VARIANTS

    payload: dict = {
        "model": "qwen3.8-27b-dflash",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                source_lang=source_lang,
                files=json.dumps(
                    [{"file": p.name, "size_bytes": p.stat().st_size,
                      "sample": sample_text(p)[:1500]}
                     for p in candidates],
                    ensure_ascii=False, indent=1),
            )},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
        "stream": False,
    }
    if extra_payload:
        payload.update(extra_payload)

    import requests

    for thinking_off in _THINKING_OFF_VARIANTS:
        attempt = dict(payload)
        for key, value in thinking_off.items():
            attempt.setdefault(key, value)
        try:
            resp = requests.post(endpoint, json=attempt, timeout=timeout)
            resp.raise_for_status()
            content = (resp.json()["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            logger.warning("Context classification LLM call failed (%s): %s", endpoint, e)
            return None
        if content:
            parsed = _parse_classification(content)
            if parsed is not None:
                return parsed
            # parse failed → retry with the next thinking-off variant
            continue
        logger.warning("Context classification returned empty content; retrying "
                       "with thinking disabled another way")
    logger.warning("Context classification failed after %d attempt(s)",
                   len(_THINKING_OFF_VARIANTS))
    return None


# Cache ----------------------------------------------------------------------

def _file_sig(path: Path) -> str:
    st = path.stat()
    return f"{path.name}:{st.st_size}:{st.st_mtime_ns}"


_CACHE_SCHEMA = 1


def _load_cache(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != _CACHE_SCHEMA:
            return {}
        return data.get("entries", {})
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(path: Path, entries: dict) -> None:
    try:
        path.write_text(
            json.dumps({"schema": _CACHE_SCHEMA, "entries": entries},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("Failed to write context scan cache %s: %s", path, e)


def classify_explicit(
    files: list[Path],
    endpoint: str | None,
    extra_payload: dict | None = None,
    source_lang: str = "Japanese",
) -> dict[str, ScannedContext]:
    """Classify explicitly-passed --context/--context-file paths.

    Returns {path-string: ScannedContext}. On LLM failure every file is
    conservatively tagged synopsis (explicit files keep working).
    """
    files = [Path(p) for p in files if Path(p).is_file()]
    if not files:
        return {}
    if not endpoint:
        return {str(p): ScannedContext(p, "synopsis") for p in files}
    classified = _classify(files, endpoint, extra_payload, source_lang)
    if classified is None:
        logger.warning("Classification unavailable for explicit context files "
                       "— treating all as synopsis")
        return {str(p): ScannedContext(p, "synopsis") for p in files}
    by_name = {item["file"]: item for item in classified}
    out: dict[str, ScannedContext] = {}
    for p in files:
        item = by_name.get(p.name, {})
        kind = item.get("kind", "synopsis")
        if kind == "skip":
            kind = "synopsis"  # explicit files are never dropped
        out[str(p)] = ScannedContext(
            path=p, kind=kind,
            confidence=item.get("confidence", 1.0),
            reason=item.get("reason", ""))
    return out


# Public API -----------------------------------------------------------------

def _fallback_readmes(base_dir: Path) -> list[ScannedContext]:
    """Legacy name-based filter (README*.txt/.md/.pdf) — degradation path."""
    from main import _discover_readmes

    return [ScannedContext(path=p, kind="synopsis") for p in _discover_readmes(base_dir)]


def scan_context_files(
    base_dir: Path,
    endpoint: str | None = None,
    extra_payload: dict | None = None,
    source_lang: str = "Japanese",
    cache_file: Path = Path(".context_scan_cache.json"),
    use_llm: bool = True,
) -> list[ScannedContext]:
    """Content-aware context discovery for one media directory.

    Returns ScannedContext entries (kind ∈ {synopsis, script}). On LLM
    failure degrades to the legacy README* filter so context is never lost.
    """
    candidates = _collect_candidates(base_dir)
    if not candidates:
        return []

    if not use_llm or not endpoint:
        logger.info("Context scan disabled/failed → legacy README filter for %s", base_dir)
        return _fallback_readmes(base_dir)

    # Cache lookup (per-file signature; partial hits classify only the new ones)
    cache = _load_cache(cache_file)
    uncached: list[Path] = []
    cached: list[tuple[Path, dict]] = []
    for p in candidates:
        sig = _file_sig(p)
        entry = cache.get(f"{base_dir}:{sig}")
        if entry is not None:
            cached.append((p, entry))
        else:
            uncached.append(p)

    results: dict[str, ScannedContext] = {}
    # Keep the candidate's own path: candidates come from up to 3 directory
    # levels, so rebuilding as candidates[0].parent/name pointed cache hits
    # from parent dirs at nonexistent paths (context then silently dropped).
    for p, entry in cached:
        results[p.name] = ScannedContext(
            path=p,
            kind=entry["kind"], confidence=entry.get("confidence", 1.0),
            reason=entry.get("reason", "cache"))

    if uncached:
        classified = _classify(uncached, endpoint, extra_payload, source_lang)
        if classified is None:
            logger.warning("Context classification unavailable for %s — "
                           "falling back to README* filter", base_dir)
            return _fallback_readmes(base_dir) + list(results.values())
        for p, item in zip(uncached, classified):
            entry = {"kind": item["kind"], "confidence": item["confidence"],
                     "reason": item["reason"]}
            cache[f"{base_dir}:{_file_sig(p)}"] = entry
            results[item["file"]] = ScannedContext(
                path=p, kind=item["kind"], confidence=item["confidence"],
                reason=item["reason"])
        _save_cache(cache_file, cache)

    # Preserve candidate order, drop skip.
    out: list[ScannedContext] = []
    seen_names: set[str] = set()
    for p in candidates:
        sc = results.get(p.name)
        if sc is None or sc.kind == "skip":
            continue
        if p.name in seen_names:
            continue
        seen_names.add(p.name)
        out.append(sc)

    kept = [sc for sc in out if sc.kind != "skip"]
    if kept:
        logger.info("Context scan [%s]: %s", base_dir,
                    ", ".join(f"{sc.path.name}→{sc.kind}" for sc in kept))
    return out
