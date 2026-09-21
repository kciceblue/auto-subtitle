"""Short-lived forced realignment of repaired source utterances.

Only the optional worker loads CUDA/the forced aligner. Original ASR metadata,
source strings and cue intervals are immutable. Failed rows retain original tokens
and are returned explicitly so callers can revert their provisional source edit.
Successful tokens use utterance-relative offsets, tagged accordingly; consumers
must not interpret these offsets as positions in the original raw ASR window.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Iterable

from src.asr_consensus import AudioWindow, aligned_tokens, compact
from src.config import TranscribeConfig
from src.ensemble_asr import ALIGNER
from src.translate import SrtBlock
from src.workflow_state import file_hash, fingerprint, read_json, write_json

logger = logging.getLogger(__name__)
VERSION = "source-realign-3"
# Provisional admission policy for the 0.6B aligner, not a measured memory claim.
# The isolated worker records actual peak allocations for subsequent calibration.
REQUIRED_GPU_GB = 3.0
HARD_FLOOR_GPU_GB = 2.0
EDGE_JITTER_SECONDS = .081
MAX_CLIP_SECONDS = 30.0
MAX_TOKEN_SECONDS = 4.0


@dataclass
class SourceRealignmentResult:
    metadata: dict
    successful_ids: tuple[int, ...]
    failed_ids: dict[int, str]
    report: dict
    output_path: Path | None = None


def _seconds(value: str) -> float:
    if not re.fullmatch(r"\d{2}:\d{2}:\d{2},\d{3}", value):
        raise ValueError("Invalid source timestamp")
    h, m, tail = value.split(":")
    second, ms = tail.split(",")
    if int(m) >= 60 or int(second) >= 60:
        raise ValueError("Invalid source timestamp fields")
    return int(h)*3600+int(m)*60+int(second)+int(ms)/1000


def _core(block: SrtBlock) -> tuple[float, float]:
    parts = block.ts_line.split(" --> ")
    if len(parts) != 2:
        raise ValueError("Invalid source cue interval")
    start, end = map(_seconds, parts)
    if end <= start:
        raise ValueError("Source cue interval is not positive")
    return start, end


def _coalesce_boundary_zeros(items: list[dict]) -> list[dict]:
    """Join at most two zero-bin characters at an existing positive boundary.

    No time is extended or invented. A distant, long or wholly collapsed run
    remains invalid. The original raw item evidence stays in the sidecar.
    """
    result = []
    index = 0
    while index < len(items):
        item = deepcopy(items[index])
        if item["end_time"] > item["start_time"]:
            item["alignment_group_items"] = [deepcopy(items[index])]
            result.append(item)
            index += 1
            continue
        end = index + 1
        while end < len(items) and items[end]["end_time"] == items[end]["start_time"]:
            end += 1
        run = items[index:end]
        at = item["start_time"]
        if (len(run) > 2 or sum(len(compact(r["text"])) for r in run) > 2
                or any(abs(r["start_time"]-at) > .001 for r in run)):
            raise ValueError("Excessive zero-duration token run remains unaligned")
        text = "".join(r["text"] for r in run)
        if result and abs(result[-1]["end_time"]-at) <= .001:
            result[-1]["text"] += text
            result[-1]["alignment_group_items"].extend(deepcopy(run))
        elif end < len(items) and abs(items[end]["start_time"]-at) <= .001:
            following = deepcopy(items[end])
            following["text"] = text + following["text"]
            following["alignment_group_items"] = deepcopy(run) + [deepcopy(items[end])]
            result.append(following)
            end += 1
        else:
            raise ValueError("Zero-duration token has no adjacent positive interval")
        index = end
    return result


def _validate_items(text: str, items: list[dict], *, source_id: int,
                    core_start: float, core_end: float,
                    clip_start: float, clip_end: float) -> list[dict]:
    """Conservative mechanical validation; never manufacture word durations."""
    bounds = [core_start, core_end, clip_start, clip_end]
    if (not all(math.isfinite(x) for x in bounds) or clip_start < 0
            or not clip_start <= core_start < core_end or core_end > clip_end+.001
            or clip_end-clip_start > MAX_CLIP_SECONDS+.001):
        raise ValueError("Invalid/oversized original-audio crop")
    if not isinstance(items, list) or not items or not compact(text):
        raise ValueError("Missing text or forced-alignment items")
    previous_end = -1.0
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ValueError("Malformed forced-alignment item")
        start, end = item.get("start_time"), item.get("end_time")
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end < start or start < previous_end-.001):
            raise ValueError("Nonfinite, collapsed, reversed or overlapping token time")
        if end-start > MAX_TOKEN_SECONDS:
            raise ValueError("Implausibly long aligned token; source timing needs review")
        previous_end = end
    items = _coalesce_boundary_zeros(items)
    # Own the entire padded crop here so no dropped edge word can be mistaken
    # for complete coverage. Core clipping is separately checked below.
    window = AudioWindow(source_id, clip_start, clip_end, clip_start, clip_end)
    tokens = aligned_tokens(text, items, window, require_complete=True)
    if len(tokens) != len(items):
        raise ValueError("Forced alignment lost a token at crop ownership boundary")
    if not tokens:
        raise ValueError("No forced-aligned source tokens")
    if tokens[-1]["finish"] < len(text):
        if compact(text[tokens[-1]["finish"]:]):
            raise ValueError("Unaligned source suffix")
        tokens[-1]["finish"] = len(text)  # Preserve literal trailing punctuation/space.
    cursor = 0
    for token, item in zip(tokens, items):
        token["alignment_group_items"] = item["alignment_group_items"]
        token["alignment_grain"] = "group" if len(item["alignment_group_items"]) > 1 else "word"
        if token["begin"] != cursor:
            raise ValueError("Noncontiguous aligned source character coverage")
        token["text"] = text[token["begin"]:token["finish"]]
        cursor = token["finish"]
        # Only a single 80ms quantization bin may be trimmed at the unchanged
        # cue edges. Larger displacement is an unresolved timing problem.
        if (token["start"] < core_start-EDGE_JITTER_SECONDS
                or token["end"] > core_end+EDGE_JITTER_SECONDS):
            raise ValueError("Token lies outside original cue beyond bounded edge jitter")
        token["start"] = max(core_start, token["start"])
        token["end"] = min(core_end, token["end"])
        if token["end"]-token["start"] < .01:
            raise ValueError("Clamping would collapse a token; no synthetic duration allowed")
        token.update(offset_scope="utterance", source_id=source_id,
                     source_hash=fingerprint(text), origin="source_realign")
    if cursor != len(text) or "".join(t["text"] for t in tokens) != text:
        raise ValueError("Forced tokens do not conserve exact repaired source")
    if any(a["end"] > b["start"]+.001 for a, b in zip(tokens, tokens[1:])):
        raise ValueError("Aligned core token times overlap")
    return tokens


def _launch_worker(spec: dict, timeout: float) -> dict:
    with tempfile.TemporaryDirectory(prefix="autosub-source-realign-") as directory:
        job_path = Path(directory)/"job.json"
        result_path = Path(directory)/"result.json"
        write_json(job_path, spec)
        process = subprocess.run([sys.executable, "-m", "src.source_realign", "--worker",
                                  str(job_path), str(result_path)],
                                 cwd=Path(__file__).resolve().parents[1], timeout=timeout,
                                 check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if process.returncode:
            raise RuntimeError(f"Forced-aligner worker exited {process.returncode}: {process.stderr[-1500:]}")
        result = read_json(result_path)
        if not isinstance(result, dict):
            raise RuntimeError("Forced-aligner worker returned no valid result")
        return result


def realign_changed(media: Path, source: list[SrtBlock], changed_ids: Iterable[int],
                    original_metadata: dict, config: TranscribeConfig | None = None,
                    output_path: Path | None = None, *, padding_seconds: float = .32,
                    worker_timeout: float = 180.0) -> SourceRealignmentResult:
    """Return copied token metadata and explicit failed IDs; never apply source edits.

    ``output_path``, when supplied, is a separate JSON sidecar containing the
    original metadata, validated replacement rows and failure/provenance ledger.
    It must not point to the raw ``.asr.json`` or to the source/media artifacts.
    The caller must revert source changes for every ID in ``failed_ids``.
    """
    started = time.monotonic()
    media = Path(media)
    if not media.is_file():
        raise FileNotFoundError(media)
    if not isinstance(original_metadata, dict):
        raise ValueError("Original ASR metadata must be an object")
    if not math.isfinite(padding_seconds) or not 0 <= padding_seconds <= 1:
        raise ValueError("Padding must be between zero and one second")
    if not math.isfinite(worker_timeout) or worker_timeout <= 0:
        raise ValueError("Worker timeout must be positive")
    if output_path is not None:
        output_path = Path(output_path)
        if (output_path.resolve() == media.resolve() or output_path.suffix != ".json"
                or output_path.name.endswith(".asr.json")):
            raise ValueError("Use a separate realignment JSON sidecar, never a raw artifact")
    ids = list(changed_ids)
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 1 for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Changed IDs must be distinct positive integers")
    blocks = {block.index: block for block in source}
    if len(blocks) != len(source):
        raise ValueError("Duplicate selected-source IDs")
    copied = deepcopy(original_metadata)
    groups = copied.get("utterance_tokens", [])
    segments = original_metadata.get("segments", [])
    positions = {block.index: i for i, block in enumerate(source)}
    failed, jobs = {}, []
    for source_id in ids:
        block = blocks.get(source_id)
        try:
            if block is None:
                raise ValueError("Changed source ID does not exist")
            index = positions[source_id]
            if not isinstance(groups, list) or len(groups) != len(source):
                raise ValueError("Original utterance-token row count mismatch")
            start, end = _core(block)
            if end-start > MAX_CLIP_SECONDS:
                raise ValueError("Changed utterance exceeds bounded aligner crop")
            if not compact(block.text):
                raise ValueError("Changed source is empty or punctuation-only")
            if segments:
                original = segments[index]
                if (original.get("index") != source_id
                        or abs(float(original["start"])-start) > .0015
                        or abs(float(original["end"])-end) > .0015):
                    raise ValueError("Source patch changed the original cue interval or ID")
            jobs.append({"id": source_id, "text": block.text, "source_hash": fingerprint(block.text),
                         "core_start": start, "core_end": end,
                         "original_tokens_hash": fingerprint(groups[index])})
        except (ValueError, TypeError, IndexError, KeyError) as exc:
            failed[source_id] = str(exc)
    cfg = config or TranscribeConfig()
    language = getattr(cfg, "language", None) or getattr(cfg, "source_lang", "Japanese")
    language = {"ja": "Japanese", "zh": "Chinese", "en": "English", "ko": "Korean",
                "auto": "Japanese"}.get(language, language)
    media_hash = file_hash(media)
    spec = {"version": VERSION, "media": str(media.resolve()), "media_hash": media_hash,
            "original_metadata_hash": fingerprint(original_metadata), "jobs": jobs,
            "padding_seconds": padding_seconds, "language": language,
            "warden_admin_url": getattr(cfg, "warden_admin_url", "http://127.0.0.1:8089/admin"),
            "unload_warden": getattr(cfg, "unload_warden_before_asr", True),
            "aligner_model": str(ALIGNER), "required_gpu_gb": REQUIRED_GPU_GB,
            "hard_floor_gpu_gb": HARD_FLOOR_GPU_GB}
    spec["request_key"] = fingerprint(spec)
    worker, successes, validated = {}, [], {}
    if jobs:
        try:
            worker = _launch_worker(spec, worker_timeout)
            if worker.get("request_key") != spec["request_key"] or not isinstance(worker.get("rows"), dict):
                raise ValueError("Stale/malformed forced-aligner worker response")
            if set(worker["rows"]) != {str(job["id"]) for job in jobs}:
                raise ValueError("Worker response must cover exact changed source IDs")
            for job in jobs:
                source_id = job["id"]
                try:
                    row = worker["rows"][str(source_id)]
                    if not isinstance(row, dict) or row.get("source_hash") != job["source_hash"]:
                        raise ValueError("Stale/malformed worker source text")
                    if row.get("error"):
                        raise ValueError(row["error"])
                    tokens = _validate_items(job["text"], row["items"], source_id=source_id,
                        core_start=job["core_start"], core_end=job["core_end"],
                        clip_start=row["clip_start"], clip_end=row["clip_end"])
                    copied["utterance_tokens"][positions[source_id]] = tokens
                    successes.append(source_id)
                    validated[str(source_id)] = {"tokens": tokens, "source_hash": job["source_hash"],
                        "core_start": job["core_start"], "core_end": job["core_end"],
                        "original_tokens_hash": job["original_tokens_hash"],
                        "status": "mechanically_aligned_source_still_unresolved"}
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    failed[source_id] = str(exc)
        except (RuntimeError, ValueError, TypeError, OSError, subprocess.TimeoutExpired) as exc:
            failed.update({job["id"]: str(exc) for job in jobs})
    try:
        media_unchanged = file_hash(media) == media_hash
    except OSError:
        media_unchanged = False
    if not media_unchanged:
        # A mid-run media change invalidates every acoustic result.
        copied = deepcopy(original_metadata)
        successes, validated = [], {}
        failed.update({source_id: "Original media changed during forced alignment" for source_id in ids})
    report = {"version": VERSION, "request_key": spec["request_key"],
        "media": str(media.resolve()), "media_hash": media_hash,
        "original_metadata_hash": spec["original_metadata_hash"], "original_metadata": deepcopy(original_metadata),
        "requested_ids": ids, "successful_ids": successes, "failed_ids": {str(k): v for k, v in failed.items()},
        "replacement_rows": validated, "worker": worker, "wall_seconds": time.monotonic()-started,
        "token_offset_scope": "Successful replacement offsets refer to the repaired utterance, not raw ASR windows",
        "source_status": "Unresolved; mechanical alignment is not acoustic or semantic approval",
        "headroom_policy": {"required_gb": REQUIRED_GPU_GB, "hard_floor_gb": HARD_FLOOR_GPU_GB,
                            "status": "Provisional, unmeasured; inspect worker peak memory"}}
    if output_path is not None:
        write_json(output_path, report)
    return SourceRealignmentResult(copied, tuple(successes), failed, report, output_path)


def _load_aligner(spec: dict):
    import torch
    from qwen_asr import Qwen3ForcedAligner
    from src.warden import ensure_gpu_headroom
    path = Path(spec["aligner_model"])
    if not path.is_dir():
        raise FileNotFoundError(f"Local forced-aligner model not found: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the isolated source forced-aligner worker")
    torch.set_num_threads(2)
    free = ensure_gpu_headroom(required_gb=spec["required_gpu_gb"], hard_floor_gb=spec["hard_floor_gpu_gb"],
        admin_url=spec["warden_admin_url"], enabled=spec["unload_warden"], caller="source forced realignment")
    torch.cuda.reset_peak_memory_stats()
    model = Qwen3ForcedAligner.from_pretrained(str(path), dtype=torch.bfloat16,
                                              device_map="cuda", local_files_only=True)
    return model, free


def _worker(spec: dict) -> dict:
    import torch
    from src.adjudicate import load_full_wav
    result = {"request_key": spec["request_key"], "rows": {}, "aligner_model": spec["aligner_model"]}
    started = time.monotonic()
    try:
        media = Path(spec["media"])
        if file_hash(media) != spec["media_hash"]:
            raise ValueError("Original media hash differs from alignment request")
        sr, audio = load_full_wav(media)
        duration = len(audio)/sr
        model, free = _load_aligner(spec)
        result["free_cuda_bytes_before_load"] = free
        for job in spec["jobs"]:
            row = {"source_hash": job["source_hash"]}
            result["rows"][str(job["id"])] = row
            before = time.monotonic()
            try:
                if job["core_end"] > duration+.001:
                    raise ValueError("Original cue interval extends beyond available media")
                padding = min(spec["padding_seconds"], max(0.0, (MAX_CLIP_SECONDS-(job["core_end"]-job["core_start"]))/2))
                left = max(0, int((job["core_start"]-padding)*sr))
                right = min(len(audio), math.ceil((job["core_end"]+padding)*sr))
                if right <= left or (right-left)/sr > MAX_CLIP_SECONDS:
                    raise ValueError("Empty or oversized original-audio crop")
                clip = audio[left:right]
                row.update(clip_start=left/sr, clip_end=right/sr)
                with torch.inference_mode():
                    aligned = model.align(audio=[(clip, sr)], text=[job["text"]], language=spec["language"])
                if len(aligned) != 1:
                    raise ValueError("Aligner returned the wrong number of utterances")
                items = [asdict(item) for item in aligned[0].items]
                row["items"] = items  # Keep rejected raw timings for diagnosis; never apply them.
                # Validate inside the isolated worker too, then independently in
                # the parent before merging any returned token row.
                _validate_items(job["text"], items, source_id=job["id"],
                    core_start=job["core_start"], core_end=job["core_end"],
                    clip_start=left/sr, clip_end=right/sr)
                row["items"] = items
            except Exception as exc:
                row["error"] = str(exc)
            row["seconds"] = time.monotonic()-before
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    except Exception as exc:
        result["error"] = str(exc)
        for job in spec["jobs"]:
            result["rows"][str(job["id"])] = {"source_hash": job["source_hash"], "error": str(exc)}
    result["seconds"] = time.monotonic()-started
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("Usage: python -m src.source_realign --worker job.json result.json")
    write_json(Path(sys.argv[3]), _worker(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))))
