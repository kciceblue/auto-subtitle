"""GPU-grouped, evidence-first workflow. Legacy pipeline remains available.

All LLM context preparation precedes the ASR workers. The parent never loads a
GPU model. Every stage records input fingerprints and output hashes for resume.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import MEDIA_EXTENSIONS, TranslateConfig, TranscribeConfig
from src.evidence import build_evidence
from src.quality import (VERSION, draft_translation_settings, resolve_source, selective_qa, translate_scenes,
                         validate_blocks, write_quality_result)
from src.translate import SrtBlock, parse_srt, write_translated_srt
from src.workflow_state import StageState, artifact_path, file_hash, fingerprint, read_json, write_json

logger = logging.getLogger(__name__)


@dataclass
class Job:
    media: Path
    source: Path
    unit: str
    config: TranslateConfig
    media_hash: str = ""
    hotwords: list[str] | None = None
    utterance_mode: bool = False
    successful: bool = False
    empty: bool = False

    @property
    def asr_source(self) -> Path:
        return artifact_path(self.source, ".utterances.srt") if self.utterance_mode else self.source

    @property
    def state_path(self) -> Path:
        return artifact_path(self.source, ".workflow.json")


def asr_outputs(job: Job, *, nemo: bool = False) -> list[Path]:
    """Match the worker's complete nonempty artifact set on run and resume."""
    outputs = [job.asr_source, artifact_path(job.source, ".asr.json")]
    if nemo:
        outputs.append(artifact_path(job.asr_source, ".nemo-prepass.json"))
    return outputs


def collect_jobs(args) -> list[Job]:
    """Resolve paths without basename searches; combine pending input and output."""
    if args.input_file:
        media = [args.input_file]
    else:
        media = sorted(p for p in Path("input").rglob("*") if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS)
        media += sorted(p for p in Path("output").rglob("*") if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS)
    if not media:
        raise ValueError("No media found in input/ or output/")
    if len(media) > 1 and any((args.output, args.translate_output, args.adjudication, args.reference_srt)):
        raise ValueError("Custom output/adjudication/reference paths require a single input file")
    import json
    extra = json.loads(args.extra_payload) if args.extra_payload else {"model": "qwen3.8-27b-dflash", "temperature": 0.3}
    jobs, seen = [], {}
    for path in media:
        if not path.is_file():
            raise FileNotFoundError(path)
        absolute = path.resolve()
        rel = None
        for root in (Path("input"), Path("output")):
            try:
                rel = absolute.relative_to(root.resolve())
                source = (Path("output") / rel).with_suffix(".srt") if root.name == "input" else path.with_suffix(".srt")
                unit = rel.parts[0] if len(rel.parts) > 1 else path.stem
                organized = Path("output") / unit / "final"
                if root.name == "input" and organized.is_dir():
                    within = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(path.name)
                    source = (organized / within).with_suffix(".srt")
                break
            except ValueError:
                continue
        if rel is None:
            source, unit = Path("output") / (path.stem + ".srt"), path.stem
        if args.output:
            source = args.output
        # Input takes precedence over an identical loose output media path.
        identity = str(source.resolve())
        if identity in seen:
            if file_hash(seen[identity]) != file_hash(path):
                raise ValueError(f"Different media map to {source}: {seen[identity]} and {path}; archive or rename the old work first")
            continue
        seen[identity] = path
        cfg = TranslateConfig(
            output_srt=args.translate_output, endpoint=args.endpoint,
            source_lang=args.source_lang, target_lang=args.target_lang,
            chunk_size=args.chunk_size, timeout=args.timeout, retries=args.retries,
            max_tokens=args.max_tokens, extra_payload=extra, vocab_file=args.vocab,
            translation_reasoning_budget=args.translation_reasoning_budget,
            translation_episode_context=args.translation_episode_context,
            coherence_polish=getattr(args, "coherence_polish", False),
            coherence_recipe=getattr(args, "coherence_recipe", None),
            structured_translation=getattr(args, "structured_translation", False),
            script_file=args.script, reference_srt=args.reference_srt,
            adjudication=args.adjudication, review_chunk_size=args.review_chunk_size,
            scene_max_chars=args.scene_max_chars, structured_qa=args.structured_qa, delta_verification=args.delta_verification, entity_placeholders=args.entity_placeholders, source_span_repair=args.source_span_repair, pause_layout=args.pause_layout, telemetry_path=artifact_path(source, ".metrics.jsonl"),
        )
        utterance_mode = args.asr_strategy == "ensemble" and args.arbitrate and args.adjudication is None
        cfg.pack_scenes = utterance_mode
        jobs.append(Job(path, source, unit, cfg, utterance_mode=utterance_mode))
    return jobs


def prepare_context(jobs: list[Job], args) -> None:
    from src.context_scan import classify_explicit, scan_context_files
    from src.hotwords import generate_hotwords, filter_asr_hotwords
    from src.title_context import extract_title, clean_release_title
    from src.work_context import prepare_work_context

    deterministic_hotwords = (generate_hotwords(vocab_file=args.vocab, hotwords_file=args.hotwords_file,
                                          source_lang=args.source_lang, mode=args.hotword_mode)
                         if args.hotword_mode != "model" else None)
    groups = defaultdict(list)
    scan_cache = {}
    for job in jobs:
        cfg = job.config
        explicit = args.context_file or args.context
        if explicit:
            key = tuple(explicit)
            if key not in scan_cache:
                scan_cache[key] = list(classify_explicit(explicit, cfg.endpoint, cfg.extra_payload, cfg.source_lang).values())
            scanned = scan_cache[key]
        elif args.no_auto_context:
            scanned = []
        else:
            scanned = []
            for directory in dict.fromkeys([job.media.parent, artifact_path(job.source, ".context.json").parent]):
                if not directory.is_dir():
                    continue  # New output directories do not exist during preflight.
                if directory not in scan_cache:
                    scan_cache[directory] = scan_context_files(directory, cfg.endpoint, cfg.extra_payload,
                                                               cfg.source_lang, use_llm=not args.no_context_scan)
                scanned.extend(scan_cache[directory])
        cfg.context_files = list(dict.fromkeys(item.path for item in scanned))
        cfg.context_kinds = {str(item.path): item.kind for item in scanned}
        cfg.title = clean_release_title(extract_title(job.media))
        if cfg.script_file is None:
            cfg.script_file = next((item.path for item in scanned if item.kind == "script"), None)
        if cfg.script_file and cfg.script_file not in cfg.context_files:
            cfg.context_files.append(cfg.script_file)
            cfg.context_kinds[str(cfg.script_file)] = "script"
        groups[job.unit].append(job)
    for group in groups.values():
        first = group[0]
        paths = list(dict.fromkeys(p for job in group for p in job.config.context_files))
        titles = [job.config.title for job in group if job.config.title]
        context_cache = artifact_path(first.source, ".work-context.json")
        if args.force:
            context_cache.unlink(missing_ok=True)
        context = prepare_work_context(paths, titles, first.config, context_cache)
        hotword_result = deterministic_hotwords or (generate_hotwords(
            context_files=paths, vocab_file=args.vocab or args.hotwords_file,
            source_lang=args.source_lang, endpoint=args.hotword_endpoint or args.endpoint,
            extra_payload=first.config.extra_payload,
            cache_file=artifact_path(first.source, ".hotword-cache.json"), title="；".join(titles) or None,
            force=args.force,
        ) if paths or titles or args.vocab or args.hotwords_file else None)
        for job in group:
            job.config.context_summary = context
            job.hotwords = (list(hotword_result.hotwords) if deterministic_hotwords is not None else
                            filter_asr_hotwords(hotword_result.hotwords if hotword_result else [],
                                                args.source_lang, args.vocab or args.hotwords_file))


def run_worker(module: str, spec: dict, directory: Path, name: str) -> int:
    manifest = directory / name
    write_json(manifest, spec)
    command = [sys.executable, "-m", module]
    if module == "src.adjudicate":
        command.append("--batch")
    result = subprocess.run(command + [str(manifest)], check=False)
    return result.returncode


def configuration_key(config: TranslateConfig) -> dict:
    return {"version": VERSION, "model": config.extra_payload, "endpoint": config.endpoint,
            "source_lang": config.source_lang, "target_lang": config.target_lang,
            "context": config.context_summary, "chunk_size": config.chunk_size,
            "scene_max_chars": config.scene_max_chars, "review_chunk_size": config.review_chunk_size,
            "pack_scenes": config.pack_scenes, "structured_qa": config.structured_qa,
            "delta_verification": config.delta_verification,
            "entity_placeholders": config.entity_placeholders,
            "source_span_repair": config.source_span_repair,
            "pause_layout": config.pause_layout,
            "pause_layout_code": file_hash(Path(__file__).with_name("pause_layout.py")) if config.pause_layout else None,
            "source_repair_code": [file_hash(Path(__file__).with_name(name)) for name in ("source_repair.py", "source_spans.py", "source_realign.py")] if config.source_span_repair else None,
            "entities_code": file_hash(Path(__file__).with_name("entities.py")) if config.entity_placeholders or config.source_span_repair else None,
            "max_tokens": config.max_tokens,
            "vocab": file_hash(config.vocab_file) if config.vocab_file else None,
            "script": file_hash(config.script_file) if config.script_file else None,
            "reference": file_hash(config.reference_srt) if config.reference_srt else None}


def process_language(job: Job, args) -> None:
    cfg = job.config
    state = StageState(job.state_path)
    translated = cfg.resolve_translate_output(job.source)
    semantic_target = artifact_path(job.source, f".utterances.{cfg.target_lang_code}.srt") if job.utterance_mode else translated
    protected = [translated, job.source, semantic_target] if job.utterance_mode else [translated]
    for path in protected:
        previous_hash = state.data.get("quality", {}).get("outputs", {}).get(path.name)
        if previous_hash and path.exists() and file_hash(path) != previous_hash and not args.force:
            raise RuntimeError(f"Subtitles were edited after QA: {path}; preserve those edits or use --force explicitly")
    source = parse_srt(job.asr_source)
    validate_blocks(source)
    evidence = build_evidence(source, cfg)
    common = configuration_key(cfg)
    source_key = fingerprint([common, file_hash(job.asr_source), [asdict(e) for e in evidence]])
    state = StageState(job.state_path)
    cache = artifact_path(job.source, ".requests.json")
    decisions_path = artifact_path(job.source, ".source-decisions.json")
    base_decisions_path = artifact_path(job.source, ".source-resolution.json") if cfg.source_span_repair else decisions_path
    display_metadata = read_json(artifact_path(job.source, ".asr.json"), {})
    started = time.monotonic()
    if not args.force and state.matches("resolution", source_key, [base_decisions_path]):
        data = read_json(base_decisions_path)
        resolved = [SrtBlock(**row) for row in data["source"]]
        decisions = data["decisions"]
    else:
        resolved, decisions = resolve_source(source, evidence, cfg, cache)
        write_json(base_decisions_path, {"source": [asdict(b) for b in resolved], "decisions": decisions})
        state.save("resolution", source_key, [base_decisions_path],
                   status="complete" if all(d["status"] != "unchecked" for d in decisions) else "failed",
                   seconds=time.monotonic() - started)
    source_stage_complete = True
    if cfg.source_span_repair:
        from src.source_repair import repair_source_spans
        from src.entities import extract_entity_lexicon
        from src.translate import read_context_file
        span_started = time.monotonic()
        aligned_path = artifact_path(job.source, ".source-aligned.json")
        span_report = artifact_path(job.source, ".source-spans.json")
        original_context = "\n\n".join(read_context_file(p) for p in cfg.context_files)[:12000]
        span_key = fingerprint([common, file_hash(base_decisions_path), display_metadata,
                                original_context, file_hash(job.media)])
        span_outputs = [decisions_path, aligned_path, span_report]
        if not args.force and state.matches("source_spans", span_key, span_outputs):
            data = read_json(decisions_path)
            resolved = [SrtBlock(**row) for row in data["source"]]
            decisions = data["decisions"]
            display_metadata = read_json(aligned_path)
            source_stage_complete = read_json(span_report, {}).get("complete") is True
        else:
            lexicon = extract_entity_lexicon(original_context, cfg, artifact_path(job.source, ".entity-lexicon.json"))
            if lexicon.degraded:
                for decision in decisions:
                    decision.update(status="unchecked", reason=decision["reason"] + "; source-span identity exclusions could not be grounded")
                report = {"complete": False, "error": "Grounded entity exclusions unavailable", "issues": lexicon.issues}
                complete_spans = False
            else:
                locks = tuple(sorted({surface for entity in lexicon.entities for surface in entity.surfaces()}))
                tc = TranscribeConfig(warden_admin_url=args.warden_admin,
                                      unload_warden_before_asr=not args.no_warden_unload)
                result = repair_source_spans(job.media, resolved, decisions, display_metadata, cfg,
                    artifact_path(job.source, ".source-span-requests.json"),
                    artifact_path(job.source, ".source-realignment.json"),
                    protected_surfaces=locks, transcribe_config=tc)
                resolved, decisions, display_metadata = result.source, result.decisions, result.metadata
                report, complete_spans = result.report, result.complete
            source_stage_complete = complete_spans
            write_json(decisions_path, {"source": [asdict(b) for b in resolved], "decisions": decisions})
            write_json(aligned_path, display_metadata)
            write_json(span_report, report)
            state.save("source_spans", span_key, span_outputs,
                       status="complete" if complete_spans else "failed", seconds=time.monotonic()-span_started)
    cfg.line_notes = {d["line"]: [f"源文决定[{d['status']}] {d['candidate']}: {d['text']}；依据: {d['reason']}"]
                      for d in decisions}
    started = time.monotonic()
    entity_plan, entity_render = None, {}
    entity_plan_path = artifact_path(job.source, ".entity-plan.json")
    entity_render_path = artifact_path(job.source, ".entity-render.json")
    if cfg.entity_placeholders:
        from src.entities import extract_entity_lexicon, classify_entity_occurrences, build_entity_plan
        from src.translate import read_context_file
        # Read original supplied documents, never the model-compressed summary.
        original_context = "\n\n".join(read_context_file(p) for p in cfg.context_files)[:12000]
        lexicon = extract_entity_lexicon(original_context, cfg, artifact_path(job.source, ".entity-lexicon.json"))
        labels = classify_entity_occurrences(resolved, lexicon, cfg, artifact_path(job.source, ".entity-classify.json"))
        entity_plan = build_entity_plan(resolved, labels, lexicon, display_metadata)
        write_json(entity_plan_path, {"plan": entity_plan.to_dict(), "lexicon": lexicon.to_dict(), "classifications": labels.to_dict()})
        cfg.entity_notes = {ln: [f"Provisional local name hint: {r.source} → {r.target}. Identity remains unresolved; preserve the source grammatical role and do not infer a speaker." for r in row.replacements]
                            for ln, row in entity_plan.rows.items() if row.replacements}
    draft_settings = draft_translation_settings(cfg)
    translation_key = fingerprint([common, file_hash(decisions_path), entity_plan.key if entity_plan else None, draft_settings])
    draft_path = artifact_path(job.source, ".translation.json")
    translation_outputs = [draft_path] + ([entity_render_path] if entity_plan else [])
    if not args.force and state.matches("translation", translation_key, translation_outputs):
        draft = [SrtBlock(**row) for row in read_json(draft_path)]
        entity_render = read_json(entity_render_path, {}) if entity_plan else {}
    else:
        draft = translate_scenes(resolved, evidence, cfg, cache, entity_plan=entity_plan, entity_render=entity_render)
        write_json(draft_path, [asdict(b) for b in draft])
        if entity_plan:
            write_json(entity_render_path, entity_render)
        state.save("translation", translation_key, translation_outputs, seconds=time.monotonic() - started,
                   request_settings=draft_settings)
    translated = cfg.resolve_translate_output(job.source)
    report = artifact_path(job.source, f".{cfg.target_lang_code}.md")
    report = report.with_name("REVIEW-" + report.name)
    ledger_path = artifact_path(job.source, f".{cfg.target_lang_code}.quality.json")
    display_inputs = None
    if job.utterance_mode:
        # Timing depends on word alignment as well as semantic text. Hash the
        # display implementation too so changes to grouping/hold rules cannot
        # leave stale final cues behind an otherwise matching quality record.
        display_inputs = [fingerprint(display_metadata),
                          file_hash(Path(__file__).with_name("display.py"))]
    coherence_inputs = [cfg.coherence_polish, file_hash(Path(__file__).with_name("coherence.py")) if cfg.coherence_polish else None]
    if cfg.coherence_recipe is not None:
        from src.coherence_workflow import recipe_binding
        coherence_inputs.append(recipe_binding(cfg.coherence_recipe, cfg, display_metadata))
    quality_key = fingerprint([common, translation_key, file_hash(draft_path), file_hash(decisions_path), coherence_inputs,
                               args.proofread, args.review, display_inputs,
                               entity_plan.key if entity_plan else None, entity_render])
    quality_outputs = [translated, report, ledger_path]
    coherence_path = artifact_path(job.source, ".coherence.json")
    if cfg.coherence_polish or cfg.coherence_recipe is not None:
        quality_outputs.append(coherence_path)
    if job.utterance_mode:
        quality_outputs += [job.source, semantic_target, artifact_path(job.source, ".display.json")]
    if not args.force and state.matches("quality", quality_key, quality_outputs):
        job.successful = source_stage_complete
        logger.info("Resume: quality complete for %s", job.media)
        return
    started = time.monotonic()
    if args.proofread or args.review:
        final, ledger = selective_qa(resolved, draft, evidence, decisions, cfg, cache)
    else:
        final = draft
        ledger = [{"line": e.line, "timestamp": source[e.line - 1].ts_line,
                   "status": "unchecked", "reason": "QA disabled by user",
                   "evidence": asdict(e), "source_decision": decisions[e.line - 1]} for e in evidence]
    if entity_plan is not None:
        for row in ledger:
            ln = row["line"]
            hints = [record for record in entity_plan.ledger if record["occurrence"]["source_id"] == ln]
            row["entity_evidence"] = hints
            row["entity_render"] = entity_render.get(str(ln), "unchecked")
            protected_name = bool(entity_plan.rows[ln].replacements)
            failed_identity = lexicon.degraded or any(
                not item.get("classification", {}).get("checked", False) for item in hints)
            unknown_identity = any(item.get("classification", {}).get("decision") == "UNKNOWN" for item in hints)
            if failed_identity:
                row["status"] = "unchecked"
                row["reason"] += "; local entity extraction/classification incomplete"
            if unknown_identity and row["status"] != "unchecked":
                row["status"] = "unresolved"
                row["reason"] += "; local entity identity unknown"
            if protected_name or row["entity_render"] in {"fallback_identity_unchecked", "translation_missing", "unchecked"}:
                if row["status"] != "unchecked":
                    row["status"] = "unresolved"
                row["reason"] += "; local name identity requires independent validation"
    if cfg.coherence_polish or cfg.coherence_recipe is not None:
        if cfg.coherence_recipe is not None:
            from src.coherence_workflow import refine_with_recipe
            final, coherence_ledger = refine_with_recipe(resolved, final, display_metadata, cfg,
                artifact_path(job.source, ".coherence-requests.json"),
                expected_binding=coherence_inputs[-1])
        else:
            from src.coherence import polish_coherence
            final, coherence_ledger = polish_coherence(resolved, final, cfg,
                artifact_path(job.source, ".coherence-requests.json"))
        write_json(coherence_path, coherence_ledger)
        for row, edit in zip(ledger, coherence_ledger):
            row["coherence_edit"] = edit
            if edit["before"] != edit["after"]:
                if row["status"] != "unchecked":
                    row["status"] = "unresolved"
                row["reason"] += "; local coherence edit requires independent evaluation"
    # Keep the semantic QA ledger tied to immutable utterance IDs. Display IDs
    # have an explicit mapping and never feed back as new recognition evidence.
    if not semantic_target.exists():
        write_translated_srt(draft, semantic_target)
    complete = write_quality_result(semantic_target, final, ledger, report, ledger_path)
    if job.utterance_mode:
        from src.display import build_display
        from src.translate import make_snapshot
        display_source, display_target, mapping = build_display(
            resolved, final, display_metadata, cfg,
            artifact_path(job.source, ".display-requests.json"))
        for destination in (job.source, translated):
            if destination.exists():
                make_snapshot(destination, "pre-review",
                              directory=artifact_path(destination, ".pre-review.srt").parent)
        write_translated_srt(display_source, job.source)
        write_translated_srt(display_target, translated)
        write_json(artifact_path(job.source, ".display.json"), {"version": VERSION, "cues": mapping})
    enabled = args.proofread or args.review
    job.successful = source_stage_complete and (complete or not enabled) and all(d["status"] != "unchecked" for d in decisions)
    state.save("quality", quality_key, quality_outputs,
               status="complete" if job.successful else "failed", seconds=time.monotonic() - started,
               qa_enabled=bool(enabled))


def run_workflow(args) -> int:
    if getattr(args, "coherence_recipe", None) is not None:
        from src.coherence_workflow import load_recipe
        if args.coherence_polish:
            raise ValueError("Choose one of --coherence-polish and --coherence-recipe")
        load_recipe(args.coherence_recipe)  # Fail before ASR or any model switch.
    if args.chunk_size <= 0 or args.scene_max_chars <= 0 or args.review_chunk_size <= 0 or args.asr_batch_size <= 0:
        raise ValueError("Chunk and batch sizes must be positive")
    for path in [args.script, args.vocab, args.hotwords_file, args.reference_srt, args.adjudication,
                 *(args.context or []), *(args.context_file or [])]:
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if args.arbitrate and args.source_lang.lower() not in {"ja", "japanese"}:
        raise ValueError("Three-model arbitration currently requires Japanese (Zipformer-ja). Disable arbitration for other languages.")
    ensemble = args.asr_strategy == "ensemble" and args.arbitrate and args.adjudication is None
    if not ensemble and (args.ensemble_source != "consensus" or not args.qwen_asr_hotwords or args.asr_window_seconds != 20.0):
        raise ValueError("Ensemble source/context controls require --asr-strategy ensemble and automatic --arbitrate")
    if getattr(args, "coherence_recipe", None) is not None and not ensemble:
        raise ValueError("Coherence recipes require automatic ensemble ASR with raw window evidence")
    if args.source_pause_units and not ensemble:
        raise ValueError("Source-pause units require automatic ensemble ASR and arbitration")
    if args.source_span_repair and not ensemble:
        raise ValueError("Source-span repair requires automatic ensemble ASR and arbitration")
    if args.ensemble_source == "nemo" and args.source_span_repair:
        raise ValueError("Source-span repair currently requires original Qwen ownership; disable it for NeMo source")
    asr_config = {"language": args.language, "model": args.model, "compute_type": args.compute_type,
                  "beam_size": args.beam_size, "no_demucs": args.no_demucs, "keep_temp": args.keep_temp,
                  "hotword_mode": args.hotword_mode,
                  "ensemble_source": args.ensemble_source, "qwen_asr_hotwords": args.qwen_asr_hotwords,
                  "verbose": args.verbose, "warden_admin_url": args.warden_admin,
                  "unload_warden_before_asr": not args.no_warden_unload}
    window_key = None
    if args.asr_window_seconds != 20.0:
        asr_config["asr_window_seconds"] = args.asr_window_seconds
        # Validate experimental geometry before context or any model call.
        TranscribeConfig(asr_window_seconds=args.asr_window_seconds)
        window_key = {"code": {name: file_hash(Path(__file__).with_name(name))
                               for name in ("ensemble_asr.py", "asr_consensus.py")}}
    if args.ensemble_source == "nemo" or args.nemo_model is not None or args.nemo_python is not None:
        asr_config.update(nemo_model=str(args.nemo_model) if args.nemo_model else None,
                          nemo_python=str(args.nemo_python) if args.nemo_python else None)
        checked = TranscribeConfig(**asr_config)  # Reject bad paths before context/model calls.
        asr_config.update(nemo_model=str(checked.nemo_model), nemo_python=str(checked.nemo_python))
    jobs = collect_jobs(args)
    nemo_runtime_key = None
    nemo_asr_key = None
    if args.ensemble_source == "nemo":
        from src.nemo_asr import runtime_identity
        nemo_runtime_key = runtime_identity(asr_config)
        nemo_asr_key = {"runtime": nemo_runtime_key, "consumers": {
            name: file_hash(Path(__file__).with_name(name)) for name in
            ("ensemble_asr.py", "asr_consensus.py", "evidence.py", "workflow.py")}}
    prepare_context(jobs, args)  # All LLM calls happen before GPU workers start.
    source_units_key = None
    if args.source_pause_units:
        asr_config["source_pause_units"] = True
        source_units_key = {"code": {name: file_hash(Path(__file__).with_name(name))
                                     for name in ("source_units.py", "pause_layout.py")}}
    active = []
    temporary_context = (nullcontext(tempfile.mkdtemp(prefix="autosub-workflow-")) if args.keep_temp
                         else tempfile.TemporaryDirectory(prefix="autosub-workflow-"))
    with temporary_context as temporary:
        directory = Path(temporary)
        if args.keep_temp:
            logger.info("Shared audio/worker manifests retained at %s", directory)
        pending = []
        for index, job in enumerate(jobs):
            job.media_hash = file_hash(job.media)
            key = fingerprint([VERSION, job.media_hash, asr_config, job.hotwords,
                               "ensemble" if ensemble else "sequential", args.asr_batch_size]
                              + ([source_units_key] if source_units_key else [])
                              + ([nemo_asr_key] if nemo_asr_key else [])
                              + ([window_key] if window_key else []))
            state = StageState(job.state_path)
            if args.force:
                artifact_path(job.source, ".requests.json").unlink(missing_ok=True)
                artifact_path(job.source, ".requests.repair-delta.json").unlink(missing_ok=True)
                artifact_path(job.source, ".display-requests.json").unlink(missing_ok=True)
                artifact_path(job.source, ".display-requests.pauses.json").unlink(missing_ok=True)
                for suffix in (".entity-lexicon.json", ".entity-classify.json", ".entity-plan.json", ".entity-render.json", ".source-span-requests.json"):
                    artifact_path(job.source, suffix).unlink(missing_ok=True)
            metadata = artifact_path(job.source, ".asr.json")
            previous = state.data.get("asr", {})
            outputs = [] if previous.get("status") == "empty" else asr_outputs(
                job, nemo=args.ensemble_source == "nemo")
            asr_matches = state.matches("asr", key, outputs)
            if ensemble and asr_matches and previous.get("status") != "empty":
                # Independent transcripts are part of the ensemble ASR result.
                # Missing/tampered evidence must rerun that worker, not silently
                # fall back to old recognition on already-cut subtitle cues.
                adj_key = fingerprint([VERSION, job.media_hash, file_hash(job.asr_source),
                                       args.source_lang, args.asr_batch_size])
                asr_matches = state.matches("arbitration", adj_key,
                                            [artifact_path(job.source, ".adjudication.json")])
            if not args.force and asr_matches:
                job.empty = previous.get("status") == "empty"
                job.successful = job.empty
                logger.info("Resume: ASR %s for %s", previous["status"], job.media)
            elif not ensemble and not args.force and job.source.exists() and not previous:
                # Adopt pre-existing raw SRTs from the legacy workflow once, explicitly.
                validate_blocks(parse_srt(job.source))
                if not metadata.exists():
                    write_json(metadata, {"provenance": "imported legacy SRT; ASR metrics unavailable"})
                state.save("asr", key, [job.asr_source, metadata], provenance="legacy import")
            else:
                pending.append({"media": str(job.media), "source": str(job.asr_source),
                                "state": str(job.state_path), "hotwords": job.hotwords,
                                "audio": str(directory / f"audio-{index}.wav"), "key": key,
                                "media_hash": job.media_hash,
                                "metadata": str(metadata),
                                "adjudication": str(artifact_path(job.source, ".adjudication.json"))})
            active.append((job, key, directory / f"audio-{index}.wav"))
        if pending:
            run_worker("src.ensemble_asr" if ensemble else "src.workflow_asr",
                       {"config": asr_config, "jobs": pending, "version": VERSION,
                        "batch_size": args.asr_batch_size, "source_lang": args.source_lang,
                        "nemo_runtime_identity": nemo_runtime_key},
                       directory, "asr.json")
        ready, arbitration = [], []
        for job, key, audio in active:
            state = StageState(job.state_path)
            record = state.data.get("asr", {})
            if record.get("status") == "empty" and state.matches("asr", key, []):
                job.empty = job.successful = True
                continue
            if not state.matches("asr", key, asr_outputs(job, nemo=args.ensemble_source == "nemo")):
                logger.error("ASR incomplete for %s; keeping input for resume", job.media)
                continue
            source = parse_srt(job.asr_source)
            if not source:
                continue
            ready.append(job)
            if args.arbitrate and job.config.adjudication is None:
                out = artifact_path(job.source, ".adjudication.json")
                adj_key = fingerprint([VERSION, job.media_hash, file_hash(job.asr_source), args.source_lang, args.asr_batch_size])
                job.config.adjudication = out
                if ensemble:
                    continue  # Worker owns evidence; language preflight checks its hashes.
                if args.force or not state.matches("arbitration", adj_key, [out]):
                    arbitration.append({"media": str(audio if audio.exists() else job.media),
                                        "source": str(job.asr_source), "out": str(out), "cues": len(source),
                                        "state": str(job.state_path), "key": adj_key})
        if arbitration:
            run_worker("src.adjudicate", {"jobs": arbitration, "batch_size": args.asr_batch_size,
                       "warden_admin": args.warden_admin, "unload_warden": not args.no_warden_unload},
                       directory, "arbitration.json")
        # Both GPU workers have exited; the 27B model may now remain resident.
        for job in ready:
            try:
                if args.arbitrate and not args.adjudication:
                    adj_key = fingerprint([VERSION, job.media_hash, file_hash(job.asr_source), args.source_lang, args.asr_batch_size])
                    if not StageState(job.state_path).matches("arbitration", adj_key, [job.config.adjudication]):
                        raise RuntimeError("Arbitration incomplete")
                process_language(job, args)
                StageState(job.state_path).save("attempt", job.media_hash, [],
                                               status="complete" if job.successful else "failed")
            except (RuntimeError, ValueError, OSError) as exc:
                StageState(job.state_path).save("attempt", job.media_hash, [], status="failed", error=str(exc))
                logger.exception("Workflow failed for %s: %s", job.media, exc)
        if args.organize:
            from src.organize import organize_unit
            for unit in sorted({job.unit for job in jobs}):
                members = [job for job in jobs if job.unit == unit]
                if all(job.successful for job in members):
                    organize_unit(unit)
                else:
                    logger.warning("Keep incomplete work %s in place for resume", unit)
    failures = sum(not job.successful for job in jobs)
    logger.info("Workflow complete: %d succeeded (%d empty), %d incomplete",
                len(jobs) - failures, sum(job.empty for job in jobs), failures)
    return int(failures > 0)
