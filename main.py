#!/usr/bin/env python3
"""Unified CLI for auto-subtitle: transcribe, translate, or full pipeline.

Usage:
    python main.py transcribe movie.mkv -l ja
    python main.py translate output/movie.srt --endpoint http://127.0.0.1:5000/v1/chat/completions
    python main.py pipeline -l ja --target-lang "Simplified Chinese"
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger("auto-subtitle")


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def add_transcribe_args(parser: argparse.ArgumentParser) -> None:
    """Add transcription-related arguments to a parser."""
    parser.add_argument(
        "input_file",
        type=Path,
        nargs="?",
        default=None,
        help="Input media file. If omitted, processes all files in input/ directory.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output .srt path (default: output/<input_stem>.srt)",
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default="auto",
        help="Source language code: ja, zh, en, auto (default: auto)",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="large-v3",
        help="Whisper model name (default: large-v3)",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="float16",
        choices=["float16", "int8_float16", "int8", "float32"],
        help="Compute type for inference (default: float16)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (default: 5)",
    )
    parser.add_argument(
        "--no-demucs",
        action="store_true",
        help="Skip vocal separation",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate temporary files",
    )
    parser.add_argument(
        "--hotwords-file",
        type=Path,
        default=None,
        help="File with hotword candidates (one per line, 'A->B' = correction). "
             "Merged with LLM-generated hotwords before ASR.",
    )
    parser.add_argument(
        "--context",
        type=Path,
        action="append",
        default=None,
        metavar="FILE",
        help="Context file (e.g. README.txt; PDF supported) used to LLM-generate ASR hotwords. "
             "Repeat the flag for several files. Omit to auto-discover context "
             "files near each input (content-aware scan, any filename).",
    )
    parser.add_argument(
        "--no-auto-context",
        action="store_true",
        help="Disable context auto-discovery entirely (no hotword context).",
    )
    parser.add_argument(
        "--no-context-scan",
        action="store_true",
        help="Disable the content-aware context scan; fall back to the legacy "
             "README* filename filter.",
    )
    parser.add_argument(
        "--hotword-endpoint",
        type=str,
        default=None,
        help="LLM endpoint for hotword generation (default: translation endpoint / "
             "http://127.0.0.1:8089/v1/chat/completions)",
    )
    parser.add_argument(
        "--no-warden-unload",
        action="store_true",
        help="Do NOT evict the llama-warden LLM from the GPU before ASR. "
             "By default the pipeline unloads warden's resident model when "
             "VRAM headroom is short, because whisper + the 27B translator "
             "exceed 32 GB together. Disable only if warden is actively "
             "serving other requests.",
    )
    parser.add_argument(
        "--warden-admin",
        type=str,
        default="http://127.0.0.1:8089/admin",
        help="llama-warden admin base URL (default: http://127.0.0.1:8089/admin)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )


def add_translate_args(parser: argparse.ArgumentParser, *, positional: bool = True) -> None:
    """Add translation-related arguments to a parser."""
    if positional:
        parser.add_argument(
            "input_srt",
            type=Path,
            help="Input .srt file to translate.",
        )
    parser.add_argument(
        "--translate-output",
        type=Path,
        default=None,
        help="Output translated .srt path (default: <input_stem>.translated.srt)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://127.0.0.1:5000/v1/chat/completions",
        help="LLM API endpoint (default: http://127.0.0.1:5000/v1/chat/completions)",
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default="Japanese",
        help="Source language name for translation prompt (default: Japanese)",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="Simplified Chinese",
        help="Target language name for translation prompt (default: Simplified Chinese)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Number of subtitle lines per LLM request (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="LLM request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries on LLM failure (default: 2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Token budget per LLM request (default: 16384 — raised for review "
             "DeepFix thinking; streaming stops at EOS so translation is unaffected). "
             "Sent explicitly so "
             "a reasoning model cannot spend the server's default budget on "
             "thinking and return an empty answer",
    )
    parser.add_argument(
        "--extra-payload",
        type=str,
        default=None,
        help='Extra JSON payload for LLM requests, e.g. \'{"temperature": 0.3}\'',
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=None,
        help="Path to a vocabulary file with context words/terms for translation",
    )
    parser.add_argument(
        "--context-file",
        type=Path,
        action="append",
        default=None,
        metavar="FILE",
        help="Context file injected into the translation prompt (plain text "
             "or PDF; content-kind auto-detected — a 台词台本 script gets "
             "script-aware sampling). "
             "Repeat the flag for several files "
             "(e.g. --context-file ../README.pdf --context-file ../NOTES.md)",
    )
    parser.add_argument(
        "--proofread",
        action="store_true",
        help="Run a whole-file proofread pass after translation "
             "(fixes consistency/alignment, verified 1:1 against the source)",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="After proofread, run a thinking-enabled semantic review pass "
             "(evidence-based gate: verified lines are released, unresolvable "
             "lines get a 存疑 flag; writes a REVIEW-*.md report)",
    )
    parser.add_argument(
        "--review-chunk-size",
        type=int,
        default=8,
        help="Lines per review LLM request (default: 8)",
    )
    parser.add_argument(
        "--reference-srt",
        type=Path,
        default=None,
        help="Reference SRT (hand-corrected/accepted translation) — its lines "
             "are matched by timestamp overlap and given to the review LLM "
             "as semantic anchors",
    )
    parser.add_argument(
        "--adjudication",
        type=Path,
        default=None,
        help="Adjudication JSON from src/adjudicate.py (main.adjudication.json) — "
             "enables the FTDC fast review path: suspicious lines are packed and "
             "re-reviewed with audio evidence, critic runs incrementally (≤2 rounds)",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=None,
        help="Dialogue script (台词台本) of the work — ground truth for review. "
             "Anchored to source lines (src/script_align.py) and injected as "
             "review evidence; outranks audio arbitration. When omitted, the "
             "first 'script'-kind auto-discovered context file is used.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto-subtitle: transcribe, translate, or full pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # transcribe
    sub_transcribe = subparsers.add_parser(
        "transcribe",
        help="Transcribe media file(s) to SRT.",
    )
    add_transcribe_args(sub_transcribe)
    # `pipeline` gets --vocab from add_translate_args; transcribe needs its own
    # (hotword generation parses the same 误->正 correction file).
    sub_transcribe.add_argument(
        "--vocab",
        type=Path,
        default=None,
        help="Vocabulary file (one term per line, 'A->B' = correction) used to "
             "ground ASR hotword generation",
    )

    # translate
    sub_translate = subparsers.add_parser(
        "translate",
        help="Translate an existing SRT file via LLM.",
    )
    add_translate_args(sub_translate, positional=True)
    sub_translate.add_argument(
        "--no-context-scan",
        action="store_true",
        help="Disable the content-aware context scan; fall back to the legacy "
             "README* filename filter.",
    )
    sub_translate.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    # pipeline
    sub_pipeline = subparsers.add_parser(
        "pipeline",
        help="Transcribe then translate in one step.",
    )
    add_transcribe_args(sub_pipeline)
    add_translate_args(sub_pipeline, positional=False)

    # review (review-only entry; the unified runner uses this after the
    # arbitration subprocess so FTDC evidence exists before DeepFix runs)
    sub_review = subparsers.add_parser(
        "review",
        help="Run the FTDC semantic review pass on an existing translation "
             "(no translation — source SRT + translated SRT in place).",
    )
    add_translate_args(sub_review, positional=True)
    sub_review.add_argument(
        "--translated",
        type=Path,
        default=None,
        help="Translated SRT to review (default: <input_stem>.zh.srt)",
    )
    sub_review.add_argument(
        "--no-context-scan",
        action="store_true",
        help="Disable the content-aware context scan; fall back to the legacy "
             "README* filename filter.",
    )
    sub_review.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    # organize (run.sh Phase 4): per-work-unit output/<unit>/{final,review}/
    # layout + input-side cleanup. No LLM involved.
    sub_organize = subparsers.add_parser(
        "organize",
        help="Reorganize pipeline output into output/<unit>/{final,review}/ "
             "(final = media + SRTs; review = context/intermediates) and "
             "clean the emptied input/<unit> folders. Incomplete units "
             "(media without a translated SRT) are left untouched.",
    )
    sub_organize.add_argument(
        "units", nargs="*", metavar="UNIT",
        help="Work unit name(s) (top-level input/ entry). Omit to organize "
             "every discoverable unit.",
    )
    sub_organize.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging",
    )

    # archive (manual hand-off): output/<unit> + zip → ready_for_human_review/
    sub_archive = subparsers.add_parser(
        "archive",
        help="Move organized output/<unit> folder(s) plus a zip of each into "
             "ready_for_human_review/. Manual step — run after checking the "
             "REVIEW report.",
    )
    sub_archive.add_argument(
        "units", nargs="*", metavar="UNIT",
        help="Unit name(s) to archive. Omit to archive every organized unit "
             "under output/.",
    )
    sub_archive.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging",
    )

    return parser


def _discover_readmes(base_dir: Path) -> list[Path]:
    """Find README*.txt/.md/.pdf files near a media file's directory chain (nearest first)."""
    from src.context_scan import is_project_root_file

    found: list[Path] = []
    d = base_dir
    for _ in range(3):
        for r in sorted(d.glob("README*")):
            if (r.is_file() and r.suffix.lower() in {".txt", ".md", ".pdf"}
                    and not is_project_root_file(r)):
                found.append(r)
        if d.parent == d:
            break
        d = d.parent
    # de-dup, preserve order
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in found:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def _missing_paths(candidates: list[tuple[str, Path | None]]) -> list[str]:
    """Report CLI-supplied paths that do not exist, as '<flag>: <path>'.

    Every one of these is validated inside the config dataclasses too, but
    that happens *after* transcription — hours of ASR followed by a
    FileNotFoundError. Callers run this before run_transcribe so a typo fails
    in the first second instead of the last.
    """
    missing: list[str] = []
    for label, path in candidates:
        if path is not None and not Path(path).exists():
            missing.append(f"{label}: {path}")
    return missing


def _check_input_paths(args: argparse.Namespace) -> list[str]:
    """Validate every path the user passed explicitly on the command line.

    Auto-discovered READMEs are skipped: they exist by construction.
    """
    candidates: list[tuple[str, Path | None]] = [
        ("--vocab", getattr(args, "vocab", None)),
        ("--hotwords-file", getattr(args, "hotwords_file", None)),
        ("--reference-srt", getattr(args, "reference_srt", None)),
    ]
    for p in getattr(args, "context", None) or []:
        candidates.append(("--context", p))
    for p in getattr(args, "context_file", None) or []:
        candidates.append(("--context-file", p))
    return _missing_paths(candidates)


def cmd_transcribe(args: argparse.Namespace) -> int:
    from src.config import TranscribeConfig
    from src.hotwords import generate_hotwords
    from transcribe import run_transcribe

    missing = _check_input_paths(args)
    if missing:
        for entry in missing:
            logger.error("File not found — %s", entry)
        return 1

    # Hotword generation: context (content-aware scan or --context) + vocab
    from src.context_scan import classify_explicit, scan_context_files
    from src.hotwords import DEFAULT_ENDPOINT

    extra_payload = getattr(args, "extra_payload", None)
    scan_payload = json.loads(extra_payload) if extra_payload else None
    scan_endpoint = args.hotword_endpoint or DEFAULT_ENDPOINT
    context_kinds: dict[str, str] = {}
    context_files: list[Path] = list(args.context or [])
    if context_files:
        explicit = classify_explicit(
            context_files, scan_endpoint, scan_payload,
            source_lang=getattr(args, "source_lang", "Japanese"),
        )
        context_kinds = {k: v.kind for k, v in explicit.items()}
    elif not args.no_auto_context:
        if args.input_file is not None:
            scans = scan_context_files(
                args.input_file.parent, scan_endpoint, scan_payload,
                use_llm=not args.no_context_scan,
            )
            context_files = [sc.path for sc in scans]
            context_kinds = {str(sc.path): sc.kind for sc in scans}
        else:
            # Batch mode: gather context from all input media directories
            from src.config import MEDIA_EXTENSIONS
            input_dir = Path("input")
            if input_dir.is_dir():
                media_files = sorted(
                    f for f in input_dir.rglob("*")
                    if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS
                )
                seen: set[Path] = set()
                for f in media_files:
                    for sc in scan_context_files(
                        f.parent, scan_endpoint, scan_payload,
                        use_llm=not args.no_context_scan,
                    ):
                        if sc.path not in seen:
                            seen.add(sc.path)
                            context_files.append(sc.path)
                            context_kinds[str(sc.path)] = sc.kind
        if context_files:
            logger.info("Auto-discovered context: %s", [str(p) for p in context_files])

    config = TranscribeConfig(
        input_file=args.input_file,
        output_file=args.output,
        language=args.language,
        model=args.model,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        no_demucs=args.no_demucs,
        keep_temp=args.keep_temp,
        verbose=args.verbose,
        context_files=context_files,
        context_kinds=context_kinds,
        hotwords_file=args.hotwords_file,
        hotword_endpoint=args.hotword_endpoint or getattr(args, "endpoint", None),
        hotword_extra_payload=(
            json.loads(extra_payload) if extra_payload else None
        ),
        warden_admin_url=args.warden_admin,
        unload_warden_before_asr=not args.no_warden_unload,
    )

    if context_files or args.hotwords_file or args.vocab:
        # Media title (filename) is itself context: 8/30 lesson — a work run
        # without context produced 31% 存疑; the title alone would anchor
        # the domain terms that plain ASR garbled.
        media_list = [args.input_file] if args.input_file else config.collect_input_files()
        title = _collect_titles(media_list)
        if title:
            logger.info("Media title as hotword context: %s", title)
        result = generate_hotwords(
            context_files=config.context_files,
            # --vocab is the file that actually carries the user's 误->正
            # corrections; --hotwords-file is the fallback alias.
            vocab_file=args.vocab or config.hotwords_file,
            source_lang=getattr(args, "source_lang", "Japanese"),
            endpoint=config.hotword_endpoint,
            extra_payload=config.hotword_extra_payload,
            cache_file=Path(".hotword_cache.json"),
            title=title,
        )
        config.hotwords = result.hotwords
        logger.info(
            "ASR hotwords (%s, %d): %s",
            result.source, len(result.hotwords), result.hotwords[:12],
        )

    exit_code, _ = run_transcribe(config)
    return exit_code


def _collect_titles(media_files: list[Path]) -> str | None:
    """Meaningful media titles from a file list, deduped, joined with ；.

    None when none of the filenames carry content signal (batch-named
    track01.wav / audio_001.wav etc.). See src/title_context.
    """
    from src.title_context import extract_title

    titles: list[str] = []
    seen: set[str] = set()
    for f in media_files:
        t = extract_title(f)
        if t and t not in seen:
            seen.add(t)
            titles.append(t)
    return "；".join(titles) if titles else None


def _find_media_for_srt(srt_path: Path) -> Path | None:
    """Locate the media file an SRT belongs to (same stem, nearest first).

    translate/review subcommands only receive SRT paths; the media file
    (whose filename carries the content title, and whose directory holds the
    context files) is looked up beside the SRT and recursively under
    input/ + output/ (media may sit in a subfolder, e.g.
    input/RJ123456/foo.wav while the SRT is output/RJ123456/foo.srt).
    """
    from src.config import MEDIA_EXTENSIONS

    stems = {srt_path.stem, srt_path.with_suffix("").name}
    candidates: list[Path] = [srt_path.parent]
    # Media moves input/ → output/ after pipeline; check both trees.
    for root in (Path("input"), Path("output")):
        if root.is_dir():
            candidates.append(root)
    for base in candidates:
        for stem in stems:
            for ext in MEDIA_EXTENSIONS:
                if base == srt_path.parent:
                    cand = base / f"{stem}{ext}"
                else:
                    # recursive: the media may be nested (input/RJ123456/…)
                    cand = next(base.rglob(f"{stem}{ext}"), None)
                if cand is not None and cand.is_file():
                    return cand
    return None


def _title_from_srt(srt_path: Path) -> str | None:
    """Best-effort meaningful media title for an SRT (see _find_media_for_srt)."""
    from src.title_context import extract_title

    media = _find_media_for_srt(srt_path)
    return extract_title(media) if media is not None else None


def _resolve_context(
    srt_path: Path,
    explicit: list[Path] | None,
    endpoint: str,
    extra_payload: dict | None,
    no_scan: bool,
    script_arg: Path | None,
    source_lang: str = "Japanese",
) -> tuple[list[Path], dict[str, str], Path | None]:
    """Resolve (context_files, context_kinds, script_file) for a source SRT.

    Explicit --context-file list wins (still classified for content kind);
    otherwise a content-aware scan of the SRT's directory AND of the matching
    media file's directory (single-file runs leave media + context in input/
    while the SRT lives in output/ — scanning only the SRT side lost the
    context there). The script file (ground truth for review) is the explicit
    --script, else the first 'script'-kind context file.
    """
    from src.context_scan import classify_explicit, scan_context_files

    script_file = script_arg
    if explicit:
        files = list(explicit)
        kinds = {k: v.kind for k, v in
                 classify_explicit(files, endpoint, extra_payload, source_lang).items()}
    else:
        scan_dirs: list[Path] = [srt_path.parent]
        media = _find_media_for_srt(srt_path)
        if media is not None and media.parent not in scan_dirs:
            scan_dirs.append(media.parent)
        files = []
        kinds = {}
        for d in scan_dirs:
            for sc in scan_context_files(
                d, endpoint, extra_payload, source_lang,
                use_llm=not no_scan,
            ):
                if sc.path not in files:
                    files.append(sc.path)
                    kinds[str(sc.path)] = sc.kind
    if script_file is None:
        for p in files:
            if kinds.get(str(p)) == "script":
                script_file = p
                break
    return files, kinds, script_file


def cmd_translate(args: argparse.Namespace) -> int:
    from src.config import TranslateConfig
    from src.translate import AllChunksFailedError, translate_srt
    from src.proofread import proofread_srt
    from src.review import review_srt

    missing = _check_input_paths(args)
    if missing:
        for entry in missing:
            logger.error("File not found — %s", entry)
        return 1

    extra = None
    if args.extra_payload:
        extra = json.loads(args.extra_payload)

    context_files, context_kinds, script_file = _resolve_context(
        args.input_srt, args.context_file, args.endpoint, extra,
        getattr(args, "no_context_scan", False), args.script, args.source_lang,
    )
    if context_files:
        logger.info("Translation context: %s", [str(p) for p in context_files])
    if script_file is not None:
        logger.info("Script (ground truth for review): %s", script_file)

    config = TranslateConfig(
        input_srt=args.input_srt,
        output_srt=args.translate_output,
        endpoint=args.endpoint,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        chunk_size=args.chunk_size,
        timeout=args.timeout,
        retries=args.retries,
        max_tokens=args.max_tokens,
        extra_payload=extra,
        vocab_file=args.vocab,
        context_files=context_files,
        context_kinds=context_kinds,
        script_file=script_file,
        title=_title_from_srt(args.input_srt),
        proofread=args.proofread,
        review=args.review,
        review_chunk_size=args.review_chunk_size,
        reference_srt=args.reference_srt,
        adjudication=args.adjudication,
        verbose=args.verbose,
    )

    input_path = config.input_srt
    if input_path is None or not input_path.exists():
        logger.error("Input SRT not found: %s", input_path)
        return 1

    output = translate_srt(input_path, config)
    logger.info("Translation complete: %s", output)

    try:
        if args.proofread:
            logger.info("Starting proofread pass: %s", output)
            proofread_srt(input_path, output, config)

        if args.review:
            logger.info("Starting semantic review pass (thinking-enabled): %s", output)
            report = review_srt(input_path, output, config, ref_srt=config.reference_srt,
                                adjudication=config.adjudication)
            logger.info("Review report: %s", report)
    except AllChunksFailedError as e:
        # The pass never ran: the translation on disk is untouched, but the
        # run must not report success.
        logger.error("%s", e)
        return 1

    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Review-only entry: FTDC semantic review over an existing translation.

    The unified runner (run.sh) calls this AFTER the arbitration subprocess,
    so the adjudication evidence (B+/B- audio votes) exists before DeepFix
    runs. Never translates — the .zh.srt must already exist.
    """
    from src.config import TranslateConfig
    from src.review import review_srt

    source = args.input_srt
    translated = args.translated or source.with_name(source.stem + ".zh.srt")
    if not source.is_file():
        logger.error("Source SRT not found: %s", source)
        return 1
    if not translated.is_file():
        logger.error("Translated SRT not found: %s (use --translated)", translated)
        return 1

    extra = None
    if args.extra_payload:
        try:
            extra = json.loads(args.extra_payload)
        except json.JSONDecodeError as e:
            logger.error("--extra-payload is not valid JSON: %s", e)
            return 1

    context_files, context_kinds, script_file = _resolve_context(
        source, args.context_file, args.endpoint, extra,
        getattr(args, "no_context_scan", False), args.script, args.source_lang,
    )
    if script_file is not None:
        logger.info("Script (ground truth for review): %s", script_file)

    config = TranslateConfig(
        input_srt=source,
        output_srt=translated,
        endpoint=args.endpoint,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        chunk_size=args.chunk_size,
        timeout=args.timeout,
        retries=args.retries,
        max_tokens=args.max_tokens,
        extra_payload=extra,
        vocab_file=args.vocab,
        context_files=context_files,
        context_kinds=context_kinds,
        script_file=script_file,
        title=_title_from_srt(source),
        review=True,
        review_chunk_size=args.review_chunk_size,
        reference_srt=args.reference_srt,
        adjudication=args.adjudication,
        verbose=args.verbose,
    )
    report = review_srt(source, translated, config, ref_srt=config.reference_srt,
                        adjudication=config.adjudication)
    logger.info("Review report: %s", report)
    return 0


def cmd_organize(args: argparse.Namespace) -> int:
    """Organize output into per-unit final/review folders (run.sh Phase 4)."""
    from src.organize import discover_units, organize_all

    units = list(args.units) or discover_units()
    if not units:
        logger.info("Organize: nothing to do (no work units found)")
        return 0
    organized = organize_all(units)
    logger.info("Organize: %d/%d unit(s) reorganized", organized, len(units))
    return 0


def cmd_archive(args: argparse.Namespace) -> int:
    """Archive organized units into ready_for_human_review/ (manual step)."""
    from src.organize import archivable_units, archive_unit

    units = list(args.units) or archivable_units()
    if not units:
        logger.error(
            "Archive: no organized unit found under output/ "
            "(run `main.py organize` first, or name the unit explicitly)"
        )
        return 1
    failed = 0
    for unit in units:
        try:
            dest = archive_unit(unit)
        except (FileNotFoundError, FileExistsError, OSError) as e:
            logger.error("Archive %s failed: %s", unit, e)
            failed += 1
            continue
        logger.info("Archived: %s (+ %s.zip)", dest, dest)
    return 1 if failed else 0


def _move_input_to_output(input_dir: Path, output_dir: Path) -> None:
    """Move all files from input_dir to output_dir, preserving structure.

    Skips files that already exist in output (e.g. SRTs just created).
    Removes empty directories left in input_dir after moving.
    """
    if not input_dir.is_dir():
        return

    for src in sorted(input_dir.rglob("*")):
        if not src.is_file():
            continue
        # Skip .gitkeep
        if src.name == ".gitkeep":
            continue
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        if dst.exists():
            # Remove from input only when the output copy matches (size
            # check); a differing file must not be silently deleted.
            if dst.stat().st_size == src.stat().st_size:
                logger.debug("Already in output, skipping: %s", rel)
                src.unlink()
            else:
                logger.warning(
                    "Output already has a DIFFERENT %s — keeping the input "
                    "copy for manual resolution", rel,
                )
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        logger.info("Moved: %s → %s", rel, dst)

    # Clean up empty directories in input (bottom-up)
    for d in sorted(input_dir.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass  # not empty

    logger.info("Input directory cleaned up")


def cmd_pipeline(args: argparse.Namespace) -> int:
    from src.config import TranscribeConfig, TranslateConfig, MEDIA_EXTENSIONS
    from src.hotwords import generate_hotwords
    from src.translate import AllChunksFailedError, translate_srt
    from src.proofread import proofread_srt
    from src.review import review_srt
    from transcribe import run_transcribe

    # Fail fast: every one of these is only checked inside TranslateConfig,
    # which is built *after* transcription. A typo must not cost a full ASR run.
    missing = _check_input_paths(args)
    if missing:
        for entry in missing:
            logger.error("File not found — %s", entry)
        return 1

    input_dir = Path("input")
    output_dir = Path("output")

    # Step 0: Scan input directory
    if args.input_file is not None:
        logger.info("Single file mode: %s", args.input_file)
        media_files = [args.input_file]
    else:
        media_files = sorted(
            f for f in input_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS
        )
        if not media_files:
            logger.error("No media files found in %s/", input_dir)
            return 1
        # Log the discovered structure
        dirs = sorted({f.parent.relative_to(input_dir) for f in media_files})
        logger.info(
            "Scanned %s/: %d media file(s) in %d location(s)",
            input_dir, len(media_files), len(dirs),
        )
        for d in dirs:
            count = sum(1 for f in media_files if f.parent.relative_to(input_dir) == d)
            label = str(d) if str(d) != "." else "(root)"
            logger.info("  %s: %d file(s)", label, count)

    # Context: explicit --context wins (classified for content kind); else a
    # content-aware scan of every media directory (src/context_scan — cached,
    # one LLM call per directory). The legacy README* filter is only the
    # scan's own degradation path now, same as the other subcommands.
    from src.context_scan import ScannedContext, classify_explicit, scan_context_files

    extra = json.loads(args.extra_payload) if args.extra_payload else None
    scan_endpoint = args.hotword_endpoint or args.endpoint
    context_files: list[Path] = list(args.context or [])
    context_kinds: dict[str, str] = {}
    dir_scans: dict[Path, list[ScannedContext]] = {}
    if context_files:
        explicit_kinds = classify_explicit(
            context_files, scan_endpoint, extra, args.source_lang)
        context_kinds = {k: v.kind for k, v in explicit_kinds.items()}
    elif not args.no_auto_context:
        for d in sorted({f.parent for f in media_files}):
            scans = scan_context_files(
                d, scan_endpoint, extra,
                args.source_lang, use_llm=not args.no_context_scan,
            )
            dir_scans[d] = scans
            for sc in scans:
                if sc.path not in context_files:
                    context_files.append(sc.path)
                    context_kinds[str(sc.path)] = sc.kind
        if context_files:
            logger.info("Auto-discovered context: %s", [str(p) for p in context_files])

    # Explicit --context-file overrides translation context for every file
    # (classified once for content kind); otherwise translation context is
    # resolved per media directory inside the loop below, so a batch over
    # several unrelated works does not leak one work's context into another.
    explicit_translate_context: list[Path] | None = None
    explicit_translate_kinds: dict[str, str] = {}
    if args.context_file:
        explicit_translate_context = list(args.context_file)
        explicit_translate_kinds = {
            k: v.kind for k, v in classify_explicit(
                explicit_translate_context, args.endpoint, extra, args.source_lang
            ).items()
        }
    elif args.context:
        explicit_translate_context = context_files
        explicit_translate_kinds = context_kinds

    # Step 1: Transcribe (with context-grounded hotwords when available)
    transcribe_config = TranscribeConfig(
        input_file=args.input_file,
        output_file=args.output,
        language=args.language,
        model=args.model,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        no_demucs=args.no_demucs,
        keep_temp=args.keep_temp,
        verbose=args.verbose,
        context_files=context_files,
        context_kinds=context_kinds,
        hotwords_file=args.hotwords_file,
        hotword_endpoint=args.hotword_endpoint or args.endpoint,
        hotword_extra_payload=extra,
        warden_admin_url=args.warden_admin,
        unload_warden_before_asr=not args.no_warden_unload,
    )

    if context_files or args.hotwords_file or args.vocab:
        # Media title (filename) is itself context — same wiring as
        # cmd_transcribe. 8/30: this was missing here, so pipeline runs
        # over context-less folders transcribe with ZERO hotwords (the
        # title's domain terms like 去勢 never biased the ASR).
        title = _collect_titles(media_files)
        if title:
            logger.info("Media title as hotword context: %s", title)
        result = generate_hotwords(
            context_files=transcribe_config.context_files,
            # --vocab is the file that actually carries the user's 误->正
            # corrections; --hotwords-file is the fallback alias.
            vocab_file=args.vocab or transcribe_config.hotwords_file,
            source_lang=args.source_lang,
            endpoint=transcribe_config.hotword_endpoint,
            extra_payload=transcribe_config.hotword_extra_payload,
            cache_file=Path(".hotword_cache.json"),
            title=title,
        )
        transcribe_config.hotwords = result.hotwords
        logger.info(
            "ASR hotwords (%s, %d): %s",
            result.source, len(result.hotwords), result.hotwords[:12],
        )

    exit_code, srt_paths = run_transcribe(transcribe_config)
    if exit_code != 0 or not srt_paths:
        logger.error("Transcription failed, skipping translation")
        return exit_code

    # Step 2: Translate each output SRT with ITS OWN context: run_transcribe
    # resolves outputs from the same sorted media collection, so srt_paths
    # and media_files pair up by index.
    media_by_srt: dict[Path, Path] = {}
    if len(srt_paths) == len(media_files):
        media_by_srt = dict(zip(srt_paths, media_files))

    from src.title_context import extract_title

    failed_files: list[Path] = []
    for srt_path in srt_paths:
        if not srt_path.is_file():
            # Per-file ASR failure (the worker keeps going; only all-failed
            # exits non-zero). Skip it here instead of crashing the batch.
            logger.error("ASR produced no SRT for %s — skipping translation", srt_path)
            failed_files.append(srt_path)
            continue

        media = media_by_srt.get(srt_path)
        if explicit_translate_context is not None:
            file_context = explicit_translate_context
            file_kinds = explicit_translate_kinds
        elif media is not None:
            scans = dir_scans.get(media.parent, [])
            file_context = [sc.path for sc in scans]
            file_kinds = {str(sc.path): sc.kind for sc in scans}
        else:
            file_context = context_files
            file_kinds = context_kinds
        script_file = args.script
        if script_file is None:
            for p in file_context:
                if file_kinds.get(str(p)) == "script":
                    script_file = p
                    break

        translate_config = TranslateConfig(
            output_srt=args.translate_output,
            endpoint=args.endpoint,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            chunk_size=args.chunk_size,
            timeout=args.timeout,
            retries=args.retries,
            max_tokens=args.max_tokens,
            extra_payload=extra,
            vocab_file=args.vocab,
            context_files=file_context,
            context_kinds=file_kinds,
            script_file=script_file,
            title=extract_title(media) if media is not None else _title_from_srt(srt_path),
            proofread=args.proofread,
            review=args.review,
            review_chunk_size=args.review_chunk_size,
            reference_srt=args.reference_srt,
            adjudication=args.adjudication,
            verbose=args.verbose,
        )

        logger.info("Translating: %s", srt_path)
        translated = translate_srt(srt_path, translate_config)
        logger.info("Translated: %s", translated)
        try:
            if args.proofread:
                logger.info("Proofreading: %s", translated)
                proofread_srt(srt_path, translated, translate_config)
            if args.review:
                logger.info("Semantic review (thinking-enabled): %s", translated)
                report = review_srt(
                    srt_path, translated, translate_config,
                    ref_srt=translate_config.reference_srt,
                    adjudication=translate_config.adjudication,
                )
                logger.info("Review report: %s", report)
        except AllChunksFailedError as e:
            # Keep going through the batch — this file's translation is on disk
            # and untouched — but the run as a whole must exit non-zero.
            logger.error("%s", e)
            failed_files.append(translated)

    # Step 3: Move everything from input/ to output/ (preserving structure)
    if args.input_file is None:
        logger.info("Moving input files to output directory...")
        _move_input_to_output(input_dir, output_dir)

    if failed_files:
        logger.error(
            "%d file(s) failed (ASR produced no SRT, or a pass failed on "
            "every chunk): %s",
            len(failed_files), [str(p) for p in failed_files],
        )
        return 1

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "transcribe":
        return cmd_transcribe(args)
    elif args.command == "translate":
        return cmd_translate(args)
    elif args.command == "pipeline":
        return cmd_pipeline(args)
    elif args.command == "review":
        return cmd_review(args)
    elif args.command == "organize":
        return cmd_organize(args)
    elif args.command == "archive":
        return cmd_archive(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
