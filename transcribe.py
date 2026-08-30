#!/usr/bin/env python3
"""Auto-subtitle: transcribe video/audio to SRT using faster-whisper.

Usage:
    python transcribe.py                           # process all files in input/
    python transcribe.py movie.mkv                 # process a single file
    python transcribe.py movie.mkv -o out.srt      # single file, custom output

Examples:
    python transcribe.py -l ja
    python transcribe.py anime.mkv -l ja --no-demucs
    python transcribe.py podcast.mp4 -l zh -o chinese_subs.srt
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

from src.config import TranscribeConfig
from src.audio import extract_audio, separate_vocals
from src.asr import transcribe
from src.srt import write_srt

logger = logging.getLogger("auto-subtitle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe video/audio to SRT subtitles using faster-whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
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
        help="Skip vocal separation (faster, but less accurate on BGM-heavy audio)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate temporary files (extracted WAV, demucs output)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def process_file(
    input_path: Path,
    output_path: Path,
    config: TranscribeConfig,
    model,
) -> bool:
    """Process a single media file. Returns True on success."""
    logger.info("Processing: %s → %s", input_path, output_path)
    t_start = time.monotonic()
    temp_dir = Path(tempfile.mkdtemp(prefix="autosub_"))

    try:
        # Step 1: Extract audio
        raw_wav = temp_dir / "audio_raw.wav"
        extract_audio(input_path, raw_wav)

        # Step 2: Vocal separation (optional)
        if config.no_demucs:
            audio_for_asr = raw_wav
            logger.info("Skipping vocal separation (--no-demucs)")
        else:
            try:
                audio_for_asr = separate_vocals(raw_wav, temp_dir)
            except RuntimeError as e:
                logger.warning("Vocal separation failed, proceeding without it: %s", e)
                audio_for_asr = raw_wav

        # Step 3: Transcribe
        segments = transcribe(model, audio_for_asr, config)

        if not segments:
            logger.error("No speech segments detected in %s", input_path.name)
            return False

        # Step 4: Write SRT
        write_srt(segments, output_path)

        elapsed = time.monotonic() - t_start
        logger.info(
            "Done in %.1fs — %d segments → %s",
            elapsed,
            len(segments),
            output_path,
        )
        return True

    except Exception:
        logger.exception("Failed to process %s", input_path.name)
        return False

    finally:
        if not config.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            logger.info("Temp files kept at: %s", temp_dir)


def run_transcribe(config: TranscribeConfig) -> tuple[int, list[Path]]:
    """Run transcription pipeline in a dedicated subprocess.

    Returns:
        Tuple of (exit_code, list_of_output_srt_paths).

    GPU isolation: whisper's CUDA memory (~3.9 GB incl. context/workspace)
    cannot be returned inside a long-lived process, and the warden LLM needs
    ~30 GB back before translation. The actual model load + transcription
    therefore runs in a short-lived child process (src.asr_worker.py); when
    it exits, its CUDA context is destroyed and all VRAM returns to the GPU.
    """
    # Collect files to process
    if config.input_file is not None:
        files = [config.input_file]
    else:
        files = config.collect_input_files()
        if not files:
            logger.error("No media files found in %s/", config.input_dir)
            return 1, []
        logger.info("Found %d file(s) in %s/", len(files), config.input_dir)

    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for input_path in files:
        output_path = config.resolve_output(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_paths.append(output_path)

    spec = {
        "input_dir": str(config.input_dir),
        "output_dir": str(config.output_dir),
        "language": config.language,
        "model": config.model,
        "compute_type": config.compute_type,
        "beam_size": config.beam_size,
        "no_demucs": config.no_demucs,
        "keep_temp": config.keep_temp,
        "verbose": config.verbose,
        "hotwords": config.hotwords,
        "warden_admin_url": config.warden_admin_url,
        "unload_warden_before_asr": config.unload_warden_before_asr,
        "files": [str(f) for f in files],
        "outputs": [str(o) for o in output_paths],
    }

    import os
    import subprocess
    import sys

    env = dict(os.environ)
    # Make `src` importable from the child regardless of CWD (run.sh cds to
    # the project dir, but direct invocations may not).
    env.setdefault("PYTHONPATH", str(Path(__file__).parent))
    logger.info("Spawning ASR subprocess: %d file(s) via %s", len(files), sys.executable)
    proc = subprocess.run(
        [sys.executable, "-m", "src.asr_worker", json.dumps(spec)],
        env=env,
    )
    exit_code = proc.returncode
    # The worker logs per-file success/failure itself (inherited stdout);
    # a non-zero exit means every file failed.
    if exit_code != 0:
        logger.error("ASR subprocess exited %d — all %d file(s) failed", exit_code, len(files))
    return exit_code, output_paths


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

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
    )

    exit_code, _ = run_transcribe(config)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
