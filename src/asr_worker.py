"""Subprocess worker for GPU ASR work — see transcribe.run_transcribe.

Whisper's CUDA memory (~3.9 GB, including the ctranslate2 CUDA context and
cublas workspace) cannot be fully returned inside a long-lived process:
`del model` + `torch.cuda.empty_cache()` leaves ~3.9 GB resident, and the
warden LLM needs ~30 GB back before translation. Running ASR in a
short-lived subprocess means the CUDA context dies with it and every byte
of VRAM returns to the GPU.

This module is the subprocess entry point. The parent (transcribe.py) builds
a JSON spec and spawns `python -m src.asr_worker <spec>`. Logging inherits
the parent's stdout/stderr; the exit code is 0 only when every file was
transcribed.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("auto-subtitle.worker")


def main(spec: dict) -> int:
    from src.asr import load_model, transcribe
    from src.config import TranscribeConfig
    from src.srt import write_srt
    import transcribe as T  # reuse process_file (ffmpeg → demucs → whisper)

    logging.basicConfig(
        level=logging.DEBUG if spec.get("verbose") else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    files = [Path(p) for p in spec["files"]]
    outputs = [Path(p) for p in spec["outputs"]]

    config = TranscribeConfig(
        input_file=None,
        input_dir=Path(spec["input_dir"]),
        output_dir=Path(spec["output_dir"]),
        language=spec["language"],
        model=spec["model"],
        compute_type=spec["compute_type"],
        beam_size=spec["beam_size"],
        no_demucs=spec["no_demucs"],
        keep_temp=spec["keep_temp"],
        verbose=spec["verbose"],
        hotwords=spec.get("hotwords") or [],
        warden_admin_url=spec["warden_admin_url"],
        unload_warden_before_asr=spec["unload_warden_before_asr"],
    )

    logger.info(
        "ASR worker: %d file(s), model=%s (%s), hotwords=%d",
        len(files), config.model, config.compute_type, len(config.hotwords),
    )
    # Evicts the warden LLM first when headroom is short (src.warden).
    model = load_model(config)

    succeeded = 0
    failed = 0
    for i, (input_path, output_path) in enumerate(zip(files, outputs), start=1):
        if len(files) > 1:
            logger.info("=== [%d/%d] %s ===", i, len(files), input_path.name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if T.process_file(input_path, output_path, config, model):
            succeeded += 1
        else:
            failed += 1

    if len(files) > 1:
        logger.info("ASR worker complete: %d succeeded, %d failed", succeeded, failed)
    # No explicit release: this process exiting is what returns all VRAM.
    return 1 if failed == len(files) else 0


if __name__ == "__main__":
    sys.exit(main(json.loads(sys.argv[1])))
