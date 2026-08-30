from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.cuda_paths import setup_nvidia_lib_path

# Must run BEFORE the faster_whisper import: ctranslate2 dlopens cublas/cudnn.
setup_nvidia_lib_path()

from faster_whisper import WhisperModel

from src.config import TranscribeConfig
from src.warden import ASR_MIN_FREE_GB, ensure_gpu_headroom

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A single transcribed segment with timestamps."""

    index: int
    start: float
    end: float
    text: str


def load_model(config: TranscribeConfig) -> WhisperModel:
    """Load the faster-whisper model.

    Automatically falls back to CPU if CUDA is unavailable.
    """
    import torch

    if torch.cuda.is_available():
        # whisper (~4 GB) and the warden-resident 27B translator (~29 GB)
        # cannot share the 32 GB card: evict the LLM first when headroom is
        # short. See src/warden.py for the measured numbers.
        ensure_gpu_headroom(
            required_gb=ASR_MIN_FREE_GB,
            admin_url=config.warden_admin_url,
            enabled=config.unload_warden_before_asr,
        )
        device = "cuda"
        logger.info(
            "Using CUDA: %s (%.1f GB VRAM)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    else:
        device = "cpu"
        logger.warning("CUDA not available, falling back to CPU (this will be slow)")
        if config.compute_type == "float16":
            config.compute_type = "int8"
            logger.info("Switched compute_type to int8 for CPU inference")

    logger.info("Loading model: %s (compute_type=%s)", config.model, config.compute_type)
    model = WhisperModel(
        config.model,
        device=device,
        compute_type=config.compute_type,
    )
    logger.info("Model loaded successfully")
    return model


def transcribe(model: WhisperModel, audio_path: Path, config: TranscribeConfig) -> list[Segment]:
    """Transcribe audio file and return list of timed segments.

    Args:
        model: Loaded WhisperModel instance.
        audio_path: Path to 16kHz mono WAV file.
        config: Transcription configuration.

    Returns:
        List of Segment objects with timestamps and text.
    """
    logger.info(
        "Transcribing: %s (language=%s, beam_size=%d, hotwords=%d)",
        audio_path,
        config.language or "auto-detect",
        config.beam_size,
        len(config.hotwords),
    )

    segments_gen, info = model.transcribe(
        str(audio_path),
        language=config.language,
        beam_size=config.beam_size,
        vad_filter=True,
        vad_parameters=config.vad_parameters,
        condition_on_previous_text=False,
        hotwords=" ".join(config.hotwords) or None,
    )

    detected_lang = info.language
    lang_prob = info.language_probability
    logger.info("Detected language: %s (probability=%.2f)", detected_lang, lang_prob)

    # Consume the generator — it can only be iterated once
    segments: list[Segment] = []
    for i, seg in enumerate(segments_gen, start=1):
        text = seg.text.strip()
        if not text:
            continue
        segments.append(Segment(index=i, start=seg.start, end=seg.end, text=text))
        if config.verbose:
            logger.debug("[%.2fs → %.2fs] %s", seg.start, seg.end, text)

    # Re-index after filtering empty segments
    for i, seg in enumerate(segments, start=1):
        seg.index = i

    logger.info("Transcription complete: %d segments", len(segments))
    return segments


def release_model(model: WhisperModel) -> None:
    """Release a loaded WhisperModel so other CUDA users can have the VRAM.

    faster-whisper is a ctranslate2 model, not a torch module, so `del` +
    gc.collect() + torch.cuda.empty_cache() is what actually returns the
    memory (measured 2026-08-30: 32.9 GB free → load → 32.9 GB after
    release). Call this after ASR so the warden LLM can reload for
    translation.
    """
    import gc

    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Whisper model released — GPU free for the translation LLM")
