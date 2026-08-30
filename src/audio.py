from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def check_ffmpeg() -> None:
    """Verify ffmpeg is available on PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it: sudo apt install ffmpeg"
        )


def extract_audio(input_file: Path, output_wav: Path) -> Path:
    """Extract audio from media file as 16kHz mono WAV.

    Args:
        input_file: Path to input media file.
        output_wav: Path to write the extracted WAV.

    Returns:
        Path to the extracted WAV file.
    """
    check_ffmpeg()
    logger.info("Extracting audio: %s → %s", input_file, output_wav)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_file),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error("ffmpeg stderr:\n%s", result.stderr)
        raise RuntimeError(f"ffmpeg failed with return code {result.returncode}")

    logger.info("Audio extracted successfully (%.1f MB)", output_wav.stat().st_size / 1e6)
    return output_wav


def separate_vocals(audio_wav: Path, output_dir: Path) -> Path:
    """Use demucs to isolate vocals from audio.

    Args:
        audio_wav: Path to input WAV file.
        output_dir: Directory for demucs output.

    Returns:
        Path to the vocals-only WAV file.
    """
    try:
        # Probe what `-m demucs` actually runs: demucs.api does not exist in
        # demucs 4.0.x, so probing it declared a working install "missing"
        # and separation silently never ran.
        import demucs.separate  # noqa: F401 — just checking availability
    except ImportError:
        raise RuntimeError(
            "demucs not installed. Install with: pip install demucs\n"
            "Or skip vocal separation with --no-demucs"
        )

    logger.info("Running vocal separation with demucs (htdemucs)...")

    # torchcodec (torchaudio's save backend) dlopens NVIDIA NPP libs from
    # pip's nvidia-* packages; export their lib dirs for the child process.
    from src.cuda_paths import setup_nvidia_lib_path
    setup_nvidia_lib_path()

    # sys.executable, not bare "python": the venv is not activated when run.sh
    # invokes .venv/bin/python3 directly, so PATH's "python" is the system
    # interpreter without demucs — separation then silently degrades every run.
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", "htdemucs",
        "-o", str(output_dir),
        str(audio_wav),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error("demucs stderr:\n%s", result.stderr)
        raise RuntimeError(f"demucs failed with return code {result.returncode}")

    # demucs outputs to: output_dir/htdemucs/<stem_name>/vocals.wav
    vocals_path = output_dir / "htdemucs" / audio_wav.stem / "vocals.wav"
    if not vocals_path.exists():
        raise FileNotFoundError(
            f"Expected demucs vocals output at {vocals_path}, but file not found. "
            f"Check demucs output in {output_dir}"
        )

    logger.info("Vocal separation complete: %s", vocals_path)
    return vocals_path
