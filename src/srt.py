from __future__ import annotations

import logging
from pathlib import Path

from src.asr import Segment

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm.

    Args:
        seconds: Time in seconds (float).

    Returns:
        Formatted timestamp string.
    """
    if seconds < 0:
        seconds = 0.0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))

    # Clamp millis to 999 to avoid rounding to 1000
    if millis >= 1000:
        millis = 999

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segment_to_srt_block(segment: Segment) -> str:
    """Format a single segment as an SRT entry.

    Args:
        segment: Transcribed segment with timing and text.

    Returns:
        SRT-formatted string block.
    """
    start = format_timestamp(segment.start)
    end = format_timestamp(segment.end)
    return f"{segment.index}\n{start} --> {end}\n{segment.text}\n"


def write_srt(segments: list[Segment], output_path: Path) -> Path:
    """Write segments to an SRT subtitle file.

    Args:
        segments: List of transcribed segments.
        output_path: Path to write the .srt file.

    Returns:
        Path to the written SRT file.
    """
    if not segments:
        logger.warning("No segments to write — output file will be empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(segment_to_srt_block(segment))
            f.write("\n")

    logger.info("SRT written: %s (%d entries)", output_path, len(segments))
    return output_path
