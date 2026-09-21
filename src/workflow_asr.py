"""One GPU ASR worker for an evidence-first batch, with per-file status."""
from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
from pathlib import Path

from src.workflow_state import StageState, artifact_path

logger = logging.getLogger(__name__)


def run(spec: dict) -> int:
    from src.asr import load_model
    from src.config import TranscribeConfig
    from src.translate import make_snapshot
    from transcribe import process_file

    config = TranscribeConfig(**spec["config"])
    model = load_model(config)
    failed = 0
    for job in spec["jobs"]:
        source = Path(job["source"])
        source.parent.mkdir(parents=True, exist_ok=True)
        state = StageState(Path(job["state"]))
        config.hotwords = job["hotwords"]
        config.cached_audio = Path(job["audio"])
        started = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix=".asr-", dir=source.parent) as directory:
                staged = Path(directory) / source.name
                if not process_file(Path(job["media"]), staged, config, model):
                    raise RuntimeError("ASR failed; see worker logs")
                if source.exists():
                    make_snapshot(source, "pre-asr")
                if staged.exists():
                    staged.replace(source)
                    metadata = artifact_path(source, ".asr.json")
                    metadata.parent.mkdir(parents=True, exist_ok=True)
                    staged.with_suffix(".asr.json").replace(metadata)
                    state.save("asr", job["key"], [source, metadata], seconds=time.monotonic() - started)
                else:
                    source.unlink(missing_ok=True)
                    state.save("asr", job["key"], [], status="empty", seconds=time.monotonic() - started)
        except Exception as exc:
            logger.exception("ASR failed: %s", job["media"])
            state.save("asr", job["key"], [], status="failed", error=str(exc))
            failed += 1
    return int(failed > 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit(run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))))
