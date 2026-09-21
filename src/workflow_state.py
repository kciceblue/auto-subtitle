"""Content-addressed stage records and atomic JSON artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_path(source: Path, suffix: str) -> Path:
    """Keep sidecars in review/ when resuming an organized work."""
    parent = source.parent
    try:
        rel = source.resolve().relative_to(Path("output").resolve())
    except ValueError:
        rel = None
    if rel is not None and len(rel.parts) >= 3 and rel.parts[1] == "final":
        parent = Path("output") / rel.parts[0] / "review" / Path(*rel.parts[2:-1])
    return parent / (source.stem + suffix)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".autosub-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class StageState:
    def __init__(self, path: Path):
        self.path = path
        data = read_json(path, {})
        self.data = data if isinstance(data, dict) else {}

    def matches(self, stage: str, key: str, outputs: list[Path]) -> bool:
        record = self.data.get(stage, {})
        if not isinstance(record, dict) or record.get("key") != key or record.get("status") not in {"complete", "empty"}:
            return False
        try:
            return record.get("outputs") == {p.name: file_hash(p) for p in outputs}
        except OSError:
            return False

    def save(self, stage: str, key: str, outputs: list[Path], *,
             status: str = "complete", **details) -> None:
        self.data[stage] = {"key": key, "status": status,
                            "outputs": {p.name: file_hash(p) for p in outputs}, **details}
        write_json(self.path, self.data)
