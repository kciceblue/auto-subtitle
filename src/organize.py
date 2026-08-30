"""Post-pipeline output organization + manual archiving.

Every processed *work unit* (a top-level entry under input/: a folder like
input/RJ123456/, or a bare root file input/foo.wav) ends up as one folder
under output/ named after the unit:

    output/<unit>/
    ├── final/    media + source .srt + translated .srt  (the deliverable)
    └── review/   context files (README/script), adjudication JSON,
                  REVIEW-*.md reports, .pre-proofread/.pre-review snapshots,
                  and anything else the pipeline produced

organize_unit() is the workflow's closing step (run.sh Phase 4): it moves the
pipeline's mirrored output plus any files still sitting on the input side
into this layout, then prunes the emptied input/<unit> directory. It refuses
to touch an *incomplete* unit (a media file without a translated SRT), so a
partially-failed batch stays in place for a re-run.

archive_unit() is the MANUAL hand-off step (./run.sh archive [unit ...]):
it zips output/<unit> and moves both the folder and the zip into
ready_for_human_review/:

    ready_for_human_review/
    ├── <unit>/       (the organized folder, moved as-is)
    └── <unit>.zip    (zip of the same folder)
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from src.config import MEDIA_EXTENSIONS

logger = logging.getLogger(__name__)

FINAL_DIR = "final"
REVIEW_DIR = "review"

_SNAPSHOT_RE = re.compile(r"\.pre-(proofread|review)\.srt$")


def _is_snapshot(path: Path) -> bool:
    return bool(_SNAPSHOT_RE.search(path.name))


def _is_media(path: Path) -> bool:
    return path.suffix.lower() in MEDIA_EXTENSIONS


def _is_final(path: Path) -> bool:
    """final/ = media + every non-snapshot .srt (source and translated)."""
    if _is_media(path):
        return True
    return path.suffix.lower() == ".srt" and not _is_snapshot(path)


def _unit_files(
    unit: str, input_dir: Path, output_dir: Path
) -> list[tuple[Path, Path]]:
    """All loose (not yet organized) files of a unit as (abs path, rel path).

    Collects from four places: output/<unit>/ (excluding final/ and review/),
    root-level output/<unit>.* + output/REVIEW-<unit>.* (bare-file units),
    input/<unit>/ (single-file runs leave media+context there), and root-level
    input/<unit>.*.
    """
    files: list[tuple[Path, Path]] = []
    seen: set[Path] = set()

    def add(p: Path, rel: Path) -> None:
        if p.is_file() and p.name != ".gitkeep" and p not in seen:
            seen.add(p)
            files.append((p, rel))

    out_root = output_dir / unit
    if out_root.is_dir():
        for p in sorted(out_root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(out_root)
            if rel.parts and rel.parts[0] in (FINAL_DIR, REVIEW_DIR):
                continue
            add(p, rel)
    for p in sorted(output_dir.glob(f"{unit}.*")) + sorted(
        output_dir.glob(f"REVIEW-{unit}.*")
    ):
        add(p, Path(p.name))

    in_root = input_dir / unit
    if in_root.is_dir():
        for p in sorted(in_root.rglob("*")):
            if p.is_file():
                add(p, p.relative_to(in_root))
    for p in sorted(input_dir.glob(f"{unit}.*")):
        add(p, Path(p.name))

    return files


def _has_translation(
    stem: str, rel_dir: Path, files: list[tuple[Path, Path]], final_root: Path
) -> bool:
    """A translated SRT for media `<rel_dir>/<stem>.<ext>` exists somewhere.

    A translated SRT is `<stem>.<lang>.srt` (non-snapshot, not the bare
    source `<stem>.srt`), looked for among the unit's loose files and inside
    an already-organized final/ tree (partial re-runs).
    """
    prefix = stem + "."
    source_name = stem + ".srt"

    def matches(name: str) -> bool:
        return (
            name.endswith(".srt")
            and name.startswith(prefix)
            and name != source_name
            and not _SNAPSHOT_RE.search(name)
        )

    for p, rel in files:
        if rel.parent == rel_dir and matches(p.name):
            return True
    if final_root.is_dir():
        for p in final_root.rglob("*.srt"):
            if p.relative_to(final_root).parent == rel_dir and matches(p.name):
                return True
    return False


def _prune_empty_dirs(root: Path) -> None:
    """Remove empty directories under (and including) root, bottom-up."""
    if not root.is_dir():
        return
    for d in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def organize_unit(
    unit: str,
    input_dir: Path = Path("input"),
    output_dir: Path = Path("output"),
) -> bool:
    """Organize one unit into output/<unit>/{final,review}/ and clean input.

    Returns True when files were moved. An incomplete unit (media without a
    translated SRT) is left untouched with a warning so a re-run can finish
    it. Already-organized units are a no-op.
    """
    files = _unit_files(unit, input_dir, output_dir)
    if not files:
        logger.debug("Organize %s: nothing loose to organize", unit)
        return False

    unit_root = output_dir / unit
    final_root = unit_root / FINAL_DIR

    media = [(p, rel) for p, rel in files if _is_media(p)]
    missing = [
        rel for p, rel in media
        if not _has_translation(p.stem, rel.parent, files, final_root)
    ]
    if missing:
        logger.warning(
            "Organize %s: SKIPPED — %d media file(s) have no translated SRT "
            "yet (%s); re-run the pipeline to finish this unit first",
            unit, len(missing), ", ".join(str(m) for m in missing[:5]),
        )
        return False

    moved = 0
    for p, rel in files:
        sub = FINAL_DIR if _is_final(p) else REVIEW_DIR
        dst = unit_root / sub / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if dst.stat().st_size == p.stat().st_size:
                p.unlink()
            else:
                logger.warning(
                    "Organize %s: %s already exists with different size — "
                    "keeping both (source left at %s)", unit, dst, p,
                )
            continue
        shutil.move(str(p), str(dst))
        moved += 1

    # Prune emptied source-side directories (the "processed folder cleanup").
    _prune_empty_dirs(input_dir / unit)
    for d in sorted((p for p in unit_root.rglob("*") if p.is_dir()), reverse=True):
        if d.name in (FINAL_DIR, REVIEW_DIR) and d.parent == unit_root:
            continue
        try:
            d.rmdir()
        except OSError:
            pass

    logger.info(
        "Organized %s: %d file(s) → %s/{%s,%s}",
        unit, moved, unit_root, FINAL_DIR, REVIEW_DIR,
    )
    return True


def discover_units(
    input_dir: Path = Path("input"),
    output_dir: Path = Path("output"),
) -> list[str]:
    """Work-unit names present on either side.

    Directories become units by name; root-level media/SRT files become units
    by first-dot stem (foo.zh.srt → foo). REVIEW-/adjudication artifacts ride
    along via _unit_files and never define a unit on their own.
    """
    units: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name and name not in seen:
            seen.add(name)
            units.append(name)

    for root in (output_dir, input_dir):
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            if p.name == ".gitkeep":
                continue
            if p.is_dir():
                add(p.name)
            elif p.is_file() and (_is_media(p) or p.suffix.lower() == ".srt"):
                add(p.name.split(".")[0])
    return units


def organize_all(
    units: list[str] | None = None,
    input_dir: Path = Path("input"),
    output_dir: Path = Path("output"),
) -> int:
    """Organize the given units (default: every discoverable one)."""
    if not units:
        units = discover_units(input_dir, output_dir)
    organized = 0
    for unit in units:
        if organize_unit(unit, input_dir, output_dir):
            organized += 1
    if not units:
        logger.info("Organize: no work units found")
    return organized


def archive_unit(
    unit: str,
    output_dir: Path = Path("output"),
    review_dir: Path = Path("ready_for_human_review"),
) -> Path:
    """Zip output/<unit> and move folder + zip into ready_for_human_review/.

    Manual hand-off step. Refuses to overwrite an existing archive of the
    same unit. Returns the destination folder.
    """
    src = output_dir / unit
    if not src.is_dir():
        raise FileNotFoundError(f"No organized output folder to archive: {src}")
    if not (src / FINAL_DIR).is_dir():
        raise FileNotFoundError(
            f"{src} has no {FINAL_DIR}/ — run `main.py organize` first"
        )
    review_dir.mkdir(parents=True, exist_ok=True)
    dest = review_dir / unit
    zip_path = review_dir / f"{unit}.zip"
    if dest.exists() or zip_path.exists():
        raise FileExistsError(
            f"Archive target already exists ({dest} / {zip_path}) — "
            "remove or rename it first"
        )
    shutil.make_archive(str(review_dir / unit), "zip",
                        root_dir=output_dir, base_dir=unit)
    shutil.move(str(src), str(dest))
    logger.info("Archived %s → %s (+ %s)", unit, dest, zip_path)
    return dest


def archivable_units(output_dir: Path = Path("output")) -> list[str]:
    """Units under output/ that are organized (have a final/ folder)."""
    if not output_dir.is_dir():
        return []
    return sorted(
        p.name for p in output_dir.iterdir()
        if p.is_dir() and (p / FINAL_DIR).is_dir()
    )
