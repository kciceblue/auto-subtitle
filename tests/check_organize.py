"""Offline checks for src/organize.py (no LLM, no GPU).

Covers: dir-unit organize, single-file-mode input-side cleanup, root-file
units, incomplete-unit refusal, idempotency, archive move+zip, and the
clean_title repeated-strip fix.
Run: .venv/bin/python3 tests/check_organize.py
"""
from __future__ import annotations

import logging
import shutil
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.organize import (
    FINAL_DIR,
    REVIEW_DIR,
    archivable_units,
    archive_unit,
    discover_units,
    organize_all,
    organize_unit,
)
from src.title_context import clean_title

TMP = Path("/tmp/check_organize")


def _fresh() -> tuple[Path, Path, Path]:
    if TMP.exists():
        shutil.rmtree(TMP)
    inp, out, rfh = TMP / "input", TMP / "output", TMP / "ready"
    for d in (inp, out, rfh):
        d.mkdir(parents=True)
    return inp, out, rfh


def _touch(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def ok(msg: str) -> None:
    print(f"OK  {msg}")


def fail(msg: str) -> None:
    print(f"FAIL {msg}")
    sys.exit(1)


# ── 1. batch dir unit: everything already moved to output/<unit>/ ─────
inp, out, rfh = _fresh()
u = out / "RJX"
_touch(u / "track01.wav")
_touch(u / "track01.srt")
_touch(u / "track01.zh.srt")
_touch(u / "track01.zh.pre-proofread.srt")
_touch(u / "track01.zh.pre-review.srt")
_touch(u / "track01.adjudication.json")
_touch(u / "REVIEW-track01.zh.md")
_touch(u / "README.txt")
_touch(u / "script.txt")

if not organize_unit("RJX", inp, out):
    fail("1: organize_unit returned False for a complete unit")
expect_final = {"track01.wav", "track01.srt", "track01.zh.srt"}
expect_review = {
    "track01.zh.pre-proofread.srt", "track01.zh.pre-review.srt",
    "track01.adjudication.json", "REVIEW-track01.zh.md",
    "README.txt", "script.txt",
}
got_final = {p.name for p in (u / FINAL_DIR).rglob("*") if p.is_file()}
got_review = {p.name for p in (u / REVIEW_DIR).rglob("*") if p.is_file()}
if got_final != expect_final:
    fail(f"1: final/ = {got_final}, expected {expect_final}")
if got_review != expect_review:
    fail(f"1: review/ = {got_review}, expected {expect_review}")
loose = [p for p in u.iterdir() if p.name not in (FINAL_DIR, REVIEW_DIR)]
if loose:
    fail(f"1: loose entries left in unit root: {loose}")
ok("1: dir unit → final/ (media+srt) + review/ (rest), no loose files")

# idempotent second run
if organize_unit("RJX", inp, out):
    fail("1b: second organize_unit moved files again")
ok("1b: organize is idempotent")

# ── 2. single-file mode: media + context still on the input side ──────
inp, out, rfh = _fresh()
_touch(inp / "RJY" / "voice.wav")
_touch(inp / "RJY" / "README.txt")
_touch(out / "RJY" / "voice.srt")
_touch(out / "RJY" / "voice.zh.srt")
_touch(out / "RJY" / "REVIEW-voice.zh.md")

if not organize_unit("RJY", inp, out):
    fail("2: organize_unit returned False")
if not (out / "RJY" / FINAL_DIR / "voice.wav").is_file():
    fail("2: media not moved from input/ into final/")
if not (out / "RJY" / REVIEW_DIR / "README.txt").is_file():
    fail("2: context not moved from input/ into review/")
if (inp / "RJY").exists():
    fail("2: input/RJY not cleaned up")
ok("2: single-file mode — input side moved into final/review, input/<unit> pruned")

# ── 3. root-level file unit (input/foo.wav, output/foo.*) ─────────────
inp, out, rfh = _fresh()
_touch(inp / "foo.wav")
_touch(out / "foo.srt")
_touch(out / "foo.zh.srt")
_touch(out / "foo.zh.pre-review.srt")
_touch(out / "foo.adjudication.json")
_touch(out / "REVIEW-foo.zh.md")

units = discover_units(inp, out)
if units != ["foo"]:
    fail(f"3: discover_units = {units}, expected ['foo']")
organize_all(units, inp, out)
if not (out / "foo" / FINAL_DIR / "foo.wav").is_file():
    fail("3: root-file media not in final/")
if not (out / "foo" / REVIEW_DIR / "REVIEW-foo.zh.md").is_file():
    fail("3: REVIEW report not in review/")
if not (out / "foo" / REVIEW_DIR / "foo.adjudication.json").is_file():
    fail("3: adjudication json not in review/")
if (inp / "foo.wav").exists():
    fail("3: input root file not moved")
if list(out.glob("foo.*")):
    fail("3: loose output/foo.* files remain")
ok("3: root-level file unit collected from both sides into output/foo/")

# ── 4. incomplete unit is refused (media without a translated SRT) ────
inp, out, rfh = _fresh()
_touch(out / "RJZ" / "a.wav")
_touch(out / "RJZ" / "a.srt")
_touch(out / "RJZ" / "a.zh.srt")
_touch(out / "RJZ" / "b.wav")
_touch(out / "RJZ" / "b.srt")   # b has no translation yet

if organize_unit("RJZ", inp, out):
    fail("4: incomplete unit was organized")
if (out / "RJZ" / FINAL_DIR).exists():
    fail("4: final/ created for an incomplete unit")
ok("4: incomplete unit (media without .zh.srt) left untouched")

# a.zh.srt for b arrives → now complete
_touch(out / "RJZ" / "b.zh.srt")
if not organize_unit("RJZ", inp, out):
    fail("4b: completed unit still refused")
ok("4b: unit organizes once the missing translation appears")

# similar-stem media must not satisfy each other's translation check
inp, out, rfh = _fresh()
_touch(out / "RJP" / "foo.wav")
_touch(out / "RJP" / "foobar.wav")
_touch(out / "RJP" / "foobar.srt")
_touch(out / "RJP" / "foobar.zh.srt")
if organize_unit("RJP", inp, out):
    fail("4c: foo.wav (no translation) passed via foobar's zh.srt")
ok("4c: prefix-similar stems don't cross-satisfy the completeness check")

# ── 5. archive: move + zip into ready_for_human_review ────────────────
inp, out, rfh = _fresh()
u = out / "RJA"
_touch(u / "t.wav")
_touch(u / "t.srt")
_touch(u / "t.zh.srt")
_touch(u / "REVIEW-t.zh.md")
organize_unit("RJA", inp, out)

if archivable_units(out) != ["RJA"]:
    fail(f"5: archivable_units = {archivable_units(out)}")
dest = archive_unit("RJA", out, rfh)
if not (dest / FINAL_DIR / "t.zh.srt").is_file():
    fail("5: archived folder incomplete")
if (out / "RJA").exists():
    fail("5: unit left behind in output/")
zip_path = rfh / "RJA.zip"
if not zip_path.is_file():
    fail("5: zip missing")
names = zipfile.ZipFile(zip_path).namelist()
if f"RJA/{FINAL_DIR}/t.zh.srt" not in names or f"RJA/{REVIEW_DIR}/REVIEW-t.zh.md" not in names:
    fail(f"5: zip content wrong: {names}")
ok("5: archive → ready/<unit>/ + ready/<unit>.zip (full content, unit-rooted)")

try:
    archive_unit("RJA", out, rfh)
    fail("5b: second archive did not raise")
except FileNotFoundError:
    ok("5b: archiving a gone unit raises")

# refuses to overwrite an existing archive
u = out / "RJA"
_touch(u / FINAL_DIR / "t.zh.srt")
try:
    archive_unit("RJA", out, rfh)
    fail("5c: overwrote an existing archive")
except FileExistsError:
    ok("5c: existing ready/<unit> refuses overwrite")

# ── 6. clean_title strips stacked batch prefixes (repeat-until-stable) ─
if clean_title("01.【PV】両耳ささやき音声") != "両耳ささやき音声":
    fail(f"6: clean_title('01.【PV】両耳ささやき音声') = {clean_title('01.【PV】両耳ささやき音声')!r}")
if clean_title("track01") != "":
    fail("6: 'track01' should strip to nothing")
ok("6: clean_title repeats prefix strips until stable")

print("\nALL ORGANIZE CHECKS PASS")
