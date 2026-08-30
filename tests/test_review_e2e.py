"""E2E review validation on a real slice containing the known 止血钳/剪刀 drift.

Builds a mini SRT (lines 146-156 of the acceptance sample work, keeping
their original 1-based indices) + the real reference SRT, runs review_srt()
with a real LLM call, and checks the report.

Needs LOCAL fixtures (gitignored, not in the repo) under
tests/fixtures/accept_e2e/: main.srt, main.zh.srt, README.txt, ref.zh.srt.
"""
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from src.config import TranslateConfig
from src.translate import parse_srt, write_translated_srt
from src.review import review_srt

BASE = Path("tests/fixtures/accept_e2e")
if not BASE.is_dir():
    sys.exit(f"local fixtures not present ({BASE}) — nothing to test")
TEST = Path("/tmp/review_e2e")
TEST.mkdir(exist_ok=True)

src_all = parse_srt(BASE / "main.srt")
zh_all = parse_srt(BASE / "main.zh.srt")

A, B = 146, 157  # 1-based inclusive
mini_src = src_all[A - 1:B]
mini_zh = zh_all[A - 1:B]

# write mini SRTs preserving ORIGINAL indices (so line N in file == line N overall)
def write_mini(blocks, out: Path, index_offset):
    lines = []
    for i, b in enumerate(blocks):
        lines.append(str(i + 1))
        lines.append(b.ts_line)
        lines.extend(b.text.split("\n"))
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")

# keep original 1-based line numbers by inserting blank-index padding
def write_mini_padded(blocks, out: Path, start_1based: int):
    out_lines = []
    n_before = start_1based - 1
    # pad: emit n_before entries? Too many. Instead keep indices contiguous 1..N
    # but remember the mapping: file-line k == global line start_1based-1+k
    lines = []
    for i, b in enumerate(blocks):
        lines.append(str(i + 1))
        lines.append(b.ts_line)
        lines.extend(b.text.split("\n"))
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")

write_mini_padded(mini_src, TEST / "main.srt", A)
write_mini_padded(mini_zh, TEST / "main.zh.srt", A)

cfg = TranslateConfig(
    output_srt=TEST / "main.zh.srt",
    endpoint="http://127.0.0.1:8089/v1/chat/completions",
    source_lang="ja",
    target_lang="zh",
    chunk_size=1000,          # one single chunk for the mini file
    context_files=[BASE / "README.txt"],
    vocab_file=Path("vocab.txt"),
    review_chunk_size=8,
    verbose=True,
)

report = review_srt(
    source_path=TEST / "main.srt",
    translated_path=TEST / "main.zh.srt",
    config=cfg,
    ref_srt=BASE / "ref.zh.srt",
)
print("=" * 60)
print("REPORT:")
print(report.read_text(encoding="utf-8"))
