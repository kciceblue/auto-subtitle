"""Full-pipeline review smoke test: script ground-truth penetration.

Scenario A (B- audio + script mismatch): line 2 must be DeepFixed toward
デニス (script authority) — the zh line must lose the "网球" mistranslation.
Scenario B (A-grade audio vs script conflict): line 2 must NOT be auto-fixed;
the report must list the hard conflict instead.
"""
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from src.config import TranslateConfig
from src.review import review_srt

ENDPOINT = "http://127.0.0.1:8089/v1/chat/completions"
PAYLOAD = {"model": "qwen3.8-27b-dflash", "temperature": 0.3}

SRC = """1
00:00:01,000 --> 00:00:02,000
ここはとある田舎の病院。

2
00:00:03,000 --> 00:00:06,000
そうだね。彼の名前はテニスくん。

3
00:00:07,000 --> 00:00:08,000
それは大変ですね。
"""
ZH = """1
00:00:01,000 --> 00:00:02,000
这是一所乡下的医院。

2
00:00:03,000 --> 00:00:06,000
是啊。他的名字叫网球君。

3
00:00:07,000 --> 00:00:08,000
那真是辛苦了呢。
"""
SCRIPT = """第1話
（田舎の病院）
A：そうだね。彼の名前はデニスくん。
"""

ADJ_B = json.dumps([
    {"line": 1, "grade": "A", "note": "一致", "n": "ここはとある田舎の病院。"},
    {"line": 2, "grade": "B-", "note": "多数派テニスだが Q 反对 デニス", "n": "テニス", "q": "デニス"},
    {"line": 3, "grade": "A", "note": "一致", "n": "それは大変ですね。"},
], ensure_ascii=False)

ADJ_A = json.dumps([
    {"line": 1, "grade": "A", "note": "一致", "n": "ここはとある田舎の病院。"},
    {"line": 2, "grade": "A", "note": "三モデル一致 テニス", "n": "テニス"},
    {"line": 3, "grade": "A", "note": "一致", "n": "それは大変ですね。"},
], ensure_ascii=False)


def run(scenario: str, adj_json: str, expect_fix: bool) -> None:
    work = Path(f"/tmp/review_script_smoke_{scenario}")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / "main.srt").write_text(SRC, encoding="utf-8")
    (work / "main.zh.srt").write_text(ZH, encoding="utf-8")
    (work / "台本.txt").write_text(SCRIPT, encoding="utf-8")
    (work / "adj.json").write_text(adj_json, encoding="utf-8")

    cfg = TranslateConfig(
        output_srt=work / "main.zh.srt",
        endpoint=ENDPOINT,
        source_lang="ja",
        target_lang="zh",
        chunk_size=1000,
        review_chunk_size=8,
        script_file=work / "台本.txt",
        extra_payload=PAYLOAD,
    )
    print(f"\n{'='*70}\nSCENARIO {scenario}\n{'='*70}")
    report = review_srt(
        source_path=work / "main.srt",
        translated_path=work / "main.zh.srt",
        config=cfg,
        adjudication=work / "adj.json",
    )
    final = (work / "main.zh.srt").read_text(encoding="utf-8")
    print("FINAL ZH:\n", final)
    print("REPORT (script section + flags):")
    rep = report.read_text(encoding="utf-8")
    for line in rep.splitlines():
        if "台本" in line or "硬冲突" in line or "#2" in line or "修改" in line:
            print("  ", line)
    line2_zh = [l for l in final.splitlines() if "网球" in l or "デニス" in l or "尼斯" in l or "名字" in l]
    print("LINE2 evidence:", line2_zh)
    if expect_fix:
        assert not any("网球" in l for l in final.splitlines()), \
            "scenario A: 网球 must be gone after script-anchored DeepFix"
        print("  ✓ 网球 fixed")
    else:
        assert any("网球" in l for l in final.splitlines()), \
            "scenario B: A-grade conflict must NOT auto-fix"
        assert "硬冲突" in rep, "report must list the hard conflict"
        print("  ✓ conflict preserved + reported")


run("A", ADJ_B, expect_fix=True)
run("B", ADJ_A, expect_fix=False)
print("\nALL SMOKE SCENARIOS PASSED")
