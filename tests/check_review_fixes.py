"""Offline checks for the review/proofread gate fixes (no LLM calls).

Covers the round-1 C2/C3 fixes and the round-2 NEW-1..NEW-10 fixes.
Run: python3 tests/check_review_fixes.py
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.proofread as proofread_mod
import src.review as review_mod
from src.proofread import proofread_srt
from src.review import _parse_review_response, _reject_fix, review_srt
from src.config import TranslateConfig
from src.translate import (
    AllChunksFailedError,
    SrtBlock,
    has_numbered_markers,
    make_snapshot,
    parse_response,
    write_translated_srt,
)

TMP = Path("/tmp/check_review_fixes")


def _write_srt(path: Path, texts: list[str]) -> None:
    """Write a minimal well-formed SRT with one block per text."""
    parts = []
    for i, text in enumerate(texts, start=1):
        start, end = i - 1, i
        parts.append(
            f"{i}\n00:00:{start:02d},000 --> 00:00:{end:02d},000\n{text}\n"
        )
    path.write_text("\n".join(parts), encoding="utf-8")


def _fresh_tmp() -> Path:
    import shutil as _shutil
    if TMP.exists():
        _shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    return TMP


def _config(**kw) -> TranslateConfig:
    base: dict = dict(
        endpoint="http://127.0.0.1:9/unused",
        source_lang="Japanese",
        target_lang="Simplified Chinese",
        review_chunk_size=8,
    )
    base.update(kw)
    return TranslateConfig(**base)


# ── round-1: parser ───────────────────────────────────────────────────

# 1. chunk-relative valid lines parse
r = _parse_review_response("[1] 修改|新的译文内容\n[3] 存疑|源文疑似ASR错误", 8, "T1")
assert r == {1: ("fix", "新的译文内容"), 3: ("flag", "源文疑似ASR错误")}, r

# 2. answer in the old global index space -> out of range, warned, zero results
assert _parse_review_response("[147] 修改|全局编号的答案", 8, "T2") == {}

# 3. prose / refusal -> zero results (caller must count this as a failed chunk)
assert _parse_review_response("这一组字幕整体没有问题。", 8, "T3") == {}

# 4. mixed: valid line applied, junk + out-of-range warned
r = _parse_review_response("[2] 修改|好的译文\n[5] 这行没有语法\n[99] 存疑|越界", 8, "T4")
assert r == {2: ("fix", "好的译文")}, r

# 5. context markers are never parsed as results
r = _parse_review_response("[前1] 修改|上下文行\n[2] 修改|正常修正\n[后1] 存疑|上下文", 8, "T5")
assert r == {2: ("fix", "正常修正")}, r

print("OK  round-1 parser: chunk-relative space, out-of-range + junk warned")


# ── NEW-2: anchored meta-marker matching ──────────────────────────────

# Legitimate Chinese subtitle text that merely CONTAINS 正确 / 保持 / 存疑.
legit = [
    ("你说得完全正确。", "你说的完全对。"),
    ("请保持安静。", "请安静点。"),
    ("我对此存疑。", "我怀疑这一点。"),
    ("这个答案不正确", "这个答案错了"),
    ("保持这个姿势别动", "别动保持住"),
]
for value, old in legit:
    reason = _reject_fix(value, old)
    assert reason is None, f"NEW-2 false positive: {value!r} -> {reason}"

# Genuine meta-commentary must still be refused.
meta = [
    "无需修改",
    "该行正确，保持原样",
    "修改|重复标记",
    "本行译文正确",
    "保持原样",
    "存疑",
    "无需修改。",
    "[3] 修改|回声",
]
for value in meta:
    reason = _reject_fix(value, "原来的译文内容")
    assert reason is not None, f"NEW-2 meta value slipped through: {value!r}"

print("OK  NEW-2  anchored meta check: 5 legit values pass, 8 meta values rejected")


# ── NEW-3: source-anchored length bounds ──────────────────────────────
# The bounds are measured against the SOURCE line (n = len(src)), not the
# current translation. An over-translated line is long precisely because it is
# wrong, so it cannot be the yardstick for its own correction; the source says
# how much was actually said. With n <= 3 (an interjection) only the absolute
# ceiling applies — あっ may legitimately render as 啊 or as 啊，等一下.
#   n <= 3 : reject len(new) > 20
#   n >  3 : reject len(new) > max(3n, 20) or len(new) < max(1, min(0.2n, n-8))

# (a) short source, short translation, slightly expanded fix
case1 = _reject_fix("嗯，好的。", "嗯", "嗯")
assert case1 is None, f"NEW-3 (a) should PASS, got {case1}"

# (b) over-translation fix: long translation, SHORT source -> 1-char fix
case2 = _reject_fix("嗯", "我真的觉得非常舒服啊啊啊啊", "あっ")
assert case2 is None, f"NEW-3 (b) should PASS, got {case2}"

# (c) runaway expansion
case3 = _reject_fix("好" * 40, "五个字译文", "気持ちいいですね")
assert case3 is not None, f"NEW-3 (c) should FAIL, got {case3}"

# (d) dropped translation: long source collapsed to a single char
case4 = _reject_fix("好", "这是一句相当长的原始译文内容", "ゆっくり息を吸って吐いてください")
assert case4 is not None, f"NEW-3 (d) should FAIL, got {case4}"

# The discrimination (b) vs (d) buys us: same 'old', opposite verdicts,
# decided purely by how much the source line actually said.
long_old = "我真的觉得非常舒服啊啊啊啊"
assert _reject_fix("嗯", long_old, "あっ") is None
assert _reject_fix("嗯", long_old, "本当に気持ちよくてたまらないんです") is not None

# The absolute expansion allowance really is absolute, not a ratio.
assert _reject_fix("啊" * 20, "啊", "あっ") is None, "NEW-3: 20-char expansion"
assert _reject_fix("啊" * 21, "啊", "あっ") is not None, "NEW-3: 21 exceeds the cap"
# Without a source line the guard falls back to the old translation length.
assert _reject_fix("好", "这是一句相当长的原始译文内容") is not None
# Unchanged behaviour for ordinary same-length edits.
assert _reject_fix("这是一句合理的新译文", "这是一句原来的译文", "これは元の翻訳です") is None
assert _reject_fix("", "原来的译文", "元の文") is not None

print("OK  NEW-3  source-anchored bounds: (a) PASS (b) PASS (c) FAIL (d) FAIL")
print(f"    (c) {case3}")
print(f"    (d) {case4}")


# ── NEW-1: markerless response never reaches the subtitle ─────────────

assert has_numbered_markers("[1] 译文") is True
assert has_numbered_markers("很抱歉，我无法协助处理这些内容。") is False
# With the fallback disabled, prose yields nothing at all.
assert parse_response(
    "很抱歉，我无法协助处理这些内容。", 3, allow_positional_fallback=False
) == ["", "", ""]
# The translation pass keeps the fallback (a split-to-1 chunk relies on it).
assert parse_response("译好的一行", 1)[0] == "译好的一行"

tmp = _fresh_tmp()
src_p, tr_p = tmp / "main.srt", tmp / "main.zh.srt"
_write_srt(src_p, ["ソース一", "ソース二"])
_write_srt(tr_p, ["旧译文一", "旧译文二"])
before = tr_p.read_text(encoding="utf-8")

proofread_mod.call_llm = lambda *a, **k: "很抱歉，我无法协助处理这些内容。"
try:
    proofread_srt(src_p, tr_p, _config())
except AllChunksFailedError as e:
    print(f"OK  NEW-1  markerless proofread response -> AllChunksFailedError: {e}")
else:
    raise AssertionError("NEW-1: markerless response did not fail the chunk")
assert tr_p.read_text(encoding="utf-8") == before, "NEW-1: prose reached the subtitle!"
assert "无法协助" not in tr_p.read_text(encoding="utf-8")
assert not (tmp / "main.zh.pre-proofread.srt").exists(), \
    "NEW-1: aborted pass must not leave a snapshot"
print("OK  NEW-1  subtitle untouched, no prose written, no snapshot on abort")


# ── NEW-6: all-failed review raises instead of reporting success ──────

tmp = _fresh_tmp()
src_p, tr_p = tmp / "main.srt", tmp / "main.zh.srt"
_write_srt(src_p, ["ソース一", "ソース二"])
_write_srt(tr_p, ["旧译文一", "旧译文二"])
before = tr_p.read_text(encoding="utf-8")

review_mod.call_llm = lambda *a, **k: "这一组字幕整体没有问题。"
try:
    review_srt(src_p, tr_p, _config())
except AllChunksFailedError as e:
    print(f"OK  NEW-6  all-failed review -> AllChunksFailedError: {e}")
else:
    raise AssertionError("NEW-6: all-failed review returned normally")
assert tr_p.read_text(encoding="utf-8") == before, "NEW-6: file rewritten after total failure"


# ── NEW-4: a rejected 修改 is flagged 存疑, not silently released ──────

tmp = _fresh_tmp()
src_p, tr_p = tmp / "main.srt", tmp / "main.zh.srt"
_write_srt(src_p, ["ソース一", "ソース二"])
_write_srt(tr_p, ["旧译文一", "旧译文二"])

review_mod.call_llm = lambda *a, **k: "[1] 修改|无需修改\n[2] 修改|新的第二行译文"
report = review_srt(src_p, tr_p, _config())
report_text = report.read_text(encoding="utf-8")

assert "#1 [存疑]" in report_text, f"NEW-4: rejected fix not flagged 存疑\n{report_text}"
assert "元标记" in report_text, "NEW-4: rejection reason missing from the report"
assert "修改被拒绝 1 处" in report_text, "NEW-4: rejected count missing from the header"
assert "#2 [修改]" in report_text, "NEW-4: the valid fix was not applied"
final = tr_p.read_text(encoding="utf-8")
assert "旧译文一" in final, "NEW-4: rejected value overwrote line 1"
assert "新的第二行译文" in final, "NEW-4: accepted value not written to line 2"
print("OK  NEW-4  rejected 修改 -> 存疑 flag in report, count in header, line kept")


# ── NEW-5: the first snapshot wins across repeated runs ───────────────

snap = tmp / "main.zh.pre-review.srt"
assert snap.exists(), "NEW-5: first run did not snapshot"
assert "旧译文二" in snap.read_text(encoding="utf-8"), "NEW-5: snapshot is not the original"

review_mod.call_llm = lambda *a, **k: "[1] 修改|第二轮的新译文\n[2] 修改|第二轮第二行"
review_srt(src_p, tr_p, _config())
snap_text = snap.read_text(encoding="utf-8")
assert "旧译文一" in snap_text and "旧译文二" in snap_text, \
    f"NEW-5: second run clobbered the original snapshot\n{snap_text}"
assert "第二轮" not in snap_text, "NEW-5: snapshot contains post-review text"
print("OK  NEW-5  second review run preserved the FIRST pre-review snapshot")

# A snapshot that cannot be taken aborts the pass instead of overwriting.
tmp2 = _fresh_tmp() / "sub"
tmp2.mkdir(parents=True)
target = tmp2 / "x.srt"
target.write_text("ORIGINAL\n", encoding="utf-8")
first = make_snapshot(target, "pre-review")
target.write_text("MODIFIED\n", encoding="utf-8")
second = make_snapshot(target, "pre-review")
assert first == second and first.read_text(encoding="utf-8") == "ORIGINAL\n", \
    "NEW-5: make_snapshot overwrote an existing snapshot"
print("OK  NEW-5  make_snapshot is first-write-wins")


# ── NEW-7: the positional input file is no longer swallowed ───────────

from main import build_parser

parser = build_parser()
ns = parser.parse_args(
    ["pipeline", "-l", "ja", "--vocab", "vocab.txt",
     "--context", "README.txt", "input/movie.mkv"]
)
assert ns.input_file == Path("input/movie.mkv"), f"NEW-7: input_file={ns.input_file}"
assert ns.context == [Path("README.txt")], f"NEW-7: context={ns.context}"

ns2 = parser.parse_args(
    ["pipeline", "-l", "ja", "input/movie.mkv",
     "--context", "A.txt", "--context", "B.md", "--context-file", "C.txt"]
)
assert ns2.input_file == Path("input/movie.mkv")
assert ns2.context == [Path("A.txt"), Path("B.md")], ns2.context
assert ns2.context_file == [Path("C.txt")], ns2.context_file

ns3 = parser.parse_args(["pipeline", "-l", "ja"])
assert ns3.input_file is None and ns3.context is None
print("OK  NEW-7  --context binds one path per flag; positional binds to input_file")


# ── round-1 check 6: atomic write leaves the original intact ──────────

target = Path("/tmp/atomic_check.srt")
target.write_text("ORIGINAL\n", encoding="utf-8")


class Boom(SrtBlock):
    # dataclass __init__ is generated at SrtBlock creation time, so a
    # subclass __post_init__ is never invoked — arm inside __init__ instead.
    def __init__(self, index, ts_line, text):
        super().__init__(index, ts_line, text)  # stores _value via setter
        self._armed = True

    @property
    def text(self):
        if self._armed:
            raise OSError("disk full")
        return self.__dict__["_value"]

    @text.setter
    def text(self, value):
        self.__dict__["_value"] = value


write_translated_srt([SrtBlock(1, "00:00:00,000 --> 00:00:01,000", "新文本")], target)
assert "新文本" in target.read_text(encoding="utf-8")
try:
    write_translated_srt([Boom(1, "00:00:00,000 --> 00:00:01,000", "x")], target)
except OSError:
    pass
assert "新文本" in target.read_text(encoding="utf-8"), "target was clobbered"
assert not Path(str(target) + ".tmp").exists(), "tmp file left behind"
target.unlink()
print("OK  round-1 atomic write: original intact, no .tmp left behind")

# ── v2 Phase-0: _COMMENT_LINE_RE must accept the "行号:" literal prefix ─

from src.review import _parse_comment_items

# Critic-R3 实测丢线索的形态：模型把格式模板里的「行号:」当字面量输出
c, oor = _parse_comment_items("行号: 105: 源文疑似听错（そばゆい 非日语词）", 200, "T-COMMENT-1")
assert c.get(105) and "そばゆい" in c[105][0], f"T-COMMENT-1: 行号: prefix not parsed: {c}"
assert not oor, f"T-COMMENT-1: unexpected out_of_range: {oor}"
print("OK  T-COMMENT-1  '行号: 105: …' literal-prefix comment parsed")

# 原有各形态不能回归
c, oor = _parse_comment_items(
    "12: 语义冲突\n第7行: 风格不符\n[42]: 漏译\n#8: 术语漂移\n- 101: 越界行", 100, "T-COMMENT-2"
)
assert set(c) == {12, 7, 42, 8}, f"T-COMMENT-2: legacy shapes broken: {c}"
assert oor == [101], f"T-COMMENT-2: out_of_range wrong: {oor}"
print("OK  T-COMMENT-2  legacy shapes (12: / 第7行: / [42]: / #8: / bullet) still parse")

# 混合形态：行号前缀 + 普通编号共存
c, oor = _parse_comment_items("行号: 105: 意见A\n行号: 12: 意见B", 100, "T-COMMENT-3")
assert set(c) == {12}, f"T-COMMENT-3: 105 应越界 (max 100): {c}"
assert oor == [105], f"T-COMMENT-3: out_of_range wrong: {oor}"
print("OK  T-COMMENT-3  mixed literal-prefix comments, out-of-range handled")

# 普通散文（非注释）不得被误判
c, oor = _parse_comment_items("行号 105 的意见是源文有问题", 200, "T-COMMENT-4")
assert not c and not oor, f"T-COMMENT-4: prose misread as comment: {c} {oor}"
print("OK  T-COMMENT-4  prose without colon is not a comment")

print("\nALL OFFLINE CHECKS PASS")
