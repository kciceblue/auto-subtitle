"""三模型音频仲裁层（Audio Adjudication Layer）— v1.3 落地实现。

对可疑行切片，用 zipformer-ja-reazonspeech（CPU，独立先验）和
qwen3-asr-1.7b（GPU，LLM 架构补完式）复听，与 whisper 主 ASR 三方
投票（读音归一化层），产出 adjudication.json 供 DeepFix 使用。

分级（v1.3，读音层优先）：
  A     三模型音字全一致            → 源文可信
  A-音  三模型读音一致、字面不同    → 读音确认，汉字交 README/LLM 裁决
  A-    W 与 Q 音一致 ≠ N          → 现状高置信（N 独立反对记录备查）
  B+    N==Q 音一致 ≠ W            → 双独立反对，强修正信号
  B-    2:1 多数（如 W==N ≠ Q）    → 多数可能重合错，交 LLM 语义裁决
  C     三向音全分歧               → 只准存疑
  D     第二/三模型不可用          → 降级
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── 模型路径（冒烟实测确认）───────────────────────────────────────────
ZIPFORMER_DIR = Path.home() / "HF/asr-models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01"
QWEN3_DIR = Path.home() / "HF/asr-models/Qwen3-ASR-1.7B"

_PAD_S = 0.2          # 切片前后 padding
_MIN_SEG_S = 3.0      # 过短的行直接跳过复听（疑似 VAD 噪声行）


@dataclass
class Adjudication:
    line: int
    w: str                       # whisper 原文
    n: str = ""                  # zipformer
    q: str = ""                  # qwen3
    kana_w: str = ""
    kana_n: str = ""
    kana_q: str = ""
    grade: str = "D"             # A/A-音/A-/B+/B-/C/D
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "line": self.line, "w": self.w, "n": self.n, "q": self.q,
            "kana": [self.kana_w, self.kana_n, self.kana_q],
            "grade": self.grade, "note": self.note,
        }


# ── 读音归一化（pykakasi，汉字→平假名）──────────────────────────────

_kakasi = None


def _get_kakasi():
    global _kakasi
    if _kakasi is None:
        import pykakasi
        _kakasi = pykakasi.kakasi()
    return _kakasi


def kana_normalize(text: str) -> str:
    """汉字→平假名，去标点/空白/长音符/促音，统一小写假名。

    同音不同字（気筒/祈祷/キト → きとう）在此层归一。
    """
    if not text:
        return ""
    try:
        items = _get_kakasi().convert(text)
        out = "".join(it["hira"] for it in items)
    except Exception:
        out = text
    out = out.lower()
    out = re.sub(r"[\s、。，!！?？「」『』・…\-—,.·()（）:：;；]", "", out)
    out = out.replace("ー", "").replace("っ", "つ").replace("ぁ", "あ").replace("ぃ", "い")\
             .replace("ぅ", "う").replace("ぇ", "え").replace("ぉ", "お")\
             .replace("ゃ", "や").replace("ゅ", "ゆ").replace("ょ", "よ")\
             .replace("ゎ", "わ").replace("ゕ", "か").replace("ゖ", "け")
    return out


# ── 音频切片 ─────────────────────────────────────────────────────────

def load_full_wav(media: Path) -> tuple[int, np.ndarray]:
    """媒体 → 16kHz mono float32（一次 ffmpeg，内存切片）。"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = Path(f.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(media),
             "-ar", "16000", "-ac", "1", str(wav_path)],
            check=True, capture_output=True)
        with wave.open(str(wav_path), "rb") as w:
            sr = w.getframerate()
            data = np.frombuffer(w.readframes(w.getnframes()),
                                 dtype=np.int16).astype(np.float32) / 32768.0
        return sr, data
    finally:
        wav_path.unlink(missing_ok=True)


def _slice(sr: int, full: np.ndarray, start: float, end: float) -> np.ndarray:
    s = max(0, int((start - _PAD_S) * sr))
    e = min(len(full), int((end + _PAD_S) * sr))
    return full[s:e]


# ── 第二/三模型识别器 ───────────────────────────────────────────────

class ZipformerN:
    """N: zipformer-ja-reazonspeech（CPU，电视节目先验）。"""

    def __init__(self, model_dir: Path = ZIPFORMER_DIR, num_threads: int = 8):
        import sherpa_onnx
        self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(model_dir / "encoder-epoch-99-avg-1.int8.onnx"),
            decoder=str(model_dir / "decoder-epoch-99-avg-1.int8.onnx"),
            joiner=str(model_dir / "joiner-epoch-99-avg-1.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=num_threads,
        )
        logger.info("zipformer-N loaded (CPU)")

    def transcribe(self, sr: int, audio: np.ndarray) -> str:
        if len(audio) < sr * 1.0:
            return ""
        s = self._rec.create_stream()
        s.accept_waveform(sr, audio)
        self._rec.decode_stream(s)
        return s.result.text.strip()


class Qwen3Q:
    """Q: qwen3-asr-1.7b（GPU，LLM 架构补完式）。"""

    def __init__(self, model_dir: Path = QWEN3_DIR):
        import torch
        from src.warden import (
            ADJUDICATE_Q_HARD_FLOOR_GB,
            ADJUDICATE_Q_MIN_FREE_GB,
            ensure_gpu_headroom,
        )
        # qwen3-asr (~3.7 GB) and the warden-resident 27B dflash (~28 GB)
        # cannot share the 32 GB card: evict the LLM first when headroom is
        # short. Same policy as whisper's ASR load (src/asr.py) — the
        # 8/30 OOM happened exactly here (dflash was resident when this
        # class called from_pretrained(device_map="cuda")).
        ensure_gpu_headroom(
            required_gb=ADJUDICATE_Q_MIN_FREE_GB,
            hard_floor_gb=ADJUDICATE_Q_HARD_FLOOR_GB,
            caller="audio arbitration (qwen3-asr)",
        )
        from qwen_asr import Qwen3ASRModel
        self._torch = torch
        self._asr = Qwen3ASRModel.from_pretrained(
            str(model_dir), dtype=torch.float16, device_map="cuda")
        logger.info("qwen3-Q loaded (GPU)")

    def transcribe(self, sr: int, audio: np.ndarray) -> str:
        if len(audio) < sr * 1.0:
            return ""
        try:
            out = self._asr.transcribe((audio, sr), language="Japanese")
            return out[0].text.strip() if out else ""
        except Exception as e:
            logger.warning("qwen3 transcribe error: %s", e)
            return ""


# ── 投票分级（v1.3 读音层逻辑）──────────────────────────────────────

def grade_vote(w: str, n: str, q: str, kana_w: str, kana_n: str, kana_q: str) -> tuple[str, str]:
    """返回 (grade, note)。"""
    kw, kn, kq = kana_w, kana_n, kana_q
    if not n and not q:
        return "D", "第二/三模型不可用"
    if kw and kw == kn == kq:
        if w == n == q:
            return "A", "三模型一致"
        return "A-音", f"读音一致 {kw}，字面不同（W:{w} / N:{n} / Q:{q}）"
    if kn and kq and kn == kq and kn != kw:
        return "B+", f"双独立反对 W（N==Q: {n} | {q}）"
    if kw and kn and kw == kn and kw != kq:
        return "B-", "2:1 多数（W==N），多数可能重合错，交 LLM 语义裁决"
    if kw and kq and kw == kq and kw != kn:
        return "A-", "W 与 Q 一致（N 独立反对，记录备查）"
    return "C", "三向读音分歧"


# ── 主入口 ──────────────────────────────────────────────────────────

def adjudicate(
    source_srt: Path,
    media: Path,
    suspicious: list[int],
    out_json: Path,
    use_q: bool = True,
    max_workers_note: str = "",
) -> Path:
    """对可疑行做三模型复听，写 adjudication.json。

    Returns: 输出 JSON 路径。
    """
    sys_path = Path(__file__).parent
    import sys
    if str(sys_path) not in sys.path:
        sys.path.insert(0, str(sys_path))
    from src.translate import parse_srt

    source_srt = Path(source_srt)
    media = Path(media)
    out_json = Path(out_json)
    blocks = parse_srt(source_srt)
    n_blocks = len(blocks)

    def ts_sec(ts: str) -> float:
        h, mi, rest = ts.split(":")
        s, ms = rest.split(",")
        return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms) / 1000

    # 音频一次转 wav
    logger.info("Loading full audio: %s", media)
    t0 = time.monotonic()
    sr, full = load_full_wav(media)
    logger.info("Audio loaded: %.1f min (%.1fs)", len(full) / sr / 60, time.monotonic() - t0)

    # N（CPU）常驻
    n_model = ZipformerN()
    # Q（GPU）可选
    q_model = Qwen3Q() if use_q else None

    # Clip to real line numbers up front ("all" arrives as 1..99999) so the
    # progress denominator reflects actual work.
    suspicious = [ln for ln in suspicious if 1 <= ln <= n_blocks]

    results: list[Adjudication] = []
    t_all = time.monotonic()
    for i, ln in enumerate(suspicious):
        b = blocks[ln - 1]
        a, bb = b.ts_line.split("-->")
        start, end = ts_sec(a.strip()), ts_sec(bb.strip())
        if end - start < 0.3:
            continue
        audio = _slice(sr, full, start, end)
        ad = Adjudication(line=ln, w=b.text.strip())
        ad.n = n_model.transcribe(sr, audio)
        if q_model is not None:
            ad.q = q_model.transcribe(sr, audio)
        if not ad.n and not ad.q:
            ad.grade, ad.note = "D", "复听无输出"
        else:
            ad.kana_w = kana_normalize(ad.w)
            ad.kana_n = kana_normalize(ad.n)
            ad.kana_q = kana_normalize(ad.q)
            ad.grade, ad.note = grade_vote(ad.w, ad.n, ad.q, ad.kana_w, ad.kana_n, ad.kana_q)
        results.append(ad)
        if (i + 1) % 50 == 0:
            logger.info("Adjudicated %d/%d (%.0fs)",
                        i + 1, len(suspicious), time.monotonic() - t_all)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=1),
        encoding="utf-8")

    grades: dict[str, int] = {}
    for r in results:
        grades[r.grade] = grades.get(r.grade, 0) + 1
    logger.info(
        "Adjudication complete: %d lines in %.0fs; grades=%s",
        len(results), time.monotonic() - t_all, grades)
    return out_json


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--srt", required=True, type=Path)
    p.add_argument("--media", required=True, type=Path)
    p.add_argument("--suspicious", required=True, type=str,
                   help="逗号分隔行号或 'all'（全行）")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--no-q", action="store_true", help="跳过 qwen3（仅 N）")
    args = p.parse_args()

    sus = list(range(1, 100000)) if args.suspicious == "all" else \
        [int(x) for x in args.suspicious.split(",") if x.strip()]
    out = adjudicate(args.srt, args.media, sus, args.out, use_q=not args.no_q)
    print(f"Written: {out}")
