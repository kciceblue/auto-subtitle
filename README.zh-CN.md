# auto-subtitle

[English](README.md) | 中文

完全本地化的视频/音频 → 翻译字幕流水线：faster-whisper 语音识别 →
分块 LLM 翻译 → 全文校对 → 三模型音频仲裁 → 证据门控语义终审 →
按作品整理输出。

无任何云服务依赖：ASR 在本地 GPU 上运行，翻译/审校走任意本地
OpenAI 兼容 chat-completions 端点（默认
`http://127.0.0.1:8089/v1/chat/completions`）。

## 流水线

```
媒体 ─ ffmpeg 16 kHz WAV
     ─ demucs 人声分离                  （可选，默认开启）
     ─ 内容感知热词生成                  （媒体标题 + 同目录简介/台本文件）
     ─ faster-whisper large-v3 ASR (VAD) → 源语言 .srt
     ─ [N] 编号分块 LLM 翻译            → <stem>.<lang>.srt
     ─ 全文校对 pass                    （可选）
     ─ 三模型音频仲裁                    （whisper + zipformer CPU + qwen3-asr GPU）
     ─ 语义终审                         （台本 > 参考字幕 > 音频证据分级；
                                          无法定夺的行只标记存疑，绝不瞎猜）
     ─ 整理归档                         → output/<unit>/{final,review}/
```

## 环境要求

- Linux + NVIDIA GPU（单张 32 GB 显卡足够；ASR 可回退 CPU）
- `ffmpeg` 在 PATH 上
- Python 3.12 虚拟环境：`pip install -r requirements.txt`
- 一个本地 OpenAI 兼容 LLM 端点（用于翻译/审校）

## 使用

```bash
./run.sh                          # 批量：处理 input/ 下所有媒体
./run.sh input/foo.wav            # 单文件
PROOFREAD=0 ADJUDICATE=0 REVIEW=0 ORGANIZE=0 NO_DEMUCS=1 ./run.sh   # 阶段开关
CONTEXT="synopsis.pdf script.txt" ./run.sh   # 显式 context（默认自动内容扫描）
./run.sh archive                  # 手动归档：完成的单元 → ready_for_human_review/
```

更底层的子命令：

```bash
python main.py {transcribe,translate,pipeline,review,organize,archive} -h
```

## 输出结构

每个作品单元（`input/` 下的顶层条目）最终整理为：

```
output/<unit>/
├── final/    媒体 + 源语言 .srt + 译文 .srt
└── review/   context 文件、仲裁 JSON、REVIEW-*.md 审校报告、快照
```

`./run.sh archive` 把完成的单元（连同一份 zip）移入
`ready_for_human_review/`。

## 测试

```bash
python3 tests/check_review_fixes.py         # 离线 parser/gate 检查
python3 tests/check_organize.py             # 离线输出布局检查
python3 tests/test_script_review_smoke.py   # 需要 LLM 端点
```

## 许可证

MIT — 见 [LICENSE](LICENSE)。
