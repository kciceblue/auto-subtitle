# auto-subtitle

[English](README.md) | 中文

在单张 RTX 5090（32 GB）上，全本地地把日语音视频做成简体中文字幕。不调用云端 API，
也不需要人工参与：先用多个本地语音识别模型收集证据，再由 Gemma 起草整集译文，
最后由 Qwen3.8-27B 参考全部证据逐句重写。

**现状（2026-09-25）：不做微调，本地系统已到上限。**
保留流程在基准集上两次独立 [v5](QUALITY.md) 评审均为 **6/6/7**（总分/表达/忠实）。
人工字幕组为 7 分，即“良好”；6 分为“可用”。证据本身足够：同样的输入交给更强的模型能拿到 7–8 分。
瓶颈在本地模型的中文表达。下一步是微调本地写作模型。
详见[上限记录](design/local-ceiling-20260925.md)。

```bash
./run.sh                         # 处理 input/ 下所有媒体 -> output/<文件名>/final/
./run.sh input/episode.mkv       # 处理单个文件
.venv/bin/python -m src.evidence_first input/episode.mkv --until draft   # 运行到指定阶段为止
```

输出为 `output/<文件名>/final/<文件名>.zh.srt`，另附日文主识别稿 `<文件名>.ja.srt`；
中间结果在 `output/<文件名>/work/`。每个阶段都可续跑，所有 LLM 请求都有缓存。

## 流程

1. **识别**：Anime Whisper 按约 20 秒的窗口识别。
2. **证据**：Zipformer、Qwen3-ASR（盲听、短片段、BandIt 分离、自动语言、强制日语、掩码等视角）
   和 Voxtral 共提供每集约 930 条备选读法。
3. **初稿**：Gemma 4 31B QAT Q4 一次翻译整集。
4. **对齐**：Qwen3-ForcedAligner 给日文打时间戳，初稿按显示单元拆分，文字不变。
5. **重写**：Qwen3.8-27B 开启推理，为每个字幕位写出最终译文。每次请求处理 12 个窗口，
   附带全部读法、日文和初稿。
6. **成稿**：按词对齐生成字幕条，控制阅读时长，并按字幕习惯处理标点。

GPU 上的模型依次运行：识别进程和 Gemma 服务都是临时启动，结束后恢复 Warden 的 Qwen 模型。
一集 24 分钟的动画约需 35–45 分钟。阶段细节、所需模型和实测数据见 [WORKFLOW.md](WORKFLOW.md)。

## 环境要求

- Ubuntu、32 GB NVIDIA 显卡、`ffmpeg`；按 `requirements.txt` 安装 `.venv`，
  Voxtral 需要 `.venv-voxtral`。
- `127.0.0.1:8089` 上的 Warden/llama.cpp 服务，提供 `qwen3.8-27b-dflash`；
  Gemma 初稿需要 llama.cpp 的 `llama-server`（见 `profiles/long-context-gemma.json`）。
- 模型放在 `models/` 和 `~/HF/asr-models/`，清单见 [WORKFLOW.md](WORKFLOW.md)。

## 质量与历史

[QUALITY.md](QUALITY.md) 规定了 v5 评分标准和可选的外部评分命令。评分只用于评估，
不会反馈给流程。自 2026-09-14 以来的全部本地实验记录在 [design/](design/)；
此前已提交的实验代码仍可在 git 历史中找到。

```bash
.venv/bin/python -m unittest tests.test_evidence_first tests.test_aligned_display
```
