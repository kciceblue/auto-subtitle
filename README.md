# auto-subtitle

English | [中文](README.zh-CN.md)

Fully local Japanese video/audio → Simplified-Chinese subtitles on one RTX 5090
(32 GB). No cloud API and no human input: several local speech recognizers collect
evidence, Gemma drafts the episode, and Qwen3.8-27B rewrites every line from all of it.

**Status (2026-09-25): this is as far as the local system gets without fine-tuning.**
The retained workflow scores **6/6/7** (overall/expression/fidelity) in both
independent [v5](QUALITY.md) reviews on the benchmark episode. The human fansub scores
7, which is "good"; 6 is "usable". The evidence is not the limit: a stronger writer
given the same inputs scores 7–8. The limit is the local model's Chinese expression.
Fine-tuning the local writer is the next step. See the
[ceiling record](design/local-ceiling-20260925.md).

```bash
./run.sh                         # every media file in input/ -> output/<stem>/final/
./run.sh input/episode.mkv       # one file
.venv/bin/python -m src.evidence_first input/episode.mkv --until draft   # stop after a stage
```

Output: `output/<stem>/final/<stem>.zh.srt`, plus the primary Japanese transcript
`<stem>.ja.srt`. Intermediate stages live in `output/<stem>/work/`. Every stage is
resumable, and every LLM request is cached.

## Pipeline

1. **Source**: Anime Whisper transcribes ~20 s windows.
2. **Evidence**: Zipformer, Qwen3-ASR (blind, short, BandIt-separated, auto, forced and
   masked views) and Voxtral add about 930 alternative readings per episode.
3. **Draft**: Gemma 4 31B QAT Q4 translates the whole episode in one request.
4. **Align**: Qwen3-ForcedAligner times the Japanese. The draft is split onto display
   units without text changes.
5. **Write**: Qwen3.8-27B with reasoning writes the final line for every slot. It sees
   twelve windows per request, with all readings, the Japanese and the draft.
6. **Build**: word-aligned cues, reading-speed timing, subtitle punctuation.

Models run one at a time on the GPU: ASR workers and the Gemma server are short-lived,
and the Warden Qwen model is restored afterwards. A 24-minute episode takes roughly
35–45 minutes. Stage details, required models and measurements are in
[WORKFLOW.md](WORKFLOW.md).

## Requirements

- Ubuntu with an NVIDIA GPU with 32 GB, `ffmpeg`, `.venv` from `requirements.txt`, and
  `.venv-voxtral` for Voxtral.
- A Warden/llama.cpp endpoint at `127.0.0.1:8089` serving `qwen3.8-27b-dflash`, and
  llama.cpp `llama-server` for the Gemma draft (`profiles/long-context-gemma.json`).
- Models under `models/` and `~/HF/asr-models/`, listed in [WORKFLOW.md](WORKFLOW.md).

## Quality and history

[QUALITY.md](QUALITY.md) defines the v5 rubric and the optional external scoring
command. Scoring is evaluation only; it never feeds the workflow. Every local
experiment since 2026-09-14 is recorded in [design/](design/). Previously committed
trial code remains in git history.

```bash
.venv/bin/python -m unittest tests.test_evidence_first tests.test_aligned_display
```
