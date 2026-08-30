# auto-subtitle

English | [中文](README.zh-CN.md)

Fully local video/audio → translated subtitle pipeline: faster-whisper ASR →
chunked LLM translation → proofread → three-model audio arbitration →
evidence-gated semantic review → organized per-work output.

No cloud services: ASR runs on your GPU, translation/review go through any
local OpenAI-compatible chat-completions endpoint (default
`http://127.0.0.1:8089/v1/chat/completions`).

## Pipeline

```
media ─ ffmpeg 16 kHz WAV
      ─ demucs vocal separation            (optional, on by default)
      ─ context-aware hotword generation   (title + nearby synopsis/script files)
      ─ faster-whisper large-v3 ASR (VAD)  → source .srt
      ─ numbered-chunk LLM translation     → <stem>.<lang>.srt
      ─ whole-file proofread pass          (optional)
      ─ three-model audio arbitration      (whisper + zipformer CPU + qwen3-asr GPU)
      ─ semantic review                    (script > reference SRT > audio evidence;
                                            unresolvable lines flagged, never guessed)
      ─ organize                           → output/<unit>/{final,review}/
```

## Requirements

- Linux, NVIDIA GPU (single 32 GB card is enough; ASR falls back to CPU)
- `ffmpeg` on PATH
- Python 3.12 venv: `pip install -r requirements.txt`
- A local OpenAI-compatible LLM endpoint for translation/review

## Usage

```bash
./run.sh                          # batch: process everything under input/
./run.sh input/foo.wav            # single file
PROOFREAD=0 ADJUDICATE=0 REVIEW=0 ORGANIZE=0 NO_DEMUCS=1 ./run.sh   # stage toggles
CONTEXT="synopsis.pdf script.txt" ./run.sh   # explicit context (default: auto content scan)
./run.sh archive                  # manual hand-off: finished units → ready_for_human_review/
```

Lower-level subcommands:

```bash
python main.py {transcribe,translate,pipeline,review,organize,archive} -h
```

## Output layout

Each work unit (top-level entry under `input/`) ends up as:

```
output/<unit>/
├── final/    media + source .srt + translated .srt
└── review/   context files, adjudication JSON, REVIEW-*.md report, snapshots
```

`./run.sh archive` moves a finished unit (plus a zip of it) into
`ready_for_human_review/`.

## Tests

```bash
python3 tests/check_review_fixes.py         # offline parser/gate checks
python3 tests/check_organize.py             # offline output-layout checks
python3 tests/test_script_review_smoke.py   # needs the LLM endpoint
```

## License

MIT — see [LICENSE](LICENSE).
