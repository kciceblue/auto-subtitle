# Agent guide

Fully local Japanese media → Simplified-Chinese subtitles on one RTX 5090 (32 GB), with no
cloud API and no human input. There is one workflow, `src/evidence_first.py`. This page is
enough to run it; [WORKFLOW.md](WORKFLOW.md) has the stage details.

## Run

```bash
./run.sh --check               # preflight, a few seconds; every line must say ok, and each failure prints its fix
./run.sh input/episode.mkv     # one file -> output/<stem>/final/<stem>.zh.srt (+ <stem>.ja.srt)
./run.sh                       # every media file under input/
```

- One 24-minute episode takes 35–45 minutes. Run it in the background with output
  redirected to a log, e.g. `./run.sh input/x.mkv > output/x.log 2>&1`, and poll the log
  for `== <stage> done`. The episode is done when the command exits 0 and the `.zh.srt`
  exists.
- To resume after any failure or interruption, rerun the same command. Completed stages are
  verified and reused, and LLM answers are cached.
- If it fails, the error says what to do. [WORKFLOW.md § When a run fails](WORKFLOW.md#when-a-run-fails)
  lists every case, and says which folders to delete to redo a stage.
- `--until STAGE` stops after `source`, `evidence`, `draft`, `align`, `pieces`, `write` or
  `build`. `--title TEXT` sets the background title given to the draft writer.

## Rules

- **One GPU job at a time.** Every GPU stage needs the whole card. Do not start other GPU
  work during a run, and do not kill its worker processes. Leave the Warden LLM gateway
  (`127.0.0.1:8089`) running: the workflow unloads and restores it.
- **Public repository.** Never commit subtitles, reference translations, media, or
  anything under `input/`, `output/` or `models/`. Commit or push only when the user asks.
- Ask before any download of 3 GB or more.
- Completed stage outputs under `output/<stem>/work/` are frozen and hash-checked. Never
  edit them. To redo a stage, delete its folder and every later stage's folder.
- Do not change `CACHE_VERSION = 'wording-pipeline-1'` in `src/evidence_first.py`. The
  cached benchmark runs replay only under that key.
- GPU workers pin their own source hashes. Editing `src/late_audio.py`,
  `src/selected_asr.py`, `src/fresh_source_extras.py` or the modules they pin makes a
  half-finished evidence stage refuse to resume. Finish runs before changing that code.

## Stages

```
media ─[source]   Anime Whisper over ~20 s RMS windows (short-lived worker)        work/source/
      ─[evidence] Zipformer + Qwen3-ASR on blind/short/BandIt crops (late_audio, short_audio);
                  Qwen3-ASR auto/forced/masked + Voxtral (fresh_source_extras)     work/acoustic/, work/extras/
      ─[draft]    whole-episode Gemma 4 31B draft on a temporary llama-server      work/draft/
      ─[align]    Qwen3-ForcedAligner (aligned_display --align)                    work/align/
      ─[pieces]   Japanese display units + exact partition of the draft (Qwen)     work/pieces/
      ─[write]    Qwen3.8-27B, thinking on, rewrites every slot, 12 windows/request work/write/
      ─[build]    word-aligned cues; 。，；： become spaces                          final/<stem>.zh.srt
```

ASR workers unload Warden and restore it afterwards (`late_audio._gpu_lifecycle`). The Gemma
draft does the same through `local_backend.temporary_local_writer`.

## Code map

- `src/evidence_first.py`: stages and CLI. `src/preflight.py`: the readiness check.
- `src/translate.py` (`call_llm`): all LLM HTTP. An empty reasoning response is retried
  with thinking off.
- `src/late_audio.py`, `src/short_audio.py`, `src/fresh_source_extras.py`,
  `src/selected_asr.py`: ASR evidence workers. `src/local_backend.py`: the temporary Gemma
  llama-server. `src/aligned_display.py`: alignment, display units and cue placement.
- The other ~60 `src/` modules are the import-coupled acquisition stack. Every one is
  reachable from `src/evidence_first.py`.
- `scripts/review_subtitle_quality.py`: optional external v5 scoring, for evaluation only
  ([QUALITY.md](QUALITY.md)).
- `design/`: experiment history. Start at [design/README.md](design/README.md). None of it
  is needed to run the workflow.

## Conventions

- Type hints, `pathlib.Path`, `logging` rather than print, explicit UTF-8, SRT timestamps as
  `HH:MM:SS,mmm`.
- A completed stage is verified and reused, never overwritten. A failure keeps its evidence.
  `late_audio` creates files with O_EXCL and mode 0444, not a temp file plus rename.
- Committed tests are whitelisted in `.gitignore` (`tests/*` is local by default). Add new
  test files there.

## Test

Offline (no GPU, models or network), from the repository root:

```bash
.venv/bin/python -m unittest $(git ls-files 'tests/test_*.py' | grep -v voxtral | sed 's#/#.#; s#\.py$##')
.venv-voxtral/bin/python -m unittest tests.test_voxtral_source_native
```
