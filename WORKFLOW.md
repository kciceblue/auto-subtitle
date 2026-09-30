# Retained local subtitle workflow: evidence-first rewrite

**Astra v5 6/6/7 in both reviews** (overall/expression/fidelity) on the episode-10
benchmark. The human fansub scores 7/7/7 and 7/7/8. Blind pairwise review puts this
recipe one step below the human reference (−1, −1). This is the best fully local result,
and the [local ceiling without fine-tuning](design/local-ceiling-20260925.md). No cloud API
and no human input are used. The score comes from one benchmark episode; source accuracy
and playback remain unverified.

```bash
./run.sh --check                 # preflight: is this machine ready? (a few seconds)
./run.sh                         # every media file in input/ -> output/<stem>/
./run.sh input/episode.mkv       # one file
.venv/bin/python -m src.evidence_first MEDIA --out DIR [--title TEXT] [--until STAGE] [--parts 1-12,13-24]
```

Results go to `output/<stem>/final/<stem>.zh.srt`. The primary Japanese transcript is
saved beside it as `<stem>.ja.srt`, with ~20 s window cues. Intermediate stages stay in
`output/<stem>/work/`. Each stage is resumable: a completed stage is verified and reused.
An interrupted source stage is moved aside as `source.failed-<time>`, then redone.

## Run it

1. `./run.sh --check`. Every line should say `ok`. Each failure prints its fix. `./run.sh`
   repeats this check before it starts (`SKIP_PREFLIGHT=1` skips it).
2. `./run.sh input/episode.mkv`, or put media in `input/` and run `./run.sh`. The title
   given to the draft writer defaults to the file name without release tags; `--title`
   overrides it.
3. Wait about 35–45 minutes per 24-minute episode. Progress goes to stderr as
   `== <stage>` and `== <stage> done in <N>s`. For an unattended run, redirect to a log:
   `./run.sh input/episode.mkv > output/episode.log 2>&1`.
4. The episode is done when `output/<stem>/final/<stem>.zh.srt` exists and the command
   exits 0. `work/write/texts.json` lists any slot that fell back to the draft.

Rules while it runs: one run at a time, since every GPU stage needs the whole card. Leave
Warden running; the workflow unloads and restores it. Do not start other GPU jobs. To
stop a run, interrupt it (Ctrl-C). Do not kill its GPU worker processes on their own.

## When a run fails

Rerun the same command. Completed stages are verified and reused. The error message says
what to do. The usual cases:

| Message | Cause | Action |
|---|---|---|
| `Preflight failed` | A prerequisite is missing | Apply the `fix:` lines, then rerun |
| `Started acquisition cannot be rerun` / `Started short acquisition cannot be repeated` | An evidence pool was interrupted | Delete that pool's folder under `work/acoustic/` and rerun. Completed pools are kept. |
| `Incomplete <method> acquisition retained at DIR; delete it to redo` | An extra reading (auto/forced/masked/voxtral) was interrupted | Delete `DIR` and rerun |
| `Completed artifact changed: FILE (delete DIR to redo)` | A frozen output was edited | Delete `DIR` and every later stage, then rerun |
| `Frozen artifact changed: FILE` | The input changed after the stage started, for example a different `--title` | Rerun with the original input, or delete that stage folder and every later one |
| `... failed with exit N; see .../worker.log` | A GPU worker crashed | Read that `worker.log`. Out-of-memory usually means another GPU process; run `./run.sh --check`. |
| Preflight `draft port ... in use` | A Gemma llama-server was left behind by an interrupted draft | `ss -ltnp "sport = :18101"`, stop that llama-server, rerun |

Later stages only check that their own output exists. To redo a stage, delete its output
and the output of every later stage:

| Stage | Output under `output/<stem>/work/` |
|---|---|
| source | `source/` |
| evidence | `acoustic/`, `extras/`, `extra-specs/`, `acquisition-plan.json`, `base-observations.json`, `base-complete.json`, `extras-complete.json` |
| draft | `draft/` |
| align | `align/` |
| pieces | `pieces/` |
| write | `write/` (the LLM answers are cached in `write/cache.json`) |
| build | rewritten on every run into `output/<stem>/final/` |

If a crash leaves Warden without its model, nothing needs doing: Warden loads
`qwen3.8-27b-dflash` again on the next request. If Warden itself is down, run
`systemctl --user start model-warden`.

## Stages

| Stage | What runs | Output |
|---|---|---|
| source | `ffmpeg` 16 kHz mono; RMS windows of about 18–25 s cover every frame; a short-lived Anime Whisper worker transcribes them (greedy, then bounded seeded retries for rejected windows) | `work/source/` |
| evidence | Zipformer and Qwen3-ASR read blind, short and BandIt-separated crops. Qwen3-ASR auto-language, forced-Japanese and masked views and Voxtral read each window. About 930 readings per episode. | `work/acoustic/`, `work/extras/` |
| draft | Gemma 4 31B QAT Q4 (`profiles/long-context-gemma.json`) on a temporary llama-server drafts the whole episode in one request from the primary Japanese and the base readings | `work/draft/` |
| align | Qwen3-ForcedAligner-0.6B word timing of each window's primary Japanese | `work/align/` |
| pieces | Japanese display units (sentence, pause and length cuts; implausible timings repaired). The draft of each window is partitioned onto the units by local Qwen, text unchanged. | `work/pieces/` |
| write | **Qwen3.8-27B, reasoning on**, gets twelve windows per request with every reading, each slot's Japanese and its draft. It writes the final Chinese for every slot. One retry covers unanswered slots; a slot left unanswered falls back to its draft. | `work/write/` |
| build | Each slot's Chinese is placed on its aligned units. Cue timing gets reading-speed minimums and lead-out. `。，；：` become spaces. | `final/` |

GPU use is sequential on one 32 GB card. ASR workers run as short-lived processes with the
Warden LLM unloaded; Warden is restored afterwards. The Gemma draft runs on a temporary
llama-server, and the previous Warden model is restored when it exits. Qwen stages use
Warden at `127.0.0.1:8089` (`qwen3.8-27b-dflash`). Every LLM request is cached per stage, so
a rerun replays finished requests.

Measured on the 24-minute episode 10:

| Stage | Time |
|---|---|
| Source ASR | 13 s |
| Base evidence | 5.4 min |
| Extra readings | 6 min |
| Writer | 12 min |

The Gemma draft and the Qwen partition take several minutes each. Plan on roughly
35–45 minutes and about 2–3 GB of disk per episode.

## Required local assets

`./run.sh --check` verifies all of these.

- Project directories:
  - `models/anime-whisper`
  - `models/bandit-v2` (with its own `runtime/`, named in `manifest.json`)
  - `models/gemma4-31b-qat-q4`
  - `models/voxtral-mini-4b-realtime-2602`
- In `~/HF/asr-models/`:
  - `Qwen3-ASR-1.7B`
  - `Qwen3-ForcedAligner-0.6B`
  - `sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01`
- Runtimes (CUDA 12.8 wheels):
  - `.venv`: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128`
  - `.venv-voxtral`: the same with `requirements-voxtral.txt`. The Voxtral reader requires its exact pins.
  - `ffmpeg` and `ffprobe` on `PATH`
  - llama.cpp `llama-server`, at the path in `profiles/long-context-gemma.json`
  - model-warden (`systemctl --user start model-warden`) serving `qwen3.8-27b-dflash` on `127.0.0.1:8089`

## Known behaviour

- A reasoning request can exhaust its 32k budget. The HTTP client then retries with
  thinking off. On episode 10 this happened for one of six writer requests.
- Windows without primary Japanese keep the draft only when the draft has text for them.
- The writer is told to use only the evidence, not outside knowledge of the series. Names
  follow the attested readings, so canonical spellings can differ from fansubs.

## Validation

- Offline tests (no GPU, models or network), from the repository root:
  `.venv/bin/python -m unittest $(git ls-files 'tests/test_*.py' | grep -v voxtral | sed 's#/#.#; s#\.py$##')`
  and `.venv-voxtral/bin/python -m unittest tests.test_voxtral_source_native`.
- Replaying the committed stages on the cached episode-10 inputs reproduces the Astra-scored
  subtitles byte for byte: part files, all 280 slot texts, and both SRTs.
- A fresh end-to-end run on a 3-minute clip of episode 10 passed every stage in 5.2
  minutes: evidence 143 s, draft 21 s, write 134 s. All 43 slots were answered and Warden
  was restored.

Experiment history and the removed trial code are recorded in [design/](design/README.md),
starting from [local-ceiling-20260925.md](design/local-ceiling-20260925.md).
