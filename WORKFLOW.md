# Retained local subtitle workflow

**Selected: round 93, contextual score 3 in both independent reviews.**
This selection retains the simplest reproducible workflow among the top-scoring
candidates. It is a best-available baseline, not a passing four-point result.
No translation or ASR was regenerated during the rubric revision and selection.

This page describes the retained production baseline. The subsequently authorized
local quality experiments are tracked in [current results](design/quality-current-results-20260916.md).
They have not qualified a replacement. Astra is an optional final benchmark; local
generation and repairs must work without it.

## Processing

1. `ffmpeg` extracts original 16 kHz mono PCM. No vocal separation in this recipe.
2. RMS-based windowing assigns every audio frame to contiguous roughly 18–25
   second windows. The selected episode has 66 windows. A generic final tail
   above 25 seconds is split without dropping frames; the original 66-window
   geometry is unchanged. Frame coverage does not prove all speech was recognized.
3. A short-lived local Anime Whisper worker transcribes windows in batches of
   eight. Greedy decoding comes first; only structurally rejected windows receive
   bounded seeded retries, up to three attempts. No background text is sent to ASR.
4. Gemma 4 31B QAT Q4 receives all selected Japanese units plus the complete
   original context in one logical translation request. Exact IDs preserve source
   ownership. The prompt preserves ambiguity and dialogue fragments. Native
   capacity is checked against 32K; output is capped at 16,384 tokens including a
   1,536-token reasoning budget. No logical translation retries; the existing
   HTTP client's same-attempt compatibility fallback remains available.
5. Restore local Qwen3.8 27B DFlash for display layout. Every source and Chinese
   character must remain unchanged across splits. Timing is based on coarse
   windows, not fresh word alignment. The formatter retains its two configured
   retries and guarded fallback; it cannot turn uncertain source text into truth.
6. Save source and target SRTs, semantic units, display maps, model/code/input
   identities and execution receipts. Any independent benchmark is a separate
   explicit operation. No external reviewer feedback enters these steps.

The pipeline, ASR and formatter use the single GPU sequentially. Only owned model
processes may be stopped. It verifies the original Qwen profile before unloading
and restores it after the temporary worker exits. A failure preserves its receipt
and cannot silently overwrite a prior run.

## Current evidence

| Candidate | Local method | Old v3 | Revised v4 |
|---|---|---:|---:|
| **93 (retained)** | Anime ASR + whole-source Gemma Q4 | 3 | **3; confirmation 3** |
| 108 | Whole Japanese + native English auxiliary evidence | 3 | 3 |
| 85 | Qwen Q8 evidence-based editing of an earlier draft | 3 | 3 |
| 94 | Vocal-separated Anime ASR + HY-MT per-unit translation | 2 | 2 |
| 24 | Gemma critic-edit of an earlier draft | 3 | 2 |

The new rubric accepts source-supported fragments and logic jumps; it does not
require Chinese subtitles to read like standalone prose. All five reviews use
complete source/Chinese/context bundles, with no audio or video. Different
workflows have different source transcripts, so this is not a controlled
writer-only comparison or an audio fidelity benchmark. No candidate reached four.

The retained output has 383 paired cues. Structural validation passes, with 13
Chinese cues longer than ten seconds (and the same 13 source-cue warnings).
These require playback review. Its confirmed score is 3, so the contextual gate
correctly remains closed. The actual failing assessment is saved alongside it.

The earlier measured producing commands total **150.88 seconds** for the
**23 min 40 sec** episode: native ASR command 19.11 seconds, translation/layout
command 131.77 seconds. This excludes extraction/setup/downloads and independent
review; it is not a fresh end-to-end measurement of the consolidated runner.
Within the recorded translation stage, writing took about 67.50 seconds and
layout 40.78 seconds. The full-source translation is the largest measured stage.

The consolidated code passed focused offline checks and mechanical parity
validation. Permanent runtime preflight checks actual model files and versions.
No new generation was run to claim bit-identical GPU output or new throughput.
The evidence covers one episode; generalization to other media is unmeasured.

## Artifacts and recovery

- Active profile: `profiles/selected-local.json`; clean backend recipe:
  `profiles/selected-writer.json`.
- Selected subtitles and original media: `output/selected/final/`.
- Bound assessment: `output/selected/review/contextual-assessment.json`.
- Current benchmark, candidates and native review receipts:
  `docs/benchmarks/contextual-selection-20260914/`.
- Retained research trials: `trials/README.md` (LC, GT, CS, AM, V and BT).
- Model inventory: `models/README.md`; exact cleanup audit: `design/cleanup-best-six-20260916/`.

The 2026-09-16 cleanup permanently removed unused model downloads, environments
and trial artifacts. Six score-3 research candidates and their shared evidence,
producer code and required historical validation records remain at their original
paths. The selected production output, original media and current reference
bundles are preserved. Historical aggregate reports retain results for deleted
trials; links to removed raw artifacts are historical rather than runnable.
Shared production modules remain because the selected runtime imports them.

Further changes should start from this single retained recipe and a separately
specified hypothesis. The archived repair attempts remain historical. New authorized experiments must
preserve their own inputs and results; a better cached-input score alone cannot
establish fresh-media accuracy or justify replacing the selected recipe.
