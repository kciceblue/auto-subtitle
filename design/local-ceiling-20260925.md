# Local ceiling without fine-tuning (2026-09-25)

**This is as far as the fully local, automatic system gets without fine-tuning.**
The retained recipe scores Astra v5 **6/6/7 in both reviews** (overall/expression/fidelity)
on the episode-10 benchmark. The human fansub scores 7. Blind pairwise review puts the
recipe one step below the human reference (−1, −1). Scores are "usable" (6), not "good"
(7). Every model runs locally. There is no cloud API and no human in the loop.

## Retained recipe: evidence-first rewrite

`python -m src.evidence_first MEDIA` (or `./run.sh`). [WORKFLOW.md](../WORKFLOW.md) has the stage details.

1. Anime Whisper transcribes ~20 s RMS windows; this is the primary Japanese.
2. Zipformer, Qwen3-ASR (blind, short, BandIt-separated, auto, forced and masked views)
   and Voxtral add ~930 alternative readings per episode.
3. Gemma 4 31B QAT Q4 drafts the whole episode in one request.
4. Qwen3-ForcedAligner times the Japanese. The draft is split onto Japanese display units
   without text changes.
5. **Qwen3.8-27B with reasoning on writes the final subtitle for every slot.** It works
   from all the readings plus the draft, twelve windows per request.
6. Word-aligned cues are built with the subtitle punctuation convention.

This is trial Q-mirror (attempt 6). It ties the best multi-stage recipe F-g5 on Astra
(6/6/7 ×2 for both) and on pairwise review (−1, −1 for both). It uses one writer call per
twelve windows instead of about ten stages, so it was kept as the simpler of the two.
Replaying the committed runner on the cached episode-10 inputs reproduces the scored
subtitles byte for byte.

## Why this is the ceiling

| Evidence | Result |
|---|---|
| Local experiment families since 2026-09-14 (source repair, long context, reasoning, self-critique, meaning audit, pair revision, register rewrite, reconciliation + ledger + selection + guarded repair, N-best) | No local candidate reached Astra 7 |
| Presentation controls ([results](presentation-controls-results-20260925.md)) | Human text in the machine layout: 7/7. Machine text in word-aligned layout: 6/6. Layout does not limit the score. |
| Capability check: same inputs and instructions, only the writer changes ([results](wording-trials-results-20260925.md)) | Opus 5.5: 7/7/8, 8/8/8. Local Qwen3.8-27B: 6/6/7 ×2. The evidence supports 7–8; the writer does not reach it. |
| Model-tier ladder on the same task | Qwen 27B ≈ Haiku 4.5 (6/6). Sonnet 5 split 6/7. Opus 5.5 7–8. The "good" threshold lies between Sonnet 5 and Opus 5.5. |
| Larger local models tried earlier (Qwen3.5-122B-A10B, DeepSeek-V4-Flash IQ3, Gemma Q8, GLM, HY-MT) | None beat the 27–31B writers |
| Held-out episode 12 (F-g5 recipe) | 5/6/5 ×2. The human fansub scores 7/7/7 ×2, so the gap is not specific to episode 10. |

The binding limit is the local writer's Chinese expression; the evidence is sufficient.
More pipeline stages around the same 27–31B models have not moved the score, and a
stronger local writer does not fit the 32 GB GPU. The remaining lever is to **fine-tune
the local writer on paired Japanese/Chinese subtitles**.

Limits of this claim:
- The retained recipe was scored on one episode.
- It was not replayed on episode 12; only F-g5 was.
- Source accuracy and playback remain unverified.
- Claude self-review ran about one point more lenient than Astra at the 6/7 boundary, so
  only Astra and pairwise results support claims.

## Next step: fine-tuning (not started)

- Estimated data: a pilot with 10–20k aligned line pairs; for a robust result, 40–100k
  pairs across 20+ series.
- One clean source already exists: 28 episodes of 葬送的芙莉莲 bilingual JPSC `.ass`
  subtitles on the local file server, about 9k pairs.
- Blocked on approval to download base weights: about 54 GB bf16 or 16 GB 4-bit.
- Evaluation will reuse the v5 scorer ([QUALITY.md](../QUALITY.md)) and the local benchmark basis below.

## Cleanup record

The user asked on 2026-09-25 to keep only the best approach and the trial records.

- **Kept in git.**
  - The evidence-first runner and the modules it imports. The acquisition stack is
    import-coupled, so 71 `src/` modules remain.
  - The v5 scorer: `scripts/review_subtitle_quality.py` with its contract and adapter.
  - Every `design/` record, `QUALITY.md` and the offline tests of the kept code.
- **Removed from the tree.**
  - 65 unused `src/` modules, all trial runners in `scripts/`, and their tests.
  - The old pipelines: `main.py`, `transcribe.py`, the round-93 selected runner profiles,
    `continue_phase23.sh`, `ready_for_human_review/`.
  - The `trials/` aliases.
  - Earlier commits still contain the previously tracked code. A **local-only** branch,
    `archive/trials-20260925`, holds the untracked trial code as it stood before cleanup.
    It is not pushed.
- **Removed locally.**
  - About 16 GB of trial runtime output under `output/`.
  - `vendor/` (the omnilingual ASR trial) and the stray `:memory:.ses`.
- **Kept locally (gitignored).** Everything here is evaluation data and must never be committed.
  - `docs/`
  - Media
  - Model directories and `.venv`/`.venv-voxtral`
  - `output/benchmarks/`: the hash-pinned v5 review bases for episodes 10 and 12, the human
    reference SRTs, the episode-12 and episode-13 media, and the Astra `results.json`
    receipts of scored candidates.
  - The retained recipe's episode-10 subtitles, in `output/<episode stem>/final/`.

Design records written before this cleanup name their original `output/...` paths; those
paths are historical. The v5 bases were moved as follows:

| Old path | New path |
|---|---|
| `output/reference-six-methods-20260924/comparison-v5/common/` | `output/benchmarks/ep10/common/` |
| `output/reference-six-methods-20260924/comparison-v5/cases/F/target.srt` | `output/benchmarks/ep10/human-reference.srt` |
| `output/episode12-ref/common/` | `output/benchmarks/ep12/common/` |
| `output/episode12-ref/reference/target.srt` | `output/benchmarks/ep12/human-reference.srt` |
