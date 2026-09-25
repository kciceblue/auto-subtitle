# Wording trials toward seven — 2026-09-25

User authorization (this session): test hypotheses recursively until a seven;
translation stays local; development scoring may use Claude (Opus 5.5) as the
reviewer; ask before any download of 3 GB or more.

## Development reviewer

Blind fresh subagents receive the unchanged v5 rubric text, the common source
SRT, the candidate and a compact context: the v5 context with the 933 acoustic
observations reduced to 860 unique readings per window (`scripts/self_review.py`).
Prompt files carry neutral IDs outside the repository. Calibration on seven
Astra-scored candidates, three reviews each (`output/self-review/calib-1/`):

| Candidate | Astra | Self-review |
|---|---|---|
| Human reference | 7, 7 | 8, 7, 8 |
| C1 | 7, 7 | 7, 7, 7 |
| C2 | 6, 6 | 6, 7, 6 |
| LC | 6, 6 | 6, 6, 6 |
| C3 | 6, 6 | 6, 6, 6 |
| UD | 6, 6 | 6, 6, 6 |
| Fresh Qwen | 5, 5 | 5, 5, 5 |

The ordering and rounded means match Astra on all seven. Decision rule: a
development candidate is a *likely seven* only when all three self-reviews give
`quality_score` ≥ 7; it then receives two Astra v5 reviews on the common basis
before any claim. Development self-review is not a v5 receipt.

## Diagnosis

Six blind-to-score analysts compared LC with the reference per four-minute
segment using the Japanese evidence (`output/self-review/diagnosis-lc-vs-reference.json`):
164 issues, severity weight 262, about 90 % judged avoidable from the Japanese
and its alternative readings alone. Seven of ten severity-3 issues are source
selection: lines or lyrics the primary ASR garbled or missed while alternative
readings carry them. The rest is literal wording, register, pragmatics (speaker,
addressee, point of view, sentence-final force) and entity consistency.
Canon-only items (for example 静久 for 雫) are about 10 % of weight.

This diagnosis used the reference and therefore informs which generic
procedures to build; no reference wording, issue text or location enters any
local request. Rules stay generic. A seven here is development evidence; a
held-out episode is needed before any production claim.

## Levers (local only)

- **L1 source reconciliation:** per window, a local model reconciles the primary
  ASR with the alternative readings (with crop times). It may only select or
  splice attested readings, may add lines that the primary missed, and marks
  lyric/non-verbal/tic units. Character-coverage guard against invention.
- **L4 entity/tic ledger:** one episode-level pass clusters name variants (ASR
  mishearings of one person), nicknames and a character's verbal tic, and fixes
  one Chinese form for each.
- **L2+L3 translation:** per scene with reasoning on, from the reconciled
  Japanese, the ledger, and the previous scene's final Chinese; the old draft is
  only a fallible reference. Generic rules for subtitle register, speaker and
  addressee, point of view and sentence-final force.
- **Selection:** multiple local variants with a local fidelity/naturalness judge.

Each stage is scored as it lands; every candidate and score is recorded,
including failures. A blind strong-model rewrite of the same inputs may be run
as a non-deliverable diagnostic ceiling (never shipped: translation stays local).

## Capability check: does local Qwen reach good quality? (attempt 6)

User question: an earlier Astra judgment (under a since-replaced rubric) held that
the local Qwen model was capable of good-quality subtitles without fine-tuning.
Test it directly: Q-mirror gives `qwen3.8-27b-dflash` (reasoning on, thinking-off
fallback) exactly the O1 task — the same six part files (all ASR readings, slots,
LC draft) and the same instructions, output as `[slot] Chinese` lines — then builds
it with the same presentation and no extra cleanup. Scored by Astra ×2 and pairwise
against the human reference. Reading, pre-declared: Astra ≥7 in both reviews → the
claim holds and the multi-stage pipeline, not the model, was limiting; 6 → the gap
to O1 (Astra 7–8 on identical inputs) is the model on this task format, and a
follow-up separates comprehension from Chinese expression.
