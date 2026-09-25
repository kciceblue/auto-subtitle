# Wording trials: results — 2026-09-25

Design and authorization: [wording-trials-20260925.md](wording-trials-20260925.md).
Budget: 15 full attempts (a candidate built and scored); calibration and
diagnostics do not count. Translation stays local; Claude reviews only.

## Scores

Self = three blind Claude reviews (quality/expression/fidelity, compact bundle).
Pair = blind Claude pairwise margin versus the human reference (−3…+3, two
orders). Astra = two `gpt-6-astra` v5 reviews on the byte-identical common basis.

| # | Candidate | Self | Pair | Astra |
|---|---|---|---|---|
| — | Human reference | 8/8/7, 7/7/7, 8/8/8 | 0 (anchor) | 7/7/7, 7/7/8 |
| — | LC (previous best local) | 6/5/6, 6/5/6, 6/6/6 | −2, −2 | 6/6/7, 6/6/6 |
| 1 | O1 diagnostic ceiling: blind Claude rewrite of the same local evidence (never shipped) | 8/8/8 ×3 | +1, −1 | **7/7/8, 8/8/8** |
| 2 | S-g2: L1 reconcile + ledger + three local writers + Gemma line selection | 6/7/6, 7/7/6, 6/7/6 | −2, −1 | — |
| 3 | **F-g5**: S-g2 + guarded two-pass local fidelity repair | 7/7/7 ×3 | −1, −1 | **6/6/7, 6/6/7** |
| 4 | FN3: F-g5 locks + 10-candidate sampled pool, Gemma filter, Qwen pick, fidelity passes | 7/7/6, 6/7/6, 7/7/7 | −2, −1 | — |
| 5 | F-g5 recipe replayed on held-out episode 12 (below) | 5/6/5, 6/6/6, 5/6/5 | −2, −3 | **5/6/5, 5/6/5** |
| 6 | Q-mirror: local Qwen given exactly O1's inputs and instructions (capability check) | — | −1, −1 | **6/6/7, 6/6/7** |

Calibration on seven Astra-scored anchors matched Astra's ordering, but Claude
self-review is about one point more lenient at the six/seven boundary for the new
candidates (F-g5: self 7/7/7, Astra 6/6/7). Pairwise against the human reference
separates candidates that integer scores tie: all earlier Astra-six candidates
sit at −2, F-g5 at −1, O1 (Astra 7–8) at 0.

## Held-out episode 12 (attempt 5)

The user asked to keep the best recipe and test whether the gap is episode-specific.
All inputs were rebuilt for S01E12 with the unchanged episode-10 machinery
(`output/episode12-ref/`: fresh Anime Whisper source, 935 observations, LC draft)
and the F-g5 recipe was replayed end to end by `scripts/run_episode_recipe.sh 12`.
Two robustness fixes were needed and apply to any episode: a reasoning-mode call
that exhausts its retries (looping past the budget, or ending the turn without
content) is retried once with thinking off. The human reference is the fansub's
burned-in Chinese: RapidOCR on a local-network OCR service, then blind visual
verification of one crop per event (366 cues, 327 dialogue and 38 lyric).

| Episode 12 | Self | Pair | Astra |
|---|---|---|---|
| Human reference (verified OCR) | — | 0 (anchor) | **7/7/7, 7/7/7** |
| F-g5 recipe | 5/6/5, 6/6/6, 5/6/5 | −2, −3 | **5/6/5, 5/6/5** |

The gap is not episode-10-specific: on a held-out episode the recipe scores
lower (Astra 5) while the human bar is the same (7). Visible causes: harder ASR
(for example アオちゃん heard as アホちゃん, rendered as "idiot"), two wrong
ledger merges (a nickname for 蒼 merged into 羽依里; 稲荷 merged with 天善), and
ASR sentence-boundary errors that the writers turned into meaning errors. Rules
drawn from the episode-10 diagnosis did not carry the same gain.

## Capability check (attempt 6)

Given the identical task on which Claude's rewrite scored Astra 7/7/8 and 8/8/8 —
same six part files, same instructions, same presentation, no pipeline — local
Qwen3.8-27B (reasoning on) scored **6/6/7 in both Astra reviews**, pairwise −1, −1:
the same level as the best multi-stage recipe (F-g5). Fidelity is seven in both
cases; expression is six. So under v5 the local Qwen reaches "usable", with meaning
largely right, but not "good": its Chinese expression is the binding limit on this
material, and more pipeline stages around it have not moved that.

## Model-tier ladder (diagnostic, not counted as an attempt)

The identical O1 task (same part files, prompt, schema and effort; only the model
changes), built with the same presentation and scored by Astra ×2 on the common basis:

| Writer | Astra overall/expression/fidelity |
|---|---|
| Local Qwen3.8-27B (Q-mirror) | 6/6/7, 6/6/7 |
| Claude Haiku 4.5 | 6/6/7, 6/6/7 |
| Claude Sonnet 5 | 6/7/6, 7/7/7 (split) |
| Claude Opus 5.5 (O1) | 7/7/8, 8/8/8 |
| Human fansub | 7/7/7, 7/7/8 |

Quality rises with model tier, and the step to "good" sits between Sonnet 5
(borderline, one 6 and one 7) and Opus 5.5 (clearly above the human reference).
The local 27B performs like Haiku 4.5 on this task. Claude outputs here are
diagnostics only and never part of the local system.

## What each lever did

- **L1 source reconciliation** (local Qwen, attested readings only, two-family
  support filter for added lines) restored dropped dialogue and insert-song
  lyrics, fixed misheard lyrics, and rejected single-recognizer hallucinations.
- **L4 ledger** with a local same-person judge merged ASR name confusions
  (爱理→羽依里) while keeping distinct people apart (加藤≠加纳).
- **Scene translation with echo-verified alignment** fixed the line-drift bug and
  gave natural phrasing; three writers (Qwen ×2, Gemma with English gloss) plus
  Gemma selection brought expression to seven in every self-review.
- **Guarded fidelity passes** (Gemma check/repair, Qwen verify; name, no-revert and
  lyric guards) fixed the perspective error, the addressee error, a garbled lyric
  and an ASR-garbled idiom. Same-family second verification was a rubber stamp.
- **N-best sampling with local taste** changed 160 lines without net gain.

## Outcome (2026-09-25)

Six of fifteen attempts were used. The user then declared this the local ceiling without
fine-tuning ([record](local-ceiling-20260925.md)). Q-mirror became the retained workflow
(`src/evidence_first.py`): it ties F-g5 on Astra and pairwise review and is far simpler.
The trial runners and outputs were removed; this page and the design above are the record.

## Where it stood before the decision

Five of fifteen attempts used; no local candidate has reached Astra 7. The
evidence supports 7–8 (O1), so the binding limit is the local 27–31B writers'
Japanese comprehension and Chinese word choice, and the held-out episode shows
the recipe is not robust to harder ASR. The remaining levers are a stronger local
writer (needs a download decision) or treating the local result as a draft for
human post-editing.
