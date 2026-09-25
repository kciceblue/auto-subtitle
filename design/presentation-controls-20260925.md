# Presentation controls — 2026-09-25

## Why

Across all 28 v5 reviews of the episode-10 candidates, `quality_score` equalled
`expression_score` every time (the human calibration pair also matches). Twelve
reviews gave fidelity seven with overall six. The September 24 experiments
mostly changed meaning, so they could not move the binding dimension.

Every generated candidate shares a presentation the human reference lacks:
18–27 cues longer than ten seconds, 6–12 cues over 25 characters (max 73–98,
some merging several speakers), and full `。，` punctuation. The human reference
has two cues above ten seconds, none above 17 characters, and spaces instead of
`。，`. Root cause: the display step receives no word timings
(`utterance_tokens` are empty), so pause layout is disabled and timing follows
the ~20-second ASR windows; when `plan_parts` fails validation it silently keeps
the whole owner as one cue.

Hypothesis: presentation (segmentation, timing, punctuation convention) is a
material part of the expression gap between six and seven.

## Two controls, no wording changes

**C1 — human text, machine segmentation and timing (degraded control).** The
unchanged human reference (comparison case F, 343 cues) is first split into two
layers by a mechanical greedy interval partition: a cue that overlaps the last
primary cue goes to a concurrent layer. This yields exactly the six insert-song
cues at 288–331 s. Primary cues are regrouped into the same 67 owner windows by
midpoint, joined by one space (the reference's own separator), and passed
through the exact common formatter used for every generated candidate
(`src.fresh_display.format_candidate`, local Qwen layout). The six concurrent
lyric cues are then added back with their original timing, so C1 does not
interleave song lines into dialogue (a text-order defect LC never had). No words
or punctuation change. C1 is somewhat coarser than LC's own layout (see results).

**C2 — LC text, aligned presentation plus Chinese punctuation convention.** LC's
67 Chinese owner strings are kept. The common Japanese owner text is aligned to
its own window audio with the installed `Qwen3-ForcedAligner-0.6B`.
Japanese units end at sentence-final punctuation (or `…` followed by 0.3 s);
units over seven seconds split at `、`/`…` or 0.5 s pauses, never before a
particle or auxiliary and with at least three characters on each side.
The aligner's word tokens are 80 ms-quantized and often zero-length, so timing
plausibility is judged per unit: a run of units faster than ~16.7 characters/s
is spread over the gap between reliable neighbours; an owner whose characters
are mostly implausible (or which fails alignment) uses character-proportional
timing over its aligned speech extent. The local Qwen formatter partitions each
Chinese owner into one piece per unit under the existing exact-concatenation rule
(`pause_layout.INSTRUCTION`), two attempts; otherwise a deterministic monotone DP
aligns Chinese sentences to units (1:1 when counts match). Multi-unit pieces are
cut at the punctuation nearest any gap of 1 s or more and timed on speech time,
not wall time. Pieces longer than 20 visible characters split at internal
punctuation. Timing: non-overlap, then a reading minimum of max(0.8 s,
characters/9), then a 0.3 s lead-out into free time, then a seven-second display
cap. Finally `。，；：` become spaces (dropped next to `…`), edges are stripped,
and `？！…、` and every other character are kept in order. All fallbacks, repairs,
capped and below-minimum cues are counted in `C2/owners-audit.json`.

Neither control uses a glossary, honorific policy, name unification or deletion
of interjections. The human text enters C1 only as the evaluated control target;
it never enters C2 or any writer, aligner or formatter request for C2.

## Pre-generation review

An independent three-lens critique (correctness, design, real-data edge cases)
with one adversarial verifier per finding ran before C2 generation and before
any review preparation: 17 findings, 11 confirmed. The confirmed defects —
degenerate alignment timing, mid-phrase cuts, greedy fallback drift, silence-
spanning fallback pieces, unenforced minimum, stray ellipses, cached transport
errors, C1 lyric interleaving, missing interpretation rules and provenance —
were fixed as described above. The superseded unscored C1 draft is kept under
`superseded/`.

## Evaluation

Both targets are frozen before any review. Each receives two independent blind
`gpt-6-astra` v5 reviews with the byte-identical common source
(`86747512…`) and context (`a0c6c3c9…`) used by all fourteen reference-comparison
reviews and the September 24 experiments: four dispatches, primaries before
confirmations, no score-conditioned retries, neutral case aliases. All results
are reported, including disagreement.

Pre-declared interpretation, indexed by `quality_score`:

| C1 (human, machine segmentation/timing) | C2 (LC, aligned + `。，`→space) | Reading |
|---|---|---|
| 6 | 7 | Presentation is binding; the aligned bundle closes the gap for LC. |
| 6 | 6 | Machine segmentation/timing costs the human text a point, but LC also has wording limits. |
| 7 | 7 | The aligned bundle helps LC; the human text is robust to machine segmentation/timing. |
| 7 | 6 | Presentation is not the binding factor; the gap is in the Chinese wording. |

A control counts as N only when both reviews give N. A primary/confirmation split
is reported as such and makes that control inconclusive. Eight or more in both
reviews reads as the seven column, with exact values reported. Five or less in
both is reported exactly as worse than the six baseline (for C1: presentation
costs two or more points; for C2: a regression) and maps to no row. Expression
and fidelity are reported alongside; any expression/overall divergence is noted.

## Limitations

- C1 degrades segmentation and timing only; C2 changes segmentation, timing and
  punctuation together, so a C2 result cannot attribute a gain among them.
- The reviewer is text-only. It sees cue length, granularity and timestamps, not
  whether timing matches speech; C2 does not validate alignment accuracy.
- C1's machine layout is harsher than LC's own (more long cues), which could
  slightly inflate a C1 drop.
- Integer scores on one episode are coarse. A seven from C2 is a text-review
  result, not evidence of audio fidelity, full coverage or playback readiness,
  and would need a held-out episode before any production claim.

Artifacts: `output/presentation-controls-20260925/`. Runner:
`scripts/run_presentation_controls.py`; layout logic: `src/aligned_display.py`;
tests: `tests/test_aligned_display.py`.
