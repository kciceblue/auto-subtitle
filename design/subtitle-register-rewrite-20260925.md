# Subtitle-register rewrite (C3) — 2026-09-25

## Why

The presentation controls ([design](presentation-controls-20260925.md)) scored
C1 (human text, machine segmentation/timing) 7/7 and C2 (LC text, aligned
presentation) 6/6. Under the pre-declared table, presentation is not binding;
the gap is in the Chinese wording. Across 32 v5 reviews, overall has equalled
expression every time.

Earlier local "expression" passes were preserve-and-edit requests: two returned
every owner unchanged, one made two edits, and a whole-episode bilingual revision
only deleted honorifics. No attempt has asked a local model to write subtitle-
register Chinese while being free to rephrase.

A diagnostic comparison of LC with the reference on this episode (analysis only;
no reference wording enters any request) points to generic subtitle-craft gaps:
non-verbal sounds rendered as cues (laughter, sobs, breathing, humming),
transliterated honorifics and inconsistent forms for the same person, register
slips such as `您` between close friends, and stiff literal phrasing. Because
this rule selection was informed by the development episode, a result here is
development evidence only.

## Method

Input: C2's Japanese units and LC Chinese pieces (identical owner text, aligned
timing), so presentation is held at C2 and C2 (6/6) is the direct baseline.

One local Qwen3.8-27B (`qwen3.8-27b-dflash`, thinking off, temperature 0.3, the
pipeline's formatter settings) rewrites the Chinese in owner-ordered chunks of
about 40 pieces. Every request carries the complete LC episode draft as read-only
context (for voice and name consistency) and, for each piece, the common
Japanese ASR text of its unit(s) and the draft Chinese. It returns one `[N]` line
per piece. The instruction states generic professional rules only:

1. Natural, concise spoken Mandarin as in professional anime subtitles; no stiff
   or literal translationese.
2. Keep all meaning: intent, negation, who does what to whom, questions,
   uncertainty. Add nothing; drop no spoken content.
3. Non-verbal sounds (laughter, sobbing, breathing, gasps, hiccups, humming) are
   not subtitled: return an empty line. Meaningful interjections (a reply, a
   surprised question, a character's recurring verbal tic) stay, rendered
   consistently.
4. Do not transliterate honorifics (no 桑/酱); use natural Chinese address.
   Match register to the relationship.
5. One Chinese form per person, place or nickname across the episode; unify
   variant forms of the same referent in the draft.
6. The Japanese is fallible ASR. Where it looks garbled, keep the draft's meaning;
   do not invent. Song lyrics stay poetic lines.

No glossary, reference text, reviewer finding or known error location is used.
The instruction contains no episode-specific names or lines.

Deterministic guards per piece, else the draft piece is kept and counted: the
`[N]` answer must exist; an empty answer is allowed only when the draft piece is
made of interjection characters alone; lexical length must be within
0.4–1.8× the draft plus four characters; no kana. Missing IDs are retried once in
a chunk of their own. Cues are then built exactly as in C2 (speech-time placement,
reading minimum, lead-out, seven-second cap, `。，；：`→space).

## Evaluation and interpretation

Freeze before review. Two blind v5 reviews with the byte-identical common source
and context, no retries. C3 counts as N only when both reviews give
`quality_score` N; a split is inconclusive.

- 7+/7+: a generic register rewrite closes the development gap. Required next:
  the same rewrite on a held-out episode before any production claim.
- 6/6: register rules alone do not close the gap (with C1/C2, the remaining
  difference is finer wording or meaning quality than these rules address).
- 5 or less: the rewrite regresses; report and stop this direction.

Artifacts: `output/presentation-controls-20260925/C3/`. Stage `c3` in
`scripts/run_presentation_controls.py`; rewrite logic in `src/register_rewrite.py`.
