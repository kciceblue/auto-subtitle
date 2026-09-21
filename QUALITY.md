# Subtitle quality and validation

Current benchmark: **subtitle-quality-v4**. The working target is **four: usable
subtitles in the supplied dialogue context**. Subtitles are not standalone prose.
Source-supported fragments, ellipsis, repetitions, interjections, sound effects,
speaker changes and scene cuts are acceptable. Do not invent unseen visual
explanations to excuse a clear contradiction of the supplied source.

## Scale

| Score | Meaning |
|---|---|
| −10 to −1 | Increasingly worse than ordinary unreviewed machine translation. |
| 0 | Ordinary unreviewed machine translation; no vendor-specific baseline required. |
| 1–3 | Material translation-added confusion or errors still prevent usable quality. |
| **4** | Usable dialogue subtitles relative to the supplied transcript and original context; minor roughness and source-supported discontinuities are allowed. |
| 5 | Stronger naturalness and faithful expression relative to that evidence. |
| **6** | No detected errors after full independent source, completeness and playback validation; plain language is acceptable. |
| 8 | The same validated correctness plus polished professional human-quality expression. |
| 10 | Exceptional expression, nuance and presentation, with the same correctness requirement. |

Scores are ordinal judgments, not accuracy percentages. A score change caused by
revising the rubric is a **reevaluation of the same output**, not a generation
improvement. Historical v1–v3 assessments retain their original meaning. The
superseded v3 policy is preserved locally at
`docs/benchmarks/contextual-selection-20260914/historical-policy/QUALITY-v3.md`.

## Local completion and optional scoring

ASR, evidence collection, interpretation, diagnosis, writing and repair must all
run locally. Local completion must not require Astra, evaluator credentials,
previous scores or an external approval. Save the final output before optional
scoring; that score cannot modify the output or become a repair prompt.

The independent benchmark below measures a frozen result. A run without it is
**completed, unscored**, not failed and not a claimed four. A confirmed score for
one saved candidate also does not establish a consistently four-point workflow.
Current qualification requires fresh local runs, all frozen before their scores
are opened, followed by separate generalization checks. Experimental scalar
results and production defaults are recorded separately.

## Evidence and independence

Writing, ASR, diagnosis and repair stay local. The stronger evaluator is separate:
GPT-6 Astra (`gpt-6-astra`), or Fable 5.1 where a genuine validated adapter exists.
The current experiments use Astra only after a frozen local candidate. The
evidence-aware adapter receives the authorized original Japanese/context, fixed
736-observation transcript pool and generated Chinese. New acoustic observations
and local interpretations stay local. It receives no audio, video, human
reference, prior scores or workflow identities.

The experimental evidence-aware reviewer returns **only one integer score**. It
returns no confidence or reading-coverage attestation. The older contextual
adapter used for the selected-output release commands below also preserves pass,
whole-text-read and confidence metadata; those are not part of the experimental
scalar contract. Neither adapter supplies locations, categories, explanations or
replacement wording to translation, repair, retrieval or training. A score may
select a complete local workflow; it cannot teach that workflow episode-specific
repairs. Exact inputs, prompts, schemas, producers and genuine dispatches are
preserved with hashes. Complete input delivery does not measure reviewer
attention or establish a calibrated guarantee.

The Japanese transcript is unverified and may contain local repairs or recognition
errors. V4 contextual review checks translation against that evidence and has a
maximum score of **five**. Agreement with a wrong transcript can still pass this
scope. It does not establish audio truth, dialogue coverage, speaker identity,
subtitle synchronization or playback readability. Those require separate source
and playback validation before six. Minor timing warnings remain visible.

## Contextual milestone gate

1. Freeze the complete source/target/context bundle. Source and target must have
   matching cue IDs and timestamps; malformed or missing cues fail preparation.
2. Obtain an independent whole-bundle review with the fixed v4 contract.
3. Obtain a new confirmation dispatch on the same unchanged files after the first
   review completes. Both scores must be at least four. Use their minimum.
4. Bind both receipts and every final deliverable to the assessment. Check strict
   SRT structure, paired files, positive durations and absence of overlaps.
5. Edits to reviewed source, Chinese, context, receipts or final files invalidate
   that assessment. Review the newly frozen result again.

Short cues below 0.4 seconds and long cues above 10 seconds require playback
review; this text-only milestone does not certify their suitability. A contextual
milestone cannot authorize the source-verified release packaging command.

```bash
# Prepare a review locally; --execute explicitly invokes the separate evaluator.
.venv/bin/python scripts/review_contextual.py \
  --manifest /absolute/path/manifest.json --output-dir /absolute/path/new-review
# Repeat with --execute to send exactly the three prepared text inputs.

# Prepare an unapproved contextual assessment, then bind actual confirmed receipts.
.venv/bin/python main.py release output/selected \
  --prepare --scope contextual_subtitles --writer-model ACTUAL_LOCAL_MODEL
.venv/bin/python main.py release output/selected \
  --check --scope contextual_subtitles
```

Manifest version `contextual-review-bundle-v1` contains `source`, `target` and
`context`, each with a plain absolute `path` and exact `sha256`. The assessment's
`contextual_reviews` binds each final target to a manifest, primary/confirmation
directories and their receipt hashes. Templates are unapproved. Actual reviews
must be obtained; manually filling a passing scalar is insufficient.

Historical `target_coherence` v3 checks and `source_verified` release checks remain
available for existing records. The latter retains its stricter no-detected-error
requirements. Local model agreement or a successful pipeline run cannot approve
release.
