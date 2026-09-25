# Subtitle quality and validation

Current scoring rubric: **subtitle-quality-v5**, scope **subtitle_text_quality**.
A quality rating describes the supplied subtitle text. Verification of the source,
coverage and playback is recorded separately. A text-only review can award **8–10**
when the observed quality warrants it; that number is not audio certification or
release approval. Human and machine translations use exactly the same rubric.

## Quality scale

| Score | Observed subtitle quality |
|---|---|
| 0 | No usable subtitle content. |
| 1–2 | Mostly unusable; severe, pervasive confusion or loss of meaning. |
| 3–4 | Some understandable content, but recurring material problems impair viewing. |
| 5 | Understandable overall; noticeable weaknesses require substantial editing. |
| 6 | Usable, with recurring roughness or several localized problems. |
| 7 | Good, generally natural and coherent; localized weaknesses limit the polish. |
| **8** | Professional-quality, natural, concise and context-appropriate; supported meaning and character voice are preserved. Isolated minor imperfections are allowed. |
| 9 | Excellent expression, nuance and dialogue voice; rare negligible weaknesses. |
| 10 | Exceptional, sustained precision and expressive craft. Rare but attainable. |

Scores are ordinal judgments, not accuracy percentages. Ordinary machine
translation has no fixed score; human authorship gives no automatic bonus.
An isolated typo does not make an otherwise strong episode unusable. Recurring
material meaning errors cannot earn eight merely because the Chinese is fluent.
Positive evidence of quality is required, not just absence of proven mistakes.

V5 changes the anchors as well as removing the evidence-dependent ceiling. **V3,
v4, the exploratory September 24 audit, and v5 numbers are not interchangeable.**
Do not multiply old scores, relabel receipts, or call a rubric-only score change a
translation improvement. The former policy is preserved in
[the v4 policy snapshot](design/subtitle-quality-v4.md).

## What the external reviewer returns

The v5 response contains five fields, validated by
[src/subtitle_quality_contract.py](src/subtitle_quality_contract.py):

| Field | Meaning |
|---|---|
| `quality_score` | Holistic quality of assessable subtitle text, 0–10. Provisional with respect to the original media. |
| `expression_score` | Target-language naturalness, dialogue voice, readability and continuity, 0–10. |
| `fidelity_to_evidence_score` | Meaning preservation against credible supplied source passages, 0–10; `null` when evidence is insufficient. This is not fidelity certified against the audio. |
| `whole_text_read` | The reviewer's self-report that it read every supplied source, target and context item. Must be `true`; otherwise the review is incomplete. |
| `confidence` | Certainty in this text assessment, 0–1. It is not the quality score. |

Do not average the dimension scores mechanically. When fidelity is unassessable,
`quality_score` describes the assessable target expression and confidence reflects
the limitation. Unknown evidence is not a zero. Complete input delivery and a
reading self-report do not prove attention to every cue or full-media coverage.

The local receipt always records `audio_reviewed`, `video_reviewed`,
`audio_source_fidelity_certified`, `full_media_coverage_verified`,
`playback_verified`, `release_gate_checked` and `overall_quality_certified` as
**false**. The reviewer cannot switch these on. For example, **text quality 8;
source/coverage/playback unverified** is valid. An unqualified claim of a verified
whole-episode eight is not supported by this command.

## Fair treatment of subtitles and evidence

Subtitles are dialogue, not standalone prose. Accept source-supported fragments,
ellipsis, repetitions, interjections, scene and speaker changes, idiomatic
condensation and natural rephrasing that preserve meaning. Check actions,
negation, intention versus outcome, relationships and nuance across neighboring
text and time; do not demand literal word matching.

Japanese ASR and locally repaired transcripts remain fallible. Agreement with a
wrong transcript does not establish accuracy. A transcript conflict alone does
not prove the translation wrong, and empty ASR does not prove silence. Supported
meaning errors affect quality; unresolved source conflicts affect certainty.
Do not invent unseen visual explanations to excuse a well-supported contradiction.
OCR defects and translation defects are also distinct. Verify suspect glyphs
against frames when available; preserve the actual subtitle wording.

Source and target are validated independently. They may have different cue counts
and timing. Preserve human segmentation; do not rewrite it to fit the pipeline's
ASR windows. Overlapping dialogue, lyrics and on-screen text may be legitimate
layers. Supply their roles in context and assess them separately; unreliable
lyric ASR is not evidence of a bad lyric translation. Overlap acceptance is a
text-ingestion rule, not proof that the rendered layout is readable.

## Local completion, independence and calibration

ASR, evidence collection, translation, diagnosis and repair remain local. Local
completion must not require an external evaluator, credentials or a passing score.
Freeze output before optional scoring; a completed unscored run is not failed.
The external reviewer is GPT-6 Astra (`gpt-6-astra`), evaluation only, using the
explicitly authorized source/target/context text. This adapter exports no audio
or video and supplies no findings, locations or replacement wording to writers.

V5 returns numeric dimensions and reading/confidence metadata; it is deliberately
not the historical experimental one-integer response. Old campaigns retain that
old contract. Audit explanations, when separately authorized, remain audit
artifacts and must not become episode-specific writer or repair prompts.

Hide authorship, workflow identity, expected score and prior ratings during
calibration. Freeze the rubric before reviewing. Report all predeclared review
results, including disagreements; do not rerun until a desired score appears.
A human reference is a candidate or control, not a guaranteed eight. One episode
cannot establish reviewer calibration or generalization: use multiple blind
controls and deliberately degraded controls before making those broader claims.
The [September 24 calibration](design/subtitle-quality-v5-calibration-20260924.md)
records both blind v5 results for the supplied human-subtitled episode: overall
seven in both reviews, with source and playback still unverified.
The [six-candidate comparison](design/six-candidates-v5-results-20260924.md)
subsequently scored every retained candidate five in both blind reviews using
the same rubric and common evidence across candidates. Those candidates are
from another episode, so this is not a paired comparison with the human sample.
The [fresh same-video comparison](design/reference-six-methods-results-20260924.md)
completed all six local workflows and fourteen common-basis v5 reviews. Every
generated candidate scored six in both reviews; the unchanged human reference
scored seven in both. These are fresh ratings on the same episode and evidence,
not a rescaling of the historical scores or proof of playback readiness.
The [local meaning-audit pilot](design/local-meaning-episode-results-20260924.md)
then independently corrected the discussed perspective error but remained six
in both blind reviews. Its failed control gate and observed regressions remain
recorded; subsequent expression and translation experiments did not reach seven.
The [self-correction experiment log](design/local-self-correction-results-20260924.md)
records expression editing, bilingual revision, independent translation,
scene-level draft comparison, UD Qwen reasoning and whole-episode local
self-critique under the same frozen scoring basis. The best verified overall
result remains six/six. The self-critic reported only uncertainties, produced
no revision, and was not rescored. The final structured correction of the UD
draft also scored six/six in all three dimensions: one clear meaning repair
came with two lyric omissions, and the original perspective error remained.
All planned reviews are complete. None of these candidates was promoted.
The [September 25 presentation controls](design/presentation-controls-results-20260925.md)
then separated presentation from wording. The human text kept seven/seven under
the machine's coarse segmentation and timing; LC's text with word-aligned cues
and subtitle punctuation stayed six/six; a generic local register rewrite of
that result also stayed six/six. Presentation is not the binding factor on this
episode; the remaining gap is in the Chinese wording. Overall equalled expression
in all 34 v5 reviews of the episode.
The [September 25 wording trials](design/wording-trials-results-20260925.md) reached
6/6/7 in both reviews with two local recipes. The simpler one, the evidence-first
rewrite, is now the [retained workflow](WORKFLOW.md). On a held-out episode, the
multi-stage recipe scored 5/6/5 against the human reference's 7/7/7. Given the same
inputs, a stronger non-local writer scored 7–8. The
[ceiling record](design/local-ceiling-20260925.md) concludes that the local writer, not
the evidence, is the limit without fine-tuning.

## New scoring command

The manifest has `version: "subtitle-quality-bundle-v1"` and `source`, `target`,
`context` entries, each containing an absolute plain-file `path` and exact
`sha256`. Source and target are UTF-8 SRTs; context is UTF-8 text and can describe
lyric cue IDs, approximate timing, OCR limits and source uncertainty. File paths
remain in local provenance; the reviewer receives only their contents.

```bash
# Prepare exact prompts, inputs, schema and producing-code snapshots locally.
.venv/bin/python scripts/review_subtitle_quality.py \
  --manifest /absolute/path/manifest.json --output-dir /absolute/path/new-review

# Explicitly execute the frozen review once using the same arguments.
.venv/bin/python scripts/review_subtitle_quality.py \
  --manifest /absolute/path/manifest.json --output-dir /absolute/path/new-review \
  --execute

# Validate completed inputs, hashes, dispatch identity and receipt read-only.
.venv/bin/python scripts/review_subtitle_quality.py \
  --manifest /absolute/path/manifest.json --output-dir /absolute/path/new-review \
  --validate
```

Each execution preserves a real dispatch, model identity, hashes and failure
receipt; it cannot overwrite or retry an existing attempt. A confirmation uses a
new directory and the same unchanged manifest. Do not edit reviewed files; freeze
and assess a new result if content changes.

## Historical gates stay versioned

The v3 Chinese-only reviewer is capped at four; v4 contextual review is capped at
five. Those were scope-dependent scales, not conventional quality out of ten.
Historical campaign controllers, `scripts/review_contextual.py` and `main.py release`
kept their v3/v4 contracts. They were removed from the tree on 2026-09-25 and remain in
git history. Stored v3/v4 scores are not migrated to v5: the former round-93 workflow's
v4 score of three stays a v4 three.

New text scoring does not authorize publishing, promote a workflow, or replace
independent source, completeness and playback validation. The
[v4 snapshot](design/subtitle-quality-v4.md) records the historical preparation,
confirmation and release commands.
