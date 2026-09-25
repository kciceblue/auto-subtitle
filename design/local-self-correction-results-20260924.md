# Local self-correction experiments

The target is overall seven in both blind v5 reviews on the supplied episode,
using the same rubric and byte-identical source/context as the six-method and
human-reference comparison. No reference translation, external corrective
wording, reviewer diagnosis or known error location enters a local writer.
The numeric reviews are evaluation only. No score-conditioned rerolls occur.

**The target has not been reached.** The best verified overall result remains six
in both reviews, against the human reference's seven/seven. The final combined
experiment also scored six in both reviews, in all three dimensions. Structured
correction of the UD draft repaired one meaning error but introduced two lyric
omissions and missed the original perspective error. All planned reviews are
complete. These same-episode development experiments do not establish
generalization to unseen videos.

| Frozen candidate | Overall | Expression | Fidelity to evidence |
|---|---:|---:|---:|
| Original LC | 6 / 6 | 6 / 6 | 7 / 6 |
| Local meaning audit | 6 / 6 | 6 / 6 | 6 / 7 |
| Local meaning audit plus two expression edits | 6 / 6 | 6 / 6 | 7 / 7 |
| Whole-episode bilingual Qwen revision | 6 / 6 | 6 / 6 | 7 / 7 |
| Fresh source-only Qwen translation | 5 / 5 | 5 / 5 | 6 / 6 |
| Scene-level Gemma comparison of LC and fresh Qwen | 6 / 6 | 6 / 6 | 7 / 6 |
| Japanese-only UD Qwen scene translation with reasoning | 6 / 6 | 6 / 6 | 6 / 6 |
| UD Qwen draft plus structured local meaning correction | 6 / 6 | 6 / 6 | 6 / 6 |
| Human reference | 7 / 7 | 7 / 7 | 7 / 7 |

Every scored local candidate completed its common display and both planned reviews.
The user's explicit authorization covers newly generated candidates from this
dataset for the current session; it is recorded in
`output/local-self-correction-permission-20260924.json`. Each review prompt was
frozen before its first and only dispatch. The self-review output is identical
to the UD scene candidate and retains that candidate's existing six/six result.

The local procedure can correct an explicit error without external hints, but
that success is not dependable across different Chinese phrasings of the same
source. Some candidates improve fidelity without reaching seven overall; no
generated candidate reaches seven in expression. Independent source audits also find
new errors accepted by the local verifiers. These are audit findings, not
explanations supplied by the numeric-only graders.

## What worked and what failed

The first local procedure found the discussed perspective error by itself:
`询问别人是否认识她` became `询问别人是否认识我`, matching explicit Japanese
`私を知りませんか`. Every owner received uniform processing, but this was a
known development example rather than evidence of held-out generalization.
The procedure retained five of six proposals. Independent source-only review
found one clear correction, one partial recovery, one probable regression and
two uncertain changes. Overall quality stayed six. See the
[episode results](local-meaning-episode-results-20260924.md).

The strict synthetic controls remain failed: V1 corrected the intended error
in all 16 wrong cases and left all 16 clean cases unchanged, but one correction
introduced an unsupported gender commitment. An additive V2 checker missed
that same new commitment and was stopped after the decisive failure. Its fresh
20-case validation set remains unused. The later episode trials are explicitly
exploratory; they do not retroactively pass these gates.

Two Chinese-only Qwen editing attempts returned all 67 owners unchanged. The
second had a local Gemma critique identifying two awkward phrases. Neither
unchanged output was rescored. The editing instruction contained an unintended
character-preservation clause; removing that conflict in a separately frozen
version produced exactly the two locally diagnosed edits. This supports the
prompt-conflict explanation without establishing that it was the only cause.

The corrected editor's verifier accepted all four equivalent smoke examples and
rejected all four meaning-changing examples. It retained both episode edits.
An independent audit found one clear wording improvement and one possible
nuance shift. These are small development checks, not a general accuracy claim.
The candidate still scored overall six, despite fidelity seven in both reviews.
The authenticated run used ten new local calls and 173.080 seconds, excluding
the earlier reused critic, later display and external evaluation.

The next intervention gave Qwen the original LC draft plus complete Japanese
evidence for one bilingual episode revision. It tested a different local model
and full-scene interpretation; it does not cherry-pick prior repairs or repeat
source acquisition. Source-supported meaning corrections are permitted, so a
style-only equivalence gate would be inappropriate. Native execution and
structural checks remain, followed by an independent source-only audit and the
same blind evaluation. See the [frozen design](local-episode-revision-20260924.md).
This run completed in 105.565 seconds, using 144,400 input and 3,088 output
tokens. All 933 observations fit the installed 196,608-token context. Its 22
changed owners contain only 34 honorific deletions (20 桑, four 君 and ten 酱),
with no other textual change. It did not fix the known perspective error.
Native execution, complete input binding, geometry and restoration passed
audit. The deletions are a localization choice, with possible loss of register
and affectionate distinctions; they are not 22 substantive meaning repairs.
Because the full-source revision mostly copied its Chinese draft, the next
experiment tests draft anchoring: fresh Qwen translation from the same Japanese
evidence with no Chinese draft, probe, critique, reference or score in its input.
The source acquisition and other six methods are not repeated.
That fresh translation completed in 104.462 seconds and changed 62 owners.
The writer received only Japanese source/evidence/context: 139,515 input and
2,942 output tokens. Its first-person wording at owner 42 improved, but it
introduced gendered address, inconsistent names and other meaning errors. Both
reviews scored five overall, five expression and six fidelity. It is not an
improved replacement for LC. The next experiment uses its differing readings
only as a fallible second draft in local scene-level source adjudication.

That scene-level comparison completed all 14 frozen windows in 1,027.285
seconds, changing 46 of 67 owners. It used 485,845 prompt and 43,273 completion
tokens. The CPU execution audit passed: exact requests and responses, alternating
draft order, all 933 observations, owner geometry, producer hashes and backend
restoration were verified. These checks establish execution integrity, not
semantic improvement. Both common v5 reviews completed and validated: six
overall and six expression in each; fidelity seven/six. The target remains unmet.
The independent source-only audit found supported recoveries at owners 36 and
57 and several expression improvements, but mixed fidelity overall: new
unsupported gendered address at 20, a brief address omitted at 26, a boundary
duplication risk at 58, and inherited errors including the perspective at 42.
The next distinct intervention uses the already installed UD Q4 Qwen with
reasoning enabled and Japanese-only scene generation. It requires no download
and receives no Chinese draft or externally identified correction. See the
[frozen design](local-thinking-translation-20260924.md).

The UD scene generation completed all 14 windows in 922.509 seconds, changing
66 owners. Every native response had actual reasoning and normal completion.
The execution audit passed, including the direct rendered requests, complete
67-owner/933-observation coverage, exact model/runtime identity and restoration.
It used 337,056 prompt and 115,505 completion tokens, all uncached. Its common
display and paired reviews completed: both scored six in all three dimensions
(confidence 0.89 each). There are real omission recoveries but new meaning and
cross-scene consistency problems. More reasoning alone did not solve the
original perspective error. The final bounded step used whole-episode local
self-critique, allowing at most one revision authorized only by that local
critic's quote-bound defects; no external audit findings entered its request.
See the [design](local-episode-self-review-20260924.md).

The whole-episode critic generated 16 issue reports, all labeled uncertain. Thirteen
passed exact source/target quotation checks; three were rejected, also with
uncertain labels. There were zero definite defects, so the declared procedure
correctly skipped revision and rescore. Relaxing the citation checks would not
have created a revision-eligible defect. The entire final target remains
byte-identical to the raw UD candidate. One authenticated call used 152,616
prompt and 9,574 completion tokens in 201.327 native seconds; the full stage
took 211.472 seconds. Input/runtime/output bindings and backend restoration
passed the CPU audit.

The practical finding is limited but clear: a structured local comparison can
find and repair the original perspective error, yet this does not consistently
improve whole-episode quality. Fresh translation and longer reasoning recover
some omissions while introducing other mistakes; the whole-episode critic
then treats potential errors as uncertainties. Reliable source-grounded defect
detection and revision without new errors remain unresolved. Further identical
or lightly reworded attempts are not justified by these results. No candidate
is promoted into the selected production workflow.

A CPU feasibility check identified one distinct combination worth testing:
apply the existing structured V1 comparison to the UD draft. All 23 original
Japanese-analysis requests and responses are reusable byte-for-byte, while none
of the old Chinese-analysis requests matches this draft. Thus the new test
reuses the expensive source work and tests correction on genuinely different
Chinese input. It keeps the original procedure and its known limitations.
See the [combined-test design](local-ud-meaning-audit-20260924.md).

That combined experiment is now complete. It reused all 23 Japanese analyses
and made 58 new local calls across all 67 owners, with no new source inference.
New work used 591,910 prompt and 58,754 completion tokens in 1,148.037 native
seconds (1,163.843 workflow seconds). Reused source time is excluded. Exact
native replay, source/cache/producer bindings, final neighboring contexts and
backend restoration passed the execution audit. Common display preserved the
exact owner text across 329 Chinese cues. It recorded three layout warnings
and zero fallback warnings; playback remains unverified. Both frozen v5
reviews completed and validated: overall, expression and fidelity are all
six/six (confidence 0.88/0.89). A separate read-only check confirmed the same
source/context, rubric, schema and review settings as the reference comparison.

The local verifier retained all five proposed edits. The independent
[five-edit source audit](../output/local-ud-meaning-audit-20260924/SEMANTIC-AUDIT.md)
found the following; all other 62 owners remain identical to the raw UD draft.

| Owner | Verified effect of the retained edit |
|---|---|
| 4 | Changes a name and gendered title without establishing the new reading. |
| 13 | Changes only a name; the recipient error and missing question remain. |
| 44 | Corrects the purpose relation and restores the meaning of being lost. |
| 52 | Removes a supported lyric about singing so the speaker can be found. |
| 53 | Removes supported lyrics about collecting materials and a dependable bird. |

Owner 42 still says “her” where the Japanese explicitly says “me.” The very
same structured procedure and Japanese analysis corrected this in original LC
but did not flag it in the UD wording. Its earlier success therefore does not
establish robust defect detection. Owner 14's changed offer/request meaning
also remains. Five accepted edits cannot be reported as five corrected errors.

This closes the current experiment series without reaching seven. Repeating
these prompts or adding another similar verifier lacks supporting evidence.
A further intervention needs a demonstrated gain in source-grounded detection,
coverage preservation and rejection of newly introduced claims on separate
controls before another full-episode run is justified. The existing failed
strict gate remains failed; no experimental candidate replaces production.

## Artifacts and limits

- Meaning audit: `output/local-meaning-episode-pilot-20260924/`
- Unchanged editor: `output/local-expression-edit-20260924/`
- Local critic and second unchanged editor: `output/local-expression-diagnosed-20260924/`
- Corrected editor: `output/local-expression-rewrite-20260924/`
- Whole-episode bilingual revision: `output/local-episode-revision-20260924/`
- Fresh source-only translation: `output/local-source-translation-20260924/`
- Scene-level comparison: `output/local-pair-revision-20260924/`
- UD reasoning translation: `output/local-thinking-translation-20260924/`
- Whole-episode local self-review: `output/local-episode-self-review-20260924/`
- Structured correction of UD draft: `output/local-ud-meaning-audit-20260924/`

Completed scored candidates have their exact SRT under `finish/display/` and
both authenticated scores under `finish/evaluation-v5/scores.json`. Native and
semantic audits are stored with their candidate. All models and source evidence
were already local; no model or dataset download was needed. Backends are
restored after owned model use. Text grades do not establish audio truth,
exhaustive subtitle coverage, word-level timing, playback quality or production
release approval. The selected production workflow remains unchanged.
