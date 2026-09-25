# V5 rubric calibration — 2026-09-24

The previous text-review contract could not award eight: v4 enforced a maximum
of five, while the general policy required source/completeness/playback
verification for eight. V5 separates a 0–10 text-quality judgment from evidence
and release status. Eight means professional quality with isolated minor
imperfections allowed. Human authorship does not guarantee it.

After explicit authorization, the human-subtitled episode supplied in `input/`
was reviewed blind. The same frozen candidate and fallible Japanese ASR were
used throughout. No subtitle generation or repair was performed.

| Review | Quality | Expression | Fidelity to supplied evidence |
|---|---:|---:|---:|
| V5 primary | 7 | 7 | 7 |
| V5 confirmation | 7 | 7 | 8 |

Both independent `gpt-6-astra` dispatches used the exact same frozen rubric and
evidence; neither received authorship, expected score or prior ratings. Both
receipts validate. These were the two predeclared v5 reviews, with no retries.
Source fidelity, full-media coverage and playback remain unverified. The result
supports good text quality, not the expected overall eight or a certified
whole-episode quality claim.

For historical comparison only, the unchanged v4 reviewer returned four on its
scale capped at five. A separately labeled explanatory audit returned expression
six and evidence-relative quality five under different experimental anchors.
These values cannot be rescaled or directly compared with v5. No candidate text
changed, so the higher numerical v5 result is not a translation improvement.

The candidate contains 277 visually checked dialogue cues and 66 lyric cues,
plus a separately recorded title card. The source has 63 nonempty coarse ASR
cues. OCR sampling and ASR remain incomplete evidence of original-media quality.
One sample cannot establish broad calibration; multiple blind references and
degraded controls are needed before generalizing reviewer accuracy.

The new adapter preserves inputs, prompt, schema, producing code and dispatch
identity. It returns quality dimensions without repair findings and cannot
satisfy the historical release gates. Validation: 104 new and existing
review/release tests passed, plus both live v5 receipt checks.

- [Current rubric and CLI](../QUALITY.md)
- [Historical policy](subtitle-quality-v4.md)
- [Full local report](../output/human-calibration-20260924/REPORT.md)
- [Machine-readable results and receipt hashes](../output/human-calibration-20260924/evaluation/results.json)
- [V5 primary assessment](../output/subtitle-quality-v5-20260924/primary/assessment.json)
- [V5 confirmation assessment](../output/subtitle-quality-v5-20260924/confirmation/assessment.json)

Detailed media, OCR and evaluation artifacts are local and gitignored. The
frozen candidate bundle SHA-256 is
`0aaac13dba5e3e1f8235c57c95fdb07b68901ce51c7625d5a855da3467625ae3`.
