# Six retained candidates under v5 — 2026-09-24

All six retained translations received two independent blind reviews under the
unchanged `subtitle-quality-v5` rubric. **Every overall score was 5/10.** These
reviews do not distinguish an overall winner.

Each cell below shows primary / confirmation. Scores are ordinal judgments,
not accuracy percentages.

| Candidate | Approach | Overall | Chinese expression | Fidelity to supplied evidence |
|---|---|---:|---:|---:|
| LC-G1 | Long-context Gemma draft | 5 / 5 | 6 / 6 | 5 / 5 |
| GT-G1 | Gemma with thinking | 5 / 5 | 6 / 6 | 5 / 5 |
| CS-G1 | Compact source evidence | 5 / 5 | 6 / 6 | 5 / 5 |
| AM-G1 | Attention-mask-corrected ASR evidence | 5 / 5 | 6 / 6 | 5 / 5 |
| V-G1 | Voxtral source evidence | 5 / 5 | 6 / 6 | 5 / 5 |
| BT-G1 | Blind backtranslation and local repair | 5 / 5 | 6 / 5 | 5 / 5 |

Five means understandable overall, with noticeable weaknesses requiring
substantial editing. Expression generally scores above fidelity to the supplied
evidence. Numeric responses do not identify specific errors; no explanations
were inferred from hidden reviewer reasoning. Equal coarse scores do not prove
equal quality or identical defects, and BT-G1's expression variation alone does
not establish reliable inferiority.

The human reference received overall seven in both reviews under the exact same
rubric. None of these candidates reached seven. However, the reference concerns
episode 10 and these candidates concern episode 13, with different source
evidence. This is not a paired human-versus-machine performance comparison.

## Evidence and validation

The six frozen targets are exactly those in
`design/cleanup-best-six-20260916/six-trials.json`. All original hashes remain
unchanged. Each contains 66 coarse rows; JSON-to-SRT conversion preserves every
text, index and timestamp. No subtitle was regenerated, rewritten or repaired.

Every reviewer received identical common Japanese ASR, original context and all
736 sanitized source observations (733 available, three unavailable). Candidate
labels, workflow identities, prior scores and expected ratings were hidden.
Two reviews per candidate were declared before scoring; all primaries finished
before confirmations started. There were no score-dependent retries or omitted
results. The user explicitly authorized all twelve text-only OpenAI
`gpt-6-astra` dispatches; no audio or video was exported.

All twelve native receipts validate, with twelve distinct dispatch IDs and six
distinct target/prompt hashes. Each primary/confirmation pair used identical
inputs. Independent verification also confirmed unchanged original files and
the same rubric/contract hashes as the human calibration.

All reviewers self-reported reading the complete supplied text. That does not
verify original-media coverage or source truth. Audio fidelity, playback,
completeness and release readiness remain unverified; every v5 certification
flag is false. The original production selection and release gates are unchanged.

Historical v4 scores remain three for all six. V4 and v5 have different anchors;
the numerical change to five is not evidence that a translation improved.

- [V5 rubric and scoring command](../QUALITY.md)
- [Full local comparison report](../output/six-candidates-v5-20260924/REPORT.md)
- [Validated results and receipt hashes](../output/six-candidates-v5-20260924/validated-results.json)
- [Frozen twelve-review plan](../output/six-candidates-v5-20260924/plan.json)
- [Human calibration](subtitle-quality-v5-calibration-20260924.md)

Detailed subtitle and review artifacts remain local and gitignored. The common
historical evidence pool is
`cf71e2d12a6905c8ca728b00a9a548567b4f90fecd63e9d988841ead96c31a13`.
