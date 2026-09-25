# Same-video six-method comparison: completed results

All six fresh workflows completed on the supplied episode-10 video. The human
Chinese reference was withheld from source acquisition, generation, repair and
formatting. Source and method audits passed; 54 focused offline tests passed.

All fourteen blind v5 reviews completed and their native receipts validated.
The same original video, fresh Japanese evidence and rubric were used for every
candidate, with method identity and authorship hidden. Scores below are the
primary and confirmation reviews, respectively.

| Candidate | Overall /10 | Chinese expression /10 | Fidelity to supplied evidence /10 |
|---|---:|---:|---:|
| LC | 6 / 6 | 6 / 6 | 7 / 6 |
| GT | 6 / 6 | 6 / 6 | 6 / 7 |
| CS | 6 / 6 | 6 / 6 | 6 / 6 |
| AM | 6 / 6 | 6 / 6 | 6 / 6 |
| V | 6 / 6 | 6 / 6 | 6 / 6 |
| BT | 6 / 6 | 6 / 6 | 7 / 7 |
| Human reference | **7 / 7** | **7 / 7** | **7 / 7** |

None of the six reached the reference's overall score. The added reasoning,
source evidence and backtranslation did not increase overall ratings on this
episode. LC has the lowest observed writer cost among the tied candidates.
This does not establish equal underlying quality or general superiority: two
integer ratings on one episode are a coarse comparison. In particular, LC and
BT have identical translated wording but different segmentation and some
different fidelity ratings; those differences do not show better translation.

The earlier episode-13 scores of five are not a before/after control for this
episode-10 run. [Full report and receipts](../output/reference-six-methods-20260924/comparison-v5/REPORT.md)
include all dimensions, execution costs, provenance and limitations.

| Method | Final Chinese SRT | Cues | Own writer seconds | Changed owner strings versus LC |
|---|---|---:|---:|---:|
| LC | [Download](../output/reference-six-methods-20260924/methods/LC/display/subtitles.zh.srt) | 301 | 139.260 | 0 |
| GT | [Download](../output/reference-six-methods-20260924/methods/GT/display/subtitles.zh.srt) | 298 | 252.131 | 63 |
| CS | [Download](../output/reference-six-methods-20260924/methods/CS/display/subtitles.zh.srt) | 289 | 140.244 | 62 |
| AM | [Download](../output/reference-six-methods-20260924/methods/AM/display/subtitles.zh.srt) | 338 | 147.833 | 56 |
| V | [Download](../output/reference-six-methods-20260924/methods/V/display/subtitles.zh.srt) | 304 | 148.552 | 56 |
| BT | [Download](../output/reference-six-methods-20260924/methods/BT/display/subtitles.zh.srt) | 314 | 188.826 | 0 |
| Human reference | [Unchanged extraction](../output/human-calibration-20260924/evaluation/candidate.srt) | 343 | — | — |

Every generated display preserves all owner text exactly and has zero layout
warnings. All methods retain four empty owners explicitly. BT's probe and repair
returned the exact LC wording; its different cue count is a formatting difference.
BT also inherits the LC draft's 139.260 seconds, shown separately from its own cost.
Cue counts and mechanical checks are not translation-quality or playback scores.

The shared source uses 67 physical windows and 933 acoustic observations for the
comparison, with 12 unavailable readings retained as missing evidence. Shared
acquisition recorded 322.321 seconds; all additional recognizers recorded 364.603
seconds. Final native writing totaled 1,016.846 seconds; display totaled 220.702
seconds. A superseded capacity attempt cost another 321.386 native-writing seconds
and remains preserved separately. The [fixed design](reference-six-methods-20260924.md)
documents the fresh-media adaptation and reversible operational-metadata projection.

Fourteen blind v5 prompts were frozen: two per finished candidate and two for the
unchanged reference, with identical source/context and hidden method/authorship
labels. The user explicitly approved all fourteen text reviews and permission
for this test-data set throughout the current session. The earlier automatic
approval block was resolved before the first dispatch. All fourteen completed
with no failures; each review used a unique native dispatch, and each pair used
identical prompt bytes. All primary reviews finished before confirmations began.
No audio or video was exported. Earlier scores were not reused.

- [Frozen comparison plan](../output/reference-six-methods-20260924/comparison-v5/plan.json)
- [Validated scores](../output/reference-six-methods-20260924/comparison-v5/validated-results.json)
- [All fourteen score rows](../output/reference-six-methods-20260924/comparison-v5/scores.csv)
- [Independent acquisition audit](../output/reference-six-methods-20260924/ACQUISITION-AUDIT.md)
- [Independent methods audit](../output/reference-six-methods-20260924/METHODS-AUDIT.md)

Audio fidelity, exhaustive subtitle coverage and playback remain unverified.
The selected production recipe and historical release gates remain unchanged.
