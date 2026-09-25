# Local meaning-audit episode result

The exploratory local meaning audit did **not** reach seven. Under the unchanged
blind v5 rubric and byte-identical Japanese source/context, its two reviews were:

| Candidate | Overall | Expression | Fidelity to evidence |
|---|---:|---:|---:|
| Original LC | 6 / 6 | 6 / 6 | 7 / 6 |
| Local meaning audit | 6 / 6 | 6 / 6 | 6 / 7 |
| Human reference | 7 / 7 | 7 / 7 | 7 / 7 |

The local procedure independently found and corrected the previously discussed
first-person perspective error: `询问别人是否认识她` became
`询问别人是否认识我`, supported by `私を知りませんか`. The whole episode was
processed uniformly; no error location, reference Chinese, external corrective
wording or reviewer diagnosis entered its requests. Because this example was
already known during development, rediscovery does not prove unseen accuracy.

The run covered 67 owners in 23 windows with all 933 cached observations. It
made six proposals and retained five. An independent source-only semantic audit
classifies these as one clear correction, one partial recovery with lost report
framing, one probable regression caused by privileging garbled primary ASR,
one plausible but unproven correction, and one unconfirmed dialogue addition.
A likely lyric recovery was rolled back after the verifier objected to a
separately proposed neighboring change without its finding metadata. No audit
assessment was used to manually select or rewrite the final candidate.

Procedural replay authenticated all 89 native requests: 23 source analyses,
30 Chinese analyses, 23 comparisons, six repairs and seven verification calls.
Native calls consumed 1,868.287 seconds, 1,037,590 prompt tokens and 93,078
completion tokens; the full controller took 1,881.969 seconds. The backend was
restored. The display formatter preserved exact owner text, and both external
review prompts were frozen before dispatch. Neither review was rerolled.

The strict V1 control gate remains failed because one of 16 intended repairs
added an unsupported gender commitment, although all 16 intended defects were
corrected and all 16 clean controls stayed unchanged. The additive V2 commitment
checker missed the same failure and was stopped early: six completed cases,
one interrupted case, 25 unattempted cases. Its fresh 20-case validation set
remains unused. This separate episode pilot was explicitly exploratory; it does
not pass those gates or change the selected production workflow.

The next experiment targets expression using one Chinese-only Qwen editorial
pass followed by Gemma checks against old/new meaning and original evidence.
That is a new intervention, not a reroll of this candidate or a rubric change.
The first such editor returned all 67 owners exactly unchanged: one call,
19.834 seconds including lifecycle, 4,913 input and 3,133 output tokens. Its
receipt was replayed and authenticated; no checker or external review was run.
A separately frozen variant adds one Chinese-only local editorial diagnosis
before the one-shot edit. It receives no external wording suggestions.

Artifacts:

- Candidate: `output/local-meaning-episode-pilot-20260924/finish/display/subtitles.zh.srt`
- Scores: `output/local-meaning-episode-pilot-20260924/finish/evaluation-v5/scores.json`
- Native audit: `output/local-meaning-episode-pilot-20260924/NATIVE-AUDIT.json`
- Semantic audit: `output/local-meaning-episode-pilot-20260924/SEMANTIC-AUDIT.md`

These are text-quality results. Source ASR, exact word ownership and playback
remain unverified, and production release is not approved by these experiments.
