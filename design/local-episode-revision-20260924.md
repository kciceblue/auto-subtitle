# Whole-episode bilingual Qwen revision

The local meaning audit and subsequent two accepted expression edits both
remained overall six in two blind v5 reviews. The expression candidate reached
fidelity seven in both, while expression remained six. More small wording
repairs are not the next useful test.

All six original episode methods used Gemma. This experiment gives the existing
Qwen3.8-27B model the full original LC draft, complete primary Japanese, title
context and all 933 cached ASR observations for one whole-episode revision.
The input is original LC, not a manually selected mixture of prior repairs.
No human-reference Chinese, external diagnostics, expected score, local issue
list or prior candidate's text enters the revision request.

The Chinese draft is explicitly fallible. Qwen may correct source-supported
meaning errors and improve idiomatic expression and episode coherence. It must
preserve uncertainty where source readings conflict, understand correlated
observations rather than count votes, and respect owner geometry. Empty ASR
does not establish silence, and crop geometry does not prove word ownership.
All observations remain available; acquisition and baseline are not repeated.

Use one installed-Qwen generation with the existing sampler and transport:
temperature 0.3, top_p 0.95, seed 20260915, thinking off, maximum 16384 output
tokens. Its native context is 196608. Render and tokenize the actual request
before inference; require prompt + output reserve + 64 <= context. Large
downloads are neither required nor authorized by this experiment.

Reuse the pinned expression controller's model lifecycle, native capacity,
single-transport receipt and replay checks. Its internal transport tag remains
`chinese-editor`; the actual pinned request and this experiment identify the
new bilingual role. No old/new semantic-equivalence gate applies: such a gate
would reject valid meaning corrections. This candidate therefore has structural
and native execution checks, not a claim of verified semantic preservation.
An independent source-only audit follows without modifying the result.

Freeze the one output, use the common exact-text display, and freeze both v5
review prompts before either dispatch. The source, context, rubric and score
contract remain identical to the six-method/reference comparison. No retries
conditioned on content or scores. An unchanged candidate is not rescored.
Both overall scores of at least seven meet the requested text-quality target;
audio truth, playback and production release remain unverified. Earlier failed
control gates and unsuccessful experiments stay recorded.

Artifacts: `output/local-episode-revision-20260924/`.
