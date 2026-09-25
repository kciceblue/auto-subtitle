# Locally diagnosed Chinese expression experiment

The first Chinese-only Qwen editor copied all 67 owners exactly. It completed
normally in 19.834 seconds (4,913 prompt and 3,133 output tokens), so no checker
or new quality review was run. A repeated unconstrained editing request would
be redundant.

This variant adds one explicit Chinese editorial diagnosis before the single
edit. Installed Gemma, with thinking enabled, reads the full Chinese episode
and identifies concrete awkward phrasing, collocation or avoidable syntactic
clutter. Each finding must quote the actual owner text. It must preserve normal
fragments, repetition, lyrical register and uncertain content. It does not see
Japanese, reference text, external findings or scores, and supplies no rewritten
subtitle. The diagnosed-editor request labels these as fallible local opinions.
Qwen sees the complete Chinese plus these locally generated findings, rechecks
them and makes one complete edit. Neither model receives external corrective
wording or known error locations. No critique/rewrite loop follows.
If the critic identifies no valid awkward wording, stop without another editor
call or a quality review.

The starting text remains the complete frozen meaning-audit pilot, with all its
limitations. After editing, reuse the unchanged eight-case development smoke
and direct source/before/after checker from the preceding expression experiment.
Accept all four erroneous edits as rejections and at least three of four
equivalent edits as acceptances before episode checks. Then verify only changed
windows and recheck only contexts changed by rollback. An unchanged editor or
fully rolled-back result is not rescored. Preserve the original experiments,
failed meaning gates and unused V2 validation set.

The critic and checker use existing Gemma weights, context 65,536, temperature
0.2, 1,024 reasoning tokens and 8,192 output tokens. The existing Qwen editor
transport/settings and exact receipt checks are reused unchanged. One critic,
one editor, no logical retries, no downloads, and backend restoration apply.

Freeze the completed candidate and the same two v5 review prompts before either
review dispatch. Source/context and rubric remain byte-identical to the prior
comparison. Both overall scores must reach seven for the requested text-quality
target; production/audio/playback certification remains outside this test.

Artifacts: `output/local-expression-diagnosed-20260924/`. This script composes
the frozen expression controller with an additive critic contract; it does not
edit either earlier producer or silently change their recorded experiments.
