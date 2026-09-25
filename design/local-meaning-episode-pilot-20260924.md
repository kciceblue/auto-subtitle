# Exploratory episode test of the first meaning-audit procedure

This experiment continues the user's request to pursue a local, reference-blind
score of seven while avoiding redundant work. The first procedure corrected all
16 deliberately wrong control cases and preserved all 16 correct controls. One
repair introduced a plausible but unsupported gender commitment. Consequently,
the strict zero-new-errors capability gate failed. Its result remains failed.

The additional open-ended commitment checker did not catch that omission. It was
stopped after the known failure, preserving six completed cases, one interrupted
case and the untouched fresh validation set. Neither version has passed its
original production gate, and neither changes the selected production pipeline.

The remaining practical question is whether the first procedure improves actual
episode quality from six toward seven. A zero-new-error control rule is stricter
than a seven-point text-quality target. We therefore declare a separate exploratory
experiment rather than passing or silently bypassing either frozen gate. This is
an explicit change in experiment scope after the control results; it must not be
described as a prequalified or regression-free production method.

## Fixed experiment

Run the unmodified version-one source-only framing, Chinese-only framing,
raw-text comparison, minimal repair and final-neighbor verification once on all
67 original LC owners. Use the complete cached union of 933 source observations
and original title context. Existing installed Gemma weights, sampler, window
sizes, native receipt adapter and backend restoration remain unchanged. No media,
ASR, reference extraction, original translation or failed control generation is
repeated. No large download is needed.

No reference Chinese, external corrective wording, manual error list, expected
score, or known production-error location enters local requests. Every owner
gets the same declared treatment. All proposals, rejections and unresolved cases
remain in the record. The failed control audits are pinned as limitations, not
model inputs or passes. No manually repaired candidate will replace the result.

After the local result is frozen, use the same exact-text formatter and freeze
both v5 reviews before dispatch. Their source/context files and rubric are byte
identical to the LC six/six and reference seven/seven comparison. Do not rescore
an unchanged output or reroll either review based on its score. Report both
scores and every known limitation. A seven is an overall text-review result,
not evidence of zero regressions, audio truth, playback quality or release approval.

If this candidate remains below seven, inspect aggregate local outcomes before
choosing another materially different intervention. Avoid another synthetic-only
loop directed at the exposed control. The unused 20-case set remains reserved
for a later frozen method.

Artifacts: `output/local-meaning-episode-pilot-20260924/`. Its separately named
controller marks the candidate exploratory and retains the original failed
capability-gate status. V1 and V2 code, plans, results and production gates are
unchanged.
