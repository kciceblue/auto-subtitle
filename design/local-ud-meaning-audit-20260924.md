# Structured local correction of the UD draft with reused Japanese analysis

The UD scene draft recovered content but remained six/six overall. Its local
whole-episode critic noticed the original perspective mismatch, labeled all
16 reports uncertain, and made no revision. The earlier structured V1 pipeline
had independently corrected that perspective error by separately representing
Japanese and Chinese meanings before comparison. This motivates one combined
test of the existing correction component on the more complete UD draft.

Use the unchanged frozen V1 `process_case` procedure for all 67 owners. Supply
only the raw UD Chinese draft, the same primary Japanese/all 933 observations,
and original title. No external audit, known error list, numeric grade, human
reference, local whole-episode critic report or hand-written correction enters
the procedure. It independently extracts Chinese meaning, compares it with
Japanese meaning, proposes repairs and verifies source/before/after semantics.
Keep all original V1 window geometry, settings, validation and rollback rules.

All 23 original source-only Japanese requests are byte-identical for this new
draft. CPU preflight replayed their native receipts, exact SSE and validated
source frames successfully against their original backend identity. Copy and
bind only these exact source-side artifacts into the new namespace. No old
Chinese frame, comparison, repair or verification may be reused; none of the
23 old Chinese requests matches the new draft. Do not generate source frames
again. Pin every inherited source artifact, the parent UD candidate and the
unchanged producing code. The preflight record is
`output/local-thinking-translation-20260924/V1-SOURCE-REUSE-FEASIBILITY.json`.

The reused source analysis represents 570.272 seconds and 393,520 local tokens.
New work requires 23 Chinese frames and 23 comparisons, followed only by the
repairs and context-dependent verification required by V1. There are no logical
rerolls; completed exact native artifacts can only be replayed. Restore the
owned backend. Do not rescore if the complete target is unchanged from raw UD.
For a changed output, run the same exact-text display and two blind v5 reviews
against byte-identical shared source/context; both overall ratings must reach
seven. Audit source fidelity independently after freezing the output.

This is an exploratory combination, not a new passing capability gate. The V1
synthetic controls still failed because a repair added unsupported gender, and
its scene verification previously rejected a neighboring legitimate proposal.
Those limitations are retained and disclosed, not silently repaired or certified
away. Exact replay, citation validity and structural preservation do not prove
semantic truth. No production promotion, download, ASR rerun, audio/timing/
playback certification or held-out generalization claim is authorized by a score.

Artifacts: `output/local-ud-meaning-audit-20260924/`.
