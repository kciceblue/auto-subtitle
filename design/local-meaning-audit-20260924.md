# Local meaning audit toward a confirmed v5 seven

User authorization: pursue local self-correction, aiming for seven; elapsed time
is secondary to avoiding redundant work. Ask before any large download. This
campaign uses already installed models and cached source evidence. Existing
session authorization covers external text review of this test-data set; audio
and video remain local.

## Success and isolation

Target: a frozen, locally completed subtitle candidate scoring at least seven in
both predeclared independent v5 text reviews on the unchanged same-media source,
context, scope and rubric. This does not certify audio, playback or release.
Every outcome is retained. An unchanged candidate is not rescored to seek a
lucky result. A failed hypothesis needs a material, evidence-driven change before
another candidate is generated; no sampling or reviewer-score reroll sweep.

The human Chinese reference, external diagnoses, expected score and known error
locations do not enter any local model request. The already discussed owner-42
example is a known development example, never the held-out test of discovery.
All 67 owners receive the same predeclared discovery procedure. No manual list
of production errors or handwritten translation repair is supplied.

Earlier temporal frames and generic critics sometimes failed or made quality
worse. The proposed distinction is measurable, separately observed source and
target interpretation followed by raw-text-backed comparison in smaller windows.
Frames, literal quotation checks and edit counts do not prove semantic accuracy.

## First mechanism

Use local Gemma4-31B QAT Q4 with the existing native receipt/capacity adapter.
Temperature .2, top-p .95, top-k 64, repeat penalty 1, seed 20260924. A 65,536-token
context is reserved for focused calls; actual full requests and output reserves
must fit without truncation. Source/target framing and repairs use thinking off;
comparison and independent verification use a bounded reasoning allowance.
Capture every actual request, response, setting and failed attempt. Restore the
captured local backend after owned processes exit. No download is planned.

1. Source-only interpretation of each three-owner focus window, with two adjacent
   owners on either side and all cached source observations assigned to that
   context. Preserve raw transcript alternatives, crop bounds and uncertainty.
2. Independently interpret the exact Chinese in the same window, without Japanese,
   source background, reference subtitles or previous generated source frames.
3. Compare the two fallible interpretations alongside the literal source and
   Chinese. Record equivalent, supported defect or uncertain for every focus
   owner. Preserve actions, participants, quoted-speech perspective, negation,
   intention/outcome and natural target expression. A pronoun difference alone
   is not an error. Validate each quoted witness against the supplied text.
4. Repair only locally discovered, valid supported defects. Preserve complete
   owner geometry and single-line text; no arbitrary paragraph retranslation.
5. Reinterpret changed target windows and verify the assembled provisional
   candidate against raw source and original Chinese in a fresh call. Accept
   only supported repairs with no supported new defect. Uncertain changes revert.
   Check final adjacent context after any rollbacks that change a verified window.

Synthetic cases use their whole short scene rather than production-sized windows.
All unchanged production owners remain in coverage and completion accounting.
Invalid quotations are retained as rejected items, not silently rewritten into
valid evidence or used to delete other findings.

## Capability gate before production

An agent isolated from production examples authors 32 synthetic Japanese/Chinese
cases: 12 development cases (six defective, six correct) and 20 held-out cases
(ten defective, ten correct). Expected labels and split names remain outside
model inputs. Cases include legitimate paraphrases, quote/perspective changes,
uncertain speakers, role reversals, polarity, modality and omission/addition.
These are authored controls, not a real-world accuracy estimate.

Initial gate: discover at least five of six development errors and eight of ten
held-out errors, with no more than one false-positive case per split. Missing,
invalid and all-uncertain responses count as misses, not successful checks.
Repairs receive a separate source-based review of all control outputs; a procedure
that introduces errors into clean controls cannot proceed merely because its
detector has high recall. At least five development errors and eight validation
errors must actually be corrected, with zero newly introduced errors across all
controls. The signed-off audit binds to the exact fixture, metrics and outputs.
The held-out labels are inspected only after its fixed
run. Any subsequent tuning makes that validation set development data and requires
fresh controls to support an unseen-case claim.

## Production and evaluation

Start from the exact frozen fresh LC owners. Reuse the complete cached union of
933 acoustic observations plus original Japanese/title context; no fresh ASR
repetition is needed for this first hypothesis. Evidence selection is by the
uniform owner windows, not by reference differences or external error locations.
Keep all generated proposals, rejected findings, reversions and uncovered items.

After local completion, apply the existing exact-text display formatter and freeze
the candidate. Reuse byte-identical common source/context from the seven-candidate
comparison for two fixed v5 reviews. Compare against its actual LC 6/6 and reference
7/7 receipts. Report what changed, detection/control performance, regressions,
actual costs and all failed hypotheses. A score below seven informs the next
mechanism choice; it never supplies corrective wording to a local writer.

Artifacts: `output/local-meaning-audit-20260924/`. Historical candidates, rubric,
native adapters and production selection remain unchanged.

## Execution and checks

`scripts/run_local_meaning_audit.py` exposes `development`, `validation` and
`production`. Each control stage is immutable after completion; the production
stage requires current producer pins, completed control receipts and independent
semantic audits bound to the exact outputs. Replaying an identical completed
native request does not generate another response. Failed attempts are retained.

The local experiment helper `output/local-meaning-audit-20260924/finish_candidate.py`
exposes separate `display`, `prepare`, `execute` and `validate` stages. Both
external prompts must be frozen before either review. An unchanged LC candidate
is skipped instead of rescored. Only the explicit execute stage sends text to the
already authorized reviewer.

Offline checks: 30 contract tests and 10 orchestration tests pass. They cover
input separation, literal witnesses, malformed responses, allowed edits,
whole-window regression reporting, final-neighbor rechecks after rollback,
request replay and control-denominator accounting. These establish procedural
behavior, not translation quality.

## First result: capability found, production gate failed

The frozen procedure detected all six development errors and all ten validation
errors, with zero false alarms; all sixteen original clean cases stayed unchanged.
Independent source-based reading found all intended defects corrected. However,
one validation repair introduced an unsupported feminine pronoun while correcting
whose brother was mentioned. The source named the person without establishing
gender. This is unsupported specificity, not proof that the actual person is male.
The Chinese-only frame and source-based verifier both missed this new commitment.

Development: 6/6 corrected, zero new errors, 54 native calls, 299.0 seconds overall.
Validation: 10/10 intended defects corrected, 9/10 without new specificity,
one introduced error, 90 native calls, 465.7 seconds overall. Both backend
lifecycles restored the captured idle state. Source-based audits and native
receipts are preserved in the two control directories.

The predeclared zero-new-errors gate therefore failed. This version did not run
on the episode and did not receive an external score. The follow-up adds explicit
checking of new factual commitments while reusing completed work; see
`design/local-meaning-audit-v2-20260924.md`.
