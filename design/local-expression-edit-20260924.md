# Chinese expression experiment after the meaning audit

The frozen meaning-audit episode received overall 6/6, expression 6/6 and
fidelity 6/7 under the unchanged v5 rubric. Repeating its minimal semantic
repairs is unlikely to address the expression bottleneck. This experiment uses
a different installed model and a different task, without reference text or
external corrective advice entering local requests.

Qwen3.8-27B edits the complete Chinese episode once, seeing only its 67 owner
texts and geometry. It may improve natural phrasing but must preserve all
meaning, ambiguity, register, names and ownership. Empty owners remain empty.
The starting text is the entire frozen exploratory V1 result, including its
known limitations; no individual edits are selected by an external auditor.

Gemma then checks changed windows directly against old/new Chinese and the
original Japanese plus all cached observations for that window. There are no
intermediate fact inventories or feedback-to-editor loop. Checks cover both
directions of meaning, added commitments, omissions and changed neighboring
relations. All visible edited owners are explicitly identified so that a
neighboring edit is not rejected merely for being outside the current focus.
Unresolved new drift or malformed/incomplete checks cause rollback. If a
rollback changes another checked window's visible text, only that changed
context is checked again; identical requests are reused without generation.

Before episode verification, eight independent short development controls test
the same checker: four equivalent style edits and four introduced meaning
errors. The fixed smoke threshold is all four errors rejected and at least
three equivalent edits accepted. This small smoke test does not certify broad
competence or absence of regressions. It does not replace or pass either failed
V1/V2 gate, and the unused V2 validation set remains untouched.

Use installed weights only, one Qwen draft, no score-conditioned retries, and
restore each model backend. The editor runs thinking-off at temperature 0.3;
the Gemma checker uses 65,536 context, temperature 0.2, 1,024 reasoning tokens
and 8,192 maximum output tokens. Existing native capacity/stop/usage checks and
exact request records apply. Source acquisition and baseline generation are
not repeated. No download is needed.

After the result freezes, apply the common exact-text display formatter and
freeze both v5 reviews before dispatch. Use the byte-identical source/context
and rubric from the six-method/reference comparison. Two overall scores of at
least seven meet this experiment's text-quality target; they do not certify
audio truth, playback or release readiness. An unchanged result is not rescored.

Artifacts: `output/local-expression-edit-20260924/`. All results, including
rejected edits and unsuccessful hypotheses, remain recorded. The selected
production workflow is unchanged.
