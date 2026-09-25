# Whole-episode local self-critique and revision

Conditional on the preceding UD scene candidate remaining below the common
target, test one whole-episode self-correction of that generated text. The
separate scene outputs have never been jointly reviewed by their producing
model. This stage gives that model the complete draft, complete primary
Japanese, all 933 unchanged observations and original title. It receives no
external audit, reference, score, known error location or suggested correction.

Use the same installed Qwen UD and authenticated thinking profile as the scene
generation: direct localhost:18089, context 229376, reasoning 8192, maximum
output 16384, temperature 0.3, top_p 0.95, top_k 20, min_p 0, repeat penalty 1,
seed 20260924 and MTP n-max 4. One critique, then at most one revision; each has
an 840-second deadline with the same non-restarting activity refresh and prior
backend restoration. All producer code and input bindings are frozen first.

The critic assesses fidelity, natural Chinese and episode coherence. It reports
at most 32 issues, each with an exact target quote and an exact primary-source
or status-ok observation quote, a source reference, a concise explanation, and
a defect/uncertain label. Validate actual quote membership and complete input
bindings mechanically. These checks do not establish that an interpretation
is correct. Reject malformed/unbound findings; uncertainties cannot authorize
changes. No externally supplied examples or diagnostic hints enter the critic.

If no valid defect remains, stop with the unchanged candidate and no new
quality review. Otherwise freeze a revision request using only the validated
local defects alongside the full draft/source. Only defect owners may change;
all other owner texts and every timestamp/ID must remain exact. There is no
recursive criticism, score-conditioned retry or manual patch.

For a changed complete candidate, use the common exact-text display and two
blind v5 reviews with the byte-identical shared Japanese evidence. Both overall
scores must reach seven. Separately audit the changes against source evidence.
The original failed strict controls remain failed; this exploratory test does
not certify production release, audio truth, coverage, timing or playback.
No downloads or ASR reruns are needed.

Artifacts: `output/local-episode-self-review-20260924/`.
