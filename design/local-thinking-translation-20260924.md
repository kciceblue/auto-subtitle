# Japanese-only scene translation with local Qwen UD and reasoning

The preceding Gemma scene comparison remained six overall in both blind v5
reviews. A distinct remaining local intervention uses the already installed
Qwen3.8 27B UD-Q4_K_XL model with reasoning enabled. Its mixed integer
quantization differs from the earlier NVFP4 target; this alone does not establish
higher translation quality. The treatment combines quantization, reasoning and
Japanese-only scene generation, so causal attribution to one component is not
possible.

Freeze fourteen windows, each with five focused owners and two neighbors per
side. Every request receives all 67 primary Japanese owners and the original
title, plus all raw observations linked to that window and its neighbors. All
933 observations are covered across the windows. No Chinese draft, external
correction, critic, known error location, human reference or score is supplied
to the writer. Original LC is used only to calculate changes after generation.

Use the existing 17.56 GB UD model, SHA256
`3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e`.
Load through Warden once, but dispatch directly to its llama.cpp backend on
localhost:18089 so the captured payload is the actual request without a proxy
persona. Authenticate the served model, process command, executable, loaded
libraries, embedded template and rendered input capacity. Context is 229376;
reasoning budget 8192 and total output limit 16384; temperature 0.3, top_p 0.95,
top_k 20, min_p 0, repeat penalty 1, seed 20260924. MTP drafting uses n-max 4.
Require actual nonempty reasoning, normal stop, complete JSON and exact global
owner geometry. Allow one generation per window, with an 840-second deadline
within the native profile's 900-second maximum. Before every direct call,
refresh Warden's activity clock through its non-restarting same-model load,
checking that the model and process remain identical. This stays below its
configured 900-second idle eviction threshold without changing that setting.
Failed/incomplete attempts are retained and not retried under the same plan.
Restore the prior Warden state unconditionally.

One frozen candidate receives the common exact-text subtitle display and the
same two blind v5 reviews using byte-identical common Japanese source/context.
Both overall grades must reach seven. An independent source-only semantic
audit describes actual improvements and regressions but never feeds corrections
back to this writer. No score-conditioned rerolls, ASR reruns or downloads.
This remains exploratory and does not pass the earlier failed strict controls
or establish audio, timing, playback or production-release quality.

Artifacts: `output/local-thinking-translation-20260924/`.
