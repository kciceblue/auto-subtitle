# Local scene-level adjudication of two independently generated drafts

Fresh Qwen translation corrected the known first-person reading but introduced
other errors and scored 5/5 overall, below LC's 6/6. It will not replace LC as
a whole. Its differing interpretations can instead be compared with LC against
the same Japanese evidence by another local pass.

For every owner, use installed Gemma to produce a source-supported, natural
Chinese revision after comparing original LC and fresh Qwen. Both drafts are
explicitly fallible. Supply complete primary Japanese and both complete drafts
for narrative context, plus all raw observations linked to the focus window
and its neighbors. No external critique, human reference, known-error list,
numeric score or correction wording enters the request. The source corpus is
unchanged; every owner and all 933 observations are covered across the windows.

Use five focused owners with two neighbors on each side: 14 windows over all
67 owners. Swap draft A/B ordering in alternating windows according to a fixed
parity rule to avoid always putting one model first. Request the five focused
outputs only, with original global IDs and geometry. The model may choose or
rewrite either draft when source evidence supports it, preserving uncertainty,
meaningful fragments and character voice. It must not choose by majority or
assume a fluent draft is correct. There is one generation per window and no
feedback/rewrite loop.

Gemma uses the existing installed weights and native adapter: context 65536,
temperature 0.2, top_p 0.95, top_k 64, repeat penalty 1, seed 20260924, thinking
enabled with 4096 reasoning tokens and 12288 maximum output tokens. Each actual
request is rendered/tokenized and checked for capacity before inference. This
is a combined treatment of local windows, independent drafts and more reasoning;
any improvement cannot isolate which component caused it.

All outputs and actual receipts are retained. Structural checks and exact source
bindings do not certify semantic truth or pass the earlier failed capability
gates. Restore the owned backend, freeze one complete candidate, apply the common
exact-text display, and freeze both unchanged v5 reviews before dispatch. Use
the same common source/context. No rescore if unchanged, no score-conditioned
retries, no downloads, no ASR rerun. Both overall ratings must reach seven for
the requested text-quality target; production/audio/playback remain unverified.

Artifacts: `output/local-pair-revision-20260924/`.
