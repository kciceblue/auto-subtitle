# Fresh Qwen translation without a Chinese draft

Whole-episode Qwen revision of LC made only honorific deletions and scored
overall 6/6, expression 6/6, fidelity 7/7. This is consistent with anchoring on
the supplied Chinese, but does not establish that cause. The six original
writers all used Gemma; a source-only Qwen translation has not been tested here.

Make one fresh translation with installed Qwen3.8-27B. The writer receives all
67 primary Japanese owners, all 933 unchanged cached observations and the
original title context. It receives no Chinese draft, backtranslation, local
critique, reference subtitles, external diagnosis or score. The builder API has
no Chinese-draft argument. Original LC remains outside the writer only for
recording exact differences after generation and comparing the common scores.

Preserve all evidence and owner geometry. Translation may recover content when
supported by source observations; ambiguous ASR is not forced into a smooth
invented story. Repeated/correlated readings are not majority truth, empty ASR
does not establish silence, and crop geometry does not establish word ownership.
The new instruction explicitly permits natural Chinese phrasing without
character-copying clauses.

Keep Qwen settings, transport, exact receipt checks and backend restoration
unchanged from the bilingual revision: temperature 0.3, top_p 0.95, seed
20260915, thinking off, 16384 output reserve, 196608 context. Native rendering
and tokenization must fit the entire request plus reserve and margin. One
generation, no retries for wording or scores, no downloads, no source reruns.
No meaning-equivalence checker applies to a fresh translation.

After freezing the output, use the common exact-text formatter and both v5
reviews on byte-identical Japanese source/context. Both prompts freeze before
dispatch. Report both ratings and an independent source-only audit. This is
another exploratory candidate, not a passed capability gate or production
release. At least seven overall in both reviews is the unchanged quality target.

Artifacts: `output/local-source-translation-20260924/`.
