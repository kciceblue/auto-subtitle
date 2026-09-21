# V-S1 / V-G1: a different local recognizer family

Selected after AL-G1 completed with score 2. The best validated score remains 3. This bounded trial changes the acoustic recognizer family, while retaining all earlier source evidence. It does not claim that recognizer diversity establishes truth. Root alone supplies real inputs and dispatches inference. Agents implement code and invented-data tests only.

## Fixed acquisition before inference

Use the public mistralai/Voxtral-Mini-4B-Realtime-2602 checkpoint at revision 2769294da9567371363522aac9bbcfdd19447add. Use the isolated .venv-voxtral CUDA environment (Torch 2.10.0+cu128, Transformers 5.3.0, mistral-common 1.11.7); pin the full resolved runtime and model assets. Preserve the frozen Qwen environment and earlier producers. Setup/download attempts are retained separately from inference attempts.

Use exactly the same 66 original short owner windows, float32 mono PCM and shared global linear gain as AM-S1. Do not choose clips by generated content. Run two fresh, source-independent 12-second controls first: silence and PCG64 seed 20260915 uniform noise in [-0.01,0.01], float64 draws cast to float32. Maximum 68 inference calls, one model load, zero retries. Native decoded lexical text must be empty on both controls; failure ends acquisition before real calls. Do not alter the controls or decoding after seeing results.

Use native BF16 on CUDA, batch one, greedy decoding, automatic language None, default six delay tokens (480 ms), no context/hints or forced-language prompt. Use offline whole-array processing, is_streaming=False and truncation=False. Do not impose Qwen's 512/4096-token budget: native generation derives total sequence length from padded audio features, ceil(feature_frames/audio_length_per_tok), with audio_length_per_tok=8. Preserve complete raw IDs, prefix and special tokens. Decode with the pinned native Mistral tokenizer SpecialTokenPolicy.KEEP for raw text and SpecialTokenPolicy.IGNORE for the semantic view, without whitespace cleanup. Do not use the Transformers convenience skip_special_tokens=True path, which additionally removes ordinary leading lang:xx text. Only native special-token IDs may be excluded; all ordinary lexical text, including language-like prefixes and literal marker spellings, must survive. A special-token insertion proof may omit raw protocol spans from the writer only when reinsertion reconstructs raw text exactly; otherwise retain raw text visibly. Language is unavailable unless the native API actually reports it; do not fabricate a detected Japanese label.

Capture original and delivered PCM identity, native serialization/padding/features, actual attention/runtime settings, generation/cache-position ranges and covered original-audio domain. Whole-wave frontend presence is insufficient by itself. Native duration-cap completion can be valid; EOS alone does not establish coverage. Reject truncation, unsupported transformations or incomplete original-audio coverage, preserving the attempt rather than retrying. Finalize exact native frontend/stop invariants from installed code and synthetic tests before registration, never from production answers. Record PyTorch allocated/reserved peaks explicitly as allocator measurements, not total driver VRAM.

Hard local limit 1800 seconds: 1380 work and 420 cleanup. Always stop the owned worker and restore the previous backend. Preserve exact source scope and unknown within-window word alignment. Keep new Voxtral text, protocol diagnostics, audio and video local. Save reports outside sealed acquisition folders.

## One conditional local writer

Only after acquisition passes complete native replay, run one V-G1 writer. Retain the complete baseline 799 readings plus all 66 Voxtral readings once (865 observations). Use the compact reversible semantic representation: model-visible readings and physical scope; audit-only paths, hashes and protocol IDs in a separate local artifact. Preserve unavailable language, empty status and recognizer identity; same-family readings are correlated. No content-based candidate selection, semantic hand edits or human-reference access.

Use the existing local Gemma 31B profile, context 163840, output 16384, thinking off/budget zero, unchanged saved sampler. One writer, no successful rerolls. Hard local limit 3600 seconds: 3180 work and 420 cleanup. Record exact input/output/native receipts, local status and final hashes. The local command must complete without review scores, history or Astra credentials. Optional scoring cannot change subtitle bytes or local completion status.

## Final assessment and stopping rule

After the local final is frozen, optional Astra returns only a scalar on the unchanged user-authorized original Japanese/context plus 736-observation pool and generated Chinese. No Voxtral/new Qwen observations, local diagnoses, reasoning, audio, video or human reference are exported. Reuse an existing valid primary for an identical target and pool; include AM-G1 continuation and AL-G1 in duplicate indexing. A primary below 4 closes this specific configuration. A primary of at least 4 permits one independent confirmation; it does not establish consistent 4.

No additional attempts are added to rescue this configuration. If it fails controls/native execution or scores below 4, retain every artifact and move only to a separately declared mechanism. Candidate mechanisms still available include a full-current-evidence local 122B source critic/recap with one Gemma repair, and fixed-candidate acoustic likelihood diagnostics. Do not announce all local workflows impossible after one failed configuration.

A qualifying candidate must later be generalized to fresh per-file evidence rather than depending on this cached 799-reading episode corpus. Five predeclared fresh local runs, all retained and frozen before scoring, test repeatability; held-out recordings test generalization. Astra remains an optional evaluator in production, not a translation or repair stage.

## Existing channel evidence

The original media is stereo AAC at 44100 Hz, but a previous numeric audit already examined 233 active six-second windows: median L/R correlation 0.9710, median downmix loss 0.1145 dB, no active window above 6 dB loss. See quality-contingency-hypotheses-20260915.md. This weakens broad cancellation as the current bottleneck; no repeat stereo analysis or full-episode channel expansion is selected here.
