# Fresh six-method comparison on the human-reference episode

User request: run the full workflows for LC, GT, CS, AM, V and BT on the supplied
episode-10 video, then compare them with the human subtitles on the same basis.

## Fixed design before generation

- Input: the preserved `input/Summer Pockets - S01E10 - A Lifetime's Worth of
  Summer Breaks HDTV-1080p.mp4`. The human Chinese reference is withheld from
  source acquisition, writing, backtranslation, repair and display formatting.
- Acquire fresh source-only Anime Whisper over the complete PCM window timeline.
  Preserve empty observations and physical windows; never require fabricated
  Chinese merely to make an empty window nonempty.
- Shared base: original source rows, both Zipformer and Qwen on full padded
  windows and complete short cores, plus BandIt dialogue/residual observations.
  Select `round(N * 21 / 66)` BandIt owners uniformly across the full owner
  timeline, in batches of at most twenty. This is a preregistered, text-blind
  substitute for the older episode-specific diagnostic selection.
- Additional ASR: automatic Qwen, forced-Japanese Qwen for fresh eligible
  nonempty/non-Japanese/positive-VAD automatic results, corrected-mask Qwen,
  and Voxtral. Preserve native request/response and physical audio provenance.
- LC: one full-base thinking-off Gemma draft. GT: the same request with a
  4,096-token reasoning budget. CS: base plus automatic/eligible forced readings
  in lossless compact tables. AM: base plus corrected-mask automatic readings.
  V: base plus Voxtral. BT: fresh LC Chinese to a blind Japanese probe, followed
  by one full-base repair; the probe is not acoustic evidence.
- Retain Gemma4-31B QAT Q4, context 163840, output reserve 16384, temperature 1,
  top-p .95, top-k 64, repeat penalty 1, seed 20260913. No score-driven generation
  retries. Record infrastructure failures separately rather than replacing them.
- Use the same guarded local display formatter for every finished candidate;
  preserve every generated character. Audit empty-target skips and source-empty
  target-only fallbacks explicitly. Formatting is not audio/playback validation.

The old experiment controllers hardcode another episode's 66 owners, 799
observations and selected IDs. New generic adapters preserve the six method
distinctions but are explicitly fresh-media adaptations, not bit-identical
replays of those frozen benchmark controllers. Actual counts and timing will be
reported. No old outputs, receipts, source code or scores are overwritten.

## Common comparison

Freeze all six finished target SRTs and the existing 343-cue reference before
scoring. Reevaluate all seven with the unchanged v5 contract: two independent
reviews each, fourteen dispatches total, with every result retained. Give all
reviewers byte-identical fresh source evidence, neutral background, and scope;
hide method identity, authorship and previous scores. Include independently
acquired acoustic alternatives equally, including raw Voxtral observations.

Compare dialogue and audible lyrics; the separately extracted center title card
is outside these audio-only workflows and is excluded from all scoring inputs.
Do not force human cue boundaries onto generated subtitles or delete mixed
song/dialogue windows. Source disagreements remain uncertainty; the human text
is a comparison candidate, not an infallible source script. Report timing/layout
integrity separately from text quality. Audio fidelity, exhaustive coverage and
playback remain uncertified by the text reviewer.

Local artifacts: `output/reference-six-methods-20260924/`. Original inputs and
every generation/acquisition attempt retain hashes, code/model identities and
local lifecycle receipts. All GPU work is sequential and restores the captured
local-server state after owned model processes exit.

## Source-only infrastructure continuation

The first automatic Qwen acquisition passed two negative controls, completed
54 real windows, then reached its native token cap on window 55. That failed
attempt and its exact code/capture artifacts are preserved. A versioned
continuation inherits those 55 observations, marks the failed one unavailable
with no transcript, and executes only previously unattempted windows 56–67.
No attempted real window is rerun and no truncated text is promoted to evidence.
Subsequent per-window technical failures are retained as unavailable while
independent unattempted windows continue; controls, model-load or overall-deadline
failures remain terminal. Acquisition completeness means every scheduled window
was attempted, not that all observations succeeded or all speech was recognized.
This policy was fixed before writer generation and before any new quality score.

## Uniform writer representation correction before scoring

The initial native tokenizer measured 147,217 LC input tokens, while CS, AM and V
needed 160,995, 160,079 and 160,608. With the fixed 16,384-token output reserve,
the latter three could not fit context 163,840. The owned controller was stopped
and restored the idle backend. Its completed, unscored LC draft (226.502 seconds)
and interrupted GT attempt (94.884 seconds) are retained unchanged under
`writer-attempts/attempt-1-capacity-abort/`; neither enters the final comparison.
No quality score or semantic output inspection informed this correction.

All six writers restart uniformly with a versioned input projection: only
operational `observation_id`, `receipt_path` and `receipt_sha256` fields move to
local, hash-bound provenance sidecars. Every transcript, owner, crop, geometry,
status, observer and other metadata value remains exact and in order. The sidecar
reconstructs every original record. Prompt rules, model settings, evidence and
output reserve remain fixed. Exact native tokenization checks all five static
draft requests before the first new generation; BT is checked at its dependent
steps. This is a capacity correction, not a quality-conditioned candidate retry.
Final costs disclose the superseded generation attempts separately.
