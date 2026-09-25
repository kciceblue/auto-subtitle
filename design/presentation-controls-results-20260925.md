# Presentation controls and register rewrite: results — 2026-09-25

Designs: [presentation controls](presentation-controls-20260925.md) and
[register rewrite](subtitle-register-rewrite-20260925.md). All reviews used the
unchanged v5 contract, with the byte-identical common source (`86747512…`) and
context (`a0c6c3c9…`) of the six-method comparison, hidden labels, frozen prompts,
primaries before confirmations and no score-conditioned retries. All six receipts
validate.

| Candidate | Wording | Presentation | Overall | Expression | Fidelity |
|---|---|---|---:|---:|---:|
| Human reference (earlier) | human | human | 7 / 7 | 7 / 7 | 7 / 7 |
| **C1** | human | machine segmentation/timing | **7 / 7** | 7 / 7 | 7 / 7 |
| LC (earlier) | LC | machine | 6 / 6 | 6 / 6 | 7 / 6 |
| **C2** | LC | word-aligned + `。，`→space | **6 / 6** | 6 / 6 | 6 / 7 |
| **C3** | LC + register rewrite | as C2 | **6 / 6** | 6 / 6 | 6 / 6 |

## Reading (pre-declared)

C1 = 7 and C2 = 6 is the table's last row: **presentation is not the binding
factor; the gap is in the Chinese wording.** The human text kept seven in both
reviews under a layout harsher than any generated candidate's. Word-aligned
presentation removed every long or merged cue from LC without changing a score.
The generic register rewrite (C3) also stayed at six. Overall equalled
expression in all 34 v5 reviews of this episode.

| Layout | Cues | > 6 s | > 10 s | > 25 chars | Max chars | `。` |
|---|---:|---:|---:|---:|---:|---:|
| Human reference | 343 | 7 | 2 | 0 | 17 | 0 |
| C1 | 236 | 56 | 33 | 23 | 83 | 0 |
| LC | 301 | 47 | 22 | 10 | 98 | 172 |
| C2 | 302 | 27 | 0 | 0 | 20 | 0 |
| C3 | 288 | 27 | 0 | 0 | 20 | 0 |

## What each control did

**C1** kept every human character. A mechanical interval partition separated
exactly the six concurrent insert-song cues, which kept their own timing; all
other cues were regrouped into the 67 owner windows and passed through the
common formatter (230 display cues, one layout warning, zero fallbacks).

**C2** kept every LC character except the declared `。，；：` conversion. The
installed Qwen3 forced aligner aligned all 63 non-empty windows in 2.9 GPU
seconds. Fifty-eight owners used aligned timing and five (noise, humming, one
lyric window) proportional timing over the aligned extent; 15 implausibly
compressed units were re-timed. The local partition took 21 seconds: 55 owners
matched on the first model answer, one on the second, one by DP fallback and six
had a single unit. Sixteen cues hit the seven-second cap and eight could not
reach the reading minimum between neighbours.

**C3** started from C2's exact pieces. One local Qwen pass (seven requests,
18 seconds, every ID answered) rewrote 60 of 280 pieces, left 201 unchanged and
removed 11 non-verbal cues. It removed all but one transliterated honorific
(`桑`/`酱`) and `您`, and made about 25 genuine phrasing repairs. Guards kept
three drafts the model tried to blank. Defects: it left the `爱理`/`羽依里`
inconsistency (4×) and, contrary to its rule, removed the recurring verbal tic
`唔咕？` in five places. Five "kana" rejections were unchanged lines containing
`ー` and did not affect output. No reference-only form appears in any request
or answer.

## Process

Before generating C2 or preparing any review, a three-lens critique with one
adversarial verifier per finding ran on the code, design and real data:
17 findings, 11 confirmed, all fixed before freezing (see the design). The first
C1 draft, which interleaved lyric cues into dialogue, was superseded unscored.
C3's procedure was frozen before its output was inspected; its modest change
volume was reported rather than re-tuned. Six `gpt-6-astra` dispatches in total.

## Conclusions

1. **Target the wording.** Segmentation, timing and punctuation convention do not
   move the v5 text score on this episode, in either direction.
2. **Generic register polish is not enough.** Removing honorifics and non-verbal
   cues and smoothing phrasing leaves LC at six; so did the September 24 honorific-
   only revision. With every tested local procedure at six, the remaining gap is
   finer word choice, names and meaning precision.
3. **C2 is still a better subtitle for viewers.** No cue longer than ten seconds
   or twenty characters, speech-aligned timing, about 25 seconds of extra local
   compute. The text reviewer cannot see that benefit; only playback can confirm
   it. It is a candidate production change for readability, not for the score.
4. One episode, integer scores and a development-informed rule set: none of these
   results generalizes without a held-out episode.

Artifacts: `output/presentation-controls-20260925/` (C1, C2, C3, `superseded/`,
`evaluation/`, `evaluation-c3/`). The selected production workflow is unchanged.
