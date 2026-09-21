# Local subtitle optimization: trial summary, 2026-09-16

Best measured score remains **3**, below the requested4. The latest controlled block has24 distinct Chinese candidates: six scored3, seventeen scored2, and one scored−4. All24 saved score wrappers were reread for this summary. No candidate qualified for promotion or consistent4 testing. Scores are ordinal, not accuracy percentages.

Astra remains an optional final scalar evaluator. All audio processing, interpretation, writing and repairs are local. These screens mostly reuse one episode and cached acoustic evidence; they are not fresh end-to-end or cross-title benchmarks. New acoustic observations remain local, so external scoring uses the same authorized original-source/context/evidence view rather than independently hearing those observations.

## Latest24 scored candidates

The baseline below is LC-G1: local Gemma31 receives the complete saved799-observation evidence set and original context. A matching score does not prove matching errors. Most variants change multiple correlated aspects, so a lower score does not identify the cause.

| Trial | What changed | Score |
|---|---|---:|
| LC-G1 | Full-evidence, whole-episode Gemma31 translation |3|
| AU-G1 | Add66 unhinted automatic-language Qwen ASR readings |2|
| AU-G2 | Add9 forced-Japanese readings to AU-G1 |2|
| GT-G1 | Enable Gemma reasoning on the same evidence |3|
| CS-G1 | Compact the expanded evidence without dropping readings |3|
| AM-G1 | Correct Qwen audio-attention masking and replace66 readings |3|
| AL-G1 | Add12 longer,120-second Qwen recognition windows |2|
| V-G1 | Add66 Voxtral ASR readings |3|
| DG-G1 | Change Gemma temperature from1 to0 |2|
| OCV-G2 | Add Meta CTC readings with VAD guarding |2|
| FO-G1 | Generate six source units at a time, with full evidence each call |2|
| UR-G1 | Supply raw CTC readings, explicitly marked untrusted |2|
| BT-G1 | Local backtranslation probe, followed by one repair |3|
| DF-G2 | Add paired original/DeepFilter-denoised Qwen readings |2|
| Q8-G1 | Gemma ordinary Q8 deployment instead of QAT Q4 |2|
| OI-G1 | Three isolated local evidence selectors, then full-evidence writer |2|
| UE-G1 | Equalize evidence presentation and remove privileged source channel |−4|
| TU-G1 | Explicit supported/partial/unresolved output labels |2|
| RH-G1 | Add local dictionary-derived Japanese reading hints |2|
| DS-G3 | Replace writer with local DeepSeek-V4-Flash-0731 |2|
| VP-G1 | Add a global MOSS transcript of VAD-packed speech |2|
| VV-G1RA | Add packed VibeVoice transcript to all799 prior observations |2|
| VV-G1RB | Original66 source units plus the same VibeVoice global transcript |2|
| ER-G2 | Full-evidence recap, exact source excerpts, support checking, translation |2|

ER-G2 is the latest scored result:12m44s to local completion, or12m55s including its earlier failed synthetic-control attempt. All66 selected excerpts were exact originals; all330 local support judgments were supported. Those checks did not demonstrate correct transcription, and the external score remained2. The corresponding final local accounting including optional-review CPU validation was13m17s for the combined attempt; reviewer runtime is separate.

## Earlier source-aware experiments

These precede the latest24. Some used a smaller review evidence pool, so their scores cannot be treated as one controlled ranking with the later block. Reused identical targets and reevaluations are not new generation improvements.

| Trials | Approach | Result |
|---|---|---|
| S1–S3 | Selective local repairs, including deeper audio revisit |3 /3 /3; final variant unchanged baseline|
| C1–C2 | Source-only episode recap before patches |3 /3; one failed local gate, one rolled back|
| G1 | Installed GLM writer for the repair stage |3 on old pool; same Chinese2 on expanded pool|
| M1 | Assemble compatible saved patches |3; local gate failed|
| N1 | Full-coverage short-audio revisit and selective repair |3; local validation passed;41m29s revisit|
| T1 | Interpret source evidence at each crop's actual scope |2; local gate failed;84m40s|
| W1/W2 | Whole-source-unit writing with complete evidence |No generation: prompt capacity failure, including compact retry|
| L1 | Joint full-episode Qwen draft plus unit checks |3; local gate failed;84m09s|
| J1 | Same draft, broader context during unit verification |2; local gate failed;85m16s|
| R1 | Readable adjacent evidence, whole-episode draft, filtering |3; local validation passed;83m21s|
| DF-L1/DF-R1 | Score the unfiltered native L1/R1 drafts |2 /2; assessment only, no new generation|
| A1/B1 | Raw evidence only; then lossless compact encoding |2 /2; compact form reduced cost, not score|
| RT1 | Reasoning enabled for exact B1 request |2|
| LQ1 | Local Qwen3.5-122B-A10B writer on exact B1 request |2;25m50s local|

The retained production workflow remains round93: Anime Whisper ASR → whole-source Gemma31 translation → text-preserving Qwen display layout. Revised source-aware reviews scored it3 and3. Rechecked alternatives were round108(native English auxiliary evidence):3; round85(Qwen Q8 evidence editing):3; round94(vocal separation + HY-MT):2; round24(Gemma critic editing):2. Earlier v1–v3 scores are historical and are not silently remapped onto v4.

## Diagnostics and incomplete attempts: no new translation score

| Direction | Observed outcome |
|---|---|
| Context-conditioned ASR C-ASR1/1b/2 | Preparation and literal-citation failures; continuation stopped on nonempty silence/noise controls before real-audio calls.89.23s cumulative.|
| Local122B critic C122-R1 | One nonliteral citation caused validation failure before repair;23m39s.|
| Filtered critic follow-up C122-F1 | One local repair returned exactly the baseline; reused3, zero changed units.|
| Bare Meta CTC | Stopped on noise control; VAD follow-up completed and its translation scored2 above.|
| Qwen-family control CF-S1 | Silent input reached output limit without normal end; zero real-audio calls.|
| Bare MOSS recording MR-S1 | Silent control did not end normally; zero real-audio calls.|
| Continuous-original VibeVoice VC-S1 | Hit16384-token limit with a long repetitive tail;4m26s, no writer.|
| Native24k overlapping VibeVoice windows NW-S1 | First two windows completed; third hit16384-token limit;5m16s, no writer.|
| ER-G1 original synthetic control | Misclassified two valid substring quotes;11.35s, zero production calls. Narrow control-only correction led to ER-G2 above.|
| Native Whisper five-best diagnostic NB-S1 | Implementation/testing in progress; zero actual calls, no Chinese candidate or score.|

Technical failures and controls are retained as failures, not converted to score0. One failed control does not establish that a whole model family is unusable.

## What the evidence supports

1. No tested direction has demonstrated4. More passes, more source strings, larger models and higher precision did not reliably improve this episode's score.
2. A passing local checker is insufficient: N1/R1 passed local checks but scored3. ER-G2's all-supported judgments also did not establish correctness.
3. The slower repair chains often spent41–85minutes and remained2–3. The compact whole-episode methods are a better efficiency baseline, while quality remains unresolved.
4. The retained simple production baseline remains the practical default. The latest full-evidence Gemma baseline LC-G1 is a useful research reference, not a newly qualified production replacement.
5. Next diagnostic: preserve five actual Whisper search finalists for eight fixed audio segments. Measure whether there are lexical alternatives before selecting a full translation trial. Diversity alone will not count as accuracy or4.

Detailed ledgers: quality-current-results-20260916.md, quality-loop-index-20260915.md and ../WORKFLOW.md. No production wording or human reference was inspected for this summary; only saved numeric results, counts, workflow descriptions and hashes.
