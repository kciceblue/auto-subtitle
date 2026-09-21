# Current local quality results — 2026-09-16

Best observed scalar remains **3**. These twenty-four completed candidates contain six 3s, seventeen 2s and one -4, with no passing4, confirmation or demonstrated consistent4. All record successful restoration. This consolidation reports numeric/status/hash fields from result summaries, native receipts, state files and score wrappers. Text differences are counted programmatically; production texts are not semantically inspected for diagnosis, and no model calls are made by the consolidation.

Writers use Gemma31 (QAT Q4 except ordinary Q8 for Q8-G1) with context163840, except DS-G3 which uses local DeepSeek-V4-Flash-0731 UD-IQ3_XXS with context196608. All reserve16384 outputtokens; thinking is off except GT-G1 and DeepSeek. Recipe descriptions identify the declared treatment, not evidence of semantic improvement. Scores come from `score.json`; newer controllers leave `screen.json.score` null after optional review.

| Candidate / saved summary | Recipe change | Final scalar | Own local s | Fresh ASR in this step s | Incremental combined local s |
| --- | --- | ---: | ---: | ---: | ---: |
| [LC-G1](../output/quality-long-context-gemma-20260915/results-summary.json) | B799 directly to Gemma31; context163840, thinking off. | [3](../output/quality-long-context-gemma-20260915/score.json) | 190.884 | 0.000 | 190.884 |
| [AU-G1](../output/quality-auto-source-continuation-20260916/results-summary.json) | B799 +66 unhinted automatic-language Qwen readings;865 total. | [2](../output/quality-auto-source-continuation-20260916/score.json) | 237.440 | 45.174 | 282.614 |
| [AU-G2](../output/quality-forced-source-draft-20260916/results-summary.json) | Exact AU-G1 evidence +9 forced-Japanese readings;874 total. | [2](../output/quality-forced-source-draft-20260916/score.json) | 233.431 | 22.745 | 256.176 |
| [GT-G1](../output/quality-gemma-thinking-20260916/results-summary.json) | Exact B799; Gemma thinking on, requested4096 inside16384 total output. | [3](../output/quality-gemma-thinking-20260916/score.json) | 244.299 | 0.000 | 244.299 |
| [CS-G1](../output/quality-compact-source-20260916/results-summary.json) | Lossless compact view of AU-G2’s same874 readings; thinking off. | [3](../output/quality-compact-source-20260916/score.json) | 166.500 | 0.000 | 166.500 |
| [AM-G1](../output/quality-masked-source-draft-continuation-20260916/results-summary.json) | AU-G1 layout; replace66 readings with audio-attention-mask-corrected Qwen readings;865 total. | [3](../output/quality-masked-source-draft-continuation-20260916/score.json) | 204.743 | 36.619 | 241.361 |
| [AL-G1](../output/quality-long-source-draft-20260916/results-summary.json) | B799 +12 unaligned120s Qwen windows, each included once;811 total. | [2](../output/quality-long-source-draft-20260916/score.json) | 155.920 | 32.647 | 188.567 |
| [V-G1](../output/quality-voxtral-source-draft-20260916/results-summary.json) | B799 +66 Voxtral short-window readings in a compact view;865 total; language unavailable. | [3](../output/quality-voxtral-source-draft-20260916/score.json) | 183.299 | 293.299 | 476.598 |
| [DG-G1](../output/quality-greedy-source-20260916/results-summary.json) | Exact LC B799/deployment; temperature1→0 only. | [2](../output/quality-greedy-source-20260916/score.json) |160.146 |0.000 |160.146 |
| [OCV-G2](../output/quality-vad-ctc-source-draft-recovery-20260916/results-summary.json) | B799 +66 Meta CTC readings restricted to padded local VAD speech intervals. | [2](../output/quality-vad-ctc-source-draft-recovery-20260916/score.json) |183.543 |78.400 |261.943 |
| [FO-G1](../output/quality-focused-owner-20260916/results-summary.json) | Exact B799 per call; eleven fixed six-owner outputs assembled once. | [2](../output/quality-focused-owner-20260916/score.json) |499.235 |0.000 |499.235 |
| [UR-G1](../output/quality-raw-ctc-admission-20260916/results-summary.json) | B799 +66 exact raw CTC strings, explicitly untrusted after failed noise control. | [2](../output/quality-raw-ctc-admission-20260916/score.json) |186.733 |0.000 |186.733 |
| [BT-G1](../output/quality-blind-backtranslation-20260916/results-summary.json) | Chinese-only backtranslation probe plus one full-evidence repair of LC;1/66 blocks changed. | [3](../output/quality-blind-backtranslation-20260916/score.json) |204.474 |0.000 |204.474 |
| [DF-G2](../output/quality-deepfilter-paired-draft-recovery-20260916/results-summary.json) | B799 +132 matched resampled/denoised Qwen readings;931 total. | [2](../output/quality-deepfilter-paired-draft-recovery-20260916/score.json) |193.289 |76.053 |269.342 |
| [Q8-G1](../output/quality-full-context-q8-continuation-20260916/results-summary.json) | Exact full B799; ordinary Gemma31 Q8,40 GPU layers/16 CPU threads. | [2](../output/quality-full-context-q8-continuation-20260916/score.json) |1491.948 |0.000 |1491.948 |
| [OI-G1](../output/quality-observer-isolation-20260916/results-summary.json) | Three isolated observer selectors, then complete B799 plus frozen suggestions. | [2](../output/quality-observer-isolation-20260916/score.json) |358.371 |0.000 |358.371 |
| [UE-G1](../output/quality-uniform-evidence-20260916/results-summary.json) | Same799 observations; remove separately repeated source-text channel and privileged archival labels. | [-4](../output/quality-uniform-evidence-20260916/score.json) |135.157 |0.000 |135.157 |
| [TU-G1](../output/quality-typed-uncertainty-20260916/results-summary.json) | Exact B799 with typed supported/partial/unresolved output and visible unresolved marker. | [2](../output/quality-typed-uncertainty-20260916/score.json) |152.650 |0.000 |152.650 |
| [RH-G1](../output/quality-reading-hints-20260916/results-summary.json) | Exact B799 plus fallible native UniDic kana hints; one annotation pass and one writer. | [2](../output/quality-reading-hints-20260916/score.json) |252.734 |0.000 |252.734 |
| [DS-G3](../output/quality-deepseek-slot-continuation-20260916/results-summary.json) | Exact B799; local DeepSeek-V4-Flash-0731 UD-IQ3_XXS, native reasoning low. | [2](../output/quality-deepseek-slot-continuation-20260916/score.json) |1880.050 |0.000 |1880.050 |
| [VP-G1](../output/quality-vad-packed-moss-draft-20260916/results-summary.json) | B799 +one VAD-packed MOSS recording transcript and exact sample map. | [2](../output/quality-vad-packed-moss-draft-20260916/score.json) |212.006 |88.897 |300.904 |
| [VV-G1RA](../output/quality-vibevoice-packed-draft-recovery-20260916/A/results-summary.json) | B799 plus one packed VibeVoice global observation;800 readings. | [2](../output/quality-vibevoice-packed-draft-recovery-20260916/A/score.json) |203.957 |110.981 |314.938 |
| [VV-G1RB](../output/quality-vibevoice-packed-draft-recovery-20260916/B/results-summary.json) | Original66 source/context plus the same global observation;67 readings. | [2](../output/quality-vibevoice-packed-draft-recovery-20260916/B/score.json) |117.526 |110.981 shared |228.507 |
| [ER-G2](../output/quality-extractive-source-control-followup-20260916/results-summary.json) | Full B799 recap/exact excerpts, support table, then Chinese; all66 excerpt choices equal originals. | [2](../output/quality-extractive-source-control-followup-20260916/score.json) |785.372 |0.000 |796.725 incl failed control |

Times above are recorded local controller time, including preparation/replay, native generation, cleanup and local preflight/validation for the optional score. External reviewer runtime is excluded. “Incremental combined” is own local time plus the fresh acquisition assigned to that step; a zero means no new ASR in that step, not free original evidence. Local bookkeeping differs across controller generations, so these are measured trial costs rather than a standardized latency benchmark.

Every row depends on the saved B799 research inputs; VV-G1RB presents only their original66 source rows to its writer, together with the new global observation. None is a fresh end-to-end run from media, and none includes the original first pass, original B799 acquisition, earlier failed search directions or unmeasured manual diagnosis. AU-G1/AM-G1 continuation time includes its failed pre-inference preparation exactly once (15.860030s /0.320979s). New ASR observations remain local and are not added to the external pool.

AU-G2’s saved broader combined total is301.349640s:233.431205s own +22.744825s fresh forced acquisition +45.173609s inherited automatic acquisition. Its two-candidate AU block is538.789779s after also adding AU-G1’s237.440139s own time once. The table instead shows256.176030s incremental AU-G2 cost. CS-G1 acquires nothing new and reuses that complete66+9 set; its166.499664s is a cached-input trial cost. Counting the prior67.918434s acquisition alongside CS would give234.418098s, still excluding B799 and not a measured fresh-from-media run.

For AM-G1/AL-G1/V-G1, the summary’s final-local-plus-acquisition totals match241.361484s /188.566904s /476.597770s. Their `screen.json.acquisition_plus_candidate_local_seconds` fields were saved before optional-review CPU checks, so those earlier values are smaller and are not substituted for the final totals here.

| Candidate | External runner s, separate from local accounting |
| --- | ---: |
| LC-G1 | 31.017 |
| AU-G1 | 29.986 |
| AU-G2 | 35.143 |
| GT-G1 | 27.761 |
| CS-G1 | 31.343 |
| AM-G1 | 37.335 |
| AL-G1 | 38.539 |
| V-G1 | 38.445 |
| DG-G1 |34.386 |
| OCV-G2 |34.003 |
| FO-G1 |33.957 |
| UR-G1 |37.754 |
| BT-G1 |39.286 |
| DF-G2 |47.888 |
| Q8-G1 |40.093 |
| OI-G1 |34.585 |
| UE-G1 |29.616 |
| TU-G1 |31.469 |
| RH-G1 |41.220 |
| DS-G3 |36.792 |
| VP-G1 |33.164 |
| VV-G1RA |36.368 |
| VV-G1RB |34.679 |
| ER-G2 |31.072 |

The twenty-four wrappers bind twenty-four distinct target hashes and twenty-four distinct dispatch IDs, with the same original Japanese, context and evidence pool. The pool hash is `cf71e2d12a6905c8ca728b00a9a548567b4f90fecd63e9d988841ead96c31a13`. All record a fresh completed primary rather than cached-primary reuse. No receipt or model was replayed for this consolidation.

Astra is optional final scoring only, after an immutable local output; its diagnoses or scores are not local generation/repair inputs. The authorized external view remains the same736-observation pool. Source-aware v4 permits legitimate fragments: neither sentence completion nor a smoother-looking subtitle is independently a quality gain. Mechanical completion/restoration and source accuracy remain separate claims.

The equal3 scores do not establish equal error patterns or prove that one recipe is generally best. No selected/default production behavior changes here. Consistency requires the separately predeclared fresh local runs with every result retained, and one episode cannot establish generalization.

C122-R1 subsequently failed its literal-citation gate before repair:1419.324509478s local, one critic and zero repair calls, restored. Native normal-stop/schema-complete response had one nonliteral citation out of11; there is no final candidate or scalar. It is excluded from the scored table, retained separately in its RESULTS.md, and is not a score of zero. C122-F1 is a separately declared recovery policy; no prior result is rewritten.

C122-F1 subsequently completed but returned the exact LC baseline for every owner: zero changes after one local repair of the five source-uncertainty concerns. Its reused3 is not a tenth distinct candidate or a new review. Own final local cost and inherited critic cost are separated in its results-summary.json. Bare OC-S1 stopped at a noise control without a candidate (105.708 s). OCV-S1 then stopped on a wrapper shape bug (20.866 s). The separately registered shape correction OCV-S2 completed acquisition and its OCV-G2 writer scored 2. The table excludes the earlier 126.574 s from incremental cost; including it gives 388.517 s for the full raw/guarded block. Both failed attempts and all raw output are retained. FO-G1 then completed eleven fixed six-owner output groups over full B799 and scored 2. It is closed. UR-G1 subsequently made all66 saved raw CTC strings available as explicitly untrusted evidence and also scored2. Its table cost excludes inherited CTC acquisition/failures. Both raw/gated variants are closed. BT-G1 then completed its two local calls and changed1/66 blocks, scoring3. The table shows final local accounting204.474s, including local review-input replay; generation/local-completion was185.655s. Inherited LC190.884s makes395.358s combined, excluding B799 acquisition. No confirmation or promotion.

DF-S1 completed68 CPU denoising calls and4 clean ASR controls, but stopped on its sixth real ASR public call before decoding: one smallest float32 subnormal became signedzero during CPU normalization. Local44.265s/restored, no candidate or scalar. DF-S2 preserved the bits, reused all CPU preparation unchanged and completed136 fresh ASR calls in76.053s local. Its full acquisition block including earlier44.265s is120.318s. DF-G2 then scored2, changing65/66 blocks versusLC. Combined final DF block313.607s excludes originalB799 collection and external47.888s. No confirmation/promotion; denoising treatment closed.

Q8-G1 completed under its predeclared fixed profile. Its first preparation stopped0.101s before any model load because the inherited selected-backend profile loader rejects different model aliases. A separate continuation uses a strict dedicated Q8 loader and includes that failed preparation once. Local completion took1467.680s; final local accounting1491.948s includes optional-review CPU checks. Native generation1405.580s,63930 prompt/3523 completion tokens; final scalar2,65/66 changed versusLC, restored. Ordinary Q8 differs in checkpoint lineage, precision and CPU/GPU placement from QATQ4, so this does not isolate quantization. No confirmation/promotion. OI-G1 subsequently completed all four local calls in336.493s, final local358.371s, score2. Its writer used122593 prompt tokens and changed63/66 blocks versusLC. All selectors chose at least one reading; no abstentions occurred. Source selections remained local, final writer retained all799 observations, backend restored. No confirmation/promotion; treatment closed. UE-G1 subsequently completed its single call in116.112s, final local135.157s, score-4, restored. All66 outputs changed versusLC. The regression closes this combined presentation treatment without identifying a semantic cause or an individual causal component. TU-G1 subsequently completed in133.364s, final local152.650s, scalar2. It reported65 supported,1 partial and0 unresolved owners; one marker was inserted and one additional literal marker was already in native Chinese. All66 owners remain visible,65 changed versusLC; restored, no confirmation/promotion. This does not verify the65 supported assertions. The typed policy is closed. RH-G1 was separately selected before this score and is now activated; local dictionary setup/adapter preparation are in progress.


RH-G1 subsequently completed one799-observation dictionary pass and one writer in231.337s, final local 252.734s, scalar2/external41.220s. 64/66 owner strings changed, restored, no confirmation/promotion. See output/quality-reading-hints-20260916/RESULTS.md, including the post-score reporting incident. DS-G1 was selected before RH score and is now activated.


CF-S1 corrected the retained-Qwen attention configuration but stopped at its first25s silent synthetic control:512 generated tokens without normalEOS,1model load/1controlcall/0realcalls. Local25.751s includingpreparation/restoration; no writer or score, no promotion. This control result retires the declared configuration, not the entire Qwen family. Failure artifacts are preserved and the numeric report is design/quality-qwen-family-result-20260916.json.


DS-G1 loaded its fixed local DeepSeek deployment but failed the synthetic probe output cap:9380prompt/128completion tokens, finish_reason=length. Local266.721s, publicsetup2968.363s separate. Zero productionwriter calls/candidates/scores, restored. This is an insufficient synthetic probe allowance, not a translation-quality finding. Fixed attemptclosed; independently predeclaredMR nowactivated.


DS-G2 successfully passedits unchangedsynthetic lookup in183tokens; it thenstopped beforewriter on acontrollerHTTPtypebug (GET/slots returnsarray, sharedhelperrequireddict). Local385.925s/restored/zeroChinese, noqualityscore. VersionedDS-G3controllercontinuation is beingprepared; same singlequalitycandidate allowance, no additionalprobe. MR-S1 subsequently retiredatfirstsilentcontrol withno normalEOS/raw2244chars:47.876s/restored/zero realaudio/zeroChinese, noqualityscore. Neither is included in19scoredcandidates.


DS-G3 subsequently completed its sole production writer in1760.620s; final local accounting1880.050s, scalar2/external36.792s,66/66 changed/restored. No confirmation or promotion. Including both earlier DS infrastructure attempts gives2532.696s local; public setup2968.363s remains separate. VP-S1/VP-G1 now activated under the prior conditional plan.


VP-G1 subsequently completed one local writer plus its88.897s acoustic revisit in287.087s to local completion and scored2; no confirmation/promotion. The one-ASR/two-writer VibeVoice experiment is registered in quality-vibevoice-packed-trial-20260916.md. Both predetermined target variants must freeze before scoring.


VV-S1 then failed at its first public model-load attempt:26.071s local,0 successful model loads,0 acoustic calls, backend restored. A public tiny-checkpoint reproduction identified recursive SDPA assignment to convolution-only submodels; this is an infrastructure result, not a new scored candidate. A separately bounded loader-only recovery preserves the original attempt and counts its time inside the same acquisition allowance. See quality-vibevoice-packed-loader-recovery-20260916.md. The two predetermined Chinese arms remain unrun and must both finish before optional final scoring.


VV-S1R successfully acquired one packed VibeVoice observation in110.981s combined local time, including the26.071s failed initial load. Both predetermined local writers froze before scoring:VV-G1RA (all800 observations)183.593s and VV-G1RB (source-only plus global,67 observations)96.951s. Both scored2. Full acquisition/two-writer completion391.525s; no confirmation, repair or promotion. Fixed block closed. Reports:output/quality-vibevoice-packed-draft-recovery-20260916/{A,B}/results-summary.json and design/quality-vibevoice-packed-block-result-20260916.json.


VC-S1 continuous-original recognition retired at16384 tokens/no EOS in266.412s local. A token-only repetition audit found a two-token period across its final16031 tokens (115 distinct token IDs total); increasing the cap is not selected. Backend restored;0 writers/0 new scores. Next bounded treatment NW-S1/G1 combines direct native44.1→24k extraction with five overlapping five-minute recognition windows, preserving all audio. See quality-vibevoice-native24k-windows-trial-20260916.md.


NW-S1 direct native24k five-window treatment retired on its third window at the16384-token limit:316.491s local, one model load,3ASR calls,2normal completions and2unattempted windows. Original backend restored. No partial salvage, writer, score, confirmation or promotion. All artifacts retained; numeric report:quality-vibevoice-native24k-acquisition-result-20260916.json. The23-candidate score catalogue remains unchanged.


ER-G1 stopped at its six-case invented control after11.352s local/restored,1control and0production calls. The model marked two valid substring quotations invalid; no episode proposal/support/writer or score exists. This is a control failure, not a new translation score. Frozen report:quality-extractive-source-initial-failure-20260916.json.


ER-G2 completed all four local calls and froze its candidate in775.280s combined, including ER-G1's failed control. Primary score2; no confirmation/promotion. Source proposals retained all66 original rows unchanged, and the support pass labeled330/330 dimensions supported. These are diagnostic counts only. The fixed recap/support treatment is closed; see output/quality-extractive-source-control-followup-20260916/RESULTS.md.
