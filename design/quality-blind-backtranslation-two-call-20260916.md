# BT-G1: two-call blind backtranslation trial

Selected on 2026-09-16 after UR-G1 closed at primary2, before any BT input preparation or model call. Maximum one probe and one repair, one assembled candidate, zero retries. Folder: output/quality-blind-backtranslation-20260916. All earlier receipts remain unchanged; best remains3.

The inherited LC final local cost190.883754286s is reported separately from this new3600s budget. Original B799 acquisition remains excluded from a fresh end-to-end timing claim.

## Question and bounded novelty check

Does a blind Japanese rendering of what the fixed LC Chinese communicates help the
same local Gemma identify translation loss when it then sees the original evidence?
This tests a different comparison representation, not new acoustic knowledge.
The numeric motivation is limited: C122-F1 retained all66 LC strings after five
source-uncertainty concerns; FO-G1's eleven focused writes scored2; best remains3.
Those outcomes do not establish that translation loss caused the remaining plateau.

A bounded search of `src/`, `scripts/` and the current `design/` catalog found the
unselected proposal in `quality-untested-mechanisms-20260916.md`, but no implemented
or completed equivalent. This is not an exhaustive archive novelty claim.
`full_evidence_critic_requests.py` directly compares source/Chinese and produces
critic findings; its filtered recovery returned the baseline unchanged.
`staged_source_requests.py` reconstructs Japanese from source evidence before
translation (SS1 scored2). Neither first interprets Chinese while blind to source.
Earlier Chinese-only critics judge Chinese directly; they do not produce this probe.

## Fixed two-call recipe

Freeze the exact LC-G1 Chinese rows and exact B799/source/context by direct file and
content hashes, without reading their scores or requiring reviewer availability.
Use one owned Gemma31 load, the existing local native adapter, context163840,
output16384 per call, thinking off/budget0, temperature1.0/top_p0.95/top_k64,
repeat_penalty1.0/seed20260913. One probe and one repair, zero retries or fallbacks.
No extra122B critic, recap, new ASR, alternate probe, local semantic veto or sweep.

1. **Blind probe.** Input whitelist is only all66 exact Chinese strings and their
   target IDs/timestamps, plus a fixed generic instruction and closed schema.
   No original Japanese, B observations, original context/title, source filename,
   audio, prior diagnosis, score or hidden conversation history. Ask for the
   Japanese meaning actually conveyed by the whole Chinese sequence, preserving
   fragments, ambiguity, negation and unresolved participants rather than making
   a polished source reconstruction. Return exactly66 `{japanese, uncertain}`
   entries keyed by original IDs; no correction, quality score or proposed Chinese.
   The uncertainty flag is generated self-assessment, not calibrated confidence.
2. **Compare and repair once.** Input contains complete unchanged B799 and its
   original Japanese/context, exact LC Chinese, and the complete probe labelled
   `generated_unverified_backtranslation`, `audio_evidence:false`,
   `source_accuracy_verified:false`. The probe stays a separate derived field,
   never joins the ASR evidence table, replaces Japanese, or counts as an
   independent vote. In the same call, compare meanings and return all66 Chinese
   strings using the existing closed owner schema. Retention is allowed for every
   owner; even an all-uncertain probe does not require edits. No third comparison
   call or generated finding-selection stage is introduced.

## Claim constraints and limitations

Eligible *candidate* translation losses are a source-supported distinction omitted
from Chinese, a concrete Chinese addition unsupported by the original evidence,
or a supported change in negation, participant, modality or temporal relationship.
Each decision must be checked against the original evidence and the actual Chinese;
a difference found only in the probe cannot establish any of these claims.
Unresolved ASR alternatives, unknown speakers, a probe mistranslation, natural
paraphrase and legitimate fragments are not established translation defects.
Keep ambiguity when evidence cannot settle it; fluent completion is not a goal.
These are repair instructions, not machine-certified error labels. The minimal
response stays Chinese-only; mechanical validation must not claim semantic proof.

The same model may repeat its own error, regularize ambiguous Chinese, invent a
Japanese distinction, anchor on LC, or repair a fault introduced only by its probe.
Round-trip similarity is not accuracy and disagreement is not an error detector.
A gain would support this two-call input policy on one episode, not establish
acoustic truth, the causal role of any individual changed owner, or generalization.

## Bounds, stopping and optional review

Total3600 local seconds:3180 shared work and420 protected cleanup. Both calls share
one absolute work deadline; capacity uses each actual complete native payload,
including the generated probe in call2. Do not truncate evidence to fit. Capture
both exact requests, raw streams, native usage/stop and outputs; replay both.
Runtime is unknown; ceilings are not estimates. This cached-B trial excludes fresh
media acquisition/LC generation, whose inherited costs must be reported separately.

Invalid/partial probe, schema/owner mismatch, capacity failure, abnormal stop or
budget failure is terminal: preserve artifacts, no substitute and no success reroll.
Cleanup always completes; overtime records failure. Assemble exact timestamps and
strings only after valid repair; no partial candidate or probe-only scoring.
Freeze one local final before optional scalar review. Reviewer unavailability must
leave local completion intact. The unchanged authorized736 pool, original Japanese
and original context are the only reviewer evidence; the probe stays local.
Byte-identical LC output is an explicit local no-op and may reuse its same-target,
same-pool primary3 only in the optional review command. Other exact duplicates use
the normal registry; otherwise at most one fresh primary. Below4 closes the trial.
A4 permits one declared confirmation; consistency requires a separately frozen
block of fresh local runs with every result retained, not selection of a lucky4.
