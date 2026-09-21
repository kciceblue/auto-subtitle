# CS-G1: compact automatic and forced evidence

This is the bounded Direction P candidate, prepared in code and not dispatched by its author. Scheduling belongs to the experiment owner. Local preparation and generation do not require any score, reviewer, previous candidate result or evaluation bundle.

The input is the exact immutable AU-G2 request at `output/quality-forced-source-draft-20260916/request.json`, file SHA256 `4e311fa8f919fb3be16c4b9252e7d8c89235507c0eb36d6b88448c0b50821792`. Original source and context are independently bound to the fixed benchmark identities. The original 799-record B body remains exact. All 66 automatic and nine forced records remain represented. `compact_source_draft_requests` replaces repetitive automatic/forced metadata with reversible tables and appends only explicit decoding instructions before the unchanged original instruction. Every literal, language label, status, condition, sample interval, VAD count and correlation survives. Operational paths, hashes and native IDs move to `representation-audit.json`, outside the model request. Reconstruction from the actual request plus sidecar must reproduce the independently pinned AU-G2 request exactly.

One local Gemma4-31B QAT Q4 writer uses context 163840, full_swa=false, max_tokens=16384, temperature=1, top_p=.95, top_k=64, repeat_penalty=1 and seed=20260913. Thinking remains disabled, reasoning budget zero and reasoning_effort none. One owned model load and one streamed generation are allowed; there is no retry, source rewrite, semantic repair, model selection or candidate reroll. The shared native adapter records the complete payload, actual rendered template/token IDs, raw SSE, exact answer and native usage. A valid answer containing unexpected reasoning is a terminal adherence failure. Output strings and all 66 original geometries remain exact.

The candidate has 3600 local seconds: at most 3180 work and 420 cleanup. Registration, input checks, capacity, load, generation, restoration and replay are charged. Cleanup runs even after work is exhausted; any overrun fails the candidate after restoration. Earlier source acquisitions and AU-G2 runtime are reused inputs and are not charged again. This measures one additional candidate, not a fresh full-pipeline cost.

`--prepare` and `--execute-local` finish at `local_complete` without importing a reviewer or consulting score history. That status asserts structural integrity and restoration, not acoustic correctness or benchmark quality. The final subtitle is immutable before the optional `--score` command. Scoring sends only the already authorized original Japanese/context/evidence pool and final Chinese; the 66+9 new observations and private native reasoning are not exported. The existing same-pool baseline and exact-target duplicate registry are consulted only during explicit review. There is at most one primary, and only a primary >=4 allows one independent confirmation on identical inputs. No score or feedback enters writing. Review failure cannot change the local subtitle or authorize regeneration. A passing pair is a benchmark milestone, not release qualification or proof of consistency; fresh declared runs are a separate future evaluation.

Commands, each deliberate and separate:

```sh
.venv/bin/python scripts/compact_source_screen.py --prepare
.venv/bin/python scripts/compact_source_screen.py --execute-local
.venv/bin/python scripts/compact_source_screen.py --score
# Only after an exact primary >=4:
.venv/bin/python scripts/compact_source_screen.py --confirm
```

Artifacts live under `output/quality-compact-source-20260916`. No production command has been executed while implementing this plan.
