# auto-subtitle

English | [中文](README.zh-CN.md)

Local Japanese video/audio → Simplified Chinese subtitles on an RTX 5090 32 GB.
The retained workflow is **Anime Whisper → whole-episode Gemma 4 31B QAT Q4
translation → Qwen display formatting with exact-text guards**.

Selected from five unchanged candidates under the revised contextual rubric.
Independent Astra reviews scored the retained output **3, then 3**. This is the
best available workflow, **below the four-point milestone**. See [quality policy](QUALITY.md)
and [workflow details](WORKFLOW.md). Source accuracy and playback remain unverified.

```bash
# CPU preflight: verify inputs, local model assets and runtime. No inference.
./run.sh input/example.mp4 --context-file input/context.txt \
  --output-dir output/example

# Generate once into a new output directory; all writing stays local.
./run.sh input/example.mp4 --context-file input/context.txt \
  --output-dir output/example --execute
```

Use `.venv` and `ffmpeg`. Paths are configured in
[profiles/selected-local.json](profiles/selected-local.json) and
[profiles/selected-writer.json](profiles/selected-writer.json). The selected assets
are under `models/`; the existing local Warden Qwen DFlash profile must be active
before execution. The runner swaps GPU models sequentially and restores it.
Provide original context explicitly. Complete episode input must fit the 32K
writer context; the runner fails before writing if it does not fit.

The draft CLI handles one media file per invocation and requires a fresh output path.
It leaves the input media in place. Output includes source/translated SRTs,
unchanged semantic text, display mappings and local execution receipts. Running
the draft pipeline does not invoke Astra or approve release. External scoring is a
separate, explicitly executed command described in `QUALITY.md`.

The measured selected subtitles are in `output/selected/final/episode.zh.srt`.
Current review and selection evidence is in
`docs/benchmarks/contextual-selection-20260914/`. Inactive experiments, old runners,
model alternatives and outputs are preserved under
`archive/experiments-20260914/workspace/`; they are no longer the active workflow.
The archive has an original-path map; no experimental data was deleted.

## Quality-first revisit

The bounded revisit workflow starts from a cached local draft. It audits the
episode before collecting fresh acoustic evidence, then rebuilds source-grounded
recaps and revises affected context groups. The current controller uses an
initialized campaign containing 66 original owners.

```bash
# Status only; no local models or external reviewer execute.
./run.sh revisit --campaign output/quality-revisit-20260914

# Run the bounded campaign, including Astra text-only scalar reviews.
./run.sh revisit --campaign output/quality-revisit-20260914 --execute
```

Q1 audits every owner, then collects blind original-audio ASR observations. Q2 may
revisit at most 20 locally flagged owners; dialogue separation is used only there.
Q3 holds the selected Japanese and recap fixed while changing the local writer.
The limit is three rounds, 30 minutes each and 90 minutes of local model-stage time;
review-service time and setup are separate. Completed, validated stages are reused.

All recognition, diagnosis, recaps and writing stay local. Astra receives only the
authorized text evidence and returns one integer score; no audio, video or images
are exported. The campaign records candidates without replacing `output/selected`.
A finished candidate must pass the evidence-view release validator before it can
be marked contextually ready:

```bash
.venv/bin/python -m src.contextual_evidence_release --campaign output/quality-revisit-20260914
```

This requires two genuine scores of at least four on identical inputs, complete
local validation, and exact subtitle text ownership. It does not certify audio
accuracy or playback timing, and it does not promote or archive a candidate.
See [workflow, initialization and limits](design/quality-first-workflow.md).
The completed three-round campaign did not reach four: the retained version scored
three and all three revisits scored two on the same final evidence pool. See
[experiment results](output/quality-revisit-20260914/RESULTS.md); the retained
workflow remains the default.


## Bounded sentence repair experiment

The next campaign preserves the selected draft and tries exact sentence-span
patches. It keeps competing Japanese readings, checks the previous and proposed
Japanese/Chinese pairs, and collects expensive dialogue/residual audio evidence
only during the second revisit. Its three candidates use Gemma and Qwen in
alternating writer/checker roles. This experimental controller is currently bound
to the saved 66-owner episode; it is not the default media runner.

```bash
# Inspect saved state without inference.
./run.sh sparse-revisit --campaign output/quality-local-20260915

# Execute/resume the predeclared local campaign and score-only Astra reviews.
./run.sh sparse-revisit --campaign output/quality-local-20260915 --execute
```

Limits are three candidates, at most 90 minutes of local-stage time per candidate
and four hours across the campaign. Existing successful requests are cached;
failed requests have at most one technical recovery. No candidate is promoted
without two independent scores of at least four and validated local receipts.
See the [fixed experiment plan](design/quality-local-20260915.md). A passing
contextual score is not a certification of the audio transcript or playback.

## Cross-segment source recap experiment

The sparse campaign finished with scores 3, 3 and 3 on the same acoustic pool;
its third result inherited the unchanged baseline. No candidate reached four.
See [measured results](output/quality-local-20260915/RESULTS.md).

The follow-up retains competing acoustic readings in a source-only context map.
The writer and checker receive the same cited support, conflicts and alternatives
from other segments. Models generate only exact Chinese patches; the original
Japanese remains an explicitly unverified transcript. All generation stays local.
Astra provides independent scores only.

```bash
# Prepare/inspect; no inference.
./run.sh context-revisit --campaign output/quality-context-20260915

# Execute the declared C1/C2 trials.
./run.sh context-revisit --campaign output/quality-context-20260915 --execute
```

This remains an experiment on the saved 66-owner episode, with two candidates,
75 minutes per candidate and two hours of local-stage time in total. It is not
the default media runner. The [fixed plan](design/quality-context-20260915.md)
defines stopping, duplicate-score handling and the two-review threshold.

C1/C2 completed at 3/3. C1 retained four changed segments but failed its final
local regression check; C2 rolled all changes back and inherited the baseline
score. Recorded local revisit time was 68 minutes 20 seconds, including failed
recap attempts. See [context trial results](output/quality-context-20260915/RESULTS.md).

## Conditional GLM writer experiment

G1 replaces only C1's proposal writer with the installed GLM-4.7-Flash Q6_K.
It reuses the immutable baseline, source map, diagnosis and complete focused
prompts, with unchanged Qwen checking and independent scalar-only scoring.
Native tokenization must show that every full prompt fits before generation.
There is one candidate, one model load, and a 75-minute local dispatch budget;
cleanup is allowed to finish and its time is recorded even after that deadline.
This is an experiment on the same saved episode, not a new default pipeline.

```bash
# CPU-only input/model-identity preparation; C1/C2 must already be complete.
.venv/bin/python -m src.glm_revisit --campaign output/quality-glm-20260915
# Run the one prepared candidate. A started or finished run cannot restart it.
.venv/bin/python -m src.glm_revisit --campaign output/quality-glm-20260915 --execute
# Independent artifact replay; fails unless both fresh scores and local gates pass.
.venv/bin/python -m src.glm_release output/quality-glm-20260915
```

See the [G1 experiment plan](design/quality-glm-20260915.md). Existing selected
subtitles remain separate from experimental candidates.

G1 completed at **3**, retaining six changed owners with its local gate passing,
in 42 minutes 58 seconds of recorded local stage time. Native GLM generation
accounted for 3 minutes 54 seconds; repeated integrity hashing added substantial
overhead. See [G1 results](output/quality-glm-20260915/RESULTS.md).

## Combining repairs and revisiting shorter audio

M1 tests one exact combination of uncontested retained S1/S2/G1 patches, with
whole-episode rollback checks and a 45-minute local budget. It generates no new
wording. [Declared plan](design/quality-assembly-20260915.md).

N1, conditional on M1 finishing below a confirmed four, spends up to two hours
on one shorter-audio revisit. Original waveform cores target six seconds with
overlap and complete coverage. Both local ASR models provide blind observations;
longer evidence remains available. A fresh source-only recap then informs local
Gemma patches and Qwen checks. The starting subtitles and candidate are scored
against the same expanded evidence pool. [Declared plan](design/quality-short-audio-20260915.md).

```bash
.venv/bin/python -m src.assembly_revisit --campaign output/quality-assembly-20260915 --execute
.venv/bin/python -m src.assembly_revisit --campaign output/quality-assembly-20260915 --verify
.venv/bin/python -m src.short_revisit --campaign output/quality-short-20260915 --execute
.venv/bin/python -m src.short_release output/quality-short-20260915
```

These bounded saved-episode experiments do not replace the selected pipeline.
The release commands require two fresh passing scalar reviews and native
artifact replay. Neither text-based score certifies audio truth or playback.

M1 completed at **3** and failed its local regression gate. N1 completed at **3**
against a **2** starting score on the identical expanded evidence pool. N1 kept
four Chinese owner repairs, passed native replay, and took 41 minutes 29 seconds
of recorded local work, including 73 seconds of short-audio acquisition. These
are revisit timings with cached inputs. See [N1 results](output/quality-short-20260915/RESULTS.md).

## Temporal source interpretation experiment

T1 holds N1's evidence and the original G1 Chinese fixed. Local Qwen freshly
interprets every observation in chronological crop groups, then builds one
source-only episode recap. Partial crops can complement one another; omitted
words in a short crop do not themselves constitute contradictory evidence.
Gemma writes bounded Chinese patches; Qwen checks all readings and the episode.
The unchanged strict local gate and two independent scalar scores still apply.
There is one candidate and a two-hour local budget, with no new ASR or images.

```bash
.venv/bin/python -m src.temporal_revisit --campaign output/quality-temporal-20260915 --execute
.venv/bin/python -m src.temporal_release output/quality-temporal-20260915
```

[Declared T1 plan](design/quality-temporal-20260915.md). Its implementation passed
43 synthetic tests before dispatch; that is not a translation quality score.
