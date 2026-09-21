# Retained six trials

All six scored **3**, below the target4. These are saved research candidates, not six qualified production workflows. Original locations are preserved because execution receipts bind absolute paths. The short links in this folder are navigation aliases only.

| Alias | Method | Score |
|---|---|---:|
| [LC](LC/RESULTS.md) | Whole-episode Gemma with all source evidence |3|
| [GT](GT/RESULTS.md) | Same evidence, reasoning enabled |3|
| [CS](CS/RESULTS.md) | Lossless compact expanded source evidence |3|
| [AM](AM/RESULTS.md) | Corrected Qwen audio-attention masking |3|
| [V](V/RESULTS.md) | Added Voxtral recognition |3|
| [BT](BT/RESULTS.md) | Local backtranslation and one repair |3|

The selected production output remains `../output/selected/`. Shared acoustic evidence and small historical validation records remain at their original locations; they are dependencies, not additional selected trials.

See `../design/cleanup-best-six-20260916/` for the exact retention and removal manifests. The original file, human reference material and selected subtitles are protected. External model stores are outside this cleanup.

Astra is optional final scoring only. Its output is never a local repair input. These cached-input trials do not establish fresh-run consistency or generalization.
