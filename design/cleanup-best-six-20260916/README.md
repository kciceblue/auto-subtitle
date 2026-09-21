# Best-six cleanup completed

User-selected retention: LC, GT, CS, AM, V and BT, all score3. No quality promotion or inference was performed.

- Filesystem free-space increase: 660,948,496,384 bytes (615.56 GiB; measured delta may include unrelated filesystem activity).
- Deleted: 1,373 exact targets containing 174,552 file entries.
- Preserved and checked: 7,827 files, including 7,628 content hashes; large files checked by identity, size and modification time.
- All35 retained symlinks and their referents checked.
- Project models retained: Anime Whisper, Gemma31 QAT Q4, Voxtral, BandIt.
- Project environments retained: `.venv`, `.venv-voxtral`, BandIt runtime.
- Shared external Qwen ASR, Zipformer, Qwen DFlash and llama.cpp installations untouched.
- Selected production subtitle identity remains `912896c8304f35b21dcbe042402e359ee3ebc363be1a1af6698ba9852092dae2`.

Original media, current human-reference materials, selected output, six complete trials and sealed shared acoustic evidence are preserved. Small records from rejected trials remain only where retained trials require them. Original absolute locations are unchanged.

Navigation: [six trials](../../trials/README.md), [models](../../models/README.md).

Audit: `plan.json` binds every deletion; `deleted.jsonl` records performed operations; `completed.json` records measured totals; `verification.json` records subsequent pin and import verification when complete. `keep-paths.json`, `code-keep.json` and `retained-before.json` document preservation.

Focused verification: selected pipeline scoring-independence test passed (1 test,0.021s); production CLI help passed. This is not a full test-suite run or a fresh translation benchmark. The dependency audit found two pre-existing retained tests referring to already-missing historical helpers; cleanup does not claim to repair those tests.
