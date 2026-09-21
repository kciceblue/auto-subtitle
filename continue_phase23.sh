#!/usr/bin/env bash
# Resume auto-subtitle Phase 2–4 from output/ after Phase 1 exit 1.
# Do NOT use set -e: one track failure must not abort the remaining 88.
set -uo pipefail
cd /home/kciceblue/HF/auto-subtitle
export PYTHONUNBUFFERED=1
PY="./.venv/bin/python3"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:8089/v1/chat/completions}"
SOURCE_LANG="${SOURCE_LANG:-Japanese}"
TARGET_LANG="${TARGET_LANG:-Simplified Chinese}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
VOCAB="${VOCAB:-./vocab.txt}"
EXTRA_PAYLOAD="${EXTRA_PAYLOAD:-{\"model\": \"qwen3.8-27b-dflash\", \"temperature\": 0.3}}"

MEDIA=()
while IFS= read -r -d '' f; do MEDIA+=("$f"); done < <(
  find output -type f \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.m4a' \
    -o -iname '*.flac' -o -iname '*.ogg' -o -iname '*.aac' \
    -o -iname '*.mkv' -o -iname '*.mp4' -o -iname '*.webm' \) -print0 | sort -z
)

echo "═══════ Resume Phase 2+3 from output/ (${#MEDIA[@]} media) ═══════"
fail=0
skip=0
ok=0

for media in "${MEDIA[@]}"; do
  rel="${media#./}"
  rel="${rel#output/}"
  stem="${rel%.*}"
  src_srt="output/${stem}.srt"
  zh_srt="output/${stem}.zh.srt"
  adj_json="output/${stem}.adjudication.json"
  rel_dir="$(dirname "$stem")"
  base="$(basename "$stem")"
  report="output/${rel_dir}/REVIEW-${base}.zh.md"

  if [[ ! -f "$src_srt" || ! -f "$zh_srt" ]]; then
    echo "SKIP (no srt/zh): $rel"
    skip=$((skip + 1))
    continue
  fi
  if [[ ! -f "$media" ]]; then
    echo "SKIP (media gone): $rel"
    skip=$((skip + 1))
    continue
  fi

  if [[ ! -f "$adj_json" ]]; then
    echo "═══════ Phase 2: arbitrate $stem ═══════"
    if ! "$PY" -m src.adjudicate --srt "$src_srt" --media "$media" \
      --suspicious all --out "$adj_json"; then
      echo "FAIL adjudicate: $rel"
      fail=$((fail + 1))
      continue
    fi
  else
    echo "SKIP adjudicate (exists): $adj_json"
  fi

  if [[ ! -f "$adj_json" ]]; then
    echo "SKIP review (no adjudication json): $rel"
    continue
  elif [[ -f "$report" ]]; then
    echo "SKIP (report exists): $report"
    ok=$((ok + 1))
  else
    echo "═══════ Phase 3: FTDC review $stem ═══════"
    if ! "$PY" main.py review "$src_srt" --translated "$zh_srt" \
      --adjudication "$adj_json" --endpoint "$ENDPOINT" \
      --source-lang "$SOURCE_LANG" --target-lang "$TARGET_LANG" \
      --max-tokens "$MAX_TOKENS" --vocab "$VOCAB" \
      --extra-payload "$EXTRA_PAYLOAD"; then
      echo "FAIL review: $rel"
      fail=$((fail + 1))
      continue
    fi
    ok=$((ok + 1))
  fi
done

echo "═══════ Phase 4: organize output ═══════"
"$PY" main.py organize || echo "WARN organize exited $?"

echo "RESUME SUMMARY: ok=$ok skip=$skip fail=$fail media=${#MEDIA[@]}"
echo "ALL DONE"
exit 0
