#!/usr/bin/env bash
# Evidence-first local subtitle workflow (WORKFLOW.md).
#   ./run.sh                    every media file under input/ -> output/<stem>/final/
#   ./run.sh MEDIA [options]    one file; options go to `python -m src.evidence_first`
#   ./run.sh [options]          every file under input/ with the same options (e.g. --until draft)
#   ./run.sh --check [MEDIA]    only the preflight check (src/preflight.py)
# Every run starts with the preflight check; SKIP_PREFLIGHT=1 skips it.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
PY="${PY:-.venv/bin/python3}"

preflight() {
  [ "${SKIP_PREFLIGHT:-0}" = 1 ] && return
  "$PY" -m src.preflight "$@" || { echo "Preflight failed: fix the items above (SKIP_PREFLIGHT=1 overrides)." >&2; exit 1; }
}

if [ "${1:-}" = --check ]; then
  shift
  exec "$PY" -m src.preflight "$@"
fi

if [ $# -gt 0 ] && [ -f "$1" ]; then
  preflight "$1"
  exec "$PY" -m src.evidence_first "$@"
fi
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
  echo "Media file not found: $1" >&2
  exit 1
fi

media=()
while IFS= read -r -d '' file; do
  media+=("$file")
done < <(find input -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.webm' -o -iname '*.mov' \
  -o -iname '*.avi' -o -iname '*.ts' -o -iname '*.m4v' -o -iname '*.mp3' -o -iname '*.wav' \
  -o -iname '*.flac' -o -iname '*.m4a' -o -iname '*.aac' -o -iname '*.ogg' -o -iname '*.opus' \) -print0 | sort -z)
if [ ${#media[@]} -eq 0 ]; then
  echo "No media found under input/" >&2
  exit 0
fi
preflight "${media[@]}"

status=0
for file in "${media[@]}"; do
  echo "== $file"
  "$PY" -m src.evidence_first "$file" "$@" || { echo "FAILED: $file" >&2; status=1; }
done
exit $status
