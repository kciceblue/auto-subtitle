#!/usr/bin/env bash
# Evidence-first local subtitle workflow (WORKFLOW.md).
#   ./run.sh                    every media file under input/ -> output/<stem>/final/
#   ./run.sh MEDIA [options]    one file; options go to `python -m src.evidence_first`
#   ./run.sh [options]          every file under input/ with the same options (e.g. --until draft)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
PY="${PY:-.venv/bin/python3}"

if [ $# -gt 0 ] && [ -f "$1" ]; then
  exec "$PY" -m src.evidence_first "$@"
fi

status=0 count=0
while IFS= read -r -d '' media; do
  count=$((count + 1))
  echo "== $media"
  "$PY" -m src.evidence_first "$media" "$@" || { echo "FAILED: $media" >&2; status=1; }
done < <(find input -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.webm' -o -iname '*.mov' \
  -o -iname '*.avi' -o -iname '*.ts' -o -iname '*.m4v' -o -iname '*.mp3' -o -iname '*.wav' \
  -o -iname '*.flac' -o -iname '*.m4a' -o -iname '*.aac' -o -iname '*.ogg' -o -iname '*.opus' \) -print0 | sort -z)
[ "$count" -gt 0 ] || echo "No media found under input/" >&2
exit $status
