#!/usr/bin/env bash
set -euo pipefail
# Selected local workflow. Preparation is CPU-only; --execute starts generation.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
PY="${PY:-.venv/bin/python3}"
case "${1:-}" in
  context-revisit) shift; exec "$PY" -m src.context_revisit "$@" ;;
  sparse-revisit) shift; exec "$PY" -m src.sparse_revisit "$@" ;;
  revisit) shift; exec "$PY" scripts/run_revisit_loop.py "$@" ;;
  archive|release) exec "$PY" main.py "$@" ;;
esac
exec "$PY" -m src.selected_pipeline --profile profiles/selected-local.json "$@"
