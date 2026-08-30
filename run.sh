#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════
# auto-subtitle 统一流程编排器（2026-08-30 整合版）
#
# 完整流水线：ASR → 翻译 → proofread → 三模型音频仲裁 → FTDC review（台本穿透）
#             → organize（output/<unit>/{final,review}/ + input 清理）
#
# 用法：
#   ./run.sh                          # 处理 input/ 下所有媒体文件
#   ./run.sh input/foo.wav            # 处理单个文件
#   ./run.sh --no-demucs input/foo.wav   # 其余参数透传给 main.py pipeline
#   ./run.sh archive [unit ...]       # 手动归档：output/<unit> + zip →
#                                     #   ready_for_human_review/（默认全部已整理单元）
#
# 环境变量（默认全开，设 0 关闭对应阶段）：
#   PROOFREAD=0   跳过 proofread 一致性命中 pass
#   ADJUDICATE=0  跳过三模型音频仲裁（whisper+zipformer CPU+qwen3-asr GPU）
#   REVIEW=0      跳过 FTDC 语义终审（triage→DeepFix→critic）
#   ORGANIZE=0    跳过 Phase 4 整理（output/<unit>/{final,review}/ + input 清理）
#   NO_DEMUCS=1   跳过人声分离（BGM 重的素材建议保留 demucs）
#   CONTEXT="a.pdf b.txt"   显式 context（默认自动内容扫描：分类 script/synopsis）
#   WARDEN_UNLOAD=0   ASR 前不驱逐 warden LLM（GPU 空间不足时勿关）
#
# 设计要点（经验沉淀）：
#   - review 依赖仲裁证据（B+/B- 音频投票 + 台本锚点），所以分阶段编排：
#     pipeline 只到 proofread；仲裁独立子进程跑（GPU 模型短命进程，退出即
#     归还 VRAM，asr_worker 哲学）；review 由 warden lazy-load dflash；
#     最后 organize 收尾（media+SRT → final/，其余 → review/）。
#   - MEDIA 清单必须在 Phase 1 之前采集：批量模式的 pipeline 结束时把
#     input/ 整体搬去 output/，之后再 find input/ 只会得到空集（8/30 bug：
#     批量模式 Phase 2+3 因此从未运行过）。
#   - 仲裁子进程内部自动做 GPU headroom 检查（ensure_gpu_headroom 通用化：
#     显存不足自动 /admin/unload 驱逐 warden LLM），不会与 27B dflash 抢显存。
#   - review 子命令 main.py review 自动内容扫描（script 类 context 自动作为
#     台本 ground-truth 穿透 triage/DeepFix/critic），并同时扫描媒体所在目录
#     （单文件模式下 context 仍留在 input/ 侧）。
# ═══════════════════════════════════════════════════════════════════════

# ── Configurable settings ────────────────────────────────────────────
ENDPOINT="${ENDPOINT:-http://127.0.0.1:8089/v1/chat/completions}"
SOURCE_LANG="${SOURCE_LANG:-Japanese}"
TARGET_LANG="${TARGET_LANG:-Simplified Chinese}"
WHISPER_LANG="${WHISPER_LANG:-ja}"
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
COMPUTE_TYPE="${COMPUTE_TYPE:-float16}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
TIMEOUT="${TIMEOUT:-300}"
# Token budget per LLM request. Sent explicitly: with the server default a
# reasoning model can burn the whole budget on thinking and return nothing.
# 16384 = FTDC 实测（2026-08-30）：dflash DeepFix chunks 的 reasoning 常达
# 6k-13k tokens，8192 会被吃光触发重试；streaming + EOS 提前停止使上限只是保险丝。
MAX_TOKENS="${MAX_TOKENS:-16384}"
NO_DEMUCS="${NO_DEMUCS:-}"
# 阶段开关（默认全开 = 完整流程）
PROOFREAD="${PROOFREAD:-1}"
ADJUDICATE="${ADJUDICATE:-1}"
REVIEW="${REVIEW:-1}"
ORGANIZE="${ORGANIZE:-1}"
# Evict the warden LLM from the GPU before ASR when VRAM headroom is short
# (whisper + the 27B translator exceed 32 GB together). Set 0 to disable.
WARDEN_UNLOAD="${WARDEN_UNLOAD:-1}"
# temperature 0.3: the warden dflash default is 1.0, which drifts into
# English on some CJK subtitle lines (high temp). Low temperature keeps the
# numbered translation output stable.
# Model = qwen3.8-27b-dflash since 2026-08-30: FTDC 全片实测 27.8 min / 0 溢出 /
# 0 拒绝，质量优于 hauhau（hauhau 的 Q4_K_P+MTP 在 DeepFix 上 ~25% chunks
# 思考失控——65536 tokens 被吃光仍空响应）。hauhau 仍可手动 EXTRA_PAYLOAD 切换。
DEFAULT_PAYLOAD='{"model": "qwen3.8-27b-dflash", "temperature": 0.3}'
EXTRA_PAYLOAD="${EXTRA_PAYLOAD:-$DEFAULT_PAYLOAD}"
VOCAB="${VOCAB:-./vocab.txt}"
# Context file(s) for hotword + translation grounding (space-separated paths).
# Leave empty to auto-discover (content-aware scan) near each input file.
CONTEXT="${CONTEXT:-}"

# ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
PY="${PY:-.venv/bin/python3}"

# ── Subcommand: archive (manual hand-off to ready_for_human_review/) ──
if [[ "${1:-}" == "archive" ]]; then
  shift
  exec "$PY" main.py archive "$@"
fi

# Explicit context files, one --context per path (argparse action="append").
CONTEXT_FILES=()
if [[ -n "$CONTEXT" ]]; then
  read -r -a CONTEXT_FILES <<< "$CONTEXT"
fi

# Build pipeline args
PIPELINE_ARGS=(
  pipeline
  -l "$WHISPER_LANG"
  -m "$WHISPER_MODEL"
  --compute-type "$COMPUTE_TYPE"
  --endpoint "$ENDPOINT"
  --source-lang "$SOURCE_LANG"
  --target-lang "$TARGET_LANG"
  --chunk-size "$CHUNK_SIZE"
  --timeout "$TIMEOUT"
  --max-tokens "$MAX_TOKENS"
)

[[ -n "$NO_DEMUCS" ]] && PIPELINE_ARGS+=(--no-demucs)
[[ -n "$EXTRA_PAYLOAD" ]] && PIPELINE_ARGS+=(--extra-payload "$EXTRA_PAYLOAD")
[[ -n "$VOCAB" ]] && PIPELINE_ARGS+=(--vocab "$VOCAB")
[[ "$PROOFREAD" == "1" ]] && PIPELINE_ARGS+=(--proofread)
[[ "$WARDEN_UNLOAD" == "0" ]] && PIPELINE_ARGS+=(--no-warden-unload)

# User arguments — including any positional media file — go in BEFORE the
# context flags. Previously "$@" trailed the whole array, so a trailing
# --context absorbed the user's media file as another context path and
# input_file silently became None (batch mode over input/).
#
# Only arguments that LOOK like media (known extension) are remembered for
# the arbitration/review phases: a flag value such as `--vocab my.txt` is an
# existing non-dash path too, but it is not media.
USER_ARGS=()
if [[ $# -gt 0 ]]; then
  for arg in "$@"; do
    if [[ "$arg" != -* && -e "$arg" ]]; then
      case "${arg,,}" in
        *.wav|*.mp3|*.m4a|*.flac|*.ogg|*.aac|*.wma|\
        *.mkv|*.mp4|*.webm|*.avi|*.mov|*.flv|*.wmv)
          USER_ARGS+=("$arg") ;;
      esac
    fi
    PIPELINE_ARGS+=("$arg")
  done
fi

for ctx in ${CONTEXT_FILES[@]+"${CONTEXT_FILES[@]}"}; do
  PIPELINE_ARGS+=(--context "$ctx")
done

# Resolve the media list BEFORE Phase 1: batch-mode pipeline moves input/ to
# output/ when it finishes, so a find run afterwards sees an empty input/ and
# Phases 2-4 silently do nothing (the 8/30 batch bug).
if [[ ${#USER_ARGS[@]} -gt 0 ]]; then
  MEDIA=("${USER_ARGS[@]}")
else
  MEDIA=()
  while IFS= read -r -d '' f; do MEDIA+=("$f"); done < <(
    find input -type f \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.m4a' \
      -o -iname '*.flac' -o -iname '*.ogg' -o -iname '*.aac' \
      -o -iname '*.mkv' -o -iname '*.mp4' -o -iname '*.webm' \) -print0
  )
fi

# ── Phase 1: transcribe + translate (+ proofread) ────────────────────
echo "═══════ Phase 1: pipeline (ASR → translate → proofread) ═══════"
"$PY" main.py "${PIPELINE_ARGS[@]}"

if [[ "$ADJUDICATE" == "1" || "$REVIEW" == "1" ]]; then

# ── Phases 2+3: adjudication (subprocess) → FTDC review ─────────────
# Media files live in input/ while transcribing; main.py pipeline moves them
# to output/ when done (batch mode) — resolve each file in both trees.
for media in "${MEDIA[@]}"; do
  rel="${media#./}"
  rel="${rel#input/}"
  rel="${rel#output/}"
  # Media outside input/ or output/ (absolute or elsewhere): main.py falls
  # back to output/<name>.srt, so mirror that here.
  [[ "$rel" == /* ]] && rel="$(basename "$media")"
  stem="${rel%.*}"
  src_srt="output/${stem}.srt"
  zh_srt="output/${stem}.zh.srt"
  adj_json="output/${stem}.adjudication.json"
  # The review report lands NEXT TO the translated SRT, so a nested rel
  # (sub/foo) must become output/sub/REVIEW-foo.zh.md, not output/REVIEW-sub/…
  rel_dir="$(dirname "$stem")"
  base="$(basename "$stem")"
  report="output/${rel_dir}/REVIEW-${base}.zh.md"
  [[ -f "$src_srt" && -f "$zh_srt" ]] || { echo "SKIP (no srt/zh): $rel"; continue; }
  media_now="$media"
  [[ -f "$media_now" ]] || media_now="output/$rel"
  [[ -f "$media_now" ]] || { echo "SKIP (media gone): $rel"; continue; }

  # Phase 2: three-model arbitration in a SHORT-LIVED SUBPROCESS — the GPU
  # model (qwen3-asr) dies with it, returning all VRAM before warden reloads
  # dflash for review. Headroom/eviction handled inside (ensure_gpu_headroom).
  if [[ "$ADJUDICATE" == "1" && ! -f "$adj_json" ]]; then
    echo "═══════ Phase 2: arbitrate $stem ═══════"
    "$PY" -m src.adjudicate --srt "$src_srt" --media "$media_now" \
      --suspicious all --out "$adj_json"
  fi

  # Phase 3: FTDC review (auto context scan; script-kind context auto-anchors)
  if [[ "$REVIEW" == "1" ]]; then
    if [[ ! -f "$adj_json" ]]; then
      echo "SKIP review (no adjudication json — run with ADJUDICATE=1): $rel"
    elif [[ -f "$report" ]]; then
      echo "SKIP (report exists): $report"
    else
      echo "═══════ Phase 3: FTDC review $stem ═══════"
      REVIEW_ARGS=(
        main.py review "$src_srt" --translated "$zh_srt"
        --adjudication "$adj_json" --endpoint "$ENDPOINT"
        --source-lang "$SOURCE_LANG" --target-lang "$TARGET_LANG"
        --max-tokens "$MAX_TOKENS"
      )
      [[ -n "$VOCAB" ]] && REVIEW_ARGS+=(--vocab "$VOCAB")
      [[ -n "$EXTRA_PAYLOAD" ]] && REVIEW_ARGS+=(--extra-payload "$EXTRA_PAYLOAD")
      # Explicit CONTEXT applies to review too (it previously reached only
      # Phase 1, so a manual context override never informed the review).
      for ctx in ${CONTEXT_FILES[@]+"${CONTEXT_FILES[@]}"}; do
        REVIEW_ARGS+=(--context-file "$ctx")
      done
      "$PY" "${REVIEW_ARGS[@]}"
    fi
  fi
done

fi  # ADJUDICATE/REVIEW

# ── Phase 4: organize output/<unit>/{final,review}/ + input cleanup ──
if [[ "$ORGANIZE" == "1" ]]; then
  echo "═══════ Phase 4: organize output ═══════"
  "$PY" main.py organize
fi

echo "ALL DONE"
