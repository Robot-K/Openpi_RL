#!/usr/bin/env bash
# 同步推理（Sync）：每执行完一个 chunk 才发起下一次推理，延迟感明显。
# 适用于快速验证；正式部署请用 infer_async.sh。
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
HOST=127.0.0.1
PORT=8000
PROMPT="Fold clothes"
CHUNK_SIZE_EXECUTE=20
INTERPOLATE=false      # true = 启用动作插值平滑
RECORD=true            # true = 保存 MCAP 数据
RECORD_DIR=./inference_data/dagger_1570_1580


DAGGER=true          # true = 启用 DAgger 干预采集（需要主臂连接）
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
LOG_FILE="logs/infer_sync_$(date +%Y%m%d_%H%M%S).log"


exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

cd examples/airbot

cmd=(
    python airbot_inference_sync.py
    --policy-config.host "${HOST}"
    --policy-config.port "${PORT}"
    --prompt "${PROMPT}"
    --chunk-size-execute "${CHUNK_SIZE_EXECUTE}"
)

if [ "${INTERPOLATE}" = true ]; then
    cmd+=(--interpolate)
fi

if [ "${RECORD}" = true ]; then
    cmd+=(--record.record-data --record.save-dir "${RECORD_DIR}")
fi

if [ "${DAGGER}" = true ]; then
    cmd+=(--dagger.enable)
fi

"${cmd[@]}"
