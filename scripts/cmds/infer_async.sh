#!/usr/bin/env bash
# 异步推理（Async）：推理与执行并行，支持 TCS 时序 chunk 平滑，实时性更好。
# 推荐用于正式部署。
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
HOST=127.0.0.1
PORT=8000
PROMPT="Fold clothes"


# TCS（Temporal Chunk Smoothing）平滑参数：
# tcs_min_overlap      新旧 chunk 混合（blend）时的最小重叠窗口长度；
#                      重叠区内按线性权重从旧→新渐变，消除动作跳变
# max_inference_rate    最大推理频率（Hz），限制推理请求速率，避免过度切换动作；
# initial_action_wait_s 首帧启动等待时限（秒）：episode 开始时等待第一个
#                      action chunk 到达的最长时间，超时则保持当前姿态直到推理就绪
# tcs_drop_max         推理延迟补偿：新 chunk 到达时，丢弃已过期的前 N 步
#                      （N = min(实际延迟步数, tcs_drop_max)），避免执行过时动作

MAX_INFERENCE_RATE=2  # 最大推理请求频率（Hz）；0 = 不限速
STEP_RATE=200         # 动作发布频率（Hz）
INITIAL_ACTION_WAIT_S=1.0
TCS_MIN_OVERLAP=3
TCS_DROP_MAX=12

INTERPOLATE=false      # true = 启用动作插值平滑
RECORD=false            # true = 保存 MCAP 数据
RECORD_DIR=./inference_data/dagger_1550_1560
DAGGER=true            # true = 启用 DAgger 干预采集（需要主臂连接）
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
LOG_FILE="logs/infer_async_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

cd examples/airbot

cmd=(
    python airbot_inference_async.py
    --policy-config.host "${HOST}"
    --policy-config.port "${PORT}"
    --prompt "${PROMPT}"
    --step-rate "${STEP_RATE}"
    --max-inference-rate "${MAX_INFERENCE_RATE}"
    --tcs-drop-max "${TCS_DROP_MAX}"
    --tcs-min-overlap "${TCS_MIN_OVERLAP}"
    --initial-action-wait-s "${INITIAL_ACTION_WAIT_S}"
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
