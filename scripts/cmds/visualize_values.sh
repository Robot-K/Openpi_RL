#!/usr/bin/env bash
# 离线可视化 episode 的相机采样图、state/action 轨迹、VF 标签分布。
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR=./lerobot_data/fold_clothv3
OUTPUT_DIR=./assets/visualizations_values/chunk010
CHUNK=10
EPISODES=0,100,200,300,400,500,600,700,800,900          # chunk 内下标；留空则处理全部
NUM_CAMERA_SAMPLES=5
SKIP_CAMERAS=false      # true = 跳过相机图像，加快速度
SKIP_EPISODE_VIDEO=true
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs "${OUTPUT_DIR}"
LOG_FILE="logs/visualize_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

EPISODES_FLAG=""
if [[ -n "$EPISODES" ]]; then
    EPISODES_FLAG="--episodes ${EPISODES}"
fi

SKIP_FLAG=""
if [[ "$SKIP_CAMERAS" == "true" ]]; then
    SKIP_FLAG="--skip_cameras"
fi

VIDEO_FLAG=""
if [[ "$SKIP_EPISODE_VIDEO" == "true" ]]; then
    VIDEO_FLAG="--skip_episode_video"
fi

uv run scripts/tools/visualize_values.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --chunk "${CHUNK}" \
    --num_camera_samples "${NUM_CAMERA_SAMPLES}" \
    ${EPISODES_FLAG} \
    ${SKIP_FLAG} \
    ${VIDEO_FLAG}
