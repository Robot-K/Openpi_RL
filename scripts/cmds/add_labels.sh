#!/usr/bin/env bash
# 基于成功/失败标签计算 binned_value、intervention，并分配 K-fold。
# 全量重新标注：即使是 resume 后追加的新 episode 也需重跑。
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
# 合并后的 LeRobot 数据集：lerobot_data/fold_clothv3
# CONFIG 可留空；只有需要从某个 config.py 读取 FAILED_EPISODES/STAGE_BOUNDARIES 时再填写。
CONFIG=
REPO_ID=fold_clothv3
NUM_FOLDS=3
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
LOG_FILE="logs/add_labels_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

REPO_FLAG=""
if [[ -n "$REPO_ID" ]]; then
    REPO_FLAG="--repo-id ${REPO_ID}"
fi

CONFIG_FLAG=""
if [[ -n "$CONFIG" ]]; then
    CONFIG_FLAG="--config ${CONFIG}"
fi

HF_LEROBOT_HOME=./lerobot_data uv run scripts/add_returns_to_lerobot.py add_labels \
    --num-folds "${NUM_FOLDS}" \
    ${CONFIG_FLAG} \
    ${REPO_FLAG}
