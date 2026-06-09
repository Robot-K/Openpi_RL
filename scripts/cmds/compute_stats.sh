#!/usr/bin/env bash
# 计算数据集归一化统计量（均值、标准差等），供训练使用。
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
FUNC_CONFIG=pi06_rl_pretrain_airbot_clothes_folding
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
LOG_FILE="logs/compute_stats_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

HF_LEROBOT_HOME=./lerobot_data uv run scripts/compute_norm_stats.py \
    --config-name "${FUNC_CONFIG}"
