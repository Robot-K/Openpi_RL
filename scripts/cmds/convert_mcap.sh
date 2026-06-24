#!/usr/bin/env bash
# 将 mcap 采集数据转换为 LeRobot 格式。
#
# 使用场景：
#   1. 同一 mcap 目录中断后恢复（RESUME=true，不设 SKIP_EPISODES）
#      → skip_episodes 自动取现有 dataset 的 episode 数，跳过已转换的文件。
#
#   2. 将新 mcap 目录追加到已有 dataset（RESUME=true，SKIP_EPISODES=0）
#      → 新目录中的所有文件都会被追加，不跳过任何文件。
#      适用场景：fold_clothv2 已转换完毕，现在要把 fold_clothv2_dagger1 追加进去。
#
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR=/data/kding/FastWAM_PI_RL/mcap_data/fold_cloth_tv_dagger1    # mcap 数据目录（含 config.py），绝对路径
REPO_ID=fold_clothv3                  # 输出/追加到 lerobot_data 下的 dataset 名
RESUME=true                       # true = 在SKIP_EPISODES基础上追加新 episode；false = 全量转换
OVERWRITE=false                    # true = 重新转换覆盖旧数据；false = 保留旧数据（仅当 RESUME=false 时有效）
INTERVENTION_ONLY=false            # true = 仅转换连续 intervention=1 片段，每段单独成为一个 episode
MIN_INTERVENTION_FRAMES=16         # intervention_only=true 时生效；短于该帧数的 intervention 片段直接丢弃
HF_LOAD_NUM_PROC=16                # resume 时并行加载已有 LeRobot parquet，加速 "Generating train split"
# 仅在 RESUME=true 时有效：
#   -1  → 自动检测（适合在同一 mcap 目录中断恢复）
#    0  → 从头开始追加（适合把新 mcap 目录的数据追加到已有 dataset）
SKIP_EPISODES=0
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
LOG_FILE="logs/convert_mcap_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

ARGS=""
if [[ "$RESUME" == "true" ]]; then
    ARGS="--resume --skip_episodes ${SKIP_EPISODES}"
elif [[ "$OVERWRITE" == "true" ]]; then
    ARGS="--force"
fi

if [[ "$INTERVENTION_ONLY" == "true" ]]; then
    ARGS="${ARGS} --intervention-only --min-intervention-frames ${MIN_INTERVENTION_FRAMES}"
fi

# --no-sync: 复用现有 .venv（已装 av 17.0.0），避免 uv 按 lock 把 av 降级为
# 14.4.0 并从源码编译，进而撞上系统旧 FFmpeg 4.4 头文件而编译失败。
HF_LEROBOT_HOME=./lerobot_data \
LEROBOT_HF_LOAD_NUM_PROC="${HF_LOAD_NUM_PROC}" \
uv run --no-sync examples/airbot/convert_mcap_data_to_lerobot.py \
    --data_dir "${DATA_DIR}" \
    --repo_id "${REPO_ID}" \
    ${ARGS}
