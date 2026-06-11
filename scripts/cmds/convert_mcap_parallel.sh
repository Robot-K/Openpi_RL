#!/usr/bin/env bash
# 并行把多个 mcap 数据目录转换为 LeRobot 格式。
#
# 每个目录各自独立输出（由其 config.py 的 TASK_NAME 决定输出目录名），
# 互不干扰，可同时跑。每个任务一个独立日志。
#
# 用法：bash scripts/cmds/convert_mcap_parallel.sh
#   停止全部：见脚本结尾打印的 PID，或 pkill -f convert_mcap_data_to_lerobot
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
# 要并行转换的 mcap 目录（每个目录需含 config.py，且 TASK_NAME 两两不同）。
DATA_DIRS=(
    /data/kding/FastWAM/mcap_data/fold_clothv3_dagger
    /data/kding/FastWAM/mcap_data/fold_clothv3_wam_dagger
    /data/kding/FastWAM/mcap_data/fold_clothv3_wuxi
    /data/kding/FastWAM/mcap_data/fold_clothv3_wuxi_airdc
    /data/kding/FastWAM/mcap_data/fold_clothv3_wuxi_dagger
)
# --resume --skip_episodes -1：首次全量；中断后再跑会自动从已转 episode 续传，不重复。
EXTRA_ARGS="--resume --skip_episodes -1"
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
echo "并行启动 ${#DATA_DIRS[@]} 个转换任务（日志见 logs/convert_<name>_${STAMP}.log）"
echo "──────────────────────────────────────────────"

PIDS=()
for d in "${DATA_DIRS[@]}"; do
    name="$(basename "$d")"
    log="logs/convert_${name}_${STAMP}.log"
    HF_LEROBOT_HOME=./lerobot_data uv run --no-sync \
        examples/airbot/convert_mcap_data_to_lerobot.py \
        --data_dir "$d" ${EXTRA_ARGS} > "$log" 2>&1 &
    pid=$!
    PIDS+=("$pid")
    printf "  %-30s PID=%-8s log=%s\n" "$name" "$pid" "$log"
done

echo "──────────────────────────────────────────────"
echo "全部已在后台运行。PIDs: ${PIDS[*]}"
echo "实时看某个：tail -f logs/convert_<name>_${STAMP}.log"
echo "停止全部：  pkill -f convert_mcap_data_to_lerobot"
echo ""
echo "等待全部完成中（可 Ctrl+C 退出等待，不影响后台任务）..."
wait
echo "✅ 全部转换任务结束。"
