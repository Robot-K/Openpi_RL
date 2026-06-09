#!/usr/bin/env bash
# 可视化 pi0 模型矢量场 f(x_t, t) = v_t。
# 支持同时指定多个 episode:timestep，只加载所需 episode，避免全量 split 扫描。
#
# SAMPLES 格式："episode:timestep episode:timestep ..."，空格分隔
# 输出路径: {OUTPUT_DIR}/{exp_name}/{step}/episode{ep:04d}_ts{ts:04d}/t{t:.1f}.png

# 设置脚本选项：
# -e: 如果任何命令执行失败（返回非零状态），脚本立即退出
# -u: 如果使用未定义的变量，脚本立即退出并报错
# -o pipefail: 如果管道中的任何命令失败，整个管道返回该失败状态
#
# 获取当前脚本所在目录的父目录的父目录，并将其设为工作目录
# $(dirname "$0"): 获取脚本所在目录的路径
# /../..: 向上遍历两级目录
# cd: 切换到该目录作为当前工作目录
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG_NAME=pi06_rl_pretrain_airbot_clothes_folding
EXP_NAME=policy_v3_wospatiodelta_iter1
STEP=140000
# 每行一个 episode:timestep，空格分隔可写多个
SAMPLES="400:100 500:200 600:300 700:400"
GPU=4,5,6,7
N_GRID=16       # 每维格点数（生成 N_GRID×N_GRID 个样本点）
GRID_MIN=-2.5   # normalized space 扫描范围
GRID_MAX=2.5
OUTPUT_DIR=./assets/visualizations_vectors
TMP_ROOT=./.tmp/visualize_vf
# ───────────────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR="checkpoints/${CONFIG_NAME}/${EXP_NAME}/${STEP}"

mkdir -p logs "${TMP_ROOT}" "${TMP_ROOT}/jax_cache" "${TMP_ROOT}/xdg_cache" "${OUTPUT_DIR}"
LOG_FILE="logs/visualize_vf_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
echo "Checkpoint : ${CHECKPOINT_DIR}"
echo "Samples    : ${SAMPLES}"
echo "Output     : ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" \
TMPDIR="${TMP_ROOT}" \
TEMP="${TMP_ROOT}" \
TMP="${TMP_ROOT}" \
XDG_CACHE_HOME="${TMP_ROOT}/xdg_cache" \
HF_LEROBOT_HOME=./lerobot_data \
JAX_COMPILATION_CACHE_DIR="${TMP_ROOT}/jax_cache" \
    uv run scripts/tools/visualize_vector_field.py \
        --config-name "${CONFIG_NAME}" \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --samples "${SAMPLES}" \
        --output-dir "${OUTPUT_DIR}" \
        --n-grid "${N_GRID}" \
        --grid-range "${GRID_MIN}" "${GRID_MAX}" \
        --gpu "${GPU}"
