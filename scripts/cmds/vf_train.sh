#!/usr/bin/env bash
# 单次 VF 训练（不使用 K-fold）。
# 适用于快速实验；正式流程请用 kfold_train.sh + kfold_eval.sh。
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
VF_CONFIG=pi06_rl_vf_airbot_clothes_folding
EXP_NAME=vf_v1
GPUS=0,1,2,3
NUM_TRAIN_STEPS=20000
OVERWRITE=true   # false = resume
TMP_ROOT=./.tmp/vf_train
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
mkdir -p "${TMP_ROOT}" "${TMP_ROOT}/jax_cache" "${TMP_ROOT}/xdg_cache"
LOG_FILE="logs/train_vf_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
echo "Using TMP_ROOT: ${TMP_ROOT}"

FLAG="--overwrite"
[[ "$OVERWRITE" == "false" ]] && FLAG="--resume"

CUDA_VISIBLE_DEVICES="${GPUS}" \
TMPDIR="${TMP_ROOT}" \
TEMP="${TMP_ROOT}" \
TMP="${TMP_ROOT}" \
XDG_CACHE_HOME="${TMP_ROOT}/xdg_cache" \
HF_LEROBOT_HOME=./lerobot_data \
JAX_COMPILATION_CACHE_DIR="${TMP_ROOT}/jax_cache" \
    uv run scripts/train.py "${VF_CONFIG}" \
        --exp-name "${EXP_NAME}" \
        --num-train-steps "${NUM_TRAIN_STEPS}" \
        "${FLAG}"
