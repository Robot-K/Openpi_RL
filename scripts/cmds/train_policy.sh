#!/usr/bin/env bash
# 优势条件策略训练（π₀.₆*）。
# 需要数据集中已有 is_good_action 列（由 kfold_eval.sh 或 vf_label.sh 生成）。
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
POLICY_CONFIG=pi06_rl_pretrain_airbot_clothes_folding
EXP_NAME=policy_v3_wospatiodelta_iter4
GPUS=0,1,2,3,4,5,6,7
OVERWRITE=true   # false = resume
TMP_ROOT=./.tmp/train_policy
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
mkdir -p "${TMP_ROOT}" "${TMP_ROOT}/jax_cache" "${TMP_ROOT}/xdg_cache"
LOG_FILE="logs/train_policy_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
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
HF_HOME=$HOME/.cache/huggingface \
JAX_COMPILATION_CACHE_DIR="${TMP_ROOT}/jax_cache" \
    uv run scripts/train.py "${POLICY_CONFIG}" \
        --exp-name "${EXP_NAME}" \
        "${FLAG}"
