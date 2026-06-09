#!/usr/bin/env bash
# 单 VF 对全数据集打分（不使用 K-fold，旧方式）。
# K-fold 流程请用 kfold_eval.sh。
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
REPO_ID=Fold_clothes
VF_CONFIG=pi06_rl_vf_airbot_clothes_folding
VF_CHECKPOINT_DIR=checkpoints/pi06_rl_vf_airbot_clothes_folding/vf_v1/20000
GPUS=0,1
POSITIVE_FRACTION=0.3
BATCH_SIZE=32
TMP_ROOT=./.tmp/vf_label
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
mkdir -p "${TMP_ROOT}" "${TMP_ROOT}/jax_cache" "${TMP_ROOT}/xdg_cache"
LOG_FILE="logs/vf_label_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
echo "Using TMP_ROOT: ${TMP_ROOT}"

CUDA_VISIBLE_DEVICES="${GPUS}" \
TMPDIR="${TMP_ROOT}" \
TEMP="${TMP_ROOT}" \
TMP="${TMP_ROOT}" \
XDG_CACHE_HOME="${TMP_ROOT}/xdg_cache" \
HF_LEROBOT_HOME=./lerobot_data \
JAX_COMPILATION_CACHE_DIR="${TMP_ROOT}/jax_cache" \
    uv run scripts/add_returns_to_lerobot.py vf_label \
        --repo-id "${REPO_ID}" \
        --vf-config "${VF_CONFIG}" \
        --vf-checkpoint-dir "${VF_CHECKPOINT_DIR}" \
        --positive-fraction "${POSITIVE_FRACTION}" \
        --batch-size "${BATCH_SIZE}"
