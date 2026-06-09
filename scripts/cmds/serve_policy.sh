#!/usr/bin/env bash
# 启动 policy 推理服务（WebSocket server）。
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
POLICY_CONFIG=pi06_rl_pretrain_airbot_clothes_folding
CHECKPOINT_DIR=checkpoints/pi06_rl_pretrain_airbot_clothes_folding/policy_iter0/XXXXX
PORT=8000
TMP_ROOT=./.tmp/serve_policy
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
mkdir -p "${TMP_ROOT}" "${TMP_ROOT}/jax_cache" "${TMP_ROOT}/xdg_cache"
LOG_FILE="logs/serve_policy_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
echo "Using TMP_ROOT: ${TMP_ROOT}"

TMPDIR="${TMP_ROOT}" \
TEMP="${TMP_ROOT}" \
TMP="${TMP_ROOT}" \
XDG_CACHE_HOME="${TMP_ROOT}/xdg_cache" \
JAX_COMPILATION_CACHE_DIR="${TMP_ROOT}/jax_cache" \
uv run scripts/serve_policy.py \
    --port "${PORT}" \
    policy:checkpoint \
    --policy.config "${POLICY_CONFIG}" \
    --policy.dir "${CHECKPOINT_DIR}"
