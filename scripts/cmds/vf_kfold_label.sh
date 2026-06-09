#!/usr/bin/env bash
# K-fold Phase 2+3：每个 VF 对其留出 fold 推理，合并后写入 is_good_action。
# CHECKPOINT_STEP 留空时自动取 fold0 目录下最大的数字子目录。
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
REPO_ID=Fold_clothes_v3
VF_CONFIG=pi06_rl_vf_airbot_clothes_folding
GPUS=(0 1 2 3 4 5)      # 可用 GPU 列表
NUM_FOLDS=3
GPUS_PER_FOLD=2
EXP_PREFIX=kfold_v3_iter4
CHECKPOINT_STEP=20000        # 留空 = 自动检测最新 checkpoint
BATCH_SIZE=48
POSITIVE_FRACTION=0.3
GAMMA=0.985
ACTION_HORIZON=50
TMP_ROOT=./.tmp/vf_kfold_label
VALUES_DIR="${TMP_ROOT}/values"
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
mkdir -p "${TMP_ROOT}" "${TMP_ROOT}/jax_cache" "${TMP_ROOT}/xdg_cache" "${TMP_ROOT}/merge"
LOG_FILE="logs/kfold_eval_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
echo "Using TMP_ROOT: ${TMP_ROOT}"

# Auto-detect checkpoint step from fold 0 if not set
if [[ -z "$CHECKPOINT_STEP" ]]; then
    _base="checkpoints/${VF_CONFIG}/${EXP_PREFIX}_fold0"
    if [[ -d "$_base" ]]; then
        CHECKPOINT_STEP=$(ls -1 "$_base" | grep -E '^[0-9]+$' | sort -n | tail -1)
    fi
    [[ -z "$CHECKPOINT_STEP" ]] && { echo "ERROR: cannot detect checkpoint step" >&2; exit 1; }
fi
echo "Using checkpoint step: ${CHECKPOINT_STEP}"

# ── Phase 2: Inference ────────────────────────────────────────────────────────
echo ""
echo "===== Phase 2: VF inference ====="
mkdir -p "${VALUES_DIR}"

INFER_PIDS=()
for (( fold=0; fold<NUM_FOLDS; fold++ )); do
    gpu_start=$(( fold * GPUS_PER_FOLD ))
    fold_gpus=""
    for (( g=0; g<GPUS_PER_FOLD; g++ )); do
        idx=$(( gpu_start + g ))
        [[ -n "$fold_gpus" ]] && fold_gpus="${fold_gpus},"
        fold_gpus="${fold_gpus}${GPUS[$idx]}"
    done
    exp_name="${EXP_PREFIX}_fold${fold}"
    ckpt_dir="checkpoints/${VF_CONFIG}/${exp_name}/${CHECKPOINT_STEP}"

    echo "[Fold ${fold}] GPUs ${fold_gpus}, ckpt=${ckpt_dir}"

    fold_tmp_dir="${TMP_ROOT}/fold${fold}"
    mkdir -p "${fold_tmp_dir}"

    CUDA_VISIBLE_DEVICES="${fold_gpus}" \
    TMPDIR="${fold_tmp_dir}" \
    TEMP="${fold_tmp_dir}" \
    TMP="${fold_tmp_dir}" \
    XDG_CACHE_HOME="${TMP_ROOT}/xdg_cache" \
    HF_HOME=$HOME/.cache/huggingface \
    HF_LEROBOT_HOME=./lerobot_data \
    JAX_COMPILATION_CACHE_DIR="${TMP_ROOT}/jax_cache" \
        uv run scripts/add_returns_to_lerobot.py vf_label \
            --repo-id "${REPO_ID}" \
            --vf-config "${VF_CONFIG}" \
            --vf-checkpoint-dir "${ckpt_dir}" \
            --infer-fold "${fold}" \
            --batch-size "${BATCH_SIZE}" \
            --values-dir "${VALUES_DIR}" \
        > "logs/kfold_eval_fold${fold}.log" 2>&1 &

    INFER_PIDS+=($!)
    echo "  PID: ${INFER_PIDS[-1]}, log: logs/kfold_eval_fold${fold}.log"
done

echo "Waiting for all ${NUM_FOLDS} inference jobs..."
for pid in "${INFER_PIDS[@]}"; do
    wait "$pid"
    status=$?
    if [[ $status -ne 0 ]]; then
        echo "ERROR: inference PID ${pid} failed (exit ${status})" >&2
        exit 1
    fi
done
echo "===== Phase 2 COMPLETE ====="

# ── Phase 3: Merge ────────────────────────────────────────────────────────────
echo ""
echo "===== Phase 3: Merge values + compute is_good_action ====="

TMPDIR="${TMP_ROOT}/merge" \
TEMP="${TMP_ROOT}/merge" \
TMP="${TMP_ROOT}/merge" \
XDG_CACHE_HOME="${TMP_ROOT}/xdg_cache" \
HF_LEROBOT_HOME=./lerobot_data \
JAX_COMPILATION_CACHE_DIR="${TMP_ROOT}/jax_cache" \
uv run scripts/add_returns_to_lerobot.py vf_merge \
    --repo-id "${REPO_ID}" \
    --values-dir "${VALUES_DIR}" \
    --positive-fraction "${POSITIVE_FRACTION}" \
    --gamma "${GAMMA}" \
    --action-horizon "${ACTION_HORIZON}"

echo "===== Phase 3 COMPLETE — is_good_action written to dataset ====="
