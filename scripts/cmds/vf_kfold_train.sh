#!/usr/bin/env bash
# K-fold Phase 1：并行训练 K 个 Value Function（每个排除一个 fold）。
# 每 GPUS_PER_FOLD 个连续 GPU 分配给一个 fold。
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
REPO_ID=Fold_clothes_v3
VF_CONFIG=pi06_rl_vf_airbot_clothes_folding
GPUS=(1 3 4 5 6 7)      # 可用 GPU 列表
NUM_FOLDS=3
NUM_TRAIN_STEPS=40000
GPUS_PER_FOLD=2          # 每个 fold 使用多少个连续 GPU
EXP_PREFIX=kfold_v3_iter4
RESUME=false             # true = 从已有 checkpoint 继续；false = 从头训练
TMP_ROOT=./.tmp/vf_kfold_train
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
mkdir -p "${TMP_ROOT}" "${TMP_ROOT}/jax_cache" "${TMP_ROOT}/xdg_cache"
LOG_FILE="logs/kfold_train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
echo "Using TMP_ROOT: ${TMP_ROOT}"
echo "===== K-fold Phase 1: Training ${NUM_FOLDS} VFs ====="

if [[ "$RESUME" == "true" ]]; then
    RESUME_FLAG="--resume"
else
    RESUME_FLAG="--overwrite"
fi

TRAIN_PIDS=()
for (( fold=0; fold<NUM_FOLDS; fold++ )); do
    gpu_start=$(( fold * GPUS_PER_FOLD ))
    fold_gpus=""
    for (( g=0; g<GPUS_PER_FOLD; g++ )); do
        idx=$(( gpu_start + g ))
        [[ -n "$fold_gpus" ]] && fold_gpus="${fold_gpus},"
        fold_gpus="${fold_gpus}${GPUS[$idx]}"
    done
    exp_name="${EXP_PREFIX}_fold${fold}"
    fold_tmp_dir="${TMP_ROOT}/fold${fold}"
    mkdir -p "${fold_tmp_dir}"

    echo "[Fold ${fold}] GPUs ${fold_gpus}, exp=${exp_name} (${RESUME_FLAG})"

    CUDA_VISIBLE_DEVICES="${fold_gpus}" \
    TMPDIR="${fold_tmp_dir}" \
    TEMP="${fold_tmp_dir}" \
    TMP="${fold_tmp_dir}" \
    XDG_CACHE_HOME="${TMP_ROOT}/xdg_cache" \
    HF_LEROBOT_HOME=./lerobot_data \
    HF_HOME=$HOME/.cache/huggingface \
    JAX_COMPILATION_CACHE_DIR="${TMP_ROOT}/jax_cache" \
        uv run scripts/train.py "${VF_CONFIG}" \
            --exp-name "${exp_name}" \
            --data.repo-id "${REPO_ID}" \
            --data.exclude-fold "${fold}" \
            --num-train-steps "${NUM_TRAIN_STEPS}" \
            "${RESUME_FLAG}" \
        > "logs/kfold_train_fold${fold}.log" 2>&1 &

    TRAIN_PIDS+=($!)
    echo "  PID: ${TRAIN_PIDS[-1]}, log: logs/kfold_train_fold${fold}.log"
done

echo "Waiting for all ${NUM_FOLDS} training jobs..."
for pid in "${TRAIN_PIDS[@]}"; do
    wait "$pid"
    status=$?
    if [[ $status -ne 0 ]]; then
        echo "ERROR: training PID ${pid} failed (exit ${status})" >&2
        exit 1
    fi
done

echo "===== K-fold Phase 1 COMPLETE ====="
