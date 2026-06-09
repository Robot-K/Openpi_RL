"""
可视化 pi0 模型的矢量场 f(x_t, t) = v_t。

对 t = 0.1, 0.2, ..., 1.0，每个 t 生成一张图，图上有 6 个 2D quiver 子图：
  - 子图 1-3：action 维度对 (1,2), (3,4), (5,6)（1-indexed）
  - 子图 4-6：action 维度对 (8,9), (10,11), (12,13)（1-indexed）

xy 轴为 x_t 对应的两个维度取值，矢量为 v_t 对应的两个维度。
其余维度固定为 0（normalized space）。

用法:
    uv run scripts/tools/visualize_vector_field.py \\
        --config-name pi06_rl_pretrain_airbot_clothes_folding \\
        --checkpoint-dir checkpoints/pi06_rl_pretrain_airbot_clothes_folding/policy_1000_wospatio_iter0/20000 \\
        --episode 0 \\
        --timestep 10 \\
        --output-dir assets/visualizations
"""

import argparse
import dataclasses
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── 路径设置 ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as _transforms


# ── 维度对定义（1-indexed → 0-indexed） ─────────────────────
# 用户指定：dims 1-6 三组, dims 8-13 三组
DIM_PAIRS = [(0, 1), (2, 3), (4, 5), (7, 8), (9, 10), (11, 12)]
DIM_PAIR_LABELS = [
    ("dim1", "dim2"), ("dim3", "dim4"), ("dim5", "dim6"),
    ("dim8", "dim9"), ("dim10", "dim11"), ("dim12", "dim13"),
]
INSPECTED_STEP = 20


# ── 模型前向传播 ────────────────────────────────────────────

@nnx.jit
def compute_velocity(
    model,
    obs_batch: _model.Observation,
    noisy_actions,
    t_scalar,
):
    """计算 v_t = f(x_t, t)。

    Args:
        model: Pi0 模型
        obs_batch: Observation，batch_size = N
        noisy_actions: float[N, action_horizon, action_dim]，noisy x_t
        t_scalar: 标量扩散时间步 t

    Returns:
        v_t: float[N, action_horizon, action_dim]
    """
    obs_batch = _model.preprocess_observation(None, obs_batch, train=False)
    batch_size = noisy_actions.shape[0]
    time_arr = jnp.full((batch_size,), t_scalar, dtype=jnp.float32)

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs_batch)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        obs_batch, noisy_actions, time_arr
    )

    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1

    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond],
    )
    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])
    return v_t  # (N, action_horizon, action_dim)


# ── 矢量场计算 ──────────────────────────────────────────────

def compute_vector_field_2d(
    model,
    obs_single: _model.Observation,
    gt_action: np.ndarray,
    action_horizon: int,
    action_dim: int,
    dim_x: int,
    dim_y: int,
    t_val: float,
    xgrid_range: tuple,
    ygrid_range: tuple,
    n_grid: int,
):
    """对给定维度对计算 2D 矢量场。

    在 (dim_x, dim_y) 维度上构造 n_grid×n_grid 网格，其余维度固定为 0，
    批量运行前向传播，返回各点的速度向量。

    Args:
        obs_single: Observation，batch_size=1
        action_horizon, action_dim: 动作形状
        dim_x, dim_y: 要扫描的维度（0-indexed）
        t_val: 扩散时间步
        grid_range: (min, max) 扫描范围
        n_grid: 每维格点数

    Returns:
        X, Y: meshgrid，shape (n_grid, n_grid)
        U, V: 矢量分量，shape (n_grid, n_grid)
    """
    xs = np.linspace(xgrid_range[0], xgrid_range[1], n_grid)
    ys = np.linspace(ygrid_range[0], ygrid_range[1], n_grid)
    X, Y = np.meshgrid(xs, ys)

    n_batch = n_grid * n_grid

    # 构造 batch action：[n_batch, action_horizon, action_dim]，按照正态分布随机初始化
    actions_batch = np.random.randn(n_batch, action_horizon, action_dim).astype(np.float32) # * t_val + (1-t_val) * gt_action[None, :, :]
    actions_batch[:, INSPECTED_STEP - 1, dim_x] = X.ravel()
    actions_batch[:, INSPECTED_STEP - 1, dim_y] = Y.ravel()

    # 将 observation 复制 n_batch 次
    obs_batch = jax.tree.map(
        lambda x: jnp.repeat(x, n_batch, axis=0),
        obs_single,
    )

    actions_jnp = jnp.array(actions_batch, dtype=jnp.bfloat16)
    v_t = compute_velocity(model, obs_batch, actions_jnp, float(t_val))
    v_t = np.array(v_t, dtype=np.float32)  # (n_batch, action_horizon, action_dim)

    # 得到action horizon = INSPECTED_STEP 步的平均速度作为该点的矢量
    v_avg = v_t[:, INSPECTED_STEP - 1]  # (n_batch, action_dim)

    U = v_avg[:, dim_x].reshape(n_grid, n_grid)
    V = v_avg[:, dim_y].reshape(n_grid, n_grid)

    return X, Y, U, V


# ── 工具函数 ──────────────────────────────────────────────

def _parse_samples(samples_str: str) -> list[tuple[int, int]]:
    """解析 'ep:ts ep:ts ...' 格式的字符串，返回 (episode, timestep) 列表。"""
    result = []
    for token in samples_str.strip().split():
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid sample format '{token}', expected 'episode:timestep'")
        result.append((int(parts[0]), int(parts[1])))
    return result


def _sample_to_obs(sample: dict) -> _model.Observation:
    """将 transform 后的 sample dict 转为 batch=1 的 Observation。

    AirbotInputs 输出的 sample 中 "image" 是 dict，不能直接 jnp.array，
    需要用 jax.tree.map 递归处理叶子数组。
    """
    obs_keys = {"image", "image_mask", "state", "tokenized_prompt", "tokenized_prompt_mask"}
    obs_raw = {k: v for k, v in sample.items() if k in obs_keys}
    # jax.tree.map 递归遍历所有叶子（包括嵌套的 image/image_mask dict）
    obs_batched = jax.tree.map(lambda x: jnp.array(np.asarray(x))[None], obs_raw)
    return _model.Observation.from_dict(obs_batched)


# ── 主函数 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize pi0 vector field")
    parser.add_argument("--config-name", required=True, help="训练 config 名称")
    parser.add_argument("--checkpoint-dir", required=True, help="checkpoint step 目录（含 params/）")
    parser.add_argument(
        "--samples", required=True,
        help="要可视化的 episode:timestep 列表，空格分隔，例如 '400:500 401:100 402:0'",
    )
    parser.add_argument("--output-dir", default="assets/visualizations", help="输出根目录")
    parser.add_argument("--n-grid", type=int, default=12, help="每维格点数（n_grid×n_grid 个点）")
    parser.add_argument("--grid-range", type=float, nargs=2, default=[-3.0, 3.0],
                        help="扫描范围 [min max]（normalized space）")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    grid_range = tuple(args.grid_range)
    samples = _parse_samples(args.samples)
    unique_episodes = sorted({ep for ep, _ in samples})

    # ── 1. 加载训练 config ──────────────────────────────────
    print(f"Loading config: {args.config_name}")
    train_config = _config.get_config(args.config_name)

    # ── 2. 加载模型参数 ─────────────────────────────────────
    print(f"Loading model from: {checkpoint_dir}")
    model = train_config.model.load(
        _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    )
    model.eval()

    # ── 3. 加载 norm stats（从 checkpoint 的 assets） ────────
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    print(f"Loading norm stats from: {checkpoint_dir / 'assets'}")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    data_config = dataclasses.replace(data_config, norm_stats=norm_stats)

    # ── 4. 构建推理 transforms（advantage_dropout_rate=0） ───
    inference_model_transforms = []
    for t in data_config.model_transforms.inputs:
        if isinstance(t, _transforms.TokenizePrompt) and t.advantage_conditioning:
            t = dataclasses.replace(t, advantage_dropout_rate=0.0)
        inference_model_transforms.append(t)

    transform_list = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *inference_model_transforms,
    ]

    # ── 5. 只加载需要的 episode，避免全量 split 扫描 ─────────
    repo_id = data_config.repo_id
    print(f"Loading dataset: {repo_id}, episodes={unique_episodes}")

    lerobot_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    raw_lerobot_ds = lerobot_dataset.LeRobotDataset(
        repo_id,
        episodes=unique_episodes,          # 只加载所需 episode
        delta_timestamps={
            key: [t / lerobot_meta.fps for t in range(train_config.model.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )
    # 为可视化兼容缺失标签：若没有 is_good_action/intervention，则注入默认 0 -- by pyc
    dataset_columns = set(raw_lerobot_ds.hf_dataset.column_names)
    missing_cols = [c for c in ("is_good_action", "intervention") if c not in dataset_columns]
    if missing_cols:
        print(f"[WARN] Missing columns {missing_cols}. Injecting defaults for visualization.")

        def _inject_missing_columns(batch):
            n = len(batch["episode_index"])
            out = {}
            if "is_good_action" in missing_cols:
                out["is_good_action"] = [0] * n
            if "intervention" in missing_cols:
                out["intervention"] = [0] * n
            return out

        raw_lerobot_ds.hf_dataset = raw_lerobot_ds.hf_dataset.map(
            _inject_missing_columns,
            batched=True,
            desc=f"Injecting default columns: {missing_cols}",
        )

    # 修复 lerobot 已知 bug：
    # 用 episodes= 过滤加载时，episode_data_index 是 0-indexed (0,1,2...)，
    # 但 hf_dataset["episode_index"] 存的是全局 episode 号 (400,401,...)。
    # __getitem__ 里用全局号做 episode_data_index 的下标会越界。
    # 解决：先用全局号建立帧映射，再把 episode_index 列原地重映射为位置索引。
    ep_indices_global = np.array(raw_lerobot_ds.hf_dataset["episode_index"])
    ep_to_local_frames: dict[int, np.ndarray] = {}
    for ep in unique_episodes:
        idxs = np.where(ep_indices_global == ep)[0]
        if len(idxs) == 0:
            raise ValueError(f"Episode {ep} not found in dataset")
        ep_to_local_frames[ep] = idxs

    # 重映射 episode_index：global → positional（与 episode_data_index 对齐）
    ep_to_pos = {ep: pos for pos, ep in enumerate(unique_episodes)}
    raw_lerobot_ds.hf_dataset = raw_lerobot_ds.hf_dataset.map(
        lambda batch: {"episode_index": [ep_to_pos[int(e)] for e in batch["episode_index"]]},
        batched=True,
        desc="Remapping episode indices",
    )

    # 应用 transforms
    if data_config.prompt_from_task:
        base_ds = _data_loader.TransformedDataset(
            raw_lerobot_ds, [_transforms.PromptFromLeRobotTask(lerobot_meta.tasks)]
        )
    else:
        base_ds = raw_lerobot_ds
    transformed_ds = _data_loader.TransformedDataset(base_ds, transform_list)

    action_horizon = train_config.model.action_horizon
    action_dim = train_config.model.action_dim
    t_values = [round(i * 0.1, 1) for i in range(1, 11)]
    exp_name = checkpoint_dir.parent.name
    step_name = checkpoint_dir.name

    # ── 6. 遍历每个 (episode, timestep) 样本 ────────────────
    for ep, ts in samples:
        local_frames = ep_to_local_frames[ep]
        if ts >= len(local_frames):
            print(f"[WARN] Episode {ep} has {len(local_frames)} frames, skipping ts={ts}")
            continue
        local_idx = int(local_frames[ts])
        print(f"\nProcessing episode={ep}, timestep={ts} (local_idx={local_idx})")

        sample = transformed_ds[local_idx]
        obs_single = _sample_to_obs(sample)
        # ground truth action in normalized space: [action_horizon, action_dim]
        gt_action = np.array(sample["actions"], dtype=np.float32)

        out_dir = (
            Path(args.output_dir)
            / exp_name
            / step_name
            / f"episode{ep:04d}_ts{ts:04d}_adaptive"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output: {out_dir}")

        # ── 7. 遍历 t 值，生成图 ─────────────────────────────
        for t_val in t_values:
            print(f"  t={t_val:.1f} ...", flush=True)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(
                f"Vector Field  f(x_t, t) = v_t     t = {t_val:.1f}   "
                f"[ep={ep}, ts={ts}]",
                fontsize=13,
            )

            for idx, ((dim_x, dim_y), (lx, ly)) in enumerate(zip(DIM_PAIRS, DIM_PAIR_LABELS)):
                row, col = divmod(idx, 3)
                ax = axes[row, col]

                center_x = float(gt_action[INSPECTED_STEP - 1, dim_x])
                center_y = float(gt_action[INSPECTED_STEP - 1, dim_y])
                h = 0.5
                xgrid_range = (center_x - h, center_x + h)
                ygrid_range = (center_y - h, center_y + h)

                X, Y, U, V = compute_vector_field_2d(
                    model, obs_single, gt_action,
                    action_horizon, action_dim,
                    dim_x, dim_y,
                    t_val, xgrid_range, ygrid_range, args.n_grid,
                )

                magnitude = np.sqrt(U ** 2 + V ** 2)
                mag_norm = magnitude + 1e-8
                q = ax.quiver(
                    X, Y,
                    U / mag_norm, V / mag_norm,
                    magnitude,
                    cmap="viridis",
                    scale=args.n_grid * 1.2,
                    width=0.004,
                    headwidth=4,
                )
                fig.colorbar(q, ax=ax, label="|v_t|")
                # ground truth action：各 action_horizon 步的散点 + INSPECTED_STEP 步的值
                ax.scatter(
                    gt_action[:, dim_x], gt_action[:, dim_y],
                    c="red", s=15, alpha=0.5, zorder=5, label="GT steps",
                )
                ax.scatter(
                    [gt_action[INSPECTED_STEP - 1, dim_x]], [gt_action[INSPECTED_STEP - 1, dim_y]],
                    c="red", s=80, marker="*", zorder=6, label="GT inspected step",
                )
                ax.set_xlabel(lx)
                ax.set_ylabel(ly)
                ax.set_title(f"({lx}, {ly})")
                ax.set_xlim(xgrid_range)
                ax.set_ylim(ygrid_range)
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc="upper right")

            fig.tight_layout()
            save_path = out_dir / f"t{t_val:.1f}.png"
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"    saved: {save_path}")

    print(f"\nAll done.")


if __name__ == "__main__":
    main()
