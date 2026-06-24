"""
离线可视化 LeRobot v2 数据集 (Fold_clothes)
用法: python scripts/visualize_dataset.py [--data_dir PATH] [--output_dir PATH] [--episodes 0,1,2]

功能:
  1. 每个 episode 采样关键帧，展示 3 个相机的图像拼图
  2. 每个 episode 的 state / action 14维轨迹曲线
  3. 全局 binned_value 分布直方图
  4. 每个 episode 的 binned_value 随时间变化曲线
  5. state / action 各维度的统计箱线图
  6. VF 综合可视化 (predicted_value / advantage / is_good_action / 采样帧)
  7. 每个 episode 输出 base 相机视频 + 下方 predicted_value 动态折线图
"""

import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

try:
    import pyarrow.parquet as pq
except ImportError:
    raise ImportError("请先安装 pyarrow: pip install pyarrow")


# ── 常量 ──────────────────────────────────────────────────

CAM_NAMES = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
CAM_LABELS = ["Base Camera", "Left Wrist", "Right Wrist"]
GOOD_COLOR, BAD_COLOR = "#2ecc71", "#e74c3c"
# OpenCV 使用 BGR，用于 episode 视频右侧折线图
_GOOD_BGR = (113, 204, 46)
_BAD_BGR = (60, 76, 231)
_TRACE_SINGLE_BGR = (36, 132, 236)  # 无 is_good_action 时的折线色


# ── 工具函数 ──────────────────────────────────────────────

def save_fig(fig: plt.Figure, output_dir: Path, filename: str):
    """tight_layout → savefig → close → print."""
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {filename}")


def shade_good_bad(ax, t: np.ndarray, good_mask: np.ndarray,
                   ymin: float = 0, ymax: float = 1, alpha: float = 0.3):
    """在 ax 上用绿/红着色 good/bad 区域。"""
    ax.fill_between(t, ymin, ymax, where=good_mask, color=GOOD_COLOR, alpha=alpha)
    ax.fill_between(t, ymin, ymax, where=~good_mask, color=BAD_COLOR, alpha=alpha)


def _draw_boxplot(ax, data: np.ndarray, color: str, title: str):
    """绘制 N 维分布箱线图到 ax。"""
    ndim = data.shape[1]
    bp = ax.boxplot([data[:, d] for d in range(ndim)],
                    tick_labels=[str(d) for d in range(ndim)],
                    patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_title(title)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3, axis="y")


def load_episode(parquet_path: Path) -> dict:
    """读取单个 episode 的 parquet 文件，返回 numpy 数据字典。"""
    table = pq.read_table(str(parquet_path))
    data = {}

    for col in ["state", "actions", "timestamp", "frame_index",
                 "episode_index", "index", "task_index", "binned_value"]:
        arr = table.column(col).to_numpy()
        data[col] = np.stack(arr) if col in ("state", "actions") else arr

    for col in ["is_good_action", "predicted_value", "advantage"]:
        if col in table.column_names:
            data[col] = table.column(col).to_numpy()

    for cam in CAM_NAMES:
        data[cam + "_raw"] = table.column(cam)

    data["num_frames"] = len(table)
    return data


def decode_image(arrow_col, idx: int) -> np.ndarray:
    """从 arrow struct 列的第 idx 行解码 PNG → numpy RGB (H,W,3)。"""
    img_bytes = arrow_col[idx].as_py()["bytes"]
    return np.array(Image.open(io.BytesIO(img_bytes)))


def make_frame_mosaic(episode_data: dict, frame_indices: list[int],
                      cam_names: list[str]) -> np.ndarray:
    """生成帧马赛克: 行=帧, 列=相机。"""
    rows = []
    for fi in frame_indices:
        row = np.concatenate(
            [decode_image(episode_data[c + "_raw"], fi) for c in cam_names],
            axis=1,
        )
        rows.append(row)
    return np.concatenate(rows, axis=0)


def estimate_video_fps(episode_data: dict, default_fps: int = 20) -> int:
    """根据 timestamp 估计视频 fps；不可用时回退默认值。"""
    ts = episode_data.get("timestamp")
    if ts is None:
        return default_fps
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size < 2:
        return default_fps
    dts = np.diff(ts)
    dts = dts[np.isfinite(dts) & (dts > 1e-6)]
    if dts.size == 0:
        return default_fps
    fps = 1.0 / np.median(dts)
    if not np.isfinite(fps) or fps <= 0:
        return default_fps
    return int(np.clip(round(float(fps)), 1, 60))


# ── 绘图函数 ──────────────────────────────────────────────

def plot_camera_samples(episode_data: dict, ep_idx: int, output_dir: Path,
                        num_samples: int = 5):
    """为一个 episode 采样帧并保存 3 相机拼图。"""
    T = episode_data["num_frames"]
    indices = np.linspace(0, T - 1, num_samples, dtype=int)
    mosaic = make_frame_mosaic(episode_data, indices.tolist(), CAM_NAMES)

    fig, ax = plt.subplots(figsize=(18, 4 * num_samples))
    ax.imshow(mosaic)
    ax.set_axis_off()

    h_img = mosaic.shape[0] // num_samples
    w_img = mosaic.shape[1] // 3
    text_kw = dict(bbox=dict(facecolor="black", alpha=0.5, pad=2))
    for ci, label in enumerate(CAM_LABELS):
        ax.text(ci * w_img + w_img // 2, 20, label,
                ha="center", va="top", fontsize=12, color="yellow",
                fontweight="bold", **text_kw)
    for ri, fi in enumerate(indices):
        ax.text(5, ri * h_img + 20, f"frame {fi}",
                ha="left", va="top", fontsize=10, color="cyan", **text_kw)

    fig.suptitle(f"Episode {ep_idx} — Camera Samples ({T} frames)",
                 fontsize=14, fontweight="bold")
    save_fig(fig, output_dir, f"ep{ep_idx:02d}_cameras.png")


def plot_trajectory(episode_data: dict, ep_idx: int, output_dir: Path):
    """绘制 state 和 action 14 维轨迹。"""
    T = episode_data["state"].shape[0]
    t = np.arange(T)
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    configs = [
        ("state", "State Value", f"Episode {ep_idx} — State Trajectories (14 dims)"),
        ("actions", "Action Value", f"Episode {ep_idx} — Action Trajectories (14 dims)"),
    ]
    for ax, (key, ylabel, title) in zip(axes, configs):
        data = episode_data[key]
        for d in range(data.shape[1]):
            ax.plot(t, data[:, d], linewidth=0.8, label=f"dim {d}")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=7, ncol=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame Index")

    save_fig(fig, output_dir, f"ep{ep_idx:02d}_trajectories.png")


def plot_binned_value_per_episode(episode_data: dict, ep_idx: int,
                                  output_dir: Path):
    """绘制单个 episode 的 binned_value 随时间变化。"""
    rb = episode_data["binned_value"]
    t = np.arange(len(rb))

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, rb, linewidth=1.0, color="tab:green")
    ax.fill_between(t, rb, alpha=0.2, color="tab:green")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("binned_value")
    ax.set_title(f"Episode {ep_idx} — binned_value over time "
                 f"(range [{rb.min()}, {rb.max()}])")
    ax.grid(True, alpha=0.3)

    save_fig(fig, output_dir, f"ep{ep_idx:02d}_binned_value.png")


def plot_global_returns_histogram(all_returns: np.ndarray, output_dir: Path):
    """所有 episode 合并后的 binned_value 分布直方图。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.arange(all_returns.min() - 0.5, all_returns.max() + 1.5, 1)
    ax.hist(all_returns, bins=bins, color="tab:blue", edgecolor="white",
            linewidth=0.3, alpha=0.8)
    ax.set_xlabel("binned_value")
    ax.set_ylabel("Count")
    ax.set_title(f"Global binned_value Distribution "
                 f"(total {len(all_returns)} frames, "
                 f"range [{all_returns.min()}, {all_returns.max()}])")
    ax.grid(True, alpha=0.3, axis="y")

    save_fig(fig, output_dir, "global_returns_hist.png")


def plot_state_action_boxplot(all_states: np.ndarray, all_actions: np.ndarray,
                              output_dir: Path):
    """State 和 Action 14 维的分布箱线图。"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for ax, (data, color, title) in zip(axes, [
        (all_states, "tab:blue", "State Distribution per Dimension (all episodes)"),
        (all_actions, "tab:orange", "Action Distribution per Dimension (all episodes)"),
    ]):
        _draw_boxplot(ax, data, color, title)

    save_fig(fig, output_dir, "state_action_boxplot.png")


def plot_vf_scores(episode_data: dict, ep_idx: int, output_dir: Path,
                   num_camera_samples: int = 8):
    """综合 VF 可视化: is_good_action 时间线 / predicted_value / advantage / 采样帧 / 分布。

    合并了原 plot_good_bad_actions 和 plot_vf_scores 的功能，按需动态选择子图行数。
    """
    T = episode_data["num_frames"]
    t = np.arange(T)

    has_pv = "predicted_value" in episode_data
    has_adv = "advantage" in episode_data
    has_good = "is_good_action" in episode_data
    if not has_pv and not has_adv:
        return

    good_mask = episode_data["is_good_action"].astype(bool) if has_good else None

    rows = []
    if has_good:
        rows.append("timeline")
    if has_pv:
        rows.append("pred_value")
    if has_adv:
        rows.append("advantage")
    if has_good:
        rows.append("cameras")
    rows.append("histogram")

    height_ratios = [3 if r == "cameras" else 2 for r in rows]
    fig = plt.figure(figsize=(20, sum(height_ratios) * 1.5))
    gs = gridspec.GridSpec(len(rows), 1, height_ratios=height_ratios, hspace=0.35)
    ri = 0

    # ---- is_good_action 时间线 ----
    if has_good:
        ax = fig.add_subplot(gs[ri]); ri += 1
        shade_good_bad(ax, t, good_mask, alpha=0.4)
        ax.set_xlim(0, T - 1); ax.set_ylim(0, 1)
        ax.set_ylabel("is_good_action")
        n_good = int(good_mask.sum())
        ax.set_title(f"Episode {ep_idx} — Good/Bad Action Timeline "
                     f"({n_good}/{T} = {n_good / T:.1%} good)")
        ax.legend(["good", "bad"], loc="upper right")
        ax.grid(True, alpha=0.3)

    # ---- predicted_value ----
    if has_pv:
        pv = episode_data["predicted_value"]
        ax = fig.add_subplot(gs[ri]); ri += 1
        ax.plot(t, pv, color="tab:blue", linewidth=0.8, label="V(o_t)")
        if good_mask is not None:
            shade_good_bad(ax, t, good_mask, pv.min(), pv.max(), alpha=0.08)
        ax.set_ylabel("V(o_t)")
        ax.set_title(f"Episode {ep_idx} — Predicted Value  "
                     f"[{pv.min():.4f}, {pv.max():.4f}]  mean={pv.mean():.4f}")
        ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)

    # ---- advantage ----
    if has_adv:
        adv = episode_data["advantage"]
        ax = fig.add_subplot(gs[ri]); ri += 1
        ax.plot(t, adv, color="tab:orange", linewidth=0.8, label="A(o_t, a_t)")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        if good_mask is not None:
            shade_good_bad(ax, t, good_mask, adv.min(), adv.max(), alpha=0.08)
        ax.set_ylabel("Advantage")
        ax.set_title(f"Episode {ep_idx} — Advantage  "
                     f"[{adv.min():.4f}, {adv.max():.4f}]  mean={adv.mean():.4f}")
        ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)

    # ---- 采样帧相机图 (绿框=好, 红框=坏) ----
    if has_good and "cameras" in rows:
        good_idx = np.where(good_mask)[0]
        bad_idx = np.where(~good_mask)[0]
        n_g = min(num_camera_samples // 2, len(good_idx))
        n_b = min(num_camera_samples - n_g, len(bad_idx))
        if n_b < num_camera_samples - n_g:
            n_g = min(num_camera_samples - n_b, len(good_idx))

        sampled_good = (good_idx[np.linspace(0, len(good_idx) - 1, n_g, dtype=int)]
                        if n_g > 0 else [])
        sampled_bad = (bad_idx[np.linspace(0, len(bad_idx) - 1, n_b, dtype=int)]
                       if n_b > 0 else [])
        sampled = sorted(list(sampled_good) + list(sampled_bad)) or [0]

        gs_inner = gridspec.GridSpecFromSubplotSpec(
            1, len(sampled), subplot_spec=gs[ri], wspace=0.05)
        ri += 1
        is_good = episode_data["is_good_action"]
        for ci, fi in enumerate(sampled):
            ax = fig.add_subplot(gs_inner[0, ci])
            ax.imshow(decode_image(episode_data["base_0_rgb_raw"], fi))
            is_g = bool(is_good[fi])
            color = GOOD_COLOR if is_g else BAD_COLOR
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(4)
            ax.set_title(f"f{fi} [{'GOOD' if is_g else 'BAD'}]",
                         fontsize=9, color=color, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])

    # ---- 直方图 ----
    ax = fig.add_subplot(gs[ri])
    hist_items = []
    if has_pv:
        hist_items.append((episode_data["predicted_value"], "V(o_t)", "tab:blue"))
    if has_adv:
        hist_items.append((episode_data["advantage"], "Advantage", "tab:orange"))
    for vals, lbl, c in hist_items:
        ax.hist(vals, bins=80, alpha=0.5, color=c, label=lbl,
                edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Value"); ax.set_ylabel("Count")
    ax.set_title(f"Episode {ep_idx} — VF Score Distributions")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Episode {ep_idx} — VF Comprehensive Visualization",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.savefig(output_dir / f"ep{ep_idx:02d}_vf_scores.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] ep{ep_idx:02d}_vf_scores.png")


def plot_global_vf_stats(all_pred_values: list[np.ndarray],
                         all_advantages: list[np.ndarray],
                         all_is_good: list[np.ndarray],
                         ep_indices: list[int],
                         output_dir: Path):
    """全局 VF 统计图: 跨 episode 的分布和统计。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    x = np.arange(len(ep_indices))

    all_pv = np.concatenate(all_pred_values)
    all_adv = np.concatenate(all_advantages)

    # 左上: predicted_value 全局直方图
    ax = axes[0, 0]
    ax.hist(all_pv, bins=100, color="tab:blue", alpha=0.7,
            edgecolor="white", linewidth=0.3)
    ax.axvline(all_pv.mean(), color="red", linestyle="--",
               label=f"mean={all_pv.mean():.4f}")
    ax.set_title(f"Global Predicted Value Distribution (n={len(all_pv)})")
    ax.set_xlabel("V(o_t)"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 右上: advantage 全局直方图
    ax = axes[0, 1]
    ax.hist(all_adv, bins=100, color="tab:orange", alpha=0.7,
            edgecolor="white", linewidth=0.3)
    ax.axvline(all_adv.mean(), color="red", linestyle="--",
               label=f"mean={all_adv.mean():.4f}")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_title(f"Global Advantage Distribution (n={len(all_adv)})")
    ax.set_xlabel("Advantage"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 左下: 每个 episode 的 positive_fraction 柱状图
    ax = axes[1, 0]
    fracs = [g.mean() for g in all_is_good]
    colors = [GOOD_COLOR if f > 0.5 else BAD_COLOR for f in fracs]
    ax.bar(x, fracs, color=colors, alpha=0.7, edgecolor="white")
    for i, f in enumerate(fracs):
        ax.text(i, f + 0.02, f"{f:.0%}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"ep{ei}" for ei in ep_indices], fontsize=8)
    ax.set_ylabel("Positive Fraction")
    ax.set_title("is_good_action Positive Fraction per Episode")
    ax.set_ylim(0, 1.15); ax.grid(True, alpha=0.3, axis="y")

    # 右下: 每个 episode 的 mean predicted_value 柱状图
    ax = axes[1, 1]
    means = [pv.mean() for pv in all_pred_values]
    stds = [pv.std() for pv in all_pred_values]
    ax.bar(x, means, yerr=stds, color="tab:blue", alpha=0.7,
           edgecolor="white", capsize=3, error_kw={"linewidth": 1})
    for i, m in enumerate(means):
        ax.text(i, m + stds[i] + 0.01, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"ep{ei}" for ei in ep_indices], fontsize=8)
    ax.set_ylabel("Mean V(o_t)")
    ax.set_title("Mean Predicted Value per Episode (± std)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Global VF Statistics", fontsize=14, fontweight="bold")
    save_fig(fig, output_dir, "global_vf_stats.png")


def plot_episode_lengths(ep_lengths: list[int], output_dir: Path):
    """各 episode 长度柱状图。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(ep_lengths))
    ax.bar(x, ep_lengths, color="tab:purple", alpha=0.7, edgecolor="white")
    for i, v in enumerate(ep_lengths):
        ax.text(i, v + 10, str(v), ha="center", fontsize=9)
    ax.set_xlabel("Episode Index")
    ax.set_ylabel("Number of Frames")
    ax.set_title(f"Episode Lengths (total {sum(ep_lengths)} frames)")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis="y")

    save_fig(fig, output_dir, "episode_lengths.png")


def _draw_value_video_trace(
    plot: np.ndarray,
    pts_all: np.ndarray,
    end_idx_inclusive: int,
    good_mask: np.ndarray | None,
) -> None:
    """在 OpenCV 画布上绘制 predicted_value 折线；若有 good_mask 则按帧着色 positive/negative。"""
    import cv2

    if end_idx_inclusive <= 0:
        return
    n = end_idx_inclusive + 1
    if good_mask is None:
        if n > 1:
            cv2.polylines(
                plot, [pts_all[:n]], False, _TRACE_SINGLE_BGR, 3, cv2.LINE_AA)
        return
    for j in range(end_idx_inclusive):
        col = _GOOD_BGR if bool(good_mask[j]) else _BAD_BGR
        cv2.line(
            plot,
            tuple(pts_all[j]),
            tuple(pts_all[j + 1]),
            col,
            3,
            cv2.LINE_AA,
        )


def make_episode_value_video(episode_data: dict, ep_idx: int, output_dir: Path,
                             default_fps: int = 20, video_scale: float = 1.4):
    """输出一个 mp4：左侧 base 相机，右侧 predicted_value 随时间展开折线图。"""
    if "predicted_value" not in episode_data:
        print(f"  [skip] ep{ep_idx:02d}_base_value_video.mp4 (缺少 predicted_value)")
        return
    try:
        import cv2
    except ImportError as e:
        raise ImportError("生成视频需要 opencv-python，请先安装。") from e

    values = np.asarray(episode_data["predicted_value"], dtype=np.float32)
    T = len(values)
    if T == 0:
        return

    good_mask = None
    if "is_good_action" in episode_data:
        gm = np.asarray(episode_data["is_good_action"]).astype(bool).reshape(-1)
        if gm.shape[0] == T:
            good_mask = gm
        else:
            print(
                f"  [warn] ep{ep_idx:02d} is_good_action 长度 {gm.shape[0]} != T={T}，"
                "视频中折线不区分 positive/negative")

    first_img = decode_image(episode_data["base_0_rgb_raw"], 0)
    h_img, w_img = first_img.shape[:2]
    h_plot = h_img
    w_plot = max(520, int(w_img * 2.2))

    # 给坐标轴刻度文字预留边距。
    pad_l, pad_r, pad_t, pad_b = 58, 22, 20, 40
    x0, x1 = pad_l, max(pad_l + 10, w_plot - pad_r)
    y0, y1 = pad_t, max(pad_t + 10, h_plot - pad_b)
    if T == 1:
        xs = np.array([(x0 + x1) // 2], dtype=np.float32)
    else:
        xs = np.linspace(x0, x1, T, dtype=np.float32)
    vmin, vmax = float(values.min()), float(values.max())
    if np.isclose(vmin, vmax):
        eps = 1e-4 if vmin == 0 else abs(vmin) * 0.05
        vmin -= eps
        vmax += eps
    ys = y1 - (values - vmin) / (vmax - vmin) * (y1 - y0)
    pts_all = np.stack([xs, ys], axis=1).astype(np.int32)

    plot_bg = np.full((h_plot, w_plot, 3), 248, dtype=np.uint8)
    cv2.rectangle(plot_bg, (x0, y0), (x1, y1), (235, 235, 235), -1)

    # y 轴网格和刻度（只显示数值，不额外文字说明）。
    y_ticks_val = np.linspace(vmin, vmax, 4)
    for yv in y_ticks_val:
        yy = int(y1 - (yv - vmin) / (vmax - vmin) * (y1 - y0))
        cv2.line(plot_bg, (x0, yy), (x1, yy), (222, 222, 222), 1, cv2.LINE_AA)
        cv2.putText(plot_bg, f"{yv:.2f}", (6, yy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (75, 75, 75), 1, cv2.LINE_AA)

    # x 轴网格和刻度。
    x_tick_idx = np.unique(np.linspace(0, T - 1, min(8, T), dtype=int))
    last_x = -10_000
    for idx in x_tick_idx:
        xx = int(xs[idx])
        if xx - last_x < 70:
            continue
        cv2.line(plot_bg, (xx, y0), (xx, y1), (228, 228, 228), 1, cv2.LINE_AA)
        label = str(idx)
        label_x = xx - 7 * len(label) // 2
        cv2.putText(plot_bg, label, (label_x, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (75, 75, 75), 1, cv2.LINE_AA)
        last_x = xx

    cv2.line(plot_bg, (x0, y0), (x0, y1), (70, 70, 70), 1, cv2.LINE_AA)
    cv2.line(plot_bg, (x0, y1), (x1, y1), (70, 70, 70), 1, cv2.LINE_AA)
    cv2.polylines(plot_bg, [pts_all], False, (200, 200, 200), 1, cv2.LINE_AA)

    if good_mask is not None:
        leg_x = min(x1 - 5, x0 + w_plot - 200)
        leg_y = y0 + 16
        cv2.circle(plot_bg, (leg_x, leg_y), 5, _GOOD_BGR, -1, cv2.LINE_AA)
        cv2.putText(
            plot_bg, "Positive", (leg_x + 12, leg_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (55, 55, 55), 1, cv2.LINE_AA)
        cv2.circle(plot_bg, (leg_x, leg_y + 22), 5, _BAD_BGR, -1, cv2.LINE_AA)
        cv2.putText(
            plot_bg, "Negative", (leg_x + 12, leg_y + 27),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (55, 55, 55), 1, cv2.LINE_AA)

    fps = estimate_video_fps(episode_data, default_fps=default_fps)
    out_path = output_dir / f"ep{ep_idx:02d}_base_value_video.mp4"
    frame_w = w_img + w_plot
    frame_h = h_img
    video_scale = max(1.0, float(video_scale))
    enc_w = int(round(frame_w * video_scale))
    enc_h = int(round(frame_h * video_scale))
    if enc_w % 2 != 0:
        enc_w += 1
    if enc_h % 2 != 0:
        enc_h += 1
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (enc_w, enc_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {out_path}")

    for i in range(T):
        img = decode_image(episode_data["base_0_rgb_raw"], i)
        if img.shape[:2] != (h_img, w_img):
            img = cv2.resize(img, (w_img, h_img), interpolation=cv2.INTER_AREA)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        plot = plot_bg.copy()
        _draw_value_video_trace(plot, pts_all, i, good_mask)
        cur_pt = tuple(pts_all[i])
        cv2.line(plot, (cur_pt[0], y0), (cur_pt[0], y1), (145, 145, 145), 1, cv2.LINE_AA)
        head_color = (
            (_GOOD_BGR if bool(good_mask[i]) else _BAD_BGR)
            if good_mask is not None
            else (220, 80, 35)
        )
        cv2.circle(plot, cur_pt, 5, head_color, -1, cv2.LINE_AA)
        if good_mask is not None:
            tag = "P" if bool(good_mask[i]) else "N"
            tx, ty = int(cur_pt[0]) - 4, int(cur_pt[1]) - 10
            cv2.putText(
                plot, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(
                plot, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                head_color, 1, cv2.LINE_AA)

        frame_rgb = np.hstack([img, plot])
        if (enc_w, enc_h) != (frame_w, frame_h):
            frame_rgb = cv2.resize(frame_rgb, (enc_w, enc_h), interpolation=cv2.INTER_CUBIC)
        writer.write(frame_rgb[:, :, ::-1])  # RGB -> BGR

    writer.release()
    print(f"  [✓] {out_path.name}")


# ── 主流程 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="离线可视化 LeRobot v2 Fold_clothes 数据集")
    parser.add_argument("--data_dir", type=str,
                        default="/data/kding/Openpi_RL/lerobot_data/Fold_clothes_v3",
                        help="数据集根目录")
    parser.add_argument("--output_dir", type=str,
                        default="/data/kding/Openpi_RL/assets/visualizations",
                        help="输出图片保存目录")
    parser.add_argument("--episodes", type=str, default=None,
                        help="逗号分隔的 episode 编号, 如 '0,1,2'; 默认全部")
    parser.add_argument("--chunk", type=int, default=8,
                        help="要可视化的数据 chunk 编号，例如 6 表示 chunk-006")
    parser.add_argument("--num_camera_samples", type=int, default=5,
                        help="每个 episode 采样多少帧画相机拼图")
    parser.add_argument("--skip_cameras", action="store_true",
                        help="跳过相机图像可视化 (节省时间)")
    parser.add_argument("--skip_episode_video", action="store_true",
                        help="跳过每个 episode 的 base+value 视频输出")
    parser.add_argument("--video_fps", type=int, default=20,
                        help="视频默认 fps（有 timestamp 时会自动估计）")
    parser.add_argument("--video_scale", type=float, default=1.4,
                        help="视频输出放大倍数（默认 1.4）")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_dir = data_dir / "data" / f"chunk-{args.chunk:03d}"
    if not parquet_dir.exists():
        raise FileNotFoundError(f"找不到 chunk 目录: {parquet_dir}")
    parquet_files = sorted(parquet_dir.glob("episode_*.parquet"))
    print(f"发现 {len(parquet_files)} 个 episode 文件:")
    for pf in parquet_files:
        print(f"  {pf.name}")

    selected = ([int(e) for e in (p.strip() for p in args.episodes.split(",")) if e]
                if args.episodes else list(range(len(parquet_files))))
    print(f"\n将可视化 episode: {selected}\n")

    all_states, all_actions, all_returns = [], [], []
    all_pred_values, all_advantages, all_is_good = [], [], []
    ep_lengths: list[int] = []

    for ep_idx in selected:
        pf = parquet_files[ep_idx]
        print(f"─── Episode {ep_idx} ({pf.name}) ───")
        ep_data = load_episode(pf)
        T = ep_data["num_frames"]
        print(f"  帧数: {T}")

        ep_lengths.append(T)
        all_states.append(ep_data["state"])
        all_actions.append(ep_data["actions"])
        all_returns.append(ep_data["binned_value"])

        if not args.skip_cameras:
            plot_camera_samples(ep_data, ep_idx, output_dir,
                                num_samples=args.num_camera_samples)
        plot_trajectory(ep_data, ep_idx, output_dir)
        plot_binned_value_per_episode(ep_data, ep_idx, output_dir)

        if "predicted_value" in ep_data or "advantage" in ep_data:
            plot_vf_scores(ep_data, ep_idx, output_dir)
        if not args.skip_episode_video:
            make_episode_value_video(
                ep_data, ep_idx, output_dir,
                default_fps=args.video_fps,
                video_scale=args.video_scale)
        if "predicted_value" in ep_data:
            all_pred_values.append(ep_data["predicted_value"])
        if "advantage" in ep_data:
            all_advantages.append(ep_data["advantage"])
        if "is_good_action" in ep_data:
            all_is_good.append(ep_data["is_good_action"].astype(float))

    all_states_arr = np.concatenate(all_states, axis=0)
    all_actions_arr = np.concatenate(all_actions, axis=0)
    all_returns_arr = np.concatenate(all_returns, axis=0)

    print(f"\n─── 全局统计 ───")
    print(f"  总帧数: {len(all_returns_arr)}")
    print(f"  State shape: {all_states_arr.shape}")
    print(f"  Action shape: {all_actions_arr.shape}")
    print(f"  binned_value range: [{all_returns_arr.min()}, {all_returns_arr.max()}]")

    plot_global_returns_histogram(all_returns_arr, output_dir)
    plot_state_action_boxplot(all_states_arr, all_actions_arr, output_dir)
    plot_episode_lengths(ep_lengths, output_dir)

    if all_pred_values and all_advantages and all_is_good:
        plot_global_vf_stats(
            all_pred_values, all_advantages, all_is_good,
            selected, output_dir)

    print(f"\n所有可视化结果已保存到: {output_dir}")
    print("完成!")


if __name__ == "__main__":
    main()
