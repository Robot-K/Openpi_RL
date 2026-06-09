"""Analyze inter-frame timing jitter and camera-action time alignment in MCAP files.

Produces two figures:
  1. Jitter figure  — interval between consecutive frames per channel
  2. Drift figure   — per-frame time offset between each camera and the action topics

Camera-action drift definition
--------------------------------
For each frame i, both video and action have a timestamp relative to the start
of the episode:
    video_elapsed[i]  = pts[i] - pts[0]                  (ms, from mp4 PTS)
    action_elapsed[i] = (action_log_time[i] - action_log_time[0]) / 1e6  (ms)
    drift[i] = video_elapsed[i] - action_elapsed[i]       (ms)

drift > 0 → video frame arrived later than the action message
drift < 0 → video frame arrived earlier than the action message

Single MCAP:
    uv run scripts/tools/analyze_mcap_timing.py --mcap path/to/ep.mcap
    uv run scripts/tools/analyze_mcap_timing.py --mcap path/to/ep.mcap --save out.png

Dataset (aggregate across all episodes):
    uv run scripts/tools/analyze_mcap_timing.py --data-dir mcap_data/fold_clothv2
    uv run scripts/tools/analyze_mcap_timing.py --data-dir mcap_data/fold_clothv2 --limit 20 --save out.png
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    task_name: str
    folders: list[str]
    state_topics: list[str]
    action_topics: list[str]
    camera_topics: dict[str, str]   # cam_name -> ros topic string
    fps: int


def load_task_config(config_path: Path) -> TaskConfig:
    if not config_path.is_file():
        config_path = config_path / "config.py"
    spec = importlib.util.spec_from_file_location("_mcap_cfg", config_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_mcap_cfg"] = mod
    spec.loader.exec_module(mod)
    return TaskConfig(
        task_name=mod.TASK_NAME,
        folders=mod.FOLDERS,
        state_topics=mod.STATE_TOPICS,
        action_topics=mod.ACTION_TOPICS,
        camera_topics=mod.CAMERA_TOPICS,
        fps=mod.FPS,
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChannelIntervals:
    """Inter-frame intervals (ms) for one logical channel."""
    label: str
    intervals_ms: np.ndarray      # shape (N-1,)
    expected_period_ms: float


@dataclass
class CameraActionDrift:
    """Per-frame time offset between one camera and the action topics."""
    cam_label: str
    drift_ms: np.ndarray          # shape (N,): video_elapsed[i] - action_elapsed[i]
    n_frames: int


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _read_video_pts(attach_data: bytes) -> np.ndarray | None:
    """Return per-frame presentation timestamps (ms) from an mp4 attachment."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(attach_data)
        tmp_path = tmp.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        pts: list[float] = []
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            pts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        cap.release()
        return np.array(pts, dtype=np.float64) if len(pts) >= 2 else None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def extract_all(
    mcap_path: Path,
    config: TaskConfig,
) -> tuple[list[ChannelIntervals], list[CameraActionDrift]] | str:
    """Extract interval and drift data from one MCAP file.

    Returns (channels, drifts) on success, or an error string on failure.
    """
    expected_ms = 1000.0 / config.fps
    all_msg_topics = config.state_topics + config.action_topics
    ts_map: dict[str, list[int]] = {t: [] for t in all_msg_topics}
    video_pts_map: dict[str, np.ndarray] = {}   # cam_name -> pts array (ms)

    try:
        with mcap_path.open("rb") as f:
            reader = make_reader(f)
            for attach in reader.iter_attachments():
                if attach.media_type != "video/mp4":
                    continue
                for cam_name, cam_topic in config.camera_topics.items():
                    if attach.name == cam_topic:
                        pts = _read_video_pts(attach.data)
                        if pts is not None:
                            video_pts_map[cam_name] = pts

            for _schema, channel, message in reader.iter_messages(topics=all_msg_topics):
                if channel.topic in ts_map:
                    ts_map[channel.topic].append(message.log_time)
    except Exception as exc:
        return str(exc)

    # ------ build action reference timestamp array (mean of all action topics) ------
    action_arrays = [
        np.array(ts_map[t], dtype=np.int64)
        for t in config.action_topics if len(ts_map[t]) >= 2
    ]
    action_ref: np.ndarray | None = None
    if action_arrays:
        min_len = min(len(a) for a in action_arrays)
        action_ref = np.mean([a[:min_len] for a in action_arrays], axis=0)

    # ------ ChannelIntervals ------
    channels: list[ChannelIntervals] = []

    state_arrays = [np.array(ts_map[t], dtype=np.int64)
                    for t in config.state_topics if len(ts_map[t]) >= 2]
    if state_arrays:
        min_len = min(len(a) for a in state_arrays)
        mean_ts = np.mean([a[:min_len] for a in state_arrays], axis=0)
        channels.append(ChannelIntervals("state topics",
                                         np.diff(mean_ts) / 1e6, expected_ms))

    if action_ref is not None:
        channels.append(ChannelIntervals("action topics",
                                         np.diff(action_ref) / 1e6, expected_ms))

    for cam_name, pts in video_pts_map.items():
        channels.append(ChannelIntervals(f"video: {cam_name}",
                                          np.diff(pts), expected_ms))

    # Deduplicate channels with identical intervals (e.g. state ≡ action)
    i = len(channels) - 1
    while i > 0:
        for j in range(i):
            a, b = channels[i].intervals_ms, channels[j].intervals_ms
            if len(a) == len(b) and np.allclose(a, b, atol=0.01):
                channels[j] = ChannelIntervals(
                    f"{channels[j].label} + {channels[i].label}",
                    channels[j].intervals_ms, channels[j].expected_period_ms)
                channels.pop(i)
                break
        i -= 1

    # ------ CameraActionDrift ------
    drifts: list[CameraActionDrift] = []
    if action_ref is not None:
        for cam_name, pts in video_pts_map.items():
            n = min(len(pts), len(action_ref))
            video_elapsed  = pts[:n] - pts[0]                         # ms
            action_elapsed = (action_ref[:n] - action_ref[0]) / 1e6  # ms
            drift = video_elapsed - action_elapsed
            drifts.append(CameraActionDrift(cam_name, drift, n))

    return channels, drifts


# ---------------------------------------------------------------------------
# Shared plotting helpers
# ---------------------------------------------------------------------------

def _stats_box(ax, text: str, loc: str = "upper right") -> None:
    x = 0.97 if "right" in loc else 0.03
    ha = "right" if "right" in loc else "left"
    y = 0.97 if "upper" in loc else 0.03
    va = "top" if "upper" in loc else "bottom"
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8,
            va=va, ha=ha, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))


# ---------------------------------------------------------------------------
# Figure 1: Jitter (interval between consecutive frames)
# ---------------------------------------------------------------------------

def _plot_interval_row(fig, gs, row: int, ch: ChannelIntervals, color: str) -> None:
    iv = ch.intervals_ms
    ep = ch.expected_period_ms
    mean_iv, std_iv, max_iv = float(np.mean(iv)), float(np.std(iv)), float(np.max(iv))
    dropout_thresh = 2.0 * ep
    drop_idx = np.where(iv > dropout_thresh)[0]

    ax_ts = fig.add_subplot(gs[row, 0])
    x = np.arange(len(iv))
    ax_ts.plot(x, iv, lw=0.7, color=color, alpha=0.8)
    ax_ts.axhline(ep,      color="black", lw=1.0, ls="--", label=f"expected {ep:.1f} ms")
    ax_ts.axhline(mean_iv, color="gray",  lw=0.8, ls=":",  label=f"mean {mean_iv:.1f} ms")
    ax_ts.fill_between(x, mean_iv - std_iv, mean_iv + std_iv, alpha=0.15, color=color,
                       label=f"±1σ ({std_iv:.2f} ms)")
    if len(drop_idx):
        ax_ts.scatter(drop_idx, iv[drop_idx], color="red", s=18, zorder=5,
                      label=f"dropout ×{len(drop_idx)}")
    ax_ts.set_ylim(0, max(3 * ep, max_iv * 1.05))
    ax_ts.set_xlabel("Frame index", fontsize=9)
    ax_ts.set_ylabel("Interval (ms)", fontsize=9)
    ax_ts.set_title(ch.label, fontsize=10, fontweight="bold")
    ax_ts.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax_ts.tick_params(labelsize=8)

    ax_h = fig.add_subplot(gs[row, 1])
    bins = np.linspace(0, max(3 * ep, max_iv * 1.05), 60)
    ax_h.hist(iv, bins=bins, color=color, alpha=0.75, edgecolor="none")
    ax_h.axvline(ep,      color="black", lw=1.0, ls="--")
    ax_h.axvline(mean_iv, color="gray",  lw=0.8, ls=":")
    if len(drop_idx):
        ax_h.axvline(dropout_thresh, color="red", lw=0.9, ls="--",
                     label=f"2×period ({dropout_thresh:.0f} ms)")
        ax_h.legend(fontsize=7.5)
    ax_h.set_xlabel("Interval (ms)", fontsize=9)
    ax_h.set_ylabel("Count", fontsize=9)
    ax_h.tick_params(labelsize=8)
    _stats_box(ax_h, f"mean={mean_iv:.2f} ms\nσ={std_iv:.2f} ms\n"
                     f"max={max_iv:.1f} ms\ndropouts={len(drop_idx)}")


def plot_jitter_single(channels: list[ChannelIntervals], title: str,
                       save_path: Path | None) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n = len(channels)
    if n == 0:
        print("No interval data to plot.")
        return
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = plt.figure(figsize=(14, 3.8 * n), constrained_layout=True)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    gs = gridspec.GridSpec(n, 2, figure=fig, width_ratios=[3, 1.5])
    for row, ch in enumerate(channels):
        _plot_interval_row(fig, gs, row, ch, colors[row % len(colors)])
    _save_or_show(fig, save_path, "_jitter")


def plot_jitter_aggregate(all_channels: dict[str, list[np.ndarray]],
                          expected_ms: float, n_files: int, title: str,
                          save_path: Path | None) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    labels = list(all_channels.keys())
    n = len(labels)
    if n == 0:
        return
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    dropout_thresh = 2.0 * expected_ms

    fig = plt.figure(figsize=(15, 4.5 * n), constrained_layout=True)
    fig.suptitle(f"{title}  ({n_files} episodes)  — interval jitter",
                 fontsize=11, fontweight="bold")
    gs = gridspec.GridSpec(n, 2, figure=fig, width_ratios=[1.8, 2.5])

    for row, label in enumerate(labels):
        arrays = all_channels[label]
        color  = colors[row % len(colors)]
        ep_stds  = np.array([np.std(a)  for a in arrays])
        ep_drops = np.array([int(np.sum(a > dropout_thresh)) for a in arrays])

        ax_v = fig.add_subplot(gs[row, 0])
        vp = ax_v.violinplot([ep_stds], positions=[0], showmedians=True, widths=0.6)
        for body in vp["bodies"]:
            body.set_facecolor(color); body.set_alpha(0.6)
        for part in ("cmedians", "cmins", "cmaxes", "cbars"):
            vp[part].set_color("black" if part == "cmedians" else "gray")
        ax_v.scatter(np.random.uniform(-0.15, 0.15, len(ep_stds)), ep_stds,
                     s=12, alpha=0.5, color=color, zorder=5)
        ax_v.set_xticks([0]); ax_v.set_xticklabels([label], fontsize=9)
        ax_v.set_ylabel("Jitter σ (ms) per episode", fontsize=9)
        ax_v.set_title(label, fontsize=10, fontweight="bold")
        ax_v.tick_params(labelsize=8)
        _stats_box(ax_v, f"median σ={np.median(ep_stds):.2f} ms\n"
                         f"mean σ={np.mean(ep_stds):.2f} ms\n"
                         f"max σ={np.max(ep_stds):.2f} ms\n"
                         f"total dropouts={int(ep_drops.sum())}")

        ax_h = fig.add_subplot(gs[row, 1])
        pooled = np.concatenate(arrays)
        bins = np.linspace(0, max(3 * expected_ms, float(np.max(pooled)) * 1.05), 80)
        ax_h.hist(pooled, bins=bins, color=color, alpha=0.75, edgecolor="none")
        ax_h.axvline(expected_ms,         color="black", lw=1.2, ls="--",
                     label=f"expected {expected_ms:.1f} ms")
        ax_h.axvline(float(np.mean(pooled)), color="gray",  lw=1.0, ls=":",
                     label=f"mean {np.mean(pooled):.2f} ms")
        if np.any(pooled > dropout_thresh):
            ax_h.axvline(dropout_thresh, color="red", lw=0.9, ls="--",
                         label=f"2×period ({dropout_thresh:.0f} ms)")
        ax_h.set_xlabel("Interval (ms)", fontsize=9); ax_h.set_ylabel("Count", fontsize=9)
        ax_h.legend(fontsize=8, loc="upper right"); ax_h.tick_params(labelsize=8)
        _stats_box(ax_h, f"pooled mean={np.mean(pooled):.2f} ms\n"
                         f"pooled σ={np.std(pooled):.2f} ms\n"
                         f"max={np.max(pooled):.1f} ms\n"
                         f"N frames={len(pooled):,}")

    _save_or_show(fig, save_path, "_jitter")


# ---------------------------------------------------------------------------
# Figure 2: Camera-action drift
# ---------------------------------------------------------------------------

def _plot_drift_row(fig, gs, row: int, d: CameraActionDrift, color: str) -> None:
    drift = d.drift_ms
    mean_d, std_d = float(np.mean(drift)), float(np.std(drift))
    max_abs = float(np.max(np.abs(drift)))

    ax_ts = fig.add_subplot(gs[row, 0])
    x = np.arange(len(drift))
    ax_ts.plot(x, drift, lw=0.7, color=color, alpha=0.85)
    ax_ts.axhline(0,      color="black", lw=1.2, ls="--", label="zero drift")
    ax_ts.axhline(mean_d, color="gray",  lw=0.8, ls=":",  label=f"mean {mean_d:+.2f} ms")
    ax_ts.fill_between(x, mean_d - std_d, mean_d + std_d, alpha=0.15, color=color,
                       label=f"±1σ ({std_d:.2f} ms)")
    ax_ts.set_xlabel("Frame index", fontsize=9)
    ax_ts.set_ylabel("Drift (ms)\n[video_elapsed − action_elapsed]", fontsize=8)
    ax_ts.set_title(f"Camera-action drift: {d.cam_label}", fontsize=10, fontweight="bold")
    ax_ts.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax_ts.tick_params(labelsize=8)
    # symmetric y-axis centred on mean
    pad = max(max_abs * 1.1, std_d * 3, 5.0)
    ax_ts.set_ylim(mean_d - pad, mean_d + pad)

    ax_h = fig.add_subplot(gs[row, 1])
    bins = np.linspace(mean_d - pad, mean_d + pad, 60)
    ax_h.hist(drift, bins=bins, color=color, alpha=0.75, edgecolor="none")
    ax_h.axvline(0,      color="black", lw=1.2, ls="--", label="zero drift")
    ax_h.axvline(mean_d, color="gray",  lw=0.8, ls=":")
    ax_h.set_xlabel("Drift (ms)", fontsize=9)
    ax_h.set_ylabel("Count", fontsize=9)
    ax_h.legend(fontsize=7.5)
    ax_h.tick_params(labelsize=8)
    _stats_box(ax_h, f"mean={mean_d:+.2f} ms\nσ={std_d:.2f} ms\n"
                     f"max|drift|={max_abs:.1f} ms\n"
                     f"N frames={len(drift)}")


def plot_drift_single(drifts: list[CameraActionDrift], title: str,
                      save_path: Path | None) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n = len(drifts)
    if n == 0:
        print("No camera-action drift data (need both video and action topics).")
        return
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = plt.figure(figsize=(14, 3.8 * n), constrained_layout=True)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    gs = gridspec.GridSpec(n, 2, figure=fig, width_ratios=[3, 1.5])
    for row, d in enumerate(drifts):
        _plot_drift_row(fig, gs, row, d, colors[row % len(colors)])
    _save_or_show(fig, save_path, "_drift")


def plot_drift_aggregate(all_drifts: dict[str, list[np.ndarray]],
                         n_files: int, title: str,
                         save_path: Path | None) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    labels = list(all_drifts.keys())
    n = len(labels)
    if n == 0:
        return
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig = plt.figure(figsize=(15, 4.5 * n), constrained_layout=True)
    fig.suptitle(f"{title}  ({n_files} episodes)  — camera-action drift",
                 fontsize=11, fontweight="bold")
    gs = gridspec.GridSpec(n, 2, figure=fig, width_ratios=[1.8, 2.5])

    for row, label in enumerate(labels):
        arrays = all_drifts[label]   # list of 1-D drift arrays (one per episode)
        color  = colors[row % len(colors)]
        ep_stds  = np.array([np.std(a)  for a in arrays])
        ep_means = np.array([np.mean(a) for a in arrays])
        ep_maxabs= np.array([np.max(np.abs(a)) for a in arrays])

        # Left: violin of per-episode drift std
        ax_v = fig.add_subplot(gs[row, 0])
        vp = ax_v.violinplot([ep_stds], positions=[0], showmedians=True, widths=0.6)
        for body in vp["bodies"]:
            body.set_facecolor(color); body.set_alpha(0.6)
        for part in ("cmedians", "cmins", "cmaxes", "cbars"):
            vp[part].set_color("black" if part == "cmedians" else "gray")
        ax_v.scatter(np.random.uniform(-0.15, 0.15, len(ep_stds)), ep_stds,
                     s=12, alpha=0.5, color=color, zorder=5)
        ax_v.set_xticks([0]); ax_v.set_xticklabels([label], fontsize=9)
        ax_v.set_ylabel("Drift σ (ms) per episode", fontsize=9)
        ax_v.set_title(f"Camera-action drift: {label}", fontsize=10, fontweight="bold")
        ax_v.tick_params(labelsize=8)
        _stats_box(ax_v, f"median σ={np.median(ep_stds):.2f} ms\n"
                         f"mean σ={np.mean(ep_stds):.2f} ms\n"
                         f"median mean={np.median(ep_means):+.2f} ms\n"
                         f"worst |drift|={np.max(ep_maxabs):.1f} ms")

        # Right: pooled drift histogram
        ax_h = fig.add_subplot(gs[row, 1])
        pooled = np.concatenate(arrays)
        center = float(np.mean(pooled))
        half   = max(float(np.std(pooled)) * 4, float(np.max(np.abs(pooled))) * 1.05, 5.0)
        bins   = np.linspace(center - half, center + half, 80)
        ax_h.hist(pooled, bins=bins, color=color, alpha=0.75, edgecolor="none")
        ax_h.axvline(0,      color="black", lw=1.2, ls="--", label="zero drift")
        ax_h.axvline(center, color="gray",  lw=1.0, ls=":", label=f"mean {center:+.2f} ms")
        ax_h.set_xlabel("Drift (ms)", fontsize=9)
        ax_h.set_ylabel("Frame count (all episodes)", fontsize=9)
        ax_h.legend(fontsize=8, loc="upper right")
        ax_h.tick_params(labelsize=8)
        _stats_box(ax_h, f"pooled mean={center:+.2f} ms\n"
                         f"pooled σ={np.std(pooled):.2f} ms\n"
                         f"max|drift|={np.max(np.abs(pooled)):.1f} ms\n"
                         f"N frames={len(pooled):,}")

    _save_or_show(fig, save_path, "_drift")


# ---------------------------------------------------------------------------
# Save/show helper
# ---------------------------------------------------------------------------

def _save_or_show(fig, save_path: Path | None, suffix: str) -> None:
    import matplotlib.pyplot as plt
    if save_path:
        # Insert suffix before extension: out.png → out_jitter.png / out_drift.png
        p = save_path.with_stem(save_path.stem + suffix)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved → {p}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _infer_config(mcap_path: Path) -> TaskConfig | None:
    for p in [mcap_path.parent, mcap_path.parent.parent, mcap_path.parent.parent.parent]:
        if (p / "config.py").is_file():
            return load_task_config(p / "config.py")
    return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize inter-frame jitter and camera-action drift in MCAP files"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mcap",     type=str, help="Path to a single .mcap file")
    group.add_argument("--data-dir", type=str, help="Dataset directory containing config.py")
    parser.add_argument("--limit",   type=int, default=0,
                        help="Max files to process in --data-dir mode (0 = all)")
    parser.add_argument("--save",    type=str, default="",
                        help="Save plots to files (e.g. out.png → out_jitter.png, out_drift.png)")
    args = parser.parse_args()

    save_path = Path(args.save) if args.save else None

    # ------------------------------------------------------------------ single
    if args.mcap:
        mcap_path = Path(args.mcap)
        config = _infer_config(mcap_path)
        if config is None:
            config = TaskConfig(task_name=mcap_path.stem, folders=[],
                                state_topics=[], action_topics=[],
                                camera_topics={}, fps=20)
            print("No config.py found; will only analyze video timestamps.")
        else:
            print(f"Config: {config.task_name}  ({config.fps} fps)")

        result = extract_all(mcap_path, config)
        if isinstance(result, str):
            print(f"Error: {result}"); return
        channels, drifts = result

        title_base = f"{mcap_path.parent.name}/{mcap_path.name}"
        print(f"Interval channels: {[c.label for c in channels]}")
        print(f"Drift pairs:       {[d.cam_label for d in drifts]}")

        plot_jitter_single(channels, f"{title_base}  —  interval jitter", save_path)
        plot_drift_single(drifts,   f"{title_base}  —  camera-action drift", save_path)

    # --------------------------------------------------------------- directory
    else:
        data_dir = Path(args.data_dir)
        config = load_task_config(data_dir / "config.py")
        print(f"Config: {config.task_name}  ({config.fps} fps)")

        mcap_files: list[Path] = []
        for folder in config.folders:
            fp = data_dir / folder
            if not fp.is_dir():
                continue
            for fname in sorted(os.listdir(fp)):
                if fname.lower().endswith(".mcap"):
                    mcap_files.append(fp / fname)
        mcap_files.sort()
        if args.limit > 0:
            mcap_files = mcap_files[:args.limit]
        print(f"Files: {len(mcap_files)}")

        all_channels: dict[str, list[np.ndarray]] = {}
        all_drifts:   dict[str, list[np.ndarray]] = {}
        errors = 0

        for i, mcap_path in enumerate(mcap_files):
            print(f"  [{i+1}/{len(mcap_files)}] {mcap_path.name}", end="", flush=True)
            result = extract_all(mcap_path, config)
            if isinstance(result, str):
                print(f"  ERROR: {result}"); errors += 1; continue
            channels, drifts = result
            for ch in channels:
                all_channels.setdefault(ch.label, []).append(ch.intervals_ms)
            for d in drifts:
                all_drifts.setdefault(d.cam_label, []).append(d.drift_ms)
            print(f"  intervals={[c.label for c in channels]}"
                  f"  drift={[d.cam_label for d in drifts]}")

        if errors:
            print(f"\n{errors} files skipped.")

        n_ok = len(mcap_files) - errors
        title = config.task_name
        plot_jitter_aggregate(all_channels, 1000.0 / config.fps, n_ok, title, save_path)
        plot_drift_aggregate(all_drifts,  n_ok, title, save_path)


if __name__ == "__main__":
    main()
