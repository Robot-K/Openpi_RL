"""Quick health check for MCAP episodes in a data directory.

Scans every ``.mcap`` file and reports anomalies:
  - File size too large or too small (configurable thresholds)
  - Video FPS too low
  - Mismatch between video frame count and message frame count

Usage:
    python scripts/tools/quick_check_mcap.py --data-dir mcap_data/fold_clothv2
    python scripts/tools/quick_check_mcap.py --data-dir mcap_data/fold_clothv2 --min-size 1 --max-size 200
    python scripts/tools/quick_check_mcap.py --data-dir mcap_data/fold_clothv2 --fps-threshold 18
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from mcap.reader import make_reader


# ---------------------------------------------------------------------------
# Config loader (same as visualize_mcap_images.py)
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    task_name: str
    robot_type: str
    folders: list[str]
    state_topics: list[str]
    action_topics: list[str]
    camera_topics: dict[str, str]
    fps: int


def load_task_config(config_path: str | Path) -> TaskConfig:
    config_path = Path(config_path)
    if not config_path.is_file():
        config_path = config_path / "config.py"
    spec = importlib.util.spec_from_file_location("_mcap_cfg", config_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_mcap_cfg"] = mod
    spec.loader.exec_module(mod)
    return TaskConfig(
        task_name=mod.TASK_NAME,
        robot_type=mod.ROBOT_TYPE,
        folders=mod.FOLDERS,
        state_topics=mod.STATE_TOPICS,
        action_topics=mod.ACTION_TOPICS,
        camera_topics=mod.CAMERA_TOPICS,
        fps=mod.FPS,
    )


# ---------------------------------------------------------------------------
# Per-file check result
# ---------------------------------------------------------------------------

@dataclass
class McapCheckResult:
    path: str
    size_mb: float
    video_frames: int = 0
    video_fps: float = 0.0
    msg_frames: int = 0
    issues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core checks
# ---------------------------------------------------------------------------

def get_video_meta(mcap_path: Path, camera_topics: dict[str, str]) -> tuple[int, float]:
    """Return (total_frames, fps) from the first available camera attachment."""
    with mcap_path.open("rb") as f:
        reader = make_reader(f)
        for attach in reader.iter_attachments():
            if attach.media_type != "video/mp4":
                continue
            if attach.name not in camera_topics.values():
                continue
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(attach.data)
                tmp_path = tmp.name
            try:
                cap = cv2.VideoCapture(tmp_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                cap.release()
                return total, fps
            finally:
                Path(tmp_path).unlink(missing_ok=True)
    return 0, 0.0


def count_msg_frames(mcap_path: Path, config: TaskConfig) -> int:
    """Count the number of state/action message frames."""
    try:
        with mcap_path.open("rb") as f:
            reader = make_reader(f)
            msg_count = 0
            for _schema, _channel, _msg in reader.iter_messages(
                topics=config.state_topics + config.action_topics
            ):
                msg_count += 1
            num_topics = len(config.state_topics) + len(config.action_topics)
            return msg_count // num_topics if num_topics else 0
    except Exception:
        return 0


def check_one(mcap_path: Path, config: TaskConfig,
              min_size_mb: float, max_size_mb: float,
              fps_threshold: float) -> McapCheckResult:
    size_mb = mcap_path.stat().st_size / (1024 * 1024)
    result = McapCheckResult(
        path=str(mcap_path),
        size_mb=size_mb,
    )

    if size_mb < min_size_mb:
        result.issues.append(f"size too small ({size_mb:.1f} MB < {min_size_mb} MB)")
    if size_mb > max_size_mb:
        result.issues.append(f"size too large ({size_mb:.1f} MB > {max_size_mb} MB)")

    try:
        total_frames, fps = get_video_meta(mcap_path, config.camera_topics)
        result.video_frames = total_frames
        result.video_fps = fps

        if total_frames == 0:
            result.issues.append("no video frames")
        elif fps < fps_threshold:
            result.issues.append(f"low FPS ({fps:.1f} < {fps_threshold})")
    except Exception as e:
        result.issues.append(f"video read error: {e}")

    try:
        result.msg_frames = count_msg_frames(mcap_path, config)
    except Exception as e:
        result.issues.append(f"msg read error: {e}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quick health check for MCAP episodes")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Data directory containing config.py and episode folders")
    parser.add_argument("--min-size", type=float, default=5,
                        help="Minimum file size in MB (default: 5)")
    parser.add_argument("--max-size", type=float, default=30.0,
                        help="Maximum file size in MB (default: 100)")
    parser.add_argument("--fps-threshold", type=float, default=15.0,
                        help="Minimum acceptable FPS (default: 15)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max episodes to check (0 = all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config = load_task_config(data_dir / "config.py")

    print(f"Task: {config.task_name}")
    print(f"Folders: {len(config.folders)}")
    print(f"Thresholds: size [{args.min_size}, {args.max_size}] MB | FPS >= {args.fps_threshold}")
    print()

    all_results: list[McapCheckResult] = []
    bad_results: list[McapCheckResult] = []
    count = 0

    for folder_name in sorted(config.folders):
        folder_path = data_dir / folder_name
        if not folder_path.is_dir():
            continue
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith(".mcap"):
                continue
            if 0 < args.limit <= count:
                break

            mcap_path = folder_path / fname
            short = f"{folder_name}/{fname}"
            print(f"  [{count + 1}] {short} ...", end="", flush=True)

            result = check_one(mcap_path, config,
                               args.min_size, args.max_size,
                               args.fps_threshold)
            all_results.append(result)

            if result.issues:
                bad_results.append(result)
                print(f"  WARN  {result.size_mb:.1f}MB  fps={result.video_fps:.1f}"
                      f"  vframes={result.video_frames}  mframes={result.msg_frames}"
                      f"  [{', '.join(result.issues)}]")
            else:
                print(f"  OK  {result.size_mb:.1f}MB  fps={result.video_fps:.1f}"
                      f"  vframes={result.video_frames}  mframes={result.msg_frames}")

            count += 1

        if 0 < args.limit <= count:
            break

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"Checked {len(all_results)} mcap files.")

    if not bad_results:
        print("All files passed.")
        return

    print(f"\n{len(bad_results)} file(s) with issues:\n")

    size_small = [r for r in bad_results if any("too small" in i for i in r.issues)]
    size_large = [r for r in bad_results if any("too large" in i for i in r.issues)]
    low_fps = [r for r in bad_results if any("low FPS" in i for i in r.issues)]
    no_video = [r for r in bad_results if any("no video" in i for i in r.issues)]
    errors = [r for r in bad_results if any("error" in i for i in r.issues)]

    if size_small:
        print(f"--- Size too small (< {args.min_size} MB): {len(size_small)} ---")
        for r in size_small:
            print(f"  {r.size_mb:8.1f} MB  {r.path}")

    if size_large:
        print(f"\n--- Size too large (> {args.max_size} MB): {len(size_large)} ---")
        for r in size_large:
            print(f"  {r.size_mb:8.1f} MB  {r.path}")

    if low_fps:
        print(f"\n--- Low FPS (< {args.fps_threshold}): {len(low_fps)} ---")
        for r in low_fps:
            print(f"  {r.video_fps:8.1f} fps  {r.path}")

    if no_video:
        print(f"\n--- No video frames: {len(no_video)} ---")
        for r in no_video:
            print(f"  {r.path}")

    if errors:
        print(f"\n--- Read errors: {len(errors)} ---")
        for r in errors:
            errs = [i for i in r.issues if "error" in i]
            print(f"  {r.path}  [{'; '.join(errs)}]")


if __name__ == "__main__":
    main()
