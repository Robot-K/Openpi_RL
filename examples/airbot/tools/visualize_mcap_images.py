"""Generate summary images for MCAP episodes.

For each MCAP file, produces one image containing:
  - Episode name, total frames, duration
  - Three snapshots from the middle camera: start / middle / end

Usage:
    python scripts/tools/visualize_mcap_images.py --data_dir mcap_data/fold_clothes
    python scripts/tools/visualize_mcap_images.py --data_dir mcap_data/fold_clothes --limit 20
    python scripts/tools/visualize_mcap_images.py --data_dir mcap_data/fold_clothes --out_dir mcap_previews
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import flatbuffers
import flatbuffers.number_types
import flatbuffers.table
import numpy as np
from mcap.exceptions import EndOfFile
from mcap.exceptions import RecordLengthLimitExceeded
from mcap.reader import NonSeekingReader
from mcap.reader import make_reader


# ---------------------------------------------------------------------------
# MCAP helpers
# ---------------------------------------------------------------------------

def decode_float_array(data: bytes) -> np.ndarray:
    root_offset = flatbuffers.packer.uoffset.unpack_from(data, 0)[0]
    tab = flatbuffers.table.Table(bytearray(data), root_offset)
    o = flatbuffers.number_types.UOffsetTFlags.py_type(tab.Offset(4))
    if o != 0:
        return tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
    return np.array([], dtype=np.float32)


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


def iter_attachments_safe(mcap_path: Path):
    """Iterate attachments, falling back to a sequential reader when the
    MCAP summary/footer is malformed.
    """
    try:
        with mcap_path.open("rb") as f:
            reader = make_reader(f)
            yield from reader.iter_attachments()
            return
    except RecordLengthLimitExceeded:
        pass

    try:
        with mcap_path.open("rb") as f:
            reader = NonSeekingReader(f)
            yield from reader.iter_attachments()
    except EndOfFile:
        return


def iter_messages_safe(mcap_path: Path, topics: list[str]):
    """Iterate messages, falling back to a sequential reader when the MCAP
    summary/footer is malformed.
    """
    try:
        with mcap_path.open("rb") as f:
            reader = make_reader(f)
            yield from reader.iter_messages(topics=topics)
            return
    except RecordLengthLimitExceeded:
        pass

    try:
        with mcap_path.open("rb") as f:
            reader = NonSeekingReader(f)
            yield from reader.iter_messages(topics=topics)
    except EndOfFile:
        return


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_video_info(mcap_path: Path, camera_topic: str) -> tuple[list[np.ndarray], int, float]:
    """Extract start/middle/end frames, total frame count, and fps from a
    camera video embedded in *mcap_path*.

    Returns (frames_bgr_list, total_frames, video_fps).
    """
    for attach in iter_attachments_safe(mcap_path):
        if attach.media_type == "video/mp4" and attach.name == camera_topic:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(attach.data)
                tmp_path = tmp.name
            try:
                cap = cv2.VideoCapture(tmp_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

                if total <= 0:
                    cap.release()
                    return [], 0, fps

                indices = [0, total // 2, max(total - 1, 0)]
                frames = []
                for idx in indices:
                    got = None
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        got = frame
                    else:
                        for fallback in range(idx - 1, max(idx - 30, -1), -1):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, fallback)
                            ret, frame = cap.read()
                            if ret:
                                got = frame
                                break
                    if got is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        last = None
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            last = frame
                        got = last
                    frames.append(got)
                cap.release()
                return frames, total, fps
            finally:
                Path(tmp_path).unlink(missing_ok=True)
    return [], 0, 0.0


def extract_all_cameras(mcap_path: Path, config: TaskConfig) -> dict[str, tuple[list[np.ndarray], int, float]]:
    """Extract frames for all cameras defined in config.camera_topics.

    Returns dict mapping camera_name -> (frames, total_frames, fps).
    """
    results = {}
    # Read all attachments in one pass
    camera_data: dict[str, bytes] = {}
    for attach in iter_attachments_safe(mcap_path):
        if attach.media_type == "video/mp4":
            for cam_name, topic in config.camera_topics.items():
                if attach.name == topic:
                    camera_data[cam_name] = attach.data

    for cam_name, topic in config.camera_topics.items():
        if cam_name not in camera_data:
            results[cam_name] = ([], 0, 0.0)
            continue
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(camera_data[cam_name])
            tmp_path = tmp.name
        try:
            cap = cv2.VideoCapture(tmp_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            if total <= 0:
                cap.release()
                results[cam_name] = ([], 0, fps)
                continue
            indices = [0, total // 2, max(total - 1, 0)]
            frames = []
            for idx in indices:
                got = None
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    got = frame
                else:
                    for fallback in range(idx - 1, max(idx - 30, -1), -1):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, fallback)
                        ret, frame = cap.read()
                        if ret:
                            got = frame
                            break
                if got is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    last = None
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        last = frame
                    got = last
                frames.append(got)
            cap.release()
            results[cam_name] = (frames, total, fps)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    return results


def count_msg_frames(mcap_path: Path, config: TaskConfig) -> int:
    """Count the number of state/action message frames."""
    try:
        msg_count = 0
        for _schema, _channel, _msg in iter_messages_safe(mcap_path, config.state_topics + config.action_topics):
            msg_count += 1
        num_topics = len(config.state_topics) + len(config.action_topics)
        return msg_count // num_topics if num_topics else 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Image composition (pure OpenCV, no matplotlib needed)
# ---------------------------------------------------------------------------

def put_text(img: np.ndarray, text: str, org: tuple[int, int],
             scale: float = 0.6, color: tuple = (255, 255, 255),
             thickness: int = 1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def make_summary_image(
    mcap_path: Path,
    config: TaskConfig,
    all_camera_data: dict[str, tuple[list[np.ndarray], int, float]] | None = None,
) -> np.ndarray | None:
    """Build a summary image for one MCAP episode showing all cameras.

    Each camera gets one row with start/middle/end frames.
    Returns BGR ndarray or None.
    """
    if all_camera_data is None:
        all_camera_data = extract_all_cameras(mcap_path, config)

    # Filter to cameras that have valid data; use config order
    cam_names = list(config.camera_topics.keys())
    valid_cams = [(name, all_camera_data[name]) for name in cam_names
                  if name in all_camera_data and all_camera_data[name][1] > 0]
    if not valid_cams:
        return None

    msg_frames = count_msg_frames(mcap_path, config)

    # Use first valid camera for global info
    _, (_, total_video_frames, fps) = valid_cams[0]
    duration_s = total_video_frames / fps if fps > 0 else 0

    thumb_h = 200
    gap = 10
    pad_top = 60
    pad_bottom = 16
    pad_side = 16
    cam_label_w = 140  # left column for camera name
    row_gap = 14

    # Determine thumb_w from first valid frame
    thumb_w = 320
    for _, (frames, total, _) in valid_cams:
        valid_f = [f for f in frames if f is not None]
        if valid_f:
            aspect = valid_f[0].shape[1] / valid_f[0].shape[0]
            thumb_w = int(thumb_h * aspect)
            break

    canvas_w = pad_side + cam_label_w + gap + thumb_w * 3 + gap * 2 + pad_side
    num_rows = len(valid_cams)
    canvas_h = pad_top + (thumb_h + 24 + row_gap) * num_rows + pad_bottom

    canvas = np.full((canvas_h, canvas_w, 3), (30, 30, 30), dtype=np.uint8)

    folder_name = mcap_path.parent.name
    file_name = mcap_path.name
    title = f"{folder_name}/{file_name}"
    dur_str = f"{duration_s:.1f}s" if duration_s < 60 else f"{int(duration_s)//60}m{int(duration_s)%60:02d}s"
    info_str = f"Video: {total_video_frames} frames | Msg: {msg_frames} frames | FPS: {fps:.0f} | Duration: {dur_str}"

    put_text(canvas, title, (pad_side, 24), 0.65, (255, 255, 255), 1)
    put_text(canvas, info_str, (pad_side, 46), 0.45, (180, 200, 220), 1)

    labels = ["Start", "Middle", "End"]
    for row_idx, (cam_name, (frames, total, cam_fps)) in enumerate(valid_cams):
        frame_indices = [0, total // 2, max(total - 1, 0)]
        y_base = pad_top + row_idx * (thumb_h + 24 + row_gap)

        # Camera name label on the left
        put_text(canvas, cam_name, (pad_side, y_base + thumb_h // 2), 0.42, (220, 200, 100), 1)

        for col_idx, (label, fidx) in enumerate(zip(labels, frame_indices)):
            x = pad_side + cam_label_w + gap + col_idx * (thumb_w + gap)
            f = frames[col_idx] if col_idx < len(frames) else None
            if f is not None:
                thumb = cv2.resize(f, (thumb_w, thumb_h))
            else:
                thumb = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
                put_text(thumb, "N/A", (thumb_w // 2 - 20, thumb_h // 2), 0.7, (128, 128, 128))
            canvas[y_base:y_base + thumb_h, x:x + thumb_w] = thumb

            tag = f"{label} (#{fidx})"
            put_text(canvas, tag, (x + 4, y_base + thumb_h + 16), 0.40, (160, 180, 200), 1)

    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate summary images for MCAP episodes")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="", help="Output directory (default: <data_dir>/previews)")
    parser.add_argument("--limit", type=int, default=0, help="Max episodes to process (0 = all)")
    parser.add_argument("--fps-threshold", type=float, default=15.0, help="Report videos with FPS below this value (default: 15)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config = load_task_config(data_dir / "config.py")
    # Use first camera topic for FPS reporting (base camera)
    primary_cam = list(config.camera_topics.keys())[0]
    primary_topic = config.camera_topics[primary_cam]

    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Task: {config.task_name}")
    print(f"Cameras: {list(config.camera_topics.keys())}")
    print(f"Output: {out_dir}")
    print()

    count = 0
    low_fps_list: list[tuple[str, float]] = []  # (path, fps)

    for folder_name in sorted(config.folders):
        folder_path = data_dir / folder_name
        if not folder_path.is_dir():
            continue

        folder_out = out_dir / folder_name
        folder_out.mkdir(parents=True, exist_ok=True)

        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith(".mcap"):
                continue
            if 0 < args.limit <= count:
                break

            mcap_path = folder_path / fname
            out_path = folder_out / (Path(fname).stem + ".jpg")

            print(f"  [{count+1}] {folder_name}/{fname} ...", end="", flush=True)

            try:
                all_camera_data = extract_all_cameras(mcap_path, config)
                primary_data = all_camera_data.get(primary_cam, ([], 0, 0.0))
                _, total_video_frames, fps = primary_data
                if total_video_frames > 0:
                    if fps < args.fps_threshold:
                        low_fps_list.append((str(mcap_path), fps))
                    img = make_summary_image(mcap_path, config, all_camera_data=all_camera_data)
                    if img is not None:
                        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        print(f" fps={fps:.1f} -> {out_path}")
                    else:
                        print(f" fps={fps:.1f} SKIP (no video)")
                else:
                    print(" SKIP (no video)")
            except Exception as e:
                print(f" ERROR: {e}")

            count += 1

        if 0 < args.limit <= count:
            break

    print(f"\nDone. {count} episodes processed. Images saved to {out_dir}")

    if low_fps_list:
        print(f"\n[WARNING] {len(low_fps_list)} video(s) with FPS < {args.fps_threshold}:")
        for path, fps in low_fps_list:
            print(f"  {fps:.1f} fps  {path}")
    else:
        print(f"\nAll videos have FPS >= {args.fps_threshold}.")


if __name__ == "__main__":
    main()
