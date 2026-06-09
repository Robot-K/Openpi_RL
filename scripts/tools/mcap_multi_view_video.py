"""Generate a side-by-side multi-camera video from a single MCAP file.

Useful for verifying that different camera streams in an MCAP are time-aligned.
Reads every video/mp4 attachment in the file, decodes each stream frame-by-frame,
concatenates them horizontally on a per-frame basis, and writes one mp4.

Usage:
    python scripts/tools/mcap_multi_view_video.py <mcap_path> [-o output.mp4] [--height 480]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader


def extract_video_attachments(mcap_path: Path) -> dict[str, bytes]:
    """Return {topic_name: mp4_bytes} for every video/mp4 attachment in the MCAP."""
    out: dict[str, bytes] = {}
    with mcap_path.open("rb") as f:
        reader = make_reader(f)
        for attach in reader.iter_attachments():
            if attach.media_type == "video/mp4":
                out[attach.name] = attach.data
    return out


def open_video_temp(data: bytes) -> tuple[cv2.VideoCapture, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(data)
    tmp.close()
    return cv2.VideoCapture(tmp.name), tmp.name


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mcap_path", type=str)
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Output mp4 path. Defaults to <mcap_stem>_multiview.mp4 next to the input.")
    p.add_argument("--height", type=int, default=480, help="Per-view height in px (default 480).")
    p.add_argument("--fps", type=float, default=None,
                   help="Output FPS. Defaults to the FPS of the first stream.")
    p.add_argument("--order", type=str, default=None,
                   help="Comma-separated topic names to control left-to-right order. "
                        "Defaults to a stable preferred order then alphabetical.")
    args = p.parse_args()

    mcap_path = Path(args.mcap_path)
    if not mcap_path.is_file():
        sys.exit(f"MCAP file not found: {mcap_path}")

    print(f"Reading {mcap_path} ...")
    attachments = extract_video_attachments(mcap_path)
    if not attachments:
        sys.exit("No video/mp4 attachments found in MCAP.")

    if args.order:
        topics = [t.strip() for t in args.order.split(",")]
        for t in topics:
            if t not in attachments:
                sys.exit(f"Requested topic '{t}' not found. Available: {list(attachments)}")
    else:
        preferred = ["left_wrist_0_rgb", "base_0_rgb", "right_wrist_0_rgb"]
        first = [t for t in preferred if t in attachments]
        rest = sorted(t for t in attachments if t not in first)
        topics = first + rest

    print(f"Found {len(topics)} video streams:")
    for t in topics:
        print(f"  - {t} ({len(attachments[t]) // 1024} KB)")

    caps_and_files = [open_video_temp(attachments[t]) for t in topics]
    caps = [c for c, _ in caps_and_files]
    tmp_paths = [path for _, path in caps_and_files]

    try:
        widths, heights, frame_counts, fps_list = [], [], [], []
        for c, t in zip(caps, topics):
            w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = c.get(cv2.CAP_PROP_FPS) or 0.0
            widths.append(w)
            heights.append(h)
            frame_counts.append(n)
            fps_list.append(fps)
            print(f"  {t}: {w}x{h} @ {fps:.2f} fps, {n} frames")

        if len(set(frame_counts)) > 1:
            print(f"WARNING: streams have different frame counts {frame_counts}; "
                  f"output truncated to {min(frame_counts)} frames.")
        n_frames = min(frame_counts)
        if n_frames <= 0:
            sys.exit("No decodable frames in at least one stream.")

        target_h = args.height
        scaled_widths = [max(1, int(round(w * target_h / h))) for w, h in zip(widths, heights)]
        total_w = sum(scaled_widths)

        out_fps = args.fps if args.fps else (fps_list[0] or 20.0)
        out_path = Path(args.output) if args.output else \
            mcap_path.with_name(mcap_path.stem + "_multiview.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (total_w, target_h))
        if not writer.isOpened():
            sys.exit(f"Failed to open VideoWriter for {out_path}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        try:
            for i in range(n_frames):
                tiles = []
                for c, t, sw in zip(caps, topics, scaled_widths):
                    ok, frame = c.read()
                    if not ok or frame is None:
                        frame = np.zeros((target_h, sw, 3), dtype=np.uint8)
                    elif (frame.shape[1], frame.shape[0]) != (sw, target_h):
                        frame = cv2.resize(frame, (sw, target_h))
                    cv2.putText(frame, t, (8, 22), font, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, t, (8, 22), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    tiles.append(frame)
                row = cv2.hconcat(tiles)
                label = f"frame {i}/{n_frames - 1}  t={i / out_fps:.2f}s"
                cv2.putText(row, label, (8, target_h - 12), font, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(row, label, (8, target_h - 12), font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                writer.write(row)
                if (i + 1) % 50 == 0 or i + 1 == n_frames:
                    print(f"  wrote {i + 1}/{n_frames}", end="\r", flush=True)
        finally:
            writer.release()

        print(f"\nSaved {out_path} ({total_w}x{target_h}, {n_frames} frames @ {out_fps:.2f} fps)")
    finally:
        for c in caps:
            c.release()
        for path in tmp_paths:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
