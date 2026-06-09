"""Fix episodes whose intervention column is mostly 1s.

For a LeRobot v2.1 dset, scan every episode parquet under DATA/chunk-*/episode_*.parquet.
If an episode has a fraction of intervention==1 values above --threshold (default 0.8),
rewrite the whole column to all 0s and update meta/episodes_stats.jsonl to match.

Usage:
    python scripts/tools/fix_all_intervention_episodes.py \\
        --root lerobot_data/Fold_clothes_v3 [--threshold 0.8] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def episode_paths(root: Path) -> list[Path]:
    pq_root = root / "data"
    if not pq_root.is_dir():
        sys.exit(f"data directory not found: {pq_root}")
    return sorted(pq_root.glob("chunk-*/episode_*.parquet"))


def parquet_intervention(path: Path) -> np.ndarray:
    col = pq.read_table(path, columns=["intervention"]).column("intervention")
    return np.asarray(col.to_pylist(), dtype=np.int64).reshape(-1)


def rewrite_intervention_zero(path: Path) -> None:
    """Read parquet, replace intervention column with zeros, atomic-replace the file."""
    table = pq.read_table(path)
    field = table.schema.field("intervention")
    n = table.num_rows
    zeros = pa.array(np.zeros(n, dtype=np.int64), type=field.type)
    idx = table.schema.get_field_index("intervention")
    new_table = table.set_column(idx, field, zeros)
    staging = path.with_suffix(path.suffix + ".staging")
    pq.write_table(new_table, staging)
    staging.replace(path)


def update_stats_file(stats_path: Path, fixed_episodes: set[int]) -> None:
    if not stats_path.is_file():
        print(f"  (no episodes_stats.jsonl at {stats_path}, skipping stats update)")
        return
    lines = stats_path.read_text().splitlines()
    out_lines = []
    updated = 0
    for line in lines:
        if not line.strip():
            out_lines.append(line)
            continue
        rec = json.loads(line)
        if rec.get("episode_index") in fixed_episodes:
            iv = rec.get("stats", {}).get("intervention")
            if iv is not None:
                iv["min"] = [0]
                iv["max"] = [0]
                iv["mean"] = [0.0]
                iv["std"] = [0.0]
                updated += 1
        out_lines.append(json.dumps(rec))
    staging = stats_path.with_suffix(stats_path.suffix + ".staging")
    staging.write_text("\n".join(out_lines) + "\n")
    staging.replace(stats_path)
    print(f"  updated intervention stats for {updated} episodes in {stats_path.name}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, type=str,
                   help="Path to LeRobot dset root (the dir containing data/ and meta/).")
    p.add_argument("--threshold", type=float, default=0.8,
                   help="Fraction of intervention==1 above which the episode is fixed (default 0.8).")
    p.add_argument("--dry-run", action="store_true",
                   help="Only print which episodes would be modified; do not write.")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"root directory not found: {root}")

    paths = episode_paths(root)
    print(f"Scanning {len(paths)} episode parquet files in {root}")

    to_fix: list[tuple[Path, int]] = []
    for i, path in enumerate(paths):
        try:
            iv = parquet_intervention(path)
        except Exception as e:
            print(f"  [{i}] ERROR reading {path}: {e}")
            continue
        if iv.size == 0:
            continue
        # Fraction of timesteps flagged as intervention; fix the episode if it dominates.
        frac = float((iv == 1).mean())
        if frac > args.threshold:
            ep_idx = int(path.stem.split("_")[-1])
            to_fix.append((path, ep_idx))
        if (i + 1) % 500 == 0:
            print(f"  scanned {i + 1}/{len(paths)} (found {len(to_fix)} mostly-1 episodes so far)")

    print(f"\nFound {len(to_fix)} episodes with >{args.threshold:.0%} intervention.")
    if to_fix:
        preview = ", ".join(str(ep) for _, ep in to_fix[:20])
        more = "" if len(to_fix) <= 20 else f", ... (+{len(to_fix) - 20} more)"
        print(f"  episode_index: {preview}{more}")

    if args.dry_run:
        print("\n[dry-run] Skipping writes.")
        return

    if not to_fix:
        return

    print("\nRewriting parquet files...")
    for n, (path, ep_idx) in enumerate(to_fix, 1):
        rewrite_intervention_zero(path)
        if n % 100 == 0 or n == len(to_fix):
            print(f"  rewrote {n}/{len(to_fix)}")

    fixed_eps = {ep for _, ep in to_fix}
    print("\nUpdating episodes_stats.jsonl...")
    update_stats_file(root / "meta" / "episodes_stats.jsonl", fixed_eps)

    print(f"\nDone. Fixed {len(to_fix)} episodes.")


if __name__ == "__main__":
    main()
