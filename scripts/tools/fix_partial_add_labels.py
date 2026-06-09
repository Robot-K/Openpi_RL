#!/usr/bin/env python3
"""Repair a LeRobot repo whose schema was made inconsistent by an
interrupted ``add_labels`` run.

``scripts/add_returns_to_lerobot.py add_labels`` rewrites each episode parquet
by removing and then re-appending its added columns, which moves those columns
to the end. If the run is interrupted, only the already-processed episodes
have the new column order; the rest keep the old order. The set of columns is
identical, but the loader rejects the mix at concat time.

This tool reorders the deviating parquets to match the majority schema. It
also cleans up leftover ``.parquet.<staging>`` files from interrupted atomic
writes (the suffix used by add_labels.py is the same one we use here) and,
optionally, deletes parquets that fail to open.

Usage (from Openpi_RL/):
    uv run scripts/tools/fix_partial_add_labels.py --repo-id Fold_clothes_v3

    # preview without writing
    uv run scripts/tools/fix_partial_add_labels.py --repo-id Fold_clothes_v3 --dry-run
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import pathlib
import sys
from collections import Counter

import pyarrow.parquet as pq
import tqdm

STAGE_SUFFIX = ".parquet.tmp"


def _default_root() -> pathlib.Path:
    env = os.environ.get("HF_LEROBOT_HOME")
    if env:
        return pathlib.Path(env)
    return pathlib.Path("./lerobot_data")


def _find_parquets(frames_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(frames_dir.glob("chunk-*/episode_*.parquet"))


def _find_stage_files(frames_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(frames_dir.glob("chunk-*/*" + STAGE_SUFFIX))


def _classify(paths):
    schemas: Counter = Counter()
    per_path: dict = {}
    corrupted: list = []
    for p in tqdm.tqdm(paths, desc="Scanning schemas"):
        try:
            names = tuple(pq.read_schema(p).names)
        except Exception:
            corrupted.append(p)
            continue
        per_path[p] = names
        schemas[names] += 1
    return schemas, per_path, corrupted


def _reorder_one(parquet_path: pathlib.Path, target_order: tuple) -> int:
    table = pq.read_table(parquet_path)
    if tuple(table.column_names) == target_order:
        return 0
    missing = [c for c in target_order if c not in table.column_names]
    extra = [c for c in table.column_names if c not in target_order]
    if missing or extra:
        raise ValueError(
            f"{parquet_path.name}: column set differs from target "
            f"(missing={missing}, extra={extra})"
        )
    table = table.select(list(target_order))
    stage_path = parquet_path.with_suffix(STAGE_SUFFIX)
    pq.write_table(table, stage_path)
    stage_path.rename(parquet_path)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo-id", required=True,
                        help="LeRobot repo id (e.g. Fold_clothes_v3)")
    parser.add_argument("--root", type=pathlib.Path, default=None,
                        help="Root containing <repo-id>/. "
                             "Defaults to $HF_LEROBOT_HOME or ./lerobot_data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report only; make no changes.")
    parser.add_argument("--keep-stage", action="store_true",
                        help="Do not delete leftover " + STAGE_SUFFIX + " files.")
    parser.add_argument("--keep-corrupted", action="store_true",
                        help="Do not delete parquets that fail to open.")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    root = args.root or _default_root()
    repo_root = root / args.repo_id
    frames_dir = repo_root / "data"
    if not frames_dir.is_dir():
        sys.exit(f"frames directory not found: {frames_dir}")
    print(f"repo: {repo_root}")

    stage_files = _find_stage_files(frames_dir)
    print(f"Leftover {STAGE_SUFFIX} files: {len(stage_files)}")
    if stage_files and not args.keep_stage:
        if args.dry_run:
            print("  (dry-run) would delete:")
            for p in stage_files[:5]:
                print(f"    {p.relative_to(repo_root)}")
            if len(stage_files) > 5:
                print(f"    ... and {len(stage_files) - 5} more")
        else:
            for p in stage_files:
                p.unlink()
            print(f"  deleted {len(stage_files)} stage files")

    paths = _find_parquets(frames_dir)
    print(f"Parquet files: {len(paths)}")

    schemas, per_path, corrupted = _classify(paths)

    print(f"Distinct schemas (by column order): {len(schemas)}")
    for cols, n in schemas.most_common():
        print(f"  count={n}: {cols}")

    if corrupted:
        print(f"\nCorrupted (cannot open) parquets: {len(corrupted)}")
        for p in corrupted:
            print(f"  {p.relative_to(repo_root)}")
        if not args.keep_corrupted:
            if args.dry_run:
                print("  (dry-run) would delete the corrupted parquets above")
            else:
                for p in corrupted:
                    p.unlink()
                print(f"  deleted {len(corrupted)} corrupted parquet(s)")

    if not schemas:
        print("No readable parquets; nothing to reorder.")
        return

    target_order, target_count = schemas.most_common(1)[0]
    deviants = [p for p, names in per_path.items() if names != target_order]
    print(f"\nMajority schema has {target_count} parquets; "
          f"deviants to reorder: {len(deviants)}")
    if not deviants:
        print("All readable parquets already match the majority schema.")
        return

    if args.dry_run:
        print("(dry-run) would reorder the listed deviants. First few:")
        for p in deviants[:5]:
            print(f"  {p.relative_to(repo_root)}")
        return

    rewritten = 0
    errors: list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_reorder_one, p, target_order): p for p in deviants}
        for fut in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Reordering",
        ):
            p = futures[fut]
            try:
                rewritten += fut.result()
            except Exception as e:
                errors.append((p, str(e)))

    print(f"Reordered {rewritten} parquet(s).")
    if errors:
        print(f"Errors on {len(errors)} file(s):")
        for p, msg in errors[:20]:
            print(f"  {p.relative_to(repo_root)}: {msg}")


if __name__ == "__main__":
    main()
