#!/usr/bin/env python3
"""
Script to add newly added episodes to the folds.json file.

This script:
1. Scans all episode files in the data directory
2. Identifies episodes not yet in folds.json
3. Assigns them randomly to fold 0, 1, or 2
4. Maintains the existing fold assignments for old episodes
"""

import json
import random
from pathlib import Path
from typing import Dict, Set
import argparse


def get_existing_episode_ids(data_dir: Path) -> Set[int]:
    """Get all episode IDs that exist in the data directory."""
    episode_files = data_dir.glob("**/episode_*.parquet")
    episode_ids = set()
    
    for file in episode_files:
        # Extract episode number from filename (e.g., "episode_000123.parquet" -> 123)
        episode_num = int(file.stem.split("_")[1])
        episode_ids.add(episode_num)
    
    return episode_ids


def load_folds(folds_file: Path) -> Dict[str, int]:
    """Load existing folds.json file."""
    if not folds_file.exists():
        return {}
    
    with open(folds_file, 'r') as f:
        return json.load(f)


def get_folds_episode_ids(folds: Dict[str, int]) -> Set[int]:
    """Extract episode IDs from folds dictionary."""
    return {int(ep_id) for ep_id in folds.keys()}


def find_new_episodes(existing_ids: Set[int], folds_ids: Set[int]) -> Set[int]:
    """Find episodes that exist in data but not in folds."""
    return existing_ids - folds_ids


def update_folds(folds: Dict[str, int], new_episodes: Set[int], seed: int = 42) -> Dict[str, int]:
    """Add new episodes to folds with random assignment to 3 folds."""
    random.seed(seed)
    
    for episode_id in sorted(new_episodes):
        fold_assignment = random.randint(0, 2)
        folds[str(episode_id)] = fold_assignment
    
    return folds


def save_folds(folds: Dict[str, int], folds_file: Path, backup: bool = True) -> None:
    """Save updated folds to file."""
    # Create backup if requested
    if backup and folds_file.exists():
        backup_file = folds_file.with_suffix('.json.bak')
        print(f"Creating backup: {backup_file}")
        with open(folds_file, 'r') as f:
            backup_content = f.read()
        with open(backup_file, 'w') as f:
            f.write(backup_content)
    
    # Sort by episode ID (as integers) for consistent output
    sorted_folds = dict(sorted(folds.items(), key=lambda x: int(x[0])))
    
    with open(folds_file, 'w') as f:
        json.dump(sorted_folds, f, indent=2)
    
    print(f"Updated folds saved to: {folds_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Add newly added episodes to the folds.json file"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/data/kding/Openpi_RL/lerobot_data/Fold_clothes_v3/data"),
        help="Path to the data directory containing episodes"
    )
    parser.add_argument(
        "--folds-file",
        type=Path,
        default=Path("/data/kding/Openpi_RL/lerobot_data/Fold_clothes_v3/meta/folds.json"),
        help="Path to the folds.json file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible fold assignment"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a backup of the original folds.json"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without actually modifying files"
    )
    
    args = parser.parse_args()
    
    # Validate directories
    if not args.data_dir.exists():
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return 1
    
    if not args.folds_file.exists():
        print(f"Warning: Folds file does not exist, will create new one: {args.folds_file}")
    
    # Get all episode IDs
    print(f"Scanning data directory: {args.data_dir}")
    existing_ids = get_existing_episode_ids(args.data_dir)
    print(f"Total episodes in data: {len(existing_ids)}")
    
    # Load existing folds
    print(f"Loading folds from: {args.folds_file}")
    folds = load_folds(args.folds_file)
    folds_ids = get_folds_episode_ids(folds)
    print(f"Total episodes in folds: {len(folds_ids)}")
    
    # Find new episodes
    new_episodes = find_new_episodes(existing_ids, folds_ids)
    print(f"New episodes to add: {len(new_episodes)}")
    
    if not new_episodes:
        print("No new episodes to add.")
        return 0
    
    # Show sample of new episodes
    new_episodes_sorted = sorted(new_episodes)
    print(f"Sample of new episodes: {new_episodes_sorted[:10]}")
    if len(new_episodes_sorted) > 10:
        print(f"... and {len(new_episodes_sorted) - 10} more")
    
    # Show fold distribution before and after
    fold_counts_before = [0, 0, 0]
    for fold_id in folds.values():
        fold_counts_before[fold_id] += 1
    print(f"\nFold distribution before update:")
    print(f"  Fold 0: {fold_counts_before[0]}")
    print(f"  Fold 1: {fold_counts_before[1]}")
    print(f"  Fold 2: {fold_counts_before[2]}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would add new episodes to folds")
        # Show predicted fold distribution
        new_episodes_per_fold = [0, 0, 0]
        random.seed(args.seed)
        for _ in range(len(new_episodes)):
            fold_id = random.randint(0, 2)
            new_episodes_per_fold[fold_id] += 1
        
        print(f"\nPredicted fold distribution after update:")
        print(f"  Fold 0: {fold_counts_before[0]} + {new_episodes_per_fold[0]} = {fold_counts_before[0] + new_episodes_per_fold[0]}")
        print(f"  Fold 1: {fold_counts_before[1]} + {new_episodes_per_fold[1]} = {fold_counts_before[1] + new_episodes_per_fold[1]}")
        print(f"  Fold 2: {fold_counts_before[2]} + {new_episodes_per_fold[2]} = {fold_counts_before[2] + new_episodes_per_fold[2]}")
        return 0
    
    # Update folds
    print(f"\nUpdating folds with seed={args.seed}...")
    updated_folds = update_folds(folds, new_episodes, seed=args.seed)
    
    # Show fold distribution after
    fold_counts_after = [0, 0, 0]
    for fold_id in updated_folds.values():
        fold_counts_after[fold_id] += 1
    print(f"\nFold distribution after update:")
    print(f"  Fold 0: {fold_counts_after[0]}")
    print(f"  Fold 1: {fold_counts_after[1]}")
    print(f"  Fold 2: {fold_counts_after[2]}")
    
    # Save updated folds
    save_folds(updated_folds, args.folds_file, backup=not args.no_backup)
    
    print("\n✓ Successfully updated folds.json with new episodes")
    return 0


if __name__ == "__main__":
    exit(main())
