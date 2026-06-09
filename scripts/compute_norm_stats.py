"""Compute normalization statistics for a config.

Fast path (default): reads state/actions directly from parquet files,
bypassing image/video decoding entirely. Typically finishes in seconds.

Slow path (--use-dataloader): uses the full training data pipeline including
image decoding. Only needed if your data transforms modify state/actions values
in ways that depend on images (unlikely for standard setups).
"""

import pathlib
import time

import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


STAT_KEYS = ["state", "actions"]


def compute_stats_fast(
    data_config: _config.DataConfig,
    action_dim: int | None = None,
    max_frames: int | None = None,
) -> dict[str, normalize.RunningStats]:
    """Read state/actions directly from parquet — no image decoding."""
    from lerobot.common.constants import HF_LEROBOT_HOME

    dataset_path = HF_LEROBOT_HOME / data_config.repo_id
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    stats = {key: normalize.RunningStats() for key in STAT_KEYS}
    total_frames = 0

    for pf in tqdm.tqdm(parquet_files, desc="Reading parquet"):
        schema = pq.read_schema(pf)
        columns = [k for k in STAT_KEYS if k in schema.names]
        if not columns:
            continue

        table = pq.read_table(pf, columns=columns)

        for key in columns:
            arr = np.stack(table.column(key).to_numpy())  # (T, dim)
            if action_dim is not None:
                arr = transforms.pad_to_dim(arr, action_dim)
            stats[key].update(arr)

        total_frames += len(table)
        if max_frames is not None and total_frames >= max_frames:
            break

    print(f"Processed {total_frames} frames from {len(parquet_files)} files")
    return {key: s for key, s in stats.items() if s._count > 0}


# ---------------------------------------------------------------------------
# Slow path: full data pipeline (kept for RLDS or non-standard transforms)
# ---------------------------------------------------------------------------


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def compute_stats_slow(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    max_frames: int | None = None,
) -> dict[str, normalize.RunningStats]:
    """Full pipeline — decodes images, applies all transforms."""
    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = _create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = _create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size,
            config.model, config.num_workers, max_frames,
        )

    stats = {key: normalize.RunningStats() for key in STAT_KEYS}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in STAT_KEYS:
            if key in batch:
                stats[key].update(np.asarray(batch[key]))

    return {key: s for key, s in stats.items() if s._count > 0}


def _create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def _create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(config_name: str, max_frames: int | None = None, use_dataloader: bool = False):
    """Compute and save normalization statistics.

    Args:
        config_name: Training config name (e.g. pi06_rl_vf_airbot_clothes_folding).
        max_frames: Cap the number of frames to process (for quick iteration).
        use_dataloader: Use the slow full-pipeline path (needed for RLDS datasets).
    """
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    t0 = time.time()

    if use_dataloader or data_config.rlds_data_dir is not None:
        print("Using slow path (full data pipeline with image decoding)")
        stats = compute_stats_slow(config, data_config, max_frames)
    else:
        action_dim = getattr(config.model, "action_dim", None)
        print(f"Using fast path (parquet-only, action_dim={action_dim})")
        stats = compute_stats_fast(data_config, action_dim, max_frames)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}

    for key, ns in norm_stats.items():
        print(f"  {key}: mean={ns.mean[:4]}..., std={ns.std[:4]}..., "
              f"q01={ns.q01[:4]}..., q99={ns.q99[:4]}...")

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
