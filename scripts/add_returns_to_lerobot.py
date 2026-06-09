"""Label a LeRobot dataset for pi0.6* (Recap) training.

Three modes of operation:

Mode 1 -- add_labels:
    Compute binned_value (progress-based), intervention (from parquet, written
    by the convert script) and stage fields.

    Progress labeling:
      - Success episodes (all except --failed-episodes): linear 0 -> 1
      - Failed episodes: linspace(0, 1, T_max)[:n], last 200 steps decayed to 0
      - Discretized into 200 bins spanning [0, 1]

    K-fold support (--num-folds K):
      When K > 0, episodes are randomly assigned to K folds (0..K-1).
      A ``fold`` column is written to each episode's parquet, and a
      ``meta/folds.json`` file is created mapping episode_index -> fold_id.

    Config-driven (optional):
        python scripts/add_returns_to_lerobot.py add_labels \
            --repo-id MyDataset \
            --config mcap_data/fold_clothes/config.py \
            --num-folds 5

    Direct args (no config file needed):
        python scripts/add_returns_to_lerobot.py add_labels \
            --repo-id MyDataset \
            --failed-episodes 3,7,12-15 \
            --stage-boundaries 100,300

Mode 2 -- vf_label:
    Use a trained value function checkpoint to infer per-timestep progress
    values, compute advantages, and write an is_good_action boolean column
    based on a percentile threshold.

    K-fold support (--infer-fold K):
      When specified, only episodes belonging to fold K are inferred.
      This is used by the K-fold pipeline so each VF only scores its
      held-out fold.

    Advantage formula (truncated to action horizon H):
      reward(t) = progress(t+1) - progress(t)
      baseline  = 1 / mean_episode_length
      advantage(t) = sum_{i=0}^{min(H, T-t)-1} gamma^i * (reward(t+i) - baseline)

    Default: 30% of data is positive (--positive-fraction 0.3)

    python scripts/add_returns_to_lerobot.py vf_label \
        --repo-id Fold_clothes \
        --vf-config pi06_rl_vf_airbot_clothes_folding \
        --vf-checkpoint-dir checkpoints/.../9999 \
        --positive-fraction 0.3
"""

import argparse
import concurrent.futures
import json
import logging
import os
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _read_jsonl(path) -> list[dict]:
    """Read a JSONL file, handling concatenated objects on a single line."""
    decoder = json.JSONDecoder()
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pos = 0
            while pos < len(line):
                obj, end = decoder.raw_decode(line, pos)
                results.append(obj)
                while end < len(line) and line[end] in " \t":
                    end += 1
                pos = end
    return results


def read_episode_lengths(repo_id: str) -> dict[int, int]:
    """Read episode lengths from ``episodes.jsonl`` metadata (no dataset loading)."""
    from lerobot.common.constants import HF_LEROBOT_HOME

    episodes_path = HF_LEROBOT_HOME / repo_id / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found: {episodes_path}")

    return {ep["episode_index"]: ep["length"] for ep in _read_jsonl(episodes_path)}


def parse_range_string(spec: str) -> set[int]:
    """Parse a range string like ``"0-100,200,300-400"`` into a set of ints."""
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return result


def read_fold_assignments(repo_id: str) -> dict[int, int]:
    """Read fold assignments from ``meta/folds.json``."""
    from lerobot.common.constants import HF_LEROBOT_HOME

    folds_path = HF_LEROBOT_HOME / repo_id / "meta" / "folds.json"
    if not folds_path.exists():
        raise FileNotFoundError(f"folds.json not found: {folds_path}")
    with open(folds_path) as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def compute_fold_assignments(
    episode_indices: list[int],
    num_folds: int,
    seed: int = 42,
) -> dict[int, int]:
    """Randomly assign episodes to K folds.

    Returns dict mapping episode_index -> fold_id (0..num_folds-1).
    """
    rng = np.random.RandomState(seed)
    shuffled = sorted(episode_indices)
    rng.shuffle(shuffled)
    assignments = {}
    for i, ep_idx in enumerate(shuffled):
        assignments[ep_idx] = i % num_folds
    return assignments


def resolve_success_episodes(
    spec: str | list[int],
    failed: list[int] | tuple[int, ...],
    all_episodes: set[int],
) -> set[int]:
    """Resolve success episode specification to a concrete set of indices.

    Args:
        spec: ``"all"``, a list of ints, or a range string like ``"0-100,200-300"``.
        failed: episode indices to exclude (overrides spec).
        all_episodes: complete set of valid episode indices.
    """
    if spec == "all":
        result = set(all_episodes)
    elif isinstance(spec, (list, tuple)):
        result = set(int(x) for x in spec)
    elif isinstance(spec, str):
        result = parse_range_string(spec)
    else:
        raise ValueError(f"Unsupported success_episodes spec: {spec!r}")
    result -= set(int(x) for x in failed)
    return result


# ---------------------------------------------------------------------------
# Computation functions
# ---------------------------------------------------------------------------

def compute_binned_value_progress(
    episode_lengths: dict[int, int],
    success_episodes: set[int],
    num_bins: int = 200,
    return_min: float = 0.0,
    return_max: float = 1.0,
) -> dict[int, np.ndarray]:
    """Compute binned values using progress-based labeling.

    Progress represents completion percentage: 0 at episode start, 1 at end.

    Success episodes: linear progress from 0 to 1, regardless of episode length.
    Failed episodes:  use linspace(0, 1, T_max)[:n] as base progress (slower
                      progress proportional to max episode length), then linearly
                      decay the last 200 steps to 0.
    """
    bin_edges = np.linspace(return_min, return_max, num_bins + 1)
    t_max = max(episode_lengths.values())

    result = {}
    for ep_idx, ep_len in episode_lengths.items():
        if ep_idx in success_episodes:
            # Success: linear 0 -> 1 over the episode
            progress = np.linspace(0, 1, ep_len)
        else:
            # Failed: take first n values from linspace(0, 1, t_max)
            full_progress = np.linspace(0, 1, t_max)
            progress = full_progress[:ep_len].copy()
            # Linearly decay last 200 steps to 0
            decay_len = min(200, ep_len)
            if decay_len > 0:
                decay_start = ep_len - decay_len
                decay_factors = np.linspace(1, 0, decay_len)
                progress[decay_start:] *= decay_factors

        bins = np.digitize(progress, bin_edges) - 1
        bins = np.clip(bins, 0, num_bins - 1)
        result[ep_idx] = bins

    return result


def read_intervention_from_dataset(
    repo_id: str,
    episode_lengths: dict[int, int],
) -> dict[int, np.ndarray]:
    """Read per-timestep intervention labels from parquet files.

    Returns dict mapping episode_index -> np.ndarray of shape (ep_len,) with 0/1 values.
    Falls back to all-zeros if the column is missing (e.g. pre-DAgger datasets).
    """
    from lerobot.common.constants import HF_LEROBOT_HOME

    ds_path = HF_LEROBOT_HOME / repo_id
    info_path = ds_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    chunks_size = info.get("chunks_size", 1000)

    result: dict[int, np.ndarray] = {}
    for ep_idx, ep_len in episode_lengths.items():
        chunk_idx = ep_idx // chunks_size
        parquet_path = (
            ds_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
        )
        if parquet_path.exists():
            schema = pq.read_schema(parquet_path)
            if "intervention" in schema.names:
                table = pq.read_table(parquet_path, columns=["intervention"])
                col = table.column("intervention").to_pylist()
                # Handle both scalar [0, 1, ...] and nested [[0], [1], ...] storage
                if col and isinstance(col[0], (list, tuple)):
                    arr = np.array([x[0] for x in col], dtype=np.int64)
                else:
                    arr = np.array(col, dtype=np.int64)
                result[ep_idx] = arr
            else:
                result[ep_idx] = np.zeros(ep_len, dtype=np.int64)
        else:
            result[ep_idx] = np.zeros(ep_len, dtype=np.int64)
    return result


# ---------------------------------------------------------------------------
# Mode 1: write binned_value + intervention to parquet
# ---------------------------------------------------------------------------

def _write_single_episode(
    ep_idx: int,
    parquet_path: pathlib.Path,
    columns: dict[str, dict[int, np.ndarray]],
    lenient: bool,
) -> int:
    """Read-modify-write a single episode parquet file (thread-safe).

    Returns the number of rows written, or 0 on skip.
    """
    if not parquet_path.exists():
        logger.warning("Parquet not found: %s, skipping ep %d", parquet_path, ep_idx)
        return 0

    table = pq.read_table(parquet_path)

    for col_name, ep_data in columns.items():
        if ep_idx in ep_data:
            arr = ep_data[ep_idx]
            if len(arr) != table.num_rows:
                if lenient:
                    logger.warning(
                        "ep %d col %s: len %d != rows %d, adjusting",
                        ep_idx, col_name, len(arr), table.num_rows,
                    )
                    if len(arr) > table.num_rows:
                        arr = arr[: table.num_rows]
                    else:
                        arr = np.pad(arr, (0, table.num_rows - len(arr)))
                else:
                    raise ValueError(
                        f"ep {ep_idx} col {col_name}: array length {len(arr)} "
                        f"!= parquet rows {table.num_rows}. "
                        "Use --lenient to pad/truncate instead of failing."
                    )
        else:
            arr = np.zeros(table.num_rows, dtype=np.int64)

        new_col = pa.array(arr)
        if col_name in table.column_names:
            table = table.set_column(table.column_names.index(col_name), col_name, new_col)
        else:
            table = table.append_column(col_name, new_col)

    tmp_path = parquet_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp_path)
    tmp_path.rename(parquet_path)
    return table.num_rows


def write_columns_to_dataset(
    repo_id: str,
    columns: dict[str, dict[int, np.ndarray]],
    column_meta: dict[str, dict],
    output_dir: str | None = None,
    lenient: bool = False,
    max_workers: int = 8,
):
    """Write arbitrary new columns to a LeRobot dataset (v2 parquet format).

    Uses atomic writes (write .tmp then rename) and parallel I/O.
    """
    from lerobot.common.constants import HF_LEROBOT_HOME
    import shutil

    src_path = HF_LEROBOT_HOME / repo_id
    if output_dir:
        dst_path = pathlib.Path(output_dir) / repo_id
        if dst_path.exists():
            shutil.rmtree(dst_path)
        shutil.copytree(src_path, dst_path)
        logger.info("Copied dataset to %s", dst_path)
    else:
        dst_path = src_path
        logger.info("Modifying dataset in-place at %s", dst_path)

    info_path = dst_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    episodes_path = dst_path / "meta" / "episodes.jsonl"
    episodes = _read_jsonl(episodes_path)

    chunks_size = info.get("chunks_size", 1000)
    tasks = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk_idx = ep_idx // chunks_size
        parquet_path = (
            dst_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
        )
        tasks.append((ep_idx, parquet_path))

    total_written = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_write_single_episode, ep_idx, pq_path, columns, lenient): ep_idx
            for ep_idx, pq_path in tasks
        }
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Writing columns",
        ):
            total_written += future.result()

    # Update info.json features
    changed = False
    for col_name, meta in column_meta.items():
        if col_name not in info.get("features", {}):
            info["features"][col_name] = meta
            changed = True
    if changed:
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)

    logger.info("Wrote %d column(s) to %d rows -> %s", len(columns), total_written, dst_path)


# ---------------------------------------------------------------------------
# Mode 2: value-function inference -> is_good_action
# ---------------------------------------------------------------------------

def load_value_function(vf_config_name: str, vf_checkpoint_dir: str):
    """Load a trained value function model from checkpoint."""
    import jax.numpy as jnp
    import openpi.models.model as _model
    import openpi.training.config as _config

    train_config = _config.get_config(vf_config_name)
    model_config = train_config.model
    params = _model.restore_params(
        pathlib.Path(vf_checkpoint_dir) / "params", dtype=jnp.bfloat16
    )
    vf_model = model_config.load(params)
    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    logger.info("Value function loaded from %s", vf_checkpoint_dir)
    return vf_model, data_config, model_config


class _VFInferDataset:
    """Dataset for VF inference with PyTorch DataLoader multiprocessing."""

    def __init__(self, lerobot_ds, transform_fn, tasks_mapping, frame_order):
        self.ds = lerobot_ds
        self.transform_fn = transform_fn
        self.tasks_mapping = tasks_mapping
        self.frame_order = frame_order  # list of (dataset_frame_idx, episode_idx)

    def __len__(self):
        return len(self.frame_order)

    def __getitem__(self, idx):
        dataset_fi, ep_idx = self.frame_order[idx]
        sample = self.ds[dataset_fi]
        sample_dict = {}
        for key in sample:
            val = sample[key]
            sample_dict[key] = val.numpy() if hasattr(val, "numpy") else val
        if "prompt" not in sample_dict and "task_index" in sample_dict:
            task_idx = int(sample_dict["task_index"])
            sample_dict["prompt"] = self.tasks_mapping.get(task_idx, "")
        result = self.transform_fn(sample_dict)
        result["_ep_idx"] = np.int64(ep_idx)
        return result


def _np_collate(batch):
    """Recursively collate a list of dicts into batched numpy arrays."""
    elem = batch[0]
    if isinstance(elem, dict):
        return {k: _np_collate([d[k] for d in batch]) for k in elem}
    elif isinstance(elem, (np.ndarray, np.generic)):
        return np.stack([np.asarray(x) for x in batch])
    else:
        return batch


def _dl_worker_init(worker_id: int) -> None:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def _run_vf_inference(
    repo_id: str,
    vf_config_name: str,
    vf_checkpoint_dir: str,
    batch_size: int,
    return_min: float,
    return_max: float,
    target_episodes: list[int],
    values_dir: str,
):
    """Run VF inference on a subset of episodes and save per-episode values to disk."""
    import multiprocessing
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import torch.utils.data
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    import openpi.transforms as _transforms
    from openpi.shared import array_typing as at
    import openpi.models.model as _model

    dataset = lerobot_dataset.LeRobotDataset(repo_id)

    episode_lengths = read_episode_lengths(repo_id)
    target_frames = sum(episode_lengths[ep] for ep in target_episodes)
    logger.info(
        "VF inference: %d episodes, %d frames (total dataset: %d eps, %d frames)",
        len(target_episodes), target_frames,
        len(episode_lengths), sum(episode_lengths.values()),
    )

    vf_model, data_config, model_config = load_value_function(
        vf_config_name, vf_checkpoint_dir
    )

    transforms_list = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(
            data_config.norm_stats,
            use_quantiles=data_config.use_quantile_norm,
        ),
        *data_config.model_transforms.inputs,
    ]
    transform_fn = _transforms.compose(transforms_list)

    tasks_mapping = dataset.meta.tasks
    logger.info("Task mapping: %s", tasks_mapping)

    rng = jax.random.key(0)

    @nnx.jit
    def infer_batch(model, obs):
        return model.infer_value(rng, obs)

    logger.info("JIT warmup: compiling VF inference graph...")
    import time as _time
    _t0 = _time.time()
    fake_obs = model_config.fake_obs(batch_size=batch_size)
    _ = jax.device_get(infer_batch(vf_model, fake_obs))
    logger.info("JIT warmup done in %.1fs", _time.time() - _t0)

    # Build ordered frame list: (dataset_frame_idx, episode_idx)
    ep_indices = np.array(dataset.hf_dataset["episode_index"])
    frame_order = []
    for ep_idx in target_episodes:
        ep_mask = ep_indices == ep_idx
        for fi in np.where(ep_mask)[0]:
            frame_order.append((int(fi), int(ep_idx)))

    if not frame_order:
        logger.info("No frames to infer, skipping.")
        return

    num_dl_workers = min(16, len(os.sched_getaffinity(0)))
    logger.info("Using PyTorch DataLoader with %d workers, prefetch_factor=4", num_dl_workers)

    infer_ds = _VFInferDataset(dataset, transform_fn, tasks_mapping, frame_order)
    loader = torch.utils.data.DataLoader(
        infer_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_dl_workers,
        multiprocessing_context=multiprocessing.get_context("spawn"),
        collate_fn=_np_collate,
        prefetch_factor=4,
        persistent_workers=True,
        worker_init_fn=_dl_worker_init,
        drop_last=False,
    )

    def _batch_to_obs(batched: dict):
        """Convert a batched numpy dict to an Observation on GPU."""
        converted = {}
        for k, v in batched.items():
            if isinstance(v, dict):
                converted[k] = {sk: jnp.asarray(sv) for sk, sv in v.items()}
            elif isinstance(v, np.ndarray):
                converted[k] = jnp.asarray(v)
            else:
                converted[k] = v
        with at.disable_typechecking():
            return _model.Observation.from_dict(converted)

    ep_values_map: dict[int, list[np.ndarray]] = {}
    values_path = pathlib.Path(values_dir)
    values_path.mkdir(parents=True, exist_ok=True)
    pbar = tqdm.tqdm(total=target_frames, desc="VF inference", unit="frame")
    saved_eps = 0

    for batch in loader:
        ep_idxs = batch.pop("_ep_idx")  # [B]
        cur_bs = len(ep_idxs)

        # Pad to batch_size if needed (avoid JIT recompilation for last batch)
        if cur_bs < batch_size:
            def _pad(arr):
                if isinstance(arr, np.ndarray):
                    pad_width = [(0, batch_size - cur_bs)] + [(0, 0)] * (arr.ndim - 1)
                    return np.pad(arr, pad_width)
                return arr

            def _pad_tree(tree):
                if isinstance(tree, dict):
                    return {k: _pad_tree(v) for k, v in tree.items()}
                return _pad(tree)

            batch = _pad_tree(batch)

        obs = _batch_to_obs(batch)
        values = np.array(jax.device_get(infer_batch(vf_model, obs)))[:cur_bs]

        # Distribute values to per-episode buckets
        for j in range(cur_bs):
            ep = int(ep_idxs[j])
            if ep not in ep_values_map:
                ep_values_map[ep] = []
            ep_values_map[ep].append(values[j])
        pbar.update(cur_bs)

        # Check if any episode is complete and flush to disk
        for ep in list(ep_values_map.keys()):
            if len(ep_values_map[ep]) == episode_lengths[ep]:
                ep_vals = np.array(ep_values_map[ep], dtype=np.float32)
                ep_vals = np.clip(ep_vals, return_min, return_max)
                np.save(values_path / f"ep_{ep:06d}.npy", ep_vals)
                saved_eps += 1
                pbar.set_postfix(ep=ep, vmin=f"{ep_vals.min():.4f}", vmax=f"{ep_vals.max():.4f}")
                del ep_values_map[ep]

    # Flush any remaining episodes
    for ep, vals_list in ep_values_map.items():
        ep_vals = np.array(vals_list, dtype=np.float32)
        ep_vals = np.clip(ep_vals, return_min, return_max)
        np.save(values_path / f"ep_{ep:06d}.npy", ep_vals)
        saved_eps += 1

    pbar.close()
    logger.info("VF inference done. Saved %d episode values to %s", saved_eps, values_dir)


def merge_and_label(
    repo_id: str,
    values_dir: str,
    positive_fraction: float = 0.3,
    gamma: float = 0.99,
    action_horizon: int = 50,
    pre_intervention_frames: int = 50,
    output_dir: str | None = None,
):
    """Load per-episode VF progress values from disk, compute advantages and is_good_action, write to dataset.

    Advantage formula (discounted returns truncated to action horizon H):
        reward(t) = progress(t+1) - progress(t)
        baseline  = 1 / mean_episode_length
        advantage(t) = sum_{i=0}^{min(H, T-t)-1} gamma^i * (reward(t+i) - baseline)

    DAgger intervention handling:
        - Frames with intervention=1: is_good_action forced to True.
        - The ``pre_intervention_frames`` frames immediately BEFORE each intervention
          start: is_good_action forced to False.
        - Both groups are EXCLUDED from the percentile threshold computation so they
          do not distort the positive_fraction ratio over the autonomous data.
    """
    episode_lengths = read_episode_lengths(repo_id)
    values_path = pathlib.Path(values_dir)

    all_values: dict[int, np.ndarray] = {}
    for ep_idx in sorted(episode_lengths.keys()):
        fpath = values_path / f"ep_{ep_idx:06d}.npy"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing value file for episode {ep_idx}: {fpath}")
        all_values[ep_idx] = np.load(fpath)

    total_frames = sum(len(v) for v in all_values.values())
    logger.info("Loaded values for %d episodes, %d frames from %s", len(all_values), total_frames, values_dir)

    # Validate lengths match
    for ep_idx, vals in all_values.items():
        expected = episode_lengths[ep_idx]
        if len(vals) != expected:
            raise ValueError(f"Episode {ep_idx}: value length {len(vals)} != expected {expected}")

    # Read per-timestep intervention labels from parquet (written by convert script)
    intervention_map = read_intervention_from_dataset(repo_id, episode_lengths)

    # Build per-episode masks: intervention and pre-intervention
    inter_masks: dict[int, np.ndarray] = {}   # intervention=1 frames
    pre_masks: dict[int, np.ndarray] = {}     # pre_intervention_frames frames before each start
    n_inter_total = 0
    n_pre_total = 0
    for ep_idx, ep_len in episode_lengths.items():
        intv = intervention_map[ep_idx]
        inter_mask = intv.astype(bool)
        pre_mask = np.zeros(ep_len, dtype=bool)
        # Find 0->1 transitions (intervention start points)
        for t in range(ep_len):
            if intv[t] == 1 and (t == 0 or intv[t - 1] == 0):
                start = max(0, t - pre_intervention_frames)
                pre_mask[start:t] = True
        # Pre-mask must not overlap with intervention frames
        pre_mask &= ~inter_mask
        inter_masks[ep_idx] = inter_mask
        pre_masks[ep_idx] = pre_mask
        n_inter_total += int(inter_mask.sum())
        n_pre_total += int(pre_mask.sum())

    logger.info(
        "DAgger masks: %d intervention frames, %d pre-intervention frames (window=%d)",
        n_inter_total, n_pre_total, pre_intervention_frames,
    )

    # Compute advantages: gamma-discounted sum truncated to action_horizon frames ahead
    if action_horizon < 1:
        raise ValueError(f"action_horizon must be >= 1, got {action_horizon}")
    mean_ep_len = np.mean(list(episode_lengths.values()))
    baseline_reward = 1.0 / mean_ep_len
    gamma_H = gamma ** action_horizon
    logger.info(
        "Computing advantages with gamma=%.3f, action_horizon=%d, mean_ep_len=%.1f, baseline_reward=%.6f",
        gamma, action_horizon, mean_ep_len, baseline_reward,
    )
    all_advantages: dict[int, np.ndarray] = {}
    for ep_idx, values in all_values.items():
        ep_len = len(values)
        rewards = np.zeros(ep_len, dtype=np.float64)
        if ep_len > 1:
            rewards[:-1] = np.diff(values)
        adj_rewards = rewards - baseline_reward
        advantages = np.zeros(ep_len, dtype=np.float64)
        running = 0.0
        # Recurrence: W[t] = adj_rewards[t] - gamma^H * adj_rewards[t+H] + gamma * W[t+1]
        # The subtraction term is 0 when t+H >= ep_len, giving a naturally capped tail sum.
        for t in range(ep_len - 1, -1, -1):
            drop = gamma_H * adj_rewards[t + action_horizon] if (t + action_horizon) < ep_len else 0.0
            running = adj_rewards[t] - drop + gamma * running
            advantages[t] = running
        all_advantages[ep_idx] = advantages

    # Compute threshold using ONLY autonomous (non-intervention, non-pre-intervention) frames
    normal_adv_list = []
    for ep_idx, adv in all_advantages.items():
        normal_mask = ~inter_masks[ep_idx] & ~pre_masks[ep_idx]
        normal_adv_list.append(adv[normal_mask])

    all_adv_normal = np.concatenate(normal_adv_list) if normal_adv_list else np.array([0.0])
    np_percentile = (1.0 - positive_fraction) * 100.0
    threshold = float(np.percentile(all_adv_normal, np_percentile))
    logger.info(
        "Advantage stats (autonomous frames only, n=%d): mean=%.4f, std=%.4f, "
        "threshold(p%.0f, positive_fraction=%.2f)=%.4f",
        len(all_adv_normal), all_adv_normal.mean(), all_adv_normal.std(),
        np_percentile, positive_fraction, threshold,
    )

    # Compute is_good_action
    is_good_action: dict[int, np.ndarray] = {}
    n_good = n_total = 0
    for ep_idx, adv in all_advantages.items():
        labels = (adv > threshold).astype(np.int64)
        # Pre-intervention frames: forced False
        labels[pre_masks[ep_idx]] = 0
        # Intervention frames: forced True
        labels[inter_masks[ep_idx]] = 1
        is_good_action[ep_idx] = labels
        n_good += int(labels.sum())
        n_total += len(labels)
    logger.info(
        "is_good_action: %d/%d = %.1f%% positive "
        "(includes %d forced-true intervention + %d forced-false pre-intervention frames)",
        n_good, n_total, 100 * n_good / n_total, n_inter_total, n_pre_total,
    )

    # Store
    predicted_value = {ep: vals.astype(np.float32) for ep, vals in all_values.items()}
    advantage = {ep: adv.astype(np.float32) for ep, adv in all_advantages.items()}

    write_columns_to_dataset(
        repo_id,
        columns={
            "is_good_action": is_good_action,
            "predicted_value": predicted_value,
            "advantage": advantage,
        },
        column_meta={
            "is_good_action": {"dtype": "int64", "shape": [1], "names": ["is_good_action"]},
            "predicted_value": {"dtype": "float32", "shape": [1], "names": ["predicted_value"]},
            "advantage": {"dtype": "float32", "shape": [1], "names": ["advantage"]},
        },
        output_dir=output_dir,
    )


def infer_values_for_dataset(
    repo_id: str,
    vf_config_name: str,
    vf_checkpoint_dir: str,
    positive_fraction: float = 0.3,
    gamma: float = 0.99,
    action_horizon: int = 50,
    pre_intervention_frames: int = 50,
    return_min: float = 0.0,
    return_max: float = 1.0,
    output_dir: str | None = None,
    batch_size: int = 32,
    infer_fold: int | None = None,
    values_dir: str | None = None,
):
    """Run VF inference on a LeRobot dataset and write is_good_action labels.

    When infer_fold is specified:
      - Only episodes belonging to that fold are inferred (reads meta/folds.json).
      - Values are saved to values_dir. Use ``vf_merge`` afterwards to combine
        results from all K folds and compute final labels.

    When infer_fold is None (default): runs inference on ALL episodes and
    immediately computes advantages + is_good_action.
    """
    if values_dir is None:
        values_dir = f"/tmp/vf_values_{repo_id}"

    episode_lengths = read_episode_lengths(repo_id)

    if infer_fold is not None:
        # K-fold mode: only infer on episodes belonging to this fold
        fold_map = read_fold_assignments(repo_id)
        target_episodes = sorted(ep for ep, fold in fold_map.items() if fold == infer_fold)
        logger.info("Fold %d: %d episodes to infer", infer_fold, len(target_episodes))
        _run_vf_inference(
            repo_id=repo_id,
            vf_config_name=vf_config_name,
            vf_checkpoint_dir=vf_checkpoint_dir,
            batch_size=batch_size,
            return_min=return_min,
            return_max=return_max,
            target_episodes=target_episodes,
            values_dir=values_dir,
        )
        return

    # Non-fold mode: inference on all episodes + merge in one go
    all_episodes = sorted(episode_lengths.keys())
    _run_vf_inference(
        repo_id=repo_id,
        vf_config_name=vf_config_name,
        vf_checkpoint_dir=vf_checkpoint_dir,
        batch_size=batch_size,
        return_min=return_min,
        return_max=return_max,
        target_episodes=all_episodes,
        values_dir=values_dir,
    )
    merge_and_label(
        repo_id=repo_id,
        values_dir=values_dir,
        positive_fraction=positive_fraction,
        gamma=gamma,
        action_horizon=action_horizon,
        pre_intervention_frames=pre_intervention_frames,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Label a LeRobot dataset for pi0.6* (Recap) training"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # -- Mode 1: add_labels ------------------------------------------------
    p1 = subparsers.add_parser(
        "add_labels",
        help="Add binned_value + intervention + stage columns",
    )
    p1.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="LeRobot dataset repo ID. Required unless --config provides TASK_NAME.",
    )
    p1.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a config.py file. Reads TASK_NAME, FAILED_EPISODES, "
             "STAGE_BOUNDARIES. Any value can be overridden by the corresponding CLI flag.",
    )
    p1.add_argument(
        "--failed-episodes",
        type=str,
        default=None,
        help="Comma/range spec of failed episode indices, e.g. '3,7,12-15'. "
             "All other episodes are treated as success. Overrides config.",
    )
    p1.add_argument(
        "--stage-boundaries",
        type=str,
        default=None,
        help="Comma-separated frame indices for stage boundaries, e.g. '100,300'. "
             "Overrides config.",
    )
    p1.add_argument("--num-bins", type=int, default=200)
    p1.add_argument("--return-min", type=float, default=0.0)
    p1.add_argument("--return-max", type=float, default=1.0)
    p1.add_argument("--output-dir", type=str, default=None)
    p1.add_argument(
        "--lenient",
        action="store_true",
        default=False,
        help="Pad/truncate on length mismatch instead of failing.",
    )
    p1.add_argument(
        "--num-folds",
        type=int,
        default=0,
        help="Number of folds for K-fold cross-validation. 0 = no folding.",
    )
    p1.add_argument(
        "--fold-seed",
        type=int,
        default=42,
        help="Random seed for fold assignment shuffle.",
    )

    # -- Mode 2: vf_label --------------------------------------------------
    p2 = subparsers.add_parser(
        "vf_label",
        help="Use trained VF to label is_good_action via advantage threshold",
    )
    p2.add_argument("--repo-id", required=True, help="LeRobot dataset repo ID")
    p2.add_argument("--vf-config", required=True)
    p2.add_argument("--vf-checkpoint-dir", required=True)
    p2.add_argument("--gamma", type=float, default=0.98, help="Discount factor for advantage computation")
    p2.add_argument("--action-horizon", type=int, default=50, help="Advantage summation horizon (frames): sum only reward[t..t+H-1]")
    p2.add_argument("--positive-fraction", type=float, default=0.3)
    p2.add_argument("--batch-size", type=int, default=32)
    p2.add_argument("--return-min", type=float, default=0.0)
    p2.add_argument("--return-max", type=float, default=1.0)
    p2.add_argument("--output-dir", type=str, default=None)
    p2.add_argument("--infer-fold", type=int, default=None, help="Only infer on episodes in this fold (reads meta/folds.json)")
    p2.add_argument("--values-dir", type=str, default=None, help="Directory for intermediate per-episode value files")
    p2.add_argument("--pre-intervention-frames", type=int, default=50, help="Frames before each intervention start to force is_good_action=False")

    # -- Mode 3: vf_merge --------------------------------------------------
    p3 = subparsers.add_parser(
        "vf_merge",
        help="Merge sharded VF values and compute is_good_action labels",
    )
    p3.add_argument("--repo-id", required=True, help="LeRobot dataset repo ID")
    p3.add_argument("--values-dir", required=True, help="Directory with per-episode value .npy files")
    p3.add_argument("--gamma", type=float, default=0.98, help="Discount factor for advantage computation")
    p3.add_argument("--action-horizon", type=int, default=50, help="Advantage summation horizon (frames): sum only reward[t..t+H-1]")
    p3.add_argument("--positive-fraction", type=float, default=0.3)
    p3.add_argument("--return-min", type=float, default=0.0)
    p3.add_argument("--return-max", type=float, default=1.0)
    p3.add_argument("--output-dir", type=str, default=None)
    p3.add_argument("--pre-intervention-frames", type=int, default=50, help="Frames before each intervention start to force is_good_action=False")

    args = parser.parse_args()

    if args.mode == "add_labels":
        _run_add_labels(args)
    elif args.mode == "vf_label":
        _run_vf_label(args)
    elif args.mode == "vf_merge":
        _run_vf_merge(args)


def _load_generic_config(config_path: str) -> dict:
    """Load minimal task parameters from a plain Python config file.

    Reads the following module-level variables (all optional except TASK_NAME):
      - TASK_NAME         (str)  — dataset repo ID
      - FAILED_EPISODES   (list/tuple of ints, default [])
      - STAGE_BOUNDARIES  (list/tuple of ints, default [])
    """
    import importlib.util

    config_path = pathlib.Path(config_path)
    spec = importlib.util.spec_from_file_location("_add_returns_cfg", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "TASK_NAME"):
        raise ValueError(f"Config file {config_path} must define TASK_NAME")

    return {
        "task_name": mod.TASK_NAME,
        "failed_episodes": tuple(getattr(mod, "FAILED_EPISODES", ())),
        "stage_boundaries": tuple(getattr(mod, "STAGE_BOUNDARIES", ())),
    }


def _run_add_labels(args):
    """Execute Mode 1: add binned_value + intervention + stage."""

    # --- Resolve repo_id, failed_episodes, stage_boundaries ----------------
    # Load from config file first (if given), then override with CLI args.
    cfg_task_name: str | None = None
    cfg_failed: tuple[int, ...] = ()
    cfg_stage_boundaries: tuple[int, ...] = ()

    if args.config:
        cfg = _load_generic_config(args.config)
        cfg_task_name = cfg["task_name"]
        cfg_failed = cfg["failed_episodes"]
        cfg_stage_boundaries = cfg["stage_boundaries"]
        logger.info("Loaded config from %s (task=%s)", args.config, cfg_task_name)

    repo_id: str = args.repo_id or cfg_task_name  # type: ignore[assignment]
    if not repo_id:
        raise SystemExit("--repo-id is required (or set TASK_NAME in --config)")

    # CLI --failed-episodes overrides config
    if args.failed_episodes is not None:
        failed_set_raw = parse_range_string(args.failed_episodes)
        failed_list: list[int] = sorted(failed_set_raw)
    else:
        failed_list = list(cfg_failed)

    # CLI --stage-boundaries overrides config
    if args.stage_boundaries is not None:
        stage_boundaries: tuple[int, ...] = tuple(
            int(x) for x in args.stage_boundaries.split(",") if x.strip()
        )
    else:
        stage_boundaries = cfg_stage_boundaries

    logger.info("Stage boundaries: %s", stage_boundaries)

    # Read episode metadata directly (no dataset loading)
    episode_lengths = read_episode_lengths(repo_id)
    total_frames = sum(episode_lengths.values())
    logger.info("Found %d episodes, %d frames", len(episode_lengths), total_frames)

    all_ep_indices = set(episode_lengths.keys())

    # Success = all episodes except explicitly listed failed ones.
    failed_set = set(failed_list) & all_ep_indices
    unknown_failed = set(failed_list) - all_ep_indices
    if unknown_failed:
        logger.warning("Failed episodes not found in dataset (ignored): %s", sorted(unknown_failed))
    success_set = all_ep_indices - failed_set

    logger.info("Success episodes: %d / %d", len(success_set), len(episode_lengths))
    if failed_set:
        logger.info("Failed episodes: %s", sorted(failed_set))

    # Binned values (progress-based)
    binned = compute_binned_value_progress(
        episode_lengths,
        success_set,
        num_bins=args.num_bins,
        return_min=args.return_min,
        return_max=args.return_max,
    )
    binned = {k: v.astype(np.int64) for k, v in binned.items()}

    # Intervention labels: read per-frame values written by the convert script
    # (from /dagger/intervention topic).  Falls back to all-zeros for datasets
    # that pre-date DAgger support (no intervention column in parquet).
    intervention = read_intervention_from_dataset(repo_id, episode_lengths)
    intervention = {k: v.astype(np.int64) for k, v in intervention.items()}

    # Stage labels
    stage_labels = {}
    if stage_boundaries:
        logger.info("Computing stage labels with boundaries: %s", stage_boundaries)
        for ep_idx, ep_len in episode_lengths.items():
            stages = np.ones(ep_len, dtype=np.int64)
            
            for stage_idx, boundary in enumerate(stage_boundaries):
                if boundary < ep_len:
                    stages[boundary:] = stage_idx + 2
                else:
                    break
            
            stage_labels[ep_idx] = stages
            
        all_stages = np.concatenate([stage_labels[ep] for ep in sorted(stage_labels)])
        unique, counts = np.unique(all_stages, return_counts=True)
        for stage, count in zip(unique, counts):
            logger.info("Stage %d: %d frames (%.1f%%)", 
                       stage, count, 100*count/len(all_stages))
    else:
        logger.info("No stage boundaries specified, using default (all frames in stage 1)")
        for ep_idx, ep_len in episode_lengths.items():
            stage_labels[ep_idx] = np.ones(ep_len, dtype=np.int64)

    # Stats
    all_bins = np.concatenate([binned[ep] for ep in sorted(binned)])
    logger.info("binned_value range: [%d, %d], mean: %.2f", all_bins.min(), all_bins.max(), all_bins.mean())
    all_intv = np.concatenate([intervention[ep] for ep in sorted(intervention)])
    logger.info("intervention: %d/%d timesteps marked", all_intv.sum(), len(all_intv))

    write_columns_to_dataset(
        repo_id,
        columns={
            "binned_value": binned,
            "intervention": intervention,
            "stage": stage_labels
        },
        column_meta={
            "binned_value": {"dtype": "int64", "shape": [1], "names": ["binned_value"]},
            "intervention": {"dtype": "int64", "shape": [1], "names": ["intervention"]},
            "stage": {"dtype": "int64", "shape": [1], "names": ["stage"]},
        },
        output_dir=args.output_dir,
        lenient=args.lenient,
    )

    # K-fold assignment
    num_folds = args.num_folds
    if num_folds > 0:
        fold_assignments = compute_fold_assignments(
            sorted(episode_lengths.keys()), num_folds, seed=args.fold_seed,
        )

        # Write fold column to each episode's parquet
        fold_col = {}
        for ep_idx, ep_len in episode_lengths.items():
            fold_col[ep_idx] = np.full(ep_len, fold_assignments[ep_idx], dtype=np.int64)

        write_columns_to_dataset(
            repo_id,
            columns={"fold": fold_col},
            column_meta={"fold": {"dtype": "int64", "shape": [1], "names": ["fold"]}},
            output_dir=args.output_dir,
            lenient=args.lenient,
        )

        # Write meta/folds.json
        from lerobot.common.constants import HF_LEROBOT_HOME
        if args.output_dir:
            ds_path = pathlib.Path(args.output_dir) / repo_id
        else:
            ds_path = HF_LEROBOT_HOME / repo_id
        folds_path = ds_path / "meta" / "folds.json"
        with open(folds_path, "w") as f:
            json.dump({str(k): v for k, v in fold_assignments.items()}, f, indent=2)

        # Log fold distribution
        for fold_id in range(num_folds):
            fold_eps = [ep for ep, fold in fold_assignments.items() if fold == fold_id]
            fold_frames = sum(episode_lengths[ep] for ep in fold_eps)
            logger.info(
                "Fold %d: %d episodes, %d frames", fold_id, len(fold_eps), fold_frames,
            )
        logger.info("Fold assignments written to %s", folds_path)


def _run_vf_label(args):
    """Execute Mode 2: VF inference -> is_good_action."""
    infer_values_for_dataset(
        repo_id=args.repo_id,
        vf_config_name=args.vf_config,
        vf_checkpoint_dir=args.vf_checkpoint_dir,
        positive_fraction=args.positive_fraction,
        gamma=args.gamma,
        action_horizon=args.action_horizon,
        pre_intervention_frames=args.pre_intervention_frames,
        return_min=args.return_min,
        return_max=args.return_max,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        infer_fold=args.infer_fold,
        values_dir=args.values_dir,
    )


def _run_vf_merge(args):
    """Execute Mode 3: merge sharded VF values -> is_good_action."""
    merge_and_label(
        repo_id=args.repo_id,
        values_dir=args.values_dir,
        positive_fraction=args.positive_fraction,
        gamma=args.gamma,
        action_horizon=args.action_horizon,
        pre_intervention_frames=args.pre_intervention_frames,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
