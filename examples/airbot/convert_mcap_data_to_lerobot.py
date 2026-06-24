"""Convert AIRBOT mcap dataset to LeRobot format.

Usage:
    HF_LEROBOT_HOME=./lerobot_data uv run examples/airbot/convert_mcap_data_to_lerobot.py \
        --data_dir mcap_data/fold_clothv2

    # Overwrite existing dataset:
    HF_LEROBOT_HOME=./lerobot_data uv run examples/airbot/convert_mcap_data_to_lerobot.py \
        --data_dir mcap_data/fold_clothv2 --force

    # Resume (append new episodes to existing dataset):
    HF_LEROBOT_HOME=./lerobot_data uv run examples/airbot/convert_mcap_data_to_lerobot.py \
        --data_dir mcap_data/fold_clothv2 --resume

    # Write to a separate output directory (e.g. when two data dirs share the same TASK_NAME):
    HF_LEROBOT_HOME=./lerobot_data uv run examples/airbot/convert_mcap_data_to_lerobot.py \
        --data_dir mcap_data/fold_clothv2_dagger1 --repo_id Fold_clothes_dagger1

    # Convert only intervention=1 segments and split each contiguous segment
    # into its own LeRobot episode:
    HF_LEROBOT_HOME=./lerobot_data uv run examples/airbot/convert_mcap_data_to_lerobot.py \
        --data_dir mcap_data/fold_clothv2_dagger1 --intervention-only

Each data directory (e.g. mcap_data/fold_clothv2, mcap_data/fold_clothv2_dagger1) should be
placed in its own folder and converted separately. Use --repo_id to give each its own output
directory under HF_LEROBOT_HOME; without it the output name defaults to TASK_NAME from config.py.

The data directory must contain a ``config.py`` describing topics, cameras, etc.
See ``mcap_data/config.py`` for an example.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

import cv2
import flatbuffers
import flatbuffers.number_types
import flatbuffers.table
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import lerobot.common.datasets.lerobot_dataset as _lerobot_ds_mod
from mcap.reader import make_reader
import numpy as np
import tyro


# --- save_episode sub-step timing (monkey-patch) ---
_SAVE_SUBSTEP_TIMINGS = {
    "wait_image_writer": 0.0,
    "save_episode_table": 0.0,
    "compute_episode_stats": 0.0,
    "meta_save_episode": 0.0,
}
_META_PATCHED = False
_LEROBOT_ORIGINAL_LOAD_DATASET = _lerobot_ds_mod.load_dataset
_LEROBOT_HF_LOAD_PATCHED = False


def _patch_lerobot_hf_load_dataset_num_proc() -> None:
    """Allow parallel HuggingFace parquet loading when resuming a large dataset."""
    global _LEROBOT_HF_LOAD_PATCHED
    if _LEROBOT_HF_LOAD_PATCHED:
        return

    raw_num_proc = os.environ.get("LEROBOT_HF_LOAD_NUM_PROC", "")
    if not raw_num_proc:
        return

    try:
        num_proc = int(raw_num_proc)
    except ValueError:
        print(f"Warning: ignoring invalid LEROBOT_HF_LOAD_NUM_PROC={raw_num_proc!r}")
        return

    if num_proc <= 1:
        return

    def load_dataset_with_num_proc(*args, **kwargs):
        if args and args[0] == "parquet":
            kwargs.setdefault("num_proc", num_proc)
        return _LEROBOT_ORIGINAL_LOAD_DATASET(*args, **kwargs)

    _lerobot_ds_mod.load_dataset = load_dataset_with_num_proc
    _LEROBOT_HF_LOAD_PATCHED = True
    print(f"Using datasets.load_dataset num_proc={num_proc} for LeRobot parquet loading")


def _reset_save_substep_timings():
    for k in _SAVE_SUBSTEP_TIMINGS:
        _SAVE_SUBSTEP_TIMINGS[k] = 0.0


def _install_save_episode_timing():
    _orig_wait = LeRobotDataset._wait_image_writer
    _orig_save_table = LeRobotDataset._save_episode_table
    _orig_compute_stats = _lerobot_ds_mod.compute_episode_stats

    def _patched_wait(self):
        t0 = time.time()
        r = _orig_wait(self)
        _SAVE_SUBSTEP_TIMINGS["wait_image_writer"] += time.time() - t0
        return r

    def _patched_save_table(self, *a, **kw):
        t0 = time.time()
        r = _orig_save_table(self, *a, **kw)
        _SAVE_SUBSTEP_TIMINGS["save_episode_table"] += time.time() - t0
        return r

    from lerobot.common.datasets.compute_stats import get_feature_stats as _get_feature_stats

    def _patched_compute_stats(ep_buf, features):
        """Fast replacement: dummy stats for image/video keys, real stats for the rest.

        Image/video stats are not consumed by our downstream `compute_norm_stats.py`
        (fast path reads state/actions directly from parquet). Skipping the PNG-reading
        `sample_images` pass saves ~26s/episode at 626 frames × 3 cameras.
        """
        t0 = time.time()
        ep_stats = {}
        for key, val in ep_buf.items():
            ft_dtype = features[key]["dtype"]
            if ft_dtype == "string":
                continue
            if ft_dtype in ("image", "video"):
                dummy = np.zeros((3, 1, 1), dtype=np.float32)
                ep_stats[key] = {
                    "min": dummy.copy(),
                    "max": dummy.copy(),
                    "mean": dummy.copy(),
                    "std": dummy.copy(),
                    "count": np.array([len(val)]),
                }
            else:
                arr = val
                keepdims = arr.ndim == 1
                ep_stats[key] = _get_feature_stats(arr, axis=0, keepdims=keepdims)
        _SAVE_SUBSTEP_TIMINGS["compute_episode_stats"] += time.time() - t0
        return ep_stats

    # Fast PNG writing: cv2.imwrite releases the GIL (unlike PIL), so with many
    # writer threads we get true parallelism. compress_level=1 (~4x faster than
    # default 6, ~20%% larger files). Input images are expected to be RGB uint8
    # (H, W, 3) as produced by _save_frame; cv2 wants BGR.
    import lerobot.common.datasets.image_writer as _image_writer_mod
    import PIL.Image as _PILImage

    _PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]

    def _fast_write_image(image, fpath):
        try:
            if isinstance(image, _PILImage.Image):
                arr = np.asarray(image)
            elif isinstance(image, np.ndarray):
                arr = image
                if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
                    arr = arr.transpose(1, 2, 0)
                if arr.dtype != np.uint8:
                    arr = (arr * 255).astype(np.uint8)
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(fpath), arr, _PNG_PARAMS)
        except Exception as e:
            print(f"Error writing image {fpath}: {e}")

    _image_writer_mod.write_image = _fast_write_image

    LeRobotDataset._wait_image_writer = _patched_wait
    LeRobotDataset._save_episode_table = _patched_save_table
    _lerobot_ds_mod.compute_episode_stats = _patched_compute_stats


def _ensure_meta_patched(dataset: LeRobotDataset):
    global _META_PATCHED
    if _META_PATCHED:
        return
    meta_cls = type(dataset.meta)
    _orig_meta_save = meta_cls.save_episode

    def _patched_meta_save(self, *a, **kw):
        t0 = time.time()
        r = _orig_meta_save(self, *a, **kw)
        _SAVE_SUBSTEP_TIMINGS["meta_save_episode"] += time.time() - t0
        return r

    meta_cls.save_episode = _patched_meta_save
    _META_PATCHED = True


_install_save_episode_timing()


def cv2_resize_with_pad(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize *img* to (target_h, target_w) preserving aspect ratio, pad with black.

    Uses cv2 only (no PIL conversion), which is ~3-5x faster than
    ``image_tools.resize_with_pad`` for single images.
    """
    h, w = img.shape[:2]
    if h == target_h and w == target_w:
        return img
    ratio = max(w / target_w, h / target_h)
    new_w = int(w / ratio)
    new_h = int(h / ratio)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))

# ---------------------------------------------------------------------------
# Inline: config loader (replaces scripts/airbot_data_config.py)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TaskConfig:
    """Structured description of an AIRBOT mcap dataset."""
    task_name: str
    robot_type: str
    folders: list[str]
    state_topics: list[str]
    action_topics: list[str]
    camera_topics: dict[str, str]
    fps: int
    delta_action_mask: tuple[int, ...] = ()
    # Label parameters (used by add_returns_to_lerobot.py Mode 1)
    success_episodes: str | list[int] = "all"
    failed_episodes: tuple[int, ...] = ()
    all_human: bool = False
    stage_boundaries: tuple[int, ...] = ()


def load_task_config(config_path: str | Path) -> TaskConfig:
    """Load a ``config.py`` file and return a structured :class:`TaskConfig`."""
    config_path = Path(config_path)
    if not config_path.is_file():
        config_path = config_path / "config.py"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

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
        delta_action_mask=tuple(getattr(mod, "DELTA_ACTION_MASK", ())),
        success_episodes=getattr(mod, "SUCCESS_EPISODES", "all"),
        failed_episodes=tuple(getattr(mod, "FAILED_EPISODES", ())),
        all_human=getattr(mod, "ALL_HUMAN", False),
        stage_boundaries=tuple(getattr(mod, "STAGE_BOUNDARIES", ())),
    )


DAGGER_INTERVENTION_TOPIC = "/dagger/intervention"


# ---------------------------------------------------------------------------
# Inline: FlatBuffers float-array decoder (replaces openpi.airbot_type)
# ---------------------------------------------------------------------------

def decode_float_array(data: bytes) -> np.ndarray:
    """Decode an ``airbot_fbs.FloatArray`` FlatBuffers message to a numpy array."""
    root_offset = flatbuffers.packer.uoffset.unpack_from(data, 0)[0]
    tab = flatbuffers.table.Table(bytearray(data), root_offset)
    o = flatbuffers.number_types.UOffsetTFlags.py_type(tab.Offset(4))
    if o != 0:
        return tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
    return np.array([], dtype=np.float32)


class McapConverter:
    """
    A class to convert MCAP dataset to LeRobot format.
    """

    def __init__(self, data_dir: str):
        # 1. Validate data_dir
        if not Path(data_dir).is_dir():
            raise ValueError(f"data_dir '{data_dir}' is not a valid directory")
        self.data_dir = data_dir

        # 2. Locate the Python-format config file
        config_path = Path(self.data_dir) / "config.py"
        if not config_path.is_file():
            raise ValueError(f"Config file 'config.py' not found in '{data_dir}'.")

        # 3. Load the config file and initialize task parameters
        config = load_task_config(config_path)
        self.task_name = config.task_name
        self.robot_type = config.robot_type
        self.folders = config.folders
        self.state_topics = config.state_topics
        self.action_topics = config.action_topics
        self.camera_topics: dict[str:str] = config.camera_topics
        self.fps = config.fps
        self.target_height = 224
        self.target_width = 224

        # 4. Scan each folder, verify existence, and count .mcap files
        self.folder_file_counts = {}
        for folder in self.folders:
            folder_path = Path(self.data_dir) / folder
            if not folder_path.is_dir():
                raise ValueError(f"Configured folder '{folder}' not found in '{self.data_dir}'.")

            # Count MCAP files in this folder
            mcap_files = [
                f for f in os.listdir(folder_path) if Path(folder_path / f).is_file() and f.lower().endswith(".mcap")
            ]
            self.folder_file_counts[folder] = len(mcap_files)

        print(f"Found {sum(self.folder_file_counts.values())} MCAP files across {len(self.folders)} folders.")

        # 5 Read first mcap file for many infos
        self.schemas = {}
        self.schemas["airbot_fbs.FloatArray"] = decode_float_array
        self.state_length = 0
        self.action_length = 0
        self.camera_image_shape = {}

        # 6. Find the first mcap file to read schema and camera information
        mcap_file_path = None
        for folder in self.folders:
            folder_path = Path(self.data_dir) / folder
            for root, _, files in os.walk(folder_path):
                for filename in files:
                    if filename.lower().endswith(".mcap"):
                        mcap_file_path = Path(root) / filename
                        break
                if mcap_file_path:
                    break

        if mcap_file_path is None:
            raise ValueError("No MCAP files found in any of the configured folders.")

        print(f"Using {mcap_file_path} to read schema and camera information.")

        with mcap_file_path.open("rb") as f:
            reader = make_reader(f)

            # Read camera image shape from the valid mcap file
            cam_attachment_path = {}
            for attach in reader.iter_attachments():
                media_type = attach.media_type
                if media_type == "video/mp4" and attach.name in self.camera_topics.values():
                    cam_attachment_path[attach.name] = self._save_temporary_video(attach.data)

            for camera_name, topic in self.camera_topics.items():
                if topic in cam_attachment_path:
                    frame = self._get_frame_image(cam_attachment_path[topic], 0)
                    self.camera_image_shape[camera_name] = frame.shape
                else:
                    raise ValueError(f"Camera attachment for {camera_name} not found in {mcap_file_path}.")

            print(f"Camera image shapes: {self.camera_image_shape}")

            is_read_topics = {topic: False for topic in self.state_topics + self.action_topics}
            for schema_obj, channel_obj, message_obj in reader.iter_messages(
                topics=self.state_topics + self.action_topics
            ):
                if is_read_topics[channel_obj.topic]:
                    break
                is_read_topics[channel_obj.topic] = True
                if schema_obj.name not in self.schemas:
                    raise ValueError(f"Schema '{schema_obj.name}' not found in schemas.")
                if channel_obj.topic in self.state_topics:
                    self.state_length += len(self.schemas[schema_obj.name](message_obj.data))
                if channel_obj.topic in self.action_topics:
                    self.action_length += len(self.schemas[schema_obj.name](message_obj.data))

        print(f"State length: {self.state_length}, Action length: {self.action_length}")

    def create_dataset(self, repo_id: str | None = None) -> LeRobotDataset:
        """
        Placeholder for dataset creation logic.
        This method should implement the logic to convert MCAP files to LeRobot format.

        ``repo_id`` overrides the output directory name (defaults to ``self.task_name``).
        The per-frame ``task`` label still uses ``self.task_name``.
        """
        features = {}
        for camera_name in self.camera_topics:
            features[camera_name] = {
                "dtype": "image",
                "shape": (self.target_height, self.target_width, 3),
                "names": ["height", "width", "channel"],
            }
        features["state"] = {
            "dtype": "float32",
            "shape": (self.state_length,),
            "names": ["state"],
        }
        features["actions"] = {
            "dtype": "float32",
            "shape": (self.action_length,),
            "names": ["actions"],
        }
        features["intervention"] = {
            "dtype": "int64",
            "shape": (1,),
            "names": ["intervention"],
        }
        print(f"Creating LeRobot dataset with features: {features}")

        return LeRobotDataset.create(
            repo_id=repo_id or self.task_name,
            robot_type=self.robot_type,
            fps=self.fps,
            features=features,
            image_writer_threads=3,
            image_writer_processes=4,  # true parallel PNG encoding via subprocesses (bypass GIL)
        )

    def _save_temporary_video(self, bytes_data: bytes) -> str:
        """
        Save a temporary video file from the MCAP attachment.
        This method is used to handle camera attachments in the MCAP files.
        """
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_file.write(bytes_data)
            tmp_file.flush()
        return tmp_file.name

    @staticmethod
    def _get_frame_image(path: str, frame_index: int) -> np.ndarray:
        """
        Get a specific frame image from a video file.
        This method is used to extract images from camera attachments in the MCAP files.
        """
        cap = cv2.VideoCapture(path)
        frame = None
        for i in range(frame_index + 1):
            ret, new_frame = cap.read()
            if not ret:
                if i <= frame_index:
                    print(
                        f"Warning: Requested frame_index {frame_index} not found in video {path} until {i}. Using last frame instead."
                    )
                break
            frame = new_frame
        cap.release()
        if frame is None:
            raise ValueError(f"Failed to read frame {frame_index} from video {path}.")
        return frame

    def _collect_mcap_files(self) -> list:
        """Collect all .mcap paths in natural (numeric-aware) sorted order.

        Natural sort prevents lexicographic pitfalls like ``dagger_1100_1110``
        sorting before ``dagger_200_210`` or ``10.mcap`` before ``2.mcap``,
        which would misalign ``--skip_episodes`` when new folders/files are
        added.
        """
        def natural_key(s: str):
            return [int(tok) if tok.isdigit() else tok.lower()
                    for tok in re.split(r"(\d+)", s)]

        mcap_files = []
        for folder_name in sorted(self.folder_file_counts, key=natural_key):
            for root, _, files in os.walk(Path(self.data_dir) / folder_name):
                for filename in sorted(files, key=natural_key):
                    if filename.lower().endswith(".mcap"):
                        mcap_files.append(Path(root) / filename)
        return mcap_files

    def _finalize_dataset_episode(
        self,
        dataset: LeRobotDataset,
        frame_count: int,
        min_frames: int = 1,
    ):
        """Persist or drop the current buffered episode depending on its length."""
        if frame_count <= 0:
            return "empty", 0.0, {
                "wait_image_writer": 0.0,
                "save_episode_table": 0.0,
                "compute_episode_stats": 0.0,
                "meta_save_episode": 0.0,
                "other": 0.0,
            }

        if frame_count < max(1, min_frames):
            dataset.clear_episode_buffer()
            return "dropped", 0.0, {
                "wait_image_writer": 0.0,
                "save_episode_table": 0.0,
                "compute_episode_stats": 0.0,
                "meta_save_episode": 0.0,
                "other": 0.0,
            }

        _ensure_meta_patched(dataset)
        _reset_save_substep_timings()
        t0 = time.time()
        dataset.save_episode()
        t_save_episode = time.time() - t0
        timings = {
            "wait_image_writer": _SAVE_SUBSTEP_TIMINGS["wait_image_writer"],
            "save_episode_table": _SAVE_SUBSTEP_TIMINGS["save_episode_table"],
            "compute_episode_stats": _SAVE_SUBSTEP_TIMINGS["compute_episode_stats"],
            "meta_save_episode": _SAVE_SUBSTEP_TIMINGS["meta_save_episode"],
        }
        timings["other"] = max(
            0.0,
            t_save_episode
            - timings["wait_image_writer"]
            - timings["save_episode_table"]
            - timings["compute_episode_stats"]
            - timings["meta_save_episode"],
        )
        return "saved", t_save_episode, timings

    def load_to_dataset(
        self,
        dataset: LeRobotDataset,
        skip_episodes: int = 0,
        intervention_only: bool = False,
        min_intervention_frames: int = 0,
    ):
        """Load MCAP files into the LeRobot dataset.

        Each mcap file is opened **once** (single-pass): metadata validation,
        DAgger intervention extraction, video attachment extraction, and
        state/action message processing all happen in one file open.
        """
        all_mcap_files = self._collect_mcap_files()
        total_files = len(all_mcap_files)
        if skip_episodes > 0:
            print(f"Resuming: skipping first {skip_episodes} episodes.")
            all_mcap_files = all_mcap_files[skip_episodes:]

        for file_idx, mcap_file_path in enumerate(all_mcap_files, start=1):
            folder_name = mcap_file_path.relative_to(self.data_dir).parts[0]
            filename = mcap_file_path.name
            mode_desc = "intervention-only" if intervention_only else "full-episode"
            print(f"[{file_idx}/{len(all_mcap_files)}] Converting {folder_name}/{filename} ({mode_desc}) ...")

            caps: dict = {}
            last_frames: dict = {}
            cam_attachment_path: dict = {}

            t_episode_start = time.time()

            try:
                with mcap_file_path.open("rb") as f:
                    reader = make_reader(f)

                    # --- Metadata validation ---
                    t0 = time.time()
                    summary = reader.get_summary()
                    for _md in reader.iter_metadata():
                        if _md.name == "task_info":
                            task_name = _md.metadata.get("task_name", "N/A")
                            task_name_clean = task_name.strip('"').strip("'")
                            if task_name_clean != self.task_name:
                                print(f"Task name mismatch: expected {self.task_name}, got {task_name} (cleaned: {task_name_clean})")
                    t_metadata = time.time() - t0

                    # --- Check if DAgger intervention topic exists ---
                    has_dagger = summary is not None and any(
                        ch.topic == DAGGER_INTERVENTION_TOPIC
                        for ch in (summary.channels or {}).values()
                    )

                    # --- Extract video attachments ---
                    t0 = time.time()
                    for attach in reader.iter_attachments():
                        if attach.media_type == "video/mp4" and attach.name in self.camera_topics.values():
                            cam_attachment_path[attach.name] = self._save_temporary_video(attach.data)
                    t_attachments = time.time() - t0
                    if len(cam_attachment_path) != len(self.camera_topics):
                        print(
                            f"Warning: Not all camera attachments found in {mcap_file_path}. "
                            f"Expected {len(self.camera_topics)}, found {len(cam_attachment_path)}. Skipping."
                        )
                        continue

                    # Initialize VideoCaptures once
                    t0 = time.time()
                    for topic, path in cam_attachment_path.items():
                        caps[topic] = cv2.VideoCapture(path)
                        last_frames[topic] = None
                    t_videocap_init = time.time() - t0

                    # --- Single iter_messages pass: state + action + dagger ---
                    msg_topics = list(self.state_topics + self.action_topics)
                    if has_dagger:
                        msg_topics.append(DAGGER_INTERVENTION_TOPIC)

                    cnt_topics = {t: 0 for t in self.state_topics + self.action_topics}
                    state_msg: dict = {}
                    action_msg: dict = {}
                    intervention_values: list[int] = []
                    cnt = 0
                    source_frame_count = 0
                    saved_episode_count = 0
                    dropped_episode_count = 0
                    buffered_frame_count = 0

                    # Accumulators for _save_frame sub-timings
                    t_save_frame_total = 0.0
                    t_sf_video_read = 0.0
                    t_sf_resize = 0.0
                    t_sf_add_frame = 0.0
                    t_save_episode = 0.0
                    _se_wait = 0.0
                    _se_table = 0.0
                    _se_stats = 0.0
                    _se_meta = 0.0
                    _se_other = 0.0

                    t0 = time.time()
                    for schema_obj, channel_obj, message_obj in reader.iter_messages(topics=msg_topics):
                        # Collect DAgger intervention values inline
                        if channel_obj.topic == DAGGER_INTERVENTION_TOPIC:
                            decoded = decode_float_array(message_obj.data)
                            intervention_values.append(int(round(decoded[0])) if len(decoded) > 0 else 0)
                            continue

                        cnt_topics[channel_obj.topic] += 1
                        if cnt_topics[channel_obj.topic] - cnt == 2:
                            _intv = intervention_values[cnt] if cnt < len(intervention_values) else 0
                            source_frame_count += 1
                            if intervention_only:
                                if _intv == 1:
                                    sf_times = self._save_frame(
                                        dataset, state_msg, action_msg, caps, last_frames, cnt, _intv
                                    )
                                    if sf_times:
                                        t_save_frame_total += sf_times["total"]
                                        t_sf_video_read += sf_times["video_read"]
                                        t_sf_resize += sf_times["resize"]
                                        t_sf_add_frame += sf_times["add_frame"]
                                        buffered_frame_count += 1
                                else:
                                    status, save_time, save_timings = self._finalize_dataset_episode(
                                        dataset,
                                        buffered_frame_count,
                                        min_frames=min_intervention_frames,
                                    )
                                    if status == "saved":
                                        saved_episode_count += 1
                                        buffered_frame_count = 0
                                        t_save_episode += save_time
                                        _se_wait += save_timings["wait_image_writer"]
                                        _se_table += save_timings["save_episode_table"]
                                        _se_stats += save_timings["compute_episode_stats"]
                                        _se_meta += save_timings["meta_save_episode"]
                                        _se_other += save_timings["other"]
                                    elif status == "dropped":
                                        dropped_episode_count += 1
                                        buffered_frame_count = 0
                            else:
                                sf_times = self._save_frame(
                                    dataset, state_msg, action_msg, caps, last_frames, cnt, _intv
                                )
                                if sf_times:
                                    t_save_frame_total += sf_times["total"]
                                    t_sf_video_read += sf_times["video_read"]
                                    t_sf_resize += sf_times["resize"]
                                    t_sf_add_frame += sf_times["add_frame"]
                                    buffered_frame_count += 1
                            cnt += 1
                        if channel_obj.topic in self.state_topics:
                            state_msg[channel_obj.topic] = self.schemas[schema_obj.name](message_obj.data)
                        if channel_obj.topic in self.action_topics:
                            action_msg[channel_obj.topic] = self.schemas[schema_obj.name](message_obj.data)
                    t_iter_messages = time.time() - t0

                    # save last frame
                    _intv = intervention_values[cnt] if cnt < len(intervention_values) else 0
                    source_frame_count += 1
                    if intervention_only:
                        if _intv == 1:
                            sf_times = self._save_frame(dataset, state_msg, action_msg, caps, last_frames, cnt, _intv)
                            if sf_times:
                                t_save_frame_total += sf_times["total"]
                                t_sf_video_read += sf_times["video_read"]
                                t_sf_resize += sf_times["resize"]
                                t_sf_add_frame += sf_times["add_frame"]
                                buffered_frame_count += 1
                    else:
                        sf_times = self._save_frame(dataset, state_msg, action_msg, caps, last_frames, cnt, _intv)
                        if sf_times:
                            t_save_frame_total += sf_times["total"]
                            t_sf_video_read += sf_times["video_read"]
                            t_sf_resize += sf_times["resize"]
                            t_sf_add_frame += sf_times["add_frame"]
                            buffered_frame_count += 1

                    min_frames = min_intervention_frames if intervention_only else 1
                    status, save_time, save_timings = self._finalize_dataset_episode(
                        dataset,
                        buffered_frame_count,
                        min_frames=min_frames,
                    )
                    if status == "saved":
                        saved_episode_count += 1
                        buffered_frame_count = 0
                        t_save_episode += save_time
                        _se_wait += save_timings["wait_image_writer"]
                        _se_table += save_timings["save_episode_table"]
                        _se_stats += save_timings["compute_episode_stats"]
                        _se_meta += save_timings["meta_save_episode"]
                        _se_other += save_timings["other"]
                    elif status == "dropped":
                        dropped_episode_count += 1
                        buffered_frame_count = 0

            except Exception as e:
                print(f"Error processing {folder_name}/{filename}: {e}. Skipping.")
                if dataset.episode_buffer is not None:
                    dataset.clear_episode_buffer()
                continue
            finally:
                t0 = time.time()
                for cap in caps.values():
                    cap.release()
                for temp_path in cam_attachment_path.values():
                    try:
                        Path(temp_path).unlink(missing_ok=True)
                    except Exception as e:
                        print(f"Warning: Could not delete temporary file {temp_path}: {e}")
                t_cleanup = time.time() - t0

            t_episode_total = time.time() - t_episode_start
            t_msg_decode = t_iter_messages - t_save_frame_total
            drop_summary = (
                f", {dropped_episode_count} short segment(s) dropped"
                if intervention_only and min_intervention_frames > 1
                else ""
            )
            print(
                f"  [TIMING] Source episode {file_idx} "
                f"({source_frame_count} frames -> {saved_episode_count} saved episode(s){drop_summary}, {t_episode_total:.1f}s total):\n"
                f"    metadata:        {t_metadata:.2f}s\n"
                f"    attachments:     {t_attachments:.2f}s\n"
                f"    videocap_init:   {t_videocap_init:.2f}s\n"
                f"    iter_messages:   {t_iter_messages:.2f}s  (msg_decode: {t_msg_decode:.2f}s + save_frame: {t_save_frame_total:.2f}s)\n"
                f"      video_read:    {t_sf_video_read:.2f}s\n"
                f"      resize:        {t_sf_resize:.2f}s\n"
                f"      add_frame:     {t_sf_add_frame:.2f}s\n"
                f"    save_episode:    {t_save_episode:.2f}s\n"
                f"      wait_writer:   {_se_wait:.2f}s\n"
                f"      save_table:    {_se_table:.2f}s (embed_images)\n"
                f"      compute_stats: {_se_stats:.2f}s\n"
                f"      meta_save:     {_se_meta:.2f}s\n"
                f"      other:         {_se_other:.2f}s\n"
                f"    cleanup:         {t_cleanup:.2f}s"
            )

    def _save_frame(
        self,
        dataset: LeRobotDataset,
        state_msg: dict,
        action_msg: dict,
        caps: dict,
        last_frames: dict,
        cnt: int,
        intervention: int = 0,
    ):
        """
        Save a frame to the dataset.
        This method is used to handle the saving of frames after processing all messages in an episode.
        Returns a timing dict with sub-step durations, or None if frame was skipped.
        """
        _t_sf_start = time.time()

        # Check if all required state and action topics have messages
        for topic in self.state_topics:
            if topic not in state_msg:
                print(f"Warning: State topic {topic} not found in messages. Skipping frame {cnt}.")
                return None
        for topic in self.action_topics:
            if topic not in action_msg:
                print(f"Warning: Action topic {topic} not found in messages. Skipping frame {cnt}.")
                return None

        state_vec = np.array([])
        action_vec = np.array([])
        for topic in self.state_topics:
            state_vec = np.concatenate((state_vec, state_msg[topic]))
        for topic in self.action_topics:
            action_vec = np.concatenate((action_vec, action_msg[topic]))
        frame = {}
        _t_vr_total = 0.0
        _t_rs_total = 0.0
        for camera_name, topic in self.camera_topics.items():
            if topic in caps:
                # Optimized: Read the next frame directly from the open capture
                _t_vr0 = time.time()
                ret, img = caps[topic].read()
                if ret:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    last_frames[topic] = img
                elif last_frames[topic] is not None:
                    img = last_frames[topic]
                else:
                    print(f"Warning: Failed to read *any* frames for {camera_name}. Using black frame.")
                    img = np.zeros(self.camera_image_shape[camera_name], dtype=np.uint8)
                _t_vr_total += time.time() - _t_vr0

                _t_rs0 = time.time()
                img = cv2_resize_with_pad(img, self.target_height, self.target_width)
                _t_rs_total += time.time() - _t_rs0
                frame[camera_name] = img
            else:
                print(f"Warning: Camera attachment for {camera_name} not found.")

        frame["state"] = state_vec.astype(np.float32)
        frame["actions"] = action_vec.astype(np.float32)
        frame["intervention"] = np.array([intervention], dtype=np.int64)
        frame["task"] = self.task_name

        _INTERNAL_KEYS = {"task", "timestamp", "frame_index", "index", "episode_index", "task_index"}
        for key, feat in dataset.features.items():
            if key in frame or key in _INTERNAL_KEYS or feat.get("dtype") == "image":
                continue
            shape = tuple(feat["shape"])
            dtype_str = feat.get("dtype", "float32")
            dtype_map = {
                "float32": np.float32, "float64": np.float64,
                "int32": np.int32, "int64": np.int64,
                "bool": np.bool_, "uint8": np.uint8,
            }
            dtype = dtype_map.get(dtype_str, np.float32)
            frame[key] = np.zeros(shape, dtype=dtype)

        _t_af0 = time.time()
        dataset.add_frame(frame)
        _t_af = time.time() - _t_af0

        return {
            "total": time.time() - _t_sf_start,
            "video_read": _t_vr_total,
            "resize": _t_rs_total,
            "add_frame": _t_af,
        }


def main(
    data_dir: str,
    repo_id: str | None = None,
    resume: bool = False,
    force: bool = False,
    skip_episodes: int = -1,
    intervention_only: bool = False,
    min_intervention_frames: int = 0,
):
    """Convert AIRBOT MCAP dataset to LeRobot format.

    Args:
        data_dir: Path to directory containing ``config.py`` and MCAP folders.
        repo_id: Override the output directory name under ``HF_LEROBOT_HOME``. Defaults
            to ``TASK_NAME`` from config.py. Use this to give two data dirs that share the
            same ``TASK_NAME`` their own separate output datasets (e.g. when converting
            several dirs in parallel). The per-frame ``task`` label still uses ``TASK_NAME``.
        resume: Append new episodes to an existing dataset. When resuming within the
            same data directory, the already-converted episodes are skipped automatically.
            When appending from a *new* data directory into an existing dataset (e.g.
            adding dagger data to a dataset built from the base mcap folder), pass
            ``--skip_episodes 0`` so no files from the new directory are skipped.
        force: Remove existing output directory without prompting.
        skip_episodes: Override how many episodes (mcap files) to skip at the start of
            this data directory. Default (-1) auto-detects from the existing dataset when
            ``--resume`` is set, which is correct when re-running on the *same* source
            directory. Set to 0 (or another value) when appending from a different source
            directory that has not yet contributed any episodes to the dataset.
        intervention_only: Only keep contiguous DAgger ``intervention=1`` segments.
            Each such segment is saved as its own LeRobot episode. Frames with
            ``intervention=0`` are dropped entirely.
        min_intervention_frames: Minimum length for an intervention segment to be kept.
            Only applies when ``intervention_only=True``. Shorter segments are dropped.
    """
    _patch_lerobot_hf_load_dataset_num_proc()

    mcap_converter = McapConverter(data_dir)
    out_name = repo_id or mcap_converter.task_name
    output_path = HF_LEROBOT_HOME / out_name
    print(f"Output path: {output_path}")

    n_skip = 0
    dataset: LeRobotDataset

    if output_path.exists():
        if resume:
            try:
                dataset = LeRobotDataset(repo_id=out_name)
                dataset.start_image_writer(num_processes=4, num_threads=3)
                if skip_episodes >= 0:
                    n_skip = skip_episodes
                    print(f"Resuming: appending to existing dataset ({dataset.num_episodes} episodes), skipping first {n_skip} episodes from this data dir.")
                else:
                    n_skip = dataset.num_episodes
                    print(f"Resuming: found {n_skip} existing episodes.")
            except Exception as e:
                raise SystemExit(
                    f"ERROR: Could not load existing dataset for resume: {e}\n"
                    "The dataset may be corrupted. To start fresh, use --force (WARNING: deletes all data)."
                )
        elif force:
            shutil.rmtree(output_path)
            dataset = mcap_converter.create_dataset(repo_id=out_name)
        else:
            raise SystemExit(
                f"Output directory already exists: {output_path}\n"
                "Use --force to overwrite or --resume to continue."
            )
    else:
        dataset = mcap_converter.create_dataset(repo_id=out_name)

    mcap_converter.load_to_dataset(
        dataset,
        skip_episodes=n_skip,
        intervention_only=intervention_only,
        min_intervention_frames=min_intervention_frames,
    )


if __name__ == "__main__":
    tyro.cli(main)
