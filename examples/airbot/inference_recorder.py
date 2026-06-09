"""Inference data recorder module.

Records observation and action data during inference into MCAP files,
using the same format as the AIRBOT-Data-Collection pipeline.
This allows recorded data to be directly converted to LeRobot format
for re-training using the existing conversion scripts.
"""

import logging
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel

from airdc.common.samplers.mcap_sampler import McapDataSampler, McapDataSamplerConfig
from airdc.common.samplers.basis import SaveType, TaskInfo

logger = logging.getLogger(__name__)


class RecordConfig(BaseModel):
    """Configuration for inference data recording.

    Args:
        record_data: Whether to enable data recording during inference.
        save_dir: Directory to save recorded MCAP files.
        task_name: Task name stored in MCAP metadata. Auto-filled from prompt if empty.
        save_video: Video saving format for color images: "h264", "jpeg", or "raw".
    """

    record_data: bool = True
    save_dir: str = "./inference_data"  # default directory, can be changed
    task_name: str = ""
    save_video: str = "h264"


class InferenceRecorder:
    """Records inference data (observations + actions) to MCAP files.

    Uses the same McapDataSampler as the data collection pipeline to ensure
    format compatibility. Each inference episode is saved as a separate MCAP file.

    The recorded data includes:
    - Joint states from all robot components (as both follow and lead topics)
    - Camera images (encoded as H264 video by default)
    - Model-predicted actions (stored as action topics)
    - Timestamps for each step
    """

    def __init__(self, config: RecordConfig, camera_names: list[str]):
        """Initialize the inference recorder.

        Args:
            config: Recording configuration.
            camera_names: List of camera names used by the robot operator.
        """
        self._config = config
        self._camera_names = camera_names
        self._sampler: Optional[McapDataSampler] = None
        self._round: int = 0
        self._step_count: int = 0
        self._recording: bool = False
        self._round_data: dict = defaultdict(list)

        # Async encoding infrastructure
        self._queue: Optional[queue.Queue] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_worker = threading.Event()

        if not config.record_data:
            logger.info("Data recording is disabled.")
            return

        # Create save directory
        self._save_dir = Path(config.save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)

        # Determine starting round number from existing files
        existing_files = list(self._save_dir.glob("*.mcap"))
        self._round = len(existing_files)

        # Initialize McapDataSampler
        sampler_config = McapDataSamplerConfig(
            task_info=TaskInfo(task_name=config.task_name),
            save_type=SaveType(color=config.save_video),
        )
        self._sampler = McapDataSampler(sampler_config)
        if not self._sampler.configure():
            logger.error("Failed to configure McapDataSampler.")
            self._sampler = None
            return

        # Set basic info for the sampler
        self._sampler.set_info({"recorder": "inference_recorder"})

        # Start async encoding worker
        self._queue = queue.Queue(maxsize=100)
        self._worker_thread = threading.Thread(target=self._encoding_worker, daemon=True)
        self._worker_thread.start()
        logger.info(
            f"Inference recorder initialized. Save dir: {self._save_dir}, "
            f"starting round: {self._round}, format: {config.save_video}"
        )

    @property
    def enabled(self) -> bool:
        """Whether data recording is enabled and configured."""
        return self._config.record_data and self._sampler is not None

    def _encoding_worker(self):
        """Background worker that processes queued recording tasks."""
        while not self._stop_worker.is_set():
            try:
                item = self._queue.get(timeout=0.1)
                if item is None:  # Sentinel to stop
                    break
                raw_obs, action, intervention = item
                self._process_step(raw_obs, action, intervention)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Encoding worker error: {e}")

    def _copy_obs(self, raw_obs: dict) -> dict:
        """Deep copy observation data for safe queuing."""
        copied = {}
        for key, value in raw_obs.items():
            if isinstance(value, dict) and "data" in value:
                copied[key] = {"data": np.array(value["data"]).copy(), "t": value.get("t")}
            elif isinstance(value, np.ndarray):
                copied[key] = value.copy()
            else:
                copied[key] = value
        return copied

    def start_episode(self) -> None:
        """Start recording a new episode.

        Creates a new MCAP file writer for the episode.
        """
        if not self.enabled:
            return

        try:
            self._save_path = self._sampler.compose_path(self._save_dir, self._round)
            self._step_count = 0
            self._round_data = defaultdict(list)
            self._recording = True
            logger.info(f"Started recording episode {self._round} -> {self._save_path}")
        except Exception as e:
            logger.error(f"Failed to start episode recording: {e}")
            self._recording = False

    def record_step(self, raw_obs: dict, action: np.ndarray, intervention: int = 0) -> None:
        """Record a single inference step.

        Captures the raw observation data and the predicted action,
        formats them to match the data collection MCAP schema, and
        writes them to the current MCAP file.

        Args:
            raw_obs: Raw observation dict from operator.capture_observation().
                     Contains keys like "left/arm/joint_state/position",
                     "base_0_rgb/color/image_raw", etc.
            action: The action array that was sent to the robot in this step.
            intervention: 0 = autonomous policy action, 1 = human intervention.
        """
        if not self._recording:
            return

        copied_obs = self._copy_obs(raw_obs)
        copied_action = np.array(action).copy()
        self._queue.put((copied_obs, copied_action, intervention))

    def _process_step(self, raw_obs: dict, action: np.ndarray, intervention: int) -> None:
        """Core step processing logic (encoding happens here)."""
        try:
            log_stamp = time.time_ns()

            # Build the data dict matching data collection format
            data = {}

            # Copy all observation data with "/" prefix to match MCAP topic format
            for key, value in raw_obs.items():
                topic_key = f"/{key}" if not key.startswith("/") else key
                data[topic_key] = value

            # Add action data as "lead" topics (mirroring follow joint states)
            # During inference, the model's predicted action serves as the leader
            self._add_action_as_lead(data, raw_obs, action)

            # Add DAgger intervention flag
            data["/dagger/intervention"] = {
                "data": [intervention],
                "t": log_stamp,
            }

            # Add timestamp
            data["log_stamps"] = log_stamp

            # Use sampler to process and encode the data (THIS IS THE SLOW PART)
            processed = self._sampler.update(data)

            # Accumulate remaining data for batch save
            for key, value in processed.items():
                self._round_data[key].append(value)

            self._step_count += 1

        except Exception as e:
            logger.warning(f"Failed to process step {self._step_count}: {e}")

    def _add_action_as_lead(
        self, data: dict, raw_obs: dict, action: np.ndarray
    ) -> None:
        """Add the predicted action as leader joint state topics.

        Maps the action array to leader topics that mirror the follower
        observation topics, maintaining compatibility with the data
        collection MCAP format.

        For play robots:
            action = [left_arm(6), left_eef(1), right_arm(6), right_eef(1)]
            Maps to /left/lead/arm/..., /left/lead/eef/...,
                     /right/lead/arm/..., /right/lead/eef/...

        For single arm:
            action = [arm(6), eef(1)]
            Maps to /lead/arm/..., /lead/eef/...

        Args:
            data: The data dict being built for this step.
            raw_obs: Raw observation to determine robot topology.
            action: Model-predicted action array.
        """
        stamp = time.time_ns()

        # Detect robot groups from observation keys
        obs_keys = list(raw_obs.keys())

        # Check for dual-arm play robot (left/right groups)
        has_left = any(k.startswith("left/") for k in obs_keys)
        has_right = any(k.startswith("right/") for k in obs_keys)

        if has_left and has_right:
            # Dual-arm play robot: action = [left_arm(6), left_eef(1), right_arm(6), right_eef(1)]
            action_flat = np.array(action).flatten()
            left_arm_action = action_flat[:6].tolist()
            left_eef_action = action_flat[6:7].tolist()
            right_arm_action = action_flat[7:13].tolist()
            right_eef_action = action_flat[13:14].tolist()

            data["/left/lead/arm/joint_state/position"] = {
                "data": left_arm_action,
                "t": stamp,
            }
            data["/left/lead/eef/joint_state/position"] = {
                "data": left_eef_action,
                "t": stamp,
            }
            data["/right/lead/arm/joint_state/position"] = {
                "data": right_arm_action,
                "t": stamp,
            }
            data["/right/lead/eef/joint_state/position"] = {
                "data": right_eef_action,
                "t": stamp,
            }
        else:
            # Single-arm or other robots: store action as a single topic
            action_flat = np.array(action).flatten()
            if len(action_flat) >= 7:
                arm_action = action_flat[:6].tolist()
                eef_action = action_flat[6:7].tolist()
                data["/lead/arm/joint_state/position"] = {
                    "data": arm_action,
                    "t": stamp,
                }
                data["/lead/eef/joint_state/position"] = {
                    "data": eef_action,
                    "t": stamp,
                }
            else:
                data["/lead/action"] = {
                    "data": action_flat.tolist(),
                    "t": stamp,
                }

    def _drain_queue(self):
        """Wait for all queued items to be processed."""
        if self._queue is not None:
            self._queue.join()

    def discard_episode(self) -> None:
        """Discard the current episode without saving."""
        if not self._recording:
            return
        self._drain_queue()  # Wait for pending encoding to finish
        self._recording = False
        self._round_data = defaultdict(list)
        logger.info(f"Discarded episode {self._round} ({self._step_count} steps, not saved)")

    def save_episode(self, dagger_stats: dict = None) -> Optional[Path]:
        """Save the current episode's recorded data to MCAP file.

        Args:
            dagger_stats: Optional DAgger intervention statistics to log.

        Returns:
            Path to the saved MCAP file, or None if save failed.
        """
        if not self._recording:
            return None

        try:
            self._drain_queue()  # Wait for all pending encoding to finish
            self._recording = False
            result_path = self._sampler.save(self._save_path, self._round_data)
            self._round += 1
            self._round_data = defaultdict(list)

            stats_msg = ""
            if dagger_stats:
                stats_msg = (
                    f", interventions: {dagger_stats.get('total_interventions', 0)}, "
                    f"ratio: {dagger_stats.get('intervention_ratio', 0):.1%}"
                )
            logger.info(
                f"Saved episode to {result_path} ({self._step_count} steps{stats_msg})"
            )
            return Path(result_path)
        except Exception as e:
            logger.error(f"Failed to save episode: {e}")
            return None

    def shutdown(self) -> None:
        """Shutdown the recorder and release resources."""
        if self._recording:
            logger.info("Saving in-progress episode before shutdown...")
            self.save_episode()

        # Stop worker thread
        if self._worker_thread is not None:
            self._stop_worker.set()
            if self._queue is not None:
                self._queue.put(None)  # Sentinel
            self._worker_thread.join(timeout=2.0)

        if self._sampler is not None:
            self._sampler.shutdown()
            logger.info("Inference recorder shut down.")
