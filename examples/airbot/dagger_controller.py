"""DAgger controller module for π0.6*-style human intervention during inference.

Implements the DAgger (Dataset Aggregation) mechanism:
  - Press 'i' to pause inference and enter human demonstration mode
  - Leader arms smoothly align to follower arms via cosine interpolation
  - Human expert demonstrates recovery via leader arm teleoperation
  - Press 'o' to resume policy inference

Reference: kai0 (OpenDriveLab) DAgger implementation + π0.6* paper.
"""

import logging
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DaggerMode(Enum):
    """DAgger state machine modes."""
    INFERENCE = "inference"
    ALIGNING = "aligning"
    DEMONSTRATING = "demonstrating"
    RESUMING = "resuming"


class DaggerConfig(BaseModel):
    """Configuration for DAgger behavior.

    Args:
        enable: Whether to enable DAgger mode.
        key_enter_dagger: Key to enter DAgger (human intervention) mode.
        key_resume_inference: Key to resume policy inference.
        align_steps: Number of interpolation steps for leader-to-follower alignment.
        align_duration: Total duration (seconds) for the alignment trajectory.
        gripper_threshold: Threshold for gripper binarization (above = open).
        demo_motion_threshold: Per-dimension delta threshold below which a demo
            frame is considered stationary and skipped (not recorded).
    """
    enable: bool = True
    key_enter_dagger: str = "i"
    key_resume_inference: str = "o"
    align_steps: int = 50
    align_duration: float = 1.0
    gripper_threshold: float = 0.5
    demo_motion_threshold: float = 0.01


@dataclass
class DaggerStats:
    """Statistics for DAgger interventions within an episode."""
    total_interventions: int = 0
    intervention_steps: int = 0
    autonomous_steps: int = 0
    segments: list = field(default_factory=list)
    _current_segment_start: int = field(default=0, repr=False)
    _current_segment_type: str = field(default="policy", repr=False)
    _current_demo_recorded: int = field(default=0, repr=False)
    _current_demo_skipped: int = field(default=0, repr=False)

    def start_intervention(self, step: int):
        self._close_segment(step)
        self._current_segment_type = "intervention"
        self._current_segment_start = step
        self.total_interventions += 1
        self._current_demo_recorded = 0
        self._current_demo_skipped = 0

    def end_intervention(self, step: int):
        self._close_segment(step)
        self._current_segment_type = "policy"
        self._current_segment_start = step

    def _close_segment(self, step: int):
        if step > self._current_segment_start:
            seg = {
                "type": self._current_segment_type,
                "start": self._current_segment_start,
                "end": step - 1,
            }
            if self._current_segment_type == "intervention":
                seg["demo_recorded"] = self._current_demo_recorded
                seg["demo_skipped"] = self._current_demo_skipped
            self.segments.append(seg)

    def to_dict(self) -> dict:
        total = self.intervention_steps + self.autonomous_steps
        return {
            "total_interventions": self.total_interventions,
            "intervention_steps": self.intervention_steps,
            "autonomous_steps": self.autonomous_steps,
            "intervention_ratio": self.intervention_steps / max(1, total),
            "segments": self.segments,
        }

    def reset(self):
        self.total_interventions = 0
        self.intervention_steps = 0
        self.autonomous_steps = 0
        self.segments.clear()
        self._current_segment_start = 0
        self._current_segment_type = "policy"
        self._current_demo_recorded = 0
        self._current_demo_skipped = 0


class DaggerController:
    """Core DAgger controller managing mode switching, arm alignment, and statistics.

    Thread model:
      - Keyboard thread (daemon): captures single keystrokes without Enter
      - Main thread: checks mode and executes corresponding logic
      - Inference is paused/resumed via threading.Event
    """

    def __init__(self, config: DaggerConfig):
        self.config = config

        # Thread-safe mode state
        self._mode = DaggerMode.INFERENCE
        self._mode_lock = threading.Lock()
        self._pause_event = threading.Event()  # Set = inference should pause

        # Statistics
        self.stats = DaggerStats()
        self._step_counter = 0

        # Lifecycle
        self._shutdown = threading.Event()
        self._reset_requested = threading.Event()
        self._discard_requested = threading.Event()
        self._start_event = threading.Event()
        self._homing_cancel = threading.Event()  # Set to cancel in-progress leader homing
        self._keyboard_thread: Optional[threading.Thread] = None

    # ── Properties ─────────────────────────────────────────

    @property
    def mode(self) -> DaggerMode:
        with self._mode_lock:
            return self._mode

    @mode.setter
    def mode(self, value: DaggerMode):
        with self._mode_lock:
            self._mode = value

    @property
    def is_intervention(self) -> bool:
        return self.mode in (DaggerMode.DEMONSTRATING, DaggerMode.ALIGNING)

    @property
    def inference_paused(self) -> bool:
        return self._pause_event.is_set()

    @property
    def should_quit(self) -> bool:
        return self._shutdown.is_set()

    @property
    def should_reset(self) -> bool:
        return self._reset_requested.is_set()

    @property
    def should_discard(self) -> bool:
        return self._discard_requested.is_set()

    # ── Keyboard Monitoring ────────────────────────────────

    def start_keyboard_listener(self):
        """Start the keyboard monitoring daemon thread."""
        self._keyboard_thread = threading.Thread(
            target=self._keyboard_loop, daemon=True, name="dagger-keyboard"
        )
        self._keyboard_thread.start()
        logger.info(
            f"DAgger keyboard listener started. "
            f"Press '{self.config.key_enter_dagger}' to intervene, "
            f"'{self.config.key_resume_inference}' to resume, "
            f"'r' to end+save episode, 'd' to end+discard episode, 'q' to quit."
        )

    def _keyboard_loop(self):
        """Capture single keystrokes without requiring Enter (Linux termios)."""
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._shutdown.is_set():
                ch = sys.stdin.read(1)
                if ch in ("\r", "\n"):
                    self._start_event.set()
                elif ch == self.config.key_enter_dagger:
                    self._on_enter_dagger()
                elif ch == self.config.key_resume_inference:
                    self._on_resume_inference()
                elif ch == "r":
                    logger.info("[DAgger] Reset (save) requested.")
                    self._reset_requested.set()
                elif ch == "d":
                    logger.info("[DAgger] Discard requested.")
                    self._discard_requested.set()
                elif ch == "q":
                    logger.info("[DAgger] Quit requested.")
                    self._shutdown.set()
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def wait_for_start(self) -> bool:
        """Block until Enter is pressed or quit is requested.

        Returns True if the caller should quit, False if episode should start.
        """
        self._start_event.clear()
        while not self._shutdown.is_set():
            if self._start_event.wait(timeout=0.1):
                self._start_event.clear()
                return False
        return True

    def _on_enter_dagger(self):
        if self.mode != DaggerMode.INFERENCE:
            logger.debug("[DAgger] Already in DAgger mode, ignoring 'i' press.")
            return
        logger.info("[DAgger] >>> Entering DAgger mode — pausing inference...")
        # Cancel any in-progress leader homing to avoid race condition where
        # the homing thread switches leader back to PASSIVE during alignment.
        self._homing_cancel.set()
        self._pause_event.set()
        self.mode = DaggerMode.ALIGNING
        self.stats.start_intervention(self._step_counter)

    def _on_resume_inference(self):
        if self.mode not in (DaggerMode.DEMONSTRATING, DaggerMode.ALIGNING):
            logger.warning("[DAgger] Not in DAgger mode, ignoring 'o' press.")
            return
        logger.info("[DAgger] >>> Resuming inference mode...")
        self.mode = DaggerMode.RESUMING

    # ── Arm Alignment ──────────────────────────────────────

    def generate_alignment_trajectory(
        self,
        leader_qpos: np.ndarray,
        follower_qpos: np.ndarray,
        dof_per_arm: int = 7,
    ) -> list[np.ndarray]:
        """Generate cosine-interpolated trajectory from leader to follower position.

        Gripper joints (index 6, 13 for dual-arm) are NOT interpolated — they
        snap to the follower's gripper state immediately.

        Args:
            leader_qpos: Current leader arm positions, shape (n_arms * dof,).
            follower_qpos: Current follower arm positions, shape (n_arms * dof,).
            dof_per_arm: Degrees of freedom per arm (default 7: 6 joints + 1 gripper).

        Returns:
            List of waypoint arrays for the alignment trajectory.
        """
        steps = self.config.align_steps
        n_joints = len(leader_qpos)
        n_arms = n_joints // dof_per_arm
        gripper_indices = [arm_idx * dof_per_arm + (dof_per_arm - 1) for arm_idx in range(n_arms)]

        trajectory = []
        for i in range(1, steps + 1):
            alpha = 0.5 * (1 - math.cos(math.pi * i / steps))
            waypoint = (1 - alpha) * leader_qpos + alpha * follower_qpos

            # Snap gripper: binarize based on follower state
            for gi in gripper_indices:
                waypoint[gi] = follower_qpos[gi]

            trajectory.append(waypoint.copy())

        return trajectory

    def execute_alignment(
        self,
        get_leader_qpos: Callable[[], np.ndarray],
        get_follower_qpos: Callable[[], np.ndarray],
        send_leader_action: Callable[[np.ndarray], None],
        switch_leader_mode_sampling: Callable[[], None],
        switch_leader_mode_passive: Callable[[], None],
    ):
        """Execute the full alignment sequence.

        1. Switch leader to SAMPLING (position-controlled)
        2. Cosine interpolate leader → follower position
        3. Switch leader to PASSIVE (gravity compensation, human can move)
        4. Set mode to DEMONSTRATING
        """
        leader_qpos = np.array(get_leader_qpos())
        follower_qpos = np.array(get_follower_qpos())

        logger.info(f"[DAgger] Aligning leader to follower ({self.config.align_steps} steps)...")

        # Reset homing cancel event so the new homing after this intervention
        # can use it as a fresh cancellation signal.
        self._homing_cancel.clear()

        # Leader must be in position-controlled mode to receive commands
        switch_leader_mode_sampling()
        time.sleep(0.1)  # Wait for mode switch to complete

        trajectory = self.generate_alignment_trajectory(leader_qpos, follower_qpos)
        dt = self.config.align_duration / self.config.align_steps

        for waypoint in trajectory:
            # Abort if shutdown requested OR if mode changed (new intervention started)
            if self._shutdown.is_set() or self.mode != DaggerMode.ALIGNING:
                if self.mode != DaggerMode.ALIGNING:
                    logger.info("[DAgger] Alignment interrupted by new intervention.")
                # Restore leader to passive mode before aborting
                try:
                    switch_leader_mode_passive()
                except Exception as e:
                    logger.warning("[DAgger] Failed to switch leader to passive on abort: %s", e)
                return
            try:
                send_leader_action(waypoint)
            except Exception as e:
                logger.warning("[DAgger] Failed to send waypoint: %s. Aborting alignment.", e)
                try:
                    switch_leader_mode_passive()
                except Exception:
                    pass
                return
            time.sleep(dt)

        # Switch leader to passive (gravity compensation) for human control
        switch_leader_mode_passive()
        self.mode = DaggerMode.DEMONSTRATING
        logger.info("[DAgger] Alignment complete. Leader in PASSIVE mode — you may demonstrate now.")

    # ── Step Counting ──────────────────────────────────────

    def count_step(self, intervention: bool):
        """Increment step counters for statistics."""
        self._step_counter += 1
        if intervention:
            self.stats.intervention_steps += 1
        else:
            self.stats.autonomous_steps += 1

    def record_demo_frame(self, leader_qpos: np.ndarray, prev_qpos: Optional[np.ndarray]) -> bool:
        """Decide whether to record a demonstration frame based on motion threshold.

        Compares each joint dimension against demo_motion_threshold. If ALL
        deltas are below the threshold (arm is stationary), the frame is skipped.
        The first frame of each intervention (prev_qpos is None) is always recorded.

        Returns True if the frame should be recorded, False if it should be skipped.
        """
        if prev_qpos is None:
            self.stats._current_demo_recorded += 1
            return True
        delta = np.abs(np.asarray(leader_qpos) - np.asarray(prev_qpos))
        if np.all(delta < self.config.demo_motion_threshold):
            self.stats._current_demo_skipped += 1
            return False
        self.stats._current_demo_recorded += 1
        return True

    # ── Resume Completion ──────────────────────────────────

    def complete_resume(self):
        """Finalize the resume transition back to inference."""
        self.stats.end_intervention(self._step_counter)
        recorded = self.stats._current_demo_recorded
        skipped = self.stats._current_demo_skipped
        total = recorded + skipped
        logger.info(
            "[DAgger] Intervention ended: %d/%d frames recorded, %d skipped "
            "(threshold=%.4f)",
            recorded, total, skipped, self.config.demo_motion_threshold,
        )
        self.mode = DaggerMode.INFERENCE
        self._pause_event.clear()
        logger.info("[DAgger] Inference resumed.")

    # ── Episode Lifecycle ──────────────────────────────────

    def reset_episode(self):
        """Reset state for a new episode."""
        self.stats.reset()
        self._step_counter = 0
        self.mode = DaggerMode.INFERENCE
        self._pause_event.clear()
        self._reset_requested.clear()
        self._discard_requested.clear()
        self._start_event.clear()

    def shutdown(self):
        """Clean shutdown."""
        self._shutdown.set()
        final_stats = self.stats.to_dict()
        logger.info(f"[DAgger] Session stats: {final_stats}")
        return final_stats
