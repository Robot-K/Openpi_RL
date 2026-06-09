import collections
import logging
import select
import sys
import termios
import threading
import time
import tty
from typing import Any

from airdc.common.systems.basis import System
from airdc.common.systems.basis import SystemMode
from airdc.utils import init_logging
from dagger_controller import DaggerConfig
from dagger_controller import DaggerController
from dagger_controller import DaggerMode
from inference_recorder import InferenceRecorder
from inference_recorder import RecordConfig
import numpy as np
from pydantic import BaseModel
from robot_config import RobotConfig
import torch
import tyro

init_logging(logging.INFO)

# Suppress cosmetic asyncio warning emitted at exit when V4L2Camera tasks are
# garbage-collected before they can be cancelled.  The cameras are already
# closed by operator.shutdown(); no data is lost.
logging.getLogger("asyncio").addFilter(
    type("_", (logging.Filter,), {"filter": staticmethod(lambda r: "Task was destroyed but it is pending" not in r.getMessage())})()
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RemotePolicyConfig(BaseModel):
    """Configuration for the remote policy client."""

    host: str = "localhost"
    port: int = 8000


class InferConfig(BaseModel):
    """Configuration for the inference script."""

    policy_config: RemotePolicyConfig
    max_steps: int = 500000
    step_rate: int = 200
    reset_action: list[float] = [
        -0.,
        -1.,
        0.8,
        1.6,
        -0.6,
        -1.7,
        0.0,
        -0.,
        -1.,
        0.8,
        -1.6,
        0.6,
        1.7,
        0.0,
    ]
    # -1 means keep inference running continuously. Non-negative values trigger
    # a new inference only when the remaining queue length drops below threshold.
    inference_trigger_threshold: int = -1
    tcs_drop_max: int = 12
    tcs_min_overlap: int = 8
    initial_action_wait_s: float = 10.0
    max_inference_rate: float = 2  # Maximum inference frequency in riiHz (0 = unlimited)
    interpolate: bool = False
    step_length: list[float] = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.25]
    debug: bool = False
    prompt: str = ""
    record: RecordConfig = RecordConfig(record_data=False)
    dagger: DaggerConfig = DaggerConfig(enable=False)
    robot_config: RobotConfig


class AutoConfig(BaseModel):
    chunk_size_predict: int = 0
    state_dim: int = -1
    camera_names: list[str] = []
    observation: dict[str, Any] = {"qpos": None, "images": {}}


auto_config = AutoConfig()

config = tyro.cli(InferConfig, config=[tyro.conf.ConsolidateSubcommandArgs])
robot_config = config.robot_config
if robot_config.robot_type == "play":
    from play_operator import Robot
else:
    raise ValueError("Unsupported robot type. Please use a valid config path for the robot.")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def interpolate_action(step_length, prev_action, cur_action):
    """Interpolate actions to make the robot move smoothly."""
    steps = np.asarray(step_length, dtype=np.float32)
    if steps.size != prev_action.size:
        repeats = int(np.ceil(prev_action.size / steps.size))
        steps = np.tile(steps, repeats)[: prev_action.size]
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


def clone_policy_observation(policy_observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "qpos": np.asarray(policy_observation["qpos"], dtype=np.float32).copy(),
        "images": {
            camera_name: np.asarray(image, dtype=np.uint8).copy()
            for camera_name, image in policy_observation["images"].items()
        },
    }


def update_observation(camera_names: list[str], operator: System) -> tuple[dict, dict[str, Any]]:
    """Update the cached observation and return raw + policy observations."""
    obs = operator.capture_observation()
    qpos = operator.get_qpos(obs)
    image_dict = {}
    for camera_name in camera_names:
        image_dict[camera_name] = np.asarray(obs[f"{camera_name}/color/image_raw"]["data"]).copy()
    auto_config.observation = {"qpos": np.asarray(qpos, dtype=np.float32), "images": image_dict}
    return obs, clone_policy_observation(auto_config.observation)


def build_policy_input(policy_observation: dict[str, Any], prompt: str) -> dict[str, Any]:
    return {
        "state": policy_observation["qpos"],
        "prompt": prompt,
        "advantage": True,
    } | policy_observation["images"]


def inference_once(policy, prompt: str, policy_observation: dict[str, Any] | None = None) -> np.ndarray:
    """Run policy inference on a specific observation snapshot."""
    observation = policy_observation or auto_config.observation
    obs = build_policy_input(observation, prompt)
    action_chunk = policy.infer(obs)["actions"] if policy is not None else np.zeros([64, 7], dtype=np.float32)
    action_chunk = action_chunk
    auto_config.chunk_size_predict = action_chunk.shape[0]
    auto_config.state_dim = action_chunk.shape[1]
    return np.asarray(action_chunk, dtype=np.float32)


def smooth_action_chunks(
    old_actions: list[np.ndarray],
    new_chunk: np.ndarray,
    executed_since_request: int,
    drop_max: int,
    min_overlap: int,
) -> np.ndarray:
    """Apply paper-style temporal chunk-wise smoothing."""
    drop = min(max(executed_since_request, 0), drop_max)
    if drop >= len(new_chunk):
        return np.asarray(old_actions, dtype=np.float32) if old_actions else np.empty((0, 0), dtype=np.float32)

    new_remaining = np.asarray(new_chunk[drop:], dtype=np.float32)
    if not old_actions:
        return new_remaining

    overlap = min(min_overlap, len(new_remaining))
    if overlap <= 0:
        return new_remaining

    old_buffer = np.asarray(old_actions, dtype=np.float32)
    old_for_overlap = old_buffer[:overlap]
    if len(old_for_overlap) < overlap:
        pad_count = overlap - len(old_for_overlap)
        pad = np.repeat(old_for_overlap[-1][np.newaxis, :], pad_count, axis=0)
        old_for_overlap = np.concatenate([old_for_overlap, pad], axis=0)

    if overlap == 1:
        weights = np.array([1.0], dtype=np.float32)
    else:
        weights = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
    blended = weights[:, np.newaxis] * old_for_overlap[:overlap] + (1.0 - weights)[:, np.newaxis] * new_remaining[:overlap]
    suffix = new_remaining[overlap:]
    if len(suffix) == 0:
        return blended
    return np.concatenate([blended, suffix], axis=0)


class AsyncChunkController:
    """Runs policy inference asynchronously and maintains a smoothed action buffer."""

    def __init__(self, policy, prompt: str, config: InferConfig, dagger_ctrl: DaggerController | None = None):
        self._policy = policy
        self._prompt = prompt
        self._config = config
        self._dagger_ctrl = dagger_ctrl

        self._buffer: collections.deque[np.ndarray] = collections.deque()
        self._latest_observation: dict[str, Any] | None = None
        self._latest_observation_version = 0
        self._executed_steps = 0

        self._lock = threading.Lock()
        self._observation_event = threading.Event()
        self._actions_ready = threading.Event()
        self._stop_event = threading.Event()
        self._last_inference_time = 0.0  # Track last inference time for rate limiting
        self._last_inference_step = 0  # Track step count at last inference
        self._thread = threading.Thread(target=self._inference_loop, daemon=True, name="async-policy-inference")

    def start(self):
        self._thread.start()

    def shutdown(self):
        self._stop_event.set()
        self._observation_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def update_observation(self, policy_observation: dict[str, Any]):
        with self._lock:
            self._latest_observation = clone_policy_observation(policy_observation)
            self._latest_observation_version += 1
        self._observation_event.set()

    def wait_for_actions(self, timeout: float) -> bool:
        return self._actions_ready.wait(timeout=timeout)

    def pop_action(self) -> np.ndarray | None:
        with self._lock:
            if not self._buffer:
                self._actions_ready.clear()
                return None
            action = self._buffer.popleft()
            if not self._buffer:
                self._actions_ready.clear()
            return np.asarray(action, dtype=np.float32).copy()

    def mark_step_executed(self):
        with self._lock:
            self._executed_steps += 1

    def clear_actions(self):
        with self._lock:
            self._buffer.clear()
            self._actions_ready.clear()

    def _should_pause(self) -> bool:
        return self._dagger_ctrl is not None and self._dagger_ctrl.mode != DaggerMode.INFERENCE

    def _should_request(self, obs_version: int, last_requested_version: int, remaining_actions: int) -> bool:
        if obs_version <= last_requested_version:
            return False
        if self._should_pause():
            return False
        if self._config.inference_trigger_threshold < 0:
            return True
        return remaining_actions <= self._config.inference_trigger_threshold

    def _inference_loop(self):
        last_requested_version = 0
        while not self._stop_event.is_set():
            if not self._observation_event.wait(timeout=0.05):
                continue
            self._observation_event.clear()
            if self._stop_event.is_set():
                break

            with self._lock:
                policy_observation = (
                    None if self._latest_observation is None else clone_policy_observation(self._latest_observation)
                )
                observation_version = self._latest_observation_version
                request_step = self._executed_steps
                remaining_actions = len(self._buffer)

            if policy_observation is None:
                continue
            if not self._should_request(observation_version, last_requested_version, remaining_actions):
                continue

            # Rate limiting: enforce minimum interval between inferences
            if self._config.max_inference_rate > 0:
                min_interval = 1.0 / self._config.max_inference_rate
                elapsed = time.monotonic() - self._last_inference_time
                if elapsed < min_interval:
                    continue

            last_requested_version = observation_version
            infer_start = time.monotonic()
            action_chunk = inference_once(self._policy, self._prompt, policy_observation)
            robot_dof = len(self._config.reset_action)
            if action_chunk.shape[1] > robot_dof:
                action_chunk = action_chunk[:, :robot_dof]
            infer_dt = time.monotonic() - infer_start
            self._last_inference_time = time.monotonic()
            
            # Track steps between inferences
            steps_since_last = request_step - self._last_inference_step
            self._last_inference_step = request_step

            if self._should_pause():
                continue

            with self._lock:
                latency_steps = self._executed_steps - request_step
                smoothed_actions = smooth_action_chunks(
                    list(self._buffer),
                    action_chunk,
                    latency_steps,
                    self._config.tcs_drop_max,
                    self._config.tcs_min_overlap,
                )
                self._buffer = collections.deque(np.asarray(action, dtype=np.float32).copy() for action in smoothed_actions)
                if self._buffer:
                    self._actions_ready.set()
                else:
                    self._actions_ready.clear()
                remaining_after_fuse = len(self._buffer)

            logger.info(
                "Async inference done: %.3fs, latency_steps=%d, steps_since_last=%d, chunk=%d, queue=%d",
                infer_dt,
                latency_steps,
                steps_since_last,
                len(action_chunk),
                remaining_after_fuse,
            )


class KeyboardListener:
    """Listen for keyboard input in a separate thread."""

    def __init__(self):
        self.reset_flag = False
        self.quit_flag = False
        self.start_flag = False
        self.discard_flag = False
        self.listener_thread = None
        self.running = False
        self.old_settings = None

    def start(self):
        self.running = True
        self.old_settings = termios.tcgetattr(sys.stdin)
        self.listener_thread = threading.Thread(target=self._listen, daemon=True)
        self.listener_thread.start()
        logger.info("Keyboard listener started. Press 'Enter' to start, 'R' to reset/save, 'D' to discard, 'Q' to quit.")

    def _listen(self):
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.running:
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    if key.lower() == "r":
                        self.reset_flag = True
                        logger.info("Reset requested!")
                    elif key.lower() == "q":
                        self.quit_flag = True
                        logger.info("Quit requested!")
                    elif key.lower() == "d":
                        self.discard_flag = True
                        logger.info("Discard requested!")
                    elif key in {"\n", "\r"}:
                        self.start_flag = True
                        logger.info("Start requested!")
                time.sleep(0.01)
        except Exception as exc:
            logger.error("Keyboard listener error: %s", exc)
        finally:
            if self.old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def check_reset(self):
        if self.reset_flag:
            self.reset_flag = False
            return True
        return False

    def check_quit(self):
        return self.quit_flag

    def check_start(self):
        if self.start_flag:
            self.start_flag = False
            return True
        return False

    def check_discard(self):
        if self.discard_flag:
            self.discard_flag = False
            return True
        return False

    def stop(self):
        self.running = False
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


def model_inference(config: InferConfig, operator: System):
    auto_config.camera_names = operator.config.camera_names
    assert config.prompt, "Prompt must be provided for policy inference."

    from openpi_client import websocket_client_policy

    policy_config = config.policy_config
    logger.info("Connecting to policy server at %s:%s", policy_config.host, policy_config.port)
    policy = websocket_client_policy.WebsocketClientPolicy(host=policy_config.host, port=policy_config.port)

    record_config = config.record
    if record_config.record_data and not record_config.task_name:
        record_config = record_config.model_copy(update={"task_name": config.prompt})
    recorder = InferenceRecorder(record_config, auto_config.camera_names)

    dagger_ctrl = None
    if config.dagger.enable:
        dagger_ctrl = DaggerController(config.dagger)
        operator.init_leaders()
        dagger_ctrl.start_keyboard_listener()
        logger.info(
            "[DAgger] DAgger mode ENABLED. Press '%s' to intervene, '%s' to resume, 'q' to quit.",
            config.dagger.key_enter_dagger,
            config.dagger.key_resume_inference,
        )

    keyboard_listener = None
    if not dagger_ctrl:
        keyboard_listener = KeyboardListener()
        keyboard_listener.start()

    try:
        while True:
            if dagger_ctrl and dagger_ctrl.should_quit:
                logger.info("DAgger quit requested.")
                break
            if keyboard_listener and keyboard_listener.check_quit():
                logger.info("Quitting...")
                break

            operator.switch_mode(SystemMode.RESETTING)
            operator.send_action(config.reset_action)

            if keyboard_listener:
                logger.info("Press 'Enter' to start episode...")
                while not keyboard_listener.check_start():
                    if keyboard_listener.check_quit():
                        logger.info("Quitting...")
                        break
                    time.sleep(0.1)
                if keyboard_listener.check_quit():
                    break
            else:
                # DAgger mode: keyboard thread handles Enter and q
                logger.info("Press 'Enter' to start episode or 'q' to quit...")
                if dagger_ctrl.wait_for_start():
                    logger.info("Quitting...")
                    break

            operator.switch_mode(SystemMode.SAMPLING)
            recorder.start_episode()
            if dagger_ctrl:
                dagger_ctrl.reset_episode()

            async_controller = AsyncChunkController(policy, config.prompt, config, dagger_ctrl)
            async_controller.start()

            with torch.inference_mode():
                raw_obs, policy_observation = update_observation(auto_config.camera_names, operator)
                async_controller.update_observation(policy_observation)
                if not async_controller.wait_for_actions(config.initial_action_wait_s):
                    logger.warning(
                        "Timed out waiting for the first action chunk after %.1fs. Falling back to hold-last-action until ready.",
                        config.initial_action_wait_s,
                    )

                t = 0
                reset_requested = False
                discard_requested = False
                queue_starved_logged = False
                prev_demo_qpos = None
                
                # Frequency monitoring
                last_step_time = time.monotonic()
                step_times = []
                freq_log_interval = 100  # Log frequency every N steps
                
                # Performance profiling
                perf_obs_times = []
                perf_record_times = []
                perf_action_times = []
                last_action = np.asarray(config.reset_action, dtype=np.float32)

                try:
                    while t < config.max_steps:
                        step_start = time.monotonic()  # Track step start for deadline-based timing

                        if dagger_ctrl and dagger_ctrl.should_quit:
                            break
                        if dagger_ctrl and dagger_ctrl.should_discard:
                            logger.info("Discarding episode...")
                            discard_requested = True
                            break
                        if dagger_ctrl and dagger_ctrl.should_reset:
                            logger.info("Resetting episode...")
                            reset_requested = True
                            break
                        if keyboard_listener:
                            if keyboard_listener.check_discard():
                                logger.info("Discarding episode...")
                                discard_requested = True
                                break
                            if keyboard_listener.check_reset():
                                logger.info("Resetting episode...")
                                reset_requested = True
                                break
                            if keyboard_listener.check_quit():
                                logger.info("Quitting...")
                                break

                        if not dagger_ctrl or dagger_ctrl.mode == DaggerMode.INFERENCE:
                            t_obs_start = time.monotonic()
                            raw_obs, policy_observation = update_observation(auto_config.camera_names, operator)
                            async_controller.update_observation(policy_observation)
                            perf_obs_times.append(time.monotonic() - t_obs_start)

                            action = async_controller.pop_action()
                            if action is None:
                                action = last_action.copy()
                                if not queue_starved_logged:
                                    logger.warning("Action buffer empty. Holding the last commanded action until a new chunk arrives.")
                                    queue_starved_logged = True
                            else:
                                queue_starved_logged = False

                            t_record_start = time.monotonic()
                            recorder.record_step(raw_obs, action, intervention=0)
                            if dagger_ctrl:
                                dagger_ctrl.count_step(intervention=False)
                            perf_record_times.append(time.monotonic() - t_record_start)

                            # Execute action (with optional interpolation)
                            t_action_start = time.monotonic()
                            if config.interpolate:
                                interp_actions = interpolate_action(config.step_length, last_action, action)
                                for act in interp_actions:
                                    operator.send_action(act)
                                    time.sleep(1.0 / config.step_rate)
                            else:
                                operator.send_action(action)
                            perf_action_times.append(time.monotonic() - t_action_start)
                            last_action = action.copy()

                            # Sleep for remaining time to hit target step_rate
                            if not config.interpolate:
                                step_duration = 1.0 / config.step_rate
                                elapsed = time.monotonic() - step_start
                                if elapsed < step_duration:
                                    time.sleep(step_duration - elapsed)


                            async_controller.mark_step_executed()
                            t += 1
                            
                            # Monitor actual execution frequency
                            current_time = time.monotonic()
                            step_times.append(current_time - last_step_time)
                            last_step_time = current_time
                            if t % freq_log_interval == 0 and len(step_times) >= freq_log_interval:
                                avg_step_time = sum(step_times[-freq_log_interval:]) / freq_log_interval
                                actual_freq = 1.0 / avg_step_time if avg_step_time > 0 else 0
                                
                                # Calculate average times for each operation
                                avg_obs = sum(perf_obs_times[-freq_log_interval:]) / min(len(perf_obs_times), freq_log_interval) * 1000
                                avg_record = sum(perf_record_times[-freq_log_interval:]) / min(len(perf_record_times), freq_log_interval) * 1000
                                avg_action = sum(perf_action_times[-freq_log_interval:]) / min(len(perf_action_times), freq_log_interval) * 1000
                                
                                logger.info(
                                    "Actual freq: %.1f Hz (target: %d Hz) | Timing: obs=%.1fms, record=%.1fms, action=%.1fms",
                                    actual_freq, config.step_rate, avg_obs, avg_record, avg_action
                                )

                        elif dagger_ctrl.mode == DaggerMode.ALIGNING:
                            logger.info("[DAgger] Starting alignment...")
                            prev_demo_qpos = None  # reset for new intervention
                            async_controller.clear_actions()
                            dagger_ctrl.execute_alignment(
                                get_leader_qpos=operator.get_leader_qpos,
                                get_follower_qpos=operator.get_follower_qpos,
                                send_leader_action=operator.send_leader_action,
                                switch_leader_mode_sampling=lambda: operator.switch_leader_mode(SystemMode.SAMPLING),
                                switch_leader_mode_passive=lambda: operator.switch_leader_mode(SystemMode.PASSIVE),
                            )

                        elif dagger_ctrl.mode == DaggerMode.DEMONSTRATING:
                            raw_obs, _policy_observation = update_observation(auto_config.camera_names, operator)
                            leader_qpos = operator.get_leader_qpos()
                            operator.send_action(leader_qpos)
                            if dagger_ctrl.record_demo_frame(leader_qpos, prev_demo_qpos):
                                recorder.record_step(raw_obs, leader_qpos, intervention=1)
                            dagger_ctrl.count_step(intervention=True)
                            prev_demo_qpos = leader_qpos.copy()

                            # Sleep for remaining time to hit target step_rate
                            step_duration = 1.0 / config.step_rate
                            elapsed = time.monotonic() - step_start
                            if elapsed < step_duration:
                                time.sleep(step_duration - elapsed)

                            t += 1

                        elif dagger_ctrl.mode == DaggerMode.RESUMING:
                            logger.info("[DAgger] Resuming inference...")
                            prev_demo_qpos = None
                            async_controller.clear_actions()

                            def _home_leaders(cancel_event):
                                try:
                                    operator.switch_leader_mode(SystemMode.RESETTING)
                                    operator.send_leader_action(np.asarray(config.reset_action, dtype=np.float32))
                                    if cancel_event.wait(timeout=1.0):
                                        logger.info("[DAgger] Leader homing cancelled by new intervention.")
                                        return
                                    operator.switch_leader_mode(SystemMode.PASSIVE)
                                except Exception as exc:
                                    logger.warning("[DAgger] Leader homing failed: %s", exc)

                            threading.Thread(target=_home_leaders, args=(dagger_ctrl._homing_cancel,), daemon=True).start()
                            dagger_ctrl.complete_resume()
                            raw_obs, policy_observation = update_observation(auto_config.camera_names, operator)
                            async_controller.update_observation(policy_observation)
                            # Defensive: clear again in case the inference thread
                            # pushed stale actions between complete_resume() (which
                            # sets mode=INFERENCE) and now.
                            async_controller.clear_actions()
                            # Update last_action to current robot position so the
                            # hold-last-action fallback won't jump back to the
                            # pre-dagger pose while waiting for fresh inference.
                            last_action = np.asarray(policy_observation["qpos"], dtype=np.float32).copy()

                finally:
                    async_controller.shutdown()

                dagger_stats = dagger_ctrl.stats.to_dict() if dagger_ctrl else None
                if discard_requested:
                    recorder.discard_episode()
                else:
                    recorder.save_episode(dagger_stats=dagger_stats)

                if keyboard_listener and keyboard_listener.check_quit():
                    break

                if not reset_requested and not (dagger_ctrl and dagger_ctrl.should_quit):
                    logger.info("Episode completed.")
    finally:
        if keyboard_listener:
            keyboard_listener.stop()
        if dagger_ctrl:
            dagger_ctrl.shutdown()
        recorder.shutdown()
        operator.shutdown()


def main():
    model_inference(config, Robot(config.robot_config))


if __name__ == "__main__":
    main()
