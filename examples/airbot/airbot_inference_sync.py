import logging
import time
import threading
import select
import sys
import termios
import tty

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
    """Configuration for the remote policy client.
    Args:
        host (str): Hostname or IP address of the policy server.
        port (int): Port number of the policy server.
    """

    host: str = "localhost"
    port: int = 8000


class InferConfig(BaseModel):
    """Configuration for the inference script.
    Args:
        max_steps (int): Maximum number of action publishing steps.
        step_rate (int): The rate at which to publish the actions.
        reset_action (list[float]): Initial action to reset the robot arm.
    """

    policy_config: RemotePolicyConfig
    max_steps: int = 500000
    step_rate: int = 25
    reset_action: list[float] = [
        -0.001618136651813984,
        -1.0361113548278809,
        0.8421794176101685,
        1.6158959865570068,
        -0.6345375776290894,
        -1.6957406997680664,
        0.0,
        0.1323927342891693,
        -1.2208569049835205,
        1.0429750680923462,
        -2.0076663494110107,
        0.840582549571991,
        2.0390350818634033,
        0.0,
    ]
    chunk_size_execute: int = 25
    interpolate: bool = False
    step_length: list[float] = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.25]
    debug: bool = False
    prompt: str = ""
    record: RecordConfig = RecordConfig(record_data=False)
    dagger: DaggerConfig = DaggerConfig(enable=False)
    robot_config: RobotConfig


class AutoConfig(BaseModel):
    chunk_size_predict: int = 0
    state_dim: int = -1
    camera_names: list[str] = []
    observation: dict = {"qpos": None, "images": {}}


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


# Update the observation in auto_config and return raw obs for recording
def update_observation(camera_names: list[str], operator: System) -> dict:
    obs = operator.capture_observation()
    qpos = operator.get_qpos(obs)
    image_dict = {}
    for camera_name in camera_names:
        image_dict[camera_name] = obs[f"{camera_name}/color/image_raw"]["data"]
    auto_config.observation = {"qpos": np.array(qpos), "images": image_dict}
    return obs


def inference_once(policy, prompt: str) -> np.ndarray:
    """Perform a single inference step using the trained policy.

    When the model is trained with delta actions (extra_delta_transform=True),
    the policy server automatically converts delta predictions back to absolute
    actions via the AbsoluteActions output transform. The returned action_chunk
    contains absolute joint angles ready for execution.
    """
    obs = {"state": auto_config.observation["qpos"], "prompt": prompt, "advantage": True} | auto_config.observation["images"]

    action_chunk = policy.infer(obs)["actions"] if policy is not None else np.zeros([64, 7], dtype=np.float32)
    auto_config.chunk_size_predict = action_chunk.shape[0]
    auto_config.state_dim = action_chunk.shape[1]
    return action_chunk


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
        """Start the keyboard listener thread."""
        self.running = True
        self.old_settings = termios.tcgetattr(sys.stdin)
        self.listener_thread = threading.Thread(target=self._listen, daemon=True)
        self.listener_thread.start()
        logger.info("Keyboard listener started. Press 'Enter' to start, 'R' to reset/save, 'D' to discard, 'Q' to quit.")

    def _listen(self):
        """Listen for keyboard input."""
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.running:
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    if key.lower() == 'r':
                        self.reset_flag = True
                        logger.info("Reset requested!")
                    elif key.lower() == 'q':
                        self.quit_flag = True
                        logger.info("Quit requested!")
                    elif key.lower() == 'd':
                        self.discard_flag = True
                        logger.info("Discard requested!")
                    elif key == '\n' or key == '\r':
                        self.start_flag = True
                        logger.info("Start requested!")
                time.sleep(0.01)
        except Exception as e:
            logger.error(f"Keyboard listener error: {e}")
        finally:
            if self.old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def check_reset(self):
        """Check if reset was requested and clear the flag."""
        if self.reset_flag:
            self.reset_flag = False
            return True
        return False

    def check_quit(self):
        """Check if quit was requested."""
        return self.quit_flag

    def check_start(self):
        """Check if start was requested and clear the flag."""
        if self.start_flag:
            self.start_flag = False
            return True
        return False

    def check_discard(self):
        """Check if discard was requested and clear the flag."""
        if self.discard_flag:
            self.discard_flag = False
            return True
        return False

    def stop(self):
        """Stop the keyboard listener."""
        self.running = False
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


def model_inference(config: InferConfig, operator: System):
    auto_config.camera_names = operator.config.camera_names
    assert config.prompt, "Prompt must be provided for policy inference."

    from openpi_client import websocket_client_policy

    # Initialize the remote policy client
    policy_config = config.policy_config
    logger.info(f"Connecting to policy server at {policy_config.host}:{policy_config.port}")
    policy = websocket_client_policy.WebsocketClientPolicy(host=policy_config.host, port=policy_config.port)

    # Initialize inference data recorder
    record_config = config.record
    if record_config.record_data and not record_config.task_name:
        record_config = record_config.model_copy(update={"task_name": config.prompt})
    recorder = InferenceRecorder(record_config, auto_config.camera_names)

    # Initialize DAgger controller if enabled
    dagger_ctrl = None
    if config.dagger.enable:
        dagger_ctrl = DaggerController(config.dagger)
        operator.init_leaders()
        dagger_ctrl.start_keyboard_listener()
        logger.info(
            "[DAgger] DAgger mode ENABLED. "
            f"Press '{config.dagger.key_enter_dagger}' to intervene, "
            f"'{config.dagger.key_resume_inference}' to resume, 'q' to quit."
        )

    # Start keyboard listener (used when DAgger is disabled)
    keyboard_listener = None
    if not dagger_ctrl:
        keyboard_listener = KeyboardListener()
        keyboard_listener.start()

    try:
        while True:
            # Check for quit
            if dagger_ctrl and dagger_ctrl.should_quit:
                logger.info("DAgger quit requested.")
                break
            if keyboard_listener and keyboard_listener.check_quit():
                logger.info("Quitting...")
                break

            # Initialize position
            operator.switch_mode(SystemMode.RESETTING)
            operator.send_action(config.reset_action)

            if keyboard_listener:
                logger.info("Press 'Enter' to start episode...")
                # Wait for Enter key
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
            # Start recording a new episode
            recorder.start_episode()
            if dagger_ctrl:
                dagger_ctrl.reset_episode()

            # Initialize the previous action to be the initial robot state
            with torch.inference_mode():
                raw_obs = update_observation(auto_config.camera_names, operator)
                pre_action = np.array(config.reset_action)
                action_chunk = None
                t = 0
                reset_requested = False
                discard_requested = False
                prev_demo_qpos = None
                
                # Frequency monitoring
                last_step_time = time.monotonic()
                step_times = []
                freq_log_interval = 100
                
                # Performance profiling
                perf_obs_times = []
                perf_infer_times = []
                perf_record_times = []
                perf_action_times = []

                while t < config.max_steps:
                    step_start = time.monotonic()  # Track step start for deadline-based timing

                    # Check for quit/reset/discard
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

                    # ── DAgger disabled or INFERENCE mode: original logic ──
                    if not dagger_ctrl or dagger_ctrl.mode == DaggerMode.INFERENCE:
                        # Capture observation
                        t_obs_start = time.monotonic()
                        raw_obs = update_observation(auto_config.camera_names, operator)
                        perf_obs_times.append(time.monotonic() - t_obs_start)

                        # When coming to the start of a new action chunk, run inference
                        action_index = t % config.chunk_size_execute
                        if action_index == 0:
                            start_time = time.monotonic()
                            logger.info("Start inference...")
                            action_chunk = inference_once(policy, config.prompt).copy()
                            robot_dof = len(config.reset_action)
                            if action_chunk.shape[1] > robot_dof:
                                action_chunk = action_chunk[:, :robot_dof]
                            infer_time = time.monotonic() - start_time
                            perf_infer_times.append(infer_time)
                            logger.info(f"Inference time: {infer_time} s")

                        action: np.ndarray = action_chunk[action_index]

                        # Record observation and action for this step
                        t_record_start = time.monotonic()
                        recorder.record_step(raw_obs, action, intervention=0)
                        if dagger_ctrl:
                            dagger_ctrl.count_step(intervention=False)
                        perf_record_times.append(time.monotonic() - t_record_start)

                        # Execute action (with optional interpolation)
                        t_action_start = time.monotonic()
                        if config.interpolate:
                            interp_actions = interpolate_action(config.step_length, pre_action, action)
                            for act in interp_actions:
                                operator.send_action(act)
                                time.sleep(1.0 / config.step_rate)
                        else:
                            operator.send_action(action)
                        perf_action_times.append(time.monotonic() - t_action_start)
                        pre_action = action.copy()

                        # Sleep for remaining time to hit target step_rate
                        if not config.interpolate:
                            step_duration = 1.0 / config.step_rate
                            elapsed = time.monotonic() - step_start
                            if elapsed < step_duration:
                                time.sleep(step_duration - elapsed)

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
                            
                            # Calculate average inference time (only when inference happens)
                            if perf_infer_times:
                                avg_infer = sum(perf_infer_times) / len(perf_infer_times) * 1000
                                logger.info(
                                    "Actual freq: %.1f Hz (target: %d Hz) | Timing: obs=%.1fms, infer=%.1fms, record=%.1fms, action=%.1fms",
                                    actual_freq, config.step_rate, avg_obs, avg_infer, avg_record, avg_action
                                )
                            else:
                                logger.info(
                                    "Actual freq: %.1f Hz (target: %d Hz) | Timing: obs=%.1fms, record=%.1fms, action=%.1fms",
                                    actual_freq, config.step_rate, avg_obs, avg_record, avg_action
                                )

                    # ── ALIGNING: move leader arms to follower position ──
                    elif dagger_ctrl.mode == DaggerMode.ALIGNING:
                        logger.info("[DAgger] Starting alignment...")
                        prev_demo_qpos = None  # reset for new intervention
                        dagger_ctrl.execute_alignment(
                            get_leader_qpos=operator.get_leader_qpos,
                            get_follower_qpos=operator.get_follower_qpos,
                            send_leader_action=operator.send_leader_action,
                            switch_leader_mode_sampling=lambda: operator.switch_leader_mode(SystemMode.SAMPLING),
                            switch_leader_mode_passive=lambda: operator.switch_leader_mode(SystemMode.PASSIVE),
                        )

                    # ── DEMONSTRATING: human controls leader, follower follows ──
                    elif dagger_ctrl.mode == DaggerMode.DEMONSTRATING:
                        # Capture observation (follower + cameras)
                        raw_obs = update_observation(auto_config.camera_names, operator)

                        # Read leader arm positions as the demonstrated action
                        leader_qpos = operator.get_leader_qpos()

                        # Send leader positions to follower arms
                        operator.send_action(leader_qpos)

                        # Record only if motion exceeds threshold
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

                    # ── RESUMING: transition back to policy inference ──
                    elif dagger_ctrl.mode == DaggerMode.RESUMING:
                        logger.info("[DAgger] Resuming inference...")
                        prev_demo_qpos = None

                        # Move leader arms back to home in background thread.
                        # Use dagger_ctrl._homing_cancel so a subsequent 'i' press
                        # can abort the sleep and prevent overwriting the new SAMPLING mode.
                        def _home_leaders(cancel_event):
                            try:
                                operator.switch_leader_mode(SystemMode.RESETTING)
                                operator.send_leader_action(np.array(config.reset_action))
                                if cancel_event.wait(timeout=1.0):
                                    logger.info("[DAgger] Leader homing cancelled by new intervention.")
                                    return
                                operator.switch_leader_mode(SystemMode.PASSIVE)
                            except Exception as e:
                                logger.warning(f"[DAgger] Leader homing failed: {e}")

                        threading.Thread(target=_home_leaders, args=(dagger_ctrl._homing_cancel,), daemon=True).start()

                        # Force new inference on next INFERENCE step by resetting chunk index
                        t = (t // config.chunk_size_execute + 1) * config.chunk_size_execute

                        # Complete the resume transition
                        dagger_ctrl.complete_resume()

                # Save or discard the recorded episode data
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
