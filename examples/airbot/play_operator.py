# from airbot_ie.robots.airbot_play_mock import AIRBOTPlay, AIRBOTPlayConfig
from airbot_ie.robots.airbot_play import AIRBOTPlay, AIRBOTPlayConfig
from airdc.common.devices.cameras.v4l2 import V4L2Camera, V4L2CameraConfig
from airdc.common.systems.basis import SystemMode
from robot_config import RobotConfig

import logging
import numpy as np
import time

logger = logging.getLogger(__name__)


class Robot:
    """Robot class for the AIRBOT Play robot."""

    def __init__(self, config: RobotConfig):
        self.config = config
        self.robots = {
            name: AIRBOTPlay(AIRBOTPlayConfig(port=port))
            for name, port in zip(self.config.robot_groups, self.config.robot_ports, strict=True)
        }
        self.cameras = {
            name: V4L2Camera(V4L2CameraConfig(
                camera_index=index,
                pixel_format=pixel_format,
                width=width,
                height=height,
                fps=fps,
            ))
            for name, index, pixel_format, width, height, fps in zip(
                self.config.camera_names,
                self.config.camera_index,
                self.config.camera_pixel_format,
                self.config.camera_width,
                self.config.camera_height,
                self.config.camera_fps,
                strict=True,
            )
        }
        self.keys = list(self.robots.keys()) + list(self.cameras.keys())
        self.values = list(self.robots.values()) + list(self.cameras.values())
        for key, value in zip(self.keys, self.values, strict=True):
            if not value.configure():
                raise RuntimeError(f"Failed to configure {key}.")
            if key in self.robots:
                value.switch_mode(SystemMode.RESETTING)

        # Leader arms (initialized lazily via init_leaders for DAgger)
        self.leaders: dict[str, AIRBOTPlay] = {}

    def init_leaders(self):
        """Initialize leader (master) arm connections for DAgger mode.

        Uses leader_ports from RobotConfig. Each leader arm corresponds
        to a follower arm in the same robot_group.
        """
        if self.leaders:
            logger.info("Leader arms already initialized.")
            return

        for name, port in zip(self.config.robot_groups, self.config.leader_ports, strict=True):
            leader = AIRBOTPlay(AIRBOTPlayConfig(port=port))
            if not leader.configure():
                raise RuntimeError(f"Failed to configure leader arm '{name}' on port {port}.")
            leader.switch_mode(SystemMode.PASSIVE)
            self.leaders[name] = leader
            logger.info(f"Leader arm '{name}' initialized on port {port}.")

        # Start leaders in PASSIVE mode (gravity compensation)
        self.switch_leader_mode(SystemMode.PASSIVE)
        logger.info("All leader arms initialized in PASSIVE mode.")

    def switch_mode(self, mode):
        """Switch the mode of the follower robots (backward compatible)."""
        for robot in self.robots.values():
            robot.switch_mode(mode)

    def switch_follower_mode(self, mode: SystemMode):
        """Switch the mode of follower (slave) arms."""
        for robot in self.robots.values():
            robot.switch_mode(mode)

    def switch_leader_mode(self, mode: SystemMode):
        """Switch the mode of leader (master) arms.

        PASSIVE = gravity compensation (human can freely move the arm)
        SAMPLING = position servo (arm follows commanded positions)
        """
        for leader in self.leaders.values():
            leader.switch_mode(mode)

    def capture_observation(self) -> dict:
        """Capture the current observation from the robot."""
        obs = {}
        for name, ins in zip(self.keys, self.values, strict=True):
            for key, value in ins.capture_observation().items():
                full_key = f"{name}/{key}"
                # Convert BGR to RGB for camera images
                if "image" in key and isinstance(value.get("data"), np.ndarray):
                    image_data = value["data"]
                    if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                        value = value.copy()
                        value["data"] = image_data[..., ::-1]
                obs[full_key] = value
        return obs

    def send_action(self, action):
        """Send the action to the follower robot."""
        for index, (_group, robot) in enumerate(self.robots.items()):
            joint_target = [float(v) for v in action[index * 7 : (index + 1) * 7]]
            stamp = time.time_ns()
            joint_target[6] *= 0.072 / 0.0471
            robot.send_action(
                {
                    "arm/joint_state/position": {"data": joint_target[:6], "t": stamp},
                    "eef/joint_state/position": {"data": joint_target[6:7], "t": stamp},
                }
            )

    def send_leader_action(self, action):
        """Send position command to leader (master) arms.

        Used during alignment to move leader arms to follower positions.
        Leader must be in SAMPLING mode to accept position commands.
        """
        for index, (_group, leader) in enumerate(self.leaders.items()):
            joint_target = [float(v) for v in action[index * 7 : (index + 1) * 7]]
            stamp = time.time_ns()
            leader.send_action(
                {
                    "arm/joint_state/position": {"data": joint_target[:6], "t": stamp},
                    "eef/joint_state/position": {"data": joint_target[6:7], "t": stamp},
                }
            )

    def get_qpos(self, obs: dict) -> list[float]:
        """Get the joint positions of the follower robot."""
        qpos = []
        for group in self.config.robot_groups:
            qpos.extend(obs[f"{group}/arm/joint_state/position"]["data"])
            qpos.extend(obs[f"{group}/eef/joint_state/position"]["data"])
        return qpos

    def get_follower_qpos(self) -> np.ndarray:
        """Get current follower (slave) arm joint positions as numpy array."""
        qpos = []
        for group in self.config.robot_groups:
            obs = self.robots[group].capture_observation()
            qpos.extend(obs["arm/joint_state/position"]["data"])
            qpos.extend(obs["eef/joint_state/position"]["data"])
        return np.array(qpos)

    def get_leader_qpos(self) -> np.ndarray:
        """Get current leader (master) arm joint positions as numpy array."""
        qpos = []
        for group in self.config.robot_groups:
            obs = self.leaders[group].capture_observation()
            qpos.extend(obs["arm/joint_state/position"]["data"])
            qpos.extend(obs["eef/joint_state/position"]["data"])
        return np.array(qpos)

    def shutdown(self) -> bool:
        """Shutdown the robot."""
        for robot in self.robots.values():
            robot.shutdown()
        for leader in self.leaders.values():
            leader.shutdown()
        for camera in self.cameras.values():
            camera.shutdown()
        return True
