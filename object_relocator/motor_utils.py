"""Motor bus creation and smooth motion utilities for SO-101."""

import json
import time
from pathlib import Path

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode, MotorCalibration

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def create_motor_bus(robot_port: str, robot_id: str = None) -> FeetechMotorsBus:
    """Create a FeetechMotorsBus with SO-101 motor configuration.

    Loads calibration from the standard lerobot cache if robot_id is provided.
    """
    calibration = None
    if robot_id:
        cal_dir = (
            Path.home() / ".cache" / "huggingface" / "lerobot"
            / "calibration" / "robots" / "so101_follower"
        )
        cal_path = cal_dir / f"{robot_id}.json"
        if cal_path.is_file():
            with open(cal_path) as f:
                cal_dict = json.load(f)
            calibration = {
                motor: MotorCalibration(**cal_data)
                for motor, cal_data in cal_dict.items()
            }
            print(f"  Loaded calibration from {cal_path}")
        else:
            print(f"  Warning: No calibration file found at {cal_path}")

    bus = FeetechMotorsBus(
        port=robot_port,
        motors={
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        },
        calibration=calibration,
    )
    bus.connect()
    return bus


def move_smooth(bus: FeetechMotorsBus, target: dict, steps: int = 50,
                delay_s: float = 0.02):
    """Smoothly interpolate from current position to target.

    Args:
        bus: Connected motor bus.
        target: Dict of joint_name -> target position. Joints not in target
                are held at their current position.
        steps: Number of interpolation steps.
        delay_s: Delay between steps in seconds.
    """
    current = bus.sync_read("Present_Position")

    for step in range(1, steps + 1):
        alpha = step / steps
        interpolated = {}
        for joint in JOINT_NAMES:
            if joint in target:
                start = current[joint]
                end = target[joint]
                interpolated[joint] = start + alpha * (end - start)
            else:
                interpolated[joint] = current[joint]
        bus.sync_write("Goal_Position", interpolated)
        time.sleep(delay_s)


def calibration_dir(robot_id: str) -> Path:
    """Return the workspace calibration directory for a robot."""
    return (
        Path.home() / ".cache" / "huggingface" / "lerobot"
        / "workspace_calibration" / robot_id
    )
