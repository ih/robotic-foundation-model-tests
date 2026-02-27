#!/usr/bin/env python
"""Recording script for single_action policy with per-episode task descriptions.

Wraps lerobot-record with:
1. Windows camera patches (DSHOW backend, sync read)
2. Policy registration for the single_action type
3. Monkey-patched record_loop for per-episode task descriptions and reset-phase
   secondary joint commands
4. Auto-calculated episode_time_s and reset_time_s
"""

import sys
import logging

# Patch camera backend to use DSHOW on Windows (before importing camera modules)
import platform
if platform.system() == "Windows":
    import cv2
    import lerobot.cameras.utils as cam_utils

    _original_get_cv2_backend = cam_utils.get_cv2_backend

    def _patched_get_cv2_backend():
        """Use DSHOW instead of MSMF on Windows for better camera compatibility."""
        if platform.system() == "Windows":
            return int(cv2.CAP_DSHOW)
        return _original_get_cv2_backend()

    cam_utils.get_cv2_backend = _patched_get_cv2_backend

    # Patch async_read to use synchronous read on Windows (threading issues with DSHOW)
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    _original_async_read = OpenCVCamera.async_read

    def _patched_async_read(self, timeout_ms: float = 200):
        """Use synchronous read on Windows to avoid threading issues with DSHOW."""
        return self.read()

    OpenCVCamera.async_read = _patched_async_read

# Import custom policy to trigger registration before lerobot parses args
from lerobot_policy_single_action.configuration_single_action import SingleActionConfig  # noqa: F401
from lerobot_policy_single_action.modeling_single_action import SingleActionPolicy  # noqa: F401
from lerobot_policy_single_action.processor_single_action import make_single_action_pre_post_processors  # noqa: F401

# Patch lerobot factory to recognize our policy type
import lerobot.policies.factory as _factory_mod

_original_get_policy_class = _factory_mod.get_policy_class
_original_make_pre_post_processors = _factory_mod.make_pre_post_processors


def _patched_get_policy_class(name):
    if name == "single_action":
        return SingleActionPolicy
    return _original_get_policy_class(name)


def _patched_make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs):
    if policy_cfg.type == "single_action":
        return make_single_action_pre_post_processors(policy_cfg)
    return _original_make_pre_post_processors(policy_cfg, pretrained_path=pretrained_path, **kwargs)


_factory_mod.get_policy_class = _patched_get_policy_class
_factory_mod.make_pre_post_processors = _patched_make_pre_post_processors

# Bypass sanity_check_dataset_name — our policy is for data collection, not evaluation,
# so the dataset name shouldn't require the "eval_" prefix.
import lerobot.utils.control_utils as _control_utils
_control_utils.sanity_check_dataset_name = lambda *args, **kwargs: None

# Patch lerobot record_loop for per-episode task descriptions and reset phase
import time as _time_module
import lerobot.scripts.lerobot_record as _record_mod
_record_mod.sanity_check_dataset_name = lambda *args, **kwargs: None

_original_record_loop = _record_mod.record_loop
_reset_state = {}


def _patched_record_loop(*args, **kwargs):
    """Patch record_loop for single_action policy.

    Two functions:
    1. During episode recording: replace single_task with the policy's
       current_task_description so each episode gets a unique task.
    2. During reset phase (no policy/teleop/dataset): call policy.reset()
       to pick new joints/direction, command secondary servo, and sleep.
    """
    policy = kwargs.get('policy')
    teleop = kwargs.get('teleop')
    dataset = kwargs.get('dataset')
    control_time_s = kwargs.get('control_time_s', 0)

    # Episode recording call
    if policy is not None:
        # On the very first episode, reset() hasn't been called yet (no reset phase
        # before episode 0). Call it here so the policy picks its first action.
        # For subsequent episodes, the reset-phase patch already called reset() and
        # set _secondary_target_locked=True, so this reset() will be a no-op.
        if 'policy' not in _reset_state:
            policy.reset()
            policy._secondary_target_locked = True
        _reset_state['policy'] = policy

        # Override single_task with per-episode task description.
        # This runs after our reset(), so current_task_description is valid.
        # record_loop will call reset() again internally, but _secondary_target_locked
        # makes that a no-op (preserves joint, direction, secondary target).
        if hasattr(policy, 'current_task_description'):
            task = policy.current_task_description
            kwargs['single_task'] = task
            logging.info(f"Episode task: {task}")

    # Reset phase (no policy, no teleop, no dataset)
    if policy is None and teleop is None and dataset is None and control_time_s > 0:
        stored_policy = _reset_state.get('policy')
        robot = kwargs.get('robot')

        if stored_policy is not None and hasattr(stored_policy, 'get_reset_motor_targets'):
            stored_policy.reset()
            motor_targets = stored_policy.get_reset_motor_targets()

            if motor_targets and robot is not None and hasattr(robot, 'bus'):
                try:
                    robot.bus.sync_write("Goal_Position", motor_targets)
                    logging.info(
                        f"Reset: commanding {motor_targets}, "
                        f"waiting {control_time_s:.1f}s to settle"
                    )
                except Exception as e:
                    logging.warning(f"Reset servo command failed: {e}")

            # Lock target so next episode's reset() preserves it
            stored_policy._secondary_target_locked = True

            # Log upcoming episode info
            if hasattr(stored_policy, 'current_task_description'):
                logging.info(f"Next episode: {stored_policy.current_task_description}")
        else:
            logging.info(f"Reset phase: waiting {control_time_s:.1f}s...")

        _time_module.sleep(control_time_s)
        return

    return _original_record_loop(*args, **kwargs)


_record_mod.record_loop = _patched_record_loop

# Now import record function
from lerobot.scripts.lerobot_record import record


def parse_arg(name: str) -> str | None:
    """Parse a command line argument value."""
    for i, arg in enumerate(sys.argv):
        if arg.startswith(f"--{name}="):
            return arg.split("=", 1)[1]
        elif arg == f"--{name}" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def check_and_clean_dataset_cache():
    """Check if dataset cache exists and prompt user to remove it."""
    import shutil
    import time
    from pathlib import Path

    repo_id = parse_arg("dataset.repo_id")
    if not repo_id:
        return

    cache_base = Path.home() / ".cache" / "huggingface" / "lerobot"
    cache_path = cache_base / repo_id

    if cache_path.exists():
        print(f"\nDataset cache already exists at:\n  {cache_path}\n")
        while True:
            response = input("Remove existing cache? [y/n]: ").strip().lower()
            if response in ("y", "yes"):
                shutil.rmtree(cache_path)
                for _ in range(50):
                    if not cache_path.exists():
                        break
                    time.sleep(0.1)
                if cache_path.exists():
                    print(f"ERROR: Failed to remove cache at {cache_path}")
                    print("Please close any programs using this directory and try again.")
                    sys.exit(1)
                print(f"Removed: {cache_path}\n")
                break
            elif response in ("n", "no"):
                print("Aborting. Please use a different --dataset.repo_id or remove the cache manually.")
                sys.exit(1)
            else:
                print("Please enter 'y' or 'n'")


def inject_episode_time():
    """Auto-calculate episode_time_s from action_duration + buffer."""
    if parse_arg("dataset.episode_time_s") is not None:
        return

    action_duration_str = parse_arg("policy.action_duration")
    action_duration = float(action_duration_str) if action_duration_str else 1.0

    episode_time = action_duration + 3.0
    sys.argv.append(f"--dataset.episode_time_s={episode_time}")
    print(f"Auto-calculated episode time: {episode_time:.1f}s "
          f"({action_duration}s action + 3s buffer)")


def inject_reset_time():
    """Auto-inject reset_time_s for secondary joint settling between episodes."""
    if parse_arg("dataset.reset_time_s") is not None:
        return

    action_duration_str = parse_arg("policy.action_duration")
    action_duration = float(action_duration_str) if action_duration_str else 1.0

    reset_time = max(3.0, action_duration * 3)
    sys.argv.append(f"--dataset.reset_time_s={reset_time}")
    print(f"Auto-set reset_time_s={reset_time:.1f}s for secondary joint settling")


def _create_motor_bus(robot_port: str, robot_id: str = None):
    """Create a FeetechMotorsBus with SO-101 motor configuration."""
    import json
    from pathlib import Path
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors import Motor, MotorNormMode, MotorCalibration

    calibration = None
    if robot_id:
        cal_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so101_follower"
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


def capture_starting_position(robot_port: str, robot_id: str = None) -> dict:
    """Capture all motor positions before recording starts."""
    try:
        bus = _create_motor_bus(robot_port, robot_id)
        positions = bus.sync_read("Present_Position")
        bus.disconnect()
        print(f"  Captured starting positions: {positions}")
        return positions
    except Exception as e:
        print(f"  Warning: Could not capture starting position: {e}")
        return {}


def return_arm_to_start(robot_port: str, starting_positions: dict,
                        robot_id: str = None):
    """Move the arm back to its starting position."""
    import time

    try:
        print("\nReturning arm to starting position...")
        bus = _create_motor_bus(robot_port, robot_id)

        current_positions = bus.sync_read("Present_Position")

        steps = 50
        for step in range(steps + 1):
            alpha = step / steps
            target = {}
            for joint_name, start_pos in starting_positions.items():
                current_pos = current_positions[joint_name]
                target[joint_name] = current_pos + alpha * (start_pos - current_pos)

            bus.sync_write("Goal_Position", target)
            time.sleep(0.02)

        print("Arm returned to starting position.")
        bus.disconnect()

    except Exception as e:
        print(f"Warning: Could not return arm to start position: {e}")


def print_command_line():
    """Print the full command for debugging."""
    args = ["lerobot-record"] + sys.argv[1:]
    command = " ".join(args)
    print("\n=== Executing lerobot-record ===")
    print(command)
    print("================================\n")


if __name__ == "__main__":
    check_and_clean_dataset_cache()

    inject_episode_time()
    inject_reset_time()

    # Inject a placeholder single_task if not provided (required by DatasetRecordConfig,
    # but overridden per-episode by the patched record_loop)
    if parse_arg("dataset.single_task") is None:
        sys.argv.append("--dataset.single_task=single action")


    # Get robot port and ID
    robot_port = parse_arg("robot.port")
    robot_id = parse_arg("robot.id")

    # Capture starting position
    starting_positions = {}
    if robot_port:
        starting_positions = capture_starting_position(robot_port, robot_id)

    print_command_line()

    # Run the recording
    try:
        record()
    finally:
        if robot_port and starting_positions:
            return_arm_to_start(robot_port, starting_positions, robot_id)
