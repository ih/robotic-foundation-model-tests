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
import types

# Block broken groot policy import (dataclass ordering bug in lerobot 0.5.0 + Python 3.12)
_groot_cfg = types.ModuleType("lerobot.policies.groot.configuration_groot")
_groot_cfg.GrootConfig = type("GrootConfig", (), {})
sys.modules["lerobot.policies.groot"] = types.ModuleType("lerobot.policies.groot")
sys.modules["lerobot.policies.groot.configuration_groot"] = _groot_cfg

# Windows camera patches (DSHOW backend is now set via camera config `backend: DSHOW`)
import platform
if platform.system() == "Windows":
    import cv2

    # In lerobot 0.5.0, both read() and async_read() rely on a background thread.
    # DSHOW on Windows has threading issues, so we disable the thread entirely and
    # make all reads synchronous via _read_from_hardware().
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    # Disable background read thread
    OpenCVCamera._start_read_thread = lambda self: None

    import time as _cam_time

    def _sync_capture(self):
        """Capture a frame synchronously and update internal state."""
        raw_frame = self._read_from_hardware()
        processed = self._postprocess_image(raw_frame)
        with self.frame_lock:
            self.latest_frame = processed
            self.latest_timestamp = _cam_time.perf_counter()
        return processed

    def _patched_async_read(self, timeout_ms: float = 200):
        """Direct synchronous read — avoids DSHOW threading issues on Windows."""
        return self._sync_capture()

    def _patched_read(self, color_mode=None):
        """Synchronous read bypassing thread-alive check for Windows DSHOW."""
        return self._sync_capture()

    def _patched_read_latest(self, max_age_ms: int = 500):
        """Synchronous read_latest — no background thread on Windows DSHOW."""
        return self._sync_capture()

    OpenCVCamera._sync_capture = _sync_capture
    OpenCVCamera.async_read = _patched_async_read
    OpenCVCamera.read = _patched_read
    OpenCVCamera.read_latest = _patched_read_latest

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

# Populated from the CLI flag --starting-positions-json=<json> before the
# recorder starts its first episode. When non-empty, the custom reset hook
# feeds this dict directly to policy.set_starting_positions() instead of
# reading live Present_Position — which is stale if the arm drooped under
# gravity between the caller's `hardware.disconnect()` and subprocess boot.
_STARTING_POSITIONS_OVERRIDE: dict = {}

# Populated from --probe-script-json=<JSON>: a list of [start_pos, direction]
# tuples consumed one per episode. Each reset pops the next entry and
# stamps `_forced_direction` + `_forced_primary_start` on the policy so
# the following reset() + get_reset_motor_targets() call uses exactly
# that direction and start position. Used by the learner's VERIFY path
# to drive error-weighted probes through the same recorder pipeline
# as EXPLORE.
_PROBE_SCRIPT_QUEUE: list = []


def _apply_next_probe_script_entry(policy) -> None:
    """Pop the next `(start_pos, direction)` tuple off the probe script
    queue (if any) and stamp it onto `policy` so the upcoming reset()
    uses that direction and forced primary-start position instead of
    random sampling. No-op if the queue is empty.
    """
    if not _PROBE_SCRIPT_QUEUE:
        return
    try:
        pos, direction = _PROBE_SCRIPT_QUEUE.pop(0)
    except (ValueError, IndexError):
        return
    try:
        pos = float(pos)
    except (TypeError, ValueError):
        return
    direction = str(direction)
    if direction not in ("positive", "negative", "none"):
        return
    # Cleared by reset()'s primary-target resample logic; we override
    # both the direction and the cached primary start on the policy
    # instance so get_reset_motor_targets() returns our pos.
    policy._forced_direction = direction
    policy._forced_primary_start = pos
    logging.info(
        f"probe_script: episode uses direction={direction} start_pos={pos}"
    )


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
        # set _diversity_targets_locked=True, so this reset() will be a no-op.
        if 'policy' not in _reset_state:
            _apply_next_probe_script_entry(policy)
            policy.reset()
            policy._diversity_targets_locked = True
            # Prefer a caller-provided override (set via --starting-positions-json
            # on the CLI) so the learner can force a known "home" pose even
            # after the motor bus was handed off between processes and the
            # arm drooped under gravity. Otherwise fall back to the live
            # Present_Position read from robot.bus.
            robot = kwargs.get('robot')
            override = _STARTING_POSITIONS_OVERRIDE
            if override:
                policy.set_starting_positions(dict(override))
                logging.info(f"Using override starting positions: {override}")
            elif robot is not None and hasattr(robot, 'bus'):
                try:
                    sp = robot.bus.sync_read("Present_Position")
                    policy.set_starting_positions(sp)
                    logging.info(f"Captured starting positions for drift correction: {sp}")
                except Exception as e:
                    logging.warning(f"Could not capture starting positions: {e}")
        _reset_state['policy'] = policy

        # Override single_task with per-episode task description.
        # This runs after our reset(), so current_task_description is valid.
        # record_loop will call reset() again internally, but _diversity_targets_locked
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
            _apply_next_probe_script_entry(stored_policy)
            stored_policy.reset()
            motor_targets = stored_policy.get_reset_motor_targets()

            if motor_targets and robot is not None and hasattr(robot, 'bus'):
                max_retries = getattr(stored_policy.config, 'max_reset_retries', 3)
                for attempt in range(max_retries + 1):
                    try:
                        robot.bus.sync_write("Goal_Position", motor_targets)
                        label = "Reset" if attempt == 0 else f"Reset retry {attempt}"
                        logging.info(
                            f"{label}: commanding {motor_targets}, "
                            f"waiting {control_time_s:.1f}s to settle"
                        )
                    except Exception as e:
                        logging.warning(f"Reset servo command failed: {e}")
                        break

                    _time_module.sleep(control_time_s)

                    # Verify positions reached
                    try:
                        actual = robot.bus.sync_read("Present_Position")
                        corrected = stored_policy.verify_reset_position(actual)
                        if not corrected:
                            logging.info("Reset positions verified OK")
                            break
                        else:
                            logging.warning(
                                f"Reset position error detected: {corrected}. "
                                f"Accepting actual positions."
                            )
                            # Targets already updated by verify_reset_position,
                            # re-read targets for next attempt
                            motor_targets = stored_policy.get_reset_motor_targets()
                            if attempt == max_retries:
                                logging.warning(
                                    f"Max reset retries ({max_retries}) reached, "
                                    f"proceeding with current positions"
                                )
                    except Exception as e:
                        logging.warning(f"Could not verify reset positions: {e}")
                        break

            # Lock target so next episode's reset() preserves it
            stored_policy._diversity_targets_locked = True

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
    """Auto-calculate episode_time_s from start_buffer + action_duration + end_buffer."""
    if parse_arg("dataset.episode_time_s") is not None:
        return

    action_duration_str = parse_arg("policy.action_duration")
    action_duration = float(action_duration_str) if action_duration_str else 1.0

    start_buffer_str = parse_arg("policy.start_buffer")
    start_buffer = float(start_buffer_str) if start_buffer_str else 1.0

    end_buffer_str = parse_arg("policy.end_buffer")
    end_buffer = float(end_buffer_str) if end_buffer_str else 1.0

    episode_time = start_buffer + action_duration + end_buffer
    sys.argv.append(f"--dataset.episode_time_s={episode_time}")
    print(f"Auto-calculated episode time: {episode_time:.1f}s "
          f"({start_buffer}s start + {action_duration}s action + {end_buffer}s end)")


def inject_reset_time():
    """Auto-inject reset_time_s for secondary joint settling between episodes."""
    if parse_arg("dataset.reset_time_s") is not None:
        return

    action_duration_str = parse_arg("policy.action_duration")
    action_duration = float(action_duration_str) if action_duration_str else 1.0

    reset_time = max(3.0, action_duration * 3)
    sys.argv.append(f"--dataset.reset_time_s={reset_time}")
    print(f"Auto-set reset_time_s={reset_time:.1f}s for secondary joint settling")


def inject_discrete_action_log_dir():
    """Inject --policy.discrete_action_log_dir so discrete actions are logged.

    Sets the log directory to meta/discrete_action_logs/ inside the HuggingFace
    dataset cache, matching where LeRobot stores dataset metadata.
    """
    if parse_arg("policy.discrete_action_log_dir") is not None:
        return

    repo_id = parse_arg("dataset.repo_id")
    if not repo_id:
        return

    from pathlib import Path
    cache_base = Path.home() / ".cache" / "huggingface" / "lerobot"
    log_dir = cache_base / repo_id / "meta" / "discrete_action_logs"

    sys.argv.append(f"--policy.discrete_action_log_dir={log_dir}")
    print(f"Discrete action logs: {log_dir}")


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
    inject_discrete_action_log_dir()

    # Inject a placeholder single_task if not provided (required by DatasetRecordConfig,
    # but overridden per-episode by the patched record_loop)
    if parse_arg("dataset.single_task") is None:
        sys.argv.append("--dataset.single_task=single action")


    # Get robot port and ID
    robot_port = parse_arg("robot.port")
    robot_id = parse_arg("robot.id")

    # Optional caller-provided override: a JSON dict of joint -> position.
    # When set, it (a) populates the module-global used by the reset hook
    # and (b) replaces the live-read baseline used by return_arm_to_start.
    override_raw = parse_arg("starting-positions-json")
    if override_raw is not None:
        import json as _json
        try:
            parsed = _json.loads(override_raw)
            if isinstance(parsed, dict):
                _STARTING_POSITIONS_OVERRIDE = {
                    str(k): float(v) for k, v in parsed.items()
                }
                # Strip the flag so lerobot-record's parser doesn't reject it.
                sys.argv = [
                    a for a in sys.argv
                    if not a.startswith("--starting-positions-json")
                ]
        except (ValueError, TypeError) as e:
            print(f"Warning: could not parse --starting-positions-json: {e}")

    # Probe-script queue: a JSON list of [start_pos, direction] tuples
    # consumed one per episode. Used by the learner's VERIFY path to
    # drive error-weighted probes through the recorder without going
    # through the normal random direction/start-position sampling.
    probe_raw = parse_arg("probe-script-json")
    if probe_raw is not None:
        import json as _json
        try:
            parsed = _json.loads(probe_raw)
            if isinstance(parsed, list):
                _PROBE_SCRIPT_QUEUE.clear()
                for entry in parsed:
                    if isinstance(entry, (list, tuple)) and len(entry) == 2:
                        _PROBE_SCRIPT_QUEUE.append((float(entry[0]), str(entry[1])))
                sys.argv = [
                    a for a in sys.argv
                    if not a.startswith("--probe-script-json")
                ]
                print(f"  probe script loaded: {len(_PROBE_SCRIPT_QUEUE)} entries")
        except (ValueError, TypeError) as e:
            print(f"Warning: could not parse --probe-script-json: {e}")

    # Capture starting position (or use the override for return-to-start).
    starting_positions = {}
    if _STARTING_POSITIONS_OVERRIDE:
        starting_positions = dict(_STARTING_POSITIONS_OVERRIDE)
        print(f"  Using override starting positions: {starting_positions}")
    elif robot_port:
        starting_positions = capture_starting_position(robot_port, robot_id)

    print_command_line()

    # Run the recording
    try:
        record()
    finally:
        if robot_port and starting_positions:
            return_arm_to_start(robot_port, starting_positions, robot_id)
