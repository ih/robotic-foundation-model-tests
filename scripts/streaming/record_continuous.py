#!/usr/bin/env python
"""Continuous-stream recorder for SO-101 single-action exploration.

Records back-to-back actions into a single LeRobot v3.0 dataset using
the official `LeRobotDataset.create(streaming_encoding=True)` API. Each
"logical episode" is one `action_duration`-second window in which the
robot moves one joint by `position_delta`. Episodes are saved through
`dataset.save_episode()` so frames flow into chunked parquet/MP4
storage and the resulting dataset is uploadable to the HuggingFace Hub
via `push_to_hub` and consumable by `canvas-world-model/create_dataset.py`
without changes.

Why this exists
---------------
The legacy `run_single_action_record.py` records each episode as a
separate lerobot-record invocation with `start_buffer + action +
end_buffer + reset_time_s` ~= 7.5s of scripted dead time per episode,
plus per-episode `verify_reset_position` retries and per-episode video
finalize overhead. Measured wall time: ~21s/episode. For a 20-episode
EXPLORE burst that's ~7-8 minutes, of which only ~3s/episode is the
actual moving-arm window we care about. This recorder removes the
per-episode overhead by streaming actions back-to-back into one
recording session.

CLI examples
------------
Smoke test, 5 actions, dry layout sanity:
    python -m scripts.streaming.record_continuous \\
        --robot-port=COM3 \\
        --base-camera=1 --wrist-camera=0 \\
        --num-actions=5 --action-duration=1.0 --fps=10 \\
        --output-repo-id=local/streaming-smoke

Two-joint pooled run, 60 actions, push to HF:
    python -m scripts.streaming.record_continuous \\
        --robot-port=COM3 \\
        --base-camera=1 --wrist-camera=0 \\
        --num-actions=60 --action-duration=1.0 --fps=10 \\
        --joints shoulder_pan.pos elbow_flex.pos \\
        --joint-range shoulder_pan.pos -60 60 \\
        --joint-range elbow_flex.pos 50 90 \\
        --output-repo-id=irvinh/streaming-shoulder-elbow-60 \\
        --push-to-hub
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Windows DSHOW camera patches.
# These mirror the patches in `run_single_action_record.py` lines 22-61 and
# must be applied BEFORE any LeRobot import path that touches OpenCVCamera.
# Lerobot's default OpenCVCamera spawns a background reader thread; on
# Windows DSHOW that thread fights with sync grabs and frames disappear.
# Disable the thread, swap reads to direct synchronous capture.
# ---------------------------------------------------------------------------
def _apply_windows_camera_patches() -> None:
    if platform.system() != "Windows":
        return

    import cv2  # noqa: F401  (loaded by lerobot anyway)
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    OpenCVCamera._start_read_thread = lambda self: None

    def _sync_capture(self):
        raw = self._read_from_hardware()
        processed = self._postprocess_image(raw)
        with self.frame_lock:
            self.latest_frame = processed
            self.latest_timestamp = time.perf_counter()
        return processed

    def _patched_async_read(self, timeout_ms: float = 200):
        return self._sync_capture()

    def _patched_read(self, color_mode=None):
        return self._sync_capture()

    def _patched_read_latest(self, max_age_ms: int = 500):
        return self._sync_capture()

    OpenCVCamera._sync_capture = _sync_capture
    OpenCVCamera.async_read = _patched_async_read
    OpenCVCamera.read = _patched_read
    OpenCVCamera.read_latest = _patched_read_latest


# Apply patches at import time so any subsequent OpenCVCamera use is safe.
_apply_windows_camera_patches()


# ---------------------------------------------------------------------------
# Late imports: must come after the camera patches above.
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402

from .sequencer import ActionSequencer, ActionTarget  # noqa: E402

# SO-101 joint constants — keep in sync with single_action policy.
SO101_JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
JOINT_INDEX = {j: i for i, j in enumerate(SO101_JOINTS)}

# Discrete action codes: must match SingleActionPolicy.select_action mapping
# in `modeling_single_action.py` lines 481-488 so canvas-world-model's
# `create_dataset.py` interprets streaming-recorded datasets identically
# to legacy episodic ones.
ACTION_HOLD = 0
ACTION_POSITIVE = 1
ACTION_NEGATIVE = 2
ACTION_NONE = 3

DIRECTION_TO_DISCRETE = {
    "positive": ACTION_POSITIVE,
    "negative": ACTION_NEGATIVE,
    "none": ACTION_NONE,
}


# ---------------------------------------------------------------------------
# Hardware adapters
# ---------------------------------------------------------------------------
@dataclass
class HardwareConfig:
    port: str
    robot_id: str
    base_camera_index: int
    wrist_camera_index: int
    base_camera_name: str
    wrist_camera_name: str
    camera_width: int
    camera_height: int
    camera_fps: int
    camera_warmup_s: float = 2.0


class HardwareSession:
    """Owns motor bus + two cameras for the duration of one recording.

    Mirrors the connection logic in `canvas-robot-control/control/robot_interface.py`
    so calibration files and DSHOW handling match the rest of the stack.
    """

    def __init__(self, hw_cfg: HardwareConfig):
        self.cfg = hw_cfg
        self.bus = None
        self.base_camera = None
        self.wrist_camera = None

    def connect(self) -> None:
        self._connect_motors()
        self._connect_cameras()

    def disconnect(self) -> None:
        # Cameras first — see CLAUDE.md memory: never taskkill /F a camera
        # owner, always disconnect cleanly so the DSHOW handle is released.
        if self.base_camera is not None:
            try:
                self.base_camera.disconnect()
            except Exception as e:
                logging.warning(f"base camera disconnect failed: {e}")
            self.base_camera = None
        if self.wrist_camera is not None:
            try:
                self.wrist_camera.disconnect()
            except Exception as e:
                logging.warning(f"wrist camera disconnect failed: {e}")
            self.wrist_camera = None
        if self.bus is not None:
            try:
                self.bus.disconnect()
            except Exception as e:
                logging.warning(f"motor bus disconnect failed: {e}")
            self.bus = None

    def _connect_motors(self) -> None:
        from lerobot.motors.feetech import FeetechMotorsBus
        from lerobot.motors import Motor, MotorNormMode, MotorCalibration

        cal_dir = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "lerobot"
            / "calibration"
            / "robots"
            / "so101_follower"
        )
        cal_path = cal_dir / f"{self.cfg.robot_id}.json"
        calibration = None
        if cal_path.is_file():
            with open(cal_path) as f:
                cal_dict = json.load(f)
            calibration = {
                motor: MotorCalibration(**cal_data)
                for motor, cal_data in cal_dict.items()
            }
        else:
            logging.warning(f"calibration file not found at {cal_path} — proceeding uncalibrated")

        self.bus = FeetechMotorsBus(
            port=self.cfg.port,
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
        self.bus.connect()

    def _connect_cameras(self) -> None:
        import cv2
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera, OpenCVCameraConfig
        from lerobot.cameras.configs import Cv2Backends

        def _make(idx: int) -> OpenCVCamera:
            cfg = OpenCVCameraConfig(
                index_or_path=idx,
                width=self.cfg.camera_width,
                height=self.cfg.camera_height,
                fps=self.cfg.camera_fps,
                rotation=180,
                backend=Cv2Backends.DSHOW,
                warmup_s=self.cfg.camera_warmup_s,
            )
            cam = OpenCVCamera(cfg)
            cam.connect()
            if hasattr(cam, "videocapture"):
                # Lock white balance — auto WB makes red look blue with these
                # USB cameras under desk lighting.
                cam.videocapture.set(cv2.CAP_PROP_AUTO_WB, 0)
                cam.videocapture.set(cv2.CAP_PROP_WB_TEMPERATURE, 6500)
            return cam

        self.base_camera = _make(self.cfg.base_camera_index)
        self.wrist_camera = _make(self.cfg.wrist_camera_index)

    def read_motor_positions(self) -> dict[str, float]:
        return self.bus.sync_read("Present_Position")

    def read_synced_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """Capture base+wrist frames from one camera tick.

        DSHOW multi-camera correctness: grab() both first, then retrieve()
        each. Sequential read() of two cameras can return the first
        camera's frame for the second camera. See robot_interface.py:124-132
        for the same pattern.
        """
        for _ in range(3):
            self.base_camera.videocapture.grab()
            self.wrist_camera.videocapture.grab()
        _, base_raw = self.base_camera.videocapture.retrieve()
        _, wrist_raw = self.wrist_camera.videocapture.retrieve()
        base_rgb = self.base_camera._postprocess_image(base_raw)
        wrist_rgb = self.wrist_camera._postprocess_image(wrist_raw)
        return base_rgb, wrist_rgb

    def send_goal(self, goal_positions: dict[str, float]) -> None:
        self.bus.sync_write("Goal_Position", goal_positions)


# ---------------------------------------------------------------------------
# Streaming recorder
# ---------------------------------------------------------------------------
@dataclass
class StreamingConfig:
    repo_id: str
    output_root: Optional[Path]   # None -> default HF_LEROBOT_HOME
    fps: int
    action_duration: float
    settle_duration: float        # extra hold time after each action_duration
    verify_every: int             # 0 disables verification checkpoints
    push_to_hub: bool
    private_hub: bool
    starting_positions: Optional[dict[str, float]]


def _build_features(
    image_height: int,
    image_width: int,
    base_name: str,
    wrist_name: str,
) -> dict:
    """Match `meta/info.json` schema produced by the legacy recorder.

    Specifically: action + observation.state are 6-vector float32 with
    SO-101 joint names; cameras are video-typed HxWx3.

    Shapes are tuples (not lists) because `validate_frame` does a strict
    `actual_shape != expected_shape` comparison against `value.shape`,
    which is always a tuple. JSON serialization to `meta/info.json` will
    flatten tuples to lists at write time.
    """
    image_shape = (image_height, image_width, 3)
    return {
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": list(SO101_JOINTS),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": list(SO101_JOINTS),
        },
        f"observation.images.{base_name}": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        f"observation.images.{wrist_name}": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
    }


def _write_action_log_header(
    log_path: Path,
    target: ActionTarget,
    action_duration: float,
    position_delta: float,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "type": "header",
        "joint_name": target.joint,
        "action_duration": action_duration,
        "position_delta": position_delta,
        "vary_target_joint": True,
        "random_seed": None,
        "direction": target.direction,
        "diversity_joints": [],
    }
    with open(log_path, "w") as f:
        f.write(json.dumps(header) + "\n")


def _append_action_log_entry(
    log_path: Path,
    timestamp: float,
    discrete_action: int,
    frame_index: int,
) -> None:
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "type": "action",
            "timestamp": timestamp,
            "discrete_action": discrete_action,
            "frame_index": frame_index,
        }) + "\n")


def run_streaming_session(
    hw_cfg: HardwareConfig,
    seq_cfg: dict,
    stream_cfg: StreamingConfig,
    num_actions: int,
) -> Path:
    """Record `num_actions` back-to-back actions into one LeRobot v3.0 dataset.

    Returns the on-disk root path of the saved dataset.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    sequencer = ActionSequencer(
        joints=seq_cfg["joints"],
        joint_ranges=seq_cfg["joint_ranges"],
        joint_indices=JOINT_INDEX,
        position_delta=seq_cfg["position_delta"],
        vary_target=seq_cfg["vary_target"],
        seed=seq_cfg.get("seed"),
    )

    hw = HardwareSession(hw_cfg)
    hw.connect()
    logging.info(
        f"Hardware connected: bus={hw_cfg.port} robot={hw_cfg.robot_id} "
        f"cameras=base:{hw_cfg.base_camera_index},wrist:{hw_cfg.wrist_camera_index} "
        f"resolution={hw_cfg.camera_width}x{hw_cfg.camera_height}@{hw_cfg.camera_fps}fps"
    )

    features = _build_features(
        image_height=hw_cfg.camera_height,
        image_width=hw_cfg.camera_width,
        base_name=hw_cfg.base_camera_name,
        wrist_name=hw_cfg.wrist_camera_name,
    )

    dataset = LeRobotDataset.create(
        repo_id=stream_cfg.repo_id,
        fps=stream_cfg.fps,
        features=features,
        root=stream_cfg.output_root,
        robot_type="so_follower",
        use_videos=True,
        # streaming_encoding=True keeps a single ffmpeg encoder open for
        # the whole session; per-episode save_episode() is then cheap
        # (no encoder spin-up/finalize per logical episode).
        streaming_encoding=True,
        batch_encoding_size=1,
    )
    dataset_root = Path(dataset.root)
    log_dir = dataset_root / "meta" / "discrete_action_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    base_key = f"observation.images.{hw_cfg.base_camera_name}"
    wrist_key = f"observation.images.{hw_cfg.wrist_camera_name}"

    frames_per_action = max(1, int(round(
        (stream_cfg.action_duration + stream_cfg.settle_duration) * stream_cfg.fps
    )))
    frame_period = 1.0 / float(stream_cfg.fps)

    # If the caller pre-specifies a home, snap there before the first
    # action so the run starts from a known pose. Otherwise just read
    # the live state.
    if stream_cfg.starting_positions:
        try:
            hw.send_goal(stream_cfg.starting_positions)
            time.sleep(1.5)  # one-time settle, not per-episode
        except Exception as e:
            logging.warning(f"starting-positions snap failed: {e}")

    session_start = time.perf_counter()
    for action_idx in range(num_actions):
        motor_positions = hw.read_motor_positions()
        target = sequencer.next_action(motor_positions)

        # Build the full goal vector: hold all joints at current pos
        # except the active one, which moves to target.
        goal = dict(motor_positions)
        goal[target.motor_name] = target.target_pos
        hw.send_goal(goal)
        action_send_t = time.perf_counter()

        # Per-episode discrete action log file (matches legacy schema so
        # canvas-world-model's load_decision_bearing_logs picks it up).
        log_path = log_dir / f"episode_{action_idx:06d}.jsonl"
        _write_action_log_header(
            log_path, target, stream_cfg.action_duration, sequencer.position_delta
        )
        action_discrete = DIRECTION_TO_DISCRETE.get(target.direction, ACTION_HOLD)

        ep_start = time.perf_counter()
        for f_idx in range(frames_per_action):
            # Pace at fps. We don't sleep before the first frame so the
            # action's effect is captured as quickly as possible.
            if f_idx > 0:
                next_t = ep_start + f_idx * frame_period
                slack = next_t - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)

            base_frame, wrist_frame = hw.read_synced_frames()
            cur_state = hw.read_motor_positions()
            state_vec = np.array(
                [cur_state.get(j.replace(".pos", ""), 0.0) for j in SO101_JOINTS],
                dtype=np.float32,
            )
            action_vec = np.array(
                [goal.get(j.replace(".pos", ""), state_vec[i])
                 for i, j in enumerate(SO101_JOINTS)],
                dtype=np.float32,
            )

            frame_dict = {
                "action": action_vec,
                "observation.state": state_vec,
                base_key: base_frame,
                wrist_key: wrist_frame,
                "task": target.task_description,
            }
            dataset.add_frame(frame_dict)

            # Discrete action: only the very first frame in this logical
            # episode carries the actual decision; the rest are HOLD.
            decision = action_discrete if f_idx == 0 else ACTION_HOLD
            _append_action_log_entry(
                log_path,
                timestamp=time.time(),
                discrete_action=decision,
                frame_index=f_idx,
            )

        dataset.save_episode()
        ep_wall = time.perf_counter() - ep_start
        logging.info(
            f"action {action_idx + 1}/{num_actions} "
            f"joint={target.motor_name} dir={target.direction} "
            f"target={target.target_pos:.2f} frames={frames_per_action} "
            f"wall={ep_wall:.2f}s"
        )

        # Periodic verification checkpoint. Bus chatter only — no extra
        # camera reads or motor moves. Cheap drift detection.
        if (
            stream_cfg.verify_every > 0
            and (action_idx + 1) % stream_cfg.verify_every == 0
        ):
            try:
                actual = hw.read_motor_positions()
                err = abs(actual.get(target.motor_name, 0.0) - target.target_pos)
                logging.info(
                    f"verify@{action_idx + 1}: {target.motor_name} cmd={target.target_pos:.2f} "
                    f"actual={actual.get(target.motor_name, 0.0):.2f} err={err:.2f}"
                )
            except Exception as e:
                logging.warning(f"verify@{action_idx + 1} failed: {e}")

    total_wall = time.perf_counter() - session_start
    logging.info(
        f"streaming done: {num_actions} actions in {total_wall:.1f}s "
        f"= {total_wall / max(1, num_actions):.2f}s/action"
    )

    # Disconnect hardware before any push-to-hub work — the upload can
    # take minutes and we don't want to hold the cameras hostage.
    hw.disconnect()

    if stream_cfg.push_to_hub:
        logging.info(f"pushing to hub: {stream_cfg.repo_id} (private={stream_cfg.private_hub})")
        try:
            dataset.push_to_hub(private=stream_cfg.private_hub)
            logging.info(f"push complete: https://huggingface.co/datasets/{stream_cfg.repo_id}")
        except Exception as e:
            logging.error(f"push_to_hub failed: {e}")
            raise

    return dataset_root


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_joint_ranges(
    pairs: Optional[list[list[str]]],
    joints: list[str],
) -> dict[str, tuple[float, float]]:
    """Defaults applied for any joint missing from --joint-range."""
    defaults = {
        "shoulder_pan.pos": (-60.0, 60.0),
        "shoulder_lift.pos": (-100.0, 40.0),
        "elbow_flex.pos": (50.0, 90.0),
        "wrist_flex.pos": (-50.0, 50.0),
        "wrist_roll.pos": (-100.0, 100.0),
        "gripper.pos": (0.0, 100.0),
    }
    out = {j: defaults[j] for j in joints if j in defaults}
    if pairs:
        for triple in pairs:
            if len(triple) != 3:
                raise ValueError(f"--joint-range expects 3 args, got {triple}")
            joint, lo, hi = triple
            if joint not in joints:
                raise ValueError(f"--joint-range {joint!r} not in --joints {joints}")
            out[joint] = (float(lo), float(hi))
    for j in joints:
        if j not in out:
            raise ValueError(f"no range for joint {j!r}; pass --joint-range {j} <lo> <hi>")
    return out


def _parse_starting_positions(s: Optional[str]) -> Optional[dict[str, float]]:
    if not s:
        return None
    raw = json.loads(s)
    return {str(k): float(v) for k, v in raw.items()}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Hardware
    p.add_argument("--robot-port", default="COM3")
    p.add_argument("--robot-id", default="my_so101_follower")
    p.add_argument("--base-camera", type=int, default=1)
    p.add_argument("--wrist-camera", type=int, default=0)
    p.add_argument("--base-camera-name", default="base_0_rgb",
                   help="Feature key suffix; matches legacy datasets.")
    p.add_argument("--wrist-camera-name", default="left_wrist_0_rgb")
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--camera-fps", type=int, default=10)

    # Recording cadence
    p.add_argument("--num-actions", type=int, required=True)
    p.add_argument("--action-duration", type=float, default=1.0,
                   help="Seconds the motor has to reach target per action.")
    p.add_argument("--settle-duration", type=float, default=0.0,
                   help="Extra hold time after action_duration. Default 0 "
                        "(back-to-back actions).")
    p.add_argument("--fps", type=int, default=10,
                   help="Dataset fps. Should match --camera-fps.")
    p.add_argument("--verify-every", type=int, default=10,
                   help="Run a motor sync_read drift check every N actions. "
                        "0 to disable.")

    # Sequencer
    p.add_argument("--joints", nargs="+", default=["shoulder_pan.pos"],
                   help="Joint pool for vary-target sampling.")
    p.add_argument("--joint-range", action="append", nargs=3, metavar=("JOINT", "LO", "HI"),
                   help="Per-joint clamp. Repeatable. Defaults applied for unspecified joints.")
    p.add_argument("--position-delta", type=float, default=10.0)
    p.add_argument("--no-vary-target", action="store_true",
                   help="Always use joints[0] instead of sampling.")
    p.add_argument("--seed", type=int, default=None)

    # Output / HF
    p.add_argument("--output-repo-id", required=True,
                   help="Used as both the on-disk subdir under HF_LEROBOT_HOME "
                        "and the destination repo for --push-to-hub.")
    p.add_argument("--output-root", default=None,
                   help="Override HF_LEROBOT_HOME. Defaults to the standard cache.")
    p.add_argument("--push-to-hub", action="store_true",
                   help="Upload to the HuggingFace Hub after the run completes. "
                        "Requires HF_TOKEN with write access in the env.")
    p.add_argument("--public-hub", action="store_true",
                   help="Make the pushed dataset public. Default is private.")

    # Initial pose
    p.add_argument("--starting-positions-json", default=None,
                   help="JSON dict of motor_name -> position for a one-shot "
                        "snap-to-home before action 0. Skipped if omitted.")

    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.fps != args.camera_fps:
        logging.warning(
            f"--fps={args.fps} differs from --camera-fps={args.camera_fps}; "
            "frame timestamps will be paced to --fps but the camera may not "
            "deliver fresh frames at that rate."
        )

    hw_cfg = HardwareConfig(
        port=args.robot_port,
        robot_id=args.robot_id,
        base_camera_index=args.base_camera,
        wrist_camera_index=args.wrist_camera,
        base_camera_name=args.base_camera_name,
        wrist_camera_name=args.wrist_camera_name,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
    )

    joints = list(args.joints)
    for j in joints:
        if j not in SO101_JOINTS:
            raise SystemExit(f"unknown joint {j!r}; expected one of {SO101_JOINTS}")
    joint_ranges = _parse_joint_ranges(args.joint_range, joints)

    seq_cfg = {
        "joints": joints,
        "joint_ranges": joint_ranges,
        "position_delta": args.position_delta,
        "vary_target": not args.no_vary_target and len(joints) > 1,
        "seed": args.seed,
    }

    stream_cfg = StreamingConfig(
        repo_id=args.output_repo_id,
        output_root=Path(args.output_root) if args.output_root else None,
        fps=args.fps,
        action_duration=args.action_duration,
        settle_duration=args.settle_duration,
        verify_every=args.verify_every,
        push_to_hub=args.push_to_hub,
        private_hub=not args.public_hub,
        starting_positions=_parse_starting_positions(args.starting_positions_json),
    )

    dataset_root = run_streaming_session(
        hw_cfg=hw_cfg,
        seq_cfg=seq_cfg,
        stream_cfg=stream_cfg,
        num_actions=args.num_actions,
    )
    print(f"\nstreaming session complete:\n  dataset_root: {dataset_root}\n  episodes: {args.num_actions}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
