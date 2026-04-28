"""Interactive workspace calibration for the SO-101 object relocator.

Teaches reachable positions across the desk by physically moving the arm
(torque disabled) and recording joint angles + camera pixel coordinates.
The robot pushes objects rather than grasping, so no gripper calibration
is needed.

Usage:
    python -m object_relocator.calibrate --port COM3 --robot-id my_so101_follower --base-camera 0
"""

import argparse
import json
import platform

import cv2
import numpy as np

from .motor_utils import (
    JOINT_NAMES, create_motor_bus, calibration_dir,
)

# Global for mouse callback
_clicked_point = None


def _mouse_callback(event, x, y, flags, param):
    global _clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicked_point = (x, y)


def _read_positions(bus) -> dict:
    """Read current joint positions as a plain dict."""
    positions = bus.sync_read("Present_Position")
    return {name: float(positions[name]) for name in JOINT_NAMES}


def _wait_enter(prompt: str):
    """Print prompt and wait for Enter."""
    input(f"\n  {prompt}\n  Press Enter when ready...")


def _capture_from_camera(camera_index: int) -> np.ndarray | None:
    """Capture a single frame from camera."""
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        print(f"  Error: Could not open camera {camera_index}")
        return None
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  Error: Could not read frame from camera {camera_index}")
        return None
    return frame


def _get_click_position(frame: np.ndarray, window_name: str) -> tuple:
    """Show frame and wait for user to click a point. Returns (x, y)."""
    global _clicked_point
    _clicked_point = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _mouse_callback)
    cv2.imshow(window_name, frame)
    print("  Click on the object position in the image window...")

    while _clicked_point is None:
        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # Escape
            cv2.destroyWindow(window_name)
            return None

    # Draw the clicked point
    display = frame.copy()
    cv2.circle(display, _clicked_point, 8, (0, 0, 255), -1)
    cv2.circle(display, _clicked_point, 12, (0, 255, 0), 2)
    cv2.imshow(window_name, display)
    cv2.waitKey(500)
    cv2.destroyWindow(window_name)

    point = _clicked_point
    _clicked_point = None
    return point


def run_calibration(port: str, robot_id: str, base_camera: int,
                    wrist_camera: int = None):
    """Run the interactive calibration workflow."""
    print("\n=== SO-101 Workspace Calibration (Push Mode) ===\n")

    cal_path = calibration_dir(robot_id)
    cal_path.mkdir(parents=True, exist_ok=True)
    print(f"  Calibration will be saved to: {cal_path}\n")

    bus = create_motor_bus(port, robot_id)
    calibration = {
        "base_camera_index": base_camera,
        "wrist_camera_index": wrist_camera,
        "interpolation_steps": 50,
        "step_delay_ms": 20,
        "push_settle_ms": 300,
        "detection_threshold": 30,
        "min_object_area": 500,
    }

    try:
        # --- Step 1: Home position ---
        print("=== Step 1: Home Position ===")
        print("  CAUTION: All torque will be disabled. Support the arm!")
        _wait_enter("Hold the arm, then move it to a safe HOME position above the workspace.")
        bus.disable_torque()
        _wait_enter("Position the arm at HOME (safe position above desk).")
        calibration["home"] = _read_positions(bus)
        bus.enable_torque()
        print(f"  Home: {calibration['home']}")

        # --- Step 2: Approach offset ---
        print("\n=== Step 2: Approach Offset ===")
        print("  Teach the height difference between push height and approach height.")
        print("  First, move the arm to PUSH height (where it contacts the object).")
        _wait_enter("Hold the arm, then move it to push height on the desk surface.")
        bus.disable_torque()
        _wait_enter("Position the arm at PUSH height (touching the desk/object level).")
        push_ref = _read_positions(bus)
        bus.enable_torque()
        print(f"  Push reference: {push_ref}")

        print("  Now lift the arm to APPROACH height (above the desk, clear of objects).")
        _wait_enter("Hold the arm, then lift it to approach height.")
        bus.disable_torque()
        _wait_enter("Position the arm at APPROACH height (above desk).")
        approach_ref = _read_positions(bus)
        bus.enable_torque()
        print(f"  Approach reference: {approach_ref}")

        approach_offset = {
            name: approach_ref[name] - push_ref[name]
            for name in JOINT_NAMES
        }
        calibration["approach_offset"] = approach_offset
        print(f"  Computed approach offset: {approach_offset}")

        # --- Step 3: Teach desk positions ---
        print("\n=== Step 3: Teach Desk Positions ===")
        print("  Teach reachable positions across your desk.")
        print("  For each position:")
        print("    1. Move the arm to push height at a desk location")
        print("    2. Place your object at the arm's position")
        print("    3. Click the object in the camera view")
        print("  Type 'q' when done teaching positions.\n")

        positions_list = []
        pos_index = 0

        while True:
            response = input(f"  Teach position {pos_index}? [Enter=yes, q=done]: ").strip()
            if response.lower() == 'q':
                if len(positions_list) < 2:
                    print("  Need at least 2 positions. Keep teaching.")
                    continue
                break

            # Teach push position
            print(f"\n  --- Position {pos_index} ---")
            print("  CAUTION: Torque will be disabled. Support the arm!")
            _wait_enter("Move the arm to PUSH height at this desk location.")
            bus.disable_torque()
            _wait_enter("Arm at push height on desk.")
            push_joints = _read_positions(bus)
            bus.enable_torque()

            # Compute approach position using offset
            approach_joints = {
                name: push_joints[name] + approach_offset[name]
                for name in JOINT_NAMES
            }

            print(f"  Push:     {push_joints}")
            print(f"  Approach: {approach_joints}")

            # Capture camera image and get click
            _wait_enter("Place your object at the arm's position, then step back.")
            frame = _capture_from_camera(base_camera)
            if frame is None:
                print("  Skipping position (camera error).")
                continue

            click = _get_click_position(frame, f"Position {pos_index} - Click object center")
            if click is None:
                print("  Skipping position (no click).")
                continue

            positions_list.append({
                "name": f"pos_{pos_index}",
                "pixel_x": click[0],
                "pixel_y": click[1],
                "approach": approach_joints,
                "push": push_joints,
            })
            print(f"  Saved position {pos_index} at pixel ({click[0]}, {click[1]})")
            pos_index += 1

        calibration["positions"] = positions_list
        print(f"\n  Taught {len(positions_list)} positions.")

        # --- Step 4: Reference images ---
        print("\n=== Step 4: Reference Images ===")
        _wait_enter("Clear ALL objects from the desk for reference image capture.")

        ref_base = _capture_from_camera(base_camera)
        if ref_base is not None:
            ref_path = cal_path / "reference_base.png"
            cv2.imwrite(str(ref_path), ref_base)
            print(f"  Saved base camera reference: {ref_path}")
        else:
            print("  Warning: Could not capture base camera reference image.")

        if wrist_camera is not None:
            ref_wrist = _capture_from_camera(wrist_camera)
            if ref_wrist is not None:
                ref_path = cal_path / "reference_wrist.png"
                cv2.imwrite(str(ref_path), ref_wrist)
                print(f"  Saved wrist camera reference: {ref_path}")

        # --- Step 5: Save ---
        json_path = cal_path / "calibration.json"
        with open(json_path, "w") as f:
            json.dump(calibration, f, indent=2)
        print(f"\n  Calibration saved to: {json_path}")
        print(f"  Total positions: {len(positions_list)}")
        print("\n=== Calibration Complete ===\n")

    finally:
        # Always re-enable torque on exit
        try:
            bus.enable_torque()
        except Exception:
            pass
        bus.disconnect()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate SO-101 workspace for object relocation (push mode)"
    )
    parser.add_argument("--port", required=True, help="Serial port (e.g., COM3)")
    parser.add_argument("--robot-id", required=True, help="Robot ID for calibration lookup")
    parser.add_argument("--base-camera", type=int, default=0,
                        help="Base camera index (default: 0)")
    parser.add_argument("--wrist-camera", type=int, default=None,
                        help="Wrist camera index (optional)")
    args = parser.parse_args()

    run_calibration(args.port, args.robot_id, args.base_camera, args.wrist_camera)


if __name__ == "__main__":
    main()
