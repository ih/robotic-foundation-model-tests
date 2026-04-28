"""Vision-based object relocator for SO-101.

Detects an object on the desk via background subtraction, then pushes it
from its current location to a randomly selected calibrated position.
"""

import json
import math
import random
import time

import cv2
import numpy as np

from .motor_utils import (
    JOINT_NAMES, create_motor_bus, move_smooth, calibration_dir,
)
from .detection import detect_object, capture_frame


class ObjectRelocator:
    """Push-based object relocation with vision detection.

    Uses background subtraction to find objects on the desk, and calibrated
    joint-space waypoints to push objects between positions.

    Motion sequence for a push:
        home -> approach(object) -> push(object) -> push(target) -> approach(target) -> home
    """

    def __init__(self, robot_port: str, robot_id: str,
                 calibration_path: str = None):
        self.robot_port = robot_port
        self.robot_id = robot_id

        cal_dir = calibration_dir(robot_id)
        self._cal_dir = cal_dir

        if calibration_path:
            cal_file = calibration_path
        else:
            cal_file = str(cal_dir / "calibration.json")

        with open(cal_file) as f:
            self.cal = json.load(f)

        # Load reference image for background subtraction
        ref_path = cal_dir / "reference_base.png"
        if ref_path.exists():
            self.reference_image = cv2.imread(str(ref_path))
        else:
            raise FileNotFoundError(
                f"Reference image not found at {ref_path}. "
                "Run calibration first: python -m object_relocator.calibrate"
            )

        self.bus = None

    def connect(self):
        """Connect to the motor bus."""
        self.bus = create_motor_bus(self.robot_port, self.robot_id)

    def disconnect(self):
        """Disconnect the motor bus."""
        if self.bus:
            self.bus.disconnect()
            self.bus = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    # --- Detection ---

    def detect_object(self, show_detection: bool = False) -> tuple | None:
        """Detect an object on the desk using background subtraction.

        Returns (centroid_x, centroid_y, bbox) or None.
        """
        camera_index = self.cal.get("base_camera_index", 0)
        frame = capture_frame(camera_index)
        if frame is None:
            print("Error: Could not capture frame from base camera.")
            return None

        result = detect_object(
            frame, self.reference_image,
            threshold=self.cal.get("detection_threshold", 30),
            min_area=self.cal.get("min_object_area", 500),
            debug=show_detection,
        )

        if result is None:
            print("No object detected on desk.")
        else:
            cx, cy, bbox = result
            print(f"Object detected at pixel ({cx}, {cy}), "
                  f"bbox=({bbox[0]}, {bbox[1]}, {bbox[2]}x{bbox[3]})")

        return result

    # --- Position mapping ---

    def find_nearest_position(self, pixel_x: int, pixel_y: int) -> int:
        """Find the calibrated position closest to the given pixel coordinates.

        Returns the index into self.cal["positions"].
        """
        positions = self.cal["positions"]
        best_idx = 0
        best_dist = float("inf")

        for i, pos in enumerate(positions):
            dx = pixel_x - pos["pixel_x"]
            dy = pixel_y - pos["pixel_y"]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        print(f"Nearest calibrated position: {positions[best_idx]['name']} "
              f"(pixel dist: {best_dist:.0f}px)")
        return best_idx

    def pick_random_target(self, exclude_index: int = None) -> int:
        """Pick a random position index, excluding the given one."""
        positions = self.cal["positions"]
        candidates = [i for i in range(len(positions)) if i != exclude_index]
        if not candidates:
            return 0
        return random.choice(candidates)

    # --- Motion execution ---

    def _move_to(self, joint_positions: dict, steps: int = None,
                 delay_s: float = None):
        """Move smoothly to a joint position dict."""
        s = steps or self.cal.get("interpolation_steps", 50)
        d = delay_s or self.cal.get("step_delay_ms", 20) / 1000.0
        move_smooth(self.bus, joint_positions, steps=s, delay_s=d)

    def _go_home(self):
        """Move to the home (safe) position."""
        self._move_to(self.cal["home"])

    # --- Main relocation ---

    def relocate(self, show_detection: bool = False) -> bool:
        """Detect object, then push it to a random new position.

        Motion sequence:
            1. home
            2. approach above object position
            3. descend to push height at object position
            4. sweep at push height from object to target position
            5. lift to approach height at target position
            6. home

        Returns True on success, False if object not detected.
        """
        if self.bus is None:
            raise RuntimeError("Not connected. Call connect() first.")

        # 1. Detect object
        detection = self.detect_object(show_detection=show_detection)
        if detection is None:
            return False

        cx, cy, _ = detection

        # 2. Find nearest calibration position to the object
        pick_idx = self.find_nearest_position(cx, cy)
        pick_pos = self.cal["positions"][pick_idx]

        # 3. Pick random target position (different from pick)
        target_idx = self.pick_random_target(exclude_index=pick_idx)
        target_pos = self.cal["positions"][target_idx]

        print(f"\nPushing: {pick_pos['name']} -> {target_pos['name']}")

        # 4. Execute push sequence
        print("  Moving to home...")
        self._go_home()

        print(f"  Approaching above {pick_pos['name']}...")
        self._move_to(pick_pos["approach"])

        print(f"  Descending to push height...")
        self._move_to(pick_pos["push"])

        print(f"  Pushing to {target_pos['name']}...")
        self._move_to(target_pos["push"])

        settle_s = self.cal.get("push_settle_ms", 300) / 1000.0
        time.sleep(settle_s)

        print(f"  Lifting from {target_pos['name']}...")
        self._move_to(target_pos["approach"])

        print("  Returning to home...")
        self._go_home()

        print(f"Push complete: {pick_pos['name']} -> {target_pos['name']}\n")
        return True

    def relocate_multiple(self, count: int, show_detection: bool = False) -> int:
        """Run multiple relocations in sequence.

        Returns the number of successful relocations.
        """
        successes = 0
        for i in range(count):
            print(f"--- Relocation {i + 1}/{count} ---")
            if self.relocate(show_detection=show_detection):
                successes += 1
            else:
                print(f"  Relocation {i + 1} failed (no object detected).")
            if i < count - 1:
                time.sleep(1.0)
        print(f"\nCompleted {successes}/{count} relocations.")
        return successes
