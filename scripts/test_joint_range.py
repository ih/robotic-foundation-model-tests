"""Interactive joint position tester.

Usage: python scripts/test_joint_range.py [joint_name] [port] [robot_id]

Examples:
  python scripts/test_joint_range.py elbow_flex
  python scripts/test_joint_range.py shoulder_pan
  python scripts/test_joint_range.py wrist_flex COM3 my_so101_follower

Available joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

Type a number (in degrees) to move the joint to that position.
Type 'q' to quit and return to starting position.
"""
import sys
import time
import json
from pathlib import Path

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode, MotorCalibration

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def create_bus(port="COM3", robot_id="my_so101_follower"):
    cal_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so101_follower"
    cal_path = cal_dir / f"{robot_id}.json"
    calibration = None
    if cal_path.is_file():
        with open(cal_path) as f:
            cal_dict = json.load(f)
        calibration = {
            motor: MotorCalibration(**cal_data)
            for motor, cal_data in cal_dict.items()
        }

    bus = FeetechMotorsBus(
        port=port,
        motors={
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
            "gripper": Motor(6, "sts3215", MotorNormMode.DEGREES),
        },
        calibration=calibration,
    )
    bus.connect()
    return bus


def main():
    joint = sys.argv[1] if len(sys.argv) > 1 else "elbow_flex"
    port = sys.argv[2] if len(sys.argv) > 2 else "COM3"
    robot_id = sys.argv[3] if len(sys.argv) > 3 else "my_so101_follower"

    if joint not in JOINTS:
        print(f"Unknown joint '{joint}'. Available: {', '.join(JOINTS)}")
        sys.exit(1)

    bus = create_bus(port, robot_id)

    positions = bus.sync_read("Present_Position")
    start_pos = positions[joint]
    # Hold all other joints at their current positions
    hold_positions = {j: positions[j] for j in JOINTS if j != joint}
    print(f"All positions: {positions}")
    print(f"Starting {joint}: {start_pos:.1f}")
    print()
    print(f"Enter a number (in degrees) to move {joint} to that position.")
    print("Type 'q' to quit and return to starting position.")
    print()

    try:
        while True:
            val = input(f"{joint} > ").strip()
            if val.lower() == 'q':
                break
            try:
                target = float(val)
            except ValueError:
                print("  Invalid input. Enter a number or 'q'.")
                continue

            if target < -180 or target > 180:
                print("  Out of range. Must be -180 to 180.")
                continue

            # Write target joint + hold all others at their positions
            goal = dict(hold_positions)
            goal[joint] = target
            bus.sync_write("Goal_Position", goal)
            time.sleep(1.5)
            actual = bus.sync_read("Present_Position")
            print(f"  Commanded: {target:.1f}  Actual: {actual[joint]:.1f}")
    finally:
        print(f"\nReturning {joint} to start ({start_pos:.1f})...")
        # Restore all joints to their original positions
        restore = dict(hold_positions)
        restore[joint] = start_pos
        bus.sync_write("Goal_Position", restore)
        time.sleep(2)
        bus.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
