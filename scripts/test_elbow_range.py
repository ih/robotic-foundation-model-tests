"""Interactive elbow_flex position tester.

Type a number (-100 to 100) to move elbow_flex to that position.
Type 'q' to quit and return to starting position.
"""
import sys
import time
import json
from pathlib import Path

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode, MotorCalibration


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


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM3"
    robot_id = sys.argv[2] if len(sys.argv) > 2 else "my_so101_follower"

    bus = create_bus(port, robot_id)

    positions = bus.sync_read("Present_Position")
    start_elbow = positions['elbow_flex']
    print(f"All positions: {positions}")
    print(f"Starting elbow_flex: {start_elbow:.1f}")
    print()
    print("Enter a number (-100 to 100) to move elbow_flex to that position.")
    print("Type 'q' to quit and return to starting position.")
    print()

    try:
        while True:
            val = input("elbow_flex > ").strip()
            if val.lower() == 'q':
                break
            try:
                target = float(val)
            except ValueError:
                print("  Invalid input. Enter a number or 'q'.")
                continue

            if target < -100 or target > 100:
                print("  Out of range. Must be -100 to 100.")
                continue

            bus.sync_write("Goal_Position", {"elbow_flex": target})
            time.sleep(1.5)
            actual = bus.sync_read("Present_Position")
            print(f"  Commanded: {target:.1f}  Actual: {actual['elbow_flex']:.1f}")
    finally:
        print(f"\nReturning to start ({start_elbow:.1f})...")
        bus.sync_write("Goal_Position", {"elbow_flex": start_elbow})
        time.sleep(2)
        bus.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
