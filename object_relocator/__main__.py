"""CLI entrypoint for the object relocator.

Usage:
    python -m object_relocator --port COM3 --robot-id my_so101_follower
    python -m object_relocator --port COM3 --robot-id my_so101_follower --count 5
    python -m object_relocator --port COM3 --robot-id my_so101_follower --show-detection
"""

import argparse

from .relocator import ObjectRelocator


def main():
    parser = argparse.ArgumentParser(
        description="Relocate an object on the desk using the SO-101 robot arm"
    )
    parser.add_argument("--port", required=True, help="Serial port (e.g., COM3)")
    parser.add_argument("--robot-id", required=True,
                        help="Robot ID for calibration lookup")
    parser.add_argument("--count", type=int, default=1,
                        help="Number of relocations to perform (default: 1)")
    parser.add_argument("--show-detection", action="store_true",
                        help="Show detection visualization windows")
    parser.add_argument("--calibration", default=None,
                        help="Path to calibration JSON (default: auto from robot-id)")
    args = parser.parse_args()

    relocator = ObjectRelocator(
        robot_port=args.port,
        robot_id=args.robot_id,
        calibration_path=args.calibration,
    )

    with relocator:
        if args.count == 1:
            success = relocator.relocate(show_detection=args.show_detection)
            if not success:
                print("Relocation failed. Is an object on the desk?")
        else:
            relocator.relocate_multiple(
                count=args.count,
                show_detection=args.show_detection,
            )


if __name__ == "__main__":
    main()
