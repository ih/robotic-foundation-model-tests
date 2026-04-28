"""Action sequencer for continuous-stream recording.

Replaces `SingleActionPolicy.reset()` + `get_reset_motor_targets()` with
a single `next_action()` call that picks the next (joint, direction,
target_pos) on the fly without going through LeRobot's record_loop.

Why a separate class instead of reusing SingleActionPolicy:
  - SingleActionPolicy's lifecycle is `reset() -> compute action per
    frame -> reset()`, driven by record_loop's call cadence and gated
    on internal flags like `_diversity_targets_locked`. Stripping that
    state machine apart for back-to-back actions is more invasive than
    re-implementing the action-choice logic in ~50 lines.
  - The sequencer is a pure decision function: given current motor
    state + config, return the next target. Trivial to test.

Action diversity matches the existing per-episode reset behavior:
  - If `vary_target_joint=True`: pick joint uniformly from the pool
    each call.
  - Direction: positive/negative; forced away from range edges so we
    never command a no-op clamp.
  - Target position: current_pos +/- position_delta, clamped to the
    joint's configured (lo, hi) range.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class ActionTarget:
    """One sequencer decision."""

    joint: str                 # e.g. "shoulder_pan.pos"
    motor_name: str            # e.g. "shoulder_pan" (without .pos suffix)
    joint_index: int           # index into the SO-101 joint list
    direction: str             # "positive" | "negative" | "none"
    target_pos: float          # absolute target in motor-space units
    magnitude: float           # signed delta from current_pos

    @property
    def task_description(self) -> str:
        """Same string format as SingleActionPolicy.current_task_description.

        Kept identical so downstream consumers (create_dataset.py, the
        learner's task-string parsers) need no changes.
        """
        if self.direction == "none":
            return "No movement"
        friendly = self.motor_name.replace("_", " ")
        return (
            f"Move {friendly} {self.direction} by {abs(self.magnitude):.1f} units"
        )


class ActionSequencer:
    """Picks the next action target without a per-episode reset cycle.

    Args:
        joints: pool of joint names (with `.pos` suffix). When `vary_target`
            is True, one is chosen uniformly per call.
        joint_ranges: dict of `joint -> (lo, hi)` clamping bounds.
        joint_indices: dict of `joint -> motor index`. SO-101 default is
            [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper].
        position_delta: magnitude of each step in motor units.
        vary_target: if True, sample joint per call; else use joints[0].
        seed: optional RNG seed for reproducible action sequences.
    """

    def __init__(
        self,
        joints: list[str],
        joint_ranges: dict[str, tuple[float, float]],
        joint_indices: dict[str, int],
        position_delta: float,
        vary_target: bool = True,
        seed: int | None = None,
    ):
        if not joints:
            raise ValueError("joints pool must be non-empty")
        for j in joints:
            if j not in joint_ranges:
                raise ValueError(f"joint {j!r} missing from joint_ranges")
            if j not in joint_indices:
                raise ValueError(f"joint {j!r} missing from joint_indices")

        self.joints = list(joints)
        self.joint_ranges = dict(joint_ranges)
        self.joint_indices = dict(joint_indices)
        self.position_delta = float(position_delta)
        self.vary_target = bool(vary_target)
        self._rng = random.Random(seed)

    def next_action(self, motor_positions: dict[str, float]) -> ActionTarget:
        """Pick the next action target from the current motor state.

        Args:
            motor_positions: dict of `motor_name -> current_pos` (no `.pos`
                suffix), as returned by `bus.sync_read("Present_Position")`.
        """
        joint = (
            self._rng.choice(self.joints) if self.vary_target else self.joints[0]
        )
        motor_name = joint.replace(".pos", "")
        joint_index = self.joint_indices[joint]
        lo, hi = self.joint_ranges[joint]

        current = float(motor_positions.get(motor_name, 0.0))

        # Force direction away from range edges so we always make a real
        # delta. Inside the safe interior, sample uniformly.
        if current >= hi - self.position_delta:
            direction = "negative"
        elif current <= lo + self.position_delta:
            direction = "positive"
        else:
            direction = self._rng.choice(["positive", "negative"])

        sign = 1.0 if direction == "positive" else -1.0
        target = current + sign * self.position_delta
        target = max(lo, min(hi, target))
        magnitude = target - current

        return ActionTarget(
            joint=joint,
            motor_name=motor_name,
            joint_index=joint_index,
            direction=direction,
            target_pos=float(target),
            magnitude=float(magnitude),
        )
