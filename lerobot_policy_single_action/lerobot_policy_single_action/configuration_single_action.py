"""Configuration for SingleAction policy.

A policy where each episode executes a single joint movement. Both the target
joint and direction are randomly chosen each episode. A secondary joint varies
between episodes for visual diversity.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from lerobot.configs.policies import PreTrainedConfig


# SO-101 joint names
SO101_JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# Previous ranges (RANGE_M100_100 units, not degrees):
# DEFAULT_JOINT_RANGES: Dict[str, Tuple[float, float]] = {
#     "shoulder_pan.pos":  (-90.0, 90.0),
#     "shoulder_lift.pos": (-100.0, 40.0),
#     "elbow_flex.pos":    (-90.0, 0.0),
#     "wrist_flex.pos":    (-50.0, 50.0),
#     "wrist_roll.pos":    (-100.0, 100.0),
#     "gripper.pos":       (10.0, 100.0),
# }
DEFAULT_JOINT_RANGES: Dict[str, Tuple[float, float]] = {
    "shoulder_pan.pos":  (-60.0, 60.0),
    "shoulder_lift.pos": (-60.0, -10.0),
    "elbow_flex.pos":    (50.0, 96.0),
    "wrist_flex.pos":    (-96.0, 96.0),
    "wrist_roll.pos":    (-91.0, 91.0),
    "gripper.pos":       (-63.0, 63.0),
}


@PreTrainedConfig.register_subclass("single_action")
@dataclass
class SingleActionConfig(PreTrainedConfig):
    """Configuration for the SingleAction policy.

    Each episode:
    1. Picks a target joint and direction (positive or negative)
    2. Executes a single movement of that joint by position_delta
    3. A secondary joint is moved to a new random position between episodes
       for visual diversity

    The task description for each episode is generated from the action,
    e.g. "Move shoulder pan positive by 10 degrees".

    Joint selection modes:
    - Fixed: Set joint_name and secondary_joint_name for specific joints
    - Random: Set vary_target_joint=True to randomly pick from joints list each episode

    Args:
        joint_name: Fixed target joint (used when vary_target_joint=False)
        secondary_joint_name: Fixed secondary joint (used when vary_target_joint=False)
        vary_target_joint: If True, randomly pick target and secondary from joints list
        joints: List of joints for random selection (used when vary_target_joint=True)
        position_delta: How far the target joint moves per action (degrees)
        secondary_position_delta: Magnitude for random secondary joint changes
        action_duration: How long to hold the action per episode (seconds)
        task_template: Template for generating per-episode task descriptions
        random_seed: Seed for reproducible random behavior
    """

    # Fixed joint mode (default)
    joint_name: str = "shoulder_pan.pos"
    secondary_joint_name: str = "elbow_flex.pos"
    tertiary_joint_name: Optional[str] = None

    # Random joint mode
    vary_target_joint: bool = False
    randomize_all_joints_on_reset: bool = False
    joints: List[str] = field(default_factory=lambda: list(SO101_JOINTS))

    # No-movement action
    include_no_movement: bool = True

    # Joint locking: hold inactive joints at their episode-start positions
    # to prevent gravity-induced drift during episodes
    lock_inactive_joints: bool = True

    # Randomize primary joint start position each episode for better range coverage
    randomize_primary_start: bool = False

    # Movement parameters
    position_delta: float = 10.0
    secondary_position_delta: float = 10.0
    joint_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_JOINT_RANGES)
    )
    action_duration: float = 1
    start_buffer: float = 1.0
    end_buffer: float = 1.0
    reset_position_tolerance: float = 5.0
    max_reset_retries: int = 3

    # Task description template
    # Available placeholders: {joint_friendly_name}, {direction}, {delta}
    task_template: str = "Move {joint_friendly_name} {direction} by {delta} units"

    # Reproducibility
    random_seed: Optional[int] = None

    # Discrete action logging
    discrete_action_log_dir: Optional[str] = None
    discrete_action_log_path: Optional[str] = None

    # Required PreTrainedConfig fields
    n_obs_steps: int = 1
    n_action_steps: int = 1

    # Required abstract property implementations
    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    # Computed fields
    joint_index: int = field(init=False, default=0)
    secondary_joint_index: int = field(init=False, default=0)
    tertiary_joint_index: int = field(init=False, default=0)

    def __post_init__(self):
        super().__post_init__()

        if self.randomize_all_joints_on_reset and not self.vary_target_joint:
            raise ValueError(
                "randomize_all_joints_on_reset requires vary_target_joint=True"
            )

        if self.vary_target_joint:
            # Random mode: validate joints list
            for joint in self.joints:
                if joint not in SO101_JOINTS:
                    raise ValueError(
                        f"Unknown joint '{joint}'. Valid joints: {SO101_JOINTS}"
                    )
            min_joints = 3 if self.tertiary_joint_name else 2
            if len(self.joints) < min_joints:
                raise ValueError(
                    f"At least {min_joints} joints required (need distinct target"
                    f" and diversity joints)"
                )
        else:
            # Fixed mode: validate specific joints
            if self.joint_name not in SO101_JOINTS:
                raise ValueError(
                    f"Unknown joint '{self.joint_name}'. Valid joints: {SO101_JOINTS}"
                )
            if self.secondary_joint_name not in SO101_JOINTS:
                raise ValueError(
                    f"Unknown secondary joint '{self.secondary_joint_name}'. "
                    f"Valid joints: {SO101_JOINTS}"
                )
            fixed_joints = {self.joint_name, self.secondary_joint_name}
            if len(fixed_joints) < 2:
                raise ValueError("Primary and secondary joints must be different")
            if self.tertiary_joint_name is not None:
                if self.tertiary_joint_name not in SO101_JOINTS:
                    raise ValueError(
                        f"Unknown tertiary joint '{self.tertiary_joint_name}'. "
                        f"Valid joints: {SO101_JOINTS}"
                    )
                if self.tertiary_joint_name in fixed_joints:
                    raise ValueError(
                        "Tertiary joint must be different from primary and secondary"
                    )
                self.tertiary_joint_index = SO101_JOINTS.index(self.tertiary_joint_name)
            self.joint_index = SO101_JOINTS.index(self.joint_name)
            self.secondary_joint_index = SO101_JOINTS.index(self.secondary_joint_name)

        # Validate movement parameters
        if self.position_delta <= 0:
            raise ValueError("position_delta must be positive")
        if self.secondary_position_delta <= 0:
            raise ValueError("secondary_position_delta must be positive")
        if self.action_duration <= 0:
            raise ValueError("action_duration must be positive")
        if self.start_buffer < 0:
            raise ValueError("start_buffer must be non-negative")
        if self.end_buffer < 0:
            raise ValueError("end_buffer must be non-negative")
        for joint_name, (lo, hi) in self.joint_ranges.items():
            if lo >= hi:
                raise ValueError(
                    f"Invalid range for {joint_name}: min ({lo}) must be < max ({hi})"
                )

    def get_joint_range(self, joint_name: str) -> Tuple[float, float]:
        """Return (min, max) for the given joint, falling back to (-60, 60)."""
        return self.joint_ranges.get(joint_name, (-60.0, 60.0))

    def get_optimizer_preset(self):
        return None

    def get_scheduler_preset(self):
        return None

    def validate_features(self):
        if not hasattr(self, 'input_features') or self.input_features is None:
            return
        has_state = any(
            'state' in key.lower()
            for key in self.input_features.keys()
        )
        if not has_state:
            raise ValueError(
                "SingleActionPolicy requires 'observation.state' in input_features"
            )
