"""Configuration for SingleAction policy.

A policy where each episode executes a single joint movement. Both the target
joint and direction are randomly chosen each episode. A secondary joint varies
between episodes for visual diversity.
"""

from dataclasses import dataclass, field
from typing import List, Optional

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

    # Random joint mode
    vary_target_joint: bool = False
    joints: List[str] = field(default_factory=lambda: list(SO101_JOINTS))

    # Movement parameters
    position_delta: float = 10.0
    primary_min: float = -60.0
    primary_max: float = 60.0
    secondary_position_delta: float = 5.0
    secondary_min: float = 70.0
    secondary_max: float = 100.0
    action_duration: float = 1

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

    def __post_init__(self):
        super().__post_init__()

        if self.vary_target_joint:
            # Random mode: validate joints list
            for joint in self.joints:
                if joint not in SO101_JOINTS:
                    raise ValueError(
                        f"Unknown joint '{joint}'. Valid joints: {SO101_JOINTS}"
                    )
            if len(self.joints) < 2:
                raise ValueError(
                    "At least 2 joints required (need distinct target and secondary)"
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
            if self.joint_name == self.secondary_joint_name:
                raise ValueError("Primary and secondary joints must be different")
            self.joint_index = SO101_JOINTS.index(self.joint_name)
            self.secondary_joint_index = SO101_JOINTS.index(self.secondary_joint_name)

        # Validate movement parameters
        if self.position_delta <= 0:
            raise ValueError("position_delta must be positive")
        if self.secondary_position_delta <= 0:
            raise ValueError("secondary_position_delta must be positive")
        if self.action_duration <= 0:
            raise ValueError("action_duration must be positive")

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
