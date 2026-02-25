"""SingleAction policy implementation for LeRobot.

Each episode executes a single joint movement:
- Target joint and direction are randomly chosen
- A secondary joint varies between episodes for visual diversity
- Task description reflects the chosen action
"""

import random
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_single_action import SingleActionConfig, SO101_JOINTS


class SingleActionPolicy(PreTrainedPolicy):
    """A policy that executes one joint movement per episode.

    Each episode:
    1. Randomly picks a target joint from config.joints
    2. Randomly picks positive or negative direction
    3. Executes that single movement for the duration of the episode
    4. A secondary joint (randomly chosen from remaining joints) holds
       a random position that was set between episodes

    The current_task_description property provides a natural language
    description of the action for per-episode task labeling.
    """

    config_class = SingleActionConfig
    name = "single_action"

    def __init__(self, config: SingleActionConfig, **kwargs):
        super().__init__(config)
        self.config = config

        self._rng = random.Random(config.random_seed)

        # Current episode state
        self._current_joint_name: Optional[str] = None
        self._current_joint_index: int = 0
        self._current_direction: str = "positive"  # "positive" or "negative"
        self._current_secondary_joint_name: Optional[str] = None
        self._current_secondary_joint_index: int = 1
        self._current_secondary_target: Optional[float] = None
        self._secondary_target_locked: bool = False

        # Action computation state
        self._target_position: Optional[float] = None
        self._action_applied: bool = False
        self._last_state: Optional[Tensor] = None

        # Dummy parameter for device placement
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    @property
    def current_task_description(self) -> str:
        """Natural language description of this episode's action."""
        if self._current_joint_name is None:
            return "Idle"
        friendly = self._current_joint_name.replace(".pos", "").replace("_", " ")
        return self.config.task_template.format(
            joint_friendly_name=friendly,
            direction=self._current_direction,
            delta=self.config.position_delta,
        )

    def reset(self):
        """Reset for new episode. Picks new target joint, direction, and secondary.

        When _secondary_target_locked is True (set by the reset-phase patch),
        this is a double-reset from LeRobot's record_loop. In that case, only
        reset the action computation state but preserve the already-chosen
        joint, direction, and secondary target.
        """
        if self._secondary_target_locked:
            # Double-reset: preserve all choices, just reset action computation
            self._secondary_target_locked = False
            self._target_position = None
            self._action_applied = False
            return

        # Read current secondary position before resetting
        current_secondary_pos = None
        if self._last_state is not None and self._current_secondary_joint_index is not None:
            current_secondary_pos = self._last_state[0, self._current_secondary_joint_index].item()
        elif self._current_secondary_target is not None:
            current_secondary_pos = self._current_secondary_target

        # Reset action state
        self._target_position = None
        self._action_applied = False

        # Pick direction randomly (always random regardless of mode)
        self._current_direction = self._rng.choice(["positive", "negative"])

        if self.config.vary_target_joint:
            # Random mode: pick target and secondary from joints pool
            self._current_joint_name = self._rng.choice(self.config.joints)
            self._current_joint_index = SO101_JOINTS.index(self._current_joint_name)

            remaining = [j for j in self.config.joints if j != self._current_joint_name]
            self._current_secondary_joint_name = self._rng.choice(remaining)
            self._current_secondary_joint_index = SO101_JOINTS.index(
                self._current_secondary_joint_name
            )
        else:
            # Fixed mode: use configured joints
            self._current_joint_name = self.config.joint_name
            self._current_joint_index = self.config.joint_index
            self._current_secondary_joint_name = self.config.secondary_joint_name
            self._current_secondary_joint_index = self.config.secondary_joint_index

        # Pick new secondary position
        if current_secondary_pos is not None:
            delta = self.config.secondary_position_delta
            change = self._rng.choice([-delta, delta])
            self._current_secondary_target = max(-100, min(100, current_secondary_pos + change))
        else:
            # First episode: no position data yet
            self._current_secondary_target = None

    def get_reset_motor_targets(self) -> dict:
        """Return secondary motor target to command between episodes.

        Called by the recording script's reset-phase patch to physically move
        the secondary joint before the next episode begins recording.
        """
        if self._current_secondary_target is None:
            return {}
        motor_name = self._current_secondary_joint_name.replace(".pos", "")
        return {motor_name: self._current_secondary_target}

    def _compute_action(self, batch: Dict[str, Tensor]) -> Tensor:
        """Execute the single action for this episode.

        On the first frame, computes the target position for the active joint.
        All subsequent frames hold that target. Secondary joint is held at its
        episode target. All other joints maintain current positions.
        """
        state = batch.get("observation.state")
        if state is None:
            raise ValueError("observation.state not found in batch")

        self._last_state = state.clone()

        # Compute target on first frame
        if not self._action_applied:
            current_pos = state[0, self._current_joint_index].item()
            if self._current_direction == "positive":
                self._target_position = current_pos + self.config.position_delta
            else:
                self._target_position = current_pos - self.config.position_delta
            self._target_position = max(-100, min(100, self._target_position))
            self._action_applied = True

        # Start with current positions (hold all joints)
        action = state.clone()

        # Set target joint
        if self._target_position is not None:
            action[:, self._current_joint_index] = self._target_position

        # Hold secondary joint at episode target
        if self._current_secondary_target is not None:
            action[:, self._current_secondary_joint_index] = self._current_secondary_target

        return action

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, Tensor], **kwargs) -> Tensor:
        action = self._compute_action(batch)
        return action.unsqueeze(1)

    @torch.no_grad()
    def select_action(self, batch: Dict[str, Tensor]) -> Tensor:
        return self._compute_action(batch)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        device = next(iter(batch.values())).device
        return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

    def get_optim_params(self):
        return []

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    def to(self, device):
        super().to(device)
        return self
