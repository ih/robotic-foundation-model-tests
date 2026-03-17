"""SingleAction policy implementation for LeRobot.

Each episode executes a single joint movement:
- Target joint and direction are randomly chosen
- A secondary joint varies between episodes for visual diversity
- Task description reflects the chosen action
"""

import json
import random
import time
from pathlib import Path
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
        self._action_just_applied: bool = False
        self._last_state: Optional[Tensor] = None
        self._frame_counter: int = 0
        self._episode_start_time: Optional[float] = None

        # Starting positions for drift correction of inactive joints
        self._starting_positions: Optional[Dict[str, float]] = None

        # Dummy parameter for device placement
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def set_starting_positions(self, positions: Dict[str, float]):
        """Store starting positions for resetting inactive joints between episodes."""
        self._starting_positions = positions

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

    def _write_log_header(self):
        """Write metadata header to the discrete action log file."""
        if not self.config.discrete_action_log_path:
            return

        log_path = Path(self.config.discrete_action_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        header = {
            "type": "header",
            "joint_name": self._current_joint_name or self.config.joint_name,
            "action_duration": self.config.action_duration,
            "position_delta": self.config.position_delta,
            "vary_target_joint": self.config.vary_target_joint,
            "random_seed": self.config.random_seed,
            "direction": self._current_direction,
            "secondary_joint_name": self._current_secondary_joint_name,
        }

        with open(log_path, "w") as f:
            f.write(json.dumps(header) + "\n")

    def _log_discrete_action(self, timestamp: float, discrete_action: int,
                             frame_index: int):
        """Log a discrete action decision to the log file."""
        if not self.config.discrete_action_log_path:
            return

        with open(self.config.discrete_action_log_path, "a") as f:
            f.write(json.dumps({
                "type": "action",
                "timestamp": timestamp,
                "discrete_action": discrete_action,
                "frame_index": frame_index,
            }) + "\n")

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
            self._frame_counter = 0
            self._episode_start_time = None
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
        self._frame_counter = 0
        self._episode_start_time = None

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

        # Pick primary direction: force away from bounds if near min/max
        current_primary_pos = None
        if self._last_state is not None:
            current_primary_pos = self._last_state[0, self._current_joint_index].item()

        if current_primary_pos is not None and current_primary_pos >= self.config.primary_max - self.config.position_delta:
            self._current_direction = "negative"
        elif current_primary_pos is not None and current_primary_pos <= self.config.primary_min + self.config.position_delta:
            self._current_direction = "positive"
        else:
            self._current_direction = self._rng.choice(["positive", "negative"])

        # Pick new secondary position: force away from bounds if near min/max
        if current_secondary_pos is not None:
            delta = self.config.secondary_position_delta
            if current_secondary_pos >= self.config.secondary_max - delta:
                change = -delta
            elif current_secondary_pos <= self.config.secondary_min + delta:
                change = delta
            else:
                change = self._rng.choice([-delta, delta])
            self._current_secondary_target = max(
                self.config.secondary_min,
                min(self.config.secondary_max, current_secondary_pos + change)
            )
        else:
            # First episode: no position data yet
            self._current_secondary_target = None

        # Set up per-episode discrete action log file
        if self.config.discrete_action_log_dir:
            log_dir = Path(self.config.discrete_action_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            # Clean up previous header-only (spurious) log file from double-reset
            prev_path = getattr(self, '_last_log_path', None)
            if prev_path:
                prev = Path(prev_path)
                if prev.exists():
                    with open(prev) as f:
                        line_count = sum(1 for _ in f)
                    if line_count <= 1:  # Header only, no action entries
                        prev.unlink()

            existing = sorted(log_dir.glob("episode_*.jsonl"))
            episode_num = len(existing)
            self.config.discrete_action_log_path = str(
                log_dir / f"episode_{episode_num:06d}.jsonl"
            )
            self._last_log_path = self.config.discrete_action_log_path
            self._write_log_header()

    def get_reset_motor_targets(self) -> dict:
        """Return motor targets to command between episodes.

        Includes:
        - Secondary joint at its new random target
        - All non-active joints reset to starting positions (corrects drift)

        Called by the recording script's reset-phase patch to physically move
        joints before the next episode begins recording.
        """
        targets = {}

        # Secondary joint target
        if self._current_secondary_target is not None:
            motor_name = self._current_secondary_joint_name.replace(".pos", "")
            targets[motor_name] = self._current_secondary_target

        # Reset inactive joints to starting positions
        if self._starting_positions:
            active_motor_names = set()
            if self._current_joint_name:
                active_motor_names.add(self._current_joint_name.replace(".pos", ""))
            if self._current_secondary_joint_name:
                active_motor_names.add(self._current_secondary_joint_name.replace(".pos", ""))

            for motor_name, start_pos in self._starting_positions.items():
                if motor_name not in active_motor_names:
                    targets[motor_name] = start_pos

        return targets

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

        # Track episode start time for start buffer
        if self._episode_start_time is None:
            self._episode_start_time = time.time()

        # Reset per-frame flag
        self._action_just_applied = False

        # During start buffer, hold current positions
        elapsed = time.time() - self._episode_start_time
        if elapsed < self.config.start_buffer:
            action = state.clone()
            if self._current_secondary_target is not None:
                action[:, self._current_secondary_joint_index] = self._current_secondary_target
            return action

        # Compute target on first frame after start buffer
        if not self._action_applied:
            current_pos = state[0, self._current_joint_index].item()
            if self._current_direction == "positive":
                self._target_position = current_pos + self.config.position_delta
            else:
                self._target_position = current_pos - self.config.position_delta
            self._target_position = max(
                self.config.primary_min,
                min(self.config.primary_max, self._target_position)
            )
            self._action_applied = True
            self._action_just_applied = True

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
        action = self._compute_action(batch)

        # Log discrete action for this frame
        if self.config.discrete_action_log_path:
            # Map direction to discrete action: positive=1, negative=2
            # The action is applied on the first frame after the start buffer.
            # _action_just_applied is set by _compute_action on that frame only.
            if self._action_just_applied:
                if self._current_direction == "positive":
                    discrete = 1
                else:
                    discrete = 2
            else:
                discrete = 0

            self._log_discrete_action(
                timestamp=time.time(),
                discrete_action=discrete,
                frame_index=self._frame_counter,
            )

        self._frame_counter += 1
        return action

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
