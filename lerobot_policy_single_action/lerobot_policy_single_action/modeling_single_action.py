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
        # Diversity joints: list of dicts with keys: name, index, target
        self._diversity_joints: list = []
        self._diversity_targets_locked: bool = False

        # Action computation state
        self._target_position: Optional[float] = None
        self._action_applied: bool = False
        self._action_just_applied: bool = False
        self._last_state: Optional[Tensor] = None
        self._frame_counter: int = 0
        self._episode_start_time: Optional[float] = None
        self._locked_positions: Optional[Tensor] = None
        # Cached random primary-start target per episode; re-sampled at reset().
        self._cached_reset_primary_target: Optional[float] = None

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
        if self._current_direction == "none":
            return "No movement"
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
            "diversity_joints": [dj["name"] for dj in self._diversity_joints],
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

    def _pick_random_position(self, joint_name: str) -> float:
        """Pick a uniformly random position within the joint's range."""
        jmin, jmax = self.config.get_joint_range(joint_name)
        return self._rng.uniform(jmin, jmax)

    def _pick_diversity_target(self, dj: dict) -> Optional[float]:
        """Pick a new target for a diversity joint, respecting bounds."""
        current_pos = None
        if self._last_state is not None:
            current_pos = self._last_state[0, dj["index"]].item()
        elif dj["target"] is not None:
            current_pos = dj["target"]

        if current_pos is None:
            return None

        jmin, jmax = self.config.get_joint_range(dj["name"])
        delta = self.config.secondary_position_delta
        if current_pos >= jmax - delta:
            change = -delta
        elif current_pos <= jmin + delta:
            change = delta
        else:
            change = self._rng.choice([-delta, delta])
        return max(jmin, min(jmax, current_pos + change))

    def reset(self):
        """Reset for new episode. Picks new target joint, direction, and diversity joints.

        When _diversity_targets_locked is True (set by the reset-phase patch),
        this is a double-reset from LeRobot's record_loop. In that case, only
        reset the action computation state but preserve the already-chosen
        joint, direction, and diversity targets.
        """
        if self._diversity_targets_locked:
            # Double-reset: preserve all choices, just reset action computation
            self._diversity_targets_locked = False
            self._target_position = None
            self._action_applied = False
            self._frame_counter = 0
            self._episode_start_time = None
            self._locked_positions = None
            return

        # Reset action state
        self._target_position = None
        self._action_applied = False
        self._frame_counter = 0
        self._episode_start_time = None
        self._locked_positions = None
        # Cleared so the new episode re-samples its primary start below.
        self._cached_reset_primary_target: Optional[float] = None

        if self.config.vary_target_joint:
            # Random mode: pick target and diversity joints from pool
            self._current_joint_name = self._rng.choice(self.config.joints)
            self._current_joint_index = SO101_JOINTS.index(self._current_joint_name)

            remaining = [j for j in self.config.joints if j != self._current_joint_name]
            if self.config.randomize_all_joints_on_reset:
                chosen = remaining
            else:
                num_diversity = 2 if self.config.tertiary_joint_name is not None else 1
                num_diversity = min(num_diversity, len(remaining))
                chosen = self._rng.sample(remaining, num_diversity)
            self._diversity_joints = [
                {"name": j, "index": SO101_JOINTS.index(j), "target": None}
                for j in chosen
            ]
        else:
            # Fixed mode: use configured joints
            self._current_joint_name = self.config.joint_name
            self._current_joint_index = self.config.joint_index
            self._diversity_joints = [
                {
                    "name": self.config.secondary_joint_name,
                    "index": self.config.secondary_joint_index,
                    "target": None,
                }
            ]
            if self.config.tertiary_joint_name is not None:
                self._diversity_joints.append({
                    "name": self.config.tertiary_joint_name,
                    "index": self.config.tertiary_joint_index,
                    "target": None,
                })

        # Carry over previous targets so _pick_diversity_target can read current pos
        # from _last_state or fall back to old target
        prev_targets = {
            dj["name"]: dj["target"]
            for dj in getattr(self, '_prev_diversity_joints', [])
            if dj["target"] is not None
        }
        for dj in self._diversity_joints:
            if dj["name"] in prev_targets:
                dj["target"] = prev_targets[dj["name"]]

        # Pick primary direction: force away from bounds if near min/max
        current_primary_pos = None
        if self._last_state is not None:
            current_primary_pos = self._last_state[0, self._current_joint_index].item()

        primary_min, primary_max = self.config.get_joint_range(self._current_joint_name)
        if current_primary_pos is not None and current_primary_pos >= primary_max - self.config.position_delta:
            self._current_direction = "negative"
        elif current_primary_pos is not None and current_primary_pos <= primary_min + self.config.position_delta:
            self._current_direction = "positive"
        else:
            directions = ["positive", "negative"]
            if self.config.include_no_movement:
                directions.append("none")
            self._current_direction = self._rng.choice(directions)

        # Pick new diversity joint targets
        for dj in self._diversity_joints:
            if self.config.randomize_all_joints_on_reset:
                dj["target"] = self._pick_random_position(dj["name"])
            else:
                dj["target"] = self._pick_diversity_target(dj)

        # Save for next reset's carry-over
        self._prev_diversity_joints = [dict(dj) for dj in self._diversity_joints]

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
        - Diversity joints at their new random targets
        - All non-active joints reset to starting positions (corrects drift)

        Called by the recording script's reset-phase patch to physically move
        joints before the next episode begins recording.
        """
        targets = {}

        # Diversity joint targets
        for dj in self._diversity_joints:
            if dj["target"] is not None:
                motor_name = dj["name"].replace(".pos", "")
                targets[motor_name] = dj["target"]

        if self.config.randomize_all_joints_on_reset:
            # Also randomize the primary joint's starting position. Cached
            # per reset() so verify + retries see the same target.
            if self._current_joint_name:
                motor_name = self._current_joint_name.replace(".pos", "")
                if getattr(self, "_cached_reset_primary_target", None) is None:
                    self._cached_reset_primary_target = self._pick_random_position(
                        self._current_joint_name
                    )
                targets[motor_name] = self._cached_reset_primary_target
        else:
            # Reset inactive joints to starting positions
            if self._starting_positions:
                active_motor_names = set()
                if self._current_joint_name:
                    active_motor_names.add(self._current_joint_name.replace(".pos", ""))
                for dj in self._diversity_joints:
                    active_motor_names.add(dj["name"].replace(".pos", ""))

                for motor_name, start_pos in self._starting_positions.items():
                    if motor_name not in active_motor_names:
                        targets[motor_name] = start_pos

            # Randomize primary joint start position for better range coverage.
            # Cached per reset() so repeat calls (verify + retries) all see the
            # same target — otherwise resampling turns each retry into a chase
            # of a new random pose and the tolerance check never converges.
            if self.config.randomize_primary_start and self._current_joint_name:
                motor_name = self._current_joint_name.replace(".pos", "")
                if getattr(self, "_cached_reset_primary_target", None) is None:
                    self._cached_reset_primary_target = self._pick_random_position(
                        self._current_joint_name
                    )
                targets[motor_name] = self._cached_reset_primary_target

        return targets

    def verify_reset_position(self, actual_positions: Dict[str, float]) -> Dict[str, float]:
        """Check if reset motor targets were reached; return corrected targets if not.

        Compares commanded targets from get_reset_motor_targets() against actual
        positions read from the robot. If any joint's error exceeds
        reset_position_tolerance, updates the target to the actual position
        (accepts where the joint actually is as the new target).

        Args:
            actual_positions: Dict of motor_name -> actual position from robot.

        Returns:
            Empty dict if all positions are within tolerance.
            Dict of motor_name -> corrected target for joints that failed.
        """
        tolerance = self.config.reset_position_tolerance
        targets = self.get_reset_motor_targets()
        corrected = {}

        # Build lookup of diversity joint motor names for state update
        diversity_motor_map = {
            dj["name"].replace(".pos", ""): dj
            for dj in self._diversity_joints
        }

        for motor_name, commanded in targets.items():
            actual = actual_positions.get(motor_name)
            if actual is None:
                continue
            error = abs(commanded - actual)
            if error > tolerance:
                corrected[motor_name] = actual
                # Update internal state to match reality
                if motor_name in diversity_motor_map:
                    diversity_motor_map[motor_name]["target"] = actual

        return corrected

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

        # During start buffer, hold positions
        elapsed = time.time() - self._episode_start_time
        if elapsed < self.config.start_buffer:
            if self.config.lock_inactive_joints:
                if self._locked_positions is None:
                    self._locked_positions = state.clone()
                action = self._locked_positions.clone()
            else:
                action = state.clone()
            for dj in self._diversity_joints:
                if dj["target"] is not None:
                    action[:, dj["index"]] = dj["target"]
            return action

        # Compute target on first frame after start buffer
        if not self._action_applied:
            if self._current_direction == "none":
                # No movement: hold primary joint at current position
                self._target_position = None
            else:
                current_pos = state[0, self._current_joint_index].item()
                if self._current_direction == "positive":
                    self._target_position = current_pos + self.config.position_delta
                else:
                    self._target_position = current_pos - self.config.position_delta
                primary_min, primary_max = self.config.get_joint_range(self._current_joint_name)
                self._target_position = max(
                    primary_min,
                    min(primary_max, self._target_position)
                )
            self._action_applied = True
            self._action_just_applied = True

        # Start with locked or current positions (hold all joints)
        if self.config.lock_inactive_joints:
            if self._locked_positions is None:
                self._locked_positions = state.clone()
            action = self._locked_positions.clone()
        else:
            action = state.clone()

        # Set target joint
        if self._target_position is not None:
            action[:, self._current_joint_index] = self._target_position

        # Hold diversity joints at episode targets
        for dj in self._diversity_joints:
            if dj["target"] is not None:
                action[:, dj["index"]] = dj["target"]

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
            # Map direction to discrete action: positive=1, negative=2, none=3
            # The action is applied on the first frame after the start buffer.
            # _action_just_applied is set by _compute_action on that frame only.
            if self._action_just_applied:
                if self._current_direction == "positive":
                    discrete = 1
                elif self._current_direction == "negative":
                    discrete = 2
                else:
                    discrete = 3  # no movement
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
