"""Single-action episode policy for LeRobot SO-101 arm.

Each episode executes a single joint movement (randomly chosen joint and direction).
A secondary joint varies between episodes for visual diversity.
Designed for generating VLM fine-tuning datasets with per-episode task descriptions.

Usage:
    pip install -e .

    python scripts/run_single_action_record.py \\
        --robot.type=so101_follower \\
        --policy.type=single_action \\
        --policy.position_delta=10 \\
        ...
"""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package:\n"
        "  pip install lerobot\n"
        "Or follow the LeRobot installation instructions."
    )

from .configuration_single_action import SingleActionConfig
from .modeling_single_action import SingleActionPolicy
from .processor_single_action import make_single_action_pre_post_processors

__all__ = [
    "SingleActionConfig",
    "SingleActionPolicy",
    "make_single_action_pre_post_processors",
]

__version__ = "0.1.0"
