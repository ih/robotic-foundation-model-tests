"""Processor functions for SingleAction policy.

Creates identity (passthrough) pre/post processors since this is a non-learned
policy that works directly with absolute joint positions.
"""

from lerobot.policies.factory import (
    PolicyProcessorPipeline,
    batch_to_transition,
    transition_to_batch,
    policy_action_to_transition,
    transition_to_policy_action,
)


def make_single_action_pre_post_processors(config, pretrained_path=None, **kwargs):
    """Create identity pre/post processors for SingleAction policy."""
    preprocessor = PolicyProcessorPipeline(
        steps=[],
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline(
        steps=[],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor
