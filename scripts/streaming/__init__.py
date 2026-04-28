"""Continuous-stream recorder for SO-101 single-action exploration.

Records back-to-back actions into a single LeRobot v3.0 dataset without
the per-episode start_buffer / end_buffer / reset_time_s overhead that
the legacy `run_single_action_record.py` pipeline pays. Each "logical
episode" is one action_duration window; episodes are saved through the
official `LeRobotDataset` streaming-encoder API so the resulting dataset
is bit-compatible with what the existing canvas-world-model pipeline
consumes and is uploadable to the HuggingFace Hub via `push_to_hub`.

See README.md for the design rationale and CLI examples.
"""
