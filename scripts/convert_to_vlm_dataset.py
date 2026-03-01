#!/usr/bin/env python
"""Convert a LeRobot dataset with per-episode task descriptions to LLaVA JSON
format for VLM fine-tuning (e.g., Cosmos Reason 2).

Reads the LeRobot v3.0 dataset metadata to extract per-episode task strings
and video paths, then generates a LLaVA-format JSON with conversation pairs
including reasoning traces.

Usage:
    python scripts/convert_to_vlm_dataset.py \
        --dataset-path ~/.cache/huggingface/lerobot/irvinh/single-action-all-joints \
        --output-dir ./vlm_training_data \
        --camera base_0_rgb
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq


def load_tasks(dataset_path: Path) -> dict:
    """Load task index -> task string mapping from tasks.parquet."""
    tasks_path = dataset_path / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        raise FileNotFoundError(f"tasks.parquet not found at {tasks_path}")

    table = pq.read_table(tasks_path)
    df = table.to_pandas()

    tasks = {}
    for idx, row in df.iterrows():
        task_index = row["task_index"]
        # Task text may be in a 'task' column or stored as the DataFrame index
        if "task" in row.index:
            task = row["task"]
        else:
            task = str(idx)
        tasks[task_index] = task

    return tasks


def load_episodes(dataset_path: Path) -> list:
    """Load episode metadata from episodes parquet files."""
    episodes_dir = dataset_path / "meta" / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"episodes directory not found at {episodes_dir}")

    episodes = []
    # Read all chunk/file parquet files
    for parquet_file in sorted(episodes_dir.rglob("*.parquet")):
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        for _, row in df.iterrows():
            episodes.append(row.to_dict())

    # Sort by episode_index
    episodes.sort(key=lambda e: e["episode_index"])
    return episodes


def get_video_path(dataset_path: Path, episode: dict, camera_key: str) -> Path:
    """Construct the video file path for an episode and camera."""
    video_key = f"observation.images.{camera_key}"

    # Try to get chunk/file indices from episode metadata
    chunk_col = f"videos/{video_key}/chunk_index"
    file_col = f"videos/{video_key}/file_index"

    if chunk_col in episode and file_col in episode:
        chunk_index = int(episode[chunk_col])
        file_index = int(episode[file_col])
    else:
        # Fallback: assume chunk-000/file-000
        chunk_index = 0
        file_index = 0

    video_path = (
        dataset_path / "videos" / video_key
        / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
    )
    return video_path


def parse_action_from_task(task: str) -> dict:
    """Parse joint name, direction, and delta from a task description string.

    Expected format: "Move {joint_friendly_name} {direction} by {delta} units"
    """
    parts = task.lower().split()
    try:
        # Find "move" ... "by" ... "units"
        move_idx = parts.index("move")
        by_idx = parts.index("by")
        units_idx = parts.index("units")

        joint_friendly = " ".join(parts[move_idx + 1 : by_idx - 1])
        direction = parts[by_idx - 1]
        delta = parts[by_idx + 1]

        return {
            "joint_friendly_name": joint_friendly,
            "direction": direction,
            "delta": delta,
        }
    except (ValueError, IndexError):
        return {"joint_friendly_name": "unknown", "direction": "unknown", "delta": "unknown"}


def generate_think_trace(task: str) -> str:
    """Generate a reasoning trace from the task description."""
    info = parse_action_from_task(task)
    return (
        f"I observe a SO-101 robot arm. "
        f"The {info['joint_friendly_name']} joint moves in the "
        f"{info['direction']} direction by approximately "
        f"{info['delta']} units. All other joints remain stationary."
    )


def convert_dataset(dataset_path: Path, output_dir: Path, camera_key: str):
    """Convert LeRobot dataset to LLaVA JSON format."""
    print(f"Loading dataset from: {dataset_path}")

    tasks = load_tasks(dataset_path)
    episodes = load_episodes(dataset_path)

    print(f"Found {len(tasks)} tasks and {len(episodes)} episodes")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    training_data = []
    skipped = 0

    for episode in episodes:
        ep_idx = episode["episode_index"]
        ep_id = f"episode_{ep_idx:06d}"

        # Get task string for this episode
        ep_tasks = episode.get("tasks")
        if ep_tasks is None or len(ep_tasks) == 0:
            print(f"  Warning: episode {ep_idx} has no tasks, skipping")
            skipped += 1
            continue

        # ep_tasks is a list of task strings (usually 1)
        task = ep_tasks[0] if isinstance(ep_tasks[0], str) else tasks.get(ep_tasks[0], "Unknown")

        # Find and copy video
        src_video = get_video_path(dataset_path, episode, camera_key)
        if not src_video.exists():
            print(f"  Warning: video not found at {src_video}, skipping episode {ep_idx}")
            skipped += 1
            continue

        dst_video = videos_dir / f"{ep_id}.mp4"
        shutil.copy2(src_video, dst_video)

        # Generate conversation pair
        think_trace = generate_think_trace(task)
        entry = {
            "id": ep_id,
            "video": f"videos/{ep_id}.mp4",
            "conversations": [
                {
                    "from": "human",
                    "value": "<video>\nWhat action does the robot perform in this video?",
                },
                {
                    "from": "gpt",
                    "value": f"<think>\n{think_trace}\n</think>\n{task}",
                },
            ],
        }
        training_data.append(entry)

    # Write JSON
    output_json = output_dir / "training_data.json"
    with open(output_json, "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"\nConversion complete:")
    print(f"  Episodes converted: {len(training_data)}")
    print(f"  Episodes skipped: {skipped}")
    print(f"  Output JSON: {output_json}")
    print(f"  Videos directory: {videos_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot dataset to LLaVA JSON for VLM fine-tuning"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the LeRobot dataset (local cache directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for the LLaVA JSON and videos",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="base_0_rgb",
        help="Camera key to extract videos from (default: base_0_rgb)",
    )

    args = parser.parse_args()
    convert_dataset(
        dataset_path=Path(args.dataset_path),
        output_dir=Path(args.output_dir),
        camera_key=args.camera,
    )


if __name__ == "__main__":
    main()
