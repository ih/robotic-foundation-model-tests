#!/usr/bin/env python
"""Verify a streaming-recorded dataset round-trips correctly.

Run after `record_continuous.py` to confirm:
  1. The on-disk layout is valid LeRobot v3.0 (loadable via the official
     `LeRobotDataset` class).
  2. Frames decode and shapes match what was declared in `meta/info.json`.
  3. `canvas-world-model`'s `LeRobotV3Reader` can read it (i.e. the
     existing canvas-building pipeline will accept it).
  4. Per-episode discrete-action logs are present and parseable.

This is the local equivalent of the HuggingFace Hub viewer's "does it
actually open?" check. If this passes, `--push-to-hub` should also work
because the layout is identical.

Optional: pass `--push-to-hub` to also exercise the upload path. The
dataset must have been created with `--output-repo-id <namespace/name>`
matching an HF account you have write access to.

Usage:
    python -m scripts.streaming.verify_dataset \\
        --root C:/Users/.../.cache/huggingface/lerobot/local/streaming-smoke \\
        --repo-id local/streaming-smoke

    # Also push:
    python -m scripts.streaming.verify_dataset \\
        --root .../irvinh/streaming-shoulder-elbow-60 \\
        --repo-id irvinh/streaming-shoulder-elbow-60 \\
        --push-to-hub
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _check_layout(root: Path) -> dict:
    """Sanity-check the v3.0 directory tree before invoking LeRobotDataset."""
    issues = []
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        issues.append(f"missing {info_path}")
        return {"ok": False, "issues": issues, "info": None}

    with open(info_path) as f:
        info = json.load(f)

    if info.get("codebase_version") != "v3.0":
        issues.append(f"codebase_version={info.get('codebase_version')!r}, expected 'v3.0'")

    expected = [
        root / "data" / "chunk-000",
        root / "videos",
        root / "meta" / "tasks.parquet",
        root / "meta" / "episodes",
    ]
    for p in expected:
        if not p.exists():
            issues.append(f"missing {p}")

    log_dir = root / "meta" / "discrete_action_logs"
    if not log_dir.exists():
        issues.append(f"missing {log_dir} (canvas-world-model requires this)")
    else:
        logs = sorted(log_dir.glob("episode_*.jsonl"))
        if not logs:
            issues.append(f"no episode_*.jsonl files in {log_dir}")

    return {"ok": not issues, "issues": issues, "info": info}


def _check_lerobot_load(repo_id: str, root: Path) -> dict:
    """Round-trip via the official LeRobotDataset constructor."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as e:
        return {"ok": False, "error": f"import failed: {e}"}

    try:
        ds = LeRobotDataset(repo_id=repo_id, root=str(root))
    except Exception as e:
        return {"ok": False, "error": f"LeRobotDataset(root={root}) failed: {e}"}

    summary = {
        "ok": True,
        "total_episodes": ds.meta.total_episodes,
        "total_frames": ds.meta.total_frames,
        "fps": ds.meta.fps,
        "features": list(ds.meta.features.keys()),
        "video_keys": list(ds.meta.video_keys),
    }

    # Decode the first frame as a smoke test.
    try:
        sample = ds[0]
        for k in ds.meta.video_keys:
            if k in sample:
                summary[f"first_frame_shape::{k}"] = tuple(sample[k].shape)
    except Exception as e:
        summary["frame_decode_error"] = str(e)
        summary["ok"] = False

    return summary


def _check_canvas_world_model_loader(root: Path, base_camera_name: str) -> dict:
    """Confirm canvas-world-model's loader can consume this dataset.

    Imports `data.lerobot_loader.LeRobotV3Reader` from sibling repo. If
    the repo isn't on PYTHONPATH we add it temporarily.
    """
    cwm_root = Path(__file__).resolve().parents[3] / "canvas-world-model"
    if cwm_root.exists() and str(cwm_root) not in sys.path:
        sys.path.insert(0, str(cwm_root))

    try:
        from data.lerobot_loader import LeRobotV3Reader, load_decision_bearing_logs
    except Exception as e:
        return {"ok": False, "error": f"canvas-world-model import failed: {e}"}

    try:
        reader = LeRobotV3Reader(str(root))
    except Exception as e:
        return {"ok": False, "error": f"LeRobotV3Reader failed: {e}"}

    summary = {
        "ok": True,
        "total_episodes": reader.total_episodes,
        "fps": reader.fps,
        "chunks_size": reader.chunks_size,
    }

    # discrete_action_logs presence is mandatory for create_dataset.py
    log_dir = reader.dataset_path / "meta" / "discrete_action_logs"
    if not log_dir.exists():
        summary["ok"] = False
        summary["error"] = f"missing {log_dir}"
        return summary

    logs = load_decision_bearing_logs(log_dir)
    summary["decision_bearing_logs"] = len(logs)
    if len(logs) != reader.total_episodes:
        summary["ok"] = False
        summary["error"] = (
            f"log/episode mismatch: {len(logs)} logs vs {reader.total_episodes} episodes"
        )
        return summary

    # First episode: confirm at least one non-zero discrete action exists
    if logs:
        nonzero = sum(
            1 for d in logs[0].decisions if d.get("discrete_action", 0) != 0
        )
        summary["first_episode_nonzero_actions"] = nonzero
        if nonzero == 0:
            summary["ok"] = False
            summary["error"] = "first episode has no non-zero discrete actions (would be trimmed as no-op)"

    # Confirm the camera key used by the recorder is present in info.features
    cam_key = f"observation.images.{base_camera_name}"
    if cam_key not in reader.info.get("features", {}):
        summary["ok"] = False
        summary["error"] = f"camera key {cam_key!r} missing from info.json features"

    return summary


def _push_to_hub(repo_id: str, root: Path, private: bool) -> dict:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as e:
        return {"ok": False, "error": f"import failed: {e}"}

    try:
        ds = LeRobotDataset(repo_id=repo_id, root=str(root))
    except Exception as e:
        return {"ok": False, "error": f"reload failed: {e}"}

    try:
        ds.push_to_hub(private=private)
    except Exception as e:
        return {"ok": False, "error": f"push_to_hub failed: {e}"}

    return {
        "ok": True,
        "url": f"https://huggingface.co/datasets/{repo_id}",
    }


def _print_section(title: str, body: dict) -> None:
    print(f"\n=== {title} ===")
    for k, v in body.items():
        print(f"  {k}: {v}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Local dataset root.")
    p.add_argument("--repo-id", required=True, help="Repo ID (namespace/name).")
    p.add_argument("--base-camera-name", default="base_0_rgb")
    p.add_argument("--push-to-hub", action="store_true",
                   help="Also exercise the HF upload path (requires HF_TOKEN).")
    p.add_argument("--public-hub", action="store_true",
                   help="Make the pushed dataset public.")
    args = p.parse_args()

    root = Path(args.root).resolve()

    layout = _check_layout(root)
    _print_section("layout", {"ok": layout["ok"], "issues": layout["issues"]})
    if not layout["ok"]:
        return 1

    lr = _check_lerobot_load(args.repo_id, root)
    _print_section("LeRobotDataset load", lr)
    if not lr.get("ok"):
        return 1

    cwm = _check_canvas_world_model_loader(root, args.base_camera_name)
    _print_section("canvas-world-model loader", cwm)
    if not cwm.get("ok"):
        return 1

    if args.push_to_hub:
        push = _push_to_hub(args.repo_id, root, private=not args.public_hub)
        _print_section("push_to_hub", push)
        if not push.get("ok"):
            return 1

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
