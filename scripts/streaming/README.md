# `scripts/streaming/` — continuous-stream SO-101 recorder

Phase-1 implementation of the continuous-stream EXPLORE pipeline planned in
`canvas-autonomous-learner/PLAN.md`. Records back-to-back single-action
episodes into one LeRobot v3.0 dataset, eliminating the per-episode
`start_buffer / end_buffer / reset_time_s / verify_reset_position` overhead
that the legacy `run_single_action_record.py` pays.

## Why this exists

The legacy recorder spends ~21 wall-seconds per episode (measured across 76
historical EXPLORE bursts). About 7.5s of that is scripted dead time and the
remaining ~13.5s is per-episode video/parquet finalize plus reset retries.
For a 20-episode burst that's ~7-8 minutes, of which only ~3s/episode is the
actual moving-arm window we care about. Streaming the actions back-to-back
into a single LeRobot recording session removes most of that overhead.

Expected speedup at equivalent canvas yield: **2-5×**, growing with batch
size as fixed overhead amortizes.

## What this is NOT

- It is not a replacement for `run_single_action_record.py` — that script
  remains in place and the autonomous learner will continue to use it
  until Phase 2 wires the streaming path into `learner/explorer.py`.
- It is not a separate raw-then-split pipeline. The original plan called
  for two stages, but LeRobot v3.0 already chunks all episodes into a
  single MP4 + parquet pair via `streaming_encoding=True`, so single-stage
  recording is both simpler and equally fast.

## Files

```
scripts/streaming/
  __init__.py
  sequencer.py            # ActionSequencer — picks (joint, direction, target_pos)
                          # without per-episode reset.
  record_continuous.py    # CLI: connect hardware, drive sequencer, write
                          # via LeRobotDataset.create(streaming_encoding=True),
                          # call save_episode() per logical episode, optional
                          # push_to_hub.
  verify_dataset.py       # CLI: load dataset back via LeRobotDataset, run
                          # canvas-world-model's LeRobotV3Reader against it,
                          # optional push_to_hub round-trip.
  README.md               # this file
```

## CLI examples

### Smoke test (5 actions, local only)

```bash
python -m scripts.streaming.record_continuous \
    --robot-port=COM3 \
    --base-camera=1 --wrist-camera=0 \
    --num-actions=5 \
    --action-duration=1.0 --fps=10 \
    --joints shoulder_pan.pos \
    --output-repo-id=local/streaming-smoke
```

Output lands at `~/.cache/huggingface/lerobot/local/streaming-smoke/`.

### Two-joint pooled, 60 actions, pushed to HF

```bash
python -m scripts.streaming.record_continuous \
    --robot-port=COM3 \
    --base-camera=1 --wrist-camera=0 \
    --num-actions=60 \
    --action-duration=1.0 --fps=10 \
    --joints shoulder_pan.pos elbow_flex.pos \
    --joint-range shoulder_pan.pos -60 60 \
    --joint-range elbow_flex.pos 50 90 \
    --starting-positions-json='{"shoulder_pan": 0, "shoulder_lift": -100, "elbow_flex": 70, "wrist_flex": 30, "wrist_roll": 0, "gripper": 50}' \
    --output-repo-id=irvinh/streaming-shoulder-elbow-60 \
    --push-to-hub
```

`--push-to-hub` requires `HF_TOKEN` in the environment with write access to
the namespace.

### Verify a recorded dataset

```bash
# Local layout + canvas-world-model loader compatibility
python -m scripts.streaming.verify_dataset \
    --root ~/.cache/huggingface/lerobot/local/streaming-smoke \
    --repo-id local/streaming-smoke

# Also exercise push_to_hub
python -m scripts.streaming.verify_dataset \
    --root ~/.cache/huggingface/lerobot/irvinh/streaming-shoulder-elbow-60 \
    --repo-id irvinh/streaming-shoulder-elbow-60 \
    --push-to-hub
```

The verifier walks four checks:

1. **Layout** — `meta/info.json`, `data/chunk-000/`, `videos/`,
   `meta/episodes/`, `meta/discrete_action_logs/` all present.
2. **`LeRobotDataset` load** — round-trips via the official constructor,
   decodes one frame to confirm video shapes match `info.json`.
3. **`canvas-world-model` loader** — imports `data.lerobot_loader.LeRobotV3Reader`
   from the sibling repo, confirms the dataset is consumable, validates
   the discrete-action logs are present and decision-bearing.
4. **Push to hub** (optional) — only when `--push-to-hub` is passed.

If checks 1-3 pass, the dataset is HF-uploadable and consumable by
`canvas-world-model/create_dataset.py`. The push step is the same code
path the autonomous learner will use later in Phase 2.

## Hardware safety

- Connect/disconnect is in a `finally` block — interrupted runs still
  release the FeetechMotorsBus and DSHOW camera handles. (See the
  `feedback_camera_shutdown.md` memory: never `taskkill /F` a camera
  owner; always disconnect cleanly.)
- A periodic verification checkpoint (default every 10 actions) reads
  motor positions and logs drift versus the last commanded target.
  This is the streaming equivalent of the legacy
  `verify_reset_position` retries, but cheaper since it only chats with
  the bus rather than running a settle + retry loop per episode.
- The sequencer clamps every target to the configured per-joint
  `(lo, hi)` range. Pass `--joint-range <joint> <lo> <hi>` per joint;
  defaults applied for unspecified joints.

## Compatibility with `canvas-world-model/create_dataset.py`

Streaming-recorded datasets are bit-compatible with what
`canvas-world-model/create_dataset.py` consumes:

- Per-episode `meta/discrete_action_logs/episode_NNNNNN.jsonl` files
  with the same header + per-frame entry schema as the legacy recorder.
- `action` discrete codes match: `1=positive`, `2=negative`, `3=none`,
  `0=hold` — the same mapping used by `SingleActionPolicy.select_action`.
- `task` strings match the legacy
  `"Move <joint_friendly> <direction> by <delta> units"` template, so
  any downstream parser already handles them.
- Camera feature keys (`observation.images.<base_name>`,
  `observation.images.<wrist_name>`) and motor schemas
  (`action`, `observation.state` as 6-vector float32) are identical.

## Phase status

- **Phase 1 — done.** Standalone recorder + verifier on hardware
  (this directory).
- **Phase 2 — done.** `canvas-autonomous-learner` now has
  `learner.explorer.collect_batch_continuous()` and a
  `cadence.continuous_explore: true` config flag. The orchestrator
  dispatches between the legacy episodic recorder and this streaming
  recorder based on that flag. Verifier path always uses legacy
  (streaming sequencer doesn't support `probe_script`). Default is
  `false` in `configs/simultaneous.yaml` until Phase 3.
- **Phase 3 — pending.** Overnight A/B soak with two arms (one legacy,
  one streaming), measuring wall time and val MSE convergence.
- **Phase 4 — pending.** Flip the default to `true` in the production
  configs, leave the legacy path as a one-cycle fallback, then delete
  it.

## Open items before Phase 2

1. **30+ second OpenCV+DSHOW stability** — the streaming session
   keeps cameras open continuously. Validate on hardware before
   committing the learner-integration code.
2. **Frame timestamps** — the recorder paces with `time.perf_counter()`
   so dropped camera frames are surfaced via timestamp gaps rather
   than silently shifting the action-frame index. Check
   `verify_dataset.py` output for any gaps after a real run.
3. **Verification cadence tuning** — default `--verify-every=10` is a
   placeholder. The right value depends on observed motor drift in
   real recordings.
