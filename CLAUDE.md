# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A three-notebook pipeline that trains an **ACT (Action Chunking Transformer)** policy on teleop demos from a **Lebai LM3** 6-DOF arm + parallel gripper and runs it on the real robot. There is no application code — all logic lives in the notebooks.

The pipeline (run in order):

1. [convert_alllog_to_lerobot.ipynb](convert_alllog_to_lerobot.ipynb) — converts raw collection logs into a LeRobot dataset under `$HF_LEROBOT_HOME` (default `~/.cache/huggingface/lerobot/`).
2. [act_training.ipynb](act_training.ipynb) — trains ACT on that dataset; saves checkpoints to `./checkpoints/act_run01/`.
3. [act_inference.ipynb](act_inference.ipynb) — loads the final checkpoint and runs a 10 Hz control loop against the live robot + camera service.

Two scripts are referenced but **not in this repo**: `save_state_img.py` (the upstream data collector that writes the CSV + per-frame JSON/JPEG archive) and `grasp_to_the_bowl.py` (the scripted-control reference whose SDK call pattern — `start_sys` → `init_claw` → non-blocking `movej` — the inference notebook mirrors).

## Data layout

Raw input expected by the converter:

```
<LOG_ROOT>/log<NNNN>.csv                       # flat per-frame index across all episodes
<LOG_ROOT>/log<NNNN>/ep<NNN>/color/000000.jpg  # base RGB
<LOG_ROOT>/log<NNNN>/ep<NNN>/wrist/000000.jpg  # optional wrist RGB
<LOG_ROOT>/log<NNNN>/ep<NNN>/state/000000.json # full SDK state snapshot
```

Demo data lives in [Data/](Data/) in this repo (gitignored) — 10 logs (`log0006`–`log0015`), one episode each, ~9.2k frames at 10 Hz, task `"put the duck in the bowl"`, no wrist camera. The converter defaults to `LOG_ROOT = Path("./Data")` and `RUN_NUMBER = None` (merge all runs). `REPO_ID = "local/lebai_duck_pick"` is shared between the converter and the training notebook's `DATASET_REPO_ID` — change both together.

## State and action conventions

These shapes are baked into all three notebooks and the trained checkpoint — changing one without the others will silently produce a broken policy:

- **`observation.state`**: `(7,)` float32 = `[jp0..jp5, claw_amplitude]` — current joint positions (rad) + current gripper opening (0–100).
- **`action`**: `(7,)` float32 = `[tgt_jp0..tgt_jp5, next_claw_amplitude]` — commanded joint targets + **next-frame** gripper amplitude.

The "next-frame gripper" trick compensates for the slow gripper actuator: the command issued at time `t` only shows up in the actual amplitude around `t+1`, so the converter pulls `claw_amplitude` from `next_row`. If `tgt_jp*` is empty (some firmwares don't populate it in teaching mode), `build_action` falls back to `next_row.jp*`.

`INCLUDE_GRIPPER = False` in the converter drops the 7th dim → both state and action become `(6,)`. All three notebooks must agree on this.

## Hyperparameters that aren't the defaults

ACT defaults assume **bimanual ALOHA at 50 Hz**. This setup is **6-DOF at 10 Hz**, so a few values are deliberately different from the LeRobot defaults — preserve them when editing:

- `chunk_size = 32` (≈3.2 s lookahead at 10 Hz; ALOHA uses 100 at 50 Hz = 2 s).
- `n_decoder_layers = 1` — matches the original ACT implementation, not the paper's claim of 7.
- `FPS = 10` throughout (converter, dataset metadata, inference loop period).
- Image resolution `(480, 640, 3)` — fixed by the camera service in `save_state_img.py`.

## Inference loop invariants

The control loop in [act_inference.ipynb](act_inference.ipynb) has a few non-obvious requirements:

- **Call `policy.reset()` before each run.** It clears the temporal-ensembling action queue; skipping it makes the first ~chunk_size actions reuse stale history.
- **`movej` is called *without* `wait_move()`.** The blend radius (`BLEND_RADIUS = 0.05`) is what makes successive non-blocking joint commands stitch together smoothly. Re-adding `wait_move()` blocks the 100 ms tick budget and the loop will fall behind.
- **Gripper commands are rate-limited** via `GRIPPER_THRESHOLD = 5.0`. The policy outputs continuous amplitudes; without thresholding, the gripper twitches every tick.
- **`start_sys()` and `init_claw()` must run before any motion command** — the SDK silently no-ops otherwise.
- Connection endpoints (`ROBOT_IP = 192.168.31.254`, `CAMERA_URL = http://192.168.31.192:8000`) are hard-coded; they must match the network the demos were recorded on, because the policy was trained on that exact camera.

## Running

These are notebooks, so there is no build/lint/test target. Execution = open in Jupyter (or VS Code) and run cells top-to-bottom. Before the first run install the deps (commented out at the top of each notebook):

```bash
pip install lerobot pandas opencv-python   # converter + training
pip install lebai_sdk requests             # inference only
```

Training needs a CUDA GPU with ≥8 GB VRAM (12 GB comfortable). The inference notebook needs the live Lebai arm and camera service reachable on the LAN.
