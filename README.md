# Lebai_train_ACT

Train an **ACT (Action Chunking Transformer)** policy on teleoperated demonstrations of a **Lebai LM3** 6-DOF arm + parallel gripper, and run the trained policy on the real robot.

The repository is a three-notebook pipeline: convert raw collection logs → train ACT → run inference on the live arm.

## What's in the repo

| Path | Purpose |
|---|---|
| [convert_alllog_to_lerobot.ipynb](convert_alllog_to_lerobot.ipynb) | Local notebook: convert raw `save_state_img.py` logs into a [LeRobot](https://github.com/huggingface/lerobot) dataset. |
| [act_training.ipynb](act_training.ipynb) | Local notebook: train an ACT policy on that dataset; writes checkpoints to `./checkpoints/`. |
| [act_inference.ipynb](act_inference.ipynb) | Load the final checkpoint and drive the arm in a real-time 10 Hz control loop. |
| [convert_local.py](convert_local.py) / [verify_local.py](verify_local.py) | Headless command-line converter + verifier — same logic as the notebook, used to produce `result/` for upload to Colab. |
| [convert_alllog_to_lerobot_colab.ipynb](convert_alllog_to_lerobot_colab.ipynb) | Colab variant of the converter (reads from Drive). |
| [act_training_colab.ipynb](act_training_colab.ipynb) | Colab variant of the trainer (extracts dataset from Drive tarball, saves checkpoints to Drive). |
| [recover_lerobot_dataset.py](recover_lerobot_dataset.py) | One-off recovery for older LeRobot dataset layouts (legacy; v0.5+ format is different — see Troubleshooting). |
| [Data/](Data/) | Raw demonstrations (gitignored). 10 logs of "put the duck in the bowl", ~9.2k frames at 10 Hz. |
| [CLAUDE.md](CLAUDE.md) | Notes for the Claude Code assistant — also useful as a quick technical reference. |

`save_state_img.py` (the upstream data collector) and `grasp_to_the_bowl.py` (a scripted-control reference) live in a separate Lebai data-collection repo, not here.

## Data layout

Each demonstration session produces one `log<NNNN>` directory and a matching CSV index:

```
Data/log<NNNN>.csv                       # flat per-frame index across all episodes in this run
Data/log<NNNN>/ep<NNN>/color/000000.jpg  # base RGB
Data/log<NNNN>/ep<NNN>/wrist/000000.jpg  # optional wrist RGB
Data/log<NNNN>/ep<NNN>/state/000000.json # full SDK state snapshot
Data/log<NNNN>/ep<NNN>/episode_meta.json # task string, frame count, timestamps
```

The included `Data/` covers logs `0006`–`0015`, one episode each, all labelled `"put the duck in the bowl"`, no wrist camera.

## State and action

Both shaped `(7,)`:

- **observation.state** = `[jp0..jp5, claw_amplitude]` — current joint positions (rad) + current gripper opening (0–100).
- **action** = `[tgt_jp0..tgt_jp5, next_claw_amplitude]` — commanded joint targets + the *next-frame* gripper amplitude (compensates for the slow gripper actuator).

Drop `INCLUDE_GRIPPER` to `False` in the converter for `(6,)` state/action. All three notebooks must agree on this choice.

## Requirements

- Python 3.10+
- A CUDA GPU with ≥8 GB VRAM for training (12 GB comfortable)
- Linux/macOS, Jupyter or VS Code
- For inference only: the live Lebai arm + camera service reachable on the LAN

Install:

```bash
pip install lerobot pandas opencv-python matplotlib tqdm   # converter + training
pip install lebai_sdk requests                             # inference only
```

## How to use this repo

### Step 1 — Convert raw logs to a LeRobot dataset

Open [convert_alllog_to_lerobot.ipynb](convert_alllog_to_lerobot.ipynb) and run every cell top to bottom.

Defaults already point at this repo's data:

```python
LOG_ROOT    = Path("./Data")
RUN_NUMBER  = None                  # merges all logs under Data/
REPO_ID     = "local/lebai_duck_pick"
INCLUDE_WRIST   = True              # auto-disabled if no wrist data
INCLUDE_GRIPPER = True
FPS = 10
```

What you should see:

- The episode summary cell prints one row per `(run, episode)` — 10 rows for the bundled data.
- The single-frame sanity check renders the first image and prints state/action with a small non-zero delta.
- The conversion loop reports `saved log<NNNN>/ep000: NNNN frames` per episode and ends with `10 episodes, ~9245 frames total`.

The dataset is written to `~/.cache/huggingface/lerobot/local/lebai_duck_pick/` (or wherever `$HF_LEROBOT_HOME` points). The converter wipes that directory at the start of each run, so re-running is safe.

### Step 2 — Train ACT

Open [act_training.ipynb](act_training.ipynb) and run every cell top to bottom.

`DATASET_REPO_ID` already matches the converter's `REPO_ID`. Key hyperparameters tuned for 10 Hz / 6-DOF (different from the LeRobot/ACT defaults that assume bimanual ALOHA at 50 Hz):

| Setting | Value | Why |
|---|---|---|
| `CHUNK_SIZE` | 32 | ≈3.2 s of action lookahead at 10 Hz |
| `n_decoder_layers` | 1 | Matches the original ACT implementation |
| `BATCH_SIZE` | 8 | Fits in 12 GB VRAM; raise to 16/32 on bigger cards |
| `NUM_STEPS` | 50,000 | Sensible default for ~10 episodes; extend if loss is still descending |
| `LR` | 1e-4 (vision backbone 1e-5 internally) | Standard for ACT |

What you should see:

- Visualization cells render the first image plus state/action trajectories — state and action should be *related but not identical*, joints should look smooth, gripper should make discrete open/close transitions.
- Loss drops rapidly in the first few thousand steps, then flattens.
- Intermediate checkpoints land in `./checkpoints/act_run01/step_NNNNNN/`; the final one in `./checkpoints/act_run01/final/`.

Watching loss alone is misleading — always re-check the visualization plots after editing the converter.

### Step 3 — Run inference on the real robot

Open [act_inference.ipynb](act_inference.ipynb) **only when the arm is in a safe pose and the e-stop is within reach.**

Set these to your network before running:

```python
ROBOT_IP   = "192.168.31.254"
CAMERA_URL = "http://192.168.31.192:8000"
```

Other defaults — `CHECKPOINT_PATH = Path("./checkpoints/act_run01/final")`, `PERIOD_S = 0.1`, conservative joint velocity/acceleration limits — line up with the training notebook.

The notebook walks through, in order:

1. Load the checkpoint, print its expected input/output shapes.
2. Open the camera service and connect to the Lebai SDK (`start_sys` then `init_claw`).
3. Define `build_observation()` and `send_action()` helpers.
4. **Dry run** — predict one action and print it without moving the robot. Catch shape and normalization bugs here, *not* during motion.
5. Control loop at 10 Hz with temporal ensembling. Stops automatically after `MAX_DURATION_S` seconds or on `Ctrl-C` / Jupyter Interrupt.
6. Cleanup: `cam.stop_all()`, `lebai.stop_sys()`.

The pre-loop **safety checklist** is non-optional:

1. Clear workspace, no one within sweep range.
2. E-stop within arm's reach of the operator.
3. Pendant velocity factor turned **down** for the first runs.
4. Start the robot in a pose similar to one of the training episodes' first frames.

## Running on Google Colab (recommended for training)

Inference (Step 3) cannot run on Colab — it needs LAN access to the Lebai SDK and camera service. For Steps 1 and 2, the **recommended flow is local-convert + Colab-train**: conversion writes thousands of small files which is slow and corruption-prone on Drive, while training writes a few large checkpoints which Drive handles well.

### 1. Convert locally

From the repo root (Python 3.10+):

```bash
python3.13 -m venv .venv               # or any Python ≥3.10
.venv/bin/pip install lerobot pandas opencv-python tqdm
.venv/bin/python convert_local.py      # reads Data/, writes result/
.venv/bin/python verify_local.py       # reads result/ in a fresh process
```

`verify_local.py` is intentionally a separate process — `lerobot 0.5.x` has an async meta-parquet writer whose flush lags `save_episode()`, producing a misleading `Parquet magic bytes not found in footer` error if you try to re-open the dataset in the same Python process the converter ran in.

You should see `Dataset OK.` and `10 episodes, 9246 frames` (for the bundled data).

### 2. Pack and upload to Drive

```bash
cd result && tar -czf lebai_duck_pick.tar.gz local/lebai_duck_pick
# ~3.5 GB; upload via drag-drop to MyDrive/Lebai_train_ACT/
```

### 3. Train on Colab

Open [act_training_colab.ipynb](act_training_colab.ipynb), set runtime to **GPU**, run all cells. The paths cell:

- Extracts `MyDrive/Lebai_train_ACT/lebai_duck_pick.tar.gz` to `/content/lerobot_cache/` (Colab SSD, fast).
- Sets `HF_LEROBOT_HOME` to the local SSD path.
- Points `CHECKPOINT_DIR` at `MyDrive/Lebai_train_ACT/checkpoints/act_run01/` on Drive (persists across runtime resets).

`RESUME = True` (cell 7) picks up from the latest `step_*` checkpoint, so a Colab disconnect mid-training is recoverable. Once `final/` is saved on Drive, download it to a machine on the robot's LAN and point [act_inference.ipynb](act_inference.ipynb)'s `CHECKPOINT_PATH` at it.

### Alternative: Drive-only Colab flow

[convert_alllog_to_lerobot_colab.ipynb](convert_alllog_to_lerobot_colab.ipynb) does conversion entirely inside Colab (reads `Data/` from Drive, writes the dataset to Drive). Slower and more fragile due to the many-small-writes corruption risk — kept around for cases where you'd rather not have a local Python environment, but use the local converter when you can.

## Switching to your own data

1. Drop your `save_state_img.py` output into `Data/` (or change `LOG_ROOT`).
2. If you collected with a wrist camera, leave `INCLUDE_WRIST = True` — the converter auto-detects.
3. Pick a new `REPO_ID` and set the same value in `DATASET_REPO_ID` in the training notebook.
4. Re-run all three notebooks.

If your task has clearly multi-modal demonstrations (e.g. "grasp from either side"), ACT may average the modes into a failing motion. Same dataset works with LeRobot's Diffusion Policy class as a drop-in replacement.
