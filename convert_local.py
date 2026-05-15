"""Convert raw save_state_img.py logs into a LeRobot dataset, locally.

Reads from ./Data/, writes to ./result/local/lebai_duck_pick/.
Run from the repo root:

    python convert_local.py
"""

import logging
import os
import shutil
import sys
from pathlib import Path

# Surface errors from lerobot's image writer (it logger.error()s and otherwise
# swallows them, which makes silent flush failures look like FileNotFoundError
# from compute_episode_stats. Make those visible so we can see what's going wrong.
logging.basicConfig(level=logging.WARNING,
                    format="[%(levelname)s] %(name)s: %(message)s")
logging.getLogger("lerobot.datasets.image_writer").setLevel(logging.DEBUG)

REPO_ROOT = Path(__file__).resolve().parent
LOG_ROOT = REPO_ROOT / "Data"
RESULT_ROOT = REPO_ROOT / "result"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

# Must be set before importing lerobot so the dataset lands in ./result/
os.environ["HF_LEROBOT_HOME"] = str(RESULT_ROOT)

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


# Configuration --------------------------------------------------------------

RUN_NUMBER = None                       # int to pick one log<NNNN>, None to merge all
ACTION_MODE = "relative"                # "absolute" = joint targets in rad
                                        # "relative" = delta-joint per tick in rad (recommended)

# When ACTION_MODE == "relative", multiply raw per-tick joint deltas by this
# factor before saving to the dataset. Raw 10 Hz deltas are typically ±0.05 rad,
# which trains slowly because the targets are tiny — scaling by 20× puts them
# in roughly ±1.0, a much better range for the policy network. The gripper
# dimension is left alone (it's an absolute 0–100 value, not a delta).
#
# At inference, run_inference.py divides the policy's predicted delta by this
# same factor before adding it to the current joint state. Converter + inference
# values MUST match; if you change it here, change ACTION_DELTA_SCALE in
# run_inference.py too.
ACTION_DELTA_SCALE = 100.0

if ACTION_MODE == "relative":
    REPO_ID = f"local/lebai_duck_pick_delta_x{int(ACTION_DELTA_SCALE)}"
else:
    REPO_ID = "local/lebai_duck_pick"

INCLUDE_WRIST = True                    # auto-disabled if no wrist data is present
INCLUDE_GRIPPER = True
FPS = 10
IMG_H, IMG_W = 480, 640                 # camera service resolution


# Helpers --------------------------------------------------------------------

def build_state(row):
    s = [float(row[f"jp{i}"]) for i in range(6)]
    if INCLUDE_GRIPPER:
        amp = row.get("claw_amplitude", 0.0)
        s.append(float(amp) if pd.notna(amp) else 0.0)
    return np.array(s, dtype=np.float32)


def build_action(row, next_row):
    """First 6 dims = joint commands (absolute targets OR scaled per-tick deltas, per ACTION_MODE).
    Last dim (if INCLUDE_GRIPPER) = next-frame gripper amplitude, always absolute (0–100).
    """
    if ACTION_MODE == "relative":
        # Action is the per-tick joint delta: next_jp - current_jp (radians).
        # At the last frame of an episode next_row == row, so delta = 0 (stay still).
        # Multiply by ACTION_DELTA_SCALE so the network sees roughly ±1 values
        # instead of ±0.05 — easier to learn, less affected by float precision.
        joints = [float((next_row[f"jp{i}"] - row[f"jp{i}"]) * ACTION_DELTA_SCALE) for i in range(6)]
    elif ACTION_MODE == "absolute":
        tgt = [row.get(f"tgt_jp{i}") for i in range(6)]
        if any(pd.isna(v) for v in tgt):
            joints = [float(next_row[f"jp{i}"]) for i in range(6)]
        else:
            joints = [float(v) for v in tgt]
    else:
        raise ValueError(f"ACTION_MODE must be 'absolute' or 'relative', got {ACTION_MODE!r}")

    if INCLUDE_GRIPPER:
        amp = next_row.get("claw_amplitude", row.get("claw_amplitude", 0.0))
        joints.append(float(amp) if pd.notna(amp) else 0.0)
    return np.array(joints, dtype=np.float32)


def load_rgb(run_dir, rel_path):
    full = Path(run_dir) / rel_path
    bgr = cv2.imread(str(full), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {full}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    # 1. Load CSV index(es)
    if RUN_NUMBER is not None:
        csv_paths = [LOG_ROOT / f"log{RUN_NUMBER:04d}.csv"]
    else:
        csv_paths = sorted(LOG_ROOT.glob("log[0-9][0-9][0-9][0-9].csv"))
    if not csv_paths:
        sys.exit(f"No log CSVs found under {LOG_ROOT}")
    print(f"CSVs to convert: {[p.name for p in csv_paths]}")

    dfs = []
    for p in csv_paths:
        d = pd.read_csv(p)
        d["__run_dir__"] = str(LOG_ROOT / p.stem)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(df)}")

    # 2. Wrist detection (treat empty / 'nan' strings as missing)
    wrist_strs = df["wrist"].fillna("").astype(str).str.strip()
    all_have_wrist = ((wrist_strs != "") & (wrist_strs.str.lower() != "nan")).all()
    has_wrist_data = INCLUDE_WRIST and all_have_wrist
    print(f"has_wrist_data = {has_wrist_data}")

    # 3. Build feature schema
    state_dim = 7 if INCLUDE_GRIPPER else 6
    action_dim = state_dim
    features = {
        "observation.images.base": {
            "dtype": "image",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": ["state"]},
        "action":            {"dtype": "float32", "shape": (action_dim,), "names": ["action"]},
    }
    if has_wrist_data:
        features["observation.images.wrist"] = {
            "dtype": "image",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        }

    # 4. Wipe any previous version of this dataset and create fresh
    out_path = RESULT_ROOT / REPO_ID
    if out_path.exists():
        print(f"Removing existing dataset at {out_path}")
        shutil.rmtree(out_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        robot_type="lebai_lm3",
        fps=FPS,
        features=features,
        # threads=0 / processes=0 disables image writes entirely in lerobot 0.5.1.
        # We need at least one worker. On 8 GB M1, 4 threads + 2 processes OOM'd —
        # 2 threads with no subprocess uses much less memory. Combined with the
        # explicit wait_until_done() drain below, conversion stays reliable.
        # Raise these on larger machines for speed.
        image_writer_threads=2,
        image_writer_processes=0,
    )
    print(f"Created empty dataset at {out_path}")
    print(f"  state_dim={state_dim}  action_dim={action_dim}  wrist={has_wrist_data}  action_mode={ACTION_MODE}")

    # 5. Conversion loop
    total_frames = 0
    total_episodes = 0
    groups = sorted(df.groupby(["__run_dir__", "episode"]), key=lambda kv: kv[0])

    for (run_dir, ep_idx), ep_df in groups:
        ep_df = ep_df.sort_values("frame").reset_index(drop=True)
        task = str(ep_df["task"].iloc[0])
        src_name = f"{Path(run_dir).name}/ep{ep_idx:03d}"

        if not task or task == "nan":
            print(f"  skip (empty task): {src_name}")
            continue
        if len(ep_df) < 5:
            print(f"  skip (only {len(ep_df)} frames): {src_name}")
            continue

        desc = f"{src_name} [{task[:30]}]"
        # Drain the writer queue every DRAIN_EVERY frames so it can't grow
        # unbounded — without this, on 8 GB M1 we OOM mid-episode at ~1000 frames.
        DRAIN_EVERY = 200
        iw = getattr(dataset.writer, "image_writer", None)
        for i in tqdm(range(len(ep_df)), desc=desc, leave=False):
            row = ep_df.iloc[i]
            next_row = ep_df.iloc[i + 1] if i + 1 < len(ep_df) else row
            frame = {
                "observation.images.base": load_rgb(row["__run_dir__"], row["color"]),
                "observation.state": build_state(row),
                "action": build_action(row, next_row),
                "task": task,
            }
            if has_wrist_data and isinstance(row["wrist"], str) and row["wrist"]:
                frame["observation.images.wrist"] = load_rgb(row["__run_dir__"], row["wrist"])
            dataset.add_frame(frame)
            if iw is not None and (i + 1) % DRAIN_EVERY == 0:
                iw.wait_until_done()

        # Final drain before save_episode — its compute_episode_stats reads
        # the PNGs back synchronously, so they must all be on disk.
        if iw is not None:
            iw.wait_until_done()

        dataset.save_episode()
        total_episodes += 1
        total_frames += len(ep_df)
        print(f"  saved {src_name}: {len(ep_df)} frames")

    print(f"\nDone. {total_episodes} episodes, {total_frames} frames.")
    print(f"Dataset written to {out_path}")
    print()
    print("NOTE: Do not call LeRobotDataset(REPO_ID) in the same Python process —")
    print("lerobot 0.5.1's async meta-parquet writer can lag the read by a beat and")
    print("you'll see a misleading 'Parquet magic bytes not found' error. Run:")
    print()
    print("    python verify_local.py")
    print()
    print("from a fresh shell to confirm the dataset is readable.")


if __name__ == "__main__":
    main()
