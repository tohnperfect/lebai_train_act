"""Test a trained ACT model against the training dataset itself.

Two complementary checks:

  1. Forward-pass eval (--mode forward, default)
     For N random frames, runs policy.forward(batch) and compares the chunked
     prediction to the ground-truth chunk. Mirrors what training loss measures.
     Use this to confirm the policy can at least reproduce frames it was
     trained on.

  2. Open-loop rollout (--mode rollout)
     For a chosen episode, walks through every frame sequentially, calling
     policy.select_action(obs) the same way run_inference.py does. Returns a
     CSV of (frame, ground-truth action, predicted action) so you can compare
     trajectories visually.

Run from a fresh shell (NOT the same process the converter ran in):

    .venv/bin/python test_model_offline.py \
        --checkpoint checkpoints_2/act_run01/final \
        --repo-id local/lebai_duck_pick_delta_x20 \
        --n-frames 200

    .venv/bin/python test_model_offline.py \
        --checkpoint checkpoints_2/act_run01/final \
        --mode rollout --episode 0 \
        --out /tmp/rollout_ep0.csv

A healthy model:
  - per-dim L1 well below the std of the corresponding action dimension
  - mean prediction std (across the sample) close to ground-truth std
    (if pred std << gt std, the model collapsed to predicting a constant)
  - rollout CSV shows predicted trajectory visually tracking the ground truth

A collapsed model:
  - per-dim L1 ≈ std of the action (predictions are basically the mean)
  - pred std ≈ 0 (everything mapped to the same constant)
  - rollout trajectory is flat or nearly flat across an episode
"""

import argparse
import csv
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to a trained ACT checkpoint directory.")
    p.add_argument("--dataset-path", type=Path, default=Path("./result"),
                   help="HF_LEROBOT_HOME root. Default: ./result")
    p.add_argument("--repo-id", default=None,
                   help="Dataset repo-id under --dataset-path. If omitted, "
                        "guesses from the checkpoint's policy config.")
    p.add_argument("--mode", choices=["forward", "rollout"], default="forward",
                   help="forward: random-frame eval (default). rollout: one full episode.")
    p.add_argument("--n-frames", type=int, default=200,
                   help="Number of frames to sample in --mode forward. Default: 200")
    p.add_argument("--episode", type=int, default=0,
                   help="Episode index to roll out in --mode rollout. Default: 0")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for sampling.")
    p.add_argument("--out", type=Path, default=None,
                   help="CSV path for rollout output. Default: ./rollout_ep{N}.csv")
    return p.parse_args()


def make_policy_features(features_dict, FeatureType, PolicyFeature):
    out = {}
    for name, spec in features_dict.items():
        if name.startswith("observation.images."):
            ft = FeatureType.VISUAL
        elif name == "observation.state":
            ft = FeatureType.STATE
        elif name == "action":
            ft = FeatureType.ACTION
        else:
            continue
        out[name] = PolicyFeature(type=ft, shape=tuple(spec["shape"]))
    return out


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    os.environ["HF_LEROBOT_HOME"] = str(args.dataset_path.resolve())

    import numpy as np
    import torch
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.act.modeling_act import ACTPolicy
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.common.policies.act.modeling_act import ACTPolicy

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading {args.checkpoint}  (device: {device})")
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to(device).eval()
    chunk_size = policy.config.chunk_size
    expected_keys = sorted(policy.config.input_features.keys())
    print(f"  chunk_size={chunk_size}  expects={expected_keys}")

    # Guess repo-id from input_features if user didn't pass one
    repo_id = args.repo_id
    if repo_id is None:
        # No reliable signal in the checkpoint itself; warn and use a default.
        repo_id = "local/lebai_duck_pick_delta_x20"
        print(f"  WARN: --repo-id not given, defaulting to {repo_id!r}")

    # Load dataset
    base = LeRobotDataset(repo_id)
    fps = base.meta.fps
    delta_timestamps = {"action": [t / fps for t in range(chunk_size)]}
    ds = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
    print(f"\nDataset: {repo_id}")
    print(f"  episodes: {base.meta.total_episodes}  frames: {base.meta.total_frames}")

    if args.mode == "forward":
        forward_eval(args, ds, policy, device, np, torch)
    else:
        rollout_eval(args, ds, base, policy, device, np)


def forward_eval(args, ds, policy, device, np, torch):
    rng = np.random.default_rng(args.seed)
    n = min(args.n_frames, len(ds))
    indices = rng.choice(len(ds), size=n, replace=False)

    preds, targets = [], []
    print(f"\nRunning forward pass on {n} random frames ...")

    with torch.inference_mode():
        for k, idx in enumerate(indices):
            sample = ds[int(idx)]
            # For each sampled frame we want a fresh prediction (no temporal
            # ensembling carry-over from the previous random frame), so reset
            # the policy's internal action queue between calls.
            policy.reset()
            obs = {k_: v.unsqueeze(0).to(device) for k_, v in sample.items()
                   if k_.startswith("observation.")}
            pred = policy.select_action(obs)
            preds.append(pred[0].cpu().numpy())
            # First step of the chunk = ground-truth action for this frame.
            targets.append(sample["action"][0].cpu().numpy())
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{n}")

    preds = np.stack(preds)        # (n, action_dim)
    targets = np.stack(targets)    # (n, action_dim)

    # Per-dim metrics
    action_dim = preds.shape[1]
    print("\n=== Per-dimension stats ===")
    print(f"{'dim':>4}  {'gt_mean':>10}  {'gt_std':>10}  {'pred_mean':>10}  "
          f"{'pred_std':>10}  {'L1':>8}  {'L1/gt_std':>10}  {'corr':>6}")
    for d in range(action_dim):
        gt = targets[:, d]
        pr = preds[:, d]
        l1 = np.mean(np.abs(pr - gt))
        gt_std = gt.std()
        ratio = l1 / gt_std if gt_std > 1e-6 else float("inf")
        corr = np.corrcoef(pr, gt)[0, 1] if pr.std() > 1e-6 else 0.0
        label = "(gripper)" if d == 6 else ""
        print(f"{d:>4}  {gt.mean():10.4f}  {gt_std:10.4f}  {pr.mean():10.4f}  "
              f"{pr.std():10.4f}  {l1:8.4f}  {ratio:10.3f}  {corr:6.3f} {label}")

    # Aggregate
    overall_l1 = np.mean(np.abs(preds - targets))
    print(f"\nOverall mean L1: {overall_l1:.4f}")
    print()
    print("Interpretation:")
    print("  L1 / gt_std  <  ~0.3   ->  model is learning that dimension well")
    print("  L1 / gt_std  >  ~0.8   ->  predictions are nearly the dataset mean (collapse)")
    print("  pred_std    <<  gt_std ->  model collapsed; predicts a constant")
    print("  corr near 1            ->  predictions track ground truth")
    print("  corr near 0            ->  predictions uncorrelated with ground truth")


def rollout_eval(args, ds_chunked, ds_single, policy, device, np):
    import torch

    ep_meta = ds_single.meta.episodes[args.episode]
    fr = int(ep_meta["dataset_from_index"])
    to = int(ep_meta["dataset_to_index"])
    print(f"\nRolling out episode {args.episode}: frames [{fr}, {to})  "
          f"length={to-fr}")

    out_path = args.out or Path(f"./rollout_ep{args.episode}.csv")
    f = open(out_path, "w", newline="")
    w = csv.writer(f)

    action_dim = ds_single.features["action"]["shape"][0]
    state_dim = ds_single.features["observation.state"]["shape"][0]
    w.writerow(
        ["frame"]
        + [f"state_{i}" for i in range(state_dim)]
        + [f"gt_action_{i}" for i in range(action_dim)]
        + [f"pred_action_{i}" for i in range(action_dim)]
    )

    policy.reset()
    diffs = []
    with torch.inference_mode():
        for i, abs_idx in enumerate(range(fr, to)):
            sample = ds_single[abs_idx]
            obs = {
                k: v.unsqueeze(0).to(device) for k, v in sample.items()
                if k.startswith("observation.")
            }
            pred = policy.select_action(obs)
            pred_np = pred[0].cpu().numpy()
            gt = sample["action"].cpu().numpy()
            state = sample["observation.state"].cpu().numpy()
            w.writerow([i] + list(state) + list(gt) + list(pred_np))
            diffs.append(pred_np - gt)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{to - fr}")
    f.close()

    diffs = np.stack(diffs)
    print(f"\nRollout saved to {out_path}")
    print(f"Per-dim mean abs error (pred - gt): {np.round(np.mean(np.abs(diffs), axis=0), 4)}")
    print(f"Per-dim pred std:                   {np.round(np.array([np.std([row[d] for row in diffs]) for d in range(diffs.shape[1])]), 4)}")
    print()
    print("Quick visual: open the CSV in any plotting tool and overlay")
    print("  gt_action_0..5  vs  pred_action_0..5 over the frame axis.")
    print("  If predicted trajectories track the gt trajectories, the model learned.")
    print("  If predicted trajectories are flat (constant), the model collapsed.")


if __name__ == "__main__":
    main()
