"""Hyperparameter sweep for ACT — short training runs to identify configs
that don't collapse.

Designed to answer the question: "Given my dataset, which combination of
learning rate, kl_weight, and use_vae trains a useful policy without
collapsing to constant predictions?"

For each config in the SWEEP list:
  - Builds a fresh ACTPolicy + AdamW + cosine LR scheduler
  - Trains for --steps-per-config (default 2500) on the same dataset
  - Logs per-step loss to sweep_results/<config_name>/loss_history.csv
  - No checkpoints saved (we only care about the loss trajectory here)

At the end prints a summary table:

  config        lr       use_vae kl_w   final_loss  mid_loss  drop%   status
  baseline      1e-4     False   -      0.450       0.780     42      ✓ healthy
  vae_kl10      1e-4     True    10.0   3.300       3.310     0       ✗ collapsed

A "healthy" config has:
  - final loss noticeably below mid loss (still dropping at the end)
  - final loss below 1.0 (or whatever you'd expect for normalized L1)
  - no NaN/inf

A "collapsed" config has:
  - flat loss curve (final ≈ mid)
  - loss stuck around 3.0+ (mean-prediction baseline)

Usage from the GPU box:

    .venv/bin/python hp_sweep.py                      # default: 2500 steps per config
    .venv/bin/python hp_sweep.py --steps-per-config 1500
    .venv/bin/python hp_sweep.py --configs baseline,vae_kl1  # only run these
    .venv/bin/python hp_sweep.py --batch-size 8 --num-workers 2  # smaller GPU

After it finishes, the summary table tells you which configs to try in a
real 50k-step training run. Take the best 1-2 and run train_local_gpu.py
with those hyperparameters.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# The sweep itself. Edit this list to add / remove / reorder configurations.

SWEEP = [
    # name        lr      use_vae  kl_weight
    ("baseline",  1e-4,   False,   1.0),          # most likely to work — current default
    ("lr_low",    3e-5,   False,   1.0),          # smaller LR, slower but safer
    ("lr_high",   3e-4,   False,   1.0),          # bigger LR, may be unstable
    ("vae_kl01",  1e-4,   True,    0.1),          # CVAE with very light KL pressure
    ("vae_kl1",   1e-4,   True,    1.0),          # CVAE with moderate KL pressure
    ("vae_kl10",  1e-4,   True,    10.0),         # CVAE at the original ACT default — known to collapse
]
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset-path", type=Path, default=Path("./result"),
                   help="HF_LEROBOT_HOME root. Default: ./result")
    p.add_argument("--repo-id", default="local/lebai_duck_pick_delta_x100",
                   help="Dataset repo id. Default: local/lebai_duck_pick_delta_x100")
    p.add_argument("--sweep-dir", type=Path, default=Path("./sweep_results"),
                   help="Where to write per-config loss histories + summary. Default: ./sweep_results")
    p.add_argument("--steps-per-config", type=int, default=2500,
                   help="Training steps per hyperparameter combo. Default: 2500 "
                        "(enough to clearly distinguish collapse vs. learning, "
                        "small enough that 6 configs fit in ~1h on a T4).")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size used for every config. Default: 16")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers. Default: 4")
    p.add_argument("--chunk-size", type=int, default=32,
                   help="ACT chunk size. Held constant across configs. Default: 32")
    p.add_argument("--configs", default=None,
                   help="Comma-separated config names to run (default: all in SWEEP). "
                        "Useful to re-run a subset or skip ones you've already tested.")
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


def train_one_config(name, lr, use_vae, kl_weight, args, dataset, dataset_stats,
                     input_features, output_features, device, ACTConfig, ACTPolicy,
                     torch, DataLoader, np):
    """Train a single config for args.steps_per_config and return a metrics dict."""

    cfg = ACTConfig(
        n_obs_steps=1,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        replace_final_stride_with_dilation=False,
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        n_encoder_layers=4,
        n_decoder_layers=1,
        use_vae=use_vae,
        latent_dim=32,
        n_vae_encoder_layers=4,
        kl_weight=kl_weight,
        dropout=0.1,
    )
    cfg.input_features = input_features
    cfg.output_features = output_features

    policy = ACTPolicy(cfg, dataset_stats=dataset_stats).to(device).train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps_per_config)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        drop_last=True, persistent_workers=args.num_workers > 0,
    )

    def cycle(loader):
        while True:
            for b in loader:
                yield b
    step_iter = cycle(loader)

    out_dir = args.sweep_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_csv = out_dir / "loss_history.csv"
    f = open(loss_csv, "w", newline="")
    w = csv.writer(f)
    w.writerow(["step", "loss", "l1_loss", "kld_loss"])

    print(f"\n--- {name}: lr={lr}, use_vae={use_vae}, kl_weight={kl_weight} ---")
    t0 = time.time()
    losses = []
    last_log = 0
    failed = False
    fail_reason = ""

    try:
        for step in range(args.steps_per_config):
            batch = next(step_iter)
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            loss, loss_dict = policy.forward(batch)
            if not torch.isfinite(loss):
                failed = True
                fail_reason = f"non-finite loss at step {step}"
                break

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            w.writerow([
                step, f"{loss.item():.6f}",
                f"{loss_dict.get('l1_loss', 0):.6f}",
                f"{loss_dict.get('kld_loss', 0):.6f}",
            ])

            if step - last_log >= 250 or step == args.steps_per_config - 1:
                f.flush()
                elapsed = time.time() - t0
                rate = (step + 1) / max(elapsed, 1e-6)
                window = losses[-min(50, len(losses)):]
                avg_recent = sum(window) / len(window)
                print(f"  step {step:5d}  recent_avg_loss={avg_recent:.4f}  "
                      f"({rate:.1f} step/s)")
                last_log = step
    finally:
        f.close()

    metrics = summarize_run(losses, failed, fail_reason)
    metrics.update(dict(name=name, lr=lr, use_vae=use_vae, kl_weight=kl_weight))
    return metrics


def summarize_run(losses, failed, fail_reason):
    if failed or len(losses) == 0:
        return dict(
            final_loss=float("nan"), mid_loss=float("nan"),
            drop_pct=float("nan"), status="FAILED",
            reason=fail_reason or "no steps completed",
        )

    n = len(losses)
    w = max(50, n // 20)
    final_window = losses[-w:]
    mid_start = n // 2 - w // 2
    mid_window = losses[mid_start:mid_start + w]

    final_loss = sum(final_window) / len(final_window)
    mid_loss = sum(mid_window) / len(mid_window)
    drop_pct = (mid_loss - final_loss) / mid_loss * 100.0 if mid_loss > 1e-6 else 0.0

    # Classify
    if final_loss > 2.5 and abs(drop_pct) < 5.0:
        status = "COLLAPSED"
        reason = "flat loss curve above the mean-prediction baseline (~3.3)"
    elif drop_pct > 15.0 and final_loss < 2.0:
        status = "HEALTHY"
        reason = "loss still dropping at end, well below collapse threshold"
    elif drop_pct > 5.0:
        status = "OK"
        reason = "loss dropping but slowly"
    elif final_loss < 1.0:
        status = "PLATEAU(LOW)"
        reason = "stopped dropping but at a low value — may already be converged"
    else:
        status = "MARGINAL"
        reason = "loss not dropping meaningfully and not at a low value"

    return dict(
        final_loss=final_loss, mid_loss=mid_loss,
        drop_pct=drop_pct, status=status, reason=reason,
    )


def write_summary(results, args):
    summary_path = args.sweep_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "lr", "use_vae", "kl_weight",
                    "final_loss", "mid_loss", "drop_pct", "status", "reason"])
        for r in results:
            w.writerow([
                r["name"], r["lr"], r["use_vae"], r["kl_weight"],
                f"{r['final_loss']:.4f}" if r['final_loss'] == r['final_loss'] else "nan",
                f"{r['mid_loss']:.4f}"   if r['mid_loss']   == r['mid_loss']   else "nan",
                f"{r['drop_pct']:.1f}"   if r['drop_pct']   == r['drop_pct']   else "nan",
                r["status"], r["reason"],
            ])

    print("\n" + "=" * 100)
    print(f"{'name':<12} {'lr':>8} {'vae':>5} {'kl_w':>6} "
          f"{'final':>9} {'mid':>9} {'drop%':>7} {'status':<14} reason")
    print("-" * 100)
    for r in results:
        fl = f"{r['final_loss']:.3f}" if r['final_loss'] == r['final_loss'] else "  NaN"
        ml = f"{r['mid_loss']:.3f}"   if r['mid_loss']   == r['mid_loss']   else "  NaN"
        dp = f"{r['drop_pct']:5.1f}"  if r['drop_pct']   == r['drop_pct']   else "  NaN"
        kl = f"{r['kl_weight']:.2f}" if r['use_vae'] else "  -  "
        print(f"{r['name']:<12} {r['lr']:>8.0e} {str(r['use_vae']):>5} {kl:>6} "
              f"{fl:>9} {ml:>9} {dp:>7} {r['status']:<14} {r['reason']}")
    print("=" * 100)
    print(f"\nFull summary written to {summary_path}")
    print(f"Per-config loss histories under {args.sweep_dir}/<name>/loss_history.csv")
    print()
    healthy = [r for r in results if r["status"] in ("HEALTHY", "OK", "PLATEAU(LOW)")]
    if healthy:
        best = min(healthy, key=lambda r: r["final_loss"])
        print(f"Best candidate: {best['name']!r}  (final loss {best['final_loss']:.4f})")
        print(f"  -> kick off a full training run with:")
        kl_arg = f"--kl-weight {best['kl_weight']}" if best["use_vae"] else ""
        vae_arg = "--use-vae" if best["use_vae"] else ""
        print(f"     (in train_local_gpu.py, set lr={best['lr']}, use_vae={best['use_vae']}, "
              f"kl_weight={best['kl_weight']} in the ACTConfig)")
    else:
        print("WARNING: no healthy configs. Likely causes:")
        print("  - dataset issue (re-run verify_local.py and inspect action distributions)")
        print("  - chunk_size too small relative to action variance")
        print("  - try use_vae=False with even smaller lr (e.g. 1e-5)")


def main():
    args = parse_args()

    # Filter configs if --configs was passed
    sweep = SWEEP
    if args.configs:
        wanted = set(args.configs.split(","))
        sweep = [c for c in SWEEP if c[0] in wanted]
        if not sweep:
            sys.exit(f"No configs matched --configs={args.configs}. "
                     f"Available: {[c[0] for c in SWEEP]}")
        print(f"Running {len(sweep)} of {len(SWEEP)} configs: {[c[0] for c in sweep]}")

    args.sweep_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_LEROBOT_HOME"] = str(args.dataset_path.resolve())

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.configs.types import FeatureType, PolicyFeature
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.common.policies.act.modeling_act import ACTPolicy
        from lerobot.common.policies.act.configuration_act import ACTConfig
        from lerobot.configs.types import FeatureType, PolicyFeature

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: no GPU detected. Sweep will be very slow on CPU.")
        print("         Each config = ~30+ min on CPU vs. ~5–15 min on GPU.")
        print("         Consider reducing --steps-per-config to 500 for CPU runs.")

    # Load dataset ONCE — share across configs
    base = LeRobotDataset(args.repo_id)
    fps = base.meta.fps
    print(f"Dataset: {args.repo_id} ({base.meta.total_episodes} episodes, "
          f"{base.meta.total_frames} frames, {fps} fps)")
    delta_timestamps = {"action": [t / fps for t in range(args.chunk_size)]}
    dataset = LeRobotDataset(args.repo_id, delta_timestamps=delta_timestamps)
    dataset_stats = dataset.meta.stats

    all_feats = make_policy_features(dataset.features, FeatureType, PolicyFeature)
    input_features = {k: v for k, v in all_feats.items() if k != "action"}
    output_features = {"action": all_feats["action"]}

    print(f"Sweep: {len(sweep)} configs × {args.steps_per_config} steps each")
    print(f"Output: {args.sweep_dir}/")

    results = []
    t_start_all = time.time()
    for cfg_tuple in sweep:
        name, lr, use_vae, kl_weight = cfg_tuple
        try:
            metrics = train_one_config(
                name, lr, use_vae, kl_weight, args, dataset, dataset_stats,
                input_features, output_features, device,
                ACTConfig, ACTPolicy, torch, DataLoader, np,
            )
        except Exception as e:
            metrics = dict(
                name=name, lr=lr, use_vae=use_vae, kl_weight=kl_weight,
                final_loss=float("nan"), mid_loss=float("nan"),
                drop_pct=float("nan"),
                status="ERRORED", reason=f"{type(e).__name__}: {str(e)[:80]}",
            )
            print(f"  ERROR in {name}: {e}")
        results.append(metrics)

    total = time.time() - t_start_all
    print(f"\nSweep complete in {total / 60:.1f} min")
    write_summary(results, args)


if __name__ == "__main__":
    main()
