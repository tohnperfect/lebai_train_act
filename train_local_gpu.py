"""Train ACT on a local GPU box.

Headless equivalent of `act_training_colab.ipynb`. Reads the LeRobot dataset
from a local directory, trains, writes checkpoints to a local directory.
No Drive, no Colab, no tarballs.

Examples:

    # Sanity check the install (200 steps, batch 4, no checkpoint saves).
    python train_local_gpu.py --smoke-test

    # Full training run, defaults match the Colab notebook.
    python train_local_gpu.py

    # Override hyperparameters.
    python train_local_gpu.py --num-steps 80000 --batch-size 32 --num-workers 8

    # Resume from the latest step_* checkpoint under --checkpoint-dir
    # (auto-detected; set --no-resume to force a fresh run).
    python train_local_gpu.py --no-resume

Run setup_gpu_env.sh once before the first run to install deps.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset-path", type=Path,
                   default=Path("./result"),
                   help="Directory used as HF_LEROBOT_HOME. The dataset must live "
                        "at <dataset-path>/<repo-id>/. Default: ./result")
    p.add_argument("--repo-id", default="local/lebai_duck_pick_delta_x100",
                   help="Dataset repo id under HF_LEROBOT_HOME. Default: local/lebai_duck_pick_delta_x100")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("./checkpoints/act_run01"),
                   help="Where to save step_* and final/ checkpoints. Default: ./checkpoints/act_run01")
    p.add_argument("--chunk-size", type=int, default=32,
                   help="ACT action chunk size. Default: 32 (≈3.2 s at 10 Hz)")
    p.add_argument("--num-steps", type=int, default=50_000,
                   help="Total training steps. Default: 50,000")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Per-iteration batch size. Default: 16 (raise to 32+ on >24 GB GPUs).")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers. Default: 4. Use 8+ on machines with many CPU cores.")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="AdamW learning rate. Default: 1e-4")
    p.add_argument("--log-every", type=int, default=200, help="Log every N steps. Default: 200")
    p.add_argument("--save-every", type=int, default=5_000, help="Checkpoint every N steps. Default: 5,000")
    p.add_argument("--no-resume", action="store_true",
                   help="Force fresh training even if step_* checkpoints exist.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Tiny run: 200 steps, batch 4, no saves, prints loss progression. "
                        "Use to verify the install + dataset before committing to full training.")
    return p.parse_args()


def make_policy_features(features_dict, FeatureType, PolicyFeature):
    """Wrap dataset features into PolicyFeature objects (required by lerobot 0.5+)."""
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


def find_latest_checkpoint(checkpoint_dir: Path):
    if not checkpoint_dir.exists():
        return None, 0
    ckpts = sorted(p for p in checkpoint_dir.glob("step_*") if p.is_dir())
    if not ckpts:
        return None, 0
    latest = ckpts[-1]
    try:
        step = int(latest.name.split("_")[1])
    except (IndexError, ValueError):
        return None, 0
    return latest, step


def main():
    args = parse_args()
    if args.smoke_test:
        args.num_steps = 200
        args.batch_size = 4
        args.log_every = 20
        args.save_every = 10**9       # never save during smoke test
        args.no_resume = True

    os.environ["HF_LEROBOT_HOME"] = str(args.dataset_path.resolve())

    # Imports must come AFTER HF_LEROBOT_HOME is set so lerobot picks it up.
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
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: no CUDA GPU. Training will be very slow on CPU.")

    # 1. Load dataset
    dataset_root = args.dataset_path / args.repo_id
    if not dataset_root.exists():
        sys.exit(f"Dataset not found at {dataset_root}\n"
                 f"Run convert_local.py first, or pass --dataset-path / --repo-id correctly.")

    meta = LeRobotDataset(args.repo_id).meta
    fps = meta.fps
    print(f"\nDataset: {args.repo_id}")
    print(f"  episodes: {meta.total_episodes}")
    print(f"  frames:   {meta.total_frames}")
    print(f"  fps:      {fps}")
    print(f"  features: {list(meta.features.keys())}")

    delta_timestamps = {"action": [t / fps for t in range(args.chunk_size)]}
    dataset = LeRobotDataset(args.repo_id, delta_timestamps=delta_timestamps)
    print(f"  training frames: {len(dataset)} (each yields a chunk of {args.chunk_size})")

    # 2. Build / resume policy
    all_feats = make_policy_features(dataset.features, FeatureType, PolicyFeature)
    input_features = {k: v for k, v in all_feats.items() if k != "action"}
    output_features = {"action": all_feats["action"]}

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt, start_step = (None, 0)
    if not args.no_resume:
        latest_ckpt, start_step = find_latest_checkpoint(args.checkpoint_dir)

    if latest_ckpt is not None:
        print(f"\nResuming from {latest_ckpt} (start_step={start_step})")
        policy = ACTPolicy.from_pretrained(latest_ckpt)
    else:
        print("\nBuilding ACT policy from scratch")
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
            n_decoder_layers=1,    # original ACT impl uses 1, not 7 from the paper
            # CVAE disabled: previous run at kl_weight=10.0 collapsed the latent to 0
            # by step 5000 and the decoder learned to output the data mean.
            # use_vae=False makes ACT a deterministic transformer over action chunks —
            # fine for single-task / unimodal demos like duck-in-bowl. To re-enable,
            # set use_vae=True with kl_weight=1.0 (NOT 10.0).
            use_vae=False,
            latent_dim=32,
            n_vae_encoder_layers=4,
            kl_weight=1.0,
            dropout=0.1,
        )
        cfg.input_features = input_features
        cfg.output_features = output_features
        policy = ACTPolicy(cfg, dataset_stats=dataset.meta.stats)

    policy.to(device).train()
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy: {n_params/1e6:.1f}M parameters")

    # 3. DataLoader + optimizer
    pin = device.type == "cuda"
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    def cycle(loader):
        while True:
            for b in loader:
                yield b
    step_iter = cycle(dataloader)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)
    for _ in range(start_step):
        scheduler.step()

    # 4. Training loop
    loss_csv = args.checkpoint_dir / "loss_history.csv"
    write_header = not loss_csv.exists() or start_step == 0
    f_loss = open(loss_csv, "a" if start_step > 0 else "w", newline="")
    loss_writer = csv.writer(f_loss)
    if write_header:
        loss_writer.writerow(["step", "loss", "l1_loss", "kld_loss", "lr"])
        f_loss.flush()

    print(f"\nTraining: steps {start_step} -> {args.num_steps}  bs={args.batch_size}  "
          f"-> {args.checkpoint_dir.resolve()}")
    if args.smoke_test:
        print("*** SMOKE TEST MODE: 200 steps, no checkpoints saved. ***")

    t0 = time.time()
    try:
        for step in range(start_step, args.num_steps):
            batch = next(step_iter)
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            loss, loss_dict = policy.forward(batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()

            loss_writer.writerow([
                step, f"{loss.item():.6f}",
                f"{loss_dict.get('l1_loss', 0):.6f}",
                f"{loss_dict.get('kld_loss', 0):.6f}",
                f"{scheduler.get_last_lr()[0]:.6e}",
            ])
            if step % args.log_every == 0:
                f_loss.flush()
                elapsed = time.time() - t0
                rate = (step - start_step + 1) / max(elapsed, 1e-6)
                eta_s = (args.num_steps - step) / max(rate, 1e-6)
                print(f"step {step:6d}  loss={loss.item():.4f}  "
                      f"l1={loss_dict.get('l1_loss', 0):.3f}  "
                      f"kld={loss_dict.get('kld_loss', 0):.3f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"({rate:.1f} step/s, eta {eta_s/60:.1f} min)")

            if (step + 1) % args.save_every == 0:
                ckpt_path = args.checkpoint_dir / f"step_{step+1:06d}"
                policy.save_pretrained(ckpt_path)
                print(f"  -> saved {ckpt_path}")
    finally:
        f_loss.close()

    if not args.smoke_test:
        final_ckpt = args.checkpoint_dir / "final"
        policy.save_pretrained(final_ckpt)
        print(f"\nDone. Final checkpoint: {final_ckpt}")
        print(f"Loss history CSV:        {loss_csv}")
    else:
        print(f"\nSmoke test done. Loss CSV (not saved as checkpoint): {loss_csv}")
        print("Inspect loss_history.csv to confirm the loss decreased over the 200 steps.")


if __name__ == "__main__":
    main()
