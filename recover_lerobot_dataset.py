"""
Recover a partially-corrupted LeRobot dataset.

When a writer (Colab session, training job, anything) is killed mid-`save_episode()`,
one parquet shard ends up with a missing footer. After that, opening the dataset
with `LeRobotDataset(REPO_ID)` fails with:

    ArrowInvalid: Parquet magic bytes not found in footer.

This script scans every episode parquet, finds the longest contiguous-good prefix,
and trims the dataset (parquets, videos, episodes.jsonl, stats, info.json) back
to that point so resume can pick up cleanly.

Usage in Colab (after mounting Drive):

    !python recover_lerobot_dataset.py \
        /content/drive/MyDrive/Lebai_train_ACT/lerobot_cache/local/lebai_duck_pick

Usage as a notebook cell — just paste this file's body and set DATASET_PATH
at the top.
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


def recover(dataset_path: Path, dry_run: bool = False) -> int:
    """Trim the dataset back to the last fully-readable episode.

    Returns the number of episodes kept.
    """
    data_dir = dataset_path / "data"
    meta_dir = dataset_path / "meta"

    parquets = sorted(data_dir.rglob("episode_*.parquet")) if data_dir.exists() else []
    print(f"Found {len(parquets)} parquet files under {data_dir}")

    last_good = -1
    for p in parquets:
        idx = int(p.stem.split("_")[-1])
        try:
            pq.ParquetFile(p)
        except Exception as e:
            print(f"  corrupt parquet at episode {idx}: {p.name}: {e}")
            break
        if idx == last_good + 1:
            last_good = idx
        else:
            print(f"  gap before episode {idx} (last good was {last_good})")
            break

    keep = last_good + 1
    print(f"\nLast contiguous good episode: {last_good}  ->  keeping {keep} episodes\n")

    if dry_run:
        print("(dry run — no files modified)")
        return keep

    # Delete parquets past keep
    for p in parquets:
        if int(p.stem.split("_")[-1]) >= keep:
            print(f"  rm {p.relative_to(dataset_path)}")
            p.unlink()

    # Delete videos past keep (if any — newer LeRobot versions store video)
    videos_dir = dataset_path / "videos"
    if videos_dir.exists():
        for v in list(videos_dir.rglob("*.mp4")):
            try:
                idx = int(v.stem.split("_")[-1])
            except (ValueError, IndexError):
                continue
            if idx >= keep:
                print(f"  rm {v.relative_to(dataset_path)}")
                v.unlink()

    # Truncate episodes.jsonl
    ej = meta_dir / "episodes.jsonl"
    if ej.exists():
        lines = [l for l in ej.read_text().splitlines() if l.strip()]
        if len(lines) > keep:
            new_text = ("\n".join(lines[:keep]) + "\n") if keep > 0 else ""
            ej.write_text(new_text)
            print(f"  episodes.jsonl truncated to {keep} lines")

    # Truncate per-episode stats files
    for name in ("episodes_stats.jsonl", "stats.jsonl"):
        f = meta_dir / name
        if not f.exists():
            continue
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        if len(lines) > keep:
            new_text = ("\n".join(lines[:keep]) + "\n") if keep > 0 else ""
            f.write_text(new_text)
            print(f"  {name} truncated to {keep} lines")

    # Fix info.json totals
    info_path = meta_dir / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        info["total_episodes"] = keep
        if ej.exists() and keep > 0:
            kept = [json.loads(l) for l in ej.read_text().splitlines() if l.strip()]
            info["total_frames"] = sum(e.get("length", 0) for e in kept)
        else:
            info["total_frames"] = 0
        info_path.write_text(json.dumps(info, indent=2))
        print(
            f"  info.json: total_episodes={keep} total_frames={info.get('total_frames')}"
        )

    print("\nRecovery complete. Re-run the build cell and then the loop cell.")
    return keep


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(
            "usage: python recover_lerobot_dataset.py <dataset_path> [--dry-run]"
        )
    path = Path(sys.argv[1]).expanduser()
    dry = "--dry-run" in sys.argv[2:]
    if not path.exists():
        sys.exit(f"error: {path} does not exist")
    recover(path, dry_run=dry)
