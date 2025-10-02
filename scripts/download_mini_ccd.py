#!/usr/bin/env python3
from pathlib import Path
import argparse
from eegdash.dataset import EEGChallengeDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true", help="download the mini release (fast)")
    parser.add_argument("--release", type=str, default="R5")
    parser.add_argument("--task", type=str, default="contrastChangeDetection")
    parser.add_argument("--data_dir", type=str, default=None, help="where to cache data")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir) if args.data_dir else (repo_root / "data")
    data_dir.mkdir(parents=True, exist_ok=True)

    ds = EEGChallengeDataset(task=args.task, release=args.release, cache_dir=data_dir, mini=bool(args.mini))
    print(f"Downloaded/validated: {len(ds)} recordings in {data_dir}")

if __name__ == "__main__":
    main()
