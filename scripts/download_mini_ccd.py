# scripts/download_mini_ccd.py
from pathlib import Path
from eegdash import EEGChallengeDataset
from eegcfct.config import get_data_root

def main():
    cache_dir = get_data_root()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading/caching to: {cache_dir}")

    # Mini CCD (R5) cache — same preprocessing as the challenge
    EEGChallengeDataset(
        release="R5",
        task="contrastChangeDetection",
        mini=True,
        description_fields=[
            "subject","session","run","task","age","gender","sex","p_factor"
        ],
        cache_dir=cache_dir,
    )
    print("Done. Mini CCD available.")

if __name__ == "__main__":
    main()
