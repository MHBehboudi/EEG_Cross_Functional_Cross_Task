# src/eegcfct/config.py
from pathlib import Path
import os

def get_data_root() -> Path:
    # Use env var if set; otherwise default to repo-root/data_cache/eeg_challenge_cache
    env = os.getenv("EEG2025_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # repo root = src/eegcfct -> up 2 -> project root
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "data_cache" / "eeg_challenge_cache").resolve()
