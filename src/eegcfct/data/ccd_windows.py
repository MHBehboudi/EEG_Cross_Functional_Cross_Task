# src/eegcfct/data/ccd_windows.py

from pathlib import Path
from typing import Tuple

import numpy as np
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events

from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

__all__ = [
    "SFREQ",
    "load_dataset_ccd",
    "preprocess_offline",
    "make_windows",
    "subject_splits",
]

# --------------------
# Constants (match startkit)
# --------------------
SFREQ = 100          # downsampled to 100 Hz in challenge data
N_CHANS = 129        # BioSemi 129 montage (informational)
WIN_SEC = 2.0        # 2 seconds → 200 samples
EPOCH_LEN_S = 2.0    # model input length for EEGNeX
SHIFT_AFTER_STIM = 0.5
WINDOW_LEN = 2.0
ANCHOR = "stimulus_anchor"


# --------------------
# Optional CSD helpers (SAFE)
# --------------------
def _ensure_loaded(raw):
    if not raw.preload:
        raw.load_data()
    return raw


def _ensure_montage(raw):
    """Attach a reasonable standard montage if none is present."""
    import mne
    mon = raw.get_montage()
    if mon is None or raw.info.get("dig") is None:
        try:
            mon = mne.channels.make_standard_montage("biosemi128")
            raw.set_montage(mon, match_case=False, on_missing="ignore")
        except Exception:
            # If even this fails, just continue without montage
            pass
    return raw


def _has_valid_positions(raw) -> bool:
    """Return True if most EEG channels have finite, non-zero 3D positions."""
    import mne
    picks = mne.pick_types(raw.info, eeg=True)
    if len(picks) == 0:
        return False
    pos = np.array([raw.info["chs"][pi]["loc"][:3] for pi in picks])
    if pos.size == 0 or not np.all(np.isfinite(pos)):
        return False
    ok = (np.linalg.norm(pos, axis=1) > 1e-6)
    # consider valid if >80% channels look reasonable
    return ok.mean() > 0.8


def _apply_csd(raw):
    """Compute current source density when positions are valid; otherwise skip."""
    import mne
    _ensure_loaded(raw)
    _ensure_montage(raw)
    if not _has_valid_positions(raw):
        print("[CSD] Skipping CSD: no valid channel positions.")
        return raw
    try:
        mne.preprocessing.compute_current_source_density(raw, sphere="auto", copy=False)
    except Exception as e:
        print(
            f"[CSD] Auto sphere failed ({type(e).__name__}: {e}). "
            f"Falling back to fixed sphere radius 94.2 mm."
        )
        # Fixed sphere (meters): origin=(0,0,0), radius=0.0942
        mne.preprocessing.compute_current_source_density(
            raw, sphere=(0.0, 0.0, 0.0, 0.0942), copy=False
        )
    return raw


# --------------------
# Dataset loading & preprocessing
# --------------------
def load_dataset_ccd(mini: bool, cache_dir: Path) -> BaseConcatDataset:
    """Load the EEGChallengeDataset for Contrast Change Detection (mini/full)."""
    ds = EEGChallengeDataset(
        task="contrastChangeDetection",
        release="R5",
        cache_dir=cache_dir,
        mini=bool(mini),
    )
    return ds


def preprocess_offline(dataset_ccd: BaseConcatDataset, use_csd: bool = False) -> BaseConcatDataset:
    """Apply startkit-like offline preprocessing (annotate targets, add anchors).

    CSD is **off by default** for stability. Enable via `use_csd=True` once you
    confirm channel positions are valid on your machine.
    """
    tx = [
        Preprocessor(
            annotate_trials_with_target,
            target_field="rt_from_stimulus",
            epoch_length=EPOCH_LEN_S,
            require_stimulus=True,
            require_response=True,
            apply_on_array=False,
        ),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]

    if use_csd:
        tx.append(Preprocessor(_apply_csd, apply_on_array=False))

    preprocess(dataset_ccd, tx, n_jobs=1)

    # keep only recordings that contain the desired anchor
    dataset = keep_only_recordings_with(ANCHOR, dataset_ccd)
    return dataset


def make_windows(dataset: BaseConcatDataset) -> BaseConcatDataset:
    """Create stimulus-locked windows and inject metadata columns (incl. 'target')."""
    windows = create_windows_from_events(
        dataset,
        mapping={ANCHOR: 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
        trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
    )
    windows = add_extras_columns(
        windows,
        dataset,
        desc=ANCHOR,
        keys=(
            "target",
            "rt_from_stimulus",
            "rt_from_trialstart",
            "stimulus_onset",
            "response_onset",
            "correct",
            "response_type",
        ),
    )
    return windows


def subject_splits(
    windows_ds: BaseConcatDataset, seed: int = 2025, valid_frac: float = 0.1, test_frac: float = 0.1
) -> Tuple[BaseConcatDataset, BaseConcatDataset, BaseConcatDataset]:
    """Subject-wise split, mirroring startkit behavior."""
    from sklearn.model_selection import train_test_split
    from sklearn.utils import check_random_state

    meta = windows_ds.get_metadata()
    subjects = meta["subject"].unique().tolist()

    # Filter list used in startkit examples
    sub_rm = [
        "NDARWV769JM7",
        "NDARME789TD2",
        "NDARUA442ZVF",
        "NDARJP304NK1",
        "NDARTY128YLU",
        "NDARDW550GU6",
        "NDARLD243KRE",
        "NDARUJ292JXV",
        "NDARBA381JGH",
    ]
    subjects = [s for s in subjects if s not in sub_rm]

    rng = check_random_state(seed)
    train_subj, valid_test_subject = train_test_split(
        subjects, test_size=(valid_frac + test_frac), random_state=rng, shuffle=True
    )
    valid_subj, test_subj = train_test_split(
        valid_test_subject, test_size=test_frac, random_state=check_random_state(seed + 1), shuffle=True
    )

    split_by_subject = windows_ds.split("subject")
    tr, va, te = [], [], []
    for s in split_by_subject:
        if s in train_subj:
            tr.append(split_by_subject[s])
        elif s in valid_subj:
            va.append(split_by_subject[s])
        elif s in test_subj:
            te.append(split_by_subject[s])

    return BaseConcatDataset(tr), BaseConcatDataset(va), BaseConcatDataset(te)
