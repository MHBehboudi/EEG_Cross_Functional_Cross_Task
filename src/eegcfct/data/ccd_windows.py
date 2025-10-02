from __future__ import annotations
from pathlib import Path
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

# ------- constants (match startkit) -------
SFREQ = 100          # downsampled to 100 Hz in challenge data
N_CHANS = 129        # BioSemi 129 montage
WIN_SEC = 2.0        # 2 seconds → 200 samples
EPOCH_LEN_S = 2.0    # model input length
SHIFT_AFTER_STIM = 0.5
WINDOW_LEN = 2.0
ANCHOR = "stimulus_anchor"


# ======================
# CSD helper functions
# ======================
def _ensure_loaded(raw):
    if not raw.preload:
        raw.load_data()
    return raw


def _maybe_attach_biosemi(raw):
    """Attach standard montage if missing (most HBN CCD use E1..E129 labels)."""
    import mne
    mon = mne.channels.make_standard_montage("biosemi128")
    try:
        raw.set_montage(mon, match_case=False, on_missing="ignore")
    except Exception as e:
        print(f"[CSD] Warning: set_montage failed: {e}")
    return raw


def _positions_ok(info) -> bool:
    """True if we have finite, non-zero EEG channel positions."""
    import mne
    picks = mne.pick_types(info, eeg=True, meg=False)
    if picks.size == 0:
        return False
    loc = np.array([info["chs"][p]["loc"][:3] for p in picks], float)
    if loc.ndim != 2 or loc.shape[1] != 3:
        return False
    if not np.all(np.isfinite(loc)):
        return False
    if (np.linalg.norm(loc, axis=1) == 0).any():
        return False
    return True


def _apply_csd_with_fallback(raw, sphere_mode: str = "fixed") -> bool:
    """
    Try CSD; return True if applied, False if skipped.
    Modes:
      - 'fixed': uses a fixed sphere radius (0.0942 m) for robustness
      - 'auto' : fit sphere to headshape (can fail if dig is incomplete)
    """
    import mne
    _ensure_loaded(raw)
    _maybe_attach_biosemi(raw)

    if not _positions_ok(raw.info):
        print("[CSD] Skipping: invalid or missing EEG positions.")
        return False

    try:
        if sphere_mode == "auto":
            mne.preprocessing.compute_current_source_density(raw, sphere="auto", copy=False)
        else:
            # Fixed sphere center at origin, radius ≈ 94.2 mm
            mne.preprocessing.compute_current_source_density(
                raw, sphere=(0.0, 0.0, 0.0, 0.0942), copy=False
            )
        return True
    except Exception as e:
        print(f"[CSD] Skipping due to error: {e}")
        return False


def _maybe_csd(raw, sphere_mode: str = "fixed"):
    """Wrapper used by Braindecode Preprocessor."""
    _apply_csd_with_fallback(raw, sphere_mode=sphere_mode)
    return raw


# ======================
# Dataset + windows API
# ======================
def load_dataset_ccd(mini: bool, cache_dir: Path) -> BaseConcatDataset:
    ds = EEGChallengeDataset(
        task="contrastChangeDetection",
        release="R5",
        cache_dir=cache_dir,
        mini=bool(mini),
    )
    return ds


def preprocess_offline(
    dataset_ccd: BaseConcatDataset,
    use_csd: bool = False,
    csd_sphere: str = "fixed",   # 'fixed' (robust default) or 'auto'
) -> BaseConcatDataset:
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
        tx.insert(0, Preprocessor(_maybe_csd, sphere_mode=csd_sphere, apply_on_array=False))

    preprocess(dataset_ccd, tx, n_jobs=1)
    dataset = keep_only_recordings_with(ANCHOR, dataset_ccd)
    return dataset


def make_windows(dataset: BaseConcatDataset) -> BaseConcatDataset:
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


def subject_splits(windows_ds: BaseConcatDataset, seed=2025, valid_frac=0.1, test_frac=0.1):
    """Subject-wise split, mirroring startkit behavior."""
    from sklearn.model_selection import train_test_split
    from sklearn.utils import check_random_state

    meta = windows_ds.get_metadata()
    subjects = meta["subject"].unique().tolist()

    # Same removals as our startkit-style pipeline
    sub_rm = [
        "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
        "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"
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
