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

# =========================
# CSD + montage helpers
# =========================
def _has_valid_positions(info) -> bool:
    """Return True if EEG channels have non-NaN, non-zero xyz positions."""
    if info is None or "chs" not in info:
        return False
    found_valid = False
    for ch in info["chs"]:
        if ch.get("kind", None) is None:
            continue
        loc = np.array(ch.get("loc", []), dtype=float)
        if loc.size >= 3:
            xyz = loc[:3]
            if not np.any(np.isnan(xyz)) and not np.allclose(xyz, 0.0):
                found_valid = True
            else:
                # If any channel is NaN/zero, we consider positions invalid
                return False
    return found_valid

def _ensure_loaded(raw):
    if not raw.preload:
        raw.load_data()
    return raw

def _ensure_montage(raw):
    import mne
    # If montage/dig is missing OR positions look invalid, attach standard BioSemi-128.
    need_montage = (
        raw.get_montage() is None
        or raw.info.get("dig") is None
        or not _has_valid_positions(raw.info)
    )
    if need_montage:
        mon = mne.channels.make_standard_montage("biosemi128")
        raw.set_montage(mon, match_case=False, on_missing="ignore")
    return raw

def _apply_csd(raw):
    import mne
    _ensure_loaded(raw)
    _ensure_montage(raw)
    try:
        # Try auto-fit sphere (uses dig if available)
        mne.preprocessing.compute_current_source_density(raw, sphere="auto", copy=False)
    except RuntimeError as e:
        # Fallback: fixed sphere radius ≈ 94 mm (0.0942 m)
        print(f"[CSD] Falling back to fixed sphere due to: {e}")
        mne.preprocessing.compute_current_source_density(
            raw, sphere=(0.0, 0.0, 0.0, 0.0942), copy=False
        )
    return raw

# =========================
# Dataset + preprocessing
# =========================
def load_dataset_ccd(mini: bool, cache_dir: Path) -> BaseConcatDataset:
    ds = EEGChallengeDataset(
        task="contrastChangeDetection",
        release="R5",
        cache_dir=cache_dir,
        mini=bool(mini),
    )
    return ds

def preprocess_offline(dataset_ccd: BaseConcatDataset, use_csd: bool = True) -> BaseConcatDataset:
    """
    Apply offline preprocessing to the raw recordings before windowing.
    Includes robust montage fix + CSD by default.
    """
    tx = []

    # Always ensure data in memory & montage sane (positions non-NaN)
    tx += [
        Preprocessor(_ensure_loaded, apply_on_array=False),
        Preprocessor(_ensure_montage, apply_on_array=False),
    ]

    # Spatial sharpening (toggle with use_csd)
    if use_csd:
        tx += [Preprocessor(_apply_csd, apply_on_array=False)]

    # Challenge-specific target/anchor annotation
    tx += [
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

    preprocess(dataset_ccd, tx, n_jobs=1)

    # Keep only recordings that actually have the anchor of interest
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

    # same exclusions as startkit examples
    sub_rm = [
        "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
        "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV",
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
