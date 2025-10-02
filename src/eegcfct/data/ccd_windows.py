from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
import mne
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_windows_from_events,
)
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

# --------------------
# Constants (match startkit/challenge data)
# --------------------
SFREQ = 100          # downsampled to 100 Hz in challenge data
N_CHANS = 129        # BioSemi 129 (as provided in challenge data)
WIN_SEC = 2.0        # model input length (seconds)
EPOCH_LEN_S = 2.0    # sliding window size (same as input)
SHIFT_AFTER_STIM = 0.5
WINDOW_LEN = 2.0     # total window span from anchor
ANCHOR = "stimulus_anchor"


# --------------------
# Position-free preprocessing helpers
# --------------------
def _ensure_loaded(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    if not raw.preload:
        raw.load_data()
    return raw


def _pick_eeg(raw: mne.io.BaseRaw) -> None:
    """Keep EEG channels only (drop EOG/EMG/ECG/STIM/MISC)."""
    raw.pick_types(eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)


def _avg_reference(raw: mne.io.BaseRaw) -> None:
    """Common average reference (does not require channel positions)."""
    raw.set_eeg_reference("average", projection=False)


def _clip_uV(raw: mne.io.BaseRaw, max_abs_uV: float = 150.0) -> None:
    """Winsorize amplitudes to ±max_abs_uV µV to tame spikes/outliers."""
    _ensure_loaded(raw)
    if max_abs_uV is None or max_abs_uV <= 0:
        return
    thr = float(max_abs_uV) * 1e-6  # convert to Volts
    data = raw.get_data()
    np.clip(data, -thr, thr, out=data)


def _ema_standardize(
    raw: mne.io.BaseRaw,
    factor_new: float = 1e-3,
    init_block_size: int = 1000,
) -> None:
    """Exponential moving standardization per channel (online z-score)."""
    from braindecode.preprocessing import exponential_moving_standardize
    _ensure_loaded(raw)
    X = raw.get_data()
    X_std = exponential_moving_standardize(
        X,
        factor_new=factor_new,
        init_block_size=init_block_size,
        per_channel=True,
    )
    raw._data[:] = X_std


# --------------------
# Optional CSD (OFF by default)
# --------------------
def _attach_biosemi_if_missing(raw: mne.io.BaseRaw) -> None:
    """If montage/dig is missing, attach a standard BioSemi montage (best-effort)."""
    try:
        mon = raw.get_montage()
    except Exception:
        mon = None
    if mon is None or raw.info.get("dig") is None:
        std = mne.channels.make_standard_montage("biosemi128")
        # best-effort: ignore unknown names; we only need finite positions for CSD
        raw.set_montage(std, match_case=False, on_missing="ignore")


def _apply_csd(
    raw: mne.io.BaseRaw,
    sphere: str | tuple = "auto",
) -> None:
    """Surface Laplacian / CSD. Requires valid (finite) channel positions."""
    _ensure_loaded(raw)
    _attach_biosemi_if_missing(raw)
    try:
        mne.preprocessing.compute_current_source_density(raw, sphere=sphere, copy=False)
    except Exception as e:
        raise RuntimeError(f"CSD failed: {e}")


# --------------------
# Dataset utilities
# --------------------
def load_dataset_ccd(mini: bool, cache_dir: Path) -> BaseConcatDataset:
    """Load the challenge dataset (HBN CCD)."""
    return EEGChallengeDataset(
        task="contrastChangeDetection",
        release="R5",
        cache_dir=cache_dir,
        mini=bool(mini),
    )


def preprocess_offline(
    dataset_ccd: BaseConcatDataset,
    *,
    # position-free steps (ON by default)
    use_avg_ref: bool = True,
    clip_uv: float = 150.0,
    use_ema_std: bool = True,
    ema_factor_new: float = 1e-3,
    ema_init_block: int = 1000,
    # CSD (OFF by default)
    use_csd: bool = False,
    csd_sphere: str = "auto",   # or "fixed" to use a fixed sphere
) -> BaseConcatDataset:
    """
    Apply safe, position-free preprocessing + (optionally) CSD, then
    annotate/anchor like the startkit.
    """
    tx: list[Preprocessor] = []

    # --- Position-free steps (robust baseline) ---
    tx.append(Preprocessor(_pick_eeg, apply_on_array=False))
    if use_avg_ref:
        tx.append(Preprocessor(_avg_reference, apply_on_array=False))
    if clip_uv and clip_uv > 0:
        tx.append(Preprocessor(_clip_uV, max_abs_uV=float(clip_uv), apply_on_array=False))
    if use_ema_std:
        tx.append(
            Preprocessor(
                _ema_standardize,
                factor_new=float(ema_factor_new),
                init_block_size=int(ema_init_block),
                apply_on_array=False,
            )
        )

    # --- Optional CSD ---
    if use_csd:
        sphere = "auto" if csd_sphere == "auto" else (0.0, 0.0, 0.0, 0.0942)
        tx.append(Preprocessor(_apply_csd, sphere=sphere, apply_on_array=False))

    # --- Startkit-like target annotation & anchors ---
    tx.extend(
        [
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
    )

    # Execute preprocessing
    preprocess(dataset_ccd, tx, n_jobs=1)
    dataset = keep_only_recordings_with(ANCHOR, dataset_ccd)
    return dataset


def make_windows(dataset: BaseConcatDataset) -> BaseConcatDataset:
    """Create windows / epochs anchored to stimulus with the challenge timing."""
    windows = create_windows_from_events(
        dataset,
        mapping={ANCHOR: 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
        trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
    )
    # enrich with metadata columns (includes 'target')
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
    windows_ds: BaseConcatDataset,
    seed: int = 2025,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
) -> Tuple[BaseConcatDataset, BaseConcatDataset, BaseConcatDataset]:
    """Subject-wise split (mirrors startkit behavior)."""
    from sklearn.model_selection import train_test_split
    from sklearn.utils import check_random_state

    meta = windows_ds.get_metadata()
    subjects = meta["subject"].unique().tolist()

    # filter list used in the startkit examples to keep splits sane
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
        valid_test_subject,
        test_size=test_frac,
        random_state=check_random_state(seed + 1),
        shuffle=True,
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
