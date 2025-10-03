from pathlib import Path
import numpy as np

from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import (
    preprocess, Preprocessor, create_windows_from_events
)
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


def _avg_ref_and_ema(raw):
    """Lightweight, safe preprocessing:
       - Common average reference (EEG only)
       - Exponential moving standardization (per channel)"""
    import mne
    from braindecode.preprocessing import exponential_moving_standardize

    # ensure EEG only (challenge data are EEG), then CAR
    eeg_picks = mne.pick_types(raw.info, eeg=True, stim=False, eog=False, misc=False)
    raw.pick(eeg_picks, verbose="error")
    raw.load_data()
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose="error")

    # EMA standardization (Braindecode default kw set)
    X = raw.get_data()
    X_std = exponential_moving_standardize(
        X,
        factor_new=0.001,
        init_block_size=min(1000, X.shape[1]),
        eps=1e-4,
    )
    raw._data[:] = X_std
    return raw


def load_dataset_ccd(mini: bool, cache_dir: Path) -> BaseConcatDataset:
    ds = EEGChallengeDataset(
        task="contrastChangeDetection",
        release="R5",
        cache_dir=cache_dir,
        mini=bool(mini),
    )
    return ds


def preprocess_offline(dataset_ccd: BaseConcatDataset) -> BaseConcatDataset:
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
        Preprocessor(_avg_ref_and_ema, apply_on_array=False),
    ]
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
        keys=("target", "rt_from_stimulus", "rt_from_trialstart",
              "stimulus_onset", "response_onset", "correct", "response_type"),
    )
    return windows


def subject_splits(windows_ds: BaseConcatDataset, seed=2025, valid_frac=0.1, test_frac=0.1):
    """Subject-wise split, mirroring startkit behavior."""
    from sklearn.model_selection import train_test_split
    from sklearn.utils import check_random_state

    meta = windows_ds.get_metadata()
    subjects = meta["subject"].unique().tolist()

    # keep splits sane
    sub_rm = ["NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
              "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"]
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
