from pathlib import Path
from typing import Tuple, Union, Sequence
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

# ---- constants to mirror startkit / your runner expectations ----
SFREQ = 100.0         # Hz (challenge downsampled rate)
WIN_SEC = 2.0         # seconds fed to the model (n_times)
EPOCH_LEN_S = 2.0     # sliding epoch size used by braindecode windowing
SHIFT_AFTER_STIM = 0.5  # seconds after stimulus anchor to start windowing
WINDOW_LEN = 2.0      # seconds spanned by the trial cut (stop - start)
ANCHOR = "stimulus_anchor"


def load_dataset_ccd(
    cache_dir: Union[str, Path],
    mini: bool = True,
    release: str = "R5",
) -> BaseConcatDataset:
    """
    Return the EEG Challenge dataset (CCD task), cached under cache_dir.
    """
    return EEGChallengeDataset(
        task="contrastChangeDetection",
        release=release,
        cache_dir=Path(cache_dir),
        mini=bool(mini),
    )


def preprocess_offline(dataset_ccd: BaseConcatDataset) -> None:
    """
    Offline preprocessing that injects 'target' and anchor events.
    Operates in-place on the provided BaseConcatDataset.
    """
    # 1) compute per-trial 'target' (rt_from_stimulus), require both stim & response
    # 2) add auxiliary anchors including 'stimulus_anchor'
    ops = [
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
    preprocess(dataset_ccd, ops, n_jobs=1)


def make_windows(
    dataset_ccd: BaseConcatDataset,
    anchor: str = ANCHOR,
) -> BaseConcatDataset:
    """
    Create windows locked to the given anchor and enrich their metadata
    (adds 'target', RTs, onsets, correctness, etc.).
    """
    # keep only recordings that actually contain the anchor
    dataset = keep_only_recordings_with(anchor, dataset_ccd)

    # convert timing (seconds) to samples
    start_off = int(SHIFT_AFTER_STIM * SFREQ)
    stop_off = int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ)
    win_size = int(EPOCH_LEN_S * SFREQ)
    stride = int(SFREQ)  # 1s stride, like startkit examples

    single_windows = create_windows_from_events(
        dataset,
        mapping={anchor: 0},
        trial_start_offset_samples=start_off,
        trial_stop_offset_samples=stop_off,
        window_size_samples=win_size,
        window_stride_samples=stride,
        preload=True,
    )

    # enrich the windows with extras including 'target'
    single_windows = add_extras_columns(
        single_windows,
        dataset,
        desc=anchor,
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
    return single_windows


def subject_splits(
    windows_ds: BaseConcatDataset,
    seed: int = 2025,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    exclude_subjects: Sequence[str] = (
        "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
        "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH",
    ),
) -> Tuple[BaseConcatDataset, BaseConcatDataset, BaseConcatDataset]:
    """
    Subject-wise split into train/valid/test. Uses numpy RNG (no sklearn dependency).
    """
    meta = windows_ds.get_metadata()
    subjects = np.array(sorted(set(meta["subject"].tolist())))
    # drop subjects known to cause split issues (mirrors startkit list)
    mask = np.isin(subjects, exclude_subjects, invert=True)
    subjects = subjects[mask]

    rng = np.random.RandomState(seed)
    perm = rng.permutation(subjects)
    n_total = len(perm)
    n_test = int(round(n_total * test_frac))
    n_valid = int(round(n_total * valid_frac))
    n_train = max(n_total - n_valid - n_test, 0)

    test_subj = set(perm[:n_test])
    valid_subj = set(perm[n_test:n_test + n_valid])
    train_subj = set(perm[n_test + n_valid: n_test + n_valid + n_train])

    split_by_subject = windows_ds.split("subject")
    tr, va, te = [], [], []
    for s, ds in split_by_subject.items():
        if s in train_subj:
            tr.append(ds)
        elif s in valid_subj:
            va.append(ds)
        elif s in test_subj:
            te.append(ds)

    return BaseConcatDataset(tr), BaseConcatDataset(va), BaseConcatDataset(te)
