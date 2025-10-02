from pathlib import Path
from typing import Tuple
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

SFREQ = 100          # challenge data resampled to 100 Hz
N_CHANS = 129
WIN_SEC = 2.0
EPOCH_LEN_S = 2.0
SHIFT_AFTER_STIM = 0.5
WINDOW_LEN = 2.0
ANCHOR = "stimulus_anchor"

def make_windows_ccd(dataset_ccd: BaseConcatDataset) -> BaseConcatDataset:
    """Create windows and inject extras/targets like the startkit."""
    transformation_offline = [
        Preprocessor(
            annotate_trials_with_target,
            target_field="rt_from_stimulus", epoch_length=EPOCH_LEN_S,
            require_stimulus=True, require_response=True,
            apply_on_array=False,
        ),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]
    preprocess(dataset_ccd, transformation_offline, n_jobs=1)

    dataset = keep_only_recordings_with(ANCHOR, dataset_ccd)

    single_windows = create_windows_from_events(
        dataset,
        mapping={ANCHOR: 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
        trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
    )

    single_windows = add_extras_columns(
        single_windows,
        dataset,
        desc=ANCHOR,
        keys=("target", "rt_from_stimulus", "rt_from_trialstart",
              "stimulus_onset", "response_onset", "correct", "response_type"),
    )

    return single_windows


def subject_split_concat(
    windows_ds: BaseConcatDataset, seed: int = 2025, valid_frac: float = 0.1, test_frac: float = 0.1
) -> Tuple[BaseConcatDataset, BaseConcatDataset, BaseConcatDataset]:
    from sklearn.model_selection import train_test_split
    from sklearn.utils import check_random_state

    meta = windows_ds.get_metadata()
    subjects = meta["subject"].unique()

    # filter set used by the startkit examples
    sub_rm = ["NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
              "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"]
    subjects = [s for s in subjects if s not in sub_rm]

    train_subj, valid_test_subject = train_test_split(
        subjects, test_size=(valid_frac + test_frac),
        random_state=check_random_state(seed), shuffle=True
    )
    valid_subj, test_subj = train_test_split(
        valid_test_subject, test_size=test_frac,
        random_state=check_random_state(seed + 1), shuffle=True
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
