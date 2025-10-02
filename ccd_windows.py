from pathlib import Path
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

SFREQ = 100
EPOCH_LEN_S = 2.0
ANCHOR = "stimulus_anchor"
SHIFT_AFTER_STIM = 0.5
WINDOW_LEN = 2.0

def load_dataset_ccd(cache_dir: Path, mini: bool):
    cache_dir.mkdir(parents=True, exist_ok=True)
    return EEGChallengeDataset(task="contrastChangeDetection", release="R5",
                               cache_dir=cache_dir, mini=mini)

def preprocess_offline(dataset):
    transformation_offline = [
        Preprocessor(
            annotate_trials_with_target,
            target_field="rt_from_stimulus", epoch_length=EPOCH_LEN_S,
            require_stimulus=True, require_response=True,
            apply_on_array=False,
        ),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]
    preprocess(dataset, transformation_offline, n_jobs=1)

def make_windows(dataset):
    dataset = keep_only_recordings_with(ANCHOR, dataset)
    ws = create_windows_from_events(
        dataset,
        mapping={ANCHOR: 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),               # +0.5 s
        trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ), # +2.5 s
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
    )
    ws = add_extras_columns(
        ws, dataset, desc=ANCHOR,
        keys=("target", "rt_from_stimulus", "rt_from_trialstart",
              "stimulus_onset", "response_onset", "correct", "response_type"),
    )
    return ws

def subject_splits(windows, *, valid_frac=0.1, test_frac=0.1, seed=2025):
    from sklearn.model_selection import train_test_split
    from sklearn.utils import check_random_state

    subject_split = windows.split("subject")
    meta = windows.get_metadata()
    subjects = meta["subject"].unique()

    # same removals as startkit
    sub_rm = ["NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
              "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"]
    subjects = [s for s in subjects if s not in sub_rm]

    train_subj, valid_test_subject = train_test_split(
        subjects, test_size=(valid_frac + test_frac),
        random_state=check_random_state(seed), shuffle=True,
    )
    valid_subj, test_subj = train_test_split(
        valid_test_subject, test_size=test_frac,
        random_state=check_random_state(seed + 1), shuffle=True,
    )

    train_set, valid_set, test_set = [], [], []
    for s in subject_split:
        if s in train_subj:
            train_set.append(subject_split[s])
        elif s in valid_subj:
            valid_set.append(subject_split[s])
        elif s in test_subj:
            test_set.append(subject_split[s])

    return BaseConcatDataset(train_set), BaseConcatDataset(valid_set), BaseConcatDataset(test_set)
