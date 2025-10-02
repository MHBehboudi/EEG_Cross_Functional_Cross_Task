from pathlib import Path
from functools import partial

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
N_CHANS = 129        # BioSemi 129 montage (channels labeled E1..E129)
WIN_SEC = 2.0        # 2 seconds → 200 samples fed to the model
EPOCH_LEN_S = 2.0    # model input length (seconds)
SHIFT_AFTER_STIM = 0.5
WINDOW_LEN = 2.0
ANCHOR = "stimulus_anchor"


# ---------- CSD helpers ----------
def _ensure_loaded(raw):
    """Ensure data is in memory (MNE requirement for CSD)."""
    if not raw.preload:
        raw.load_data()
    return raw


def _ensure_montage(raw):
    """Attach a standard BioSemi montage if montage/dig points are missing."""
    import mne
    need_montage = (raw.get_montage() is None) or (raw.info.get("dig") is None)
    if need_montage:
        mon = mne.channels.make_standard_montage("biosemi128")
        # We map by names; many HBN channels are E1..E129 and align well with BioSemi layout.
        raw.set_montage(mon, match_case=False, on_missing="ignore")
    return raw


def _apply_csd(raw, sphere_mode="auto"):
    """Compute surface Laplacian (CSD) safely.

    sphere_mode:
      - 'auto'  → try to fit a sphere from available dig; fallback to fixed if needed
      - 'fixed' → use a fixed sphere radius (~94.2 mm)
    """
    import mne
    _ensure_loaded(raw)
    _ensure_montage(raw)

    print(f"CSD: start  (sphere_mode={sphere_mode})")

    if sphere_mode == "fixed":
        # safe fixed sphere radius 94.2 mm
        mne.preprocessing.compute_current_source_density(
            raw, sphere=(0.0, 0.0, 0.0, 0.0942), copy=False
        )
        print("CSD: using fixed sphere (r=0.0942 m)")
    else:
        try:
            mne.preprocessing.compute_current_source_density(raw, sphere="auto", copy=False)
            print("CSD: auto sphere succeeded")
        except Exception as e:
            print(f"CSD: auto sphere failed ({e}); falling back to fixed")
            mne.preprocessing.compute_current_source_density(
                raw, sphere=(0.0, 0.0, 0.0, 0.0942), copy=False
            )
    print("CSD: done")
    return raw
# -----------------------------------------


def load_dataset_ccd(mini: bool, cache_dir: Path) -> BaseConcatDataset:
    """Load EEGChallengeDataset with CCD task (R5 release)."""
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
    csd_sphere: str = "auto",
) -> BaseConcatDataset:
    """Add target annotations & optional CSD; keep only recordings with the anchor."""
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
    preprocess(dataset_ccd, tx, n_jobs=1)

    # Optional CSD (surface Laplacian)
    if use_csd:
        print(f"CSD: enabled, sphere={csd_sphere}")
        preprocess(
            dataset_ccd,
            [Preprocessor(partial(_apply_csd, sphere_mode=csd_sphere), apply_on_array=False)],
            n_jobs=1,
        )

    dataset = keep_only_recordings_with(ANCHOR, dataset_ccd)
    return dataset


def make_windows(dataset: BaseConcatDataset) -> BaseConcatDataset:
    """Stimulus-locked windows, +0.5 s shift, 2 s stride, 2 s model input."""
    windows = create_windows_from_events(
        dataset,
        mapping={ANCHOR: 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
        trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
    )
    # inject extras including 'target'
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

    # filter list used in the startkit examples to keep splits sane
    sub_rm = [
        "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
        "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV",
        "NDARBA381JGH"
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
