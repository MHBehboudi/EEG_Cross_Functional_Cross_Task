#!/usr/bin/env python3
import os
import sys
import io
import gc
import math
import time
import copy
import zipfile
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import MSELoss
from tqdm import tqdm

# Braindecode / EEGDash
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
from braindecode.models import EEGNeX
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

# --------------------
# Defaults (match startkit)
# --------------------
SFREQ = 100          # downsampled to 100 Hz in challenge data
N_CHANS = 129        # BioSemi 129 montage
WIN_SEC = 2.0        # 2 seconds → 200 samples
EPOCH_LEN_S = 2.0    # sliding window size (samples returned to model)
SHIFT_AFTER_STIM = 0.5
WINDOW_LEN = 2.0     # long enough to include response
ANCHOR = "stimulus_anchor"

# --------------------
# Utilities
# --------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def human_size(p: Path):
    try:
        return f"{p.stat().st_size/1e6:.2f} MB"
    except Exception:
        return "n/a"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loaders(train_set, valid_set, test_set, batch_size, num_workers):
    # Braindecode Windows datasets usually return (X, y, i). Our loops handle both (X, y) and (X, y, ...).
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    return train_loader, valid_loader, test_loader


def train_one_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_fn,
    optimizer,
    scheduler,
    epoch: int,
    device
):
    model.train()
    total_loss = 0.0
    sum_sq_err = 0.0
    n_samples = 0

    for bidx, batch in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        # batch can be (X, y) or (X, y, ...)
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        preds_flat = preds.detach().view(-1)
        y_flat = y.detach().view(-1)
        sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
        n_samples += y_flat.numel()

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / max(len(dataloader), 1)
    rmse = math.sqrt(sum_sq_err / max(n_samples, 1))
    return avg_loss, rmse


@torch.no_grad()
def eval_loop(dataloader: DataLoader, model: torch.nn.Module, loss_fn, device):
    model.eval()
    total_loss, sum_sq_err, n_samples = 0.0, 0.0, 0
    for batch in dataloader:
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()
        preds = model(X)
        loss = loss_fn(preds, y).item()
        total_loss += loss
        preds_flat = preds.view(-1)
        y_flat = y.view(-1)
        sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
        n_samples += y_flat.numel()
    avg_loss = total_loss / max(len(dataloader), 1)
    rmse = math.sqrt(sum_sq_err / max(n_samples, 1))
    return avg_loss, rmse


def make_windows_ccd(dataset_ccd: BaseConcatDataset):
    """Mirror the startkit’s CCD epoching & extras injection."""
    # 1) annotate to compute 'target' as rt_from_stimulus
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

    # 2) keep only recordings that contain the anchor we want
    dataset = keep_only_recordings_with(ANCHOR, dataset_ccd)

    # 3) create windows (stimulus-locked, +0.5s shift, 2s stride)
    single_windows = create_windows_from_events(
        dataset,
        mapping={ANCHOR: 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
        trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
    )

    # 4) inject metadata columns, including 'target'
    single_windows = add_extras_columns(
        single_windows,
        dataset,
        desc=ANCHOR,
        keys=("target", "rt_from_stimulus", "rt_from_trialstart",
              "stimulus_onset", "response_onset", "correct", "response_type"),
    )

    return single_windows


def subject_split_concat(windows_ds: BaseConcatDataset, seed=2025, valid_frac=0.1, test_frac=0.1):
    from sklearn.model_selection import train_test_split
    from sklearn.utils import check_random_state

    meta = windows_ds.get_metadata()
    subjects = meta["subject"].unique()

    # filter list used in the startkit examples to keep splits sane
    sub_rm = ["NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
              "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"]
    subjects = [s for s in subjects if s not in sub_rm]

    train_subj, valid_test_subject = train_test_split(
        subjects, test_size=(valid_frac + test_frac), random_state=check_random_state(seed), shuffle=True
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


def write_submission_py(out_dir: Path):
    code = """\
import torch
from braindecode.models import EEGNeX

# Cap threads for Codabench CPU workers
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def _make(self):
        model = EEGNeX(
            n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2 * self.sfreq)
        ).to(self.device)
        model.eval()
        return model

    def get_model_challenge_1(self):
        model_challenge1 = self._make()
        state = torch.load("/app/output/weights_challenge_1.pt", map_location=self.device)
        model_challenge1.load_state_dict(state, strict=True)
        return model_challenge1

    def get_model_challenge_2(self):
        model_challenge2 = self._make()
        state = torch.load("/app/output/weights_challenge_2.pt", map_location=self.device)
        model_challenge2.load_state_dict(state, strict=True)
        return model_challenge2
"""
    (out_dir / "submission.py").write_text(code)


def build_zip(out_dir: Path, zip_name="submission-to-upload.zip"):
    # Codabench requires a single-level zip with exactly these filenames
    to_zip = [
        out_dir / "submission.py",
        out_dir / "weights_challenge_1.pt",
        out_dir / "weights_challenge_2.pt",
    ]
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in to_zip:
            zf.write(p, arcname=p.name)
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Train like startkit & build Codabench ZIP")
    parser.add_argument("--mini", action="store_true", help="use mini release (recommended to match startkit speed)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="output")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    DATA_DIR = Path(args.data_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------
    # Load challenge data (mini) exactly like startkit
    # --------------------
    dataset_ccd = EEGChallengeDataset(
        task="contrastChangeDetection",
        release="R5",
        cache_dir=DATA_DIR,
        mini=bool(args.mini),
    )
    print(f"Loaded CCD dataset (mini={bool(args.mini)}): {len(dataset_ccd)} recordings")

    # --------------------
    # Create windows + extras (this sets up 'target' in metadata)
    # --------------------
    single_windows = make_windows_ccd(dataset_ccd)
    print("Windows created. Metadata columns:", single_windows.get_metadata().columns.tolist())

    # --------------------
    # Subject-wise split and DataLoaders
    # --------------------
    train_set, valid_set, test_set = subject_split_concat(single_windows, seed=args.seed)
    print("Number of examples per split")
    print(f"  Train: {len(train_set)}")
    print(f"  Valid: {len(valid_set)}")
    print(f"  Test : {len(test_set)}")

    train_loader, valid_loader, test_loader = build_loaders(
        train_set, valid_set, test_set, args.batch_size, args.num_workers
    )

    # --------------------
    # Model (same shape as startkit’s weights)
    # --------------------
    model = EEGNeX(
        n_chans=N_CHANS,
        n_outputs=1,
        sfreq=SFREQ,
        n_times=int(WIN_SEC * SFREQ),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs - 1, 1))
    loss_fn = MSELoss()

    # Early stopping (modest patience so it mirrors startkit speed)
    patience = 50
    min_delta = 1e-4
    best_rmse = float("inf")
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0

    # --------------------
    # Train
    # --------------------
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_rmse = train_one_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, device)
        va_loss, va_rmse = eval_loop(valid_loader, model, loss_fn, device)
        print(f"[{epoch:03d}/{args.epochs}] train_loss={tr_loss:.6f} train_rmse={tr_rmse:.6f}  "
              f"val_loss={va_loss:.6f} val_rmse={va_rmse:.6f}")

        if va_rmse < best_rmse - min_delta:
            best_rmse = va_rmse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val RMSE={best_rmse:.6f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --------------------
    # Final test
    # --------------------
    te_loss, te_rmse = eval_loop(test_loader, model, loss_fn, device)
    print(f"TEST: loss={te_loss:.6f} rmse={te_rmse:.6f}")

    # --------------------
    # Save weights (challenge 1 + 2), write submission.py, zip
    # --------------------
    # Save your trained weights (challenge 1)
    path_c1 = OUT_DIR / "weights_challenge_1.pt"
    torch.save(model.state_dict(), path_c1)

    # For a startkit-like first run, reuse the same tiny model for challenge 2
    path_c2 = OUT_DIR / "weights_challenge_2.pt"
    torch.save(model.state_dict(), path_c2)

    write_submission_py(OUT_DIR)
    zip_path = build_zip(OUT_DIR, zip_name="submission-to-upload.zip")

    print(f"Saved weights: {path_c1} ({human_size(path_c1)})")
    print(f"Saved weights: {path_c2} ({human_size(path_c2)})")
    print(f"Built ZIP:     {zip_path} ({human_size(zip_path)})")
    print("Done.")


if __name__ == "__main__":
    main()
