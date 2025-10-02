# src/eegcfct/train/runner.py
from __future__ import annotations
import argparse
import copy
import math
from pathlib import Path

import torch
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from braindecode.models import EEGNeX

from ..data.ccd_windows import (
    load_dataset_ccd, preprocess_offline, make_windows, subject_splits,
    SFREQ, N_CHANS, WIN_SEC
)
from .loops import build_loaders, train_one_epoch, eval_loop
from ..preproc.channel_clustering import (
    compute_channel_clustering, ClusteredWindowsDataset,
    save_channel_clustering, load_channel_clustering
)

# ---------- small utils ----------
def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def human_size(p: Path):
    try:
        return f"{p.stat().st_size/1e6:.2f} MB"
    except Exception:
        return "n/a"

def write_submission_py(out_dir: Path, n_chans_for_model: int):
    code = f"""\
import torch
from braindecode.models import EEGNeX

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

class Submission:
    def __init__(self, SFREQ={SFREQ}, DEVICE=torch.device("cpu")):
        self.sfreq = SFREQ
        self.device = DEVICE

    def _make(self):
        model = EEGNeX(n_chans={n_chans_for_model}, n_outputs=1, sfreq=self.sfreq, n_times=int({WIN_SEC} * self.sfreq)).to(self.device)
        model.eval()
        return model

    def get_model_challenge_1(self):
        m = self._make()
        state = torch.load("/app/output/weights_challenge_1.pt", map_location=self.device)
        m.load_state_dict(state, strict=True)
        return m

    def get_model_challenge_2(self):
        m = self._make()
        state = torch.load("/app/output/weights_challenge_2.pt", map_location=self.device)
        m.load_state_dict(state, strict=True)
        return m
"""
    (out_dir / "submission.py").write_text(code)

def build_zip(out_dir: Path, zip_name="submission-to-upload.zip"):
    import zipfile
    to_zip = [out_dir / "submission.py", out_dir / "weights_challenge_1.pt", out_dir / "weights_challenge_2.pt"]
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in to_zip:
            zf.write(p, arcname=p.name)
    return zip_path

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Train like startkit & build Codabench ZIP")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--save_zip", action="store_true")

    # --- channel clustering knobs ---
    parser.add_argument("--n_chan_clusters", type=int, default=0,
                        help="If >0, cluster channels by signal similarity and average within clusters.")
    parser.add_argument("--cluster_max_windows", type=int, default=1500,
                        help="How many training windows to use when estimating channel similarity.")
    parser.add_argument("--cluster_seed", type=int, default=2025)
    parser.add_argument("--cluster_path", type=str, default="",
                        help="If set, load/save clustering here (npz). If file exists -> load; else compute+save.")

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    DATA_DIR = Path(args.data_dir); DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR = Path(args.out_dir);   OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load + preprocess + windows
    ds = load_dataset_ccd(mini=args.mini, cache_dir=DATA_DIR)
    ds = preprocess_offline(ds)
    windows = make_windows(ds)
    print("Windows ready. Columns:", windows.get_metadata().columns.tolist())

    # splits
    train_set, valid_set, test_set = subject_splits(windows, seed=args.seed)
    print(f"Split sizes → Train={len(train_set)}  Valid={len(valid_set)}  Test={len(test_set)}")

    n_chans_for_model = N_CHANS
    clustering_used = False

    # Optional: channel clustering on TRAIN only (no leakage)
    if args.n_chan_clusters and args.n_chan_clusters > 0:
        clustering_used = True
        path_npz = Path(args.cluster_path) if args.cluster_path else (OUT_DIR / "channel_clusters.npz")
        if path_npz.exists():
            print(f"Loading existing channel clustering from {path_npz}")
            cc = load_channel_clustering(path_npz)
        else:
            print(f"Computing channel clustering: K={args.n_chan_clusters}, "
                  f"max_windows={args.cluster_max_windows}")
            cc = compute_channel_clustering(
                train_set=train_set,
                n_chans=N_CHANS,
                n_clusters=int(args.n_chan_clusters),
                max_windows=int(args.cluster_max_windows),
                seed=int(args.cluster_seed),
            )
            save_channel_clustering(path_npz, cc)
            print(f"Saved channel clustering to {path_npz}")

        # Wrap datasets to apply W @ X on the fly
        train_set = ClusteredWindowsDataset(train_set, cc.W)
        valid_set = ClusteredWindowsDataset(valid_set, cc.W)
        test_set  = ClusteredWindowsDataset(test_set,  cc.W)
        n_chans_for_model = int(args.n_chan_clusters)
        print(f"Channel clustering active → {N_CHANS} → {n_chans_for_model} channels")

    # loaders
    tr, va, te = build_loaders(train_set, valid_set, test_set, args.batch_size, args.num_workers)

    # model
    model = EEGNeX(n_chans=n_chans_for_model, n_outputs=1, sfreq=SFREQ, n_times=int(WIN_SEC * SFREQ)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = CosineAnnealingLR(optim, T_max=max(args.epochs - 1, 1))
    loss_fn = MSELoss()

    # train w/ early stopping
    patience, min_delta = 50, 1e-4
    best_rmse, best_state, best_epoch, no_improve = math.inf, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        tl, trm = train_one_epoch(tr, model, loss_fn, optim, sched, epoch, device)
        vl, vrm = eval_loop(va, model, loss_fn, device)
        print(f"[{epoch:03d}/{args.epochs}] train_loss={tl:.6f} train_rmse={trm:.6f}  val_loss={vl:.6f} val_rmse={vrm:.6f}")
        if vrm < best_rmse - min_delta:
            best_rmse, best_state, best_epoch, no_improve = vrm, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val RMSE={best_rmse:.6f}).")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # final test
    tl, trm = eval_loop(te, model, loss_fn, device)
    print(f"TEST: loss={tl:.6f} rmse={trm:.6f}")

    # save weights (+ optional zip)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    p1 = OUT_DIR / "weights_challenge_1.pt"
    p2 = OUT_DIR / "weights_challenge_2.pt"
    torch.save(model.state_dict(), p1)
    torch.save(model.state_dict(), p2)
    print(f"Saved weights: {p1} ({human_size(p1)})")
    print(f"Saved weights: {p2} ({human_size(p2)})")

    if args.save_zip:
        write_submission_py(OUT_DIR, n_chans_for_model=n_chans_for_model)
        zp = build_zip(OUT_DIR, "submission-to-upload.zip")
        print(f"Built ZIP:     {zp} ({human_size(zp)})")

    print("Done.")
