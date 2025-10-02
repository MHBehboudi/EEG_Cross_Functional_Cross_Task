# src/eegcfct/train/runner.py
from __future__ import annotations
import argparse
import copy
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..data.ccd_windows import (
    load_dataset_ccd, preprocess_offline, make_windows, subject_splits,
    SFREQ, N_CHANS, WIN_SEC
)
from ..models.simple_eeg import SimpleEEGRegressor
from .loops import build_loaders, train_one_epoch, eval_loop


# ----- small utils ------------------------------------------------------------
def set_seed(seed: int):
    import random
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


# ----- channel clustering -----------------------------------------------------
def _gather_channel_matrix(ds, max_windows: int = 300, seed: int = 2025) -> np.ndarray:
    """
    Concatenate a subset of windows along time so each channel becomes a long vector.
    Returns X_cat: (C, T_total).
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ds))
    rng.shuffle(idx)
    idx = idx[:max_windows]

    X_list = []
    for i in idx:
        x = ds[i][0]  # (C, T)
        if torch.is_tensor(x):
            x = x.numpy()
        X_list.append(x)  # list of (C, T_i)

    # concat across time
    X_cat = np.concatenate(X_list, axis=1)  # (C, sum(T_i))
    # robust z-score per channel
    mu = X_cat.mean(axis=1, keepdims=True)
    sd = X_cat.std(axis=1, keepdims=True) + 1e-8
    X_cat = (X_cat - mu) / sd
    return X_cat


def compute_channel_projection(
    train_windows_ds,
    n_clusters: int,
    max_windows: int = 300,
    seed: int = 2025,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster channels using k-means on concatenated, z-scored time series per channel.
    Returns:
      W: (K, C) projection (uniform averaging per cluster)
      labels: (C,) cluster labels for channels 0..C-1
    """
    assert n_clusters >= 2, "n_clusters must be >= 2"
    X_cat = _gather_channel_matrix(train_windows_ds, max_windows=max_windows, seed=seed)
    C = X_cat.shape[0]

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(X_cat)  # (C,)

    counts = np.bincount(labels, minlength=n_clusters).astype(np.float32)
    counts[counts == 0] = 1.0  # safety

    W = np.zeros((n_clusters, C), dtype=np.float32)
    for c in range(C):
        k = int(labels[c])
        W[k, c] = 1.0 / counts[k]  # uniform average within cluster

    return W, labels


# ----- submission writer ------------------------------------------------------
def write_submission_py(out_dir: Path, n_chans_in: int, use_proj: bool):
    """
    Write a self-contained submission.py with the same SimpleEEGRegressor
    and logic to load /app/output/channel_projection.npz (+ weights).
    """
    code = f'''\
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Be nice to Codabench CPU workers
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass


class ChannelProjector(nn.Module):
    def __init__(self, W: torch.Tensor | None):
        super().__init__()
        if W is None:
            self.register_buffer("W", None)
        else:
            self.register_buffer("W", W.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.W is None:
            return x
        return torch.einsum("bct,kc->bkt", x, self.W)


class SimpleEEGRegressor(nn.Module):
    def __init__(
        self,
        n_chans: int,
        n_outputs: int = 1,
        proj_W: torch.Tensor | None = None,
        n_filters: int = 64,
        temporal_kernel: int = 25,
        depth: int = 3,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.project = ChannelProjector(proj_W)
        c_in = n_chans if proj_W is None else proj_W.shape[0]
        self.conv1 = nn.Conv1d(c_in, n_filters, kernel_size=temporal_kernel,
                               padding=temporal_kernel // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters)
        blocks = []
        for _ in range(max(depth - 1, 0)):
            blocks += [
                nn.Conv1d(n_filters, n_filters, kernel_size=15, padding=7, bias=False),
                nn.BatchNorm1d(n_filters),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.backbone = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_filters, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.float()
        x = self.project(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.backbone(x)
        return self.head(x)


class Submission:
    def __init__(self, SFREQ={SFREQ}, DEVICE=torch.device("cpu")):
        self.sfreq = SFREQ
        self.device = DEVICE

    def _load_W(self):
        import os
        w_path = "/app/output/channel_projection.npz"
        if os.path.exists(w_path):
            Z = np.load(w_path)
            W = Z.get("W", None)
            if W is not None and W.ndim == 2:
                return torch.from_numpy(W)
        return None

    def _make(self):
        W = self._load_W() if {str(use_proj)} else None
        n_chans_in = {n_chans_in}
        model = SimpleEEGRegressor(
            n_chans=n_chans_in, n_outputs=1, proj_W=W,
            n_filters=64, temporal_kernel=25, depth=3, dropout=0.10
        ).to(self.device)
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
'''
    (out_dir / "submission.py").write_text(code)


def build_zip(out_dir: Path, zip_name="submission-to-upload.zip"):
    import zipfile
    to_zip = [
        out_dir / "submission.py",
        out_dir / "weights_challenge_1.pt",
        out_dir / "weights_challenge_2.pt",
        out_dir / "channel_projection.npz",  # always present; identity if no clustering
    ]
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in to_zip:
            zf.write(p, arcname=p.name)
    return zip_path


# ----- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train + ZIP (simple model, optional channel clustering)")
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="output")
    ap.add_argument("--save_zip", action="store_true")

    # clustering
    ap.add_argument("--n_chan_clusters", type=int, default=0,
                    help="If >0, cluster channels into K groups and average (K clusters).")
    ap.add_argument("--cluster_max_windows", type=int, default=300,
                    help="How many training windows to sample when building clusters.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    DATA_DIR = Path(args.data_dir); DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR = Path(args.out_dir);   OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) load + preprocess + windows
    ds = load_dataset_ccd(mini=args.mini, cache_dir=DATA_DIR)
    ds = preprocess_offline(ds)
    windows = make_windows(ds)
    print("Windows ready. Columns:", windows.get_metadata().columns.tolist())

    # 2) splits + loaders (we'll build W on TRAIN windows only)
    train_set, valid_set, test_set = subject_splits(windows, seed=args.seed)
    print(f"Split sizes → Train={len(train_set)}  Valid={len(valid_set)}  Test={len(test_set)}")

    # 3) optional channel clustering -> W (K,C)
    use_clustering = args.n_chan_clusters and args.n_chan_clusters > 0
    if use_clustering:
        print(f"Clustering channels → K={args.n_chan_clusters} (sampling {args.cluster_max_windows} windows)")
        W_np, labels = compute_channel_projection(
            train_set, n_clusters=args.n_chan_clusters,
            max_windows=args.cluster_max_windows, seed=args.seed
        )
        K = W_np.shape[0]
    else:
        # identity projection (C -> C)
        K = N_CHANS
        W_np = np.eye(N_CHANS, dtype=np.float32)
        labels = np.arange(N_CHANS)

    # save projection to disk (always)
    np.savez(OUT_DIR / "channel_projection.npz", W=W_np, labels=labels)
    W_t = torch.from_numpy(W_np)

    # 4) loaders
    tr, va, te = build_loaders(train_set, valid_set, test_set, args.batch_size, args.num_workers)

    # 5) model
    model = SimpleEEGRegressor(
        n_chans=N_CHANS,
        n_outputs=1,
        proj_W=W_t if use_clustering or (W_np.shape[0] != N_CHANS) else None,
        n_filters=64,
        temporal_kernel=25,
        depth=3,
        dropout=0.10,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = CosineAnnealingLR(optim, T_max=max(args.epochs - 1, 1))
    loss_fn = MSELoss()

    # 6) train with early stopping
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

    # 7) final test
    tl, trm = eval_loop(te, model, loss_fn, device)
    print(f"TEST: loss={tl:.6f} rmse={trm:.6f}")

    # 8) save weights (+ optional ZIP)
    p1 = OUT_DIR / "weights_challenge_1.pt"
    p2 = OUT_DIR / "weights_challenge_2.pt"
    torch.save(model.state_dict(), p1)
    torch.save(model.state_dict(), p2)
    print(f"Saved weights: {p1} ({human_size(p1)})")
    print(f"Saved weights: {p2} ({human_size(p2)})")

    if args.save_zip:
        write_submission_py(OUT_DIR, n_chans_in=N_CHANS, use_proj=True)
        zp = build_zip(OUT_DIR, "submission-to-upload.zip")
        print(f"Built ZIP:     {zp} ({human_size(zp)})")

    print("Done.")
