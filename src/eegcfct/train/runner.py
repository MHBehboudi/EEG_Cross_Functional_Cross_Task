import argparse
import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from braindecode.models import EEGNeX

from ..data.ccd_windows import (
    load_dataset_ccd, preprocess_offline, make_windows, subject_splits,
    SFREQ, N_CHANS, WIN_SEC
)
from .loops import build_loaders, train_one_epoch, eval_loop


# ---------------- small utils ----------------
def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def human_size(p: Path):
    try:
        return f"{p.stat().st_size/1e6:.2f} MB"
    except Exception:
        return "n/a"


def write_submission_py(out_dir: Path):
    # Submission reconstructs either identity or cluster projector from weights
    code = r'''import os
from pathlib import Path
import torch
import torch.nn as nn
from braindecode.models import EEGNeX

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

class ProjectedEEGNeX(nn.Module):
    def __init__(self, in_ch: int, k_out: int, sfreq: int, n_times: int):
        super().__init__()
        self.projector = nn.Conv1d(in_ch, k_out, kernel_size=1, bias=False)
        self.backbone = EEGNeX(n_chans=k_out, n_outputs=1, sfreq=sfreq, n_times=n_times)

    def forward(self, x):
        x = self.projector(x)
        return self.backbone(x)

def _resolve(name: str) -> Path:
    for p in (Path("/app/output")/name, Path("/app/ingested_program")/name, Path(".")/name):
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing {name}")

def _make_from_weights(wp: Path, device: torch.device, sfreq: int):
    state = torch.load(wp, map_location=device)
    if "projector.weight" in state:
        pw = state["projector.weight"]  # [K, C, 1]
        k_out, c_in = int(pw.shape[0]), int(pw.shape[1])
        model = ProjectedEEGNeX(in_ch=c_in, k_out=k_out, sfreq=int(sfreq), n_times=int(2*sfreq)).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model
    # fallback: identity projector + EEGNeX(C=129)
    c_in = 129
    model = ProjectedEEGNeX(in_ch=c_in, k_out=c_in, sfreq=int(sfreq), n_times=int(2*sfreq)).to(device)
    with torch.no_grad():
        eye = torch.eye(c_in, dtype=torch.float32, device=device).unsqueeze(-1)
        model.projector.weight.copy_(eye)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = int(SFREQ)
        self.device = DEVICE if DEVICE is not None else torch.device("cpu")

    def get_model_challenge_1(self):
        return _make_from_weights(_resolve("weights_challenge_1.pt"), self.device, self.sfreq)

    def get_model_challenge_2(self):
        return _make_from_weights(_resolve("weights_challenge_2.pt"), self.device, self.sfreq)
'''
    (out_dir / "submission.py").write_text(code)


def build_zip(out_dir: Path, zip_name="submission-to-upload.zip"):
    import zipfile
    to_zip = [out_dir / "submission.py", out_dir / "weights_challenge_1.pt", out_dir / "weights_challenge_2.pt"]
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in to_zip:
            zf.write(p, arcname=p.name)
    return zip_path


# ---------------- clustering + PCA projector ----------------
def _sample_windows_for_stats(train_set, max_windows=512, num_workers=4):
    """Collect up to max_windows windows (X) to compute clustering stats."""
    loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=False)
    Xs = []
    total = 0
    for batch in loader:
        X = batch[0]  # [B, C, T]
        Xs.append(X.cpu().float().numpy())
        total += X.shape[0]
        if total >= max_windows:
            break
    if not Xs:
        raise RuntimeError("No windows sampled for clustering stats.")
    X = np.concatenate(Xs, axis=0)
    if X.shape[0] > max_windows:
        X = X[:max_windows]
    return X  # [N, C, T]


def _compute_channel_features(X):
    """
    Build a feature vector per channel using correlations with all channels.
    X: [N, C, T] windows (z-scored per-window & channel later)
    Returns: feats [C, C]
    """
    N, C, T = X.shape
    # per-window z-score each channel across time to normalize scale
    X_ = X - X.mean(axis=2, keepdims=True)
    std = X_.std(axis=2, keepdims=True) + 1e-8
    X_ = X_ / std
    # flatten time across windows: channels x (N*T)
    Xf = X_.transpose(1, 0, 2).reshape(C, N * T)
    Xf = Xf - Xf.mean(axis=1, keepdims=True)
    Xf_std = Xf.std(axis=1, keepdims=True) + 1e-8
    Xf = Xf / Xf_std
    # channel-channel correlation as features (each row = correlation profile to all chans)
    corr = np.corrcoef(Xf)  # [C, C]
    # guard numerical issues
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr.astype(np.float32)


def _kmeans_clusters(feats, num_clusters, seed):
    km = KMeans(n_clusters=num_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(feats)  # length C
    return labels


def _build_pca_projector(X, labels, num_clusters, pca_k):
    """
    For each cluster, compute top-K PCA in channel space, return W [K_out, C].
    X: [N, C, T] (already sampled); labels: per-channel ints.
    """
    N, C, T = X.shape
    # per-window z-score across time
    X_ = X - X.mean(axis=2, keepdims=True)
    std = X_.std(axis=2, keepdims=True) + 1e-8
    X_ = X_ / std
    Xf = X_.transpose(1, 0, 2).reshape(C, N * T)  # channels x samples

    rows = []
    for cl in range(num_clusters):
        idx = np.where(labels == cl)[0]
        if len(idx) == 0:
            continue
        if len(idx) == 1:
            # singleton cluster → just pass that channel
            e = np.zeros((C,), dtype=np.float32)
            e[idx[0]] = 1.0
            rows.append(e)
            continue

        D = Xf[idx, :]  # [|S|, M]
        # center across samples
        D = D - D.mean(axis=1, keepdims=True)
        # SVD on channels (fast & stable)
        # D = U S V^T, U[:, :k] are top PCs in channel space
        U, S, Vt = np.linalg.svd(D, full_matrices=False)
        k = min(pca_k, U.shape[1])
        comps = U[:, :k]  # [|S|, k]
        # turn each component into a row over all C channels
        for j in range(k):
            w = np.zeros((C,), dtype=np.float32)
            w[idx] = comps[:, j]
            rows.append(w)

    if not rows:
        # fallback to identity
        W = np.eye(C, dtype=np.float32)
    else:
        W = np.stack(rows, axis=0)  # [K_out, C]
    # optional: row-normalize to unit norm to avoid scale explosion
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    W = W / norms
    return W  # [K_out, C]


class ProjectedEEGNeX(nn.Module):
    """1x1 Conv projector (C->K) + EEGNeX(K chans)."""
    def __init__(self, in_ch: int, k_out: int, sfreq: int, n_times: int):
        super().__init__()
        self.projector = nn.Conv1d(in_ch, k_out, kernel_size=1, bias=False)
        self.backbone = EEGNeX(n_chans=k_out, n_outputs=1, sfreq=sfreq, n_times=n_times)

    def forward(self, x):
        x = self.projector(x)
        return self.backbone(x)


# ---------------- main ----------------
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

    # clustering/PCA knobs
    parser.add_argument("--use_clusters", action="store_true", help="Enable KMeans+PCA projector")
    parser.add_argument("--clusters", type=int, default=20, help="Number of KMeans clusters")
    parser.add_argument("--pca_k", type=int, default=3, help="Top-K PCs per cluster")
    parser.add_argument("--sample_windows", type=int, default=512, help="#windows for stats")

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
    tr_loader, va_loader, te_loader = build_loaders(train_set, valid_set, test_set, args.batch_size, args.num_workers)

    # projector (optional)
    in_ch = N_CHANS
    if args.use_clusters:
        print(f"[Projector] KMeans clusters={args.clusters}, PCA per cluster={args.pca_k}, sample_windows={args.sample_windows}")
        Xs = _sample_windows_for_stats(train_set, max_windows=args.sample_windows, num_workers=args.num_workers)
        feats = _compute_channel_features(Xs)
        labels = _kmeans_clusters(feats, num_clusters=args.clusters, seed=args.seed)
        W = _build_pca_projector(Xs, labels, num_clusters=args.clusters, pca_k=args.pca_k)  # [K_out, C]
        k_out = int(W.shape[0])
        print(f"[Projector] Built projector W with shape {W.shape} → EEGNeX n_chans={k_out}")
        model = ProjectedEEGNeX(in_ch=in_ch, k_out=k_out, sfreq=SFREQ, n_times=int(WIN_SEC * SFREQ)).to(device)
        with torch.no_grad():
            w_t = torch.from_numpy(W).to(device=device, dtype=torch.float32).unsqueeze(-1)  # [K, C, 1]
            model.projector.weight.copy_(w_t)
    else:
        # Baseline: identity projector (C->C)
        k_out = in_ch
        model = ProjectedEEGNeX(in_ch=in_ch, k_out=k_out, sfreq=SFREQ, n_times=int(WIN_SEC * SFREQ)).to(device)
        with torch.no_grad():
            eye = torch.eye(in_ch, dtype=torch.float32, device=device).unsqueeze(-1)
            model.projector.weight.copy_(eye)

    # optim & loss
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = CosineAnnealingLR(optim, T_max=max(args.epochs - 1, 1))
    loss_fn = MSELoss()

    # train (early stopping)
    patience, min_delta = 50, 1e-4
    best_rmse, best_state, no_improve = math.inf, None, 0
    for epoch in range(1, args.epochs + 1):
        tl, trm = train_one_epoch(tr_loader, model, loss_fn, optim, sched, epoch, device)
        vl, vrm = eval_loop(va_loader, model, loss_fn, device)
        print(f"[{epoch:03d}/{args.epochs}] train_loss={tl:.6f} train_rmse={trm:.6f}  val_loss={vl:.6f} val_rmse={vrm:.6f}")
        if vrm < best_rmse - min_delta:
            best_rmse, best_state, no_improve = vrm, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val RMSE={best_rmse:.6f}).")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # test
    tl, trm = eval_loop(te_loader, model, loss_fn, device)
    print(f"TEST: loss={tl:.6f} rmse={trm:.6f}")

    # save weights (+ zip)
    p1 = OUT_DIR / "weights_challenge_1.pt"
    p2 = OUT_DIR / "weights_challenge_2.pt"
    torch.save(model.state_dict(), p1)
    torch.save(model.state_dict(), p2)
    print(f"Saved weights: {p1} ({human_size(p1)})")
    print(f"Saved weights: {p2} ({human_size(p2)})")

    if args.save_zip:
        write_submission_py(OUT_DIR)
        zp = build_zip(OUT_DIR, "submission-to-upload.zip")
        print(f"Built ZIP:     {zp} ({human_size(zp)})")

    print("Done.")
