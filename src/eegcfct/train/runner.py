import argparse
import copy
import math
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from braindecode.models import EEGNeX

from ..data.ccd_windows import (
    load_dataset_ccd, preprocess_offline, make_windows, subject_splits,
    SFREQ, N_CHANS, WIN_SEC
)
from .loops import build_loaders, train_one_epoch, eval_loop


# -----------------------------
# Small utils
# -----------------------------
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


# -----------------------------
# Projected model (optional)
# -----------------------------
class ProjectedEEGNeX(torch.nn.Module):
    """
    EEGNeX preceded by a linear channel projector (1x1 Conv1d).
    projector_weight can be:
      - (C_out, C_in, 1) conv weight, or
      - (C_in, C_out) raw matrix
    """
    def __init__(self, sfreq: int, n_times: int, projector_weight, freeze_projector: bool = True):
        super().__init__()
        if projector_weight is None:
            raise RuntimeError("ProjectedEEGNeX requires a projector weight.")
        if isinstance(projector_weight, torch.Tensor):
            W = projector_weight.detach().clone().float()
        else:
            W = torch.tensor(projector_weight, dtype=torch.float32)

        if W.ndim == 3:  # (C_out, C_in, 1) conv format -> convert to (C_in, C_out)
            c_out, c_in, _ = W.shape
            W_ci_co = W.permute(1, 0, 2).squeeze(-1).contiguous()
        elif W.ndim == 2:  # (C_in, C_out)
            c_in, c_out = W.shape
            W_ci_co = W.contiguous()
        else:
            raise RuntimeError(f"Unexpected projector weight shape: {tuple(W.shape)}")

        self.projector = torch.nn.Conv1d(
            in_channels=W_ci_co.shape[0],
            out_channels=W_ci_co.shape[1],
            kernel_size=1,
            bias=False,
        )
        with torch.no_grad():
            self.projector.weight.copy_(W_ci_co.t().unsqueeze(-1).contiguous())

        if freeze_projector:
            for p in self.projector.parameters():
                p.requires_grad = False

        self.backbone = EEGNeX(
            n_chans=self.projector.out_channels,
            n_outputs=1,
            sfreq=sfreq,
            n_times=n_times,
        )

    def forward(self, x):
        # Ensure device/dtype alignment
        self.projector = self.projector.to(x.device, dtype=x.dtype)
        x = self.projector(x)
        return self.backbone(x)


# -----------------------------
# Submission writer
# -----------------------------
def write_submission_py(out_dir: Path, have_projector: bool = False, k: int | None = None, pcs: int | None = None):
    """
    Emit submission.py that auto-detects projector presence from the checkpoint.
    """
    code = f"""\
import torch
from braindecode.models import EEGNeX

# Cap threads for Codabench CPU workers
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# Metadata (informational only)
# have_projector={have_projector}, proj_k={k}, proj_pcs={pcs}

class ProjectedEEGNeX(torch.nn.Module):
    def __init__(self, sfreq: int, n_times: int, projector_weight):
        super().__init__()
        if isinstance(projector_weight, torch.Tensor):
            W = projector_weight.detach().clone().float()
        else:
            W = torch.tensor(projector_weight, dtype=torch.float32)
        if W.ndim == 3:  # (C_out, C_in, 1) -> (C_in, C_out)
            c_out, c_in, _ = W.shape
            W_ci_co = W.permute(1, 0, 2).squeeze(-1).contiguous()
        elif W.ndim == 2:
            W_ci_co = W.contiguous()
        else:
            raise RuntimeError(f"Unexpected projector weight shape: {{tuple(W.shape)}}")

        self.projector = torch.nn.Conv1d(
            in_channels=W_ci_co.shape[0],
            out_channels=W_ci_co.shape[1],
            kernel_size=1,
            bias=False,
        )
        with torch.no_grad():
            self.projector.weight.copy_(W_ci_co.t().unsqueeze(-1).contiguous())
        self.backbone = EEGNeX(
            n_chans=self.projector.out_channels,
            n_outputs=1,
            sfreq=sfreq,
            n_times=n_times,
        )

    def forward(self, x):
        self.projector = self.projector.to(x.device, dtype=x.dtype)
        x = self.projector(x)
        return self.backbone(x)


class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def _load_and_build(self, ckpt_path: str):
        state = torch.load(ckpt_path, map_location=self.device)
        if "projector.weight" in state:
            W = state.pop("projector.weight")
            model = ProjectedEEGNeX(
                sfreq=self.sfreq,
                n_times=int(2 * self.sfreq),
                projector_weight=W,
            ).to(self.device)
        else:
            model = EEGNeX(
                n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2 * self.sfreq)
            ).to(self.device)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    def get_model_challenge_1(self):
        return self._load_and_build("/app/output/weights_challenge_1.pt")

    def get_model_challenge_2(self):
        return self._load_and_build("/app/output/weights_challenge_2.pt")
"""
    (out_dir / "submission.py").write_text(code)


def build_zip(out_dir: Path, zip_name="submission-to-upload.zip"):
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


# -----------------------------
# Channel projection via KMeans + PCA
# -----------------------------
def _accumulate_channel_cov(train_set, max_windows: int = 200) -> np.ndarray:
    """
    Estimate channel covariance (C x C) across up to max_windows windows.
    Each window is mean-centered across time before covariance.
    """
    C = N_CHANS
    cov = np.zeros((C, C), dtype=np.float64)
    n_used = 0
    for i, sample in enumerate(train_set):
        if i >= max_windows:
            break
        X = sample[0]  # tensor (C, T)
        X = X.numpy()
        X = X - X.mean(axis=1, keepdims=True)
        cov += X @ X.T / max(X.shape[1], 1)
        n_used += 1
    if n_used > 0:
        cov /= n_used
    return cov


def _kmeans_channels(features: np.ndarray, k: int, rnd: int = 0) -> np.ndarray:
    """
    KMeans cluster channels using given features (C x F). Returns labels (C,).
    """
    from sklearn.cluster import KMeans
    C = features.shape[0]
    if k >= C:
        # degenerate case: each channel its own cluster
        return np.arange(C, dtype=int)
    km = KMeans(n_clusters=k, random_state=rnd, n_init=10)
    labels = km.fit_predict(features)
    return labels


def _pca_components_from_cov(cov: np.ndarray, p: int) -> np.ndarray:
    """
    Top-p eigenvectors of covariance matrix 'cov' (M x M). Returns (M x p).
    If p > M, returns M components.
    """
    M = cov.shape[0]
    p_eff = min(p, M)
    # eigen-decomposition (symmetric)
    w, v = np.linalg.eigh(cov)
    # sort desc by eigenvalue
    order = np.argsort(w)[::-1]
    v_desc = v[:, order]
    return v_desc[:, :p_eff].astype(np.float32, copy=False)


def build_channel_projection_from_kmeans_pca(
    train_set,
    k: int = 20,
    pcs: int = 3,
    n_win_for_pca: int = 200,
) -> np.ndarray:
    """
    Build a channel projector W of shape (C_out, C_in, 1) by:
      1) estimating channel covariance (C x C) from up to n_win_for_pca windows
      2) clustering channels via KMeans on covariance rows (C features)
      3) per cluster, take top 'pcs' eigenvectors of sub-covariance (|cluster| x |cluster|)
      4) place those eigenvectors into rows of W (zero elsewhere)
    C_out = sum over clusters of min(pcs, cluster_size).
    """
    C = N_CHANS
    cov = _accumulate_channel_cov(train_set, max_windows=n_win_for_pca)
    # Feature for each channel: its covariance profile (row of cov)
    features = cov.copy()
    labels = _kmeans_channels(features, k=k, rnd=0)

    # Count C_out
    counts = [(labels == c).sum() for c in range(k)]
    outs = [min(pcs, cnt) for cnt in counts]
    C_out = int(np.sum(outs))
    if C_out == 0:
        # fallback: identity
        W = np.zeros((C, C, 1), dtype=np.float32)
        for i in range(C):
            W[i, i, 0] = 1.0
        return W

    W_full = np.zeros((C_out, C, 1), dtype=np.float32)
    row_ptr = 0
    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        sub_cov = cov[np.ix_(idx, idx)]
        U = _pca_components_from_cov(sub_cov, pcs)  # (len(idx) x r)
        r = U.shape[1]
        # place these 'r' rows into W_full
        # each row is a length-C vector with nonzeros only at 'idx'
        for j in range(r):
            W_full[row_ptr + j, idx, 0] = U[:, j]
        row_ptr += r

    return W_full  # (C_out, C_in, 1)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train (optionally with channel projector) & build Codabench ZIP")
    # data/training
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--save_zip", action="store_true")
    # projector
    parser.add_argument("--use_projector", action="store_true",
                        help="Enable KMeans+PCA channel projection prior to EEGNeX.")
    parser.add_argument("--proj_k", type=int, default=20, help="Number of channel clusters.")
    parser.add_argument("--proj_pcs", type=int, default=3, help="PCs per cluster.")
    parser.add_argument("--proj_windows", type=int, default=200, help="Num windows to estimate PCA.")
    parser.add_argument("--proj_freeze", action="store_true", help="Freeze projector weights (default True).")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    DATA_DIR = Path(args.data_dir); DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR = Path(args.out_dir);   OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load + preprocess + windows
    ds = load_dataset_ccd(mini=args.mini, cache_dir=DATA_DIR)
    ds = preprocess_offline(ds)
    windows = make_windows(ds)
    print("Windows ready. Columns:", windows.get_metadata().columns.tolist())

    # splits + loaders
    train_set, valid_set, test_set = subject_splits(windows, seed=args.seed)
    print(f"Split sizes → Train={len(train_set)}  Valid={len(valid_set)}  Test={len(test_set)}")
    tr_loader, va_loader, te_loader = build_loaders(train_set, valid_set, test_set, args.batch_size, args.num_workers)

    # projector (optional)
    have_projector = False
    projector_weight = None
    if args.use_projector:
        print(f"[Projector] KMeans+PCA: K={args.proj_k}, PCs/cluster={args.proj_pcs}, windows={args.proj_windows}")
        W = build_channel_projection_from_kmeans_pca(
            train_set=train_set, k=args.proj_k, pcs=args.proj_pcs, n_win_for_pca=args.proj_windows
        )  # (C_out, C_in, 1)
        projector_weight = torch.from_numpy(W)
        have_projector = True
        print(f"[Projector] Built projector with C_out={W.shape[0]} from C_in={W.shape[1]}.")

    # model
    if have_projector:
        model = ProjectedEEGNeX(
            sfreq=SFREQ,
            n_times=int(WIN_SEC * SFREQ),
            projector_weight=projector_weight,
            freeze_projector=args.proj_freeze,
        ).to(device)
    else:
        model = EEGNeX(n_chans=N_CHANS, n_outputs=1, sfreq=SFREQ, n_times=int(WIN_SEC * SFREQ)).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = CosineAnnealingLR(optim, T_max=max(args.epochs - 1, 1))
    loss_fn = MSELoss()

    # train w/ early stopping
    patience, min_delta = 50, 1e-4
    best_rmse, best_state, best_epoch, no_improve = math.inf, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        tl, trm = train_one_epoch(tr_loader, model, loss_fn, optim, sched, epoch, device)
        vl, vrm = eval_loop(va_loader, model, loss_fn, device)
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
    tl, trm = eval_loop(te_loader, model, loss_fn, device)
    print(f"TEST: loss={tl:.6f} rmse={trm:.6f}")

    # save weights (+ optional zip)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    p1 = OUT_DIR / "weights_challenge_1.pt"
    p2 = OUT_DIR / "weights_challenge_2.pt"
    torch.save(model.state_dict(), p1)
    torch.save(model.state_dict(), p2)
    print(f"Saved weights: {p1} ({human_size(p1)})")
    print(f"Saved weights: {p2} ({human_size(p2)})")

    write_submission_py(OUT_DIR, have_projector=have_projector, k=args.proj_k if have_projector else None, pcs=args.proj_pcs if have_projector else None)
    if args.save_zip:
        zp = build_zip(OUT_DIR, "submission-to-upload.zip")
        print(f"Built ZIP:     {zp} ({human_size(zp)})")

    print("Done.")
