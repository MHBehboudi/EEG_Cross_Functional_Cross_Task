# src/eegcfct/train/runner.py
import argparse
import copy
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..data.ccd_windows import (
    load_dataset_ccd, preprocess_offline, make_windows, subject_splits,
    SFREQ, N_CHANS, WIN_SEC
)
from ..models.clustered_cnn import build_model_clustered
from .loops import build_loaders, train_one_epoch, eval_loop


# ---------- utils ----------
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


# ---------- clustering (build projector weight) ----------
def compute_cluster_projection(train_set,
                               max_windows: int = 400,
                               n_clusters: int = 50,
                               seed: int = 2025,
                               num_workers: int = 2) -> torch.Tensor:
    """
    Build KxC averaging matrix P (as a Conv1d weight [K, C, 1]) by:
      1) Sampling up to `max_windows` windows from train_set
      2) For each window, compute channel-by-channel correlation (C x C)
      3) Average the correlation matrices
      4) Cluster channels using KMeans on rows of the averaged matrix
      5) Create P: rows are 1/n_k on members of cluster k (channel mean)
    """
    from torch.utils.data import DataLoader
    from sklearn.cluster import KMeans

    # Small loader just for sampling raw windows
    dl = DataLoader(train_set, batch_size=32, shuffle=True,
                    num_workers=num_workers, pin_memory=False)
    C = None
    corr_sum = None
    n_seen = 0

    for X, *_ in dl:
        # X: [B, C, T]
        X = X.float().numpy()
        B, C = X.shape[0], X.shape[1]
        for b in range(B):
            x = X[b]  # [C, T]
            # per-channel standardize over time
            x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
            # corr over channels
            cc = np.corrcoef(x)  # [C, C]
            if corr_sum is None:
                corr_sum = np.zeros_like(cc, dtype=np.float64)
            # replace nan (rare) with 0 before adding
            np.nan_to_num(cc, nan=0.0, copy=False)
            corr_sum += cc
            n_seen += 1
            if n_seen >= max_windows:
                break
        if n_seen >= max_windows:
            break

    if corr_sum is None:
        # fallback: identity => clusters ~ identity
        C = N_CHANS if C is None else C
        P = np.eye(C, dtype=np.float32)
        return torch.from_numpy(P).unsqueeze(-1)  # [C, C, 1]

    avg_corr = corr_sum / max(n_seen, 1)
    # features for chan c = its row in avg_corr
    feats = avg_corr.astype(np.float32)  # [C, C]
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = kmeans.fit_predict(feats)  # [C]

    # Build averaging matrix P[K, C]
    P = np.zeros((n_clusters, C), dtype=np.float32)
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            # empty cluster: spread tiny weight uniformly (robustness)
            P[k, :] = 1.0 / C
        else:
            P[k, idx] = 1.0 / float(len(idx))

    # Conv1d weight shape: [out_ch=K, in_ch=C, 1]
    return torch.from_numpy(P).unsqueeze(-1)


# ---------- submission writer (self-contained) ----------
def write_submission_py(out_dir: Path):
    code = r'''
import os
from pathlib import Path
import torch
import torch.nn as nn

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass


class SimpleEEGRegressor(nn.Module):
    def __init__(self, n_in_ch: int, n_filters: int = 64, kernel_size: int = 11):
        super().__init__()
        self.n_in_ch = n_in_ch
        self.conv1 = nn.Conv1d(n_in_ch, n_filters, kernel_size=kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.act1 = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(n_filters, 1)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool(x).squeeze(-1)
        x = self.head(x)
        return x


class ClusteredEEGRegressor(nn.Module):
    def __init__(self, projector_in: int, projector_out: int,
                 n_filters: int = 64, kernel_size: int = 11):
        super().__init__()
        self.projector = nn.Conv1d(projector_in, projector_out, kernel_size=1, bias=False)
        self.backbone = SimpleEEGRegressor(n_in_ch=projector_out,
                                           n_filters=n_filters,
                                           kernel_size=kernel_size)

    def forward(self, x):
        x = self.projector(x)
        return self.backbone(x)


class ChannelAdapter(nn.Module):
    """Adapt incoming channels to expected projector_in via slice/pad."""
    def __init__(self, model: ClusteredEEGRegressor, projector_in: int):
        super().__init__()
        self.model = model
        self.expected_c = projector_in

    def forward(self, x):
        b, c_in, t = x.shape
        c_exp = self.expected_c
        if c_in == c_exp:
            x_in = x
        elif c_in > c_exp:
            x_in = x[:, :c_exp, :]
        else:
            pad = torch.zeros(b, c_exp - c_in, t, dtype=x.dtype, device=x.device)
            x_in = torch.cat([x, pad], dim=1)
        return self.model(x_in)


def _resolve(name: str) -> Path:
    for p in (Path("/app/output")/name, Path("/app/ingested_program")/name, Path(".")/name):
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing {name}")


def _make_from_weights(wp: Path, device: torch.device):
    state = torch.load(wp, map_location=device)

    # Infer meta
    n_filters = int(state.get("__meta_n_filters", torch.tensor(64)).item())
    ksize = 11
    if "backbone.conv1.weight" in state:
        ksize = int(state["backbone.conv1.weight"].shape[-1])

    if "projector.weight" in state:
        # Trained WITH projector (clustering)
        proj_w = state["projector.weight"]  # [K, C, 1]
        k_out, c_in = int(proj_w.shape[0]), int(proj_w.shape[1])
        model = ClusteredEEGRegressor(projector_in=c_in, projector_out=k_out,
                                      n_filters=n_filters, kernel_size=ksize)
        model.load_state_dict(state, strict=True)
        model.eval()
        return ChannelAdapter(model, projector_in=c_in)
    else:
        # Trained WITHOUT projector: create identity projector so shapes match
        if "backbone.conv1.weight" not in state:
            raise RuntimeError("Cannot infer input channel count; missing 'backbone.conv1.weight'.")
        c_in = int(state["backbone.conv1.weight"].shape[1])
        k_out = c_in

        model = ClusteredEEGRegressor(projector_in=c_in, projector_out=k_out,
                                      n_filters=n_filters, kernel_size=ksize)
        with torch.no_grad():
            eye = torch.eye(c_in, dtype=torch.float32).unsqueeze(-1)  # [C, C, 1]
            model.projector.weight.copy_(eye)

        model.load_state_dict(state, strict=False)
        model.eval()
        return ChannelAdapter(model, projector_in=c_in)


class Submission:
    # Codabench calls: Submission(SFREQ, DEVICE)
    def __init__(self, SFREQ=None, DEVICE=None, *_, **__):
        self.device = DEVICE if DEVICE is not None else torch.device("cpu")
        self.sfreq = SFREQ

    def get_model_challenge_1(self):
        return _make_from_weights(_resolve("weights_challenge_1.pt"), self.device)

    def get_model_challenge_2(self):
        return _make_from_weights(_resolve("weights_challenge_2.pt"), self.device)
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


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Train (with/without clustering) & build Codabench ZIP")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--save_zip", action="store_true")

    # clustering options
    parser.add_argument("--use_clustering", action="store_true")
    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--cluster_windows", type=int, default=400,
                        help="How many windows to sample to build the correlation-based clusters")
    parser.add_argument("--freeze_projector", action="store_true",
                        help="If set, keep the projector fixed (cluster averages) during training")

    # model minor knobs
    parser.add_argument("--n_filters", type=int, default=64)
    parser.add_argument("--kernel_size", type=int, default=11)

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

    # splits + loaders (we need train_set to build clusters)
    train_set, valid_set, test_set = subject_splits(windows, seed=args.seed)
    print(f"Split sizes → Train={len(train_set)}  Valid={len(valid_set)}  Test={len(test_set)}")

    # OPTIONAL: compute projection for clustering
    projector_weight = None
    if args.use_clustering:
        print(f"Building channel clusters: K={args.n_clusters}, windows={args.cluster_windows}")
        projector_weight = compute_cluster_projection(
            train_set, max_windows=args.cluster_windows,
            n_clusters=args.n_clusters, seed=args.seed, num_workers=min(args.num_workers, 4)
        )  # [K, C, 1]

    # loaders
    tr, va, te = build_loaders(train_set, valid_set, test_set, args.batch_size, args.num_workers)

    # model
    model = build_model_clustered(
        c_in=N_CHANS,
        use_clustering=args.use_clustering,
        n_clusters=args.n_clusters,
        projector_weight=projector_weight,
        projector_trainable=not args.freeze_projector,
        n_filters=args.n_filters,
        kernel_size=args.kernel_size,
    ).to(device)

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

    # save weights (add a tiny meta field for n_filters so submission can mirror nicely)
    def _save_weights(p: Path, state):
        state = {k: v for k, v in state.items()}
        state["__meta_n_filters"] = torch.tensor(args.n_filters)
        torch.save(state, p)

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    p1 = OUT_DIR / "weights_challenge_1.pt"
    p2 = OUT_DIR / "weights_challenge_2.pt"
    _save_weights(p1, model.state_dict())
    _save_weights(p2, model.state_dict())
    print(f"Saved weights: {p1} ({human_size(p1)})")
    print(f"Saved weights: {p2} ({human_size(p2)})")

    if args.save_zip:
        write_submission_py(OUT_DIR)
        zp = build_zip(OUT_DIR, "submission-to-upload.zip")
        print(f"Built ZIP:     {zp} ({human_size(zp)})")

    print("Done.")
