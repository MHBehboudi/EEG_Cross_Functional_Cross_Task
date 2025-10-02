# src/eegcfct/train/runner.py
from __future__ import annotations
import argparse, copy, math
from pathlib import Path
import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR

from braindecode.models import EEGNeX  # still import for baseline fallback
from ..data.ccd_windows import (
    load_dataset_ccd, preprocess_offline, make_windows, subject_splits,
    SFREQ, N_CHANS, WIN_SEC,
)
from .loops import build_loaders, train_one_epoch, eval_loop

# NEW:
from ..featurizers.cluster_features import (
    compute_channel_features_from_windows,
    cluster_channels,
    build_spatial_pca_projection,
    projection_matrix_to_conv1d_weight,
)
from ..models.projected_eegnex import ProjectedEEGNeX

# ---------- small utils ----------
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

def write_submission_py(out_dir: Path, use_projection: bool, K: int | None):
    """
    Create submission.py that rebuilds the SAME model at inference.
    If use_projection=True, we instantiate ProjectedEEGNeX and rely on
    baked-in projector weights from state_dict (no sklearn at inference).
    """
    if use_projection:
        code = f'''\
import torch
from torch import nn
from braindecode.models import EEGNeX

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

class ProjectedEEGNeX(torch.nn.Module):
    def __init__(self, P_weight: torch.Tensor, sfreq: float, n_times: int, n_outputs: int = 1):
        super().__init__()
        K, C, _ = P_weight.shape
        self.projector = nn.Conv1d(in_channels=C, out_channels=K, kernel_size=1, bias=False)
        with torch.no_grad():
            self.projector.weight.copy_(P_weight)
        for p in self.projector.parameters():
            p.requires_grad = False
        self.backbone = EEGNeX(n_chans=K, n_outputs=n_outputs, sfreq=sfreq, n_times=n_times)
    def forward(self, x):
        x = self.projector(x)
        return self.backbone(x)

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
    def _make(self):
        # Create a dummy projector; real weights come from state_dict
        K = {K if K is not None else 64}
        C = 129
        P_dummy = torch.zeros((K, C, 1), dtype=torch.float32)
        model = ProjectedEEGNeX(P_dummy, sfreq=self.sfreq, n_times=int(2 * self.sfreq)).to(self.device)
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
    else:
        code = f'''\
import torch
from braindecode.models import EEGNeX
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
        model = EEGNeX(n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2 * self.sfreq)).to(self.device)
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
    to_zip = [out_dir / "submission.py", out_dir / "weights_challenge_1.pt", out_dir / "weights_challenge_2.pt"]
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in to_zip:
            zf.write(p, arcname=p.name)
    return zip_path

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Train + build ZIP with optional unsupervised channel clustering + PCA")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--save_zip", action="store_true")

    # NEW: clustering/PCA knobs
    parser.add_argument("--cluster_method", type=str, default="kmeans", choices=["kmeans", "spectral", "ae"])
    parser.add_argument("--feature_type", type=str, default="corr", choices=["corr", "cov"])
    parser.add_argument("--n_clusters", type=int, default=20)
    parser.add_argument("--pcs_per_cluster", type=int, default=3)

    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    DATA_DIR = Path(args.data_dir); DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR = Path(args.out_dir);   OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load + preprocess + windows
    ds_all = load_dataset_ccd(mini=args.mini, cache_dir=DATA_DIR)
    ds_all = preprocess_offline(ds_all)
    windows = make_windows(ds_all)
    print("Windows ready. Columns:", windows.get_metadata().columns.tolist())

    # splits + loaders
    train_set, valid_set, test_set = subject_splits(windows, seed=args.seed)
    print(f"Split sizes → Train={len(train_set)}  Valid={len(valid_set)}  Test={len(test_set)}")
    tr_loader, va_loader, te_loader = build_loaders(train_set, valid_set, test_set, args.batch_size, args.num_workers)

    # ===== Build spatial projection (unsupervised) =====
    use_projection = (args.n_clusters > 0 and args.pcs_per_cluster > 0)
    model = None
    K_total = None

    if use_projection:
        print(f"[Clustering] method={args.cluster_method}  features={args.feature_type}  "
              f"n_clusters={args.n_clusters}  pcs_per_cluster={args.pcs_per_cluster}")

        F = compute_channel_features_from_windows(
            train_set, feature_type=args.feature_type, max_windows=200, stride=1, seed=args.seed
        )  # (C, C)
        labels = cluster_channels(F, method=args.cluster_method, n_clusters=args.n_clusters, seed=args.seed)  # (C,)

        P = build_spatial_pca_projection(
            train_set, labels,
            pcs_per_cluster=args.pcs_per_cluster,
            sample_stride=2,
            max_samples_per_cluster=120_000,
            seed=args.seed
        )  # (K, C)
        K_total = P.shape[0]
        W = projection_matrix_to_conv1d_weight(P)  # (K, C, 1)

        model = ProjectedEEGNeX(P_weight=W, sfreq=SFREQ, n_times=int(WIN_SEC * SFREQ)).to(device)
    else:
        model = EEGNeX(n_chans=N_CHANS, n_outputs=1, sfreq=SFREQ, n_times=int(WIN_SEC * SFREQ)).to(device)

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

    write_submission_py(OUT_DIR, use_projection=use_projection, K=K_total)
    if args.save_zip:
        zip_path = build_zip(OUT_DIR, "submission-to-upload.zip")
        print(f"Built ZIP: {zip_path} ({human_size(zip_path)})")
    print("Done.")
