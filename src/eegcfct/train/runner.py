# src/eegcfct/train/runner.py
from __future__ import annotations
import argparse
import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from braindecode.models import EEGNeX

from ..data.ccd_windows import (
    load_dataset_ccd, preprocess_offline, make_windows, subject_splits,
    SFREQ, N_CHANS, WIN_SEC,
)
from .loops import build_loaders, train_one_epoch, eval_loop
from ..ssl.contrastive import (
    train_ssl_encoder,
    build_channel_projection_from_ssl,
)

# ---------------- small utils ----------------
def set_seed(seed: int):
    import random
    import numpy as np
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

# ------------- model wrapper with projector -------------
class WithProjector(nn.Module):
    """
    Applies a fixed 1x1 Conv projector over channels, then runs EEGNeX.
    projector: nn.Conv1d(C_in=N_CHANS, C_out=F, kernel_size=1, bias=False)
    backbone: EEGNeX(n_chans=F, n_outputs=1, sfreq=SFREQ, n_times=WIN_SAMPLES)
    """
    def __init__(self, projector: nn.Conv1d, backbone: nn.Module):
        super().__init__()
        self.projector = projector
        self.backbone = backbone

    def forward(self, x):
        # x: (B, C, T)
        x = self.projector(x)
        return self.backbone(x)

def _make_projector_from_W(W_np: np.ndarray, device: torch.device) -> nn.Conv1d:
    """
    W_np: (C, F) channel projector, maps (B, C, T) -> (B, F, T)
    nn.Conv1d expects weight shape (F, C, 1) with out=in features arrangement.
    """
    C, F = W_np.shape
    W_t = torch.from_numpy(W_np.T).float().unsqueeze(-1)  # (F, C, 1)
    proj = nn.Conv1d(in_channels=C, out_channels=F, kernel_size=1, bias=False).to(device)
    with torch.no_grad():
        proj.weight.copy_(W_t.to(device))
    for p in proj.parameters():
        p.requires_grad_(False)  # fix the projector
    return proj

# ------------- submission writer -------------
def write_submission_py(out_dir: Path):
    code = f"""\
import torch
from torch import nn
from braindecode.models import EEGNeX

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

class WithProjector(nn.Module):
    def __init__(self, projector, backbone):
        super().__init__()
        self.projector = projector
        self.backbone = backbone
    def forward(self, x):
        x = self.projector(x)
        return self.backbone(x)

def _build_from_state(state, sfreq, device):
    # infer projector shape (F, C, 1)
    w = state['projector.weight']
    F, C, _ = w.shape
    projector = nn.Conv1d(in_channels=C, out_channels=F, kernel_size=1, bias=False).to(device)
    with torch.no_grad():
        projector.weight.copy_(w.to(device))
    for p in projector.parameters():
        p.requires_grad_(False)
    model = EEGNeX(n_chans=F, n_outputs=1, sfreq=sfreq, n_times=int({WIN_SEC} * sfreq)).to(device)
    wrapped = WithProjector(projector, model)
    return wrapped

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        state = torch.load("/app/output/weights_challenge_1.pt", map_location=self.device)
        m = _build_from_state(state, self.sfreq, self.device)
        m.load_state_dict(state, strict=True)
        m.eval()
        return m

    def get_model_challenge_2(self):
        state = torch.load("/app/output/weights_challenge_2.pt", map_location=self.device)
        m = _build_from_state(state, self.sfreq, self.device)
        m.load_state_dict(state, strict=True)
        m.eval()
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

# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser(description="Train (optionally with SSL projector) & build ZIP")
    # data / io
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--save_zip", action="store_true")
    # sup training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    # SSL projector
    parser.add_argument("--use_ssl", action="store_true")
    parser.add_argument("--ssl_epochs", type=int, default=10)
    parser.add_argument("--ssl_steps", type=int, default=150)
    parser.add_argument("--ssl_batch_size", type=int, default=64)
    parser.add_argument("--clusters", type=int, default=20)
    parser.add_argument("--pcs_per_cluster", type=int, default=3)
    parser.add_argument("--cov_batches", type=int, default=25)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    DATA_DIR = Path(args.data_dir); DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR = Path(args.out_dir);   OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) data
    ds = load_dataset_ccd(mini=args.mini, cache_dir=DATA_DIR)
    ds = preprocess_offline(ds)  # reref + robust standardize inside
    windows = make_windows(ds)
    print("Windows ready. Columns:", windows.get_metadata().columns.tolist())

    # 2) splits + loaders
    train_set, valid_set, test_set = subject_splits(windows, seed=args.seed)
    print(f"Split sizes → Train={len(train_set)}  Valid={len(valid_set)}  Test={len(test_set)}")
    tr, va, te = build_loaders(train_set, valid_set, test_set, args.batch_size, args.num_workers)

    # 3) projector (optional SSL)
    projector = None
    proj_out_ch = N_CHANS
    if args.use_ssl:
        ssl_enc = train_ssl_encoder(
            train_set,                     # pretrain on train only to avoid peeking
            device=device,
            epochs=args.ssl_epochs,        # <-- correct names
            steps_per_epoch=args.ssl_steps,
            batch_size=args.ssl_batch_size,
            num_workers=args.num_workers,
            lr=1e-3, weight_decay=1e-5, tau=0.2, verbose=True,
        )
        print(f"[SSL] Building channel projection with K={args.clusters}, PCs/cluster={args.pcs_per_cluster} ...")
        W_np = build_channel_projection_from_ssl(
            train_set, ssl_enc,
            n_clusters=args.clusters,
            pcs_per_cluster=args.pcs_per_cluster,
            cov_batches=args.cov_batches,
            batch_size=args.ssl_batch_size,
            num_workers=args.num_workers,
            device=device,
        )  # (C, F)
        projector = _make_projector_from_W(W_np, device=device)
        proj_out_ch = W_np.shape[1]
        print(f"[SSL] Projector built: in={N_CHANS} -> out={proj_out_ch}")

    # 4) model
    backbone = EEGNeX(n_chans=proj_out_ch, n_outputs=1, sfreq=SFREQ, n_times=int(WIN_SEC * SFREQ)).to(device)
    model = WithProjector(projector, backbone) if projector is not None else backbone

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = CosineAnnealingLR(optim, T_max=max(args.epochs - 1, 1))
    loss_fn = MSELoss()

    # 5) train with early stopping
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

    # 6) test
    tl, trm = eval_loop(te, model, loss_fn, device)
    print(f"TEST: loss={tl:.6f} rmse={trm:.6f}")

    # 7) save weights (+ optional ZIP)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
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
