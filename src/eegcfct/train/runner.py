# src/eegcfct/train/runner.py

from __future__ import annotations
import argparse
import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from braindecode.models import EEGNeX

from ..data.ccd_windows import (
    load_dataset_ccd, preprocess_offline, make_windows, subject_splits,
    SFREQ, N_CHANS, WIN_SEC
)
from .loops import build_loaders, train_one_epoch, eval_loop
from ..ssl.contrastive import (
    train_ssl_encoder,
    build_channel_projection_from_ssl,
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


# ---------- model wrapper for channel projection ----------
class ClusteredEEGNeX(nn.Module):
    """
    Project channels with a fixed linear mixing (1x1 Conv over channel dim),
    then run EEGNeX on the reduced channel space.
    """
    def __init__(self, projector: nn.Conv1d, backbone: EEGNeX):
        super().__init__()
        self.projector = projector
        self.backbone = backbone

    def forward(self, x):
        # x: (B, C, T)
        x = self.projector(x)          # (B, C', T)
        y = self.backbone(x)           # (B, 1)
        return y


def _make_clustered_model(W_np: np.ndarray, device: torch.device) -> nn.Module:
    """
    Build ClusteredEEGNeX from projection matrix W (C x C_proj).
    """
    C, Cproj = W_np.shape  # channels_in x channels_out
    projector = nn.Conv1d(in_channels=C, out_channels=Cproj, kernel_size=1, bias=False)
    with torch.no_grad():
        w = torch.from_numpy(W_np.T).unsqueeze(-1)  # (Cproj, C, 1)
        projector.weight.copy_(w)
    projector = projector.to(device)

    backbone = EEGNeX(
        n_chans=Cproj, n_outputs=1, sfreq=SFREQ, n_times=int(WIN_SEC * SFREQ)
    ).to(device)
    model = ClusteredEEGNeX(projector, backbone).to(device)
    return model


def _make_plain_model(device: torch.device) -> nn.Module:
    return EEGNeX(
        n_chans=N_CHANS, n_outputs=1, sfreq=SFREQ, n_times=int(WIN_SEC * SFREQ)
    ).to(device)


def write_submission_py(out_dir: Path, used_projection: bool, c_proj: int | None):
    """
    Emit a submission.py that reconstructs either:
      - plain EEGNeX (no projector), or
      - ClusteredEEGNeX if state_dict contains 'projector.weight'
    """
    code = f"""\
import torch
import torch.nn as nn
from braindecode.models import EEGNeX

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

SFREQ = {SFREQ}
WIN_SEC = {WIN_SEC}
N_CHANS = {N_CHANS}

class ClusteredEEGNeX(nn.Module):
    def __init__(self, projector: nn.Conv1d, backbone: EEGNeX):
        super().__init__()
        self.projector = projector
        self.backbone = backbone
    def forward(self, x):
        x = self.projector(x)
        return self.backbone(x)

def _make_plain(device):
    m = EEGNeX(n_chans=N_CHANS, n_outputs=1, sfreq=SFREQ, n_times=int(WIN_SEC*SFREQ)).to(device)
    m.eval()
    return m

def _make_clustered(device, state_dict):
    # infer projected channels from weight
    w = state_dict['projector.weight']  # (Cproj, C, 1)
    Cproj, C, _ = w.shape
    projector = nn.Conv1d(in_channels=C, out_channels=Cproj, kernel_size=1, bias=False).to(device)
    projector.load_state_dict({{'weight': w}})
    backbone = EEGNeX(n_chans=Cproj, n_outputs=1, sfreq=SFREQ, n_times=int(WIN_SEC*SFREQ)).to(device)
    m = ClusteredEEGNeX(projector, backbone).to(device)
    m.eval()
    return m

class Submission:
    def __init__(self, SFREQ_in, DEVICE):
        self.sfreq = SFREQ_in
        self.device = DEVICE

    def _load(self, path):
        state = torch.load(path, map_location=self.device)
        if 'projector.weight' in state:
            m = _make_clustered(self.device, state)
        else:
            m = _make_plain(self.device)
        m.load_state_dict(state, strict=True)
        m.eval()
        return m

    def get_model_challenge_1(self):
        return self._load("/app/output/weights_challenge_1.pt")

    def get_model_challenge_2(self):
        return self._load("/app/output/weights_challenge_2.pt")
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
    parser = argparse.ArgumentParser(description="Train like startkit & build Codabench ZIP (with optional SSL clustering)")
    # data/training
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--save_zip", action="store_true")

    # SSL / clustering knobs
    parser.add_argument("--use_ssl", action="store_true", help="enable self-supervised channel clustering")
    parser.add_argument("--ssl_epochs", type=int, default=5)
    parser.add_argument("--ssl_steps", type=int, default=200)
    parser.add_argument("--clusters", type=int, default=20)
    parser.add_argument("--pcs_per_cluster", type=int, default=3)
    parser.add_argument("--cov_batches", type=int, default=25)

    # CSD controls (plumbed through to preprocessing if you later enable it in ccd_windows.py)
    parser.add_argument("--use_csd", action="store_true")
    parser.add_argument("--csd_sphere", type=str, default="auto", choices=["auto", "fixed", "none"])

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    DATA_DIR = Path(args.data_dir); DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR = Path(args.out_dir);   OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load + preprocess + windows
    ds = load_dataset_ccd(mini=args.mini, cache_dir=DATA_DIR)
    ds = preprocess_offline(ds, use_csd=args.use_csd, csd_sphere=args.csd_sphere)
    windows = make_windows(ds)
    print("Windows ready. Columns:", windows.get_metadata().columns.tolist())

    # splits + loaders
    train_set, valid_set, test_set = subject_splits(windows, seed=args.seed)
    print(f"Split sizes → Train={len(train_set)}  Valid={len(valid_set)}  Test={len(test_set)}")
    tr_loader, va_loader, te_loader = build_loaders(train_set, valid_set, test_set, args.batch_size, args.num_workers)

    # optional SSL clustering
    used_projection = False
    W_np = None
    if args.use_ssl:
        print(f"[SSL] Pretraining encoder for {args.ssl_epochs} epochs x {args.ssl_steps} steps...")
        ssl_enc = train_ssl_encoder(
            tr_loader, device=device,
            ssl_epochs=args.ssl_epochs, ssl_steps=args.ssl_steps,
            log=print,
        )
        print(f"[SSL] Building channel projection with K={args.clusters}, PCs/cluster={args.pcs_per_cluster} ...")
        W_np = build_channel_projection_from_ssl(
            ssl_enc, tr_loader, device=device,
            n_clusters=args.clusters, pcs_per_cluster=args.pcs_per_cluster,
            cov_batches=args.cov_batches, log=print,
        )
        used_projection = True

    # model
    if used_projection and W_np is not None and W_np.size > 0:
        model = _make_clustered_model(W_np, device)
    else:
        model = _make_plain_model(device)

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

    write_submission_py(OUT_DIR, used_projection=used_projection, c_proj=(None if W_np is None else W_np.shape[1]))
    if args.save_zip:
        zp = build_zip(OUT_DIR, "submission-to-upload.zip")
        print(f"Built ZIP:     {zp} ({human_size(zp)})")

    print("Done.")
