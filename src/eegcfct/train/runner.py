# src/eegcfct/train/runner.py
import argparse
import copy
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from braindecode.models import EEGNeX

from ..data.ccd_windows import (
    load_dataset_ccd, preprocess_offline, make_windows, subject_splits,
    SFREQ, N_CHANS, WIN_SEC
)
from .loops import build_loaders, train_one_epoch, eval_loop

# === SSL utilities (already in your repo) ===
from ..ssl.contrastive_over_channel import (
    train_ssl_encoder,
    build_channel_projection_from_ssl,
)

# === Projected backbone container ===
import torch.nn as nn
class ClusteredEEGNeX(nn.Module):
    """
    Applies a fixed 1x1 conv projector (C_in->C_out), then EEGNeX over C_out channels.
    We freeze projector weights so they are purely from SSL+PCA.
    """
    def __init__(self, in_chans: int, out_chans: int, sfreq: int):
        super().__init__()
        self.projector = nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False)
        self.backbone  = EEGNeX(n_chans=out_chans, n_outputs=1, sfreq=sfreq, n_times=int(WIN_SEC * sfreq))

    def forward(self, x):
        x = self.projector(x)
        return self.backbone(x)

# ---------- helpers ----------
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

def _build_zip(out_dir: Path, zip_name="submission-to-upload.zip"):
    """
    Build a FLAT zip with exactly:
      - submission.py (from repo root; NOT generated here)
      - weights_challenge_1.pt
      - weights_challenge_2.pt
    """
    import zipfile
    to_zip = [
        Path("submission.py"),                    # from repo root
        out_dir / "weights_challenge_1.pt",
        out_dir / "weights_challenge_2.pt",
    ]
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in to_zip:
            zf.write(p, arcname=p.name)
    return zip_path

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Train (SSL+cluster+PCA projector) and build Codabench ZIP")
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="output")
    ap.add_argument("--save_zip", action="store_true")

    # SSL + projector options
    ap.add_argument("--n_clusters", type=int, default=20)
    ap.add_argument("--pcs_per_cluster", type=int, default=3)
    ap.add_argument("--ssl_epochs", type=int, default=10)
    ap.add_argument("--ssl_steps", type=int, default=150)
    ap.add_argument("--ssl_batch", type=int, default=16)
    ap.add_argument("--ssl_crop", type=int, default=150)

    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    DATA_DIR = Path(args.data_dir); DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR  = Path(args.out_dir);  OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load + preprocess + windows
    ds = load_dataset_ccd(mini=args.mini, cache_dir=DATA_DIR)
    ds = preprocess_offline(ds)                # keep as in startkit
    windows = make_windows(ds)
    print("Windows ready. Columns:", windows.get_metadata().columns.tolist())

    # 2) Subject splits + loaders
    train_set, valid_set, test_set = subject_splits(windows, seed=args.seed)
    print(f"Split sizes → Train={len(train_set)}  Valid={len(valid_set)}  Test={len(test_set)}")
    tr, va, te = build_loaders(train_set, valid_set, test_set, args.batch_size, args.num_workers)

    # 3) SSL pretraining → cluster+PCA → projector W
    print(f"[SSL] Pretraining encoder for {args.ssl_epochs} epochs x {args.ssl_steps} steps...")
    ssl_enc = train_ssl_encoder(
        train_set,
        epochs=args.ssl_epochs,
        steps_per_epoch=args.ssl_steps,
        batch_size=args.ssl_batch,
        crop_len=args.ssl_crop,
        device=device,
    )
    print(f"[SSL] Building channel projection with K={args.n_clusters}, PCs/cluster={args.pcs_per_cluster} ...")
    W = build_channel_projection_from_ssl(
        train_set,
        ssl_enc,
        n_clusters=args.n_clusters,
        pcs_per_cluster=args.pcs_per_cluster,
        device=device,
    )  # (C_out, C_in) = (K * pcs, 129)
    C_out, C_in = W.shape
    assert C_in == N_CHANS, f"Expected in_chans={N_CHANS}, got {C_in}"

    # 4) Model with frozen projector
    model = ClusteredEEGNeX(in_chans=C_in, out_chans=C_out, sfreq=SFREQ).to(device)
    with torch.no_grad():
        w = torch.from_numpy(W).float().unsqueeze(-1)  # (C_out, C_in, 1)
        model.projector.weight.copy_(w.to(model.projector.weight.device))
    model.projector.weight.requires_grad_(False)

    # 5) Train + early stop
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = CosineAnnealingLR(optim, T_max=max(args.epochs - 1, 1))
    loss_fn = MSELoss()

    patience, min_delta = 50, 1e-4
    best_rmse, best_state, no_improve = math.inf, None, 0
    for epoch in range(1, args.epochs + 1):
        tl, trm = train_one_epoch(tr, model, loss_fn, optim, sched, epoch, device)
        vl, vrm = eval_loop(va, model, loss_fn, device)
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

    # 6) Test
    tl, trm = eval_loop(te, model, loss_fn, device)
    print(f"TEST: loss={tl:.6f} rmse={trm:.6f}")

    # 7) Save weights (these INCLUDE projector.weight)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    p1 = OUT_DIR / "weights_challenge_1.pt"
    p2 = OUT_DIR / "weights_challenge_2.pt"
    torch.save(model.state_dict(), p1)
    torch.save(model.state_dict(), p2)
    print(f"Saved weights: {p1} ({human_size(p1)})")
    print(f"Saved weights: {p2} ({human_size(p2)})")

    # 8) Build flat ZIP (uses your repo's submission.py)
    if args.save_zip:
        zp = _build_zip(OUT_DIR, "submission-to-upload.zip")
        print(f"Built ZIP:     {zp} ({human_size(zp)})")

    print("Done.")
