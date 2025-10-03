# src/eegcfct/train/runner.py
import argparse
import copy
import math
from pathlib import Path

import torch
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..data.ccd_windows import (
    load_dataset_ccd, preprocess_offline, make_windows, subject_splits,
    SFREQ, N_CHANS, WIN_SEC
)
from .loops import build_loaders, train_one_epoch, eval_loop

# --- import the new simple models ---
from ..models.simple import SimpleEEGRegressor, ProjectedRegressor

# (optional) SSL bits – keep as you have them
from ..ssl.contrastive import (
    train_ssl_encoder,
    build_channel_projection_from_ssl,
)

# ---------- small utils ----------
def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def human_size(p: Path):
    try:
        return f"{p.stat().st_size/1e6:.2f} MB"
    except Exception:
        return "n/a"

# ---------- write submission.py (embedded simple model) ----------
def write_submission_py(out_dir: Path):
    """
    We embed the minimal model directly so Codabench needs only:
      - submission.py
      - weights_challenge_1.pt
      - weights_challenge_2.pt
    No external imports from your package are required.
    """
    code = r'''import torch
import torch.nn as nn

# Cap threads for Codabench CPU workers
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

class SimpleEEGRegressor(nn.Module):
    def __init__(self, n_chans: int, n_times: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_chans, 64, kernel_size=7, padding="same", bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding="same", bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 128, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x).squeeze(-1)
        return self.head(x)

class ProjectedRegressor(nn.Module):
    def __init__(self, n_chans_in: int, n_chans_out: int, n_times: int):
        super().__init__()
        self.projector = nn.Conv1d(n_chans_in, n_chans_out, kernel_size=1, bias=False)
        for p in self.projector.parameters():
            p.requires_grad = False
        self.backbone = SimpleEEGRegressor(n_chans_out, n_times)

    def forward(self, x):
        x = self.projector(x)
        return self.backbone(x)

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        self.n_times = int(2 * self.sfreq)

    def _build_from_state(self, state_path: str):
        # Peek state to detect if a projector is present and its shape
        state = torch.load(state_path, map_location="cpu")
        proj_key = "projector.weight"
        if any(k == proj_key for k in state.keys()):
            Cout, Cin, _ = state[proj_key].shape
            model = ProjectedRegressor(Cin, Cout, self.n_times).to(self.device)
            # load and move projector to device first
            model.load_state_dict(torch.load(state_path, map_location=self.device), strict=True)
        else:
            model = SimpleEEGRegressor(n_chans=129, n_times=self.n_times).to(self.device)
            model.load_state_dict(torch.load(state_path, map_location=self.device), strict=True)
        model.eval()
        return model

    def get_model_challenge_1(self):
        return self._build_from_state("/app/output/weights_challenge_1.pt")

    def get_model_challenge_2(self):
        return self._build_from_state("/app/output/weights_challenge_2.pt")
'''
    (out_dir / "submission.py").write_text(code)

def build_zip(out_dir: Path, zip_name="submission-to-upload.zip"):
    import zipfile
    to_zip = [out_dir / "submission.py",
              out_dir / "weights_challenge_1.pt",
              out_dir / "weights_challenge_2.pt"]
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in to_zip:
            zf.write(p, arcname=p.name)
    return zip_path

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Train simple backbone (optionally with projector) & build Codabench ZIP")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--save_zip", action="store_true")

    # projector/SSL knobs (if you’re using your SSL path; harmless if you skip)
    parser.add_argument("--use_ssl", action="store_true")
    parser.add_argument("--ssl_epochs", type=int, default=5)
    parser.add_argument("--ssl_steps_per_epoch", type=int, default=150)
    parser.add_argument("--proj_k", type=int, default=20)
    parser.add_argument("--proj_pcs", type=int, default=3)

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

    # splits + loaders
    train_set, valid_set, test_set = subject_splits(windows, seed=args.seed)
    print(f"Split sizes → Train={len(train_set)}  Valid={len(valid_set)}  Test={len(test_set)}")
    tr, va, te = build_loaders(train_set, valid_set, test_set, args.batch_size, args.num_workers)

    # build optional projector from SSL
    W = None
    if args.use_ssl:
        print(f"[SSL] Pretraining encoder for {args.ssl_epochs} epochs x {args.ssl_steps_per_epoch} steps...")
        ssl_enc = train_ssl_encoder(
            windows_ds=train_set,
            device=device,
            epochs=args.ssl_epochs,
            steps_per_epoch=args.ssl_steps_per_epoch,
        )
        print(f"[SSL] Building channel projection with K={args.proj_k}, PCs/cluster={args.proj_pcs} ...")
        with torch.no_grad():
            W = build_channel_projection_from_ssl(
                windows_ds=train_set,
                encoder=ssl_enc,
                device=device,
                n_clusters=args.proj_k,
                pcs_per_cluster=args.proj_pcs,
                n_win_for_pca=150,
            )  # numpy array (C_out, C_in)

    # model
    if W is not None:
        C_out, C_in = W.shape
        model = ProjectedRegressor(n_chans_in=C_in, n_chans_out=C_out, n_times=int(WIN_SEC * SFREQ)).to(device)
        with torch.no_grad():
            model.projector.weight.copy_(torch.from_numpy(W).float().unsqueeze(-1))
        for p in model.projector.parameters():
            p.requires_grad = False
    else:
        model = SimpleEEGRegressor(n_chans=N_CHANS, n_times=int(WIN_SEC * SFREQ)).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = CosineAnnealingLR(optim, T_max=max(args.epochs - 1, 1))
    loss_fn = MSELoss()

    # train w/ early stopping
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
        write_submission_py(OUT_DIR)
        zp = build_zip(OUT_DIR, "submission-to-upload.zip")
        print(f"Built ZIP:     {zp} ({human_size(zp)})")
    print("Done.")
