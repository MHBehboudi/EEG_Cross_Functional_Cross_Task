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
from ..ssl.contrastive import (
    train_ssl_encoder, compute_channel_embeddings, kmeans, build_projection_matrix
)
from ..models.projector_model import ProjectorEEGNeX

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

def write_submission_py(out_dir: Path, proj_dim: int):
    code = f"""\
import torch
import torch.nn as nn
from braindecode.models import EEGNeX

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

class ChannelProjector(nn.Module):
    def __init__(self, P: torch.Tensor):
        super().__init__()
        proj_dim, n_ch = P.shape
        self.conv = nn.Conv1d(n_ch, proj_dim, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(P.to(torch.float32).unsqueeze(-1))
        for p in self.parameters(): p.requires_grad_(False)
    def forward(self, x): return self.conv(x)

class ProjectorEEGNeX(nn.Module):
    def __init__(self, P: torch.Tensor, sfreq: int, n_times: int):
        super().__init__()
        self.projector = ChannelProjector(P)
        self.backbone = EEGNeX(n_chans=P.shape[0], n_outputs=1, sfreq=sfreq, n_times=n_times)
    def forward(self, x):
        x = self.projector(x)
        return self.backbone(x)

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def _make(self):
        # P is stored inside the state dict (as conv weight), so we just create
        # a dummy P with right shape; it will be overwritten when loading.
        dummy_P = torch.eye({proj_dim}, {N_CHANS})
        m = ProjectorEEGNeX(dummy_P, sfreq=self.sfreq, n_times=int({WIN_SEC}*self.sfreq)).to(self.device)
        m.eval()
        return m

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

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Train + SSL channel clustering + ZIP")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--save_zip", action="store_true")

    # SSL clustering flags
    parser.add_argument("--use_ssl", action="store_true", help="enable contrastive SSL for channel clustering")
    parser.add_argument("--clusters", type=int, default=20)
    parser.add_argument("--pcs_per_cluster", type=int, default=3)
    parser.add_argument("--ssl_epochs", type=int, default=3)
    parser.add_argument("--ssl_steps", type=int, default=200)
    parser.add_argument("--ssl_batch", type=int, default=128)
    parser.add_argument("--ssl_temp", type=float, default=0.2)
    parser.add_argument("--ssl_samples_per_ch", type=int, default=256)

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    DATA_DIR = Path(args.data_dir); DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR = Path(args.out_dir);   OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load + preprocess + windows
    ds = load_dataset_ccd(mini=args.mini, cache_dir=DATA_DIR)
    ds = preprocess_offline(ds)  # (we kept EMA, band-pass already applied by challenge)
    windows = make_windows(ds)
    print("Windows ready. Columns:", windows.get_metadata().columns.tolist())

    # splits + loaders
    train_set, valid_set, test_set = subject_splits(windows, seed=args.seed)
    print(f"Split sizes → Train={len(train_set)}  Valid={len(valid_set)}  Test={len(test_set)}")
    tr_loader, va_loader, te_loader = build_loaders(train_set, valid_set, test_set, args.batch_size, args.num_workers)

    # ---------- build projector P via SSL (or identity) ----------
    if args.use_ssl:
        print(f"[SSL] Training encoder: epochs={args.ssl_epochs}, steps/epoch={args.ssl_steps}")
        ssl_enc = train_ssl_encoder(
            train_loader=tr_loader,
            device=device,
            epochs=args.ssl_epochs,
            steps_per_epoch=args.ssl_steps,
            batch_size=args.ssl_batch,
            emb_dim=128,
            temperature=args.ssl_temp,
            lr=1e-3,
            seed=args.seed,
        )
        print("[SSL] Computing per-channel embeddings…")
        ch_emb = compute_channel_embeddings(
            ssl_enc, tr_loader, device, n_channels=N_CHANS, samples_per_channel=args.ssl_samples_per_ch
        )  # [C, D]
        print("[SSL] KMeans on channel embeddings…")
        assign, _ = kmeans(ch_emb, K=args.clusters, iters=50, seed=args.seed)
        print("[SSL] Cluster-wise PCA → projection matrix")
        P = build_projection_matrix(tr_loader, device, n_channels=N_CHANS, assign=assign,
                                    pcs_per_cluster=args.pcs_per_cluster, max_windows=256)  # [proj_dim,C]
        proj_dim = P.shape[0]
    else:
        print("[SSL] Disabled → identity projection")
        P = torch.eye(N_CHANS, device=device)
        proj_dim = N_CHANS

    # ---------- model (projector + EEGNeX) ----------
    model = ProjectorEEGNeX(P, sfreq=SFREQ, n_times=int(WIN_SEC * SFREQ)).to(device)
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-5)
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
    p1 = OUT_DIR / "weights_challenge_1.pt"
    p2 = OUT_DIR / "weights_challenge_2.pt"
    torch.save(model.state_dict(), p1)
    torch.save(model.state_dict(), p2)
    print(f"Saved weights: {p1} ({human_size(p1)})")
    print(f"Saved weights: {p2} ({human_size(p2)})")

    if args.save_zip:
        write_submission_py(OUT_DIR, proj_dim=proj_dim)
        import zipfile
        zip_path = OUT_DIR / "submission-to-upload.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(OUT_DIR / "submission.py", arcname="submission.py")
            zf.write(p1, arcname="weights_challenge_1.pt")
            zf.write(p2, arcname="weights_challenge_2.pt")
        print(f"Built ZIP:     {zip_path} ({human_size(zip_path)})")

    print("Done.")
