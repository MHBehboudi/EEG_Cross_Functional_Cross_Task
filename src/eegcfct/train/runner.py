# src/eegcfct/train/runner.py
from __future__ import annotations
import argparse
import copy
import math
from pathlib import Path

import torch
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from braindecode.models import EEGNeX

from ..data.ccd_windows import (
    load_dataset_ccd, preprocess_offline, make_windows, subject_splits,
    SFREQ, N_CHANS, WIN_SEC
)
from ..models.projected_eegnex import ProjectedEEGNeX
from .loops import build_loaders, train_one_epoch, eval_loop


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


# ---------- optional KMeans init for projector ----------
def _try_kmeans_init_projector(projector: torch.nn.Conv1d, train_loader, max_batches: int = 4):
    """
    Initialize projector weights (out_ch=K, in_ch=C) via KMeans on
    channel patterns from a small sample of training data.
    Falls back silently if sklearn unavailable.
    """
    try:
        from sklearn.cluster import KMeans
    except Exception:
        return  # keep default random init

    device = projector.weight.device
    projector.eval()
    projector.requires_grad_(True)

    # Collect per-channel feature vectors by averaging over time & batches
    with torch.no_grad():
        feats = []
        n_collected = 0
        for b_idx, batch in enumerate(train_loader):
            if b_idx >= max_batches:
                break
            X = batch[0]  # [B, C, T] or (X, y, ...)
            if isinstance(X, (list, tuple)):
                X = X[0]
            X = X.to(device).float()
            # summarize channel dynamics by mean across (B, T)
            # -> one vector per channel (length = C)
            # We'll stack multiple batches and then cluster channels.
            # For stability, compute channel-wise energy (std across T) then mean over batch.
            # shape: [B, C, T] -> [B, C] std over T -> mean over B -> [C]
            ch_std = X.std(dim=-1)        # [B, C]
            ch_feat = ch_std.mean(dim=0)  # [C]
            feats.append(ch_feat.cpu())
            n_collected += 1

        if n_collected == 0:
            return

        ch_feat_all = torch.stack(feats, dim=0).mean(dim=0).numpy()  # [C]
        C = ch_feat_all.shape[0]
        K = projector.weight.shape[0]  # out_ch

        # Build a simple (C x 1) feature matrix and cluster channels into K groups
        X_feat = ch_feat_all.reshape(-1, 1)
        kmeans = KMeans(n_clusters=K, n_init="auto", random_state=0)
        labels = kmeans.fit_predict(X_feat)  # [C]

        # Create a mixing matrix: each cluster gets equal weight from its members
        import numpy as np
        W = np.zeros((K, C), dtype="float32")
        for c in range(C):
            W[labels[c], c] = 1.0
        # normalize rows to sum to 1 (avoid zero rows if rare)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W /= row_sums

        with torch.no_grad():
            # projector.weight shape: [K, C, 1]
            w = torch.from_numpy(W).to(device).unsqueeze(-1)
            projector.weight.copy_(w)
    # done


# ---------- submission writer (EEGNeX + optional projector) ----------
def write_submission_py(out_dir: Path):
    code = r'''
import os
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
    def __init__(self, in_ch: int, k_out: int, sfreq: int, n_times: int, eegnex_kwargs=None):
        super().__init__()
        self.projector = nn.Conv1d(in_ch, k_out, kernel_size=1, bias=False)
        kw = dict(n_chans=k_out, n_outputs=1, sfreq=sfreq, n_times=n_times)
        if eegnex_kwargs:
            kw.update(eegnex_kwargs)
        self.backbone = EEGNeX(**kw)

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

    # Two cases:
    # 1) projector present (clustered) -> build ProjectedEEGNeX with sizes from weight
    # 2) no projector -> create identity projector and pass through EEGNeX with inferred C
    if "projector.weight" in state:
        pw = state["projector.weight"]              # [K, C, 1]
        k_out, c_in = int(pw.shape[0]), int(pw.shape[1])

        # infer n_times from backbone first conv if present; fallback to 200 (2s*100Hz)
        n_times = 200
        # try to infer from any conv in EEGNeX that encodes time-length (best effort)
        # if not available, keep default 200 which matches the challenge windows
        model = ProjectedEEGNeX(in_ch=c_in, k_out=k_out, sfreq=sfreq, n_times=n_times)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    else:
        # no projector saved => infer input C from backbone
        # look for first conv weight in EEGNeX
        c_in = None
        for k, v in state.items():
            if k.endswith("conv1.weight") and v.ndim == 3:
                c_in = int(v.shape[1])  # [out, C, k]
                break
        if c_in is None:
            # last resort
            c_in = 129
        k_out = c_in
        n_times = 200

        model = ProjectedEEGNeX(in_ch=c_in, k_out=k_out, sfreq=sfreq, n_times=n_times)
        with torch.no_grad():
            eye = torch.eye(c_in, dtype=torch.float32).unsqueeze(-1)  # [C, C, 1]
            model.projector.weight.copy_(eye)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model


class Submission:
    # Codabench calls: Submission(SFREQ, DEVICE)
    def __init__(self, SFREQ=None, DEVICE=None, *_, **__):
        self.device = DEVICE if DEVICE is not None else torch.device("cpu")
        self.sfreq = int(SFREQ) if SFREQ is not None else 100

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


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Train like startkit (clustered) & build Codabench ZIP")
    # standard
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--save_zip", action="store_true")
    # clustering
    parser.add_argument("--use_clusters", action="store_true", help="Enable channel clustering projector")
    parser.add_argument("--n_clusters", type=int, default=50, help="Number of channel clusters (projector out channels)")
    parser.add_argument("--projector_init", choices=["learned", "kmeans"], default="learned",
                        help="Initialization for projector; always trained end-to-end.")

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

    # model
    n_times = int(WIN_SEC * SFREQ)
    if args.use_clusters:
        model = ProjectedEEGNeX(in_ch=N_CHANS, k_out=args.n_clusters, sfreq=SFREQ, n_times=n_times).to(device)
        if args.projector_init == "kmeans":
            _try_kmeans_init_projector(model.projector, tr, max_batches=4)
    else:
        # baseline (no projector)
        model = EEGNeX(n_chans=N_CHANS, n_outputs=1, sfreq=SFREQ, n_times=n_times).to(device)

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

    # save weights (challenge 1 & 2)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    p1 = OUT_DIR / "weights_challenge_1.pt"
    p2 = OUT_DIR / "weights_challenge_2.pt"

    state = model.state_dict()
    torch.save(state, p1)
    torch.save(state, p2)
    print(f"Saved weights: {p1} ({human_size(p1)})")
    print(f"Saved weights: {p2} ({human_size(p2)})")

    # zip
    if args.save_zip:
        write_submission_py(OUT_DIR)
        zp = build_zip(OUT_DIR, "submission-to-upload.zip")
        print(f"Built ZIP:     {zp} ({human_size(zp)})")

    print("Done.")
