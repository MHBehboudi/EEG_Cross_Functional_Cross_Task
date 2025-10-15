# src/eegcfct/train/runner.py
import argparse
import copy
import math
from pathlib import Path
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
from ..models.clustered_eegnex import ClusteredEEGNeX
from ..models.demega import DeMEGA, ClusteredDeMEGA
from ..ssl.contrastive_over_channel import train_ssl_encoder, build_channel_projection_from_ssl

# ---------- utils ----------
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

def _write_submission_py(out_dir: Path, arch: str, use_projector: bool):
    """
    Create a submission.py that can reconstruct the model from weights.
    It embeds DeMEGA + ClusteredDeMEGA locally to avoid extra imports.
    """
    code = r'''import torch
import torch.nn as nn
from braindecode.models import EEGNeX

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

class DeMEGA(nn.Module):
    def __init__(self, n_chans, n_times, d_model=128, n_heads=4, depth=2):
        super().__init__()
        self.stem = nn.Conv1d(n_chans, d_model, kernel_size=5, padding=2, bias=False)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                               batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        h = self.stem(x)          # (B, d_model, T)
        h = h.transpose(1, 2)     # (B, T, d_model)
        h = self.encoder(h)       # (B, T, d_model)
        h = h.transpose(1, 2)     # (B, d_model, T)
        return self.head(h)       # (B, 1)

class ClusteredDeMEGA(nn.Module):
    def __init__(self, in_chans, out_chans, n_times, d_model=128, n_heads=4, depth=2):
        super().__init__()
        self.projector = nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False)
        self.backbone  = DeMEGA(out_chans, n_times, d_model=d_model, n_heads=n_heads, depth=depth)
    def forward(self, x):
        x = self.projector(x)
        return self.backbone(x)

class ClusteredEEGNeX(torch.nn.Module):
    def __init__(self, in_chans: int, out_chans: int, sfreq: int):
        super().__init__()
        self.projector = nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False)
        self.backbone  = EEGNeX(n_chans=out_chans, n_outputs=1, sfreq=sfreq, n_times=int(2 * sfreq))
    def forward(self, x):
        x = self.projector(x)
        return self.backbone(x)

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq  = SFREQ
        self.device = DEVICE

    def _load(self, path: str):
        sd = torch.load(path, map_location=self.device)

        # If there's a 1x1 projector in weights, read its shape
        proj_key = "projector.weight"
        if proj_key in sd:
            out_c, in_c, _ = sd[proj_key].shape
            # Disambiguate backbone by presence of DeMEGA keys (Transformer) vs EEGNeX
            is_demega = any(k.startswith("backbone.encoder.layers.0.self_attn") for k in sd.keys())
            if is_demega:
                model = ClusteredDeMEGA(in_chans=in_c, out_chans=out_c, n_times=int(2*self.sfreq))
            else:
                model = ClusteredEEGNeX(in_chans=in_c, out_chans=out_c, sfreq=self.sfreq)
        else:
            # No projector → plain backbone; guess by keys
            is_demega = any(k.startswith("encoder.layers.0.self_attn") or k.startswith("stem.") for k in sd.keys())
            if is_demega:
                model = DeMEGA(n_chans=129, n_times=int(2*self.sfreq))
            else:
                model = EEGNeX(n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2*self.sfreq))
        model.to(self.device).eval()
        model.load_state_dict(sd, strict=True)
        return model

    def get_model_challenge_1(self):
        m = self._load("/app/input/res/weights_challenge_1.pt"); m.eval(); return m

    def get_model_challenge_2(self):
        m = self._load("/app/input/res/weights_challenge_2.pt"); m.eval(); return m
'''
    (out_dir / "submission.py").write_text(code)

def _build_zip(out_dir: Path, zip_name="submission-to-upload.zip"):
    import zipfile
    to_zip = [out_dir / "submission.py", out_dir / "weights_challenge_1.pt", out_dir / "weights_challenge_2.pt"]
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in to_zip:
            zf.write(p, arcname=p.name)
    return zip_path

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Train (SSL+cluster+PCA projector) and build Codabench ZIP")
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="output")
    ap.add_argument("--save_zip", action="store_true")

    # Model arch + projector
    ap.add_argument("--arch", type=str, default="demega", choices=["eegnex", "transformer", "demega"])
    ap.add_argument("--use_projector", type=int, default=1)

    # Projector options
    ap.add_argument("--n_clusters", type=int, default=20)
    ap.add_argument("--pcs_per_cluster", type=int, default=3)

    # SSL options
    ap.add_argument("--ssl_epochs", type=int, default=10)
    ap.add_argument("--ssl_steps", type=int, default=150)
    ap.add_argument("--ssl_batch", type=int, default=16)
    ap.add_argument("--ssl_crop", type=int, default=150)

    args = ap.parse_args()

    # banner
    print("==== RUN CONFIG ====")
    print(f"ARCH={args.arch}  USE_PROJECTOR={args.use_projector}")
    print(f"EPOCHS={args.epochs}  BATCH_SIZE={args.batch_size}  WORKERS={args.num_workers}  SEED={args.seed}")
    print(f"SSL: epochs={args.ssl_epochs} steps={args.ssl_steps} batch={args.ssl_batch} crop={args.ssl_crop}")
    print(f"Projector: K={args.n_clusters}  PCs/cluster={args.pcs_per_cluster}")
    print(f"Mini={1 if args.mini else 0}  PWD={Path().resolve()}")
    print("====================")

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    DATA_DIR = Path(args.data_dir); DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR  = Path(args.out_dir);  OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load + preprocess + windows
    ds = load_dataset_ccd(mini=args.mini, cache_dir=DATA_DIR)
    ds = preprocess_offline(ds)
    windows = make_windows(ds)
    print("Windows ready. Columns:", windows.get_metadata().columns.tolist())

    # 2) Subject splits + loaders
    train_set, valid_set, test_set = subject_splits(windows, seed=args.seed)
    print(f"Split sizes → Train={len(train_set)}  Valid={len(valid_set)}  Test={len(test_set)}")
    tr, va, te = build_loaders(train_set, valid_set, test_set, args.batch_size, args.num_workers)

    # 3) Build model (optionally with SSL projector)
    n_times = int(WIN_SEC * SFREQ)
    if args.use_projector:
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

        if args.arch == "demega":
            model = ClusteredDeMEGA(in_chans=C_in, out_chans=C_out, n_times=n_times).to(device)
        elif args.arch == "eegnex":
            model = ClusteredEEGNeX(in_chans=C_in, out_chans=C_out, sfreq=SFREQ).to(device)
        else:
            # simple transformer fallback (kept for completeness; DeMEGA is preferred)
            from ..models.transformer import ClusteredTransformer
            model = ClusteredTransformer(in_chans=C_in, out_chans=C_out, n_times=n_times).to(device)

        with torch.no_grad():
            w = torch.from_numpy(W).float().unsqueeze(-1)  # (C_out, C_in, 1)
            model.projector.weight.copy_(w.to(model.projector.weight.device))
        model.projector.weight.requires_grad_(False)

    else:
        if args.arch == "demega":
            model = DeMEGA(n_chans=N_CHANS, n_times=n_times).to(device)
        elif args.arch == "eegnex":
            model = EEGNeX(n_chans=N_CHANS, n_outputs=1, sfreq=SFREQ, n_times=n_times).to(device)
        else:
            from ..models.transformer import PlainTransformer
            model = PlainTransformer(n_chans=N_CHANS, n_times=n_times).to(device)

    # 4) Train
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

    # 5) Test
    tl, trm = eval_loop(te, model, loss_fn, device)
    print(f"TEST: loss={tl:.6f} rmse={trm:.6f}")

    # 6) Save weights (+ optional zip)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    p1 = OUT_DIR / "weights_challenge_1.pt"
    p2 = OUT_DIR / "weights_challenge_2.pt"
    torch.save(model.state_dict(), p1)
    torch.save(model.state_dict(), p2)
    print(f"Saved weights: {p1} ({human_size(p1)})")
    print(f"Saved weights: {p2} ({human_size(p2)})")

    if args.save_zip:
        _write_submission_py(OUT_DIR, args.arch, bool(args.use_projector))
        zp = _build_zip(OUT_DIR, "submission-to-upload.zip")
        print(f"Built ZIP:     {zp} ({human_size(zp)})")

    print("Done.")
