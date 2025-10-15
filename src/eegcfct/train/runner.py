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
from ..ssl.contrastive_over_channel import (
    train_ssl_encoder, build_channel_projection_from_ssl
)
from ..models.demega import DeMEGA, ClusteredDeMEGA  # NEW

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

# ---------- write submission ----------
def _write_submission_py(out_dir: Path):
    code = r'''import torch
import torch.nn as nn
from braindecode.models import EEGNeX

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# ---- DeMEGA (same as training, compact) ----
class DeMEGA(torch.nn.Module):
    def __init__(self, in_chans, d_model=64, nhead=4, depth=2, k_per_channel=4, mlp_hidden=64, dropout=0.0):
        super().__init__()
        self.k = k_per_channel
        self.stem = nn.Sequential(
            nn.Conv1d(in_chans, in_chans*self.k, 7, padding=3, groups=in_chans, bias=False),
            nn.BatchNorm1d(in_chans*self.k),
            nn.GELU(),
            nn.Conv1d(in_chans*self.k, in_chans*self.k, 5, padding=2, groups=in_chans*self.k, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(self.k, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, 4*d_model, dropout, activation="gelu", batch_first=False, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head = nn.Sequential(nn.Linear(d_model, mlp_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, 1))
        self.pos = nn.Parameter(torch.zeros(in_chans, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.register_buffer("arch_id", torch.tensor([2], dtype=torch.int32))  # 2 = demega

    def forward(self, x):
        B, C, _ = x.shape
        h = self.stem(x).view(B, C, self.k)     # (B,C,k)
        h = self.proj(h) + self.pos.unsqueeze(0)  # (B,C,d)
        h = h.transpose(0,1)                    # (C,B,d)
        h = self.encoder(h).mean(dim=0)         # (B,d)
        return self.head(h)

class ClusteredDeMEGA(torch.nn.Module):
    def __init__(self, in_chans, out_chans, d_model=64, depth=2, nhead=4):
        super().__init__()
        self.projector = nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False)
        self.backbone = DeMEGA(out_chans, d_model=d_model, depth=depth, nhead=nhead)
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

    def _decode_arch(self, sd):
        # sd may be a state_dict or {"arch": "...", "state_dict": ...}
        arch = None
        state = sd
        if isinstance(sd, dict) and "state_dict" in sd:
            state = sd["state_dict"]
            arch = sd.get("arch", None)

        # If arch provided, return it.
        if isinstance(arch, str):
            return arch.lower(), state

        # Else infer from keys
        keys = list(state.keys())
        has_proj = any(k.startswith("projector.") for k in keys)
        if any(k.startswith("backbone.block_1") for k in keys):
            return ("eegnex_proj" if has_proj else "eegnex"), state
        if any(k.startswith("backbone.encoder.layers") for k in keys) and any(k.startswith("backbone.stem") for k in keys):
            return ("demega_proj" if has_proj else "demega"), state
        # Fallback: plain EEGNeX
        return "eegnex", state

    def _load(self, fname: str):
        raw = torch.load(fname, map_location=self.device)
        arch, sd = self._decode_arch(raw)

        if arch == "eegnex":
            model = EEGNeX(n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2*self.sfreq)).to(self.device)
        elif arch == "eegnex_proj":
            # infer in/out from projector
            oc, ic, _ = sd["projector.weight"].shape
            model = ClusteredEEGNeX(in_chans=ic, out_chans=oc, sfreq=self.sfreq).to(self.device)
        elif arch == "demega":
            # infer d_model from head input
            d_model = sd["head.0.weight"].shape[1]
            in_ch   = sd["pos"].shape[0]
            model = DeMEGA(in_chans=in_ch, d_model=d_model).to(self.device)
        elif arch == "demega_proj":
            oc, ic, _ = sd["projector.weight"].shape
            d_model = sd["backbone.head.0.weight"].shape[1]
            model = ClusteredDeMEGA(in_chans=ic, out_chans=oc, d_model=d_model).to(self.device)
        else:
            # last resort
            model = EEGNeX(n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2*self.sfreq)).to(self.device)

        model.load_state_dict(sd, strict=True)
        model.eval()
        return model

    def get_model_challenge_1(self):
        return self._load("/app/input/res/weights_challenge_1.pt")

    def get_model_challenge_2(self):
        return self._load("/app/input/res/weights_challenge_2.pt")
'''
    (out_dir / "submission.py").write_text(code)

def _save_weights(model: torch.nn.Module, path: Path, arch: str):
    payload = {"arch": arch, "state_dict": model.state_dict()}
    torch.save(payload, path)

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

    # NEW: backbone selection + projector toggle
    ap.add_argument("--arch", type=str, default="demega", choices=["eegnex", "transformer", "demega"])
    ap.add_argument("--use_projector", type=int, default=1)  # 1=True, 0=False

    # clustering / SSL options (training-time only)
    ap.add_argument("--n_clusters", type=int, default=20)
    ap.add_argument("--pcs_per_cluster", type=int, default=3)
    ap.add_argument("--ssl_epochs", type=int, default=10)
    ap.add_argument("--ssl_steps", type=int, default=150)
    ap.add_argument("--ssl_batch", type=int, default=16)
    ap.add_argument("--ssl_crop", type=int, default=150)

    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()

    print("==== RUN CONFIG ====")
    print(f"ARCH={args.arch}  USE_PROJECTOR={args.use_projector}")
    print(f"EPOCHS={args.epochs}  BATCH_SIZE={args.batch_size}  WORKERS={args.num_workers}  SEED={args.seed}")
    print(f"SSL: epochs={args.ssl_epochs} steps={args.ssl_steps} batch={args.ssl_batch} crop={args.ssl_crop}")
    print(f"Projector: K={args.n_clusters}  PCs/cluster={args.pcs_per_cluster}")
    print(f"Mini={1 if args.mini else 0}  PWD={Path.cwd()}")
    print("====================")

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

    # 3) Build model
    arch_name = args.arch.lower()
    using_proj = bool(args.use_projector)

    if using_proj:
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
            train_set, ssl_enc,
            n_clusters=args.n_clusters,
            pcs_per_cluster=args.pcs_per_cluster,
            device=device,
        )  # (C_out, C_in) = (K*pcs, 129)
        C_out, C_in = W.shape
        assert C_in == N_CHANS, f"Expected in_chans={N_CHANS}, got {C_in}"

        if arch_name == "demega":
            model = ClusteredDeMEGA(in_chans=C_in, out_chans=C_out, d_model=64, depth=2, nhead=4).to(device)
        elif arch_name == "eegnex":
            model = ClusteredEEGNeX(in_chans=C_in, out_chans=C_out, sfreq=SFREQ).to(device)
        else:
            # If you had a plain transformer backbone, you'd wrap similarly.
            model = ClusteredDeMEGA(in_chans=C_in, out_chans=C_out, d_model=64, depth=2, nhead=4).to(device)

        with torch.no_grad():
            w = torch.from_numpy(W).float().unsqueeze(-1)  # (C_out, C_in, 1)
            model.projector.weight.copy_(w.to(model.projector.weight.device))
        model.projector.weight.requires_grad_(False)
        arch_for_save = f"{arch_name}_proj"
    else:
        if arch_name == "demega":
            model = DeMEGA(in_chans=N_CHANS, d_model=64, depth=2, nhead=4).to(device)
        else:
            model = EEGNeX(n_chans=N_CHANS, n_outputs=1, sfreq=SFREQ, n_times=int(WIN_SEC * SFREQ)).to(device)
        arch_for_save = arch_name

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
    _save_weights(model, p1, arch_for_save)
    _save_weights(model, p2, arch_for_save)
    print(f"Saved weights: {p1} ({human_size(p1)})")
    print(f"Saved weights: {p2} ({human_size(p2)})")

    if args.save_zip:
        _write_submission_py(OUT_DIR)
        zp = _build_zip(OUT_DIR, "submission-to-upload.zip")
        print(f"Built ZIP:     {zp} ({human_size(zp)})")

    print("Done.")
