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

# ✅ use your new SSL (contrastiveOverChannel.py renamed to contrastive_over_channel.py)
from ..ssl.contrastive_over_channel import (
    train_ssl_encoder,
    build_channel_projection_from_ssl,
)

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

def _write_submission_py(out_dir: Path):
    """
    Write a Codabench-ready submission.py that:
      • Loads projection.npy (M x 129) from /app/input/res (or identity if missing)
      • Applies fixed 1x1 conv with that projection to map 129->M channels
      • Builds EEGNeX(n_chans=M) as backbone
      • Loads only backbone weights (ignores projector/SSL heads), tolerant (strict=False)
    """
    code = r'''import os
import numpy as np
import torch
import torch.nn as nn
from braindecode.models import EEGNeX

RES_DIR_ENV = "EEG2025_RES_DIR"

def _res_dir():
    return os.environ.get(RES_DIR_ENV, "/app/input/res")

def _load_projection_or_identity(n_in: int = 129):
    path = os.path.join(_res_dir(), "projection.npy")
    if os.path.isfile(path):
        W = np.load(path)
        if W.ndim != 2 or W.shape[1] != n_in:
            raise RuntimeError(f"projection.npy has wrong shape {W.shape}; expected (M,{n_in})")
        return torch.tensor(W, dtype=torch.float32)
    return torch.eye(n_in, dtype=torch.float32)

class ProjectThenEEGNeX(nn.Module):
    def __init__(self, W: torch.Tensor, sfreq: int, device: torch.device):
        super().__init__()
        M, C = W.shape
        self.proj = nn.Conv1d(in_channels=C, out_channels=M, kernel_size=1, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(W.unsqueeze(-1))  # (M, C, 1)
        for p in self.proj.parameters():
            p.requires_grad_(False)
        self.backbone = EEGNeX(n_chans=M, n_outputs=1, sfreq=sfreq, n_times=int(2 * sfreq))
        self.to(device)

    def forward(self, x):
        x = self.proj(x)
        return self.backbone(x)

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def _clean_state_dict(self, sd: dict) -> dict:
        # unwrap common wrappers
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        cleaned = {}
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[len("module."):]
            # drop projector/ssl/head params
            if k.startswith("projector.") or k.startswith("ssl_head.") or k.startswith("head."):
                continue
            cleaned[k] = v
        return cleaned

    def _load_weights_backbone_only(self, model: ProjectThenEEGNeX, filename: str):
        path = os.path.join(_res_dir(), filename)
        if not os.path.isfile(path):
            return
        sd = torch.load(path, map_location=self.device)
        sd = self._clean_state_dict(sd)
        # Map "backbone.*" -> backbone, or accept plain EEGNeX keys
        bb_sd = {}
        for k, v in sd.items():
            if k.startswith("backbone."):
                bb_sd[k[len("backbone."):]] = v
            else:
                bb_sd[k] = v
        model.backbone.load_state_dict(bb_sd, strict=False)

    def _build_model(self):
        W = _load_projection_or_identity(n_in=129).to(self.device)
        return ProjectThenEEGNeX(W, self.sfreq, self.device)

    def get_model_challenge_1(self):
        m = self._build_model()
        self._load_weights_backbone_only(m, "weights_challenge_1.pt")
        return m

    def get_model_challenge_2(self):
        m = self._build_model()
        self._load_weights_backbone_only(m, "weights_challenge_2.pt")
        return m
'''
    (out_dir / "submission.py").write_text(code)

def _build_zip(out_dir: Path, zip_name="submission-to-upload.zip"):
    import zipfile
    to_zip = [out_dir / "submission.py", out_dir / "weights_challenge_1.pt", out_dir / "weights_challenge_2.pt"]
    # include projection if present
    proj = out_dir / "projection.npy"
    if proj.exists():
        to_zip.append(proj)
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in to_zip:
            zf.write(p, arcname=p.name)
    return zip_path

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Train (optionally SSL+cluster+PCA) and build Codabench ZIP")
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="output")
    ap.add_argument("--save_zip", action="store_true")

    # clustering / SSL options (training-time only)
    ap.add_argument("--cluster_mode", type=str, default="ssl_pca", choices=["none", "ssl_pca"])
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

    # 3) Build model (optionally with clustered projector from SSL+PCA)
    if args.cluster_mode == "ssl_pca":
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

        # ✅ persist projection for Codabench inference
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.save(OUT_DIR / "projection.npy", W.astype(np.float32))

        C_out, C_in = W.shape
        assert C_in == N_CHANS, f"Expected in_chans={N_CHANS}, got {C_in}"

        model = ClusteredEEGNeX(in_chans=C_in, out_chans=C_out, sfreq=SFREQ).to(device)
        with torch.no_grad():
            w = torch.from_numpy(W).float().unsqueeze(-1)  # (C_out, C_in, 1)
            model.projector.weight.copy_(w.to(model.projector.weight.device))
        model.projector.weight.requires_grad_(False)
    else:
        model = EEGNeX(n_chans=N_CHANS, n_outputs=1, sfreq=SFREQ, n_times=int(WIN_SEC * SFREQ)).to(device)

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
        _write_submission_py(OUT_DIR)
        zp = _build_zip(OUT_DIR, "submission-to-upload.zip")
        print(f"Built ZIP:     {zp} ({human_size(zp)})")

    print("Done.")
