#!/usr/bin/env python3
# one_click_train_and_package.py
# End-to-end: data -> SSL -> clustering+PCA -> train -> save -> package

import os
import zipfile
import math
import copy
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---- import your repo's data utilities ----
from eegcfct.data.ccd_windows import (
    load_dataset_ccd, preprocess_offline, make_windows, subject_splits,
    SFREQ, N_CHANS, WIN_SEC
)

# ========= utils =========
def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def human_size(p: Path):
    try: return f"{p.stat().st_size/1e6:.2f} MB"
    except Exception: return "n/a"

# ========= SSL: channel-wise Conv+LSTM encoder =========
class TinyChLSTMEncoder(nn.Module):
    """
    Input:  (B, C, T)
    Output: (B, C, D)  — per-channel embeddings
    """
    def __init__(self, in_ch: int, emb_dim: int = 32):
        super().__init__()
        self.H = 16  # per-direction hidden size

        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=5, padding=2, groups=in_ch),
            nn.BatchNorm1d(in_ch),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Conv1d(in_ch, in_ch, kernel_size=5, padding=2, groups=in_ch),
            nn.BatchNorm1d(in_ch),
            nn.Dropout(0.2),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.H, bidirectional=True, batch_first=False)
        self.lin = nn.Linear(2 * self.H, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        h = self.conv(x)                          # (B, C, T)
        h = h.reshape(B * C, T).unsqueeze(-1)     # (B*C, T, 1)
        h = h.transpose(0, 1)                     # (T, B*C, 1)
        _, (h_n, _) = self.lstm(h)                # (2, B*C, H)
        h = h_n.transpose(0, 1).reshape(B * C, -1)  # (B*C, 2H)
        h = self.lin(h)                           # (B*C, D)
        return h.reshape(B, C, h.shape[-1])       # (B, C, D)

def random_crop_pair(x: torch.Tensor, crop_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    T = x.shape[-1]
    crop_len = min(crop_len, T)
    if crop_len == T:
        return x, x
    max_start = T - crop_len + 1
    s1 = torch.randint(0, max_start, (1,), device=x.device).item()
    s2 = torch.randint(0, max_start, (1,), device=x.device).item()
    return x[..., s1:s1 + crop_len], x[..., s2:s2 + crop_len]

def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    B, C, D = z1.shape
    z1 = F.normalize(z1.reshape(B * C, D), dim=-1)
    z2 = F.normalize(z2.reshape(B * C, D), dim=-1)
    reps = torch.cat([z1, z2], dim=0)        # (2*N, D)
    sim = reps @ reps.T                      # (2*N, 2*N)
    N = sim.shape[0]
    sim = sim / tau
    sim = sim.masked_fill(torch.eye(N, device=sim.device, dtype=torch.bool), float("-inf"))
    N_eff = B * C
    labels = torch.arange(N_eff, device=sim.device)
    pos = torch.cat([labels + N_eff, labels], dim=0)
    return F.cross_entropy(sim, pos)

def train_ssl_encoder(
    windows_ds,
    *,
    epochs: int = 10,
    steps_per_epoch: int = 150,
    batch_size: int = 25,
    crop_len: int = 150,
    device: torch.device,
) -> nn.Module:
    probe = DataLoader(windows_ds, batch_size=1, shuffle=True)
    X0 = next(iter(probe))[0]    # (1, C, T)
    C = X0.shape[1]

    enc = TinyChLSTMEncoder(in_ch=C, emb_dim=32).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(windows_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    it = iter(loader)

    enc.train()
    for ep in range(1, epochs + 1):
        losses: List[float] = []
        for _ in range(steps_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader); batch = next(it)
            X = batch[0].to(device).float()
            v1, v2 = random_crop_pair(X, crop_len)
            z1, z2 = enc(v1), enc(v2)
            loss = nt_xent(z1, z2, tau=0.2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"[SSL {ep:02d}/{epochs}] contrastive_loss={np.mean(losses):.4f}")
    enc.eval()
    return enc

# ========= build projection W by clustering channels + PCA =========
@torch.no_grad()
def compute_channel_embeddings(
    windows_ds,
    encoder: nn.Module,
    *,
    batches: int = 64,
    batch_size: int = 16,
    crop_len: int = 150,
    device: torch.device,
) -> np.ndarray:
    probe = DataLoader(windows_ds, batch_size=1, shuffle=True)
    X0 = next(iter(probe))[0].to(device).float()
    C = X0.shape[1]
    T_probe = min(crop_len, X0.shape[-1])
    D = encoder(torch.zeros(1, C, T_probe, device=device)).shape[-1]

    acc = torch.zeros(C, D, device=device)
    n = 0
    dl = DataLoader(windows_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    it = iter(dl)
    encoder.eval()

    for _ in range(batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)
        X = batch[0].to(device).float()
        v1, _ = random_crop_pair(X, min(crop_len, X.shape[-1]))
        z = encoder(v1)                     # (B, C, D)
        acc += z.mean(dim=0)                # (C, D)
        n += 1

    return (acc / max(n, 1)).detach().cpu().numpy()  # (C, D)

def _pca_basis_from_raw(
    windows_ds,
    chan_idx: np.ndarray,
    *,
    pcs_per_cluster: int,
    samples: int = 80,
    batch_size: int = 16,
) -> np.ndarray:
    dl = DataLoader(windows_ds, batch_size=batch_size, shuffle=True)
    it = iter(dl)
    X_blocks: List[np.ndarray] = []; got = 0
    while got < samples:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)
        X = batch[0][:, chan_idx, :].cpu().numpy()  # (B, nc, T)
        X_blocks.append(X.transpose(1, 0, 2).reshape(len(chan_idx), -1))
        got += 1

    Xcat = np.concatenate(X_blocks, axis=1)        # (nc, bigT)
    Xcat = (Xcat - Xcat.mean(axis=1, keepdims=True)) / (Xcat.std(axis=1, keepdims=True) + 1e-8)
    p = min(pcs_per_cluster, Xcat.shape[0])
    U = PCA(n_components=p, svd_solver="full").fit(Xcat.T).components_.T  # (nc, p)
    return U

def build_channel_projection_from_ssl(
    windows_ds,
    encoder: nn.Module,
    *,
    n_clusters: int,
    pcs_per_cluster: int,
    device: torch.device,
) -> np.ndarray:
    emb = compute_channel_embeddings(windows_ds, encoder, device=device)   # (C, D)
    C, _ = emb.shape
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(emb)                                           # (C,)
    W_rows: List[np.ndarray] = []
    for g in range(n_clusters):
        idx = np.where(labels == g)[0]
        if len(idx) == 0:
            continue
        U = _pca_basis_from_raw(windows_ds, idx, pcs_per_cluster=pcs_per_cluster)  # (nc, p)
        Wg = np.zeros((U.shape[1], C), dtype=np.float32)                            # (p, C)
        Wg[:, idx] = U.T
        W_rows.append(Wg)
    if len(W_rows) == 0:
        return np.eye(C, dtype=np.float32)
    return np.concatenate(W_rows, axis=0)  # (sum_p, C)

# ========= model =========
class ClusteredEEGNeX(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, sfreq: int):
        super().__init__()
        self.projector = nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False)
        from braindecode.models import EEGNeX
        self.backbone = EEGNeX(n_chans=out_chans, n_outputs=1, sfreq=sfreq, n_times=int(2 * sfreq))
    def forward(self, x):
        x = self.projector(x)
        return self.backbone(x)

# ========= training loops =========
def build_loaders(train_set, valid_set, test_set, batch_size: int, num_workers: int):
    tr = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=True)
    va = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    te = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return tr, va, te

def _batch_xy(batch, device):
    # train_set yields (X, y, infos) or (X, y, crop_inds, infos)
    if len(batch) >= 2:
        X = batch[0].to(device=device, dtype=torch.float32)
        y = batch[1].to(device=device, dtype=torch.float32).view(-1, 1)
        return X, y
    raise RuntimeError("Unexpected batch structure")

def train_one_epoch(loader, model, loss_fn, optim, sched, epoch, device):
    model.train()
    losses = []; sqerr = []; ycount = 0
    for batch in loader:
        X, y = _batch_xy(batch, device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optim.zero_grad(set_to_none=True)
        loss.backward(); optim.step()
        losses.append(float(loss.item()))
        with torch.no_grad():
            sqerr.append(((y_pred - y) ** 2).sum().item())
            ycount += y.numel()
    if sched is not None: sched.step()
    rmse = math.sqrt(sum(sqerr) / max(ycount, 1))
    return float(np.mean(losses)), rmse

@torch.no_grad()
def eval_loop(loader, model, loss_fn, device):
    model.eval()
    losses = []; sqerr = []; ycount = 0
    for batch in loader:
        X, y = _batch_xy(batch, device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        losses.append(float(loss.item()))
        sqerr.append(((y_pred - y) ** 2).sum().item())
        ycount += y.numel()
    rmse = math.sqrt(sum(sqerr) / max(ycount, 1))
    return float(np.mean(losses)), rmse

# ========= write Codabench submission.py (projector-in-weights; /app/input/res) =========
def write_submission_py(out_dir: Path):
    code = r'''import os
import torch
import torch.nn as nn
from braindecode.models import EEGNeX

# keep runtime lean
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

RES_DIR = os.environ.get("EEG2025_RES_DIR", "/app/input/res")

class ClusteredEEGNeX(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, sfreq: int):
        super().__init__()
        self.projector = nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False)
        self.backbone  = EEGNeX(n_chans=out_chans, n_outputs=1, sfreq=sfreq, n_times=int(2 * sfreq))
    def forward(self, x):
        x = self.projector(x)
        return self.backbone(x)

def _strip_module(sd: dict) -> dict:
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq  = SFREQ
        self.device = DEVICE

    def _load_weights_and_shape(self, filename: str):
        path = os.path.join(RES_DIR, filename)
        sd = torch.load(path, map_location=self.device)
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        sd = _strip_module(sd)

        if "projector.weight" in sd:
            out_c, in_c, _ = sd["projector.weight"].shape
            model = ClusteredEEGNeX(in_chans=in_c, out_chans=out_c, sfreq=self.sfreq).to(self.device)
        else:
            model = EEGNeX(n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2 * self.sfreq)).to(self.device)

        model.eval()
        model.load_state_dict(sd, strict=True)
        return model

    def get_model_challenge_1(self):
        return self._load_weights_and_shape("weights_challenge_1.pt")

    def get_model_challenge_2(self):
        return self._load_weights_and_shape("weights_challenge_2.pt")
'''
    (out_dir / "submission.py").write_text(code)

def build_zip(out_dir: Path, zip_name="submission-to-upload.zip"):
    to_zip = [
        out_dir / "submission.py",
        out_dir / "weights_challenge_1.pt",
        out_dir / "weights_challenge_2.pt",
    ]
    zp = out_dir / zip_name
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in to_zip:
            zf.write(p, arcname=p.name)
    return zp

# ========= main =========
def main():
    import argparse
    ap = argparse.ArgumentParser("One-click EEG2025 training + packaging")
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="output")

    # SSL + clustering/PCA options
    ap.add_argument("--n_clusters", type=int, default=20)
    ap.add_argument("--pcs_per_cluster", type=int, default=3)
    ap.add_argument("--ssl_epochs", type=int, default=5)
    ap.add_argument("--ssl_steps", type=int, default=80)
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
    ds = preprocess_offline(ds)
    windows = make_windows(ds)
    print("Windows ready. Columns:", windows.get_metadata().columns.tolist())

    # 2) Splits + loaders
    train_set, valid_set, test_set = subject_splits(windows, seed=args.seed)
    print(f"Split sizes → Train={len(train_set)}  Valid={len(valid_set)}  Test={len(test_set)}")
    tr, va, te = build_loaders(train_set, valid_set, test_set, args.batch_size, args.num_workers)

    # 3) SSL pretraining → projection W
    print(f"[SSL] Pretraining encoder for {args.ssl_epochs} epochs x {args.ssl_steps} steps...")
    ssl_enc = train_ssl_encoder(
        train_set, epochs=args.ssl_epochs, steps_per_epoch=args.ssl_steps,
        batch_size=args.ssl_batch, crop_len=args.ssl_crop, device=device
    )
    print(f"[SSL] Building channel projection with K={args.n_clusters}, PCs/cluster={args.pcs_per_cluster} ...")
    W = build_channel_projection_from_ssl(
        train_set, ssl_enc, n_clusters=args.n_clusters, pcs_per_cluster=args.pcs_per_cluster, device=device
    )  # (C_out, C_in) = (sum_p, 129)

    # 4) Build ClusteredEEGNeX and load projector
    C_out, C_in = W.shape
    assert C_in == N_CHANS, f"Expected in_chans={N_CHANS}, got {C_in}"
    model = ClusteredEEGNeX(in_chans=C_in, out_chans=C_out, sfreq=SFREQ).to(device)
    with torch.no_grad():
        w = torch.from_numpy(W).float().unsqueeze(-1)  # (C_out, C_in, 1)
        model.projector.weight.copy_(w.to(model.projector.weight.device))
    model.projector.weight.requires_grad_(False)

    # 5) Train / early stop
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

    # 7) Save weights (include projector.weight) and package
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    p1 = OUT_DIR / "weights_challenge_1.pt"
    p2 = OUT_DIR / "weights_challenge_2.pt"
    torch.save(model.state_dict(), p1)
    torch.save(model.state_dict(), p2)
    print(f"Saved weights: {p1} ({human_size(p1)})")
    print(f"Saved weights: {p2} ({human_size(p2)})")

    write_submission_py(OUT_DIR)
    zp = build_zip(OUT_DIR, "submission-to-upload.zip")
    print(f"Built ZIP:     {zp} ({human_size(zp)})")
    print("Done.")

if __name__ == "__main__":
    main()
