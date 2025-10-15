# src/eegcfct/train/runner.py
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
from ..ssl.contrastive_over_channel import train_ssl_encoder, build_channel_projection_from_ssl
from ..models.temporal_transformer import TemporalTransformerRegressor
from ..models.channel_gnn import ChannelGCNRegressor

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
    import zipfile
    to_zip = [Path("submission.py"), out_dir / "weights_challenge_1.pt", out_dir / "weights_challenge_2.pt"]
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in to_zip:
            zf.write(p, arcname=p.name)
    return zip_path

class ClusteredEEGNeX(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, sfreq: int):
        super().__init__()
        self.projector = nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False)
        self.backbone  = EEGNeX(n_chans=out_chans, n_outputs=1, sfreq=sfreq, n_times=int(WIN_SEC * sfreq))
    def forward(self, x): return self.backbone(self.projector(x))

def _bn_recalibration(train_loader, model, device):
    was_training = model.training
    model.train()
    with torch.no_grad():
        for X, *rest in train_loader:
            X = X.to(device).float()
            _ = model(X)
    if not was_training:
        model.eval()

def _compute_corr_knn_adjacency(train_set, k=8, samples=64, batch_size=16):
    from torch.utils.data import DataLoader
    dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    it = iter(dl)
    C = N_CHANS
    corr = np.zeros((C, C), dtype=np.float64)
    got = 0
    while got < samples:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)
        X = batch[0].numpy()                      # (B, C, T)
        X = X.transpose(1, 0, 2).reshape(C, -1)   # (C, B*T)
        X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)
        corr += (X @ X.T) / X.shape[1]
        got += 1
    corr = corr / max(got, 1)
    corr = np.abs(corr)
    np.fill_diagonal(corr, 0.0)
    adj = np.zeros_like(corr, dtype=np.float32)
    k = min(k, C-1)
    idx = np.argpartition(-corr, kth=k-1, axis=1)[:, :k]
    rows = np.repeat(np.arange(C)[:, None], idx.shape[1], axis=1)
    adj[rows, idx] = 1.0
    adj = np.maximum(adj, adj.T)                 # symmetrize
    adj = adj + np.eye(C, dtype=np.float32)      # self-loops
    d = adj.sum(1); d_inv_sqrt = (d + 1e-8) ** -0.5
    A_hat = (adj * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
    return A_hat.astype(np.float32)

def main():
    ap = argparse.ArgumentParser("Train (SSL+cluster+PCA projector OR GNN/Transformer) and build Codabench ZIP")
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="output")
    ap.add_argument("--save_zip", action="store_true")

    ap.add_argument("--arch", type=str, default="transformer", choices=["eegnex", "transformer", "gnn"])
    ap.add_argument("--use_projector", type=int, default=1)

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

    ds = load_dataset_ccd(mini=args.mini, cache_dir=DATA_DIR)
    ds = preprocess_offline(ds)
    windows = make_windows(ds)
    print("Windows ready. Columns:", windows.get_metadata().columns.tolist())

    train_set, valid_set, test_set = subject_splits(windows, seed=args.seed)
    print(f"Split sizes → Train={len(train_set)}  Valid={len(valid_set)}  Test={len(test_set)}")
    tr, va, te = build_loaders(train_set, valid_set, test_set, args.batch_size, args.num_workers)

    # Projector via SSL (unless gnn or disabled)
    C_in = N_CHANS
    proj_W = None
    use_proj = bool(args.use_projector)
    if args.arch == "gnn":
        use_proj = False
    if use_proj:
        print(f"[SSL] Pretraining encoder for {args.ssl_epochs} epochs x {args.ssl_steps} steps...")
        ssl_enc = train_ssl_encoder(train_set, epochs=args.ssl_epochs, steps_per_epoch=args.ssl_steps,
                                    batch_size=args.ssl_batch, crop_len=args.ssl_crop, device=device)
        print(f"[SSL] Building channel projection with K={args.n_clusters}, PCs/cluster={args.pcs_per_cluster} ...")
        proj_W = build_channel_projection_from_ssl(train_set, ssl_enc,
                                                   n_clusters=args.n_clusters, pcs_per_cluster=args.pcs_per_cluster,
                                                   device=device)  # (C_out, C_in)
        C_out, C_in_chk = proj_W.shape
        assert C_in_chk == C_in
    else:
        C_out = C_in

    # Model
    arch = args.arch
    if arch == "eegnex":
        if use_proj:
            model = ClusteredEEGNeX(in_chans=C_in, out_chans=C_out, sfreq=SFREQ).to(device)
            with torch.no_grad():
                w = torch.from_numpy(proj_W).float().unsqueeze(-1)
                model.projector.weight.copy_(w.to(model.projector.weight.device))
            model.projector.weight.requires_grad_(False)
        else:
            model = EEGNeX(n_chans=C_in, n_outputs=1, sfreq=SFREQ, n_times=int(WIN_SEC * SFREQ)).to(device)
    elif arch == "transformer":
        if use_proj:
            class ProjectThenTransformer(nn.Module):
                def __init__(self, W: np.ndarray, sfreq: int):
                    super().__init__()
                    M, C = W.shape
                    self.projector = nn.Conv1d(C, M, kernel_size=1, bias=False)
                    with torch.no_grad():
                        self.projector.weight.copy_(torch.from_numpy(W).float().unsqueeze(-1))
                    for p in self.projector.parameters(): p.requires_grad_(False)
                    self.backbone = TemporalTransformerRegressor(in_chans=M, sfreq=sfreq)
                def forward(self, x): return self.backbone(self.projector(x))
            model = ProjectThenTransformer(proj_W, SFREQ).to(device)
        else:
            model = TemporalTransformerRegressor(in_chans=C_in, sfreq=SFREQ).to(device)
    else:  # gnn
        model = ChannelGCNRegressor(in_chans=C_in, sfreq=SFREQ).to(device)
        A_hat = _compute_corr_knn_adjacency(train_set, k=8, samples=64, batch_size=16)
        with torch.no_grad(): model.adj.copy_(torch.from_numpy(A_hat))

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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

    _bn_recalibration(tr, model, device)

    tl, trm = eval_loop(te, model, loss_fn, device)
    print(f"TEST: loss={tl:.6f} rmse={trm:.6f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "arch": arch}
    p1 = OUT_DIR / "weights_challenge_1.pt"
    p2 = OUT_DIR / "weights_challenge_2.pt"
    torch.save(payload, p1)
    torch.save(payload, p2)
    print(f"Saved weights: {p1} ({human_size(p1)})")
    print(f"Saved weights: {p2} ({human_size(p2)})")

    if args.save_zip:
        zp = _build_zip(OUT_DIR, "submission-to-upload.zip")
        print(f"Built ZIP:     {zp} ({human_size(zp)})")

    print("Done.")
