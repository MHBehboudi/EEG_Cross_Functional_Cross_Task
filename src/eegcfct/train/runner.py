# src/eegcfct/train/runner.py
import argparse, copy, math
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
from ..ssl.contrastive_over_channel import train_ssl_encoder, build_channel_projection_from_ssl
from ..models.graph_transformer import ClusteredDeMega

# ---------- utils ----------
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

# ---------- submission writer ----------
def _write_submission_py(out_dir: Path, arch: str, use_projector: bool):
    """
    Create a submission.py matching the chosen arch and weight keys.
    """
    code = f'''import torch, torch.nn as nn
from braindecode.models import EEGNeX

try:
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
except Exception: pass

# === light graph+transformer (same as in training) ===
class DepthwiseTemporalTokenizer(nn.Module):
    def __init__(self, n_chans, k=9, p=4, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(n_chans, n_chans, kernel_size=k, padding=p, groups=n_chans, bias=False)
        self.bn   = nn.BatchNorm1d(n_chans)
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")
    def forward(self, x):
        h = self.conv(x); h = self.bn(h); h = nn.functional.gelu(h); h = self.drop(h)
        return h.transpose(1, 2)

class ChannelGraphBlock(nn.Module):
    def __init__(self, n_chans, hid):
        super().__init__()
        self.A = nn.Parameter(torch.randn(n_chans, n_chans)*0.02)
        self.lin = nn.Linear(n_chans, n_chans)
        self.proj = nn.Linear(n_chans, hid)
        self.norm = nn.LayerNorm(hid)
    def forward(self, x):
        B,T,C = x.shape
        A = nn.functional.softplus(self.A) + 1e-6
        A = A / (A.sum(dim=-1, keepdim=True)+1e-6)
        y = torch.matmul(x, A.t())
        y = nn.functional.gelu(self.lin(y)) + x
        y = self.proj(y)
        return self.norm(y)

class TimeTransformer(nn.Module):
    def __init__(self, dim, depth=2, heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=int(dim*mlp_ratio),
                                           dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(self.enc(x))

class DeMegaLikeBackbone(nn.Module):
    def __init__(self, n_chans, sfreq, win_sec, use_graph=True, token_k=9, d_model=128, depth=2, heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.tokenizer = DepthwiseTemporalTokenizer(n_chans, k=token_k, p=token_k//2, dropout=dropout)
        self.proj_in   = nn.Linear(n_chans, d_model)
        self.graph = ChannelGraphBlock(n_chans, hid=d_model) if use_graph else None
        self.time_tf = TimeTransformer(d_model, depth=depth, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
    def forward(self, x):
        tok = self.tokenizer(x)      # (B,T,C)
        g = self.graph(tok) if self.graph is not None else self.proj_in(tok)
        y = self.time_tf(g).mean(dim=1)
        return self.head(y)

class ClusteredDeMega(nn.Module):
    def __init__(self, in_chans, out_chans, sfreq, win_sec, **kw):
        super().__init__()
        if out_chans is not None and out_chans != in_chans:
            self.projector = nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False)
            n_backbone_ch = out_chans
        else:
            self.projector = None
            n_backbone_ch = in_chans
        self.backbone = DeMegaLikeBackbone(n_chans=n_backbone_ch, sfreq=sfreq, win_sec=win_sec, **kw)
    def forward(self, x):
        if self.projector is not None: x = self.projector(x)
        return self.backbone(x)

class ClusteredEEGNeX(torch.nn.Module):
    def __init__(self, in_chans, out_chans, sfreq):
        super().__init__()
        self.projector = nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False)
        self.backbone  = EEGNeX(n_chans=out_chans, n_outputs=1, sfreq=sfreq, n_times=int(2 * sfreq))
    def forward(self, x):
        x = self.projector(x); return self.backbone(x)

def _build_for_weights(sd, sfreq):
    # decide arch by keys
    if "projector.weight" in sd:
        out_c, in_c, _ = sd["projector.weight"].shape
        # use matching backbone keys
        if any(k.startswith("backbone.time_tf") for k in sd.keys()):
            # demega graph/transformer
            m = ClusteredDeMega(in_chans=in_c, out_chans=out_c, sfreq=sfreq, win_sec=2.0)
        else:
            m = ClusteredEEGNeX(in_chans=in_c, out_chans=out_c, sfreq=sfreq)
    else:
        # plain backbones without projector
        if any(k.startswith("time_tf") or k.startswith("tokenizer") for k in sd.keys()):
            m = DeMegaLikeBackbone(n_chans=129, sfreq=sfreq, win_sec=2.0)
        else:
            m = EEGNeX(n_chans=129, n_outputs=1, sfreq=sfreq, n_times=int(2*sfreq))
    return m

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq  = SFREQ
        self.device = DEVICE
    def _load(self, fname):
        sd = torch.load("/app/input/res/"+fname, map_location=self.device)
        m  = _build_for_weights(sd, self.sfreq).to(self.device)
        m.load_state_dict(sd, strict=True); m.eval(); return m
    def get_model_challenge_1(self):
        return self._load("weights_challenge_1.pt")
    def get_model_challenge_2(self):
        return self._load("weights_challenge_2.pt")
'''
    (out_dir / "submission.py").write_text(code)

def _build_zip(out_dir: Path, zip_name="submission-to-upload.zip"):
    import zipfile
    to_zip = [out_dir / "submission.py", out_dir / "weights_challenge_1.pt", out_dir / "weights_challenge_2.pt"]
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in to_zip: zf.write(p, arcname=p.name)
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

    # architecture
    ap.add_argument("--arch", type=str, default="eegnex", choices=["eegnex", "transformer", "demega", "gnn"])
    ap.add_argument("--use_projector", type=int, default=1)   # 1=use SSL+cluster PCA projector; 0=plain
    # demega hyperparams
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--tf_depth", type=int, default=2)
    ap.add_argument("--tf_heads", type=int, default=4)
    ap.add_argument("--tf_mlp", type=float, default=2.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--token_k", type=int, default=9)

    # projector/SSL options
    ap.add_argument("--n_clusters", type=int, default=20)
    ap.add_argument("--pcs_per_cluster", type=int, default=3)
    ap.add_argument("--ssl_epochs", type=int, default=10)
    ap.add_argument("--ssl_steps", type=int, default=150)
    ap.add_argument("--ssl_batch", type=int, default=16)
    ap.add_argument("--ssl_crop", type=int, default=150)

    args = ap.parse_args()

    print("==== RUN CONFIG ====")
    print(f"ARCH={args.arch}  USE_PROJECTOR={args.use_projector}")
    print(f"EPOCHS={args.epochs}  BATCH_SIZE={args.batch_size}  WORKERS={args.num_workers}  SEED={args.seed}")
    print(f"SSL: epochs={args.ssl_epochs} steps={args.ssl_steps} batch={args.ssl_batch} crop={args.ssl_crop}")
    print(f"Projector: K={args.n_clusters}  PCs/cluster={args.pcs_per_cluster}")
    print(f"Mini={int(args.mini)}  PWD={Path.cwd()}")
    print("====================")

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    DATA_DIR = Path(args.data_dir); DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR  = Path(args.out_dir);  OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) load/preprocess/windows
    ds = load_dataset_ccd(mini=args.mini, cache_dir=DATA_DIR)
    ds = preprocess_offline(ds)
    windows = make_windows(ds)
    print("Windows ready. Columns:", windows.get_metadata().columns.tolist())

    # 2) splits + loaders
    train_set, valid_set, test_set = subject_splits(windows, seed=args.seed)
    print(f"Split sizes → Train={len(train_set)}  Valid={len(valid_set)}  Test={len(test_set)}")
    tr, va, te = build_loaders(train_set, valid_set, test_set, args.batch_size, args.num_workers)

    # 3) projector (optional)
    projector_W = None
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
        projector_W = build_channel_projection_from_ssl(
            train_set, ssl_enc, n_clusters=args.n_clusters, pcs_per_cluster=args.pcs_per_cluster, device=device
        )  # (C_out, C_in)
        assert projector_W.shape[1] == N_CHANS

    # 4) build backbone
    if args.arch == "eegnex":
        if projector_W is not None:
            model = ClusteredEEGNeX(in_chans=N_CHANS, out_chans=projector_W.shape[0], sfreq=SFREQ).to(device)
        else:
            model = EEGNeX(n_chans=N_CHANS, n_outputs=1, sfreq=SFREQ, n_times=int(WIN_SEC*SFREQ)).to(device)
    else:
        # DeMEGA-like or GNN/Transformer variants
        use_graph = (args.arch in ["gnn", "demega"])
        out_ch = projector_W.shape[0] if projector_W is not None else None
        model = ClusteredDeMega(
            in_chans=N_CHANS, out_chans=out_ch, sfreq=SFREQ, win_sec=WIN_SEC,
            use_graph=use_graph, token_k=args.token_k, d_model=args.d_model,
            depth=args.tf_depth, heads=args.tf_heads, mlp_ratio=args.tf_mlp, dropout=args.dropout
        ).to(device)

    # load projector weights if present and freeze
    if projector_W is not None:
        with torch.no_grad():
            w = torch.from_numpy(projector_W).float().unsqueeze(-1)  # (C_out,C_in,1)
            model.projector.weight.copy_(w.to(model.projector.weight.device))
        model.projector.weight.requires_grad_(False)

    # 5) train
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

    # 6) test
    tl, trm = eval_loop(te, model, loss_fn, device)
    print(f"TEST: loss={tl:.6f} rmse={trm:.6f}")

    # 7) save + zip
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    p1 = OUT_DIR / "weights_challenge_1.pt"
    p2 = OUT_DIR / "weights_challenge_2.pt"
    torch.save(model.state_dict(), p1); torch.save(model.state_dict(), p2)
    print(f"Saved weights: {p1} ({human_size(p1)})")
    print(f"Saved weights: {p2} ({human_size(p2)})")

    if args.save_zip:
        _write_submission_py(OUT_DIR, args.arch, bool(args.use_projector))
        zp = _build_zip(OUT_DIR, "submission-to-upload.zip")
        print(f"Built ZIP:     {zp} ({human_size(zp)})")
    print("Done.")
