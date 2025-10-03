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
from ..ssl.contrastive import (
    train_ssl_encoder,
    build_channel_projection_from_ssl,
)


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


# ---------- model wrapper (optional projector) ----------
class ProjectedEEGNeX(torch.nn.Module):
    """Optional 1x1 Conv projector over channels, then EEGNeX backbone."""
    def __init__(self, projector_weight: np.ndarray | None, sfreq: int = SFREQ, n_times: int = int(WIN_SEC * SFREQ)):
        super().__init__()
        if projector_weight is None:
            self.projector = None
            in_ch = N_CHANS
        else:
            W = torch.from_numpy(projector_weight.astype(np.float32))  # (C_in, C_out)
            self.projector = torch.nn.Conv1d(
                in_channels=W.shape[0], out_channels=W.shape[1],
                kernel_size=1, bias=False
            )
            with torch.no_grad():
                self.projector.weight.copy_(W.T[:, :, None])  # (C_out, C_in, 1)
            in_ch = W.shape[1]

        self.backbone = EEGNeX(
            n_chans=in_ch, n_outputs=1, sfreq=sfreq, n_times=n_times
        )

    def forward(self, x):
        if self.projector is not None:
            self.projector = self.projector.to(x.device, dtype=x.dtype)
            x = self.projector(x)
        return self.backbone(x)


def write_submission_py(out_dir: Path, have_projector: bool, k: int = 0, pcs: int = 0):
    if have_projector:
        code = f"""\
import torch
from braindecode.models import EEGNeX

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

class ProjectedEEGNeX(torch.nn.Module):
    def __init__(self, sfreq, n_times, projector_weight):
        super().__init__()
        if projector_weight is None:
            self.projector = None
            in_ch = 129
        else:
            W = torch.tensor(projector_weight, dtype=torch.float32)
            self.projector = torch.nn.Conv1d(in_channels=W.shape[0], out_channels=W.shape[1], kernel_size=1, bias=False)
            with torch.no_grad():
                self.projector.weight.copy_(W.t().unsqueeze(-1))
            in_ch = W.shape[1]
        self.backbone = EEGNeX(n_chans=in_ch, n_outputs=1, sfreq=sfreq, n_times=n_times)

    def forward(self, x):
        if self.projector is not None:
            self.projector = self.projector.to(x.device, dtype=x.dtype)
            x = self.projector(x)
        return self.backbone(x)

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def _make(self):
        model = ProjectedEEGNeX(sfreq=self.sfreq, n_times=int(2 * self.sfreq), projector_weight=None).to(self.device)
        return model

    def get_model_challenge_1(self):
        m = self._make()
        state = torch.load("/app/output/weights_challenge_1.pt", map_location=self.device)
        m.load_state_dict(state, strict=True)
        m.eval()
        return m

    def get_model_challenge_2(self):
        m = self._make()
        state = torch.load("/app/output/weights_challenge_2.pt", map_location=self.device)
        m.load_state_dict(state, strict=True)
        m.eval()
        return m
"""
    else:
        code = f"""\
import torch
from braindecode.models import EEGNeX

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def _make(self):
        model = EEGNeX(n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2 * self.sfreq)).to(self.device)
        model.eval()
        return model

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


def build_zip(out_dir: Path, zip_name="submission-to-upload.zip"):
    import zipfile
    to_zip = [out_dir / "submission.py", out_dir / "weights_challenge_1.pt", out_dir / "weights_challenge_2.pt"]
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in to_zip:
            zf.write(p, arcname=p.name)
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Train like startkit & build Codabench ZIP")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--save_zip", action="store_true")

    # SSL / clustering params
    parser.add_argument("--use_ssl", action="store_true")
    parser.add_argument("--ssl_epochs", type=int, default=10)
    parser.add_argument("--ssl_steps_per_epoch", type=int, default=150)
    parser.add_argument("--ssl_channels_per_step", type=int, default=32)
    parser.add_argument("--ssl_emb_dim", type=int, default=48)
    parser.add_argument("--ssl_amp", action="store_true")  # enable AMP
    parser.add_argument("--proj_k", type=int, default=20)
    parser.add_argument("--proj_pcs", type=int, default=3)
    parser.add_argument("--n_win_for_pca", type=int, default=50)

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

    # Optional SSL pretraining + channel projection
    projector_W = None
    have_projector = False
    if args.use_ssl:
        print(f"[SSL] Pretraining encoder for {args.ssl_epochs} epochs x {args.ssl_steps_per_epoch} steps...")
        ssl_enc = train_ssl_encoder(
            train_loader=tr,
            ssl_epochs=args.ssl_epochs,
            ssl_steps_per_epoch=args.ssl_steps_per_epoch,
            lr=1e-3,
            wd=1e-5,
            tau=0.1,
            device=device,
            in_ch=N_CHANS,
            emb_dim=args.ssl_emb_dim,
            windows_ds=windows,  # accepted but not used (for compat)
            channels_per_step=args.ssl_channels_per_step,
            use_amp=args.ssl_amp,
        )
        print(f"[SSL] Building channel projection with K={args.proj_k}, PCs/cluster={args.proj_pcs} ...")
        with torch.no_grad():
            projector_W = build_channel_projection_from_ssl(
                encoder=ssl_enc,
                train_loader=tr,
                k_clusters=args.proj_k,
                pcs_per_cluster=args.proj_pcs,
                n_win_for_pca=args.n_win_for_pca,
                device=device,
            )
        have_projector = True

    # model
    model = ProjectedEEGNeX(
        projector_weight=projector_W if have_projector else None,
        sfreq=SFREQ, n_times=int(WIN_SEC * SFREQ)
    ).to(device)

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

    # save weights (+ optional zip)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    p1 = OUT_DIR / "weights_challenge_1.pt"
    p2 = OUT_DIR / "weights_challenge_2.pt"
    torch.save(model.state_dict(), p1)
    torch.save(model.state_dict(), p2)
    print(f"Saved weights: {p1} ({human_size(p1)})")
    print(f"Saved weights: {p2} ({human_size(p2)})")

    # write submission (keeps projector inside state_dict → no extra files)
    write_submission_py(OUT_DIR, have_projector=have_projector, k=args.proj_k, pcs=args.proj_pcs)
    if args.save_zip:
        zp = build_zip(OUT_DIR, "submission-to-upload.zip")
        print(f"Built ZIP:     {zp} ({human_size(zp)})")

    print("Done.")
