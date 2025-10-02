# src/eegcfct/train/runner.py
import argparse
import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import MSELoss
from braindecode.models import EEGNeX
from sklearn.cluster import KMeans

from ..data.ccd_windows import (
    load_dataset_ccd, preprocess_offline, make_windows, subject_splits,
    SFREQ, N_CHANS, WIN_SEC
)
from .loops import build_loaders, train_one_epoch, eval_loop


# ---------------- small utils ----------------
def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
            print(f"[Device] CUDA: {name}")
        except Exception:
            print("[Device] CUDA")
    else:
        print("[Device] CPU")
    return dev

def human_size(p: Path):
    try:
        return f"{p.stat().st_size/1e6:.2f} MB"
    except Exception:
        return "n/a"

def pick_amp_dtype(amp_arg: str, device: torch.device):
    if device.type != "cuda":
        return None
    if amp_arg == "off":
        return None
    if amp_arg == "fp16":
        return torch.float16
    if amp_arg == "bf16":
        return torch.bfloat16
    # auto
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# ---------------- model with projector ----------------
class ProjectedEEGNeX(nn.Module):
    """Conv1×1 projector (C->K) + EEGNeX(K chans)."""
    def __init__(self, in_ch: int, k_out: int, sfreq: int, n_times: int, eegnex_kwargs=None):
        super().__init__()
        self.projector = nn.Conv1d(in_ch, k_out, kernel_size=1, bias=False)
        kw = dict(n_chans=k_out, n_outputs=1, sfreq=sfreq, n_times=n_times)
        if eegnex_kwargs:
            kw.update(eegnex_kwargs or {})
        self.backbone = EEGNeX(**kw)

    def forward(self, x):
        x = self.projector(x)
        return self.backbone(x)


# ---------------- clustering projector init ----------------
@torch.no_grad()
def init_projector_kmeans(train_loader, n_clusters: int, device: torch.device) -> torch.Tensor:
    """
    Build K cluster-averaging projector: for each cluster, average member channels.

    Returns weight W with shape [K, C, 1] such that y_k = mean_{c in cluster k} x_c.
    """
    # Collect a few batches, average across batch dimension to get [C, T] features per channel
    feats = []
    max_batches = 4
    for i, batch in enumerate(train_loader):
        X, _ = batch[0], batch[1]  # [B, C, T]
        X = X.float()
        X = X.mean(dim=0)  # [C, T]
        feats.append(X.cpu().numpy())
        if i + 1 >= max_batches:
            break
    F = np.mean(np.stack(feats, axis=0), axis=0)  # [C, T]
    C, _ = F.shape

    # KMeans on channels; each channel is a sample with T features
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(F)  # [C]

    W = np.zeros((n_clusters, C), dtype=np.float32)
    for c, lab in enumerate(labels):
        W[lab, c] = 1.0
    counts = W.sum(axis=1, keepdims=True)
    counts[counts == 0] = 1.0
    W /= counts  # average within cluster
    W = torch.from_numpy(W).unsqueeze(-1)  # [K, C, 1]
    return W


# ---------------- Codabench submission writer ----------------
def write_submission_py(out_dir: Path):
    """Emit a submission.py that works on CPU or GPU and supports projector weights."""
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

    # If projector is present, infer C and K from its weight
    if "projector.weight" in state:
        pw = state["projector.weight"]              # [K, C, 1]
        k_out, c_in = int(pw.shape[0]), int(pw.shape[1])
        n_times = 200  # 2s * 100 Hz
        model = ProjectedEEGNeX(in_ch=c_in, k_out=k_out, sfreq=sfreq, n_times=n_times).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    # Fallback: identity projection
    c_in = 129
    n_times = 200
    model = ProjectedEEGNeX(in_ch=c_in, k_out=c_in, sfreq=sfreq, n_times=n_times).to(device)
    with torch.no_grad():
        eye = torch.eye(c_in, dtype=torch.float32, device=device).unsqueeze(-1)
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


# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description="GPU-ready training + cluster projector + Codabench zip")
    # data/run
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--save_zip", action="store_true")
    # clusters
    parser.add_argument("--use_clusters", action="store_true")
    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--projector_init", type=str, default="kmeans", choices=["kmeans", "identity", "random"])
    # GPU/AMP
    parser.add_argument("--amp", type=str, default="auto", choices=["auto", "off", "bf16", "fp16"])
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # GPU perf toggles
    if device.type == "cuda":
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    amp_dtype = pick_amp_dtype(args.amp, device)
    print(f"[AMP] dtype = {amp_dtype}")

    DATA_DIR = Path(args.data_dir); DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR = Path(args.out_dir);   OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------- data pipeline -------
    ds = load_dataset_ccd(mini=args.mini, cache_dir=DATA_DIR)
    ds = preprocess_offline(ds)
    windows = make_windows(ds)
