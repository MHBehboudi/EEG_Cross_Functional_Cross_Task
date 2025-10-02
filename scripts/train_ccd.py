#!/usr/bin/env python3
import math, copy, zipfile, argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import MSELoss
from tqdm import tqdm

from eegdash.dataset import EEGChallengeDataset
from braindecode.datasets import BaseConcatDataset

from eegcfct.data import make_windows_ccd, subject_split_concat, SFREQ, N_CHANS, WIN_SEC
from eegcfct.model import make_eegnex

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_loaders(train_set, valid_set, test_set, batch_size, num_workers):
    tl = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers,
                    pin_memory=torch.cuda.is_available())
    vl = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=torch.cuda.is_available())
    te = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=torch.cuda.is_available())
    return tl, vl, te

def train_one_epoch(loader, model, loss_fn, optimizer, scheduler, device):
    model.train()
    total_loss, sum_sq, n = 0.0, 0.0, 0
    for batch in tqdm(loader, total=len(loader), leave=False):
        X, y = batch[0].to(device).float(), batch[1].to(device).float()
        optimizer.zero_grad(set_to_none=True)
        pred = model(X); loss = loss_fn(pred, y)
        loss.backward(); optimizer.step()
        total_loss += float(loss.item())
        sum_sq += torch.sum((pred.detach().view(-1) - y.detach().view(-1))**2).item()
        n += y.numel()
    if scheduler is not None: scheduler.step()
    return total_loss / max(len(loader), 1), math.sqrt(sum_sq / max(n, 1))

@torch.no_grad()
def eval_loop(loader, model, loss_fn, device):
    model.eval()
    total_loss, sum_sq, n = 0.0, 0.0, 0
    for batch in loader:
        X, y = batch[0].to(device).float(), batch[1].to(device).float()
        pred = model(X); total_loss += loss_fn(pred, y).item()
        sum_sq += torch.sum((pred.view(-1) - y.view(-1))**2).item()
        n += y.numel()
    return total_loss / max(len(loader), 1), math.sqrt(sum_sq / max(n, 1))

def write_submission_py(out_dir: Path):
    code = """\
import torch
from braindecode.models import EEGNeX

try:
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
except Exception:
    pass

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ; self.device = DEVICE

    def _make(self):
        m = EEGNeX(n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2*self.sfreq)).to(self.device)
        m.eval(); return m

    def get_model_challenge_1(self):
        m = self._make()
        m.load_state_dict(torch.load("/app/output/weights_challenge_1.pt", map_location=self.device), strict=True)
        return m

    def get_model_challenge_2(self):
        m = self._make()
        m.load_state_dict(torch.load("/app/output/weights_challenge_2.pt", map_location=self.device), strict=True)
        return m
"""
    (out_dir / "submission.py").write_text(code)

def build_zip(out_dir: Path):
    z = out_dir / "submission-to-upload.zip"
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ["submission.py", "weights_challenge_1.pt", "weights_challenge_2.pt"]:
            zf.write(out_dir / name, arcname=name)
    return z

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mini", action="store_true")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir) if args.data_dir else (repo_root / "data")
    out_dir  = Path(args.out_dir)  if args.out_dir  else (repo_root / "output")
    data_dir.mkdir(exist_ok=True, parents=True)
    out_dir.mkdir(exist_ok=True, parents=True)

    ds = EEGChallengeDataset(task="contrastChangeDetection", release="R5",
                             cache_dir=data_dir, mini=bool(args.mini))
    print(f"Loaded CCD (mini={bool(args.mini)}): {len(ds)} recordings")

    windows = make_windows_ccd(ds)
    tr, va, te = subject_split_concat(windows, seed=args.seed)
    tl, vl, te_l = build_loaders(tr, va, te, args.batch_size, args.num_workers)

    model = make_eegnex(n_chans=N_CHANS, sfreq=SFREQ, win_sec=WIN_SEC, n_outputs=1).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = CosineAnnealingLR(optim, T_max=max(args.epochs-1, 1))
    loss_fn = MSELoss()

    patience, min_delta = 50, 1e-4
    best_rmse, best_state, no_improve = float("inf"), None, 0

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_rmse = train_one_epoch(tl, model, loss_fn, optim, sched, device)
        va_loss, va_rmse = eval_loop(vl, model, loss_fn, device)
        print(f"[{epoch:03d}/{args.epochs}] train_loss={tr_loss:.6f} train_rmse={tr_rmse:.6f}  "
              f"val_loss={va_loss:.6f} val_rmse={va_rmse:.6f}")
        if va_rmse < best_rmse - min_delta:
            best_rmse, best_state, no_improve = va_rmse, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val RMSE={best_rmse:.6f})."); break

    if best_state is not None: model.load_state_dict(best_state)
    te_loss, te_rmse = eval_loop(te_l, model, loss_fn, device)
    print(f"TEST: loss={te_loss:.6f} rmse={te_rmse:.6f}")

    torch.save(model.state_dict(), out_dir / "weights_challenge_1.pt")
    torch.save(model.state_dict(), out_dir / "weights_challenge_2.pt")
    write_submission_py(out_dir)
    z = build_zip(out_dir)
    print(f"Built ZIP: {z}")
    print("Done.")

if __name__ == "__main__":
    main()
