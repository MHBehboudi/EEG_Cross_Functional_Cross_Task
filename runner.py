import copy, zipfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .loops import train_one_epoch, valid_model
from ..models.eegnex_model import build_eegnex
from ..data.ccd_windows import load_dataset_ccd, preprocess_offline, make_windows, subject_splits, SFREQ

def train_and_build_zip(*, outdir="output", mini=True, epochs=100, batch_size=128, num_workers=4, seed=2025):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    cache_dir = Path("data")
    ds = load_dataset_ccd(cache_dir, mini=mini)
    preprocess_offline(ds)
    windows = make_windows(ds)
    train_set, valid_set, test_set = subject_splits(windows, seed=seed)
    print(f"Splits  train={len(train_set)}  valid={len(valid_set)}  test={len(test_set)}")

    # Loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model
    model = build_eegnex(sfreq=SFREQ, n_chans=129, n_outputs=1, n_times=int(2 * SFREQ), device=device)

    # Optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 1)
    loss_fn = torch.nn.MSELoss()

    # Train (early stop)
    patience, min_delta = 50, 1e-4
    best_rmse, best_state, best_epoch, no_improve = float("inf"), None, None, 0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_rmse = train_one_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, device)
        va_loss, va_rmse = valid_model(valid_loader, model, loss_fn, device)
        print(f"[{epoch:03d}/{epochs}] train_loss={tr_loss:.6f} train_rmse={tr_rmse:.6f}  "
              f"val_loss={va_loss:.6f} val_rmse={va_rmse:.6f}")

        if va_rmse < best_rmse - min_delta:
            best_rmse, best_state, best_epoch, no_improve = va_rmse, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val RMSE={best_rmse:.6f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_rmse = valid_model(test_loader, model, loss_fn, device, print_batch_stats=False)
    print(f"TEST: loss={te_loss:.6f} rmse={te_rmse:.6f}")

    # --- Package submission (code-only repo; weights go to output, ignored by git) ---
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    w1 = outdir / "weights_challenge_1.pt"
    w2 = outdir / "weights_challenge_2.pt"
    torch.save(model.state_dict(), w1)
    torch.save(model.state_dict(), w2)
    print(f"Saved weights: {w1} / {w2}")

    sub_py = outdir / "submission.py"
    sub_py.write_text("""from braindecode.models import EEGNeX
import torch

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        model_challenge1 = EEGNeX(
            n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2 * self.sfreq)
        ).to(self.device)
        try:
            model_challenge1.load_state_dict(
                torch.load('/app/output/weights_challenge_1.pt', map_location=self.device)
            )
        except Exception:
            pass
        return model_challenge1

    def get_model_challenge_2(self):
        model_challenge2 = EEGNeX(
            n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2 * self.sfreq)
        ).to(self.device)
        try:
            model_challenge2.load_state_dict(
                torch.load('/app/output/weights_challenge_2.pt', map_location=self.device)
            )
        except Exception:
            pass
        return model_challenge2
""")

    zip_path = outdir / "submission-to-upload.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sub_py, arcname="submission.py")
        z.write(w1, arcname="weights_challenge_1.pt")
        z.write(w2, arcname="weights_challenge_2.pt")
    print(f"Built ZIP: {zip_path}")
    return zip_path
