# src/eegcfct/train/loops.py
import math
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

def build_loaders(train_set, valid_set, test_set, batch_size, num_workers):
    pin = torch.cuda.is_available()
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_set, shuffle=True,  **common)
    valid_loader = DataLoader(valid_set, shuffle=False, **common)
    test_loader  = DataLoader(test_set,  shuffle=False, **common)
    return train_loader, valid_loader, test_loader

def _move(batch, device):
    # braindecode WindowsDataset returns (X, y, i) or (X, y)
    X, y = batch[0], batch[1]
    return X.to(device, non_blocking=True).float(), y.to(device, non_blocking=True).float()

def train_one_epoch(
    dataloader, model, loss_fn, optimizer, scheduler, epoch, device,
    amp_dtype=None, grad_clip=None
):
    model.train()
    use_amp = (device.type == "cuda" and amp_dtype is not None)
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    total_loss = 0.0
    sse = 0.0
    n = 0

    for batch in dataloader:
        X, y = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(dtype=amp_dtype):
                preds = model(X)
                loss = loss_fn(preds, y)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        sse += torch.sum((preds.detach().view(-1) - y.detach().view(-1)) ** 2).double().cpu().item()
        n += y.numel()

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / max(len(dataloader), 1)
    rmse = math.sqrt(sse / max(n, 1))
    return avg_loss, rmse

@torch.no_grad()
def eval_loop(dataloader, model, loss_fn, device, amp_dtype=None):
    model.eval()
    use_amp = (device.type == "cuda" and amp_dtype is not None)
    total_loss = 0.0
    sse = 0.0
    n = 0

    for batch in dataloader:
        X, y = _move(batch, device)
        if use_amp:
            with autocast(dtype=amp_dtype):
                preds = model(X)
                loss = loss_fn(preds, y)
        else:
            preds = model(X)
            loss = loss_fn(preds, y)
        total_loss += float(loss.detach().cpu())
        sse += torch.sum((preds.detach().view(-1) - y.detach().view(-1)) ** 2).double().cpu().item()
        n += y.numel()

    avg_loss = total_loss / max(len(dataloader), 1)
    rmse = math.sqrt(sse / max(n, 1))
    return avg_loss, rmse
