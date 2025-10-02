import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
from torch.optim.lr_scheduler import LRScheduler

def train_one_epoch(dataloader: DataLoader, model, loss_fn, optimizer,
                    scheduler: Optional[LRScheduler], epoch: int, device,
                    print_batch_stats: bool = True):
    model.train()
    total_loss = 0.0
    sum_sq_err, n_samples = 0.0, 0
    bar = tqdm(enumerate(dataloader), total=len(dataloader), disable=not print_batch_stats)
    for b, batch in bar:
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds_flat, y_flat = preds.detach().view(-1), y.detach().view(-1)
        sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
        n_samples += y_flat.numel()
        if print_batch_stats:
            rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
            bar.set_description(f"[{epoch:03d}] loss={loss.item():.6f} rmse={rmse:.6f}")
    if scheduler is not None:
        scheduler.step()
    avg_loss = total_loss / len(dataloader)
    rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
    return avg_loss, rmse

@torch.no_grad()
def valid_model(dataloader: DataLoader, model, loss_fn, device, print_batch_stats: bool = True):
    model.eval()
    total_loss = 0.0
    sum_sq_err, n_samples = 0.0, 0
    bar = tqdm(enumerate(dataloader), total=len(dataloader), disable=not print_batch_stats)
    for b, batch in bar:
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()
        preds = model(X)
        l = loss_fn(preds, y).item()
        total_loss += l
        preds_flat, y_flat = preds.detach().view(-1), y.detach().view(-1)
        sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
        n_samples += y_flat.numel()
        if print_batch_stats:
            rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
            bar.set_description(f"[val] loss={l:.6f} rmse={rmse:.6f}")
    avg_loss = total_loss / len(dataloader)
    rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
    return avg_loss, rmse
