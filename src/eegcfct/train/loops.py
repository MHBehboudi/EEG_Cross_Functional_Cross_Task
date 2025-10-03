# src/eegcfct/train/loops.py
import math
import torch
from torch.utils.data import DataLoader

def build_loaders(train_set, valid_set, test_set, batch_size, num_workers):
    pin = torch.cuda.is_available()
    tr = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    va = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    te = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return tr, va, te

def _rmse_from_batches(loss_sum_sq_err, n_samples):
    return math.sqrt(loss_sum_sq_err / max(n_samples, 1))

def train_one_epoch(dataloader, model, loss_fn, optimizer, scheduler, epoch, device):
    model.train()
    total_loss = 0.0
    sum_sq_err = 0.0
    n_samples = 0
    for batch in dataloader:
        X, y = batch[0].to(device).float(), batch[1].to(device).float()
        optimizer.zero_grad(set_to_none=True)
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        sum_sq_err += torch.sum((preds.detach().view(-1) - y.detach().view(-1)) ** 2).item()
        n_samples += y.numel()
    if scheduler is not None:
        scheduler.step()
    avg_loss = total_loss / max(len(dataloader), 1)
    rmse = _rmse_from_batches(sum_sq_err, n_samples)
    return avg_loss, rmse

@torch.no_grad()
def eval_loop(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0.0
    sum_sq_err = 0.0
    n_samples = 0
    for batch in dataloader:
        X, y = batch[0].to(device).float(), batch[1].to(device).float()
        preds = model(X)
        loss = loss_fn(preds, y).item()
        total_loss += loss
        sum_sq_err += torch.sum((preds.view(-1) - y.view(-1)) ** 2).item()
        n_samples += y.numel()
    avg_loss = total_loss / max(len(dataloader), 1)
    rmse = _rmse_from_batches(sum_sq_err, n_samples)
    return avg_loss, rmse
