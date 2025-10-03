# src/eegcfct/ssl/contrastive.py
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# --------------------------
# Utils & augmentations
# --------------------------
def _get_X(batch):
    # Braindecode windows can be (X, y) or (X, y, i, ...)
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch

def aug_noise(x, sigma=0.02):
    return x + sigma * torch.randn_like(x)

def aug_time_mask(x, max_frac=0.1):
    B, C, T = x.shape
    L = max(1, int(T * max_frac * random.random()))
    s = random.randint(0, max(0, T - L))
    x = x.clone()
    x[:, :, s:s+L] = 0
    return x

def aug_channel_dropout(x, p=0.1):
    B, C, T = x.shape
    mask = (torch.rand(B, C, 1, device=x.device) > p).float()
    return x * mask

def augment(x):
    ops = [aug_noise, aug_time_mask, aug_channel_dropout]
    # apply two random augs
    for _ in range(2):
        x = random.choice(ops)(x)
    return x


# --------------------------
# Two-view sampler
# --------------------------
class TwoAugmentSampler:
    """Yield (v1, v2) augmented views for contrastive SSL."""
    def __init__(self, dataloader, steps_per_epoch: int):
        self.loader = dataloader
        self.steps = steps_per_epoch

    def __iter__(self):
        it = iter(self.loader)
        for _ in range(self.steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.loader)
                batch = next(it)
            X = _get_X(batch).float()
            yield augment(X.clone()), augment(X.clone())


# --------------------------
# Very small channel-wise encoder
# --------------------------
class ChanEncoder(nn.Module):
    """Depthwise temporal convs per channel + per-channel 1x1 to D, GAP over time."""
    def __init__(self, in_ch: int = 129, emb_dim: int = 64):
        super().__init__()
        self.in_ch = in_ch
        self.emb_dim = emb_dim
        self.conv1 = nn.Conv1d(in_ch, in_ch, kernel_size=15, padding=7, groups=in_ch)
        self.bn1   = nn.BatchNorm1d(in_ch)
        self.conv2 = nn.Conv1d(in_ch, in_ch, kernel_size=7,  padding=3, groups=in_ch)
        self.bn2   = nn.BatchNorm1d(in_ch)
        # per-channel projection to D
        self.proj  = nn.Conv1d(in_ch, in_ch * emb_dim, kernel_size=1, groups=in_ch)

    def forward(self, x):           # x: (B, C, T)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.proj(x)            # (B, C*D, T)
        B, CD, T = x.shape
        D, C = self.emb_dim, self.in_ch
        x = x.view(B, C, D, T).mean(-1)   # (B, C, D)
        x = F.normalize(x, dim=-1)
        return x


# --------------------------
# NT-Xent contrastive loss
# --------------------------
def nt_xent(z1, z2, tau=0.1):
    """z1,z2: (N, D), L2-normalized."""
    N, D = z1.shape
    reps = torch.cat([z1, z2], dim=0)      # (2N, D)
    sim = (reps @ reps.t()) / tau          # (2N, 2N)
    eye = torch.eye(2*N, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(eye, 0.0)
    # positives are (i, i+N) and (i+N, i)
    pos = torch.exp((z1 * z2).sum(-1) / tau)  # (N,)
    denom_i = torch.exp(sim[:N]).sum(dim=1)
    denom_j = torch.exp(sim[N:]).sum(dim=1)
    loss = (-torch.log(pos / denom_i) - torch.log(pos / denom_j)).mean() * 0.5
    return loss


# --------------------------
# Train SSL encoder
# --------------------------
def train_ssl_encoder(
    train_loader,
    ssl_epochs: int = 5,
    ssl_steps_per_epoch: int = 150,
    lr: float = 1e-3,
    wd: float = 1e-5,
    tau: float = 0.1,
    device: torch.device = torch.device("cpu"),
    in_ch: int = 129,
    emb_dim: int = 64,
):
    enc = ChanEncoder(in_ch=i_
