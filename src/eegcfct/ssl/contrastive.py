# src/eegcfct/ssl/contrastive.py
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- tiny 1D encoder (raw channel -> embedding) ----------
class Tiny1DCNNEncoder(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, emb_dim),
        )

    def forward(self, x):  # x: [B,1,T]
        h = self.fe(x)
        z = self.head(h)
        z = F.normalize(z, dim=-1)
        return z  # [B, D]

# ---------- simple time-series augmentations ----------
@dataclass
class AugParams:
    jitter_std: float = 0.02
    time_mask_ratio: float = 0.06   # fraction of T to zero
    amp_scale_low: float = 0.9
    amp_scale_high: float = 1.1

def augment_ts(x: torch.Tensor, params: AugParams) -> torch.Tensor:
    # x: [B,1,T]
    B, _, T = x.shape
    out = x.clone()

    # amplitude jitter
    out = out + torch.randn_like(out) * params.jitter_std

    # random amplitude scaling
    scale = torch.empty(B, 1, 1, device=x.device).uniform_(params.amp_scale_low, params.amp_scale_high)
    out = out * scale

    # time mask
    L = int(T * params.time_mask_ratio)
    if L > 0:
        starts = torch.randint(low=0, high=max(1, T - L), size=(B,), device=x.device)
        for i in range(B):
            out[i, :, starts[i]:starts[i]+L] = 0.0
    return out

# ---------- NT-Xent (SimCLR) ----------
def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    # z1, z2: [B, D], L2-normalized
    B, D = z1.shape
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.mm(z, z.t()) / temperature  # cosine sim since normalized

    # mask self-sim
    mask = torch.eye(2*B, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)

    # positives: (i, i+B) and (i+B, i)
    targets = torch.arange(B, device=z.device)
    loss_i = F.cross_entropy(sim[:B], targets + B)
    loss_j = F.cross_entropy(sim[B:], targets)
    return (loss_i + loss_j) * 0.5

# ---------- SSL loader (on-the-fly channel sampling) ----------
class ChannelWindowSampler(torch.utils.data.IterableDataset):
    """
    Iterates over (two augmented views) from random (window, channel) pairs
    pulled from a braindecode WindowsDataset/DataLoader batch.
    """
    def __init__(self, loader, n_steps_per_epoch: int, device: torch.device, aug_params: AugParams):
        super().__init__()
        self.loader = loader
        self.n_steps = n_steps_per_epoch
        self.device = device
        self.aug_params = aug_params

    def __iter__(self):
        it = iter(self.loader)
        for _ in range(self.n_steps):
            try:
                X, y = next(it)
            except StopIteration:
                it = iter(self.loader)
                X, y = next(it)
            # X: [B, C, T]
            X = X.to(self.device, non_blocking=True).float()
            B, C, T = X.shape

            # pick one random channel per sample
            idx = torch.randint(low=0, high=C, size=(B,), device=self.device)
            seg = torch.stack([X[i, idx[i]] for i in range(B)], dim=0).unsqueeze(1)  # [B,1,T]

            v1 = augment_ts(seg, self.aug_params)
            v2 = augment_ts(seg, self.aug_params)
            yield v1, v2

# ---------- train SSL encoder ----------
@torch.no_grad()
def _peek_time_length(loader) -> int:
    for X, _ in loader:
        return X.shape[-1]
    return 200

def train_ssl_encoder(train_loader, device: torch.device, epochs: int = 5,
                      steps_per_epoch: int = 200, batch_size: int = 128,
                      emb_dim: int = 128, temperature: float = 0.2,
                      lr: float = 1e-3, seed: int = 2025) -> Tiny1DCNNEncoder:
    torch.manual_seed(seed); random.seed(seed)

    enc = Tiny1DCNNEncoder(emb_dim=emb_dim).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr)
    sampler = ChannelWindowSampler(train_loader, steps_per_epoch, device, AugParams())

    for ep in range(1, epochs+1):
        enc.train()
        total = 0.0
        for v1, v2 in sampler:
            z1 = enc(v1)
            z2 = enc(v2)
            loss = nt_xent(z1, z2, temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"[SSL] epoch {ep}/{epochs} loss={total/max(steps_per_epoch,1):.4f}")
    return enc

# ---------- channel embeddings ----------
@torch.no_grad()
def compute_channel_embeddings(encoder: Tiny1DCNNEncoder, train_loader, device: torch.device,
                               n_channels: int, samples_per_channel: int = 256) -> torch.Tensor:
    encoder.eval()
    # collect per-channel embeddings by averaging across random windows
    sums = torch.zeros(n_channels, encoder.head[-1].out_features, device=device)
    counts = torch.zeros(n_channels, device=device)

    it = iter(train_loader)
    T = _peek_time_length(train_loader)
    need = n_channels * samples_per_channel
    have = 0

    while have < need:
        try:
            X, _ = next(it)
        except StopIteration:
            it = iter(train_loader)
            X, _ = next(it)
        X = X.to(device).float()  # [B, C, T]
        B, C, T = X.shape
        # sample many (channel, sample) pairs from this batch
        m = min((need - have), B * 4)
        ch_idx = torch.randint(low=0, high=C, size=(m,), device=device)
        samp_idx = torch.randint(low=0, high=B, size=(m,), device=device)
        seg = X[samp_idx, ch_idx].unsqueeze(1)  # [m,1,T]
        z = encoder(seg)  # [m,D]
        for i in range(m):
            c = int(ch_idx[i].item())
            sums[c] += z[i]
            counts[c] += 1
        have += m
    emb = sums / counts.clamp_min(1.0).unsqueeze(-1)
    emb = F.normalize(emb, dim=-1)
    return emb  # [C, D]

# ---------- k-means (torch, simple) ----------
@torch.no_grad()
def kmeans(X: torch.Tensor, K: int, iters: int = 50, seed: int = 2025) -> Tuple[torch.Tensor, torch.Tensor]:
    # X: [N, D]
    g = torch.Generator(device=X.device).manual_seed(seed)
    N, D = X.shape
    perm = torch.randperm(N, generator=g, device=X.device)
    cent = X[perm[:K]].clone()
    for _ in range(iters):
        # assign
        d2 = torch.cdist(X, cent, p=2.0)  # [N,K]
        assign = torch.argmin(d2, dim=1)  # [N]
        # update
        for k in range(K):
            mask = (assign == k)
            if mask.any():
                cent[k] = X[mask].mean(dim=0)
    return assign, cent  # [N], [K,D]

# ---------- cluster-wise PCA to build projection ----------
@torch.no_grad()
def _accum_cov(train_loader, device: torch.device, ch_idx: List[int], max_windows: int = 256) -> torch.Tensor:
    # returns covariance [n_ch,n_ch] accumulated across windows
    n_ch = len(ch_idx)
    cov = torch.zeros(n_ch, n_ch, device=device)
    count = 0
    it = iter(train_loader)
    while count < max_windows:
        try: X,_ = next(it)
        except StopIteration:
            it = iter(train_loader); X,_ = next(it)
        X = X.to(device).float()  # [B,C,T]
        B, C, T = X.shape
        m = min(B, max_windows - count)
        Xb = X[:m, ch_idx]        # [m, n_ch, T]
        Xb = Xb - Xb.mean(dim=-1, keepdim=True)
        # cov over time then average across batch
        cov += torch.einsum('bct,bC t->cC', Xb, Xb) / float(T)  # (bct)*(bCt)
        count += m
    cov /= max(1, count)
    return cov

@torch.no_grad()
def build_projection_matrix(train_loader, device: torch.device,
                            n_channels: int, assign: torch.Tensor,
                            pcs_per_cluster: int = 3, max_windows: int = 256) -> torch.Tensor:
    """
    Returns P: [proj_dim, n_channels] mapping orig->projected via y = P @ x.
    """
    K = int(assign.max().item()) + 1
    rows = []
    for k in range(K):
        idx = torch.where(assign == k)[0].tolist()
        if len(idx) == 0:  # empty (rare)
            continue
        cov = _accum_cov(train_loader, device, idx, max_windows=max_windows)  # [n_k,n_k]
        # eigh gives ascending eigenvals -> take last pcs
        evals, evecs = torch.linalg.eigh(cov)  # evecs: [n_k,n_k] cols = eigenvectors
        comps = evecs[:, -pcs_per_cluster:]    # [n_k, Pk]
        comps = comps.t().contiguous()         # [Pk, n_k]
        # build block row for full 129
        row = torch.zeros(comps.shape[0], n_channels, device=device)
        row[:, idx] = comps
        rows.append(row)
    if not rows:
        # fallback to identity
        P = torch.eye(n_channels, device=device)
    else:
        P = torch.cat(rows, dim=0)  # [proj_dim, n_channels]
    # L2 normalize rows for stability
    P = F.normalize(P, dim=1)
    return P
