from __future__ import annotations
import math
from typing import Optional, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ---------------------------
# Small per-channel encoder
# ---------------------------
class ChanEncoder(nn.Module):
    """Encode each channel's 2s window into an embedding vector.

    Forward expects X of shape (B, C, T) and returns (B, C, D).
    We reshape to (B*C, 1, T) and apply a tiny temporal ConvNet.
    """
    def __init__(self, in_ch: int = 129, emb_dim: int = 64):
        super().__init__()
        self.in_ch = in_ch
        self.emb_dim = emb_dim

        self.fe = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(64, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x = x.reshape(B * C, 1, T)
        h = self.fe(x)
        h = self.gap(h).squeeze(-1)        # (B*C, 64)
        z = self.proj(h)                   # (B*C, D)
        z = z.view(B, C, self.emb_dim)     # (B, C, D)
        return z


# ---------------------------
# Contrastive utilities
# ---------------------------
def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """InfoNCE across (N) positives; z1,z2: (N,D) normalized."""
    N, D = z1.shape
    z = torch.cat([z1, z2], dim=0)                    # (2N,D)
    sim = torch.matmul(z, z.t())                      # (2N,2N)
    # mask out self
    mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim = sim / tau
    sim = sim.masked_fill(mask, float("-inf"))

    # positives: (i,N+i) and (N+i,i)
    pos = torch.arange(N, device=z.device)
    labels = torch.cat([pos + N, pos], dim=0)
    loss = F.cross_entropy(sim, labels)
    return loss


def _rand_time_crop(x: torch.Tensor, min_keep: float = 0.8) -> torch.Tensor:
    """Random temporal crop on last dim. x: (B,C,T)"""
    B, C, T = x.shape
    keep = torch.randint(int(min_keep * T), T + 1, (1,), device=x.device).item()
    start = torch.randint(0, T - keep + 1, (1,), device=x.device).item()
    out = x[..., start:start + keep]
    # pad back to T (right-pad zeros)
    if out.shape[-1] < T:
        pad = T - out.shape[-1]
        out = F.pad(out, (0, pad))
    return out


def _add_noise(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    return x + sigma * torch.randn_like(x)


def _chan_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    if p <= 0:
        return x
    B, C, T = x.shape
    mask = (torch.rand(B, C, 1, device=x.device) > p).float()
    return x * mask


def _augment(x: torch.Tensor) -> torch.Tensor:
    # Compose a couple of simple, fast ops
    x = _rand_time_crop(x, min_keep=0.8)
    x = _add_noise(x, sigma=0.02)
    x = _chan_dropout(x, p=0.1)
    return x


class TwoAugmentSampler:
    """Yield (view1, view2, ch_idx) for contrastive learning.

    To prevent OOM, we subselect `channels_per_step` channels each step.
    """
    def __init__(self, loader, steps_per_epoch: int = 150, channels_per_step: int = 32):
        self.loader = loader
        self.steps = steps_per_epoch
        self.channels_per_step = channels_per_step

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        it = iter(self.loader)
        for _ in range(self.steps):
            batch = next(it, None)
            if batch is None:
                it = iter(self.loader)
                batch = next(it)
            X = batch[0]  # (B,C,T)
            B, C, T = X.shape
            if self.channels_per_step < C:
                # choose a new random subset each step
                ch_idx = torch.randperm(C)[: self.channels_per_step]
                X = X[:, ch_idx, :]
            else:
                ch_idx = torch.arange(C)
            v1 = _augment(X)
            v2 = _augment(X)
            yield v1, v2, ch_idx


# ---------------------------
# SSL training + projection
# ---------------------------
@torch.no_grad()
def compute_channel_embeddings(
    encoder: ChanEncoder,
    loader,
    n_batches: int = 20,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Average per-channel embeddings across a few batches → (C, D) numpy."""
    encoder.eval()
    embs = None
    n_seen = 0
    it = iter(loader)
    for _ in range(n_batches):
        batch = next(it, None)
        if batch is None:
            it = iter(loader)
            batch = next(it)
        X = batch[0].to(device, non_blocking=True)  # (B,C,T)
        Z = encoder(X)  # (B,C,D)
        Z = Z.mean(dim=0)  # (C,D) batch-avg
        if embs is None:
            embs = Z
        else:
            embs += Z
        n_seen += 1
    embs = (embs / max(n_seen, 1)).detach().cpu().numpy()  # (C,D)
    return embs


def train_ssl_encoder(
    train_loader,
    ssl_epochs: int = 5,
    ssl_steps_per_epoch: int = 150,
    lr: float = 1e-3,
    wd: float = 1e-5,
    tau: float = 0.1,
    device: torch.device = torch.device("cpu"),
    in_ch: int = 129,
    emb_dim: int = 48,
    windows_ds=None,  # accepted for compatibility with runner; not used
    channels_per_step: int = 32,
    use_amp: bool = True,
) -> ChanEncoder:
    enc = ChanEncoder(in_ch=in_ch, emb_dim=emb_dim).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=wd)
    sampler = TwoAugmentSampler(
        train_loader, steps_per_epoch=ssl_steps_per_epoch, channels_per_step=channels_per_step
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    enc.train()
    for ep in range(1, ssl_epochs + 1):
        tot, n = 0.0, 0
        for v1, v2, _ch_idx in sampler:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                z1 = enc(v1).reshape(-1, enc.emb_dim)  # (B*channels_per_step, D)
                z2 = enc(v2).reshape(-1, enc.emb_dim)
                z1 = F.normalize(z1, dim=-1)
                z2 = F.normalize(z2, dim=-1)
                loss = nt_xent(z1, z2, tau=tau)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tot += float(loss.item()); n += 1
        print(f"[SSL {ep:02d}/{ssl_epochs}] contrastive_loss={tot/max(n,1):.4f}")

    enc.eval()
    return enc


@torch.no_grad()
def build_channel_projection_from_ssl(
    encoder: ChanEncoder,
    train_loader,
    k_clusters: int = 20,
    pcs_per_cluster: int = 3,
    n_win_for_pca: int = 50,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Return W of shape (C, K*P). For each cluster, PCA over *signals* of its channels
    pooled across ~n_win_for_pca windows, then put top P loadings into W."""
    # 1) Embed channels & cluster them
    C = train_loader.dataset[0][0].shape[0]  # infer channels from one sample
    embs = compute_channel_embeddings(encoder, train_loader, n_batches=10, device=device)  # (C,D)
    kmeans = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embs)  # (C,)

    # 2) Collect a small pool of windows to compute PCA bases
    it = iter(train_loader)
    pool = []
    total = 0
    while total < n_win_for_pca:
        batch = next(it, None)
        if batch is None:
            it = iter(train_loader)
            batch = next(it)
        X = batch[0]  # (B,C,T)
        pool.append(X.detach().cpu())
        total += X.shape[0]
    X_pool = torch.cat(pool, dim=0)  # (N,C,T)
    N, C_check, T = X_pool.shape
    assert C_check == C

    # Flatten time -> samples x channels
    SxC = X_pool.permute(0, 2, 1).reshape(N * T, C)  # (S, C)
    SxC = SxC.numpy()

    # 3) Build W by cluster
    out_dim = k_clusters * pcs_per_cluster
    W = np.zeros((C, out_dim), dtype=np.float32)
    col_ptr = 0
    for k in range(k_clusters):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            col_ptr += pcs_per_cluster
            continue

        Xc = SxC[:, idx]  # (S, |idx|)
        # z-score across samples for stability
        Xc = (Xc - Xc.mean(axis=0, keepdims=True)) / (Xc.std(axis=0, keepdims=True) + 1e-6)

        p_eff = min(pcs_per_cluster, idx.size)
        pca = PCA(n_components=p_eff, svd_solver="auto", random_state=42)
        pca.fit(Xc)
        U = pca.components_.T.astype(np.float32)  # (|idx|, p_eff)

        W[idx, col_ptr:col_ptr + p_eff] = U
        col_ptr += pcs_per_cluster

    return W  # (C, K*P)
