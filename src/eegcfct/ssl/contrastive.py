# src/eegcfct/ssl/contrastive.py

from __future__ import annotations
import math
from typing import Iterable, Tuple, Callable, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# -----------------------------
# Utilities
# -----------------------------
def _take_X(batch) -> torch.Tensor:
    """Return the data tensor X from a dataloader batch, ignoring labels/indices."""
    if isinstance(batch, (tuple, list)):
        X = batch[0]
    else:
        X = batch
    if not torch.is_tensor(X):
        X = torch.as_tensor(X)
    # Expected shape (B, C, T) or (B, C, T, 1). If last dim is 1, squeeze it.
    if X.ndim == 4 and X.shape[-1] == 1:
        X = X[..., 0]
    return X.float()


def _random_time_mask(x: torch.Tensor, max_frac: float = 0.12) -> torch.Tensor:
    """Mask a random contiguous time segment with zeros."""
    # x: (..., T)
    T = x.shape[-1]
    if T < 4:
        return x
    L = int(np.random.uniform(0.0, max_frac) * T)
    if L <= 0:
        return x
    start = np.random.randint(0, max(T - L, 1))
    out = x.clone()
    out[..., start:start + L] = 0.0
    return out


def _jitter(x: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    """Add small Gaussian noise."""
    return x + sigma * torch.randn_like(x)


def _scale(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Random scaling per-sample."""
    # scale per sample (last 2 dims are C,T or 1,T if per-channel later)
    shape = [x.shape[0]] + [1] * (x.ndim - 1)
    s = torch.randn(shape, device=x.device) * sigma + 1.0
    return x * s


def _two_augs(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two simple time-series augmentations."""
    # Compose lightweight augs to keep it stable
    v1 = _jitter(_random_time_mask(_scale(x), max_frac=0.12), sigma=0.01)
    v2 = _jitter(_random_time_mask(_scale(x), max_frac=0.12), sigma=0.01)
    return v1, v2


# -----------------------------
# Channel-wise encoder
# -----------------------------
class TinyChannelSSL(nn.Module):
    """
    A tiny 1D CNN applied to single-channel sequences.
    Forward expects (N, 1, T). Returns L2-normalized projection vectors for contrastive learning.
    Use .encode() to get the backbone (pre-projection) features.
    """
    def __init__(self, emb_dim: int = 64, proj_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # (N, 64, 1)
        )
        self.feat = nn.Linear(64, emb_dim)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 1, T)
        returns normalized feature vectors (N, emb_dim)
        """
        h = self.backbone(x).squeeze(-1)  # (N, 64)
        h = self.feat(h)                  # (N, emb_dim)
        h = F.normalize(h, dim=-1)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 1, T)
        returns normalized projection vectors (N, proj_dim)
        """
        h = self.encode(x)
        z = self.proj(h)
        z = F.normalize(z, dim=-1)
        return z


# -----------------------------
# Two-view channel iterator
# -----------------------------
class TwoViewChannelIterator:
    """
    Wrap a DataLoader over windows (yielding (X, y, ...)), and produce
    two augmented channel-wise batches for contrastive learning.

    For each input window X (B, C, T), we create two views and then reshape to (B*C, 1, T).
    Optionally subsample channels to limit batch size via `max_pairs`.
    """
    def __init__(
        self,
        loader: Iterable,
        max_pairs: Optional[int] = None,
        device: Optional[torch.device] = None,
        aug_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]] = _two_augs,
    ):
        self.loader = loader
        self.max_pairs = max_pairs
        self.device = device
        self.aug_fn = aug_fn

    def __iter__(self):
        while True:
            for batch in self.loader:
                X = _take_X(batch)  # (B, C, T)
                if self.device is not None:
                    X = X.to(self.device, non_blocking=True)

                # Two augmentations at the window level (B, C, T)
                v1, v2 = self.aug_fn(X)

                # Reshape into per-channel samples (B*C, 1, T)
                B, C, T = v1.shape
                v1 = v1.transpose(0, 1).contiguous().view(C * B, 1, T)  # (C*B, 1, T)
                v2 = v2.transpose(0, 1).contiguous().view(C * B, 1, T)

                # Optionally subsample to keep batch size reasonable
                if self.max_pairs is not None and self.max_pairs < v1.shape[0]:
                    idx = torch.randperm(v1.shape[0], device=v1.device)[: self.max_pairs]
                    v1 = v1.index_select(0, idx)
                    v2 = v2.index_select(0, idx)

                yield v1, v2


# -----------------------------
# Contrastive loss
# -----------------------------
def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.2) -> torch.Tensor:
    """
    NT-Xent (SimCLR) loss.
    z1, z2: (N, D) normalized projections.
    """
    N, D = z1.shape
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim = torch.matmul(z, z.T) / temp  # (2N, 2N)

    # mask out self similarities
    mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -9e15)

    # positives are off-diagonal pairs (i <-> i+N and i+N <-> i)
    targets = torch.arange(N, device=z.device)
    # logits for first N rows: positives at indices N + i
    logits_1 = sim[:N, N:]
    # logits for last N rows: positives at indices i
    logits_2 = sim[N:, :N]

    loss_1 = F.cross_entropy(logits_1, targets)
    loss_2 = F.cross_entropy(logits_2, targets)
    return 0.5 * (loss_1 + loss_2)


# -----------------------------
# Public API: train SSL + build projection
# -----------------------------
def train_ssl_encoder(
    train_loader,
    device: torch.device,
    ssl_epochs: int = 5,
    ssl_steps: int = 200,
    temp: float = 0.2,
    samples_per_ch: int = 128,
    lr: float = 1e-3,
    log: Callable[[str], None] = print,
) -> TinyChannelSSL:
    """
    Self-supervised pretraining of a channel-wise encoder.
    Returns a trained TinyChannelSSL model.
    """
    model = TinyChannelSSL(emb_dim=64, proj_dim=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # max_pairs caps the per-step batch size ~ samples_per_ch * (#channels ~=129) but we cannot
    # guarantee exact; we just use this as a soft upper-bound for (B*C).
    max_pairs = samples_per_ch * 8  # modest default; adjust if you want larger batches
    iterator = TwoViewChannelIterator(train_loader, max_pairs=max_pairs, device=device)

    step_iter = iter(iterator)
    for ep in range(1, ssl_epochs + 1):
        model.train()
        losses = []
        for _ in range(ssl_steps):
            v1, v2 = next(step_iter)  # each is (N, 1, T)
            z1 = model(v1)
            z2 = model(v2)
            loss = nt_xent(z1, z2, temp=temp)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))
        log(f"[SSL {ep:02d}/{ssl_epochs}] contrastive_loss={np.mean(losses):.4f}")

    return model


@torch.no_grad()
def build_channel_projection_from_ssl(
    encoder: TinyChannelSSL,
    loader,
    device: torch.device,
    n_clusters: int = 20,
    pcs_per_cluster: int = 3,
    cov_batches: int = 25,
    log: Callable[[str], None] = print,
) -> np.ndarray:
    """
    Use the trained encoder to:
      1) compute mean embedding per channel (C x D),
      2) cluster channels (KMeans),
      3) compute per-cluster PCA over *channel signals* to build a linear projection W (C x (K*pcs)).

    Returns: W as numpy float32 (shape C x out_dim).
    """
    encoder.eval()

    # ---- 1) Aggregate channel embeddings (C x D)
    ch_sum: Optional[torch.Tensor] = None
    ch_cnt: Optional[torch.Tensor] = None
    batches_seen = 0

    for batch in loader:
        X = _take_X(batch).to(device)  # (B, C, T)
        B, C, T = X.shape
        # Build per-channel samples
        Xch = X.transpose(0, 1).contiguous().view(C * B, 1, T)  # (C*B, 1, T)
        H = encoder.encode(Xch)  # (C*B, D)
        D = H.shape[-1]

        # reshape back to (C, B, D) then mean over B
        Hb = H.view(C, B, D).sum(dim=1)  # (C, D)
        if ch_sum is None:
            ch_sum = Hb.clone()
            ch_cnt = torch.full((C, 1), B, device=device, dtype=Hb.dtype)
        else:
            ch_sum += Hb
            ch_cnt += B
        batches_seen += 1
        if batches_seen >= max(5, cov_batches // 2):
            break

    if ch_sum is None:
        raise RuntimeError("No data seen to compute channel embeddings.")

    ch_emb = (ch_sum / ch_cnt).detach().cpu().numpy()  # (C, D)
    C, D = ch_emb.shape

    # ---- 2) Cluster channels in embedding space
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=2025)
    labels = km.fit_predict(ch_emb)

    # ---- 3) Per-cluster PCA over channel signals to make a *channel projection* W
    # Accumulate covariance within each cluster: cov_k = sum_over_batches (Xk @ Xk^T)
    # where Xk is (Ck, B*T) on each batch.
    covs: List[np.ndarray] = []
    ch_index_lists: List[np.ndarray] = []
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        ch_index_lists.append(idx)
        covs.append(np.zeros((len(idx), len(idx)), dtype=np.float64))  # high precision for stability

    batches_seen = 0
    for batch in loader:
        X = _take_X(batch)  # (B, C, T)
        B, C2, T = X.shape
        X_np = X.detach().cpu().numpy()  # keep it simple and RAM-friendly for mini
        assert C2 == C, "Channel count mismatch during covariance accumulation."

        for k, idx in enumerate(ch_index_lists):
            if idx.size == 0:
                continue
            Xk = X_np[:, idx, :]                      # (B, Ck, T)
            Xk = np.transpose(Xk, (1, 0, 2)).reshape(idx.size, -1)  # (Ck, B*T)
            covs[k] += Xk @ Xk.T

        batches_seen += 1
        if batches_seen >= cov_batches:
            break

    # Build W: (C x out_dim), block by block
    blocks: List[np.ndarray] = []
    row_ptr = 0
    W = np.zeros((C, n_clusters * pcs_per_cluster), dtype=np.float32)  # upper bound, we'll prune unused cols
    col_ptr = 0
    used_cols = 0

    for k, idx in enumerate(ch_index_lists):
        Ck = idx.size
        if Ck == 0:
            continue
        # Eigen-decomposition of covariance
        cov = covs[k] / max(1.0, cov_batches)
        # Numerical guard
        cov = 0.5 * (cov + cov.T)
        try:
            vals, vecs = np.linalg.eigh(cov)  # vecs: (Ck, Ck)
        except np.linalg.LinAlgError:
            # very degenerate cluster: fall back to identity
            vecs = np.eye(Ck, dtype=np.float64)

        # take top PCs
        order = np.argsort(vals)[::-1]
        p = min(pcs_per_cluster, Ck)
        U = vecs[:, order[:p]].astype(np.float32)  # (Ck, p)

        # place into W rows at indices 'idx'
        W[idx[:, None], col_ptr:col_ptr + p] = U
        col_ptr += p
        used_cols += p

    # trim unused cols if any clusters had < pcs_per_cluster channels
    W = W[:, :used_cols].astype(np.float32)

    log(f"[SSL] Built projection W with shape {W.shape} from {n_clusters} clusters, {pcs_per_cluster} PCs/cluster.")
    return W
