# src/eegcfct/ssl/contrastive.py
# -----------------------------------------------------------------------------
# Lightweight self-supervised channel encoder + clustering projector builder
# Works with Braindecode Window datasets that yield (X, y) or (X, y, ...)
# -----------------------------------------------------------------------------

from __future__ import annotations
import math
from typing import Iterable, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def _get_X(batch):
    """Return the input tensor X from a windows batch (X, y, *)."""
    # braindecode windows often return (X, y, i) or (X, y)
    return batch[0]


def _default_loader(dataset, batch_size: int, num_workers: int, shuffle: bool):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )


# -------------------------------------------------------------------------
# Augmentations for contrastive SSL (kept simple & fast)
# -------------------------------------------------------------------------
@torch.no_grad()
def _aug_time_mask(x: torch.Tensor, p: float = 0.2, max_frac: float = 0.1):
    """Random contiguous time masking per channel."""
    # x: (B, C, T)
    if p <= 0:
        return x
    B, C, T = x.shape
    out = x.clone()
    mask_len = max(1, int(T * max_frac))
    do = torch.rand(B, C, device=x.device) < p
    for b in range(B):
        for c in range(C):
            if do[b, c]:
                start = torch.randint(0, max(T - mask_len + 1, 1), (1,), device=x.device).item()
                out[b, c, start:start + mask_len] = 0.0
    return out


@torch.no_grad()
def _aug_jitter(x: torch.Tensor, sigma: float = 0.01):
    if sigma <= 0:
        return x
    return x + sigma * torch.randn_like(x)


@torch.no_grad()
def _aug_drop_channels(x: torch.Tensor, p: float = 0.05):
    if p <= 0:
        return x
    B, C, T = x.shape
    m = (torch.rand(B, C, device=x.device) >= p).float().unsqueeze(-1)
    return x * m


@torch.no_grad()
def make_two_views(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create two stochastically augmented views of x (B, C, T)."""
    v1 = _aug_time_mask(_aug_jitter(_aug_drop_channels(x, 0.05), 0.01), p=0.2, max_frac=0.1)
    v2 = _aug_time_mask(_aug_jitter(_aug_drop_channels(x, 0.05), 0.01), p=0.2, max_frac=0.1)
    return v1, v2


# -------------------------------------------------------------------------
# Contrastive encoder: channelwise temporal encoder + MLP head
# -------------------------------------------------------------------------
class ContrastiveChannelEncoder(nn.Module):
    """
    Encodes each channel's T-length signal into a D-dim embedding.
    Implementation: apply the same temporal stack to every channel by
    reshaping (B, C, T) -> (B*C, 1, T), then pool to 32 feats -> Linear->D.
    """
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=7, padding="same"),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # -> (B*C, 32, 1)
            nn.Flatten(),             # -> (B*C, 32)
            nn.Linear(32, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor, proj: bool = True) -> torch.Tensor:
        """
        x: (B, C, T) -> returns (B, C, D) normalized embeddings
        """
        B, C, T = x.shape
        h = self.temporal(x.reshape(B * C, 1, T))  # (B*C, D)
        h = h.reshape(B, C, -1)
        if proj:
            z = self.proj(h)
        else:
            z = h
        # normalize last dim
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        return z


# -------------------------------------------------------------------------
# Pair sampler that yields two augmented views per batch
# -------------------------------------------------------------------------
class DualViewChannelSampler:
    def __init__(self, dataset, batch_size: int, num_workers: int, device: torch.device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.loader = _default_loader(dataset, batch_size, num_workers, shuffle=True)

    def __iter__(self):
        it = iter(self.loader)
        while True:
            try:
                batch = next(it)
            except StopIteration:
                return
            X = _get_X(batch).float().to(self.device)  # (B, C, T)
            v1, v2 = make_two_views(X)
            yield v1, v2


# -------------------------------------------------------------------------
# InfoNCE loss over channels (pos: same channel across views)
# -------------------------------------------------------------------------
def info_nce_channels(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """
    z1, z2: (B, C, D) normalized
    Treat positions (b, c) across views as positives, all others negatives.
    """
    B, C, D = z1.shape
    N = B * C
    a = z1.reshape(N, D)
    b = z2.reshape(N, D)
    logits = (a @ b.T) / tau  # (N, N)
    labels = torch.arange(N, device=z1.device)
    loss1 = nn.functional.cross_entropy(logits, labels)
    loss2 = nn.functional.cross_entropy(logits.T, labels)
    return 0.5 * (loss1 + loss2)


# -------------------------------------------------------------------------
# Public API: train encoder, compute channel embeddings, build projector
# -------------------------------------------------------------------------
@torch.no_grad()
def compute_channel_embeddings(
    windows_ds,
    encoder: ContrastiveChannelEncoder,
    num_batches: int = 25,
    batch_size: int = 64,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Average per-channel embeddings over a few random mini-batches.
    Returns: np.ndarray of shape (C, D)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = _default_loader(windows_ds, batch_size, num_workers, shuffle=True)

    ch_mean: Optional[torch.Tensor] = None
    n_accum = 0
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        X = _get_X(batch).float().to(device)   # (B, C, T)
        with torch.no_grad():
            z = encoder(X)                    # (B, C, D)
        z_mean = z.mean(dim=0)                # (C, D), average over batch
        if ch_mean is None:
            ch_mean = z_mean
        else:
            ch_mean += z_mean
        n_accum += 1

    if ch_mean is None:
        raise RuntimeError("No batches seen while computing channel embeddings.")
    ch_mean = (ch_mean / float(n_accum)).detach().cpu().numpy()
    return ch_mean  # (C, D)


@torch.no_grad()
def _estimate_channel_cov(
    windows_ds,
    batches: int = 25,
    batch_size: int = 64,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Estimate channel covariance matrix (C x C) by averaging X X^T over windows.
    X is zero-mean per channel per window.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = _default_loader(windows_ds, batch_size, num_workers, shuffle=True)

    cov = None
    n_seen = 0
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        X = _get_X(batch).float().to(device)   # (B, C, T)
        X = X - X.mean(dim=-1, keepdim=True)   # zero-mean per channel
        # window-wise covariance, then average over B
        # X @ X^T / T gives (B, C, C)
        B, C, T = X.shape
        cov_b = (X @ X.transpose(1, 2)) / float(T)  # (B, C, C)
        cov_batch = cov_b.mean(dim=0)               # (C, C)
        cov = cov_batch if cov is None else (cov + cov_batch)
        n_seen += 1

    if cov is None:
        raise RuntimeError("No batches seen while estimating covariance.")
    cov = cov / float(n_seen)
    return cov  # (C, C) on device


@torch.no_grad()
def build_channel_projection_from_ssl(
    windows_ds,
    encoder: ContrastiveChannelEncoder,
    n_clusters: int = 20,
    pcs_per_cluster: int = 3,
    cov_batches: int = 25,
    batch_size: int = 64,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    1) Embed channels with the SSL encoder → (C, D)
    2) Cluster channel embeddings (KMeans K=n_clusters)
    3) Estimate channel covariance (C, C)
    4) For each cluster, take the top-P eigenvectors of the cluster sub-cov
       and place them into a global projection matrix W (C, sum_p) where
       sum_p = sum over clusters of chosen PCs (<= pcs_per_cluster).
    Returns: W as a numpy array (C, F) to be used as a 1x1 conv projector.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not _HAVE_SK:
        raise RuntimeError("scikit-learn is required for KMeans/PCA. Please install it.")

    # 1) channel embeddings (C, D)
    ch_emb = compute_channel_embeddings(
        windows_ds, encoder,
        num_batches=min(cov_batches, 50),
        batch_size=batch_size, num_workers=num_workers, device=device
    )  # (C, D)
    C, D = ch_emb.shape

    # 2) clusters
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=2025)
    labels = kmeans.fit_predict(ch_emb)  # (C,)

    # 3) channel covariance
    cov = _estimate_channel_cov(
        windows_ds, batches=cov_batches, batch_size=batch_size,
        num_workers=num_workers, device=device
    )  # (C, C)
    cov = cov.detach().cpu()

    # 4) build W
    # Compute total number of features (sum of per-cluster PCs)
    cluster_sizes = [int((labels == k).sum()) for k in range(n_clusters)]
    per_cluster_p = [min(pcs_per_cluster, max(1, s)) for s in cluster_sizes]
    F = int(sum(per_cluster_p))

    W = torch.zeros((C, F), dtype=torch.float32)  # CPU for easy numpy conversion
    col_ptr = 0

    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        m = len(idx)
        if m == 0:
            continue
        p = min(pcs_per_cluster, m)

        # take sub-cov
        sub = cov[np.ix_(idx, idx)]  # (m, m)
        # numeric safety
        # eigenvals ascending; take top-p eigenvectors
        try:
            evals, evecs = torch.linalg.eigh(sub)  # (m,), (m, m)
            U = evecs[:, -p:]                      # (m, p)
        except Exception:
            U = torch.eye(m, p)                    # fallback

        # ---- FIXED LINE (no [:, None]) ----
        W[idx, col_ptr:col_ptr + p] = U
        col_ptr += p

    return W.numpy()  # (C, F)


# -------------------------------------------------------------------------
# Training loop for SSL encoder
# -------------------------------------------------------------------------
def train_ssl_encoder(
    windows_ds,
    device: Optional[torch.device] = None,
    epochs: int = 10,
    steps_per_epoch: int = 150,
    batch_size: int = 64,
    num_workers: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    tau: float = 0.2,
    verbose: bool = True,
) -> ContrastiveChannelEncoder:
    """
    Simple contrastive pretraining: same-channel across two views is positive.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = ContrastiveChannelEncoder(embed_dim=64).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=weight_decay)

    sampler = DualViewChannelSampler(windows_ds, batch_size=batch_size,
                                     num_workers=num_workers, device=device)

    if verbose:
        print(f"[SSL] Pretraining encoder for {epochs} epochs x {steps_per_epoch} steps...")

    for ep in range(1, epochs + 1):
        enc.train()
        it = iter(sampler)
        running = 0.0
        for step in range(steps_per_epoch):
            try:
                v1, v2 = next(it)
            except StopIteration:
                # recreate iterator if we reached the end
                it = iter(sampler)
                v1, v2 = next(it)

            z1 = enc(v1)  # (B, C, D)
            z2 = enc(v2)  # (B, C, D)
            loss = info_nce_channels(z1, z2, tau=tau)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())

        if verbose:
            print(f"[SSL {ep:02d}/{epochs}] contrastive_loss={running/steps_per_epoch:.4f}")

    enc.eval()
    return enc
