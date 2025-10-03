# src/eegcfct/ssl/contrastive.py
from __future__ import annotations
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -----------------------
# Tiny channel encoder
# -----------------------
class TinyChEncoder(nn.Module):
    """
    Input: (B, C, T) -> depthwise temporal convs -> mean over time -> linear -> (B, C, D) (shared head)
    """
    def __init__(self, in_ch: int, emb_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=5, padding=2, groups=in_ch),
            nn.GELU(),
            nn.Conv1d(in_ch, in_ch, kernel_size=5, padding=2, groups=in_ch),
            nn.GELU(),
        )
        self.lin = nn.Linear(in_ch, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.conv(x)       # (B, C, T)
        h = h.mean(-1)         # (B, C)
        h = self.lin(h)        # (B, emb_dim)
        # Repeat across channels so output is (B, C, D)
        return h.unsqueeze(1).repeat(1, x.shape[1], 1)


# -----------------------
# Random crop augmentations
# -----------------------
def random_crop_pair(x: torch.Tensor, crop_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    T = x.shape[-1]
    crop_len = min(crop_len, T)
    if T == crop_len:
        s1 = s2 = 0
    else:
        s1 = torch.randint(0, T - crop_len + 1, (1,), device=x.device).item()
        s2 = torch.randint(0, T - crop_len + 1, (1,), device=x.device).item()
    return x[..., s1:s1 + crop_len], x[..., s2:s2 + crop_len]


# -----------------------
# NT-Xent loss
# -----------------------
def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    # z1, z2: (B, C, D) -> flatten channels into batch
    B, C, D = z1.shape
    z1 = z1.reshape(B * C, D)
    z2 = z2.reshape(B * C, D)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    reps = torch.cat([z1, z2], dim=0)           # (2N, D)
    sim = reps @ reps.T                         # cosine sim (since normalized)
    N = reps.shape[0]
    mask = torch.eye(N, dtype=torch.bool, device=sim.device)
    sim = sim / tau
    labels = torch.arange(B * C, device=sim.device)
    pos = torch.cat([labels + B * C, labels], dim=0)  # (2N,)
    logits = sim.masked_fill(mask, -1e9)
    loss = F.cross_entropy(logits, pos)
    return loss


# -----------------------
# SSL Training (SimCLR-lite)
# -----------------------
def train_ssl_encoder(
    windows_ds,
    *,
    epochs: int = 10,
    steps_per_epoch: int = 150,
    batch_size: int = 16,
    crop_len: int = 150,
    device: torch.device,
) -> nn.Module:
    from torch.utils.data import DataLoader
    probe = DataLoader(windows_ds, batch_size=1, shuffle=True)
    X0 = next(iter(probe))[0]  # (1, C, T)
    C = X0.shape[1]

    enc = TinyChEncoder(in_ch=C, emb_dim=32).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-3, weight_decay=1e-4)

    loader = DataLoader(windows_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    it = iter(loader)

    enc.train()
    for ep in range(1, epochs + 1):
        losses = []
        for _ in range(steps_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            X = batch[0].to(device).float()  # (B, C, T)
            v1, v2 = random_crop_pair(X, crop_len)
            z1 = enc(v1)
            z2 = enc(v2)
            loss = nt_xent(z1, z2, tau=0.2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"[SSL {ep:02d}/{epochs}] contrastive_loss={np.mean(losses):.4f}")
    enc.eval()
    return enc


@torch.no_grad()
def compute_channel_embeddings(
    windows_ds,
    encoder: nn.Module,
    *,
    batches: int = 64,
    batch_size: int = 16,
    crop_len: int = 150,
    device: torch.device,
) -> np.ndarray:
    from torch.utils.data import DataLoader
    dl = DataLoader(windows_ds, batch_size=batch_size, shuffle=True)
    it = iter(dl)

    # Discover C and D
    X0 = next(iter(DataLoader(windows_ds, batch_size=1)))[0]
    C = X0.shape[1]
    D = encoder(torch.zeros(1, C, min(crop_len, X0.shape[-1]), device=device)).shape[-1]

    acc = torch.zeros(C, D, device=device)
    n = 0
    for _ in range(batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        X = batch[0].to(device).float()
        v1, _ = random_crop_pair(X, min(crop_len, X.shape[-1]))
        z = encoder(v1)  # (B, C, D)
        acc += z.mean(dim=0)
        n += 1
    emb = (acc / max(n, 1)).detach().cpu().numpy()  # (C, D)
    return emb


def _pca_basis_from_raw(
    windows_ds,
    chan_idx: np.ndarray,
    *,
    pcs_per_cluster: int,
    samples: int = 80,
    batch_size: int = 16,
) -> np.ndarray:
    """Compute PCA basis (len(idx) x p) on raw signals aggregated over a few windows."""
    from torch.utils.data import DataLoader
    dl = DataLoader(windows_ds, batch_size=batch_size, shuffle=True)
    it = iter(dl)

    X_blocks: List[np.ndarray] = []
    got = 0
    while got < samples:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        X = batch[0][:, chan_idx, :].cpu().numpy()  # (B, nc, T)
        X_blocks.append(X.transpose(1, 0, 2).reshape(len(chan_idx), -1))  # (nc, B*T)
        got += 1

    Xcat = np.concatenate(X_blocks, axis=1)  # (nc, bigT)
    Xcat = (Xcat - Xcat.mean(axis=1, keepdims=True)) / (Xcat.std(axis=1, keepdims=True) + 1e-8)
    p = min(pcs_per_cluster, Xcat.shape[0])
    U = PCA(n_components=p, svd_solver="full").fit(Xcat.T).components_.T  # (nc, p)
    return U


@torch.no_grad()
def build_channel_projection_from_ssl(
    windows_ds,
    encoder: nn.Module,
    *,
    n_clusters: int,
    pcs_per_cluster: int,
    device: torch.device,
) -> np.ndarray:
    """
    1) Embed each channel -> (C, D)
    2) KMeans -> labels in [0..K-1]
    3) For each cluster, compute PCA over raw signals of those channels
    4) Assemble W (K*p, C), where each block row contains U^T in the cluster columns.
    """
    emb = compute_channel_embeddings(windows_ds, encoder, device=device)  # (C, D)
    C, _ = emb.shape

    # Use n_init=10 for older sklearn compatibility
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(emb)  # (C,)

    out_rows = 0
    bases = []
    for g in range(n_clusters):
        idx = np.where(labels == g)[0]
        if len(idx) == 0:
            bases.append((idx, np.zeros((0, 0), dtype=np.float32)))
            continue
        U = _pca_basis_from_raw(windows_ds, idx, pcs_per_cluster=pcs_per_cluster)  # (nc, p)
        bases.append((idx, U))
        out_rows += U.shape[1]

    W = np.zeros((out_rows, C), dtype=np.float32)
    row = 0
    for (idx, U) in bases:
        p = U.shape[1]
        if p == 0:
            continue
        W[row:row + p, idx] = U.T  # place U^T into proper columns
        row += p

    return W  # (C_out, C_in)
