#src/eegcfct/ssl/contrastive.py
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
    enc = ChanEncoder(in_ch=in_ch, emb_dim=emb_dim).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=wd)
    sampler = TwoAugmentSampler(train_loader, steps_per_epoch=ssl_steps_per_epoch)

    enc.train()
    for ep in range(1, ssl_epochs + 1):
        tot, n = 0.0, 0
        for v1, v2 in sampler:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            z1 = enc(v1).reshape(-1, enc.emb_dim)  # (B*C, D)
            z2 = enc(v2).reshape(-1, enc.emb_dim)
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)
            loss = nt_xent(z1, z2, tau=tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.item()); n += 1
        print(f"[SSL {ep:02d}/{ssl_epochs}] contrastive_loss={tot/max(n,1):.4f}")

    enc.eval()
    return enc


# --------------------------
# Build channel embeddings, then projection W
# --------------------------
@torch.no_grad()
def compute_channel_embeddings(
    windows_ds,
    encoder: nn.Module,
    device,
    n_batches: int = 32,
    batch_size: int = 32,
):
    loader = DataLoader(windows_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    C, D = encoder.in_ch, encoder.emb_dim
    sum_emb = torch.zeros(C, D, device=device)
    n_seen = 0
    for i, batch in enumerate(loader):
        X = _get_X(batch).to(device).float()       # (B, C, T)
        z = encoder(X)                             # (B, C, D)
        sum_emb += z.sum(dim=0)                    # sum over batch
        n_seen += X.shape[0]
        if i + 1 >= n_batches:
            break
    emb = sum_emb / (max(n_seen, 1))               # (C, D)
    return emb


@torch.no_grad()
def build_channel_projection_from_ssl(
    windows_ds,
    encoder: nn.Module,
    device,
    n_clusters: int = 20,
    pcs_per_cluster: int = 3,
    n_win_for_pca: int = 150,   # accepted for API compatibility
):
    """
    Returns W as (K*pcs_per_cluster, C), so the model will see C_out = K*pcs_per_cluster channels.
    """
    emb = compute_channel_embeddings(windows_ds, encoder, device=device)
    emb_np = emb.detach().cpu().numpy()
    C, D = emb_np.shape

    # Cluster channels in embedding space
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = kmeans.fit_predict(emb_np)

    C_out = n_clusters * pcs_per_cluster
    W = np.zeros((C_out, C), dtype=np.float32)

    row = 0
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            # empty cluster fallback: simple one-hot row
            W[row, 0] = 1.0
            row += pcs_per_cluster
            continue

        Xk = emb_np[idx]                  # (Nk, D)
        p = min(pcs_per_cluster, Xk.shape[0], D)
        pca = PCA(n_components=p, random_state=0).fit(Xk)
        U = pca.components_               # (p, D)

        # Project all channel embeddings onto these p PCs -> (C, p)
        proj = emb_np @ U.T
        W_block = proj.T                  # (p, C)

        # Row-normalize for stability
        norms = np.linalg.norm(W_block, axis=1, keepdims=True) + 1e-8
        W_block = W_block / norms

        W[row:row + p, :] = W_block[:p, :]
        # if p < pcs_per_cluster, remaining rows stay zeros (benign)
        row += pcs_per_cluster

    return W
