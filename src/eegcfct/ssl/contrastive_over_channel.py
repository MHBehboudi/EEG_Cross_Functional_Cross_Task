import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, List
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- PHASE 1: ENCODER MODULE ---
class TinyChLSTMEncoder(nn.Module):
    """
    Channel-wise LSTM Encoder: Extracts temporal features for each channel separately.
    Input: (B, C, T) -> Output: (B, C, D)
    """
    def __init__(self, in_ch: int, emb_dim: int = 32):
        super().__init__()
        self.HIDDEN_SIZE = 16

        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, padding=2, kernel_size=5, groups=in_ch),
            nn.BatchNorm1d(in_ch),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Conv1d(in_ch, in_ch, padding=2, kernel_size=5, groups=in_ch),
            nn.BatchNorm1d(in_ch),
            nn.Dropout(0.2),
            nn.GELU(),
        )

        # LSTM over magnitude per-channel
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.HIDDEN_SIZE,
                            bidirectional=True, batch_first=False)

        self.lin_input_size = in_ch * 2 * self.HIDDEN_SIZE
        self.lin = nn.Linear(self.lin_input_size, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        h = self.conv(x)
        h = h.reshape(B * C, T).unsqueeze(-1)  # (B*C, T, 1)
        h = h.transpose(0, 1)                  # (T, B*C, 1) for LSTM
        _, (h_n, _) = self.lstm(h)             # h_n: (2, B*C, H)
        h = h_n.transpose(0, 1).reshape(B * C, -1)  # (B*C, 2H)
        h = self.lin(h)                         # (B*C, D)
        return h.reshape(B, C, h.shape[-1])     # (B, C, D)

# --- HELPERS ---
def random_crop_pair(x: torch.Tensor, crop_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    T = x.shape[-1]
    crop_len = min(crop_len, T)
    if crop_len == T:
        return x, x
    max_start = T - crop_len + 1
    s1 = torch.randint(0, max_start, (1,), device=x.device).item()
    s2 = torch.randint(0, max_start, (1,), device=x.device).item()
    return x[..., s1:s1 + crop_len], x[..., s2:s2 + crop_len]

def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    B, C, D = z1.shape
    z1 = F.normalize(z1.reshape(B * C, D), dim=-1)
    z2 = F.normalize(z2.reshape(B * C, D), dim=-1)
    reps = torch.cat([z1, z2], dim=0)              # (2*N_eff, D)
    sim = reps @ reps.T                            # (2*N, 2*N)
    N = sim.shape[0]
    sim = sim / tau
    sim = sim.masked_fill(torch.eye(N, device=sim.device, dtype=torch.bool), float("-inf"))
    N_eff = B * C
    labels = torch.arange(N_eff, device=sim.device)
    pos = torch.cat([labels + N_eff, labels], dim=0)
    return F.cross_entropy(sim, pos)

def train_ssl_encoder(windows_ds, *, epochs: int = 10, steps_per_epoch: int = 150,
                      batch_size: int = 25, crop_len: int = 150, device: torch.device) -> nn.Module:
    probe = DataLoader(windows_ds, batch_size=1, shuffle=True)
    X0 = next(iter(probe))[0]
    C = X0.shape[1]

    enc = TinyChLSTMEncoder(in_ch=C, emb_dim=32).to(device)
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
            X = batch[0].to(device).float()
            v1, v2 = random_crop_pair(X, crop_len)
            z1, z2 = enc(v1), enc(v2)
            loss = nt_xent(z1, z2, tau=0.2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"[SSL {ep:02d}/{epochs}] contrastive_loss={np.mean(losses):.4f}")
    enc.eval()
    return enc

@torch.no_grad()
def compute_channel_embeddings(windows_ds, encoder: nn.Module, *, batches: int = 64,
                               batch_size: int = 16, crop_len: int = 150,
                               device: torch.device) -> np.ndarray:
    probe = DataLoader(windows_ds, batch_size=1, shuffle=True)
    X0 = next(iter(probe))[0].to(device).float()
    C = X0.shape[1]
    T_probe = min(crop_len, X0.shape[-1])
    D = encoder(torch.zeros(1, C, T_probe, device=device)).shape[-1]
    acc = torch.zeros(C, D, device=device); n = 0

    dl = DataLoader(windows_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    it = iter(dl); encoder.eval()
    for _ in range(batches):
        try: batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)
        X = batch[0].to(device).float()
        v1, _ = random_crop_pair(X, min(crop_len, X.shape[-1]))
        z = encoder(v1)           # (B,C,D)
        acc += z.mean(dim=0)      # (C,D)
        n += 1

    return (acc / max(n, 1)).detach().cpu().numpy()   # (C,D)

def _pca_basis_from_raw(windows_ds, chan_idx: np.ndarray, *, pcs_per_cluster: int,
                        samples: int = 80, batch_size: int = 16) -> np.ndarray:
    dl = DataLoader(windows_ds, batch_size=batch_size, shuffle=True)
    it = iter(dl)
    X_blocks: List[np.ndarray] = []; got = 0
    while got < samples:
        try: batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)
        X = batch[0][:, chan_idx, :].cpu().numpy()              # (B, nc, T)
        X_blocks.append(X.transpose(1, 0, 2).reshape(len(chan_idx), -1))
        got += 1
    Xcat = np.concatenate(X_blocks, axis=1)                     # (nc, bigT)
    Xcat = (Xcat - Xcat.mean(axis=1, keepdims=True)) / (Xcat.std(axis=1, keepdims=True) + 1e-8)
    p = min(pcs_per_cluster, Xcat.shape[0])
    U = PCA(n_components=p, svd_solver="full").fit(Xcat.T).components_.T  # (nc,p)
    return U

def build_channel_projection_from_ssl(windows_ds, encoder: nn.Module, *,
                                      n_clusters: int, pcs_per_cluster: int,
                                      device: torch.device) -> np.ndarray:
    emb = compute_channel_embeddings(windows_ds, encoder, device=device)  # (C,D)
    C, _ = emb.shape
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(emb)                                         # (C,)
    W_rows = []
    for g in range(n_clusters):
        idx = np.where(labels == g)[0]
        if len(idx) == 0:
            continue
        U = _pca_basis_from_raw(windows_ds, idx, pcs_per_cluster=pcs_per_cluster)  # (nc,p)
        Wg = np.zeros((U.shape[1], C), dtype=np.float32)                            # (p,C)
        Wg[:, idx] = U.T
        W_rows.append(Wg)
    if len(W_rows) == 0:
        return np.eye(C, dtype=np.float32)  # fallback
    W = np.concatenate(W_rows, axis=0)      # (sum_p, C)
    return W
