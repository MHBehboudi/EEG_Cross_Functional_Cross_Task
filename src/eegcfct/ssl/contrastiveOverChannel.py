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
        
        # LSTM input size is 1 because we process the magnitude of the time-series
        self.lstm = nn.LSTM(input_size=1, 
                            hidden_size=self.HIDDEN_SIZE, 
                            bidirectional=True,
                            batch_first=False) 
        
        # Linear layer input size: 2 * Hidden Size (32) * C (channels)
        self.lin_input_size = in_ch * 2 * self.HIDDEN_SIZE
        self.lin = nn.Linear(self.lin_input_size, emb_dim) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) - Batch, Channels, Time
        B, C, T = x.shape 

        # 1. Temporal Convolution: (B, C, T) -> (B, C, T)
        h = self.conv(x) 
        
        # 2. Reshape for Channel-Wise LSTM: (B, C, T) -> (T, B*C, 1)
        # Sequence first, then combined batch, then 1 feature
        h = h.reshape(B * C, T).unsqueeze(-1) # (B*C, T, 1)
        h = h.transpose(0, 1) # (T, B*C, 1) - Sequence first for default LSTM

        # 3. LSTM Processing: (T, B*C, 1) -> output is (2, B*C, H) [Final Hidden State]
        # We only care about the final hidden state h_n.
        _, (h_n, c_n) = self.lstm(h) 
        
        # 4. Process Hidden State: (2, B*C, H) -> (B*C, 2*H)
        h = h_n.transpose(0, 1).reshape(B * C, -1) 
        
        # 5. Linear Projection: (B*C, 2*H) -> (B*C, D)
        h = self.lin(h)
        
        # 6. Final Reshape to Output: (B*C, D) -> (B, C, D)
        return h.reshape(B, C, h.shape[-1])


# --- HELPER FUNCTIONS ---

def random_crop_pair(x: torch.Tensor, crop_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates two independent random crops from a batch tensor."""
    T = x.shape[-1]
    
    crop_len = min(crop_len, T)
    
    if crop_len == T:
        v1 = v2 = x
    else:
        max_start = T - crop_len + 1
        s1 = torch.randint(0, max_start, (1,), device=x.device).item()
        s2 = torch.randint(0, max_start, (1,), device=x.device).item()
        
        v1 = x[..., s1:s1 + crop_len]
        v2 = x[..., s2:s2 + crop_len]

    return v1, v2

def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """Normalized Temperature-scaled Cross-Entropy Loss."""
    B, C, D = z1.shape
    
    # Flatten B and C into effective batch size N_eff = B * C
    z1 = z1.reshape(B * C, D) 
    z2 = z2.reshape(B * C, D) 
    
    # L2-normalize
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    
    # Combine and Similarity
    reps = torch.cat([z1, z2], dim=0)          
    sim = reps @ reps.T                        
    N = reps.shape[0]                          
    mask = torch.eye(N, dtype=torch.bool, device=sim.device)
    
    # Scaling and Masking
    sim = sim / tau
    logits = sim.masked_fill(mask, -torch.inf)
    
    # Define Targets (Positives)
    N_eff = B * C
    labels = torch.arange(N_eff, device=sim.device)
    pos = torch.cat([labels + N_eff, labels], dim=0) 
    
    # Compute Cross-Entropy Loss
    loss = F.cross_entropy(logits, pos)
    return loss


def train_ssl_encoder(
    windows_ds,
    *,
    epochs: int = 10,
    steps_per_epoch: int = 150,
    batch_size: int = 25,
    crop_len: int = 150,
    device: torch.device,
) -> nn.Module:
    """The main SSL training loop for the TinyChLSTMEncoder."""
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


# --- PHASE 2: CHANNEL GROUPING (CLUSTERING AND PCA) ---

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
    """Extracts the averaged (C, D) embedding matrix from the trained encoder."""
    probe_dl = DataLoader(windows_ds, batch_size=1, shuffle=True)
    X0 = next(iter(probe_dl))[0].to(device).float()
    C = X0.shape[1]
    
    T_probe = min(crop_len, X0.shape[-1])
    dummy_input = torch.zeros(1, C, T_probe, device=device)
    D = encoder(dummy_input).shape[-1]
    
    acc = torch.zeros(C, D, device=device)
    n = 0
    
    dl = DataLoader(windows_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    it = iter(dl)
    
    encoder.eval() 
    
    for _ in range(batches):
        try:
            batch = next(it)
        except StopIteration: 
            it = iter(dl)
            batch = next(it)
            
        X = batch[0].to(device).float() 
        v1, _ = random_crop_pair(X, min(crop_len, X.shape[-1]))
        z = encoder(v1) 
        
        acc += z.mean(dim=0)
        n += 1

    emb = (acc / max(n, 1)).detach().cpu().numpy()
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
            
        # Select channels, move to CPU, convert to NumPy
        X = batch[0][:, chan_idx, :].cpu().numpy()  # (B, nc, T)
        
        # Reshape to (nc, B*T) for PCA where channels are features
        X_blocks.append(X.transpose(1, 0, 2).reshape(len(chan_idx), -1))
        got += 1

    Xcat = np.concatenate(X_blocks, axis=1)  # (nc, bigT)
    
    # Standardization (mean-subtraction and division by std)
    Xcat = (Xcat - Xcat.mean(axis=1, keepdims=True)) / (Xcat.std(axis=1, keepdims=True) + 1e-8)
    
    p = min(pcs_per_cluster, Xcat.shape[0])
    
    # PCA: Fit on Xcat.T where rows=samples (B*T), columns=features (nc)
    U = PCA(n_components=p, svd_solver="full").fit(Xcat.T).components_.T
    return U

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
    # 1. Embed and get total channels
    emb = compute_channel_embeddings(windows_ds, encoder, device=device)
    C, _ = emb.shape

    # 2. KMeans clustering
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(emb)  # (C,)

    out_rows = 0
    bases = []
    
    # 3. Compute PCA basis for each cluster
    for g in range(n_clusters):
        idx = np.where(labels == g)[0] # Channel indices belonging to cluster g
        if len(idx) == 0:
            bases.append((idx, np.zeros((0, 0), dtype=np.float32)))
            continue
            
        U = _pca_basis_from_raw(windows_ds, idx, pcs_per_cluster=pcs_per_cluster) # (nc, p)
        
        bases.append((idx, U))
        out_rows += U.shape[1] # Accumulate total output dimensions

    # 4. Assemble the final projection matrix W
    W = np.zeros((out_rows, C), dtype=np.float32)
    row = 0
    for (idx, U) in bases:
        p = U.shape[1]
        if p == 0:
            continue
        # Place U^T (the projection vectors) into the appropriate rows and columns
        W[row:row + p, idx] = U.T
        row += p

    return W # Final shape: (Sum of all PCs, Total Channels C)
