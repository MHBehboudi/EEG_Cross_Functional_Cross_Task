# src/eegcfct/featurizers/cluster_features.py
from __future__ import annotations
import math
from typing import Literal, Tuple, List
import numpy as np
import torch
from torch import nn
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA

FeatureType = Literal["corr", "cov"]
ClusterMethod = Literal["kmeans", "spectral", "ae"]

# ========== Utilities ==========

def _stack_window_corr(X: np.ndarray) -> np.ndarray:
    """X: (C, T) -> corr matrix (C, C) with Fisher-z handling."""
    C = X.shape[0]
    if X.shape[1] < 2:  # too short
        return np.zeros((C, C), dtype=np.float32)
    R = np.corrcoef(X)
    # Clamp for numerical stability then Fisher z
    R = np.clip(R, -0.999999, 0.999999)
    Z = np.arctanh(R)
    np.fill_diagonal(Z, 0.0)  # do not explode the diagonal
    return Z.astype(np.float32)

def compute_channel_features_from_windows(
    windows_ds,
    feature_type: FeatureType = "corr",
    max_windows: int = 200,
    stride: int = 1,
    seed: int = 2025,
) -> np.ndarray:
    """
    Build a channel-feature matrix F (C, C) using average Fisher-z correlations
    (or covariances) across a subset of windows.
    """
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(windows_ds))
    if max_windows is not None:
        if max_windows < len(idxs):
            idxs = rng.choice(idxs, size=max_windows, replace=False)
    sum_mat, count = None, 0
    for i in idxs:
        X = windows_ds[i][0]  # (C, T)
        X = np.asarray(X, dtype=np.float32)
        if feature_type == "corr":
            Z = _stack_window_corr(X[:, ::stride])
            if sum_mat is None:
                sum_mat = Z
            else:
                sum_mat += Z
            count += 1
        elif feature_type == "cov":
            Xc = X - X.mean(axis=1, keepdims=True)
            S = (Xc @ Xc.T) / (Xc.shape[1] - 1)
            if sum_mat is None:
                sum_mat = S
            else:
                sum_mat += S
            count += 1
        else:
            raise ValueError(f"Unsupported feature_type: {feature_type}")
    if count == 0:
        raise RuntimeError("No windows found to compute channel features.")
    M = sum_mat / float(count)
    if feature_type == "corr":
        M = np.tanh(M)  # back to correlation scale
    return M.astype(np.float32)  # (C, C)

# ========== Optional AE Embedding (fast, tiny) ==========

class TinyAE(nn.Module):
    def __init__(self, in_dim: int, emb: int = 16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, emb)
        )
        self.dec = nn.Sequential(
            nn.Linear(emb, 64), nn.ReLU(),
            nn.Linear(64, in_dim)
        )
    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat, z

def _ae_learn_embeddings(features: np.ndarray, emb_dim: int = 16, epochs: int = 30, lr=1e-3, seed=2025) -> np.ndarray:
    """
    features: (C, C) matrix -> each row is a channel feature vector
    returns embeddings: (C, emb_dim)
    """
    torch.manual_seed(seed)
    C = features.shape[0]
    x = torch.from_numpy(features).float()
    model = TinyAE(in_dim=C, emb=emb_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        x_hat, _ = model(x)
        loss = loss_fn(x_hat, x)
        loss.backward()
        opt.step()
    with torch.no_grad():
        _, z = model(x)
    return z.detach().cpu().numpy().astype(np.float32)

# ========== Clustering ==========

def cluster_channels(
    features: np.ndarray,
    method: ClusterMethod = "kmeans",
    n_clusters: int = 20,
    seed: int = 2025,
) -> np.ndarray:
    """
    features: (C, C) — channel feature vectors (rows).
    returns labels: (C,)
    """
    C = features.shape[0]
    X = features
    if method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
        labels = kmeans.fit_predict(X)
    elif method == "spectral":
        # Use similarity from |corr| if available; otherwise cosine sim
        if np.allclose(np.diag(features), 1.0, atol=1e-2):
            A = np.abs(features)
        else:
            # normalize rows, cosine sim
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            A = Xn @ Xn.T
            A = (A + A.T) / 2
            np.fill_diagonal(A, 1.0)
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=seed,
        )
        labels = sc.fit_predict(A)
    elif method == "ae":
        Z = _ae_learn_embeddings(features, emb_dim=16, epochs=40, seed=seed)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
        labels = kmeans.fit_predict(Z)
    else:
        raise ValueError(f"Unknown cluster method: {method}")
    return labels.astype(np.int32)

# ========== PCA Projection (sklearn) ==========

def build_spatial_pca_projection(
    windows_ds,
    labels: np.ndarray,
    pcs_per_cluster: int = 3,
    sample_stride: int = 2,
    max_samples_per_cluster: int = 100_000,
    seed: int = 2025,
) -> np.ndarray:
    """
    Learn a channel-space projection P (K, C) where K = sum_k pcs_per_cluster
    by fitting PCA on the (time x channels_in_cluster) matrices concatenated
    across sampled windows (sklearn PCA).
    """
    rng = np.random.default_rng(seed)
    C = windows_ds[0][0].shape[0]
    assert labels.shape[0] == C
    cluster_ids = np.unique(labels)
    K_total = pcs_per_cluster * len(cluster_ids)
    P = np.zeros((K_total, C), dtype=np.float32)

    # Precompute an order of windows to sample
    order = np.arange(len(windows_ds))
    rng.shuffle(order)

    row = 0
    for k in cluster_ids:
        ch_idx = np.where(labels == k)[0]
        if len(ch_idx) == 0:
            # degenerate cluster; leave zeros
            row += pcs_per_cluster
            continue

        # Build design matrix: stack timepoints across windows
        X_stack = []
        total = 0
        for i in order:
            X = windows_ds[i][0]  # (C, T)
            Xc = np.asarray(X[ch_idx, ::sample_stride], dtype=np.float32).T  # (T', n_ch_cluster)
            X_stack.append(Xc)
            total += Xc.shape[0]
            if total >= max_samples_per_cluster:
                break
        X_stack = np.concatenate(X_stack, axis=0)  # (N_time, n_ch_cluster)
        # Center
        X_stack -= X_stack.mean(axis=0, keepdims=True)

        n_comp = min(pcs_per_cluster, X_stack.shape[1])
        pca = PCA(n_components=n_comp, svd_solver="auto", random_state=seed)
        pca.fit(X_stack)   # components_: (n_comp, n_ch_cluster)

        # Place components into P rows at the columns of this cluster
        comps = pca.components_.astype(np.float32)
        P[row:row+n_comp, :][:, ch_idx] = comps
        # If fewer comps than requested, keep zeros in remaining rows
        row += pcs_per_cluster

    return P  # (K_total, C)

# ========== Convert P -> Conv1d weight ==========

def projection_matrix_to_conv1d_weight(P: np.ndarray) -> torch.Tensor:
    """
    P: (K, C) used as y = P @ x  where x is (C, T).
    For Conv1d(in=C, out=K, kernel=1), weight needs shape (K, C, 1)
    so that y[b, k, t] = sum_c W[k,c,0] * x[b,c,t]
    """
    W = torch.from_numpy(P).float().unsqueeze(-1)  # (K, C, 1)
    return W
