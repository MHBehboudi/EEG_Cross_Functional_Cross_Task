# src/eegcfct/ssl/contrastive.py  — replace just this function

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

@torch.no_grad()
def build_channel_projection_from_ssl(
    windows_ds,
    encoder: torch.nn.Module,
    device: torch.device,
    n_clusters: int = 20,
    pcs_per_cluster: int = 3,
    n_win_for_pca: int = 150,   # NEW: accepted for API compatibility
):
    """
    Build a fixed channel projection W (C_out x C_in) using SSL channel embeddings:
      1) Get one embedding per channel -> shape (C, D)
      2) KMeans over channels (C) into K clusters
      3) For each cluster, fit PCA on the member-channel embeddings and
         take the first 'pcs_per_cluster' components -> rows in W
    Returns:
      W as a NumPy array of shape (K * pcs_per_cluster, C)
    """
    # If you already have a helper like `compute_channel_embeddings`, use it.
    # Otherwise, inline a minimal version here. We assume you already have it.
    try:
        from .contrastive import compute_channel_embeddings  # self import is fine
    except Exception:
        raise RuntimeError(
            "compute_channel_embeddings() not found; make sure it's defined in contrastive.py"
        )

    encoder.eval()
    # Get (C, D) channel embeddings (C = number of EEG channels)
    emb = compute_channel_embeddings(windows_ds, encoder, device=device)
    # emb: torch.Tensor [C, D]
    emb_np = emb.detach().cpu().numpy()
    C, D = emb_np.shape

    # KMeans over channels
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = kmeans.fit_predict(emb_np)

    # Build W by stacking per-cluster PCA components
    C_out = n_clusters * pcs_per_cluster
    W = np.zeros((C_out, C), dtype=np.float32)

    row = 0
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            # empty cluster: put a one-hot on an arbitrary channel to keep shape stable
            W[row, 0] = 1.0
            row += pcs_per_cluster
            continue

        # PCA on the member-channel embeddings
        Xk = emb_np[idx]  # shape (Nk, D)
        p = min(pcs_per_cluster, Xk.shape[0], D)
        pca = PCA(n_components=p, random_state=0)
        pca.fit(Xk)
        U = pca.components_  # shape (p, D)

        # Project each channel embedding onto these PCs -> weights per channel
        # W_block shape (p, C): for channels in idx, fill with projection; others 0
        proj = (emb_np @ U.T)  # (C, p)
        W_block = proj.T       # (p, C)

        # Normalize each row to unit norm to keep projector well-scaled
        norms = np.linalg.norm(W_block, axis=1, keepdims=True) + 1e-8
        W_block = W_block / norms

        # Write into W
        W[row:row + p, :] = W_block[:p, :]
        # If p < pcs_per_cluster, leave the remaining rows zeros (safe default)
        row += pcs_per_cluster

    return W
