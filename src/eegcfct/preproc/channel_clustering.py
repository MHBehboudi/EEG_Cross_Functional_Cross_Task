# src/eegcfct/preproc/channel_clustering.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Optional, Sequence

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


@dataclass
class ChannelClusteringResult:
    labels: np.ndarray            # shape (n_chans,), values in [0, n_clusters-1]
    W: np.ndarray                 # shape (n_clusters, n_chans), rows sum to 1
    mean_corr: np.ndarray         # shape (n_chans, n_chans)
    n_clusters: int
    n_chans: int


def _corrcoef_abs(x: np.ndarray) -> np.ndarray:
    """Return |corr| across channels for a single window X with shape (C, T) or (1,C,T)."""
    if x.ndim == 3:  # (1, C, T)
        x = x[0]
    # channels (variables) in rows, samples (time) in columns
    c = np.corrcoef(x)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    return np.abs(c)


def _accumulate_mean_corr(train_set, n_chans: int, max_windows: int, seed: int) -> np.ndarray:
    """Accumulate mean absolute correlation across (up to) max_windows windows."""
    rng = np.random.default_rng(seed)
    n = len(train_set)
    # choose a subset for speed (deterministic with seed)
    if max_windows <= 0 or max_windows >= n:
        idxs = np.arange(n)
    else:
        idxs = rng.choice(n, size=max_windows, replace=False)
    S = np.zeros((n_chans, n_chans), dtype=np.float64)
    count = 0
    for i in idxs:
        item = train_set[i]
        # item can be (X,y) or (X,y,...) or dict-like
        if isinstance(item, (tuple, list)):
            X = item[0]
        else:
            X = item["X"] if "X" in item else item[0]  # best-effort
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        C = _corrcoef_abs(X)
        if C.shape != (n_chans, n_chans):
            # Try to coerce if a singleton dim exists
            if C.ndim == 3 and C.shape[0] == 1 and C.shape[1:] == (n_chans, n_chans):
                C = C[0]
            else:
                continue
        S += C
        count += 1
    if count == 0:
        raise RuntimeError("No windows were accumulated to compute channel correlations.")
    return (S / count).astype(np.float32)


def _linkage_labels_from_distance(D: np.ndarray, n_clusters: int) -> np.ndarray:
    """Hierarchical clustering (average linkage) from a full distance matrix."""
    # condensed vector for SciPy
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    # clusters labeled from 1..k → convert to 0..k-1
    labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1
    return labels.astype(np.int64)


def _build_cluster_matrix(labels: np.ndarray, n_chans: int, n_clusters: int) -> np.ndarray:
    """Return W in R^{K x C} with uniform weights within each cluster, rows sum to 1."""
    W = np.zeros((n_clusters, n_chans), dtype=np.float32)
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            # If empty cluster sneaks in (rare with maxclust), assign a single channel
            # to keep dimensions consistent.
            k_fill = np.argmin(np.bincount(labels, minlength=n_clusters))
            idx = np.array([k_fill], dtype=int)
        W[k, idx] = 1.0 / len(idx)
    return W


def compute_channel_clustering(
    train_set,
    n_chans: int,
    n_clusters: int,
    max_windows: int = 1500,
    seed: int = 2025,
) -> ChannelClusteringResult:
    """
    1) Aggregate |corr| across channels from a subset of training windows
    2) Convert to distance D = 1 - mean_corr
    3) Hierarchical clustering to get labels
    4) Build cluster averaging matrix W (K x C)
    """
    mean_corr = _accumulate_mean_corr(train_set, n_chans=n_chans, max_windows=max_windows, seed=seed)
    mean_corr = np.clip(mean_corr, 0.0, 1.0)
    # Turn similarity into distance
    D = 1.0 - mean_corr
    np.fill_diagonal(D, 0.0)
    labels = _linkage_labels_from_distance(D, n_clusters=n_clusters)
    W = _build_cluster_matrix(labels, n_chans=n_chans, n_clusters=n_clusters)
    return ChannelClusteringResult(labels=labels, W=W, mean_corr=mean_corr,
                                   n_clusters=n_clusters, n_chans=n_chans)


def save_channel_clustering(path: Path | str, result: ChannelClusteringResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        labels=result.labels,
        W=result.W,
        mean_corr=result.mean_corr,
        n_clusters=np.array([result.n_clusters], dtype=np.int64),
        n_chans=np.array([result.n_chans], dtype=np.int64),
    )


def load_channel_clustering(path: Path | str) -> ChannelClusteringResult:
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    return ChannelClusteringResult(
        labels=data["labels"],
        W=data["W"],
        mean_corr=data["mean_corr"],
        n_clusters=int(data["n_clusters"][0]),
        n_chans=int(data["n_chans"][0]),
    )


class ClusteredWindowsDataset(torch.utils.data.Dataset):
    """
    Wrap a windows dataset and left-multiply the channel dimension by W (K x C)
    so each sample X (C,T) becomes X' (K,T). Works with (X,y) or (X,y,...) tuples.
    """
    def __init__(self, base_ds, W: np.ndarray):
        super().__init__()
        self.base = base_ds
        # store as torch for speed
        self.W = torch.as_tensor(W, dtype=torch.float32)

    def __len__(self):
        return len(self.base)

    def _apply(self, X: torch.Tensor) -> torch.Tensor:
        # Expect X shape (C,T) or (1,C,T)
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float32)
        if X.ndim == 3:
            # (1,C,T) → drop singleton
            X = X[0]
        # W (K,C) @ X (C,T) → (K,T)
        Xr = self.W @ X
        return Xr

    def __getitem__(self, idx):
        item = self.base[idx]
        if isinstance(item, (tuple, list)):
            X, y, *rest = item
            Xr = self._apply(X)
            # Keep structure: (X', y, rest...)
            if len(rest) == 0:
                return Xr, y
            return (Xr, y, *rest)
        elif isinstance(item, dict):
            X = item["X"]; y = item.get("y", None)
            Xr = self._apply(X)
            if y is None:
                return {"X": Xr, **{k:v for k,v in item.items() if k != "X"}}
            return (Xr, y)
        else:
            X, y = item  # best effort
            Xr = self._apply(X)
            return (Xr, y)
