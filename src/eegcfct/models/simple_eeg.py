# src/eegcfct/models/simple_eeg.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelProjector(nn.Module):
    """
    Fixed linear projection over channels.
    x: (B, C, T)  ->  (B, K, T), where W is (K, C).
    If W is None, acts as identity.
    """
    def __init__(self, W: torch.Tensor | None):
        super().__init__()
        if W is None:
            self.register_buffer("W", None)
        else:
            assert W.dim() == 2, "W must be 2D (K, C)"
            self.register_buffer("W", W.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.W is None:
            return x
        # einsum: (B,C,T) x (K,C) -> (B,K,T)
        return torch.einsum("bct,kc->bkt", x, self.W)


class SimpleEEGRegressor(nn.Module):
    """
    Minimal, robust Conv1d regressor for EEG windows:
      - Optional fixed channel projection W (for clustering)
      - A few Conv1d + BN + GELU blocks
      - AdaptiveAvgPool1d(1) => time-invariant
      - Small MLP head to scalar output

    Shape safety: independent of T (window length) due to AdaptiveAvgPool1d.
    Works with any #channels because W (if provided) changes n_chans at the front.
    """
    def __init__(
        self,
        n_chans: int,
        n_outputs: int = 1,
        proj_W: torch.Tensor | None = None,
        n_filters: int = 64,
        temporal_kernel: int = 25,
        depth: int = 3,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.project = ChannelProjector(proj_W)

        c_in = n_chans if proj_W is None else proj_W.shape[0]
        self.conv1 = nn.Conv1d(c_in, n_filters, kernel_size=temporal_kernel,
                               padding=temporal_kernel // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters)

        blocks: list[nn.Module] = []
        for _ in range(max(depth - 1, 0)):
            blocks += [
                nn.Conv1d(n_filters, n_filters, kernel_size=15, padding=7, bias=False),
                nn.BatchNorm1d(n_filters),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.backbone = nn.Sequential(*blocks) if blocks else nn.Identity()

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # -> (B, n_filters, 1)
            nn.Flatten(),             # -> (B, n_filters)
            nn.Linear(n_filters, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # windows give (C,T) or (B,C,T)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.float()  # safety
        x = self.project(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.backbone(x)
        return self.head(x)
