# src/eegcfct/models/projected_eegnex.py
from __future__ import annotations
import torch
from torch import nn
from braindecode.models import EEGNeX

class ProjectedEEGNeX(nn.Module):
    """
    A fixed linear spatial projection (1x1 Conv) followed by EEGNeX.
    The projector is frozen; its weights are learned offline (clusters+PCA).
    """
    def __init__(self, P_weight: torch.Tensor, sfreq: float, n_times: int, n_outputs: int = 1):
        super().__init__()
        K, C, _ = P_weight.shape
        self.projector = nn.Conv1d(
            in_channels=C, out_channels=K, kernel_size=1, bias=False
        )
        with torch.no_grad():
            self.projector.weight.copy_(P_weight)
        for p in self.projector.parameters():
            p.requires_grad = False  # frozen

        self.backbone = EEGNeX(
            n_chans=K, n_outputs=n_outputs, sfreq=sfreq, n_times=n_times
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.projector(x)   # (B, K, T)
        return self.backbone(x)
