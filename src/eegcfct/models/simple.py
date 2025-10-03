# src/eegcfct/models/simple.py
import torch
import torch.nn as nn

class SimpleEEGRegressor(nn.Module):
    """
    Minimal 1D CNN:
      - temporal convs over each channel set (Conv1d expects input as [B, C, T])
      - global average over time -> linear head
    Works for any n_chans; time length is handled by AdaptiveAvgPool1d(1).
    """
    def __init__(self, n_chans: int, n_times: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_chans, 64, kernel_size=7, padding="same", bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding="same", bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 128, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # -> (B, 128, 1)
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x).squeeze(-1)  # (B, 128)
        return self.head(x)               # (B, 1)


class ProjectedRegressor(nn.Module):
    """
    Projector (fixed 1x1 Conv over channels) -> SimpleEEGRegressor backbone.
    The projector weight is set from your KMeans+PCA matrix and then frozen.
    """
    def __init__(self, n_chans_in: int, n_chans_out: int, n_times: int):
        super().__init__()
        self.projector = nn.Conv1d(n_chans_in, n_chans_out, kernel_size=1, bias=False)
        for p in self.projector.parameters():
            p.requires_grad = False  # projector stays fixed during supervised training
        self.backbone = SimpleEEGRegressor(n_chans_out, n_times)

    def forward(self, x):
        x = self.projector(x)  # (B, C_out, T)
        return self.backbone(x)
