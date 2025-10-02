# src/eegcfct/models/projector_model.py
import torch
import torch.nn as nn
from braindecode.models import EEGNeX

class ChannelProjector(nn.Module):
    """
    1x1 Conv implementing a fixed linear projection: y = P @ x per timepoint.
    P must be [proj_dim, n_ch].
    """
    def __init__(self, P: torch.Tensor):
        super().__init__()
        proj_dim, n_ch = P.shape
        self.conv = nn.Conv1d(n_ch, proj_dim, kernel_size=1, bias=False)
        with torch.no_grad():
            w = P.clone().to(torch.float32).unsqueeze(-1)  # [proj_dim,n_ch,1]
            self.conv.weight.copy_(w)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):  # x: [B, C, T]
        return self.conv(x)

class ProjectorEEGNeX(nn.Module):
    def __init__(self, P: torch.Tensor, sfreq: int, n_times: int):
        super().__init__()
        proj_dim = P.shape[0]
        self.projector = ChannelProjector(P)
        self.backbone = EEGNeX(n_chans=proj_dim, n_outputs=1, sfreq=sfreq, n_times=n_times)

    def forward(self, x):  # x: [B,C,T]
        x = self.projector(x)
        return self.backbone(x)
