# src/eegcfct/models/projected_eegnex.py
from __future__ import annotations
import torch
import torch.nn as nn
from braindecode.models import EEGNeX


class ProjectedEEGNeX(nn.Module):
    """
    Channel-clustered EEGNeX:
      - projector: 1x1 Conv1d mixing channels -> K clusters
      - backbone : EEGNeX configured with n_chans=K
    """
    def __init__(
        self,
        in_ch: int,
        k_out: int,
        sfreq: int,
        n_times: int,
        eegnex_kwargs: dict | None = None,
    ):
        super().__init__()
        self.projector = nn.Conv1d(in_ch, k_out, kernel_size=1, bias=False)

        kw = dict(n_chans=k_out, n_outputs=1, sfreq=sfreq, n_times=n_times)
        if eegnex_kwargs:
            kw.update(eegnex_kwargs)
        self.backbone = EEGNeX(**kw)

        # store a few meta fields (useful when loading on Codabench)
        self.in_ch = in_ch
        self.k_out = k_out
        self.sfreq = sfreq
        self.n_times = n_times

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T]
        x = self.projector(x)      # -> [B, K, T]
        y = self.backbone(x)       # -> [B, 1]
        return y
