# src/eegcfct/models/clustered_eegnex.py
import torch
import torch.nn as nn
from braindecode.models import EEGNeX

class ClusteredEEGNeX(nn.Module):
    """
    y = EEGNeX( W @ x )
    where x: (B, C_in, T), W: (C_out, C_in), implemented as Conv1d(C_in->C_out, k=1, bias=False)
    """
    def __init__(self, in_chans: int, out_chans: int, sfreq: int):
        super().__init__()
        self.projector = nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False)
        # projector weights will be loaded from state_dict; they are frozen during training (we set requires_grad_=False)
        self.backbone = EEGNeX(
            n_chans=out_chans, n_outputs=1, sfreq=sfreq, n_times=int(2 * sfreq)
        )

    def forward(self, x):
        # x: (B, C_in, T)
        x = self.projector(x)
        return self.backbone(x)
