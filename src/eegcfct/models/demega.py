# src/eegcfct/models/demega.py
import torch
import torch.nn as nn

class DeMEGA(nn.Module):
    """
    Minimal DeMEGA-like backbone for EEG/MEG:
      (B, C, T) -> (B, 1)
    """
    def __init__(self, n_chans: int, n_times: int, d_model=128, n_heads=4, depth=2):
        super().__init__()
        self.stem = nn.Conv1d(n_chans, d_model, kernel_size=5, padding=2, bias=False)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                               batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # over time
            nn.Flatten(),             # (B, d_model)
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):            # x: (B, C, T)
        h = self.stem(x)             # (B, d_model, T)
        h = h.transpose(1, 2)        # (B, T, d_model)
        h = self.encoder(h)          # (B, T, d_model)
        h = h.transpose(1, 2)        # (B, d_model, T)
        return self.head(h)          # (B, 1)


class ClusteredDeMEGA(nn.Module):
    """
    1x1 projector (fixed at train end) + DeMEGA backbone
      (B, C_in, T) --projector--> (B, C_out, T) --DeMEGA--> (B, 1)
    """
    def __init__(self, in_chans: int, out_chans: int, n_times: int,
                 d_model=128, n_heads=4, depth=2):
        super().__init__()
        self.projector = nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False)
        self.backbone = DeMEGA(out_chans, n_times, d_model=d_model,
                               n_heads=n_heads, depth=depth)

    def forward(self, x):
        x = self.projector(x)
        return self.backbone(x)
