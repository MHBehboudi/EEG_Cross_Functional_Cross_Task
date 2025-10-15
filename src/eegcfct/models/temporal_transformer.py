# src/eegcfct/models/temporal_transformer.py
import torch
import torch.nn as nn

class TemporalTransformerRegressor(nn.Module):
    """
    (B, C, T=200) -> conv1x1 -> (B, d_model, T) -> TransformerEncoder over time -> mean-pool -> MLP -> (B,1)
    """
    def __init__(
        self,
        in_chans: int,
        sfreq: int = 100,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sfreq = sfreq
        self.stem = nn.Conv1d(in_chans, d_model, kernel_size=1, bias=False)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.stem(x)          # (B, d_model, T)
        h = h.transpose(1, 2)     # (B, T, d_model)
        h = self.encoder(h)       # (B, T, d_model)
        h = h.mean(dim=1)         # (B, d_model)
        y = self.head(h)          # (B, 1)
        return y
