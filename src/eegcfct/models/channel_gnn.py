# src/eegcfct/models/channel_gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelGCNRegressor(nn.Module):
    """
    Per-channel temporal features (depthwise conv + GAP) -> two GCN layers over a fixed A_hat -> readout.
    A_hat (C x C) is a normalized adjacency registered as a buffer and learned/estimated at train time, then saved.
    """
    def __init__(self, in_chans: int, sfreq: int = 100, m: int = 8, d_time: int = 32, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.sfreq = sfreq
        self.m = m

        self.dw = nn.Conv1d(in_chans, in_chans * m, kernel_size=7, padding=3, groups=in_chans, bias=False)
        self.proj_node = nn.Linear(m, d_time, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.gcn1 = nn.Linear(d_time, hidden, bias=False)
        self.gcn2 = nn.Linear(hidden, hidden, bias=False)

        self.head = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        self.register_buffer("adj", torch.eye(in_chans))

    def _node_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.dw(x)                         # (B, C*m, T)
        h = F.gelu(h)
        h = F.adaptive_avg_pool1d(h, 1)        # (B, C*m, 1)
        B, CM, _ = h.shape
        C = self.adj.shape[0]
        h = h.view(B, C, self.m)               # (B, C, m)
        h = self.proj_node(h)                  # (B, C, d_time)
        return h

    def _gcn_layer(self, X: torch.Tensor, lin: nn.Linear) -> torch.Tensor:
        H = torch.matmul(self.adj, X)          # (B, C, F)
        H = lin(H)
        return F.gelu(H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X = self._node_features(x)             # (B, C, d_time)
        H = self._gcn_layer(X, self.gcn1)      # (B, C, hidden)
        H = self.dropout(H)
        H = self._gcn_layer(H, self.gcn2)      # (B, C, hidden)
        H = H.mean(dim=1)                      # (B, hidden)
        return self.head(H)                    # (B,1)
