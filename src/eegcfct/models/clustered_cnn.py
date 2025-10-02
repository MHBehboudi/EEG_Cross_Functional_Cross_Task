# src/eegcfct/models/clustered_cnn.py
import torch
import torch.nn as nn

class SimpleEEGRegressor(nn.Module):
    """
    Minimal time-adaptive 1D CNN:
      input  : [B, C, T]
      output : [B, 1]
    """
    def __init__(self, n_in_ch: int, n_filters: int = 64, kernel_size: int = 11):
        super().__init__()
        self.n_in_ch = n_in_ch
        self.conv1 = nn.Conv1d(n_in_ch, n_filters, kernel_size=kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.act1 = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(n_filters, 1)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool(x).squeeze(-1)  # [B, n_filters]
        x = self.head(x)              # [B, 1]
        return x


class ChannelProjector(nn.Module):
    """
    1x1 Conv over channels: C_in -> K clusters (learnable).
    We initialize weights as cluster means (averaging matrix), then optionally freeze.
    """
    def __init__(self, c_in: int, k_out: int, init_weight: torch.Tensor | None = None, trainable: bool = False):
        super().__init__()
        self.proj = nn.Conv1d(c_in, k_out, kernel_size=1, bias=False)
        if init_weight is not None:
            # init_weight: [K, C, 1]
            with torch.no_grad():
                self.proj.weight.copy_(init_weight)
        for p in self.proj.parameters():
            p.requires_grad_(bool(trainable))

    def forward(self, x):
        return self.proj(x)


class ClusteredEEGRegressor(nn.Module):
    """
    Optional projector (C_in -> K) followed by the SimpleEEGRegressor on K channels.
    If K == C_in and init is identity, this reduces to the plain model.
    """
    def __init__(self, c_in: int, k_out: int | None = None,
                 n_filters: int = 64, kernel_size: int = 11,
                 projector_weight: torch.Tensor | None = None,
                 projector_trainable: bool = False):
        super().__init__()
        self.c_in = c_in
        self.k_out = k_out if k_out is not None else c_in

        if self.k_out != self.c_in or projector_weight is not None:
            self.projector = ChannelProjector(
                c_in=self.c_in, k_out=self.k_out,
                init_weight=projector_weight, trainable=projector_trainable
            )
        else:
            self.projector = None

        self.backbone = SimpleEEGRegressor(n_in_ch=self.k_out,
                                           n_filters=n_filters,
                                           kernel_size=kernel_size)

    def forward(self, x):
        # x: [B, C_in, T]
        if self.projector is not None:
            x = self.projector(x)  # [B, K, T]
        return self.backbone(x)


def build_model_clustered(c_in: int,
                          use_clustering: bool = False,
                          n_clusters: int = 50,
                          projector_weight: torch.Tensor | None = None,
                          projector_trainable: bool = False,
                          n_filters: int = 64,
                          kernel_size: int = 11) -> nn.Module:
    if not use_clustering:
        return ClusteredEEGRegressor(c_in=c_in, k_out=c_in,
                                     projector_weight=None,
                                     projector_trainable=False,
                                     n_filters=n_filters, kernel_size=kernel_size)
    else:
        return ClusteredEEGRegressor(c_in=c_in, k_out=n_clusters,
                                     projector_weight=projector_weight,
                                     projector_trainable=projector_trainable,
                                     n_filters=n_filters, kernel_size=kernel_size)
