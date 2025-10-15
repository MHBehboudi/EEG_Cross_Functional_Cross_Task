# src/eegcfct/models/graph_transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# -------- Tokenizer (channel-wise temporal conv) ----------
class DepthwiseTemporalTokenizer(nn.Module):
    """Depthwise temporal conv to make channel tokens (B,C,T) -> (B,C,T) -> (B,T,C)"""
    def __init__(self, n_chans: int, k: int = 9, p: int = 4, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(n_chans, n_chans, kernel_size=k, padding=p, groups=n_chans, bias=False)
        self.bn   = nn.BatchNorm1d(n_chans)
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")
    def forward(self, x):           # x: (B,C,T)
        h = self.conv(x)            # (B,C,T)
        h = self.bn(h)
        h = F.gelu(h)
        h = self.drop(h)
        return h.transpose(1, 2)    # (B,T,C) tokens = time steps, dim=C

# -------- Small Channel Graph block (no extra deps) -------
class ChannelGraphBlock(nn.Module):
    """
    Message passing over channels via A (learnable). Operates per time step.
    Input: (B,T,C) -> (B,T,C). A is CxC; we softplus to keep weights positive-ish, row-normalize.
    """
    def __init__(self, n_chans: int, hid: int):
        super().__init__()
        self.A = nn.Parameter(torch.randn(n_chans, n_chans) * 0.02)
        self.lin = nn.Linear(n_chans, n_chans)
        self.proj = nn.Linear(n_chans, hid)
        self.norm = nn.LayerNorm(hid)

    def forward(self, x):           # x: (B,T,C)
        B, T, C = x.shape
        A = F.softplus(self.A) + 1e-6
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

        # message passing: for each time step, Y_t = X_t @ A^T
        y = torch.matmul(x, A.t())          # (B,T,C)
        y = F.gelu(self.lin(y)) + x         # residual in channel space
        y = self.proj(y)                    # (B,T,hid)
        return self.norm(y)

# -------- Transformer encoder over time -------------------
def _make_pe(Tmax: int, d: int, device):
    pe = torch.zeros(Tmax, d, device=device)
    pos = torch.arange(0, Tmax, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # (Tmax, d)

class TimeTransformer(nn.Module):
    def __init__(self, dim: int, depth: int = 2, heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):  # (B,T,D)
        y = self.enc(x)
        return self.norm(y)

# -------- Full models -------------------------------------
class DeMegaLikeBackbone(nn.Module):
    """
    Tokenize (channel-wise conv) -> (optional) ChannelGraph -> Time Transformer -> mean pool -> head
    Expects (B,C,T). If a projector is used before, C is the projected channel count.
    """
    def __init__(
        self, n_chans: int, sfreq: int, win_sec: float,
        use_graph: bool = True, token_k: int = 9, d_model: int = 128,
        depth: int = 2, heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1
    ):
        super().__init__()
        self.sfreq = sfreq
        self.n_times = int(win_sec * sfreq)

        self.tokenizer = DepthwiseTemporalTokenizer(n_chans, k=token_k, p=token_k // 2, dropout=dropout)  # (B,T,C)
        self.proj_in   = nn.Linear(n_chans, d_model)

        self.graph = ChannelGraphBlock(n_chans, hid=d_model) if use_graph else None
        self.time_tf = TimeTransformer(d_model, depth=depth, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x):           # x: (B,C,T)
        # 1) tokenize channels with temporal DW-conv
        tok = self.tokenizer(x)     # (B,T,C)
        if self.graph is not None:
            g  = self.graph(tok)    # (B,T,d_model) (already projected)
        else:
            g  = self.proj_in(tok)  # (B,T,d_model)

        # 2) transformer over time
        y = self.time_tf(g)         # (B,T,d_model)

        # 3) global mean over time (window pooling) -> regression head
        y = y.mean(dim=1)           # (B,d_model)
        return self.head(y)         # (B,1)

class ClusteredDeMega(nn.Module):
    """
    Optional 1x1 projector first (C_in -> C_out) then DeMegaLikeBackbone.
    """
    def __init__(self, in_chans: int, out_chans: Optional[int], sfreq: int, win_sec: float, **backbone_kwargs):
        super().__init__()
        if out_chans is not None and out_chans != in_chans:
            self.projector = nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False)
            n_backbone_ch = out_chans
        else:
            self.projector = None
            n_backbone_ch = in_chans
        self.backbone = DeMegaLikeBackbone(n_chans=n_backbone_ch, sfreq=sfreq, win_sec=win_sec, **backbone_kwargs)

    def forward(self, x):     # (B,C,T)
        if self.projector is not None:
            x = self.projector(x)
        return self.backbone(x)
