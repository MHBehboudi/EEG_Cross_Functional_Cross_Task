# src/eegcfct/models/demega.py
import torch
import torch.nn as nn

class DeMEGA(nn.Module):
    """
    DeMEGA-style backbone:
      * Tokenizer: per-channel temporal DW-CNN -> k features per channel
      * Project k -> d_model per channel (linear)
      * Graph-Transformer over channel tokens (TransformerEncoder across C)
      * Channel pooling -> MLP head -> scalar

    Input:  (B, C, T)
    Output: (B, 1)
    """
    def __init__(
        self,
        in_chans: int,
        d_model: int = 64,
        nhead: int = 4,
        depth: int = 2,
        k_per_channel: int = 4,
        mlp_hidden: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.d_model = d_model
        self.k = k_per_channel

        # Per-channel temporal tokenizer: DW-Conv → (B, C*k, T)
        self.stem = nn.Sequential(
            nn.Conv1d(in_chans, in_chans * self.k, kernel_size=7, padding=3, groups=in_chans, bias=False),
            nn.BatchNorm1d(in_chans * self.k),
            nn.GELU(),
            nn.Conv1d(in_chans * self.k, in_chans * self.k, kernel_size=5, padding=2, groups=in_chans * self.k, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # (B, C*k, 1)
        )

        # Project per-channel k features → d_model
        self.proj = nn.Linear(self.k, d_model)

        # Channel “graph” transformer: sequence length = C, embed = d_model
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=False, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # Head: mean-pool over channels → MLP → scalar
        self.head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

        # Small learnable per-channel positional embedding
        self.pos = nn.Parameter(torch.zeros(in_chans, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        # (optional) id buffer to help submission.py detect arch robustly
        self.register_buffer("arch_id", torch.tensor([2], dtype=torch.int32))  # 2 == demega

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, _ = x.shape
        h = self.stem(x)                  # (B, C*k, 1)
        h = h.view(B, C, self.k)          # (B, C, k)
        h = self.proj(h)                  # (B, C, d_model)
        h = h + self.pos.unsqueeze(0)     # (B, C, d_model)

        # Transformer expects (S, B, E)
        h = h.transpose(0, 1)             # (C, B, d_model)
        h = self.encoder(h)               # (C, B, d_model)
        h = h.mean(dim=0)                 # (B, d_model)
        y = self.head(h)                  # (B, 1)
        return y

class ClusteredDeMEGA(nn.Module):
    """
    Wrapper with frozen projector (C_in -> C_out) then DeMEGA on C_out.
    """
    def __init__(self, in_chans: int, out_chans: int, d_model: int = 64, depth: int = 2, nhead: int = 4):
        super().__init__()
        self.projector = nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False)
        self.backbone  = DeMEGA(in_chans=out_chans, d_model=d_model, depth=depth, nhead=nhead)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projector(x)
        return self.backbone(x)
