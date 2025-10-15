# submission.py
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# ---------- Minimal DeMEGA-style backbone used during training ----------
class SimplePatchEmbed(nn.Module):
    def __init__(self, n_chans: int, d_model: int = 128, token_k: int = 9, sfreq: int = 100, n_times: int = 200, dropout: float = 0.1):
        super().__init__()
        self.token_k = token_k
        # depthwise temporal conv to get local tokens per channel
        self.dw = nn.Conv1d(n_chans, n_chans, kernel_size=token_k, padding=token_k//2, groups=n_chans, bias=False)
        self.pw = nn.Conv1d(n_chans, d_model, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, C, T)
        h = self.dw(x)
        h = self.pw(h)             # (B, d_model, T)
        h = self.act(h)
        h = self.drop(h)
        # mean pool over time to get a single token sequence length T=1 (keep it simple)
        h = h.mean(-1, keepdim=True).transpose(1, 2)  # (B, 1, d_model)
        return h

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.sa = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=int(d_model*mlp_ratio),
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )

    def forward(self, x):
        return self.sa(x)

class DeMegaBackbone(nn.Module):
    def __init__(self, n_chans: int, d_model: int = 128, depth: int = 2, n_heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.1, token_k: int = 9, sfreq: int = 100, n_times: int = 200):
        super().__init__()
        self.stem = SimplePatchEmbed(n_chans, d_model, token_k, sfreq, n_times, dropout)
        self.encoder = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):  # x: (B, C, T)
        tok = self.stem(x)           # (B, 1, D)
        h = self.encoder(tok)        # (B, 1, D)
        y = self.head(h[:, 0, :])    # (B, 1)
        return y

# ---------- Projector + Backbone wrapper (matches training) ----------
class ClusteredModel(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, sfreq: int,
                 d_model=128, depth=2, n_heads=4, mlp_ratio=2.0, dropout=0.1, token_k=9, n_times=200):
        super().__init__()
        self.projector = nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False)  # frozen in training
        self.backbone  = DeMegaBackbone(
            n_chans=out_chans, d_model=d_model, depth=depth, n_heads=n_heads,
            mlp_ratio=mlp_ratio, dropout=dropout, token_k=token_k, sfreq=sfreq, n_times=n_times
        )

    def forward(self, x):
        x = self.projector(x)
        return self.backbone(x)

# ---------- Submission API ----------
class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq  = SFREQ
        self.device = DEVICE

    def _load(self, fname: str):
        # Load weights; infer projector shape to rebuild model
        sd = torch.load(f"/app/input/res/{fname}", map_location=self.device)
        if "projector.weight" in sd:
            out_c, in_c, _ = sd["projector.weight"].shape
            model = ClusteredModel(
                in_chans=in_c, out_chans=out_c, sfreq=self.sfreq,
                # keep defaults aligned with training script
                d_model=128, depth=2, n_heads=4, mlp_ratio=2.0, dropout=0.1, token_k=9, n_times=int(2*self.sfreq)
            ).to(self.device)
        else:
            # Fallback: no projector saved (unlikely in your current pipeline)
            model = DeMegaBackbone(
                n_chans=129, d_model=128, depth=2, n_heads=4, mlp_ratio=2.0, dropout=0.1, token_k=9,
                sfreq=self.sfreq, n_times=int(2*self.sfreq)
            ).to(self.device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model

    def get_model_challenge_1(self):
        return self._load("weights_challenge_1.pt")

    def get_model_challenge_2(self):
        return self._load("weights_challenge_2.pt")
