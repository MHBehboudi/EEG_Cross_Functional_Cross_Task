# submission.py
import os
from pathlib import Path
import torch
import torch.nn as nn

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass


class SimpleEEGRegressor(nn.Module):
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
        x = self.pool(x).squeeze(-1)
        x = self.head(x)
        return x


class ClusteredEEGRegressor(nn.Module):
    def __init__(self, projector_in: int, projector_out: int,
                 n_filters: int = 64, kernel_size: int = 11):
        super().__init__()
        self.projector = nn.Conv1d(projector_in, projector_out, kernel_size=1, bias=False)
        self.backbone = SimpleEEGRegressor(n_in_ch=projector_out,
                                           n_filters=n_filters,
                                           kernel_size=kernel_size)

    def forward(self, x):
        x = self.projector(x)
        return self.backbone(x)


class ChannelAdapter(nn.Module):
    """Adapt incoming channels to expected projector_in via slice/pad."""
    def __init__(self, model: ClusteredEEGRegressor, projector_in: int):
        super().__init__()
        self.model = model
        self.expected_c = projector_in

    def forward(self, x):
        b, c_in, t = x.shape
        c_exp = self.expected_c
        if c_in == c_exp:
            x_in = x
        elif c_in > c_exp:
            x_in = x[:, :c_exp, :]
        else:
            pad = torch.zeros(b, c_exp - c_in, t, dtype=x.dtype, device=x.device)
            x_in = torch.cat([x, pad], dim=1)
        return self.model(x_in)


def _resolve(name: str) -> Path:
    for p in (Path("/app/output")/name, Path("/app/ingested_program")/name, Path(".")/name):
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing {name}")

def _make_from_weights(wp: Path, device: torch.device):
    state = torch.load(wp, map_location=device)
    if "projector.weight" not in state:
        raise RuntimeError("Weights must include 'projector.weight'")
    proj_w = state["projector.weight"]  # [K, C, 1]
    k_out, c_in = int(proj_w.shape[0]), int(proj_w.shape[1])
    ksize = 11
    if "backbone.conv1.weight" in state:
        ksize = int(state["backbone.conv1.weight"].shape[-1])
    n_filters = int(state.get("__meta_n_filters", torch.tensor(64)).item())

    model = ClusteredEEGRegressor(projector_in=c_in, projector_out=k_out,
                                  n_filters=n_filters, kernel_size=ksize)
    model.load_state_dict(state, strict=True)
    model.eval()
    return ChannelAdapter(model, projector_in=c_in)


class Submission:
    def __init__(self):
        self.device = torch.device("cpu")

    def get_model_challenge_1(self):
        return _make_from_weights(_resolve("weights_challenge_1.pt"), self.device)

    def get_model_challenge_2(self):
        return _make_from_weights(_resolve("weights_challenge_2.pt"), self.device)
