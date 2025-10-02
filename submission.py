# submission.py
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Cap threads on Codabench CPUs
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass


# --------- small, time-adaptive 1D CNN regressor ----------
class SimpleEEGRegressor(nn.Module):
    """
    Input:  [B, C, T] with C = expected channels from trained weights
    Output: [B, 1]
    """
    def __init__(self, n_in_ch: int, n_filters: int = 64, kernel_size: int = 11):
        super().__init__()
        self.n_in_ch = n_in_ch
        self.conv1 = nn.Conv1d(n_in_ch, n_filters, kernel_size=kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.act1 = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)  # time-adaptive
        self.head = nn.Linear(n_filters, 1)

    def forward(self, x):
        # x: [B, C, T] where C == n_in_ch (enforced by wrapper)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool(x).squeeze(-1)  # [B, n_filters]
        x = self.head(x)              # [B, 1]
        return x


class ChannelAdapter(nn.Module):
    """
    Wraps a model to adapt incoming channels to the model's expected C.
    If incoming C_in > C_exp: keep first C_exp channels.
    If incoming C_in < C_exp: zero-pad channels to C_exp.
    """
    def __init__(self, base_model: SimpleEEGRegressor):
        super().__init__()
        self.base = base_model
        self.expected_c = base_model.n_in_ch

    def forward(self, x):
        # x: [B, C_in, T]
        b, c_in, t = x.shape
        c_exp = self.expected_c
        if c_in == c_exp:
            x_in = x
        elif c_in > c_exp:
            x_in = x[:, :c_exp, :]
        else:  # c_in < c_exp
            pad = torch.zeros(b, c_exp - c_in, t, dtype=x.dtype, device=x.device)
            x_in = torch.cat([x, pad], dim=1)
        return self.base(x_in)


# --------- weight path helpers ----------
def _candidates(name: str):
    # Try common Codabench locations + CWD
    return [
        Path("/app/output") / name,
        Path("/app/ingested_program") / name,
        Path(".") / name,
    ]

def _resolve_existing(name: str) -> Path:
    for p in _candidates(name):
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {name} in expected locations.")


def _build_model_from_weights(weight_path: Path, device: torch.device):
    state = torch.load(weight_path, map_location=device)
    # infer input channels and kernel from first conv
    if "conv1.weight" not in state:
        raise RuntimeError("Weights do not contain 'conv1.weight' to infer input channels.")
    conv_w = state["conv1.weight"]            # [n_filters, n_in_ch, k]
    n_filters = int(conv_w.shape[0])
    n_in_ch  = int(conv_w.shape[1])
    ksize    = int(conv_w.shape[2]) if conv_w.ndim >= 3 else 11

    base = SimpleEEGRegressor(n_in_ch=n_in_ch, n_filters=n_filters, kernel_size=ksize)
    base.load_state_dict(state, strict=True)
    base.eval()
    return ChannelAdapter(base)  # add channel adaptation wrapper


# --------- Codabench API ----------
class Submission:
    def __init__(self):
        self.device = torch.device("cpu")

    def _make_from(self, weight_name: str):
        wp = _resolve_existing(weight_name)
        model = _build_model_from_weights(wp, self.device)
        model.eval()
        return model

    def get_model_challenge_1(self):
        return self._make_from("weights_challenge_1.pt")

    def get_model_challenge_2(self):
        return self._make_from("weights_challenge_2.pt")
