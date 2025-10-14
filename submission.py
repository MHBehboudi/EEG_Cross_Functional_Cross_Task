# submission.py
import os
import numpy as np
import torch
import torch.nn as nn
from braindecode.models import EEGNeX

# ---- HARD CAP THREADS to avoid oversubscription / stalls ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass
# -------------------------------------------------------------

RES_DIR_ENV = "EEG2025_RES_DIR"

def _res_dir():
    return os.environ.get(RES_DIR_ENV, "/app/input/res")

def _load_projection_or_identity(n_in: int = 129):
    """Load projection.npy (M x n_in); fallback to identity if missing/bad."""
    path = os.path.join(_res_dir(), "projection.npy")
    try:
        if os.path.isfile(path):
            W = np.load(path, allow_pickle=False)
            if W.ndim == 2 and W.shape[1] == n_in and W.shape[0] > 0:
                return torch.tensor(W, dtype=torch.float32)
    except Exception:
        pass
    return torch.eye(n_in, dtype=torch.float32)

class ProjectThenEEGNeX(nn.Module):
    """Fixed 1x1 conv (W) to map 129->M channels, then EEGNeX(n_chans=M)."""
    def __init__(self, W: torch.Tensor, sfreq: int, device: torch.device):
        super().__init__()
        M, C = W.shape
        self.proj = nn.Conv1d(in_channels=C, out_channels=M, kernel_size=1, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(W.unsqueeze(-1))  # (M, C, 1)
        for p in self.proj.parameters():
            p.requires_grad_(False)
        self.backbone = EEGNeX(n_chans=M, n_outputs=1, sfreq=sfreq, n_times=int(2 * sfreq))
        self.to(device)

    def forward(self, x):
        x = self.proj(x)
        return self.backbone(x)

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def _clean_state_dict(self, sd: dict) -> dict:
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        cleaned = {}
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("projector.") or k.startswith("ssl_head.") or k.startswith("head."):
                continue
            cleaned[k] = v
        return cleaned

    def _load_weights_backbone_only(self, model: ProjectThenEEGNeX, filename: str):
        path = os.path.join(_res_dir(), filename)
        if not os.path.isfile(path):
            return
        try:
            sd = torch.load(path, map_location=self.device)
            sd = self._clean_state_dict(sd)
            bb_sd = {}
            for k, v in sd.items():
                if k.startswith("backbone."):
                    bb_sd[k[len("backbone."):]] = v
                else:
                    bb_sd[k] = v
            model.backbone.load_state_dict(bb_sd, strict=False)
        except Exception:
            # If loading fails, proceed with random init (still a valid submission)
            pass

    def _build_model(self):
        W = _load_projection_or_identity(n_in=129).to(self.device)
        return ProjectThenEEGNeX(W, self.sfreq, self.device)

    def get_model_challenge_1(self):
        m = self._build_model()
        self._load_weights_backbone_only(m, "weights_challenge_1.pt")
        return m

    def get_model_challenge_2(self):
        m = self._build_model()
        self._load_weights_backbone_only(m, "weights_challenge_2.pt")
        return m
