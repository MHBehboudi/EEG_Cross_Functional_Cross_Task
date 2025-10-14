# submission.py
import os
import numpy as np
import torch
import torch.nn as nn
from braindecode.models import EEGNeX

RES_DIR_ENV = "EEG2025_RES_DIR"

def _res_dir():
    return os.environ.get(RES_DIR_ENV, "/app/input/res")

def _load_projection_or_identity(n_in: int = 129):
    """Load projection matrix W (M x 129) from projection.npy; if missing, use identity (129 x 129)."""
    path = os.path.join(_res_dir(), "projection.npy")
    if os.path.isfile(path):
        W = np.load(path)  # expected shape (M, 129)
        if W.ndim != 2 or W.shape[1] != n_in:
            raise RuntimeError(f"projection.npy has wrong shape {W.shape}; expected (M, {n_in})")
        return torch.tensor(W, dtype=torch.float32)
    # fallback: identity (no projection)
    return torch.eye(n_in, dtype=torch.float32)

class ProjectThenEEGNeX(nn.Module):
    """Applies fixed 1x1 conv (W) to map 129->M channels, then EEGNeX(n_chans=M)."""
    def __init__(self, W: torch.Tensor, sfreq: int, device: torch.device):
        super().__init__()
        M, C = W.shape  # W: (M, 129)
        self.proj = nn.Conv1d(in_channels=C, out_channels=M, kernel_size=1, bias=False)
        with torch.no_grad():
            # Conv1d weight: (out_ch, in_ch, k) -> (M, 129, 1)
            self.proj.weight.copy_(W.unsqueeze(-1))
        for p in self.proj.parameters():
            p.requires_grad_(False)  # keep projection fixed at inference

        self.backbone = EEGNeX(
            n_chans=M, n_outputs=1, sfreq=sfreq, n_times=int(2 * sfreq)
        )
        self.to(device)

    def forward(self, x):
        # x: (B, 129, 200)
        x = self.proj(x)
        return self.backbone(x)

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def _clean_state_dict(self, sd: dict) -> dict:
        cleaned = {}
        # If wrapped like {"state_dict": {...}}
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]

        for k, v in sd.items():
            # strip DDP/DataParallel
            if k.startswith("module."):
                k = k[len("module."):]
            # keep only backbone.* params (drop projector/ssl heads)
            if k.startswith("projector.") or k.startswith("ssl_head.") or k.startswith("head."):
                continue
            cleaned[k] = v
        return cleaned

    def _load_weights_backbone_only(self, model: ProjectThenEEGNeX, filename: str):
        path = os.path.join(_res_dir(), filename)
        if not os.path.isfile(path):
            return  # run with random init (allowed)
        sd = torch.load(path, map_location=self.device)
        sd = self._clean_state_dict(sd)

        # Two possible situations:
        #  1) keys like "backbone.*" -> load directly into model.backbone
        #  2) keys without prefix (plain EEGNeX) -> also load into model.backbone
        bb_sd = {}
        for k, v in sd.items():
            if k.startswith("backbone."):
                bb_sd[k[len("backbone."):]] = v
            else:
                bb_sd[k] = v
        # allow missing/unexpected keys safely
        model.backbone.load_state_dict(bb_sd, strict=False)

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
