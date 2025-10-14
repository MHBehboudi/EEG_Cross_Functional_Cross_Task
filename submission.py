# submission.py
import os
import torch
import torch.nn as nn
from braindecode.models import EEGNeX

# Cap threads to avoid stalls
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

RES_DIR = os.environ.get("EEG2025_RES_DIR", "/app/input/res")

class ClusteredEEGNeX(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, sfreq: int):
        super().__init__()
        self.projector = nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False)
        self.backbone  = EEGNeX(n_chans=out_chans, n_outputs=1, sfreq=sfreq, n_times=int(2 * sfreq))
    def forward(self, x):
        x = self.projector(x)
        return self.backbone(x)

def _strip_wrappers(sd: dict) -> dict:
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq  = SFREQ
        self.device = DEVICE

    def _load_weights_and_shape(self, filename: str):
        path = os.path.join(RES_DIR, filename)
        sd = torch.load(path, map_location=self.device)
        sd = _strip_wrappers(sd)

        # If projector.weight is present, reconstruct ClusteredEEGNeX with the right shapes
        if "projector.weight" in sd:
            out_c, in_c, _ = sd["projector.weight"].shape
            model = ClusteredEEGNeX(in_chans=in_c, out_chans=out_c, sfreq=self.sfreq).to(self.device)
        else:
            # Fallback to plain EEGNeX (rare; mainly for baseline)
            model = EEGNeX(n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2 * self.sfreq)).to(self.device)

        model.eval()
        model.load_state_dict(sd, strict=True)
        return model

    def get_model_challenge_1(self):
        return self._load_weights_and_shape("weights_challenge_1.pt")

    def get_model_challenge_2(self):
        return self._load_weights_and_shape("weights_challenge_2.pt")
