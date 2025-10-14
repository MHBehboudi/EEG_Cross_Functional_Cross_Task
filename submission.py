# submission.py
import os
import torch
from braindecode.models import EEGNeX

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        # Codabench puts submission files here
        self.res_dir = os.environ.get("EEG2025_RES_DIR", "/app/input/res")

    # ---- model builder ----
    def _build_model(self, n_chans=129):
        # 2 s at 100 Hz => 200 samples
        m = EEGNeX(
            n_chans=n_chans,
            n_outputs=1,
            sfreq=self.sfreq,
            n_times=int(2 * self.sfreq),
        ).to(self.device)
        return m

    # ---- state_dict cleaner for weights saved from a wrapper (backbone/projector) ----
    def _clean_state_dict(self, sd: dict) -> dict:
        cleaned = {}
        for k, v in sd.items():
            # strip DP/DDP prefix
            if k.startswith("module."):
                k = k[len("module."):]
            # unwrap backbone.* -> *
            if k.startswith("backbone."):
                k = k[len("backbone."):]
            # drop projector / ssl heads / classifier heads from pretraining
            if k.startswith("projector.") or k.startswith("ssl_head.") or k.startswith("head."):
                continue
            # keep everything else
            cleaned[k] = v
        return cleaned

    def _maybe_load(self, model, filename: str):
        path = os.path.join(self.res_dir, filename)
        if os.path.isfile(path):
            sd = torch.load(path, map_location=self.device)
            # Some saves store a dict like {"state_dict": ...}
            if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                sd = sd["state_dict"]
            sd = self._clean_state_dict(sd)
            # allow missing/unexpected keys after cleaning
            missing, unexpected = model.load_state_dict(sd, strict=False)
            # Optional: you can print or log these, but Codabench hides stdout often
            # print("Missing:", missing, "Unexpected:", unexpected)
        return model

    # ---- required API ----
    def get_model_challenge_1(self):
        m = self._build_model(n_chans=129)
        return self._maybe_load(m, "weights_challenge_1.pt")

    def get_model_challenge_2(self):
        m = self._build_model(n_chans=129)
        return self._maybe_load(m, "weights_challenge_2.pt")
