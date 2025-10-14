# submission.py
import os
import torch
from braindecode.models import EEGNeX

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        # Codabench puts submission files here:
        self.res_dir = os.environ.get("EEG2025_RES_DIR", "/app/input/res")

    def _build_model(self, n_chans=129):
        # 2 seconds at 100 Hz => n_times=200
        model = EEGNeX(
            n_chans=n_chans,
            n_outputs=1,
            sfreq=self.sfreq,
            n_times=int(2 * self.sfreq),
        ).to(self.device)
        return model

    def _maybe_load(self, model, filename):
        path = os.path.join(self.res_dir, filename)
        if os.path.isfile(path):
            sd = torch.load(path, map_location=self.device)
            # Strip "module." if saved via DataParallel/DDP
            if any(k.startswith("module.") for k in sd.keys()):
                sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=True)
        # If file not present, we proceed with random init (still valid submission)
        return model

    def get_model_challenge_1(self):
        model = self._build_model(n_chans=129)
        model = self._maybe_load(model, "weights_challenge_1.pt")
        return model

    def get_model_challenge_2(self):
        model = self._build_model(n_chans=129)
        model = self._maybe_load(model, "weights_challenge_2.pt")
        return model
