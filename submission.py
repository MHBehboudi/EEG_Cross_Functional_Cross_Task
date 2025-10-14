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
        # 2 seconds at 100 Hz => 200 timepoints
        m = EEGNeX(n_chans=n_chans, n_outputs=1, sfreq=self.sfreq, n_times=int(2 * self.sfreq))
        return m.to(self.device)

    def _maybe_load(self, model, filename):
        path = os.path.join(self.res_dir, filename)
        if os.path.isfile(path):
            sd = torch.load(path, map_location=self.device)
            # strip "module." if saved via DataParallel/DDP
            if any(k.startswith("module.") for k in sd.keys()):
                sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=True)
        # else: run with random weights (OK for baseline)
        return model

    def get_model_challenge_1(self):
        m = self._build_model(n_chans=129)
        m = self._maybe_load(m, "weights_challenge_1.pt")
        return m

    def get_model_challenge_2(self):
        m = self._build_model(n_chans=129)
        m = self._maybe_load(m, "weights_challenge_2.pt")
        return m
