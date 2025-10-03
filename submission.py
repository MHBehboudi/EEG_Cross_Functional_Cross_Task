# submission.py
import torch
from braindecode.models import EEGNeX

# Cap threads for Codabench CPU workers
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def _make(self):
        # Challenge spec: 129 chans, 2s @ SFREQ
        model = EEGNeX(
            n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2 * self.sfreq)
        ).to(self.device)
        model.eval()
        return model

    def get_model_challenge_1(self):
        m = self._make()
        state = torch.load("/app/output/weights_challenge_1.pt", map_location=self.device)
        m.load_state_dict(state, strict=True)
        return m

    def get_model_challenge_2(self):
        m = self._make()
        state = torch.load("/app/output/weights_challenge_2.pt", map_location=self.device)
        m.load_state_dict(state, strict=True)
        return m
