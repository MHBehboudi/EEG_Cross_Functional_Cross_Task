import torch
from braindecode.models import EEGNeX

def make_eegnex(n_chans: int, sfreq: int, win_sec: float, n_outputs: int = 1):
    model = EEGNeX(
        n_chans=n_chans,
        n_outputs=n_outputs,
        sfreq=sfreq,
        n_times=int(win_sec * sfreq),
    )
    return model
