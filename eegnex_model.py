from braindecode.models import EEGNeX

def build_eegnex(*, sfreq=100, n_chans=129, n_times=200, n_outputs=1, device=None):
    model = EEGNeX(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, sfreq=sfreq)
    if device is not None:
        model = model.to(device)
    return model
