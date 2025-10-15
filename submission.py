# submission.py
import os
import torch
import torch.nn as nn
from braindecode.models import EEGNeX

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try:
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
except Exception:
    pass

RES_DIR = os.environ.get("EEG2025_RES_DIR", "/app/input/res")
WIN_SEC = 2

class TemporalTransformerRegressor(nn.Module):
    def __init__(self, in_chans: int, sfreq: int = 100, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Conv1d(in_chans, d_model, kernel_size=1, bias=False)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model,128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128,1))
    def forward(self, x):
        h = self.stem(x); h = h.transpose(1,2); h = self.encoder(h); h = h.mean(dim=1); return self.head(h)

class ChannelGCNRegressor(nn.Module):
    def __init__(self, in_chans: int, sfreq: int = 100, m: int = 8, d_time: int = 32, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.m = m
        self.dw = nn.Conv1d(in_chans, in_chans*m, kernel_size=7, padding=3, groups=in_chans, bias=False)
        self.proj_node = nn.Linear(m, d_time, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gcn1 = nn.Linear(d_time, hidden, bias=False)
        self.gcn2 = nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(hidden,128), nn.GELU(), nn.Linear(128,1))
        self.register_buffer("adj", torch.eye(in_chans))
    def _node_features(self, x):
        h = self.dw(x); h = torch.nn.functional.gelu(h); h = torch.nn.functional.adaptive_avg_pool1d(h, 1)
        B, CM, _ = h.shape; C = self.adj.shape[0]; h = h.view(B, C, self.m); return self.proj_node(h)
    def _gcn_layer(self, X, lin):
        H = torch.matmul(self.adj, X); H = lin(H); return torch.nn.functional.gelu(H)
    def forward(self, x):
        X = self._node_features(x); H = self._gcn_layer(X, self.gcn1); H = self.dropout(H); H = self._gcn_layer(H, self.gcn2)
        H = H.mean(dim=1); return self.head(H)

class ClusteredEEGNeX(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, sfreq: int):
        super().__init__()
        self.projector = nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False)
        self.backbone  = EEGNeX(n_chans=out_chans, n_outputs=1, sfreq=sfreq, n_times=int(2 * sfreq))
    def forward(self, x): return self.backbone(self.projector(x))

def _unwrap_payload(payload):
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload
    return {"state_dict": payload, "arch": None}

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def _load(self, fname: str) -> nn.Module:
        path = os.path.join(RES_DIR, fname)
        payload = torch.load(path, map_location=self.device)
        payload = _unwrap_payload(payload)
        sd = payload["state_dict"]; arch = payload.get("arch", None)
        keys = set(sd.keys()); has = lambda k: any(x.startswith(k) for x in keys)

        # Transformer (with or without projector)
        if arch == "transformer" or has("stem.weight"):
            if "projector.weight" in sd:
                out_c, in_c, _ = sd["projector.weight"].shape
                proj = nn.Conv1d(in_c, out_c, kernel_size=1, bias=False)
                model = nn.Sequential(proj, TemporalTransformerRegressor(in_chans=out_c, sfreq=self.sfreq)).to(self.device)
                model.load_state_dict(sd, strict=True); return model
            model = TemporalTransformerRegressor(in_chans=129, sfreq=self.sfreq).to(self.device)
            model.load_state_dict(sd, strict=True); return model

        # GNN
        if arch == "gnn" or has("adj") or has("gcn1.weight"):
            model = ChannelGCNRegressor(in_chans=129, sfreq=self.sfreq).to(self.device)
            model.load_state_dict(sd, strict=True); return model

        # EEGNeX (with/without projector)
        if "projector.weight" in sd:
            out_c, in_c, _ = sd["projector.weight"].shape
            model = ClusteredEEGNeX(in_chans=in_c, out_chans=out_c, sfreq=self.sfreq).to(self.device)
        else:
            model = EEGNeX(n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2 * self.sfreq)).to(self.device)
        model.load_state_dict(sd, strict=True); return model

    def get_model_challenge_1(self):
        m = self._load("weights_challenge_1.pt"); m.eval(); return m

    def get_model_challenge_2(self):
        m = self._load("weights_challenge_2.pt"); m.eval(); return m
