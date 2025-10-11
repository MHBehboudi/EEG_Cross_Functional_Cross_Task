import torch
import torch.nn as nn


class TinyChEncoder(nn.Module):
  def __init__ (self, in_ch: int, emb_dim: int = 32):
      super().__init__()
      self.conv = nn.Sequential(
                  nn.Conv1d(in_ch,in_ch,kernel_size = 5, padding = 2 , groups = in_ch),
                  nn.BatchNorm1d(in_ch),
                  nn.Dropout(0.1),
                  nn.GELU(),
                  nn.Conv1d(in_ch,in_ch,kernel_size = 5, padding = 2 , groups = in_ch),
                  nn.BatchNorm1d(in_ch),
                  nn.Dropout(0.1),
                  nn.GELU()
      )
      self.lin = nn.Linear(in_ch,emb_dim)
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    #h -> input (B,C,T) Batch, Channels, Time -> Output should become (B,C,D)
    h = self.conv(x) #-> (B,C,T) -> (B,C,T)
    h = h.mean(-1) #-> (B,C,T) -> (B,C)
    h = self.lin(h) # -> (B,C) -> (B,D)
    return h.unsqueze(1).repeat(1,x.shape[1],1)

class TinyChLSTMEncoder(nn.Module):
  def __init__(self, in_ch: int, emb_dim: int = 32):
    super().__init__()
    self.conv = nn.Sequential(
                nn.Conv1d(in_ch, in_ch, padding = 2, kernel_size=5 , groups = in_ch),
                nn.BatchNorm1d(in_ch),
                nn.Dropout(0.2),
                nn.GELU(),
                nn.Conv1d(in_ch, in_ch, padding = 2, kernel_size=5 , groups = in_ch),
                nn.BatchNorm1d(in_ch),
                nn.Dropout(0.2),
                nn.GELU(),
    ) #(B,C,T))
    self.lstm = nn.LSTM(input_size = 1, hidden_size = 16, bidirectional = True)# (C*2*H, T)
    self.lin = nn.Linear(2*16, emb_dim)
    def forward(self,x: torch.Tensor) -> torch.Tensor:
      #input is a time_window (B,C,T)
      B , C, T = x.shape
      h = self.conv(x) #(B,C,T) -> (B,C,T)
      h = h.reshape(B*C,T,1).transpose(0,1) #(B,C,T) -> (B*C ,T, 1)
      _, (h_n, _) = self.lstm(h) # (B*C ,T, 1) -> (2, B*C, H) 
      h = h_n.transpose(0,1).reshape(B*C,-1) # (2, B*C, H) -> (B*C, 2* H)
      h = self.lin(h) # (B*C, 2 * H)   -> (B*C,D)

      return h.reshape(B, C,emb_dim)

def random_crop_pair(x: torch.Tensor, crop_len: int = 150) -> tuple[torch.Tensor, torch.Tensor]:
  # Input is tensor batch [B,C,T]
  T = x.shape[-1]
  if crop_len >= T:
    v1 = v2 = x
  else:
    max_start = T-crop_len+1
    s1 = torch.randint(0,max_start,(1,),device=x.device).item()
    s2 = torch.randint(0,max_start,(1,),device=x.device).item()
    v1 = x[...,s1:s1+crop_len]
    v2 = x[...,s2:s2+crop_len]

return v1,v2

import torch.nn.functional as F
def NT_Xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
  B, C, D = z1.shape
  z1 = z1.reshape(B*C,-1)
  z2 = z2.reshape(B*C,-1)
  z1 = F.normalize(z1, dim = -1)
  z2 = F.normalize(z2, dim = -1)
  reps = torch.cat([z1,z2],dim = 0) # 2 (B*C,D) -> 1 (2*B*C,D)
  sim = reps@reps.T # (2*B*C,D) * (D, 2*B*C)-> (2*B*C,2*B*C)
  N = reps.shape[0]
  mask = torch.eye(N, dtype = torch.bool, device = sim.device)
  sim = sim/tau
  logits = sim.masked_fill(mask,-torch.inf)
  Neff = B*C
  labels = torch.arange(Neff, device = sim.device)
  pos = torch.cat([labels+Neff,labels],dim=0)
  loss = F.cross_entropy(logits,pos)
  return loss

from  torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
def train_ssl_encoder(
  window_ds,
  *,
  epochs: int = 10,
  steps_per_epoch: int = 150,
  batch_size: int = 25,
  crop_len: int = 150,
  device: torch.device,
) -> nn.Module:
  prob = DataLoader(window_ds,batch_size=1, shuffle = True)
  X0 = next(iter(prob))[0] # (1,C,T) -> The [0] is because of the tuple
  C = X0.shape[1]
  enc = TinyChLSTMEncoder(in_ch = C, emb_dim = 32).to(device)
  opt = torch.optim.AdamW(enc.parameters(),lr = 1e-3, weight_decay = 1e-4)
  loader  = DataLoader(window_ds, batch_size = batch_size, shuffle = True, drop_last = True)
  it = iter(loader)
  enc.train()
  for ep in range(1,epochs+1):
    losses = []
    for _ in range(steps_per_epoch):
      try:
        batch = next(it)
      except StopIteration:
        it = iter(loader)
        batch = next(it)

      X = batch[0].to(device).float()
      v1, v2 = random_crop_pair(X, crop_len)
      z1 = enc(v1)
      z2 = enc(v2)
      loss = NT_Xent(z1, z2, tau = 0.2)
      opt.zero_grad(set_to_none=True)
      loss.backward()
      opt.step()
      losses.append(loss.item())
    print(f"[SSL {ep:02d}/{epochs}] contrastive_loss = {np.mean(losses):.4f}")
    
  enc.eval()
  return enc
      




        
        
    


def compute_channel_embeddings (
  windows_ds,
  encoder: nn.Module,
  *,
  batches: int = 64,
  batch_size: int = 16,
  crop_len: int = 150,
  device: torch.device,
) -> np.ndarray:
  
  
      
                
