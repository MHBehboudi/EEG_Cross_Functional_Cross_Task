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
