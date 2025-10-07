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

def compute_channel_embeddings (

      
                
