import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self._eps = eps
        self._w = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor): # [batch_size × seq_len × emb_size]
        rms = (x.pow(2).mean(-1, keepdim=True) + self._eps) ** 0.5
        norm_x = x / rms
        return self._w * norm_x