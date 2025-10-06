import torch
from torch import nn

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor): # [batch_size × seq_len × emb_size]
        return torch.sigmoid(x) * x