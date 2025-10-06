import torch
from torch import nn
from .silu import SiLU

class SwiGLU(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1):
        super().__init__()

        self._gate = nn.Linear(emb_size, 4 * emb_size)
        self._up = nn.Linear(emb_size, 4 * emb_size)
        self._down = nn.Linear(4 * emb_size, emb_size)
        self._activation = SiLU()
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor): # [batch_size × seq_len × emb_size].
        gate_out = self._gate(x)                          # [batch, seq, 4*emb]
        activation_out = self._activation(gate_out)       # [batch, seq, 4*emb]
        up_out = self._up(x)                              # [batch, seq, 4*emb]
        out = up_out * activation_out                     # поэлементное!
        out = self._down(out)                             # [batch, seq, emb]
        return self._dropout(out)