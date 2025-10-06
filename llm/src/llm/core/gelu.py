import torch
from torch import nn

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.sqrt_2_over_pi = torch.sqrt(torch.tensor(2.0) / math.pi)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(
            self.sqrt_2_over_pi * (x + 0.044715 * torch.pow(x, 3))
        ))