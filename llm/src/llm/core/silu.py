import torch
from torch import nn

class SiLU(nn.Module):
    """
    SiLU (Swish) — современная активационная функция для нейросетей.
    
    Научная суть:
        - Формула: $SiLU(x) = x * \sigm(x)$, где $\sigm(x)$ — сигмоида.
        - Более гладкая альтернатива ReLU, улучшает поток градиентов в глубоких сетях.
        - Используется во многих «state-of-the-art» архитектурах (SwiGLU, PaLM, LLaMA).
        - Также известна как Swish (Ramachandran et al, 2017). 
    Пример:
        >>> act = SiLU()
        >>> x = torch.tensor([-1.0, 0.0, 1.0])
        >>> print(act(x))
    """
    def forward(self, x: torch.Tensor):
        return torch.sigmoid(x) * x