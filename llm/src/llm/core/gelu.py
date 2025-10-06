import torch
from torch import nn


class GELU(nn.Module):
    """
    Гауссовская Эрф-активация (GELU, Gaussian Error Linear Unit).

    Научная суть:
        - Одна из самых популярных smooth активаций для трансформеров.
        - Дает более гибкие аппроксимации, чем ReLU/SiLU, улучшает flow градиентов для больших LLM.
        - Используется в BERT, GPT, GPT2 и почти всех современных NLP-моделях.
        Формула:
            GELU(x) = 0.5 * x * (1 + tanh(\sqrt{2/π} * (x + 0.044715 x³)))
        Подробнее: Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415
    Пример:
        >>> gelu = GELU()
        >>> y = gelu(torch.tensor([-1.0, 0.0, 1.0]))
        >>> print(y)
    """

    def __init__(self):
        super().__init__()
        self.sqrt_2_over_pi = torch.sqrt(torch.tensor(2.0) / math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (1 + torch.tanh(self.sqrt_2_over_pi * (x + 0.044715 * torch.pow(x, 3))))
        )
