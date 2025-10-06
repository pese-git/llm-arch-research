"""
SwiGLU (Swish-Gated Linear Unit) - активационная функция с gating mechanism.

Комбинация Swish активации и Gating Linear Unit. Широко используется в современных
моделях типа LLaMA и PaLM благодаря улучшенной производительности.

Научная статья: "GLU Variants Improve Transformer"
https://arxiv.org/abs/2002.05202

Формула:
SwiGLU(x) = Swish(xW_g + b_g) ⊙ (xW_u + b_u) * W_d + b_d

Преимущества:
- Лучшая производительность чем у ReLU/GELU
- Gating mechanism позволяет модели лучше выбирать информацию
- Хорошо масштабируется для больших моделей
"""

import torch
from torch import nn
from typing import Optional
from .silu import SiLU


class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) — современная нелинейность для архитектур LLM (LLaMA, PaLM).

    Реализация SwiGLU активационной функции.

    Состоит из трех линейных слоев и активации SiLU:
    1. Gate слой + SiLU активация
    2. Up слой (линейное преобразование)
    3. Element-wise multiplication gate и up
    4. Down слой (линейная проекция)

    Научная суть:
        - Сохраняет преимущества GLU (раздельные гейтом и телом) + мощность Swish/SiLU активации.
        - Дает надежную гладкую активацию, хорошо работает на больших масштабах.
        - Статья: "GLU Variants Improve Transformer" (Shazeer, 2020).

        Формула:
            SwiGLU(x) = SiLU(W_g·x) * (W_u·x)
        где SiLU(x) = x*sigma(x)

    Args:
        emb_size (int): размер входов/выходов
        dropout (float): после выходной проекции
    Пример:
        >>> ff = SwiGLU(emb_size=512, dropout=0.1)
        >>> y = ff(torch.randn(2,10,512))
    """

    def __init__(self, emb_size: int, dropout: float = 0.1):
        """
        Инициализация SwiGLU слоя.

        Args:
            emb_size: Размерность входных/выходных эмбеддингов
            dropout: Вероятность dropout (по умолчанию 0.1)
        """
        super().__init__()
        self._gate = nn.Linear(emb_size, 4 * emb_size)
        self._up = nn.Linear(emb_size, 4 * emb_size)
        self._down = nn.Linear(4 * emb_size, emb_size)
        self._activation = SiLU()
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через SwiGLU слой.

        Args:
            x: Входной тензор формы [batch_size, seq_len, emb_size]

        Returns:
            Выходной тензор формы [batch_size, seq_len, emb_size]

        Алгоритм:
        1. gate = SiLU(linear_gate(x))
        2. up = linear_up(x)
        3. output = linear_down(gate ⊙ up)
        4. apply dropout
        """
        # Gate ветвь: линейное преобразование + активация
        gate_out = self._gate(x)  # [batch, seq, 4*emb]
        activation_out = self._activation(gate_out)  # [batch, seq, 4*emb]

        # Up ветвь: линейное преобразование
        up_out = self._up(x)  # [batch, seq, 4*emb]

        # Element-wise multiplication (gating mechanism)
        out = up_out * activation_out  # поэлементное умножение!

        # Final projection and dropout
        out = self._down(out)  # [batch, seq, emb]
        return self._dropout(out)

    def extra_repr(self) -> str:
        """Строковое представление для отладки."""
        return f"emb_size={self._gate.in_features}, dropout={self._dropout.p}"
