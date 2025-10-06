"""
RMSNorm (Root Mean Square Normalization) - нормализация по среднеквадратичному значению.

Упрощенная версия LayerNorm без вычисления среднего значения. Широко используется
в современных архитектурах типа LLaMA благодаря лучшей стабильности и производительности.

Научная статья: "Root Mean Square Layer Normalization"
https://arxiv.org/abs/1910.07467

Формула:
RMSNorm(x) = (x / RMS(x)) * w
где RMS(x) = sqrt(mean(x²) + eps)

Преимущества:
- Меньше вычислений (нет вычитания среднего)
- Лучшая стабильность при обучении
- Сохранение масштаба сигнала
"""

import torch
from torch import nn
from typing import Optional


class RMSNorm(nn.Module):
    """
    RMS Normalization (Root Mean Square Layer Normalization).

    Нормализует входные данные по последнему измерению используя среднеквадратичное
    значение вместо среднего, как в стандартном LayerNorm.

    Научная суть:
        - Упрощенный вариант LayerNorm без вычисления среднего, только деление на rms.
        - Лучшая численная стабильность на больших моделях, меньше вычислений.
        - Применяется в LLaMA, PaLM и др.

        Формула:
            RMSNorm(x) = (x / sqrt(mean(x²) + eps)) * w   (w — обучаемый вектор)

    Args:
        dim (int): размер последнего измерения (обычно emb_size)
        eps (float): для численной устойчивости

    Пример:
        >>> norm = RMSNorm(emb_size)
        >>> out = norm(x)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Инициализация RMSNorm слоя.

        Args:
            dim: Размерность нормализуемого измерения
            eps: Малое значение для численной стабильности (по умолчанию 1e-6)
        """
        super().__init__()
        self._eps = eps
        self._w = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через RMSNorm слой.

        Args:
            x: Входной тензор формы [..., dim]

        Returns:
            Нормализованный тензор той же формы, что и входной

        Формула:
        output = w * (x / sqrt(mean(x²) + eps))
        """
        # Вычисление RMS (Root Mean Square) по последнему измерению
        rms = (x.pow(2).mean(-1, keepdim=True) + self._eps) ** 0.5

        # Нормализация и масштабирование
        norm_x = x / rms
        return self._w * norm_x

    def extra_repr(self) -> str:
        """Строковое представление для отладки."""
        return f"dim={self._w.shape[0]}, eps={self._eps}"
