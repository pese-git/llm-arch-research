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
    RMSNorm (Root Mean Square Layer Normalization) — простая и эффективная альтернатива LayerNorm.

    Назначение:
    -----------
    - Нормализует входной тензор по последнему измерению только с помощью RMS (root mean square), без вычитания среднего.
    - Используется в LLaMA, PaLM и других крупных языковых моделях для лучшей стабильности и ускорения обучения.
    - В отличие от LayerNorm, не центрирует значения, что особенно полезно для автогерессивных трансформеров с residual-связями.

    Мотивация и математика:
    -----------------------
    - Формула для одного слоя и вектора x:
        rms = sqrt( mean( x ** 2 ) + eps )
        out = w * ( x / rms )
      где w — learnable scale, eps — небольшая константа для численной устойчивости.
    - Нет смещения/вычитания среднего — сигнал сохраняет абсолютные значения, меньше “искажает” автоагрегатные значения на накопленных резидуалах.

    Аргументы конструктора:
    -----------------------
    dim : int
        Размер последнего нормализуемого измерения (обычно совпадает с размером embedding/final head).
    eps : float, default=1e-6
        Малое значение для устойчивости (additive epsilon).

    Особенности:
    ------------
    - Нет батч-нормализации, нет зависимости от размера батча.
    - Отлично подходит для больших моделей и автогерессии — меньше шуму от residual.

    Пример использования:
    ---------------------
        >>> norm = RMSNorm(emb_size=256)
        >>> x = torch.randn(4, 10, 256)
        >>> out = norm(x)  # возвращает tensor той же формы

    References:
    -----------
    - Zhang & Sennrich, "Root Mean Square Layer Normalization", 2019: https://arxiv.org/abs/1910.07467
    - Применение в LLaMA: https://arxiv.org/abs/2302.13971
    - HuggingFace implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Инициализация RMSNorm.

        Args:
        -----
        dim : int
            Последнее нормализуемое измерение (обычно размерность embedding или hidden).
        eps : float
            Малое значение для устойчивости (по умолчанию 1e-6).

        Внутри:
        -------
        - Создаётся обучаемый scale weight w для каждой компоненты dim.
        - Сохраняется параметр eps для добавления к RMS.
        """
        super().__init__()
        self._eps = eps
        self._w = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через RMSNorm.
    
        Args:
        -----
        x : torch.Tensor
            Входной тензор любого shape с последней размерностью dim.
    
        Returns:
        --------
        torch.Tensor — тот же shape, что и вход x, но нормализованный по RMS на последнем измерении.
    
        Алгоритм:
        ---------
        - Вычислить rms = sqrt( mean( x**2, dim=-1, keepdim=True ) + eps )
        - Поделить x на rms
        - Помасштабировать обучаемым весом w
    
        Пример:
        -------
            >>> norm = RMSNorm(256)
            >>> out = norm(torch.randn(2, 10, 256))
    
        """
        # Вычисление RMS (Root Mean Square) по последнему измерению
        rms = (x.pow(2).mean(-1, keepdim=True) + self._eps) ** 0.5

        # Нормализация и масштабирование
        norm_x = x / rms
        return self._w * norm_x

    def extra_repr(self) -> str:
        """Строковое представление для отладки."""
        return f"dim={self._w.shape[0]}, eps={self._eps}"
