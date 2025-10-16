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
    SwiGLU (Swish-Gated Linear Unit) — эффективная feed-forward нелинейность для трансформеров (LLAMA, PaLM, Mistral).

    Назначение:
    -----------
    - Улучшает классический блок FeedForward (FFN) в трансформерах за счёт \"gating\" (механизма управления информационным потоком).
    - Использует нелинейность SiLU (Swish) вместо ReLU или GELU, повышая capacity блока.
    - Является дефолтом во всех современных LLM, начиная с PaLM, LLaMA и Mistral.

    Формула и математика:
    ---------------------
    Пусть x — вход, then:

        SwiGLU(x) = (SiLU(W_g x + b_g)) ⊙ (W_u x + b_u) W_d + b_d

    Типовая реализация (как здесь, по LLAMA/Mistral):
        gate = SiLU(Linear_gate(x))        # фитчерный \"gate\"
        up   = Linear_up(x)                # пропускная ветка
        mult = gate * up                   # поэлементное умножение (контроль информации)
        out  = Linear_down(mult)           # финальная проекция
        out  = Dropout(out)                # регуляризация

    Почему это работает:
    -------------------
    - Gating позволяет информации проходить \"частично\", динамически подавляя/усиливая сигналы в hidden-space.
    - SiLU обеспечивает smooth градиенты (лучше для обучения LLM).
    - В экспериментах (PaLM, LLAMA) SwiGLU consistently outperforms ReLU, GELU, обычные GLU.

    Параметры конструктора:
    -----------------------
    emb_size: int
        Размерность входного (и выходного) признакового пространства.
    dropout: float
        Dropout после final linear (обычно около 0.1).

    Пример использования:
    ---------------------
        >>> block = SwiGLU(emb_size=512, dropout=0.1)
        >>> x = torch.randn(8, 16, 512)
        >>> y = block(x)
        >>> print(y.shape)  # torch.Size([8, 16, 512])

    References:
    -----------
    - Shazeer, \"GLU Variants Improve Transformer\", 2020: https://arxiv.org/abs/2002.05202
    - PaLM: https://arxiv.org/abs/2204.02311 (Section 4.1)
    - LLaMA: https://arxiv.org/abs/2302.13971
    - Mistral: https://arxiv.org/abs/2310.06825
    - HuggingFace discussion: https://huggingface.co/docs/transformers/main/en/model_doc/llama

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
        Прямой проход через блок SwiGLU.

        Args:
        -----
        x : torch.Tensor
            Входной тензор формы [batch_size, seq_len, emb_size]

        Returns:
        --------
        torch.Tensor той же формы

        Алгоритм:
        ---------
        1. gate = SiLU(linear_gate(x))
        2. up = linear_up(x)
        3. mult = gate * up  # поэлементно
        4. out = linear_down(mult)
        5. out = dropout(out)
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
