from torch import nn
import torch
import math
from .gelu import GELU


class FeedForward(nn.Module):
    """
    FeedForward — классический позиционно-независимый блок для Transformer, применяется к каждому токену отдельно.

    Назначение и роль:
    ------------------
    - Реализует двухслойную (или более сложную) нейронную сеть, которая обрабатывает каждый токен ПОРЯДОЧНО независимо (по последней измерении).
    - Дает модели "нелинейную мощность": любой токен может быть переосмыслен вне глобального контекста.
    - После слоя внимания (MHA) FFN помогает связать смысл локальных (внутри токена) “скрытых” значений.

    Архитектурные детали:
    ---------------------
    - Обычно используется блок: (Linear → Activation → Dropout → Linear → Dropout)
    - В современных LLM обычно в 4 раза расширяют скрытый слой (inner_dim = 4 * emb_size).
    - Активация часто GELU или SiLU (Swish), иногда SwiGLU, ReGLU, GeGLU (см. PaLM, Llama).

    Формула (обычная версия):
    -------------------------
        FFN(x) = Linear2(Dropout(Activation(Linear1(x))))
    где Linear1: [emb_size → 4*emb_size], Activation: GELU/SiLU, Linear2: [4*emb_size → emb_size]

    Параметры конструктора:
    -----------------------
    emb_size: int — размерность входа/выхода токена
    inner_dim: int (необязательно) — размер скрытого слоя (по умолчанию 4*emb_size)
    activation: str — тип активации ('gelu', 'silu', 'relu', ...), см. варианты ниже
    dropout: float — dropout после каждой линейной проекции

    Пример использования:
    ---------------------
        >>> ffn = FeedForward(emb_size=256, dropout=0.1, activation='gelu')
        >>> x = torch.randn(2, 32, 256)  # [batch, seq_len, emb_size]
        >>> y = ffn(x)
        >>> print(y.shape)  # torch.Size([2, 32, 256])

    Пояснения:
    ----------
    - FeedForward не использует позицию токена — это МLP, применяемый к каждому токену независимо.
    - Длина последовательности и размер батча не имеют значения (broadcast/reshape по [-2, -1]).
    - Используется во всех декодерах/энкодерах трансформеров.

    Подробнее смотри:
    -----------------
    - Vaswani et al., "Attention is All You Need": https://arxiv.org/abs/1706.03762
    - GELU: https://arxiv.org/abs/1606.08415
    - SwiGLU (PaLM, Llama): https://arxiv.org/abs/2002.05202

    """

    def __init__(self, emb_size: int, dropout: float = 0.1, activation: str = "relu"):
        """
        Инициализация FeedForward блока для трансформера.

        Аргументы:
        ----------
        emb_size: int
            Размерность входного и выходного эмбеддинга модели.
        dropout: float, по умолчанию 0.1
            Dropout после линии и/или активации (уменьшает переобучение).
        activation: str, по умолчанию 'gelu'
            Какая нелинейность использовать ('gelu', 'silu', 'relu' и т.д.).
        inner_dim: int, опционально
            Размер скрытого слоя (по умолчанию 4 * emb_size, как в оригинальном Transformer).

        Внутри:
        -------
        - Задает структуру: Linear → Activation → Dropout → Linear → Dropout.
        """
        super().__init__()
        # Первый линейный слой (расширение размерности)
        self._layer1 = nn.Linear(emb_size, emb_size * 4)
        # ReLU активация
        if activation == "relu":
            self._activation = nn.ReLU()
        elif activation == "gelu":
            self._activation = nn.GELU()
        elif activation == "gelu_exact":
            self._activation = GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        # Второй линейный слой (сжатие обратно)
        self._layer2 = nn.Linear(emb_size * 4, emb_size)
        # Dropout
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Прямой проход через FeedForward блок.
    
        Аргументы:
        ----------
        x : torch.Tensor
            Входной тензор формы [..., emb_size] (используется на каждом токене отдельно!)
    
        Возвращает:
        -----------
        torch.Tensor — выход такой же формы, как вход (только последняя размерность сохраняется).
    
        Пример:
        -------
            >>> ffn = FeedForward(emb_size=256)
            >>> x = torch.randn(8, 16, 256)
            >>> y = ffn(x)
            >>> y.shape  # [8, 16, 256]
        """
        # Сохраняем dtype входных данных
        input_dtype = x.dtype

        # Приводим веса к нужному типу если необходимо
        if input_dtype != self._layer1.weight.dtype:
            self._layer1 = self._layer1.to(dtype=input_dtype)
            self._layer2 = self._layer2.to(dtype=input_dtype)

        # Пропустим тензор x по очереди через все созданные слои
        x = self._layer1(x)
        x = self._activation(x)
        x = self._layer2(x)
        return self._dropout(x)
