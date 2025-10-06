from torch import nn
import torch
import math
from .gelu import GELU


class FeedForward(nn.Module):
    """
    Классический слой прямого распространения (FeedForward, или FFN) для архитектуры Transformer.

    Этот слой состоит из двух линейных преобразований с расширением внутренней размерности
    в 4 раза и механизмом dropout для регуляризации. Между линейными слоями применяется
    активация ReLU.

    Научная суть:
        - После внимания каждому токену применяется одинаковая двухслойная нейросеть.
        - Дает глубокую нелинейность; позволяет модели не только сопоставлять, но и моделировать сложные связи между токенами.
        - Изначально предложен в «Attention is All You Need» (Vaswani et al., 2017).
        
        Формула:
            FFN(x) = Dropout(W2·act(W1·x))
            где act — ReLU, GELU и др., обычно expansion x4.

    Алгоритм работы:
    1. Входной тензор x (размерность: [batch_size, seq_len, emb_size])
    2. Линейное преобразование: emb_size -> 4*emb_size
    3. Активация ReLU
    4. Линейное преобразование: 4*emb_size -> emb_size
    5. Применение dropout
    6. Возврат результата (размерность: [batch_size, seq_len, emb_size])

    Предназначение:
    - Добавляет нелинейность в архитектуру трансформера
    - Обеспечивает взаимодействие между различными размерностями эмбеддингов
    - Работает независимо для каждого токена в последовательности
    
    Args:
        emb_size (int): размерность входных эмбеддингов
        dropout (float): вероятность(dropout)
        activation (str): нелинейная функция (relu, gelu, gelu_exact)
    
    Пример:
        >>> ff = FeedForward(emb_size=512, dropout=0.1)
        >>> x = torch.randn(32, 10, 512)
        >>> output = ff(x)
        >>> print(output.shape)  # torch.Size([32, 10, 512])
    """
    def __init__(self, emb_size: int, dropout: float = 0.1, activation: str = "relu"):
        """
        Инициализация слоя Feed Forward Network.
        
        Args:
            emb_size: Размерность входных эмбеддингов
            dropout: Вероятность dropout для регуляризации (по умолчанию: 0.1)
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
        Прямой проход через слой Feed Forward Network.
        
        Args:
            x: Входной тензор размерности [batch_size, seq_len, emb_size]
            
        Returns:
            Тензор той же размерности, что и входной
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