import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt

class HeadAttention(nn.Module):
    """
    Реализация одного головного механизма внимания из архитектуры Transformer.
    Выполняет scaled dot-product attention с маскированием будущих позиций (causal attention).
    
    Основной алгоритм:
    1. Линейные преобразования входных данных в Q (query), K (key), V (value)
    2. Вычисление scores = Q·K^T / sqrt(d_k)
    3. Применение causal маски (заполнение -inf будущих позиций)
    4. Softmax для получения весов внимания
    5. Умножение весов на значения V
    
    Пример использования:
    >>> attention = HeadAttention(emb_size=64, head_size=32, max_seq_len=128)
    >>> x = torch.randn(1, 10, 64)  # [batch_size, seq_len, emb_size]
    >>> output = attention(x)  # [1, 10, 32]
    
    Параметры:
        emb_size (int): Размер входного эмбеддинга
        head_size (int): Размерность выхода головы внимания
        max_seq_len (int): Максимальная длина последовательности
    
    Примечания:
    - Использует нижнетреугольную маску для предотвращения "заглядывания в будущее"
    - Автоматически адаптируется к разным версиям PyTorch
    - Поддерживает batch-обработку входных данных
    """
    def __init__(self, emb_size: int, head_size: int, max_seq_len: int):
        super().__init__()
        self._emb_size = emb_size
        self._head_size = head_size
        self._max_seq_len = max_seq_len

        # Линейные преобразования для Q, K, V
        self._k = nn.Linear(emb_size, head_size)
        self._q = nn.Linear(emb_size, head_size)
        self._v = nn.Linear(emb_size, head_size)

        # Создание causal маски
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('_tril_mask', mask.bool() if hasattr(torch, 'bool') else mask.byte())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через слой внимания.
        
        Аргументы:
            x (torch.Tensor): Входной тензор формы [batch_size, seq_len, emb_size]
            
        Возвращает:
            torch.Tensor: Выходной тензор формы [batch_size, seq_len, head_size]
            
        Исключения:
            ValueError: Если длина последовательности превышает max_seq_len
            
        Пример внутренних преобразований:
        Для входа x.shape = [2, 5, 64]:
        1. Q/K/V преобразования -> [2, 5, 32]
        2. Scores = Q·K^T -> [2, 5, 5]
        3. После маски и softmax -> [2, 5, 5]
        4. Умножение на V -> [2, 5, 32]
        """
        seq_len = x.shape[1]
        if seq_len > self._max_seq_len:
            raise ValueError(f"Длина последовательности {seq_len} превышает максимум {self._max_seq_len}")

        # 1. Линейные преобразования
        k = self._k(x)  # [B, T, hs]
        q = self._q(x)  # [B, T, hs]
        
        # 2. Вычисление scores
        scores = q @ k.transpose(-2, -1) / sqrt(self._head_size)
        
        # 3. Применение causal маски
        scores = scores.masked_fill(~self._tril_mask[:seq_len, :seq_len], float('-inf'))
        
        # 4. Softmax и умножение на V
        weights = F.softmax(scores, dim=-1)
        return weights @ self._v(x)