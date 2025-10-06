import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
from .rope import RoPE


class HeadAttention(nn.Module):
    """
    Одноголовый механизм внимания (scaled dot-product attention) — фундаментальный строительный блок всех современных Transformer.

    Научная суть:
        - Attention учит модель самостоятельно "выбирать" важные связи между словами, независимо от их положения.
        - Механизм causal mask гарантирует невозможность "заглядывания в будущее" при генерации (авторегрессия).

        Формула:
            Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V
            (Q — запросы, K — ключи, V — значения; d_k — размерность ключа)

        Поддерживает Rotary Position Encoding (RoPE) для относительного позиционного кодирования.

    Args:
        emb_size (int): размер входного эмбеддинга
        head_size (int): размерность attention-головы
        max_seq_len (int): максимальная длина последовательности
        rope (RoPE, optional): экземпляр RoPE для позиций

    Примечания:
    - Использует нижнетреугольную маску для предотвращения "заглядывания в будущее"
    - Автоматически адаптируется к разным версиям PyTorch
    - Поддерживает batch-обработку входных данных

    Пример использования:
        >>> attention = HeadAttention(emb_size=64, head_size=32, max_seq_len=128)
        >>> x = torch.randn(1, 10, 64)
        >>> output, _ = attention(x)
        >>> print(output.shape)  # torch.Size([1, 10, 32])
    """

    def __init__(
        self, emb_size: int, head_size: int, max_seq_len: int, rope: RoPE = None
    ):
        super().__init__()
        self._emb_size = emb_size
        self._head_size = head_size
        self._max_seq_len = max_seq_len
        self._rope = rope

        # Линейные преобразования для Q, K, V
        self._k = nn.Linear(emb_size, head_size)
        self._q = nn.Linear(emb_size, head_size)
        self._v = nn.Linear(emb_size, head_size)

        # Создание causal маски
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer(
            "_tril_mask", mask.bool() if hasattr(torch, "bool") else mask.byte()
        )

    def forward(
        self, x: torch.Tensor, use_cache: bool = True, cache: tuple = None
    ) -> tuple:
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
            raise ValueError(
                f"Длина последовательности {seq_len} превышает максимум {self._max_seq_len}"
            )

        k = self._k(x)  # [B, T, hs]
        q = self._q(x)  # [B, T, hs]
        v = self._v(x)  # [B, T, hs]

        if self._rope is not None:
            # ✅ Применяем RoPE к Q и K (НЕ к V!)
            q = self._rope(q)  # [B, T, hs]
            k = self._rope(k)  # [B, T, hs]

        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=1)  # [B, cache_len + T, hs]
            v = torch.cat([v_cache, v], dim=1)  # [B, cache_len + T, hs]

        scores = q @ k.transpose(-2, -1) / sqrt(self._head_size)

        if cache is None:
            scores = scores.masked_fill(
                ~self._tril_mask[:seq_len, :seq_len], float("-inf")
            )

        weights = F.softmax(scores, dim=-1)
        x_out = weights @ v  # [B, T, hs]

        if use_cache is True:
            return (x_out, (k, v))
        else:
            return (x_out, None)
