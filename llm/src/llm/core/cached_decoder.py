# llm/src/llm/core/cached_decoder.py

import torch
from torch import nn
from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention
from .rope import RoPE

class CachedDecoder(nn.Module):
    """
    Универсальный декодерный блок с dependency injection для поддержки различных архитектур.
    
    Поддерживает кэширование ключей-значений для ускорения генерации текста.
    
    Args:
        feed_forward_layer: Экземпляр слоя прямого распространения (SwiGLU, FeedForward и т.д.)
        num_heads: Количество голов механизма внимания
        emb_size: Размерность векторных представлений
        head_size: Размерность каждой головы внимания
        max_seq_len: Максимальная длина последовательности
        norm_layer: Класс слоя нормализации (LayerNorm, RMSNorm и т.д.)
        dropout: Вероятность dropout
        rope: Экземпляр RoPE для позиционного кодирования (опционально)
    """
    def __init__(
        self,
        feed_forward_layer: nn.Module,  # Обязательный параметр
        num_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        norm_layer: type = nn.LayerNorm,  # Класс
        dropout: float = 0.1,
        rope: RoPE = None,
    ):
        """
        Инициализация декодера с кэшированием.
        
        Args:
            feed_forward_layer: Слой feed-forward (должен быть экземпляром, а не классом)
            num_heads: Количество голов внимания
            emb_size: Размерность эмбеддингов
            head_size: Размерность каждой головы
            max_seq_len: Максимальная длина последовательности
            norm_layer: Класс нормализации (по умолчанию LayerNorm)
            dropout: Вероятность dropout
            rope: Rotary Positional Embeddings (опционально)
        """
        super().__init__()
        self._heads = MultiHeadAttention(
            num_heads=num_heads,
            emb_size=emb_size,
            head_size=head_size,
            max_seq_len=max_seq_len,
            rope=rope,
            dropout=dropout,
        )
        self._ff = feed_forward_layer
        self._norm1 = norm_layer(emb_size)
        self._norm2 = norm_layer(emb_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        use_cache: bool = True,
        cache: list = None,
    ):
        """
        Прямой проход через декодерный блок.
        
        Args:
            x: Входной тензор формы [batch_size, seq_len, emb_size]
            mask: Маска внимания формы [batch_size, seq_len] (опционально)
            use_cache: Флаг использования кэширования
            cache: Список кэшированных пар (key, value) тензоров
        
        Returns:
            Кортеж (output, new_cache) где:
            - output: Выходной тензор формы [batch_size, seq_len, emb_size]
            - new_cache: Обновленный кэш или None, если use_cache=False
        """
        norm1_out = self._norm1(x)
        # Передаём все cache/use_cache дальше в attention
        attention, kv_caches = self._heads(
            norm1_out, mask=mask, use_cache=use_cache, cache=cache
        )
        out = attention + x
        norm2_out = self._norm2(out)
        ffn_out = self._ff(norm2_out)
        result = ffn_out + out

        if use_cache:
            return (result, kv_caches)
        else:
            return (result, None)