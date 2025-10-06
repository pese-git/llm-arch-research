# llm/src/llm/core/cached_decoder.py

import torch
from torch import nn
from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention
from .rope import RoPE

class CachedDecoder(nn.Module):
    """
    Универсальный декодер с поддержкой кэша для autoregressive использования (GPT, LLAMA и пр).
    - Поддерживает использование past_key_values для быстрого генеративного инференса.
    """
    def __init__(
        self,
        feed_forward_layer: nn.Module,  # Обязательный параметр
        num_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        dropout: float = 0.1,
        norm_layer: type = nn.LayerNorm,  # Класс
        rope: RoPE = None,
        activation: str = "gelu",
    ):
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
        x: [batch, seq_len, emb_size]
        mask: (optional)
        use_cache: использовать ли кэширование KV-слоев (инкрементальный генератив, GPT-style)
        cache: список кэшей для голов (или None)
        Возвращает: (output, new_cache) если use_cache=True, иначе (output, None)
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