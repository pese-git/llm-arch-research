# llm/src/llm/core/cached_decoder.py

from typing import Callable

import torch
from torch import nn
from .multi_head_attention import MultiHeadAttention
from .rope import RoPE


class CachedDecoder(nn.Module):
    """
    CachedDecoder — pre-norm блок декодера с KV-кэшем и подставляемыми нормализацией и FFN.

    Назначение:
    -----------
    Общий блок, из которого собирается декодер LLaMA: attention — всегда MultiHeadAttention
    (опционально с RoPE), а нормализация и feed-forward передаются в конструктор.
    - LLaMA: norm_layer=RMSNorm, feed_forward_layer=SwiGLU, rope=RoPE(...).
    - По умолчанию norm_layer=nn.LayerNorm — тогда это классический pre-LN блок.

    Формула работы (псевдокод):
    ---------------------------
        out = x + Attention(Norm1(x))
        result = out + FFN(Norm2(out))

    Архитектурные особенности:
    --------------------------
    - Встроенная causal-маска; паддинг батча (padding, см. core/padding.py) передаётся в attention.
    - KV-кэш каждого слоя: при генерации K/V прошлых токенов не пересчитываются.

    Параметры конструктора:
    -----------------------
    feed_forward_layer : nn.Module — FFN-блок (для LLaMA — SwiGLU)
    num_heads : int — число attention heads
    emb_size : int — embedding размерность
    head_size : int — размер каждой attention head (обычно emb_size // num_heads)
    max_seq_len : int — максимально допустимая длина последовательности
    norm_layer : callable — класс или фабрика нормализации, вызывается как norm_layer(emb_size)
                 (nn.LayerNorm по умолчанию; LLaMA передаёт functools.partial(RMSNorm, eps=...))
    dropout : float — dropout в attention
    rope : RoPE — rotary positional encoding для Q и K (для LLaMA)

    Пример использования:
    ---------------------
        >>> from llm.core.swi_glu import SwiGLU
        >>> from llm.core.rms_norm import RMSNorm
        >>> from llm.core.rope import RoPE
        >>> decoder = CachedDecoder(
        ...     feed_forward_layer=SwiGLU(emb_size=256, dropout=0.1), num_heads=4, emb_size=256,
        ...     head_size=64, max_seq_len=2048, norm_layer=RMSNorm, rope=RoPE(64, 2048))
        >>> x = torch.randn(2, 100, 256)
        >>> y, kv_cache = decoder(x, use_cache=True, cache=None)
        >>> print(y.shape)  # torch.Size([2, 100, 256])

    Подробнее:
    ----------
    - LLaMA: https://arxiv.org/abs/2302.13971
    - Объяснения autoregressive cache: https://jalammar.github.io/illustrated-gpt2/

    """

    def __init__(
        self,
        feed_forward_layer: nn.Module,
        num_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        dropout: float = 0.1,
        rope: RoPE = None,
        bias: bool = True,
    ):
        """
        Конструктор CachedDecoder.

        Аргументы:
        ----------
        num_heads : int
            Сколько attention heads используется в каждом attention слое.
        emb_size : int
            Размерность входного вектора x.
        head_size : int
            Размерность каждой attention head (обычно emb_size // num_heads).
        feed_forward_layer : nn.Module
            Feed-forward слой (например, обычный двухслойный MLP), который применяется после нормы и внимания, и после второй нормы.
        max_seq_len : int
            Максимальная поддерживаемая длина последовательности (выделяет буфер для causal-маски).
        dropout : float, default=0.1
            Dropout после внимания и/или feedforward.
        bias : bool, default=True
            Есть ли bias у Q/K/V и выходной проекции attention (в LLaMA его нет).
        """
        super().__init__()
        self._heads = MultiHeadAttention(
            num_heads=num_heads,
            emb_size=emb_size,
            head_size=head_size,
            max_seq_len=max_seq_len,
            rope=rope,
            dropout=dropout,
            bias=bias,
        )
        self._ff = feed_forward_layer
        self._norm1 = norm_layer(emb_size)
        self._norm2 = norm_layer(emb_size)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = True,
        cache: list = None,
        padding=None,
    ):
        """
        Прямой проход через Decoder Block с поддержкой KV-кэша.

        В этом методе применяется:
        - Causal multi-head attention (masked, не смотрит вперёд)
        - Быстрая обработка длинных последовательностей за счёт сохранения и передачи KV-кэша
        - Нормализация (norm_layer) перед каждым подблоком
        - Feed-forward блок (feed_forward_layer)
        - Dropout

        Аргументы:
        ----------
        x : torch.Tensor
            Вход [batch, seq_len, emb_size]
        use_cache : bool, по умолчанию True
            Включать ли накопление и возврат KV-кэша для autoregressive inferece.
        cache : list, опционально
            Список предыдущего KV-кеша для attention.
        padding : Padding, опционально
            Паддинг батча (core/padding.py) — передаётся в attention; None — без паддинга.

        Возвращает:
        -----------
        x_ff_out : torch.Tensor
            Результат после attention, модуля и их рез. связей (shape == x)
        new_cache : new KV-cache (или None)

        """
        norm1_out = self._norm1(x)
        # Передаём все cache/use_cache дальше в attention
        attention, kv_caches = self._heads(norm1_out, use_cache=use_cache, cache=cache, padding=padding)
        out = attention + x
        norm2_out = self._norm2(out)
        ffn_out = self._ff(norm2_out)
        result = ffn_out + out

        if use_cache:
            return (result, kv_caches)
        else:
            return (result, None)
