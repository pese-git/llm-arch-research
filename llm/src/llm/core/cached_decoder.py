# llm/src/llm/core/cached_decoder.py

import torch
from torch import nn
from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention
from .rope import RoPE


class CachedDecoder(nn.Module):
    """
    CachedDecoder — Transformer-декодер с key/value-кэшированием (реализация накладывающегося masked multi-head attention).

    Назначение:
    -----------
    Позволяет быстро и эффективно реализовывать autoregressive генерацию текста в стиле GPT-2/3/4:
    - На шаге генерации используются только нужные токены, “прошлые” key/value значения не пересчитываются, а подаются из кэша.
    - Позволяет значительно ускорять inferece (особенно на длинных последовательностях).
    - Вдохновлено реализациями в HuggingFace transformers, GPT-2/3 и других LLM.

    Архитектурные особенности:
    --------------------------
    - Использует классическую multi-head attention (с causal mask — запрещает видеть “будущее”).
    - Предусматривает передачу и накопление KV-cache для каждого слоя (hidden state attention).
    - Поддерживает передачу внимания через стек attention-блоков.
    - Применяется layernorm и feed-forward block (GELU).

    Параметры конструктора:
    -----------------------
    num_heads : int — число attention heads
    emb_size : int — embedding размерность
    head_size : int — размер каждой attention head (обычно emb_size // num_heads)
    feed_forward_layer : nn.Module — feedforward блок (mLP), может быть любым PyTorch-слоем
    max_seq_len : int — максимально допустимая длина последовательности
    dropout : float — dropout на attention/ffn

    Пример использования:
    ---------------------
        >>> from llm.core.feed_forward import FeedForward
        >>> ff_block = FeedForward(emb_size=256, dropout=0.1, activation=\"gelu\")
        >>> decoder = CachedDecoder(num_heads=4, emb_size=256, head_size=64, feed_forward_layer=ff_block, max_seq_len=2048, dropout=0.1)
        >>> x = torch.randn(2, 100, 256)
        >>> y, kv_cache = decoder(x, use_cache=True, cache=None)
        >>> print(y.shape)  # torch.Size([2, 100, 256])

    Подробнее:
    ----------
    - GPT-2: https://cdn.openai.com/better-language-models/language-models.pdf
    - HuggingFace cache mechanics: https://huggingface.co/docs/transformers/main/en/model_doc/gpt2
    - Объяснения autoregressive cache: https://jalammar.github.io/illustrated-gpt2/

    """

    def __init__(
        self,
        feed_forward_layer: nn.Module,
        num_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        norm_layer: type = nn.LayerNorm,
        dropout: float = 0.1,
        rope: RoPE = None,
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
            Размерность каждой attention head; emb_size = num_heads * head_size должно быть True!
        feed_forward_layer : nn.Module
            Feed-forward слой (например, обычный двухслойный MLP), который применяется после нормы и внимания, и после второй нормы.
        max_seq_len : int
            Максимальная поддерживаемая длина последовательности (выделяет буфер для causal-маски).
        dropout : float, default=0.1
            Dropout после внимания и/или feedforward.
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
        Прямой проход через Decoder Block с поддержкой KV-кэша.

        В этом методе применяется:
        - Causal multi-head attention (masked, не смотрит вперёд)
        - Быстрая обработка длинных последовательностей за счёт сохранения и передачи KV-кэша
        - LayerNorm перед каждым блоком
        - Feed-forward блок и вторая LayerNorm
        - Dropout

        Аргументы:
        ----------
        x : torch.Tensor
            Вход [batch, seq_len, emb_size]
        use_cache : bool, по умолчанию True
            Включать ли накопление и возврат KV-кэша для autoregressive inferece.
        cache : list, опционально
            Список предыдущего KV-кеша для attention.

        Возвращает:
        -----------
        x_ff_out : torch.Tensor
            Результат после attention, модуля и их рез. связей (shape == x)
        new_cache : new KV-cache (или None)

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
