# llm/src/llm/core/cached_decoder.py

import torch
from torch import nn
from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention
from .rope import RoPE

class CachedDecoder(nn.Module):
    """
    Универсальный декодерный блок для современных LLM (GPT, LLaMA, др.), поддерживает кэширование key-value для эффективной генерации.

    Научная идея:
    Автопагрессивная авторегрессия в трансформерах требует быстрого доступа к ранее вычисленным self-attention ключам/значениям — этот класс позволяет прозрачно кэшировать такие состояния для быстрой инференс-генерации.

    Алгоритм:
        - Input -> LayerNorm -> Многоголовое внимание с кэшем (может быть RoPE)
        - Суммируем residual
        - LayerNorm -> FeedForward (любой, например SwiGLU) -> Residual
        - Возвращается кортеж (output, kvcache)

    Args:
        feed_forward_layer (nn.Module): FeedForward или SwiGLU слой
        num_heads (int): Количество голов внимания
        emb_size (int): Размерность эмбеддингов
        head_size (int): Размерность головы внимания
        max_seq_len (int): Максимальная длина
        norm_layer (тип nn.Module): Normalization слой (LayerNorm или RMSNorm)
        dropout (float): Dropout
        rope (RoPE|None): Экземпляр RoPE (для LLaMA)
    
    Пример (GPT2 style):
        >>> decoder = CachedDecoder(
        ...   feed_forward_layer=FeedForward(...),
        ...   norm_layer=nn.LayerNorm,
        ...   num_heads=4, emb_size=256, head_size=64, max_seq_len=128)
        >>> out, cache = decoder(x, use_cache=True)
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
        Инициализация декодера с кэшированием.
        
        Поведение аналогично блоку TransformerDecoderLayer,
        но с гибкой возможностью подмены любых подкомпонент (активация, norm, позиции).

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
        Прямой проход с поддержкой кэша.
        
        Args:
            x (Tensor[float]): [batch, seq_len, emb_size] — скрытые состояния
            mask (Optional[Tensor]): маска внимания (или causal mask), shape [seq_len, seq_len]
            use_cache (bool): использовать кэширование KV
            cache (list): кэш self-attention для быстрого авторегрессива
        Returns:
            output (Tensor[float]): выходные состояния [batch, seq_len, emb_size]
            kv_caches (list): обновленный кэш, если use_cache
        Пример:
            >>> out, new_cache = decoder(x, use_cache=True, cache=old_cache)
            >>> out.shape # [batch, seq_len, emb_size]
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