
import torch
from torch import nn

from llm.core.rms_norm import RMSNorm
from llm.core.swi_glu import SwiGLU
from llm.core.rope import RoPE
from llm.core.group_query_attention import GroupedQueryAttention

class MistralDecoder(nn.Module):
    """
    MistralDecoder — стек декодирующих блоков, реализующий архитектуру Mistral-style Transformer.

    Назначение:
    -----------
    Этот класс описывает один или несколько блоков декодера, включающих Grouped Query Attention (GQA),
    sliding window attention и SwiGLU feed-forward, как реализовано в моделях Mistral и Llama 2.

    Ключевые особенности архитектуры:
    ---------------------------------
    - Использует GQA: для каждого токена вычисляется attention c раздельным числом Q и KV голов (сильно ускоряет LLM).
    - Sliding Window Attention: внимание ограничено окном из window_size элементов (ускоряет обработку длинных текстов).
    - Rotary Positional Embedding (RoPE): позиционная информация интегрируется вращением Q/K.
    - RMSNorm перед и после внимания и FFN (устойчивое обучение).
    - SwiGLU в качестве нелинейности вместо стандартного GELU (больше capacity в модели).

    Аргументы конструктора:
    -----------------------
    num_layers : int — сколько блоков-декодеров в стеке
    параметры GQA: num_q_heads, num_kv_heads, emb_size, head_size, max_seq_len, window_size, rope, dropout
    - все они идут в каждый слой (блок) декодера

    Пример использования:
    ---------------------
        >>> decoder = MistralDecoder(
        ...     num_q_heads=8, num_kv_heads=2, emb_size=256, head_size=32,
        ...     max_seq_len=4096, window_size=256, rope=rope, dropout=0.1)
        >>> x = torch.randn(2, 512, 256)
        >>> out, cache = decoder(x)
        >>> print(out.shape)  # torch.Size([2, 512, 256])

    Подробнее:
    ----------
    - Mistral: https://arxiv.org/abs/2310.06825
    - Llama 2: https://arxiv.org/abs/2307.09288
    - Open LLM обзор: https://huggingface.co/blog/mistral

    """
    def __init__(self, 
        num_q_heads: int,
        num_kv_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        window_size: int,
        rope: RoPE,
        dropout: float = 0.1
    ):
        """
        Инициализация стека декодеров MistralDecoder.

        Аргументы:
        ----------
        num_layers : int
            Сколько слоёв (декодеров/GQA-блоков) собрать в стек.
        num_q_heads : int
            Количество Query-heads в attention (их больше, экономит память).
        num_kv_heads : int
            Количество Key/Value-heads в attention (их меньше для быстрой генерации).
        emb_size : int
            Размерность embedding (должна делиться на num_q_heads без остатка).
        head_size : int
            Размер одного attention head.
        max_seq_len : int
            Максимально обрабатываемая длина последовательности.
        window_size : int
            Размер окна для sliding window attention.
        rope : RoPE
            Rotary Positional Embedding для Q/K.
        dropout : float, опционально
            Dropout на каждом attention/FFN (по умолчанию 0.1).

        Внутри:
        -------
        - Собираются num_layers Sequential-блоков из GQA + SwiGLU + RMSNorm.
        - Все параметры передаются в каждый слой (блок).
        """
        super().__init__()
        self._heads = GroupedQueryAttention(
            num_q_heads=num_q_heads, 
            num_kv_heads=num_kv_heads,
            emb_size=emb_size, 
            head_size=head_size, 
            max_seq_len=max_seq_len,
            window_size=window_size,
            rope=rope,
            dropout=dropout
        )
        self._ff = SwiGLU(emb_size=emb_size, dropout=dropout)
        self._norm1 = RMSNorm(emb_size)
        self._norm2 = RMSNorm(emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, use_cache: bool = True, cache: list = None) -> torch.Tensor:
        """
        Прямой проход через стек MistralDecoder.

        Аргументы:
        ----------
        x : torch.Tensor
            Входные эмбеддинги (обычно shape [batch, seq_len, emb_size]).
        use_cache : bool, по умолчанию True
            Включить ли кэширование для ускорения генерации (авторегрессия).
        cache : list, опционально
            Предыдущий кеш attention-блоков (или None).

        Возвращает:
        -----------
        out : torch.Tensor
            Тензор после декодирования (shape соответствует x).
        new_cache : list (или None)
            Новый кэш attention для дальнейшей генерации (или None, если use_cache=False).

        """
        norm1_out = self._norm1(x)
        attention, kv_caches = self._heads(norm1_out, mask, use_cache=use_cache, cache=cache)
        out = attention + x
        
        norm2_out = self._norm2(out)
        ffn_out = self._ff(norm2_out)

        if use_cache is True:
            return (ffn_out + out, kv_caches)
        else:
            return (ffn_out + out, None)