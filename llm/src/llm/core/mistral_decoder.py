
import torch
from torch import nn

from llm.core.rms_norm import RMSNorm
from llm.core.swi_glu import SwiGLU
from llm.core.rope import RoPE
from llm.core.group_query_attention import GroupedQueryAttention

class MistralDecoder(nn.Module):
    """
    MistralDecoder — один блок декодера Mistral (модель собирает из них стек в Mistral.__init__).

    Назначение:
    -----------
    Блок включает Grouped Query Attention (GQA) со sliding window, RoPE и SwiGLU feed-forward
    в pre-norm схеме с RMSNorm, как в Mistral 7B.

    Ключевые особенности архитектуры:
    ---------------------------------
    - Использует GQA: для каждого токена вычисляется attention c раздельным числом Q и KV голов (сильно ускоряет LLM).
    - Sliding Window Attention: внимание ограничено окном из window_size элементов (ускоряет обработку длинных текстов).
    - Rotary Positional Embedding (RoPE): позиционная информация интегрируется вращением Q/K.
    - Pre-norm: RMSNorm перед attention и перед FFN, residual-связь вокруг каждого подблока.
    - SwiGLU в качестве нелинейности вместо стандартного GELU (больше capacity в модели).

    Формула работы (псевдокод):
    ---------------------------
        out = x + GQA(RMSNorm1(x))
        result = out + SwiGLU(RMSNorm2(out))

    Аргументы конструктора:
    -----------------------
    num_q_heads, num_kv_heads, emb_size, head_size, max_seq_len, window_size, rope, dropout —
    передаются в GroupedQueryAttention; emb_size и dropout — также в SwiGLU; norm_eps — eps обеих RMSNorm.

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
        dropout: float = 0.1,
        norm_eps: float = 1e-6,
        intermediate_size: int = None,
        bias: bool = True,
    ):
        """
        Инициализация блока MistralDecoder.

        Аргументы:
        ----------
        num_q_heads : int
            Количество Query-heads в attention; должно делиться на num_kv_heads.
        num_kv_heads : int
            Количество Key/Value-heads в attention (их меньше для быстрой генерации).
        emb_size : int
            Размерность embedding.
        head_size : int
            Размер одного attention head (num_q_heads * head_size может отличаться от emb_size).
        max_seq_len : int
            Максимально обрабатываемая длина последовательности.
        window_size : int или None
            Размер окна для sliding window attention (Mistral 7B v0.1 — 4096); None — окна нет (Mistral v0.2+).
        rope : RoPE
            Rotary Positional Embedding для Q/K.
        dropout : float, опционально
            Dropout на каждом attention/FFN (по умолчанию 0.1).
        norm_eps : float, по умолчанию 1e-6
            eps обеих RMSNorm (у Mistral 7B — 1e-5).
        intermediate_size : int, опционально
            Скрытый размер SwiGLU, по умолчанию 4 * emb_size (у Mistral 7B — 14336 = 3.5 * 4096).
        bias : bool, по умолчанию True
            Есть ли bias у всех Linear блока (в Mistral 7B его нет).

        Внутри:
        -------
        - GroupedQueryAttention (с RoPE и sliding window), SwiGLU и две RMSNorm.
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
            dropout=dropout,
            bias=bias,
        )
        self._ff = SwiGLU(emb_size=emb_size, dropout=dropout, hidden_dim=intermediate_size, bias=bias)
        self._norm1 = RMSNorm(emb_size, eps=norm_eps)
        self._norm2 = RMSNorm(emb_size, eps=norm_eps)

    def forward(self, x: torch.Tensor, use_cache: bool = True, cache: tuple = None) -> tuple:
        """
        Прямой проход через блок MistralDecoder.

        Аргументы:
        ----------
        x : torch.Tensor
            Входные эмбеддинги (обычно shape [batch, seq_len, emb_size]).
        use_cache : bool, по умолчанию True
            Включить ли кэширование для ускорения генерации (авторегрессия).
        cache : list, опционально
            KV-кэш этого слоя с предыдущих шагов: (K, V, next_pos), или None.

        Возвращает:
        -----------
        out : torch.Tensor
            Тензор после декодирования (shape соответствует x).
        new_cache : list (или None)
            Новый кэш attention для дальнейшей генерации (или None, если use_cache=False).

        """
        norm1_out = self._norm1(x)
        attention, kv_caches = self._heads(norm1_out, use_cache=use_cache, cache=cache)
        out = attention + x

        norm2_out = self._norm2(out)
        ffn_out = self._ff(norm2_out)

        if use_cache is True:
            return (ffn_out + out, kv_caches)
        else:
            return (ffn_out + out, None)