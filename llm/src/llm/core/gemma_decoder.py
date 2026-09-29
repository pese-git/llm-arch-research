import torch
from torch import nn
from llm.core.rope import RoPE
from llm.core.group_query_attention import GroupedQueryAttention
from llm.core.rms_norm import RMSNorm
from llm.core.geglu import GeGLU

class GemmaDecoder(nn.Module):
    """
    GemmaDecoder — декодерный блок архитектуры Gemma (Google DeepMind, 2024).

    Назначение:
    -----------
    Одна «ячейка» декодерного стека Gemma: pre-norm блок с RMSNorm, Multi-Query Attention
    (одна общая K/V-голова, как в Gemma 2B) с RoPE и GeGLU feed-forward.

    Архитектурные компоненты:
    -------------------------
    - RMSNorm перед attention и перед FFN
    - Multi-Query Attention с RoPE и KV-кэшем
    - GeGLU feed-forward (GELU-gated MLP)
    - Residual-связь вокруг каждого подблока
    - Dropout в attention и FFN (в оригинальной Gemma его нет, см. docs/gemma.md)

    Алгоритм прямого прохода:
    -------------------------
        1. norm1_out = RMSNorm1(x)
        2. attention_out = MQA(norm1_out)
        3. resid1 = attention_out + x
        4. norm2_out = RMSNorm2(resid1)
        5. ffn_out = GeGLU(norm2_out)
        6. output = ffn_out + resid1

    Аргументы конструктора:
    ----------------------
    num_q_heads : int
        Число голов query (K/V-голова всегда одна).
    emb_size : int
        Размерность скрытого пространства (embedding dim).
    head_size : int
        Размерность одной attention-головы.
    max_seq_len : int
        Максимальная длина последовательности (размер causal-маски).
    rope : RoPE
        Позиционное кодирование Rotary Position Embedding.
    dropout : float, optional
        Dropout для регуляризации (примерно 0.0–0.1).
    norm_eps : float, optional
        eps обеих RMSNorm (по умолчанию 1e-6, как в Gemma).

    Пример использования:
    ---------------------
        >>> decoder = GemmaDecoder(
        ...     num_q_heads=8,
        ...     emb_size=256,
        ...     head_size=32,
        ...     max_seq_len=1024,
        ...     rope=RoPE(32, 1024),
        ...     dropout=0.1,
        ... )
        >>> x = torch.randn(2, 24, 256)
        >>> out, cache = decoder(x, use_cache=True, cache=None)
        >>> print(out.shape)  # torch.Size([2, 24, 256])

    Литература и ссылки:
    --------------------
    - Gemma (официальный релиз): https://ai.google.dev/gemma
    - Gemma paper: https://arxiv.org/abs/2403.08295
    - Rotary Embedding: https://arxiv.org/abs/2104.09864
    - Multi-Query Attention: https://arxiv.org/abs/1911.02150
    """
    def __init__(self, 
        num_q_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        rope: RoPE,
        dropout: float = 0.1,
        norm_eps: float = 1e-6,
        num_kv_heads: int = 1,
        intermediate_size: int = None,
        bias: bool = True,
    ):
        """
        Конструктор слоя GemmaDecoder.

        Создаёт подслои блока: две RMSNorm, Multi-Query Attention с RoPE и GeGLU.

        Аргументы:
        ----------
        num_q_heads : int
            Количество query-голов в attention (определяет степень параллелизма внимания).
        emb_size : int
            Размер пространства эмбеддинга (embedding dim, input/output размерность слоя).
        head_size : int
            Размерность одной attention-головы. Обычно emb_size // num_q_heads.
        max_seq_len : int
            Максимальная длина последовательности, для которой поддерживается attention и маскирование.
        rope : RoPE
            Объект для rotary positional encoding (позиционное кодирование для attention).
        dropout : float, default=0.1
            Dropout после attention и feed-forward для регуляризации (обычно 0.0–0.1).
        norm_eps : float, default=1e-6
            eps обеих RMSNorm.
        num_kv_heads : int, default=1
            Число K/V-голов: 1 — Multi-Query Attention, как в Gemma 2B; num_q_heads — обычный MHA,
            как в Gemma 7B. Внимание — GroupedQueryAttention без скользящего окна.
        intermediate_size : int, опционально
            Скрытый размер GeGLU, по умолчанию 4 * emb_size (в Gemma — 8 * emb_size).
        bias : bool, default=True
            Есть ли bias у всех Linear блока (в Gemma его нет).

        Внутри:
        -------
        - MultiQueryAttention (со своей causal-маской), GeGLU, RMSNorm ×2.

        Пример:
        -------
            >>> decoder = GemmaDecoder(
            ...     num_q_heads=8, emb_size=512, head_size=64, max_seq_len=1024, rope=rope_obj, dropout=0.05
            ... )
        """
        super().__init__()
        self._heads = GroupedQueryAttention(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            emb_size=emb_size,
            head_size=head_size,
            max_seq_len=max_seq_len,
            rope=rope,
            dropout=dropout,
            bias=bias,
        )
        self._ff = GeGLU(emb_size=emb_size, dropout=dropout, hidden_dim=intermediate_size, bias=bias)
        self._norm1 = RMSNorm(emb_size, eps=norm_eps)
        self._norm2 = RMSNorm(emb_size, eps=norm_eps)

    def forward(self, x: torch.Tensor, use_cache: bool = True, cache: tuple = None) -> tuple:
        """
        Прямой проход (forward) через GemmaDecoder.

        Последовательно реализует:
        - Нормализацию входа (RMSNorm)
        - Multi-Query self-attention со встроенной causal-маской и кэшем
        - Остаточное сложение (skip connection)
        - Вторую нормализацию
        - Feed-Forward-блок GeGLU
        - Ещё одно residual сложение

        Поддерживает autoregressive режим с caching (KV-слоты attention для ускорения генерации).

        Аргументы:
        ----------
        x : torch.Tensor
            Входной скрытый тензор формы [batch_size, seq_length, emb_size].
        use_cache : bool, по умолчанию True
            Если True — возвращается кэш KV для ускорения autoregressive генерации.
        cache : list, optional
            Кэш предыдущих ключей/значений attention (если используется при инференсе).

        Возвращает:
        -----------
        Tuple[torch.Tensor, cache]:
            - Выход декодера с той же формой [batch_size, seq_length, emb_size]
            - Кэш attention (если use_cache=True), иначе None

        Пример:
        -------
            >>> out, new_cache = decoder(x, use_cache=True, cache=old_cache)
            >>> out.shape  # [batch_size, seq_len, emb_size]

        Примечания:
        -----------
        - Паддинг (attention_mask) проверяется в forward модели; блок применяет только встроенную causal-маску.
        - Для ускорения в режиме генерации рекомендуется использовать use_cache=True + передавать cache.

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