import torch
from torch import nn
import torch.nn.functional as F
from llm.core.rope import RoPE
from llm.core.multi_query_attention import MultiQueryAttention
from llm.core.rms_norm import RMSNorm
from llm.core.geglu import GeGLU

class GemmaDecoder(nn.Module):
    """
    GemmaDecoder — декодерный блок архитектуры Gemma (Google DeepMind, 2024).

    Назначение:
    -----------
    Данный блок реализует одну «ячейку» декодерного стека в модели Gemma. Архитектура схожа с современными LLM (Llama/Mistral),
    но имеет уникальные особенности attention и feed-forward слоёв, соответствующие спецификации Gemma.

    Архитектурные компоненты:
    -------------------------
    - LayerNorm или RMSNorm
    - Multi-head self-attention (обычно Multi-Query Attention)
    - Skip connection (остаточное сложение)
    - Feed-forward блок (может включать SwiGLU, GeGLU или классический FFN)
    - Повторная нормализация
    - Dropout (регуляризация на уровне attention и feed-forward)

    Алгоритм прямого прохода:
    -------------------------
        1. norm1_out = LayerNorm(x)
        2. attention_out = Attention(norm1_out, ...)
        3. resid1 = attention_out + x
        4. norm2_out = LayerNorm(resid1)
        5. ffn_out = FeedForward(norm2_out)
        6. output = ffn_out + resid1

    Теоретические детали:
    ---------------------
    - В Gemma используются техники оптимизации памяти и ускорения инференса (например, shared K/V-головы, Rope, кастомные FFN).
    - Поддержка кэширования attention для ускорения генерации (KV cache).
    - Блок проектирован для использования в стеке, повторяется N раз во всей LLM.

    Аргументы конструктора:
    ----------------------
    num_q_heads : int
        Число голов query (Query Heads) для attention.
    num_kv_heads : int
        Число ключевых/значенческих голов (Key/Value Heads).
    emb_size : int
        Размерность скрытого пространства (embedding dim).
    head_size : int
        Размерность одной attention-головы.
    max_seq_len : int
        Максимальная длина последовательности (ограничение на causal mask).
    dropout : float, optional
        Dropout для регуляризации (примерно 0.0–0.1).
    rope : RoPE, optional
        Позиционное кодирование Rotary Position Embedding.

    Пример использования:
    ---------------------
        >>> decoder = GemmaDecoder(
        ...     num_q_heads=8,
        ...     num_kv_heads=2,
        ...     emb_size=256,
        ...     head_size=32,
        ...     max_seq_len=1024,
        ...     dropout=0.1,
        ...     rope=rope_obj
        ... )
        >>> x = torch.randn(2, 24, 256)
        >>> out, cache = decoder(x, mask=None, use_cache=True, cache=None)
        >>> print(out.shape)  # torch.Size([2, 24, 256])

    Литература и ссылки:
    --------------------
    - Gemma (официальный релиз): https://ai.google.dev/gemma
    - Gemma paper: https://arxiv.org/abs/2403.07794
    - Rotary Embedding: https://arxiv.org/abs/2104.09864
    - Multi-Query Attention: https://arxiv.org/abs/1911.02150
    - Llama: https://arxiv.org/abs/2302.13971
    """
    def __init__(self, 
        num_q_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        rope: RoPE,
        dropout: float = 0.1
    ):
        """
        Конструктор слоя GemmaDecoder.

        Производит инициализацию всех подслоёв (нормализация, multi-head или multi-query attention, feed-forward блок, Dropout)
        согласно архитектуре декодера Gemma. Обеспечивает поддержку rotary-позиционирования, обучения и inference с caching.

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

        Внутри:
        -------
        - Инициализируются все слои norm, attention, rope, FFN, остаточные соединения.
        - Строится causal-маска автоагрессивного attention (если требуется).
        - Гибко поддерживает работу как на training, так и для быстрых inference/генерации.

        Пример:
        -------
            >>> decoder = GemmaDecoder(
            ...     num_q_heads=8, emb_size=512, head_size=64, max_seq_len=1024, rope=rope_obj, dropout=0.05
            ... )
        """
        super().__init__()
        self._heads = MultiQueryAttention(
            num_q_heads=num_q_heads, 
            emb_size=emb_size, 
            head_size=head_size, 
            max_seq_len=max_seq_len,
            rope=rope,
            dropout=dropout
        )
        self._ff = GeGLU(emb_size=emb_size, dropout=dropout)
        self._norm1 = RMSNorm(emb_size)
        self._norm2 = RMSNorm(emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, use_cache: bool = True, cache: list = None) -> torch.Tensor:
        """
        Прямой проход (forward) через GemmaDecoder.

        Последовательно реализует:
        - Нормализацию входа (обычно RMSNorm или LayerNorm)
        - Self-attention (multi-query или multi-head, с опциональной маской и кэшем)
        - Остаточное сложение (skip connection)
        - Вторую нормализацию
        - Feed-Forward-блок (например, GeGLU/SwiGLU)
        - Ещё одно residual сложение

        Поддерживает autoregressive режим с caching (KV-слоты attention для ускорения генерации).

        Аргументы:
        ----------
        x : torch.Tensor
            Входной скрытый тензор формы [batch_size, seq_length, emb_size].
        mask : torch.Tensor, optional
            Attention mask (например, causal или padding mask). Если None, используется встроенная causal mask.
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
            >>> out, new_cache = decoder(x, mask=att_mask, use_cache=True, cache=old_cache)
            >>> out.shape  # [batch_size, seq_len, emb_size]

        Примечания:
        -----------
        - mask используется для ограничения внимания (напр., каузальный режим GPT/LLM).
        - Для ускорения в режиме генерации рекомендуется использовать use_cache=True + передавать cache.

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