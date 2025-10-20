

from torch import nn
import torch
import torch.nn.functional as F
from llm.core.rope import RoPE
from llm.core.group_query_attention import GroupedQueryAttention
from llm.core.moe import MoE
from llm.core.rms_norm import RMSNorm

class MixtralDecoder(nn.Module):
    """
    MixtralDecoder — декодерный блок для Mixtral/MoE-трансформеров (см. Mixtral 8x7B, Mistral v0.2 и др.).

    Назначение:
    -----------
    MixtralDecoder реализует один модульный слой глубокой трансформерной архитектуры с Mixture-of-Experts (MoE) Feed-Forward Network и Grouped Query Attention (GQA).
    Поддерживает разреженную активацию и масштабируемое количество экспертов, оптимально для больших LLM.

    Архитектура блока:
    ------------------
    - RMSNorm -> Grouped Query Attention (GQA)
    - skip-connection
    - RMSNorm -> MoE (SwiGLU-эксперты)
    - skip-connection

    Для входа `x` проходит:
        1. norm1_out = RMSNorm(x)
        2. attention, kv_caches = GQA(norm1_out, ...)
        3. out = attention + x  # residual connection
        4. norm2_out = RMSNorm(out)
        5. ffn_out = MoE(norm2_out)
        6. return (ffn_out + out, kv_caches)

    Теоретическая мотивация:
    ------------------------
    - Использование MoE (см. https://arxiv.org/abs/1701.06538) позволяет кратно увеличивать capacity без роста затрат на ff-часть.
    - Grouped Query Attention эффективно масштабирует self-attention для больших моделей (см. Mistral, Llama 2/3).
    - RMSNorm (Root Mean Square LayerNorm) стабилизирует градиенты и память.
    - Является строительным блоком для стека декодеров в Mixtral-моделях (см. Mixtral, Mistral, LLaMA).

    Аргументы конструктора:
    ----------------------
    num_q_heads : int
        Число query-голов в attention.
    num_kv_heads : int
        Число key-value голов (группировка ключей/values).
    emb_size : int
        Скрытый размер эмбеддинга.
    head_size : int
        Размерность одной головы (emb_size // num_q_heads).
    max_seq_len : int
        Максимальная поддерживаемая длина последовательности.
    num_experts : int
        Количество «экспертов» (MoE).
    top_k_experts : int
        Сколько одновременно экспертов активируется для одного токена.
    window_size : int
        Размер окна внимания (используется для efficient attention).
    rope : RoPE
        Реализация позиционного кодирования RoPE.
    dropout : float
        Вероятность Dropout для регуляризации.

    Пример использования:
    ---------------------
        >>> decoder = MixtralDecoder(... параметры ...)
        >>> x = torch.randn(batch, seq, emb_size)
        >>> out, cache = decoder(x, mask=None, use_cache=True)
        >>> out.shape

    Литература и ссылки:
    --------------------
    - Mixtral 8x7B: https://mistral.ai/news/mixtral-of-experts/
    - Shazeer et al., “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer”, 2017. https://arxiv.org/abs/1701.06538
    - Mistral paper: https://arxiv.org/abs/2310.06825
    - GQA: https://arxiv.org/abs/2305.14236
    - RMSNorm: https://arxiv.org/abs/1910.07467

    """
    def __init__(self, 
        num_q_heads: int,
        num_kv_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        num_experts: int,
        top_k_experts: int,
        window_size: int,
        rope: RoPE,
        dropout: float = 0.1
    ):
        """
        Конструктор декодерного блока MixtralDecoder.

        Осуществляет инициализацию всех под-компонентов слоя: Attention (Grouped Query Attention), MoE (Mixture-of-Experts, SwiGLU)
        и нормализации (RMSNorm). Позволяет гибко настраивать архитектуру под специфику задач и размеры LLM.

        Аргументы:
        ----------
        num_q_heads : int
            Количество голов внимания (queries) в механизме GroupedQueryAttention.
            Чем больше — тем тоньше дискретизация внимания по подпространствам признаков.
        num_kv_heads : int
            Количество групп ключей/значений (key-value heads) для GQA.
            Позволяет балансировать производительность и память.
        emb_size : int
            Размерность эмбеддингового пространства внутри слоя (hidden).
        head_size : int
            Размерность одной attention-головы. Обычно emb_size // num_q_heads.
        max_seq_len : int
            Максимально поддерживаемая длина токенизированной последовательности.
        num_experts : int
            Количество экспертов в слое MoE (размер пула SwiGLU-экспертов).
        top_k_experts : int
            Сколько экспертов по роутингу активируется на 1 токен (разреженность — эффективная экономия вычислений).
        window_size : int
            Размер окна для attention (может использоваться для ограничения receptive field, как в Mistral).
        rope : RoPE
            Объект позиционного кодирования RoPE (Rotary Positional Embedding), необходим для архитектуры внимания.
        dropout : float, по умолчанию 0.1
            Вероятность зануляции выходных значений для регуляризации и борьбы с переобучением.

        Пример:
        -------
            >>> decoder = MixtralDecoder(
            ...     num_q_heads=8,
            ...     num_kv_heads=2,
            ...     emb_size=256,
            ...     head_size=32,
            ...     max_seq_len=1024,
            ...     num_experts=4,
            ...     top_k_experts=2,
            ...     window_size=128,
            ...     rope=rope_module,
            ...     dropout=0.05
            ... )

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
        self._ff = MoE(
            emb_size=emb_size, 
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            dropout=dropout
        )
        self._norm1 = RMSNorm(emb_size)
        self._norm2 = RMSNorm(emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, use_cache: bool = True, cache: list = None) -> torch.Tensor:
        """
        Прямой проход (forward) через декодерный блок MixtralDecoder.

        Данный метод реализует последовательную обработку входных скрытых состояний (x) через:
        - нормализацию (RMSNorm),
        - attention-модуль (Grouped Query Attention) с опциональным применением маски и кэша ключей/значений для ускорения инференса,
        - остаточное сложение (residual connection),
        - повторную нормализацию,
        - feed-forward блок на основе Mixture-of-Experts (MoE),
        - финальное остаточное сложение.

        Аргументы:
        ----------
        x : torch.Tensor
            Входной скрытый тензор формы [batch_size, seq_len, emb_size] — результат эмбеддинга токенов либо предыдущего слоя.
        mask : torch.Tensor, optional
            (Необязательно) Маска внимания для ограничения области self-attention (например, для автоперемешивания или causal-LLM-моделей).
        use_cache : bool, по умолчанию True
            Если True — сохраняет кэш ключей/значений attention для ускорения авторегрессии (инференса).
        cache : list, optional
            (Необязательно) Предварительно вычисленный кеш attention (для ускорения генерации длинного текста).

        Возвращает:
        -----------
        Tuple[torch.Tensor, Any]:
            - Первый элемент: скрытый тензор выхода слоя с той же формой, что вход (последовательный residual из attention и MoE-блока).
            - Второй элемент: обновлённый кэш attention (если use_cache=True), иначе None.

        Пример:
        -------
            >>> out, cache = decoder(x, mask=att_mask, use_cache=True, cache=old_cache)
            >>> out.shape  # [batch_size, seq_len, emb_size]

        Примечания:
        -----------
        - Для autoregressive-генерации (GPT-like режимов) следует передавать mask и использовать use_cache=True.
        - Реализация поддерживает произвольные батчи и длины последовательностей, в пределах max_seq_len слоя.
        - Модуль MixtralDecoder обычно используется в виде стека (несколько подряд) внутри крупной LLM.

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
