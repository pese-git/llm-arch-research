from torch import nn
import torch
from llm.core.rope import RoPE
from llm.core.group_query_attention import GroupedQueryAttention
from llm.core.moe import MoE
from llm.core.rms_norm import RMSNorm

class MixtralDecoder(nn.Module):
    """
    MixtralDecoder — декодерный блок Mixtral: GQA + Mixture-of-Experts вместо плотного FFN.

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
        Размерность одной головы (обычно emb_size // num_q_heads).
    max_seq_len : int
        Максимальная поддерживаемая длина последовательности.
    num_experts : int
        Количество «экспертов» (MoE).
    top_k_experts : int
        Сколько одновременно экспертов активируется для одного токена.
    window_size : int
        Размер скользящего окна внимания. В Mixtral 8x7B окна нет (плотное внимание на 32k),
        здесь оно унаследовано от Mistral — см. docs/mixtral.md.
    rope : RoPE
        Реализация позиционного кодирования RoPE.
    dropout : float
        Вероятность Dropout для регуляризации.
    norm_eps : float
        eps обеих RMSNorm (по умолчанию 1e-6).

    Пример использования:
    ---------------------
        >>> decoder = MixtralDecoder(
        ...     num_q_heads=8, num_kv_heads=2, emb_size=256, head_size=32, max_seq_len=1024,
        ...     num_experts=4, top_k_experts=2, window_size=128, rope=RoPE(32, 1024))
        >>> x = torch.randn(2, 16, 256)
        >>> out, cache = decoder(x, use_cache=True)
        >>> out.shape  # torch.Size([2, 16, 256])

    Литература и ссылки:
    --------------------
    - Jiang et al., "Mixtral of Experts", 2024: https://arxiv.org/abs/2401.04088
    - Mixtral 8x7B (блог): https://mistral.ai/news/mixtral-of-experts/
    - Shazeer et al., “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer”, 2017. https://arxiv.org/abs/1701.06538
    - Mistral paper: https://arxiv.org/abs/2310.06825
    - GQA (Ainslie et al., 2023): https://arxiv.org/abs/2305.13245
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
        dropout: float = 0.1,
        norm_eps: float = 1e-6,
        intermediate_size: int = None,
        bias: bool = True,
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
        window_size : int или None
            Размер скользящего окна attention, как в Mistral 7B v0.1; None — окна нет (как в Mixtral 8x7B).
        rope : RoPE
            Объект позиционного кодирования RoPE (Rotary Positional Embedding), необходим для архитектуры внимания.
        dropout : float, по умолчанию 0.1
            Вероятность зануляции выходных значений для регуляризации и борьбы с переобучением.
        norm_eps : float, по умолчанию 1e-6
            eps обеих RMSNorm.
        intermediate_size : int, опционально
            Скрытый размер SwiGLU каждого эксперта, по умолчанию 4 * emb_size (у Mixtral 8x7B — 14336 = 3.5 * 4096).
        bias : bool, по умолчанию True
            Есть ли bias у всех Linear блока (в Mixtral 8x7B его нет).

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
            dropout=dropout,
            bias=bias,
        )
        self._ff = MoE(
            emb_size=emb_size, 
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            dropout=dropout,
            hidden_dim=intermediate_size,
            bias=bias,
        )
        self._norm1 = RMSNorm(emb_size, eps=norm_eps)
        self._norm2 = RMSNorm(emb_size, eps=norm_eps)

    def forward(
        self, x: torch.Tensor, use_cache: bool = True, cache: tuple = None, padding=None
    ) -> tuple:
        """
        Прямой проход (forward) через декодерный блок MixtralDecoder.

        Данный метод реализует последовательную обработку входных скрытых состояний (x) через:
        - нормализацию (RMSNorm),
        - attention-модуль (Grouped Query Attention) со встроенной маской causal + окно и кэшем ключей/значений,
        - остаточное сложение (residual connection),
        - повторную нормализацию,
        - feed-forward блок на основе Mixture-of-Experts (MoE),
        - финальное остаточное сложение.

        Аргументы:
        ----------
        x : torch.Tensor
            Входной скрытый тензор формы [batch_size, seq_len, emb_size] — результат эмбеддинга токенов либо предыдущего слоя.
        use_cache : bool, по умолчанию True
            Если True — сохраняет кэш ключей/значений attention для ускорения авторегрессии (инференса).
        cache : list, optional
            (Необязательно) Предварительно вычисленный кеш attention (для ускорения генерации длинного текста).
        padding : Padding, опционально
            Паддинг батча (core/padding.py) — передаётся в attention; None — без паддинга.

        Возвращает:
        -----------
        Tuple[torch.Tensor, Any]:
            - Первый элемент: скрытый тензор выхода слоя с той же формой, что вход (последовательный residual из attention и MoE-блока).
            - Второй элемент: обновлённый кэш attention (если use_cache=True), иначе None.

        Пример:
        -------
            >>> out, cache = decoder(x, use_cache=True, cache=old_cache)
            >>> out.shape  # [batch_size, seq_len, emb_size]

        Примечания:
        -----------
        - Паддинг батча (padding) передаётся в attention: маска ключей и позиции для RoPE.
        - Реализация поддерживает произвольные батчи и длины последовательностей, в пределах max_seq_len слоя.
        - Модуль MixtralDecoder обычно используется в виде стека (несколько подряд) внутри крупной LLM.

        """
        norm1_out = self._norm1(x)
        attention, kv_caches = self._heads(norm1_out, use_cache=use_cache, cache=cache, padding=padding)
        out = attention + x

        norm2_out = self._norm2(out)
        ffn_out = self._ff(norm2_out)

        if use_cache is True:
            return (ffn_out + out, kv_caches)
        else:
            return (ffn_out + out, None)
