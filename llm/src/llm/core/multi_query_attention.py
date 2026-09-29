import torch
from torch import nn
import torch.nn.functional as F
from llm.core.rope import RoPE

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA) — быстрый и экономичный вариант self-attention для LLM.

    Назначение:
    -----------
    Класс реализует механизм внимания (self-attention), в котором для всех Query-голов используются одни и те же Key и Value.
    В классическом MultiHeadAttention (MHA) на каждый Query используется свой Key/Value. В MQA набор Key/Value общий для всех голов,
    что снижает требования к памяти и ускоряет работу, что особенно важно для больших LLM на inference.

    Теоретическое преимущество:
    --------------------------
    - Существенно экономит память на матрицы Key и Value: количество KV-голов обычно в 4–8 раз меньше, чем число Query-голов.
    - Позволяет достигать скорости почти обычной MHA при минимальной потере точности (см. Llama, Mistral).
    - Является стандартом де-факто для deployment и inference современных LLM.

    Архитектурная схема:
    --------------------
    - Для каждого токена во входе вычисляются Q_h (отдельные для каждой Query-головы), но K и V — общие для всех.
    - Attention внутри каждой головы формируется через матричный продукт соответствующей Q_h и общего K.
    - Выходные вектора голов конкатенируются и проецируются обратно в emb_size.

    Формулы:
    --------
        Q = Wq·x,  K = Wk·x,  V = Wv·x
        (Wq — отдельные для всех Query, Wk/Wv — общие для всех голов)
        Attention_h(x) = softmax(Q_h·K^T / sqrt(d_k))·V
        Output = Concat_h([Attention_h(x)])·W_o

    Аргументы конструктора:
    -----------------------
    emb_size : int
        Размерность скрытого пространства (hidden size, embedding dim).
    num_heads : int
        Число Query-голов (обычно 8–32 в LLM).
    kv_heads : int
        Число Key/Value-голов (обычно 1, 2, 4, 8).
    head_size : int, optional
        Размерность одной головы (обычно emb_size // num_heads).
    dropout : float, optional
        Вероятность Dropout для регуляризации внимания.

    Пример использования:
    ---------------------
        >>> mqa = MultiQueryAttention(emb_size=512, num_heads=8, kv_heads=1)
        >>> x = torch.randn(2, 16, 512)
        >>> mask = torch.ones(2, 16, 16)
        >>> out = mqa(x, mask)
        >>> print(out.shape)  # torch.Size([2, 16, 512])

    Литература и статьи:
    --------------------
    - Shazeer, N., “Fast Transformer Decoding: One Write-Head Is All You Need” (MQA): https://arxiv.org/abs/1911.02150
    - Llama: https://arxiv.org/abs/2302.13971
    - Mistral: https://arxiv.org/abs/2310.06825
    - PaLM/PaLM2, Mixtral, ChatGLM: практическое описание MQA.
    """
    def __init__(
        self,
        num_q_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        rope: RoPE = None,
        dropout: float = 0.1,
    ):
        """
        Конструктор MultiQueryAttention.

        Инициализирует все слои и буферы для реализации Multi-Query Attention с общими K/V-головами и индивидуальными Q-головами.
        Позволяет существенно ускорять инференс и экономить память при работе с большими языковыми моделями.

        Аргументы:
        ----------
        num_q_heads : int
            Число query-голов (обычно совпадает с количеством attention heads в модели).
            Определяет количество параллельных subspace для запроса.
        emb_size : int
            Размер скрытого пространства embedding (input/output размерность attention слоя).
        head_size : int
            Размерность одной attention-головы.
            Обычно emb_size // num_q_heads.
        max_seq_len : int
            Максимально поддерживаемая длина последовательности (нужна для построения треугольной маски causal attention).
        rope : RoPE, optional
            Модуль для rotary positional encoding (позиционный энкодер, улучшает обобщающую способность attention).
            Если None, positional encoding не применяется.
        dropout : float, по умолчанию 0.1
            Вероятность dropout для выходного слоя attention (регуляризация).

        Внутри:
        -------
        - Насчитывает отдельные весовые слои для Q, общие для всех голов K/V.
        - Строит causal маску для автогрессивной генерации.
        - (Опционально) использует RoPE для позиционного кодирования.
        - Dropout применяется после финального projection.

        Пример:
        -------
            >>> mqa = MultiQueryAttention(emb_size=256, num_q_heads=8, head_size=32, max_seq_len=2048, rope=None, dropout=0.1)
        """
        super().__init__()
        self._num_q_heads = num_q_heads
        self._head_size = head_size
        self._max_seq_len = max_seq_len
        self._rope = rope
        
        self._q = nn.Linear(emb_size, num_q_heads * head_size)
        self._k = nn.Linear(emb_size,  head_size)
        self._v = nn.Linear(emb_size,  head_size)

        # Создание causal маски
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        # persistent=False: маска вычисляется из max_seq_len и не нужна в чекпоинте —
        # иначе она хранилась бы в каждом слое и привязывала бы чекпоинт к max_seq_len
        self.register_buffer("_tril_mask", mask.bool(), persistent=False)
        
        self._layer = nn.Linear(num_q_heads * head_size, emb_size)
        self._dropout = nn.Dropout(dropout)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Чекпоинты, сохранённые до persistent=False, содержат маску; она строится
        # в __init__, поэтому ключ отбрасывается, и такие чекпоинты грузятся и со strict=True
        state_dict.pop(prefix + "_tril_mask", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        use_cache: bool = True,
        cache: list = None,
    ):
        """
        Прямой проход (forward) через слой MultiQueryAttention.

        Реализует multi-query self-attention для входных последовательностей с оптимизацией памяти за счёт общих K/V-голов для всех Query.
        Поддерживает работу с rotary positional encoding (RoPE), каузальной маской и кэшированием для ускорения генерации.

        Аргументы:
        ----------
        x : torch.Tensor
            Входной тензор формы [batch_size, seq_len, emb_size] — скрытые состояния после предыдущего слоя или эмбеддинга.
        mask : torch.Tensor, optional
            Необязательная маска внимания (например, для padding или custom-маскировки). По умолчанию используется встроенная causal mask.
        use_cache : bool, по умолчанию True
            Если True, возвращает кэш ключей/значений (для autoregressive inference/generation).
        cache : list, optional
            (K_cache, V_cache) — предварительный кэш KV (для ускоренного инференса). Если None, кэш не используется/создаётся заново.

        Возвращает:
        -----------
        если use_cache == True:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
                - attention_out: [batch_size, seq_len, emb_size] — результат attention после проекции и dropout.
                - (K, V): кэшированные ключи и значения (использовать для последующих forward'ов в autoregressive генерации)
        если use_cache == False:
            Tuple[torch.Tensor, None]

        Математические шаги:
        --------------------
            1. Q = Wq·x; K = Wk·x; V = Wv·x     # Q: индивидуальные для каждой головы, K/V — общие
            2. [optional] Rotary positional encoding применяется к Q и K
            3. (optional) concat c k/v cache (for autoregressive inference)
            4. attention_scores = softmax(Q·K^T / sqrt(head_size), mask)
            5. attention_out = attention_scores·V
            6. heads сливаются и проецируются в emb_size; применяется dropout.

        Пример:
        -------
            >>> out, cache = mqa(x, mask=attn_mask, use_cache=True, cache=prev_cache)
            >>> print(out.shape)   # torch.Size([batch_size, seq_len, emb_size])

        Примечания:
        -----------
        - Для каузального режима используется треугольная маска (по умолчанию).
        - Для генерации текста с cache передавайте кэш от предыдущих токенов — это ускоряет autoregressive inference.
        - Внимание! Тензоры внутри cache должны иметь форму [batch, heads, seq_len, head_size].
        """
        batch_size, seq_len, emb_size = x.shape

        # Абсолютная позиция первого нового токена = длина закэшированной последовательности
        start_pos = cache[0].size(2) if cache is not None else 0
        if start_pos + seq_len > self._max_seq_len:
            raise ValueError(
                f"Длина последовательности {start_pos + seq_len} превышает максимум {self._max_seq_len}"
            )

        # Пропустите тензор x через матрицы Wq, Wk , Wv, чтобы получить матрицы запроса, ключа и значения.
        k = self._k(x)  # [B, T, hs]
        q = self._q(x)  # [B, T, hs]
        v = self._v(x)  # [B, T, hs]

        # Шаг 2: Изменение формы для multi-head
        # [batch_size, seq_len, num_heads * head_size] 
        # -> [batch_size, seq_len, num_heads, head_size]
        q = q.reshape(batch_size, seq_len, self._num_q_heads, self._head_size)
        k = k.reshape(batch_size, seq_len, 1, self._head_size)
        v = v.reshape(batch_size, seq_len, 1, self._head_size)

        # 3. Transpose: [B, T, H, hs] -> [B, H, T, hs]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Позиция первого нового токена = длина кэша: без неё RoPE повернёт новые Q и K как позицию 0.

        # Пропустите матрицы запроса и ключа через экземпляр rope, чтобы выполнить поворот.
        if self._rope is not None:
            # Применяем RoPE к Q и K (НЕ к V!)
            q = self._rope(q, start_pos=start_pos)  # [B, T, hs]
            k = self._rope(k, start_pos=start_pos)  # [B, T, hs]


        # Если cache пришел, то объединяем кэш и одну строку из ключа и значения. Это будут новые key и value  для последующих вычислений.
        # 5. Кэширование (для autoregressive generation)
        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=2)  # Concat по seq_len (dim=2)
            v = torch.cat([v_cache, v], dim=2)


        # Перемножим матрицы запроса и ключа (транспонированную), чтобы вычислить матрицу внимания.
        # И разделить все значения в матрице внимания на корень из head_size.
        scores = q @ k.transpose(-2, -1) / (self._head_size ** 0.5)

        # Causal-маска по абсолютным позициям (с кэшем тоже — см. MultiHeadAttention)
        causal_mask = self._tril_mask[start_pos:start_pos + seq_len, :start_pos + seq_len]
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        # Применить к матрице внимания (построчно) функцию Softmax.
        weights = F.softmax(scores, dim=-1)

        # Перемножим матрицу внимания и матрицу значения.
        x_out = weights @ v  # [B, T, hs]


        # Измените форму тензора на batch_size × seq_len × num_heads*head_size.
        # Transpose обратно и concatenate heads
        x_out = x_out.transpose(1, 2)  # [B, T_q, H, hs]
        x_out = x_out.contiguous()  # Важно для reshape!
        concatenated_attention = x_out.reshape(batch_size, seq_len, self._num_q_heads * self._head_size)


        # Пропустите получившийся тензор через последний линейный слой.
        # 3. Проецируем в пространство эмбеддингов
        projected_output = self._layer(concatenated_attention)


        # 4. Применяем dropout для регуляризации
        final_output = self._dropout(projected_output)

        if use_cache is True:
            return (final_output, (k, v))
        else:
            return (final_output, None)