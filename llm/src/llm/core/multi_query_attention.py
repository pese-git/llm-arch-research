import torch
from torch import nn
import torch.nn.functional as F

from llm.core.padding import Padding
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
    - Существенно экономит память на матрицы Key и Value: K/V-голова одна на все Query-головы,
      поэтому KV-кэш в num_q_heads раз меньше, чем у MHA.
    - Почти не теряет в качестве по сравнению с MHA (используется, например, в PaLM и Gemma 2B).
    - Обобщение с несколькими K/V-головами — Grouped Query Attention (GroupedQueryAttention).

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
    num_q_heads : int
        Число Query-голов; K/V-голова всегда одна.
    emb_size : int
        Размерность скрытого пространства (hidden size, embedding dim).
    head_size : int
        Размерность одной головы (обычно emb_size // num_q_heads).
    max_seq_len : int
        Максимальная длина последовательности (размер causal-маски).
    rope : RoPE, optional
        Rotary positional encoding для Q и K.
    dropout : float, optional
        Вероятность Dropout после выходной проекции.

    Пример использования:
    ---------------------
        >>> mqa = MultiQueryAttention(num_q_heads=8, emb_size=512, head_size=64, max_seq_len=128)
        >>> x = torch.randn(2, 16, 512)
        >>> out, cache = mqa(x, use_cache=True)
        >>> print(out.shape)  # torch.Size([2, 16, 512])

    Литература и статьи:
    --------------------
    - Shazeer, N., “Fast Transformer Decoding: One Write-Head Is All You Need” (MQA): https://arxiv.org/abs/1911.02150
    - Gemma (2B использует MQA): https://arxiv.org/abs/2403.08295
    - GQA как обобщение MQA: https://arxiv.org/abs/2305.13245
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
        use_cache: bool = True,
        cache: list = None,
        padding: Padding = None,
    ):
        """
        Прямой проход (forward) через слой MultiQueryAttention.

        Реализует multi-query self-attention для входных последовательностей с оптимизацией памяти за счёт общих K/V-голов для всех Query.
        Поддерживает работу с rotary positional encoding (RoPE), каузальной маской и кэшированием для ускорения генерации.

        Аргументы:
        ----------
        x : torch.Tensor
            Входной тензор формы [batch_size, seq_len, emb_size] — скрытые состояния после предыдущего слоя или эмбеддинга.
        use_cache : bool, по умолчанию True
            Если True, возвращает кэш ключей/значений (для autoregressive inference/generation).
        cache : list, optional
            (K_cache, V_cache) — предварительный кэш KV (для ускоренного инференса). Если None, кэш не используется/создаётся заново.
        padding : Padding, опционально
            Паддинг батча (core/padding.py): маска ключей по всем слотам и позиции новых
            токенов для RoPE. None — паддинга нет: только встроенная маска и позиции start_pos, …

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
            4. attention_scores = softmax(causal_mask(Q·K^T / sqrt(head_size)))
            5. attention_out = attention_scores·V
            6. heads сливаются и проецируются в emb_size; применяется dropout.

        Пример:
        -------
            >>> out, cache = mqa(x, use_cache=True, cache=prev_cache)
            >>> print(out.shape)   # torch.Size([batch_size, seq_len, emb_size])

        Примечания:
        -----------
        - Маска встроенная, causal; с padding к ней добавляется маска ключей паддинга (core/padding.py).
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

        # 1. Проекции: Q — на num_q_heads голов, K и V — на одну общую голову
        q = self._q(x)  # [B, T, H * hs]
        k = self._k(x)  # [B, T, hs]
        v = self._v(x)  # [B, T, hs]

        # 2. Разбиение на головы: [B, T, H, hs] -> [B, H, T, hs]; у K и V H = 1
        q = q.reshape(batch_size, seq_len, self._num_q_heads, self._head_size).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, 1, self._head_size).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, 1, self._head_size).transpose(1, 2)

        # 3. RoPE поворачивает Q и K (не V) по абсолютным позициям start_pos …
        positions = padding.positions if padding is not None else None
        if self._rope is not None:
            q = self._rope(q, start_pos=start_pos, positions=positions)  # [B, H, T, hs]
            k = self._rope(k, start_pos=start_pos, positions=positions)  # [B, 1, T, hs]

        # 4. Ключи и значения из кэша идут перед новыми
        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=2)  # [B, 1, cache_len + T, hs]
            v = torch.cat([v_cache, v], dim=2)

        # 5. Scaled dot-product: общая K-голова транслируется на все Q-головы
        scores = q @ k.transpose(-2, -1) / (self._head_size ** 0.5)  # [B, H, T, T_kv]

        # 6. Causal-маска по абсолютным позициям (с кэшем тоже — см. MultiHeadAttention)
        causal_mask = self._tril_mask[start_pos:start_pos + seq_len, :start_pos + seq_len]
        if padding is not None:
            # + маска ключей паддинга, своя у каждой строки: [B, 1, T, T_kv]
            causal_mask = padding.apply(causal_mask, start_pos, key_start=0)
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        # 7. Softmax и взвешенная сумма значений
        weights = F.softmax(scores, dim=-1)
        x_out = weights @ v  # [B, H, T, hs]

        # 8. Склейка голов и проекция обратно в emb_size
        x_out = x_out.transpose(1, 2).contiguous()  # [B, T, H, hs]
        concatenated_attention = x_out.reshape(batch_size, seq_len, self._num_q_heads * self._head_size)
        final_output = self._dropout(self._layer(concatenated_attention))  # [B, T, emb_size]

        if use_cache is True:
            return (final_output, (k, v))
        else:
            return (final_output, None)