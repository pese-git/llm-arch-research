import torch
from torch import nn
import torch.nn.functional as F

from llm.core.padding import Padding
from llm.core.rope import RoPE

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)
    =============================

    Что такое Grouped Query Attention?
    ----------------------------------
    Это разновидность многоголового внимания (multi-head), где для Q (query) голов может быть больше, чем для K/V (key/value) голов:
    вместо стандартного MHA (num_q_heads == num_kv_heads) — меньшее число K/V разделяет информацию для всех Q.
    Такой подход экономит память и ускоряет инференс, сохраняя высокое качество внимания (используется, например, в Llama 2 70B и Mistral 7B).

    Зачем это нужно?
    ----------------
    - Сокращает количество вычислений и размер KV-кэша в больших LLM.
    - Позволяет эффективно масштабировать число attention-глав для моделирования сложных связей, не увеличивая размер всех матриц.

    Как работает?
    -------------
    1. Q формируется для каждого query-head (их много)
    2. K и V вычисляется только для меньшего числа KV-heads (обычно в 2-4 раза меньше, чем Q)
    3. К/V heads дублируются (repeat) так, чтобы на каждую Q-head был свой набор
    4. Всё внимание (Q,K,V) — стандартное scaled dot-product, только более эффективно и с компрессией

    Поддержка дополнительных фич:
    -----------------------------
    - Rotary Position Encoding (RoPE) для Q и K (для относительной позиции)
    - Sliding-window attention mask (можно ограничить исторический контекст, как в Mistral)
    - Кэширование Q/K/V (ускоряет генерацию автоагретивно)

    Аргументы конструктора:
    -----------------------
    num_q_heads: int — количество query голов (Q)
    num_kv_heads: int — количество key/value голов (обычно меньше Q)
    emb_size: int — embedding размерность
    head_size: int — размер каждой attention-head
    max_seq_len: int — максимальная длина последовательности
    window_size: int — ширина sliding window: токен видит window_size предыдущих позиций и себя (window_size + 1 позиций)
    rope: RoPE (по желанию) — если задан, то будет применяться RoPE для Q и K
    dropout: float — dropout после линейной проекции

    Пример использования:
    ---------------------
        >>> gqa = GroupedQueryAttention(num_q_heads=8, num_kv_heads=2, emb_size=256, head_size=32, max_seq_len=1024, window_size=256)
        >>> x = torch.randn(2, 128, 256)
        >>> y, cache = gqa(x)
        >>> print(y.shape)  # torch.Size([2, 128, 256])

    Где прочитать подробнее:
    ------------------------
    - LlamaV2 (Section 2.3): https://arxiv.org/abs/2307.09288
    - Mistral: https://arxiv.org/abs/2310.06825
    - GQA: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", 2023: https://arxiv.org/abs/2305.13245
    - Обзор: https://huggingface.co/blog/mistral

    """

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        window_size: int = None,
        rope: RoPE = None,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Инициализация слоя Grouped Query Attention (GQA).

        Этот конструктор задаёт архитектуру эффективного внимания, где Q-голов может быть больше, чем KV-голов. 
        Это экономит память/вычисления и позволяет реализовать сдвигающееся "окно" внимания (Mistral-style).

        Аргументы:
        ----------
        num_q_heads : int
            Количество Query attention heads; должно делиться на num_kv_heads (напр. 8/2, 12/4), иначе ValueError.
            Чем больше — тем богаче контекстное окно каждой позиции.
        num_kv_heads : int
            Количество Key/Value attention heads (обычно 2-4, иногда меньше, чем Query).
            В современных LLM принято уменьшать их число для оптимизации скорости/кэша.
        emb_size : int
            Размерность входного embedding (общий размер вектора на токен).
        head_size : int
            Размерность одной головы внимания.
            num_q_heads * head_size может отличаться от emb_size: выходная проекция возвращает результат в emb_size.
        max_seq_len : int
            Максимальная поддерживаемая длина входной последовательности; определяет размер триангулярной (causal/sliding window) маски.
        window_size : int или None, по умолчанию None
            Размер "скользящего окна" истории: токен видит window_size предыдущих позиций и себя (как у Mistral 7B v0.1).
            Чем меньше значение, тем локальнее работает внимание (и меньше память/время).
            None — окна нет, обычное causal-внимание на весь контекст (Mixtral, Mistral v0.2+), кэш не обрезается.
        rope : RoPE, опционально
            Если задан — применяется Rotary Positional Encoding к Q и K для относительного позиционного кодирования.
        bias : bool, по умолчанию True
            Есть ли bias у W_Q, W_K, W_V и W_O (в Mistral и Mixtral его нет).
        dropout : float, по умолчанию 0.1
            Dropout после линейной проекции attention (обычно 0.1, помогает борьбе с переобучением).

        Что создаётся внутри:
        ---------------------
        - Линейные слои для получения Q, K, V из embedding.
        - Буфер для causal/sliding window mask (матрица масок в зависимости от window_size и max_seq_len).
        - Линейный слой для финального преобразования (объединение всех голов и возврат к emb_size).
        - Dropout перед возвратом.

        Пример:
        -------
            >>> attn = GroupedQueryAttention(
            ...     num_q_heads=8, num_kv_heads=2, emb_size=256, head_size=32,
            ...     max_seq_len=1024, window_size=256, dropout=0.1)
        """
        super().__init__()
        if num_kv_heads < 1 or num_q_heads % num_kv_heads != 0:
            # Каждая K/V-голова обслуживает группу из num_q_heads // num_kv_heads Q-голов;
            # без делимости _repeat_kv_heads падал бы в первом forward с ошибкой reshape
            raise ValueError(
                f"num_q_heads={num_q_heads} должно делиться на num_kv_heads={num_kv_heads} "
                "(каждая K/V-голова обслуживает одинаковую группу Q-голов)"
            )
        self._num_heads = num_q_heads
        self._num_kv_heads = num_kv_heads
        self._head_size = head_size
        self._max_seq_len = max_seq_len
        self._rope = rope
        self._window_size = window_size

        self._q = nn.Linear(emb_size, self._num_heads * head_size, bias=bias)
        self._k = nn.Linear(emb_size, num_kv_heads * head_size, bias=bias)
        self._v = nn.Linear(emb_size, num_kv_heads * head_size, bias=bias)

        # Маска causal + скользящее окно; без окна — обычная causal-маска (окно шире любой последовательности)
        mask = self._create_sliding_window_mask(
            max_seq_len, max_seq_len if window_size is None else window_size
        )
        # persistent=False: маска вычисляется из max_seq_len и window_size и не нужна в
        # чекпоинте — иначе она хранилась бы в каждом слое и привязывала бы к ним чекпоинт
        self.register_buffer("_tril_mask", mask.bool(), persistent=False)
        
        self._layer = nn.Linear(head_size * self._num_heads, emb_size, bias=bias)
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
        Шаг внимания в режиме Grouped Query Attention — 
        реализует эффективное многооконное внимание с раздельными Q/KV и sliding/casual mask.

        Что происходит в этом методе:
        -----------------------------
        - Преобразует входной тензор x (токеновые эмбеддинги) в Q, K, V-матрицы с учётом разного числа голов для Q и KV.
        - Накладывает встроенную маску causal + sliding window; с padding — ещё маску ключей паддинга (core/padding.py).
        - Применяет RoPE (если задан) к Q и K, вносит позиционную информацию.
        - При работе с кэшем дополняет ключи и значения предыдущими (ускоряет генерацию).
        - Повторяет K/V головы для соответствия количеству Q (чтобы на каждую Q-head приходился свой KV).
        - Считает обычное scaled dot-product внимание, применяет маску (не даёт видеть будущее, как и в autoregressive).
        - Softmax, смешивание V на основе attention, объединение всех голов.
        - Dropout и финальное линейное преобразование обратно к emb_size.

        Аргументы:
        ----------
        x : torch.Tensor
            Входной тензор размера [batch, seq_len, emb_size]
        use_cache : bool, по умолчанию True
            Нужно ли использовать/возвращать кэш KV для быстрых автогенераций.
        cache : list, опционально
            Ранее сохранённый кэш KV (используется для инференса по одному токену)
        padding : Padding, опционально
            Паддинг батча (core/padding.py): маска ключей по всем слотам и позиции новых
            токенов для RoPE. None — паддинга нет: только встроенная маска и позиции start_pos, …

        Возвращает:
        -----------
        - output: torch.Tensor формы [batch, seq_len, emb_size]
        - kv_cache: (K, V, next_pos), если use_cache=True, иначе None. K и V — последние window_size
          позиций (до дублирования голов), next_pos — абсолютная позиция следующего токена для RoPE

        Важно:
        -------
        - Реализует Mistral-style attention: к каждой Q-head в итоге “приписан” собственный (но потенциально дублированный) KV-head.
        - Sliding window ограничивает область вижимости в attention (ускоряет генерацию на длинных последовательностях).
        - Использование RoPE опционально — но необходимо для современных архитектур LLM.

        Пример:
        -------
            >>> attn = GroupedQueryAttention(num_q_heads=8, num_kv_heads=2, emb_size=256, head_size=32, max_seq_len=1024, window_size=256)
            >>> x = torch.randn(2, 128, 256)
            >>> y, kv_cache = attn(x)
            >>> print(y.shape)  # torch.Size([2, 128, 256])
        """
        batch_size, seq_len, emb_size = x.shape

        # Кэш — (K, V, next_pos). K и V обрезаны до последних window_size позиций, поэтому
        # абсолютную позицию для RoPE нельзя брать из длины кэша — она хранится отдельно.
        start_pos = cache[2] if cache is not None else 0
        if start_pos + seq_len > self._max_seq_len:
            raise ValueError(
                f"Длина последовательности {start_pos + seq_len} превышает максимум {self._max_seq_len}"
            )

        # 1. Проекции: Q — на num_q_heads голов, K и V — на num_kv_heads голов
        q = self._q(x)  # [B, T, H_q * hs]
        k = self._k(x)  # [B, T, H_kv * hs]
        v = self._v(x)  # [B, T, H_kv * hs]

        # 2. Разбиение на головы: [B, T, H, hs] -> [B, H, T, hs]
        q = q.reshape(batch_size, seq_len, self._num_heads, self._head_size).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self._num_kv_heads, self._head_size).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self._num_kv_heads, self._head_size).transpose(1, 2)

        # 3. RoPE поворачивает Q и K (не V) по абсолютным позициям start_pos …
        positions = padding.positions if padding is not None else None
        if self._rope is not None:
            q = self._rope(q, start_pos=start_pos, positions=positions)  # [B, H_q, T, hs]
            k = self._rope(k, start_pos=start_pos, positions=positions)  # [B, H_kv, T, hs]

        # 4. Ключи и значения из кэша идут перед новыми (кэш хранится до дублирования голов)
        if cache is not None:
            k_cache, v_cache, _ = cache
            k = torch.cat([k_cache, k], dim=2)  # [B, H_kv, cache_len + T, hs]
            v = torch.cat([v_cache, v], dim=2)

        # 5. Каждая K/V-голова дублируется на свою группу Q-голов. Единственная K/V-голова (MQA)
        # не копируется, а транслируется в матричном умножении — как в MultiQueryAttention
        if self._num_kv_heads == 1:
            k_expanded, v_expanded = k, v  # [B, 1, T_kv, hs]
        else:
            k_expanded = self._repeat_kv_heads(k, self._num_heads, self._num_kv_heads)  # [B, H_q, T_kv, hs]
            v_expanded = self._repeat_kv_heads(v, self._num_heads, self._num_kv_heads)

        # 6. Scaled dot-product
        scores = q @ k_expanded.transpose(-2, -1) / (self._head_size ** 0.5)  # [B, H_q, T, T_kv]

        # 7. Маска causal + скользящее окно по абсолютным позициям. Строки — новые токены
        # start_pos … start_pos + seq_len − 1, столбцы — ключи: из кэша (последние
        # cache_len позиций перед start_pos) и новые. Нужна и с кэшем: при префилле кусками
        # новые токены не должны видеть друг друга «вперёд», и каждой строке нужно своё окно.
        cache_len = k.size(2) - seq_len
        window_mask = self._tril_mask[
            start_pos:start_pos + seq_len, start_pos - cache_len:start_pos + seq_len
        ]
        if padding is not None:
            # + маска ключей паддинга, своя у каждой строки: [B, 1, T, T_kv]
            window_mask = padding.apply(window_mask, start_pos, key_start=start_pos - cache_len)
        scores = scores.masked_fill(~window_mask, float("-inf"))

        # 8. Softmax и взвешенная сумма значений
        weights = F.softmax(scores, dim=-1)
        x_out = weights @ v_expanded  # [B, H_q, T, hs]

        # 9. Склейка голов и проекция обратно в emb_size
        x_out = x_out.transpose(1, 2).contiguous()  # [B, T, H_q, hs]
        concatenated_attention = x_out.reshape(batch_size, seq_len, self._num_heads * self._head_size)
        output = self._dropout(self._layer(concatenated_attention))  # [B, T, emb_size]

        if use_cache:
            # В кэш — последние window_size позиций K и V (до дублирования голов); без окна — все
            if self._window_size is not None:
                k = k[:, :, -self._window_size:, :]
                v = v[:, :, -self._window_size:, :]
            kv_cache = (k, v, start_pos + seq_len)
            return output, kv_cache
        else:
            return output, None

    def _repeat_kv_heads(
        self,
        kv: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int
    ) -> torch.Tensor:
        """
        Приводит число голов K/V к числу голов Q путём поэлементного повторения (tile) KV-голов.

        Зачем это нужно?
        ----------------
        В Grouped Query Attention (Mistral, Llama-2, GPT-4 и др.) обычно num_kv_heads < num_q_heads.
        Чтобы каждая Query-head могла смотреть на свою собственную (пусть и общую) KV, мы "нарезаем" или повторяем KV столько раз, сколько требуется — это экономит память и ускоряет генерацию.

        Алгоритм:
        ---------
        - kv имеет форму [batch_size, num_kv_heads, seq_len, head_size]
        - Для каждого KV-head делается n_repeat = num_q_heads // num_kv_heads по head-axis (обычно целое)
        - На выходе форма [batch_size, num_q_heads, seq_len, head_size], где каждый KV-head дублирован для нужного количества Q-heads.

        Args:
        -----
        kv : torch.Tensor
            Входной тензор KV (обычно после linear layer on эмбеддинги), размер [batch_size, num_kv_heads, seq_len, head_size]
        num_q_heads : int
            Сколько должно быть Q-голов (их больше!)
        num_kv_heads : int
            Сколько KV-голов было (их меньше!)

        Returns:
        --------
        torch.Tensor формы [batch_size, num_q_heads, seq_len, head_size], где KV-головы повторены как требуется.

        Пример:
        -------
            num_q_heads = 8, num_kv_heads = 2
            [KV0, KV1] -> [KV0, KV0, KV0, KV0, KV1, KV1, KV1, KV1]
            # Каждый KV-head дублируется 4 раза, чтобы покрыть все 8 Q-heads.
        """
        batch_size, num_kv_heads, seq_len, head_size = kv.shape

        if num_q_heads == num_kv_heads:
            # Нет необходимости дублировать
            return kv

        # Вычисляем сколько раз нужно повторить каждую голову
        num_repeats = num_q_heads // num_kv_heads

        # repeat_interleave дублирует каждую голову num_repeats раз
        # [B, num_kv_heads, S, hs] -> [B, num_q_heads, S, hs]
        # [B, num_kv_heads, S, hs] -> [B, num_kv_heads, 1, S, hs]
        kv = kv.unsqueeze(2)
        
        # [B, num_kv_heads, 1, S, hs] -> [B, num_kv_heads, num_repeats, S, hs]
        kv = kv.repeat(1, 1, num_repeats, 1, 1)
        
        # [B, num_kv_heads, num_repeats, S, hs] -> [B, num_q_heads, S, hs]
        kv = kv.reshape(batch_size, num_q_heads, seq_len, head_size)
        

        return kv

    def _create_sliding_window_mask(
        self,
        max_seq_len: int,
        window_size: int,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Создаёт маску для Sliding Window Attention (ограниченного окна внимания).

        Зачем нужна эта маска?
        ----------------------
        В современных LLM (например, Mistral) self-attention работает не по всей истории, а только в узком "скользящем окне":  
        каждый токен видит только предшествующие (или соседние) токены на расстоянии window_size.  
        Это ускоряет инференс на длинных текстах и экономит память, но сохраняет ключевые зависимости в пределах окна.

        Как работает алгоритм:
        ----------------------
        - Для каждого токена mask[i, j] == True только если 0 <= i - j <= window_size: токен j не правее i и не дальше window_size позиций.
        - Главное: mask всегда "нижнетреугольная" (causal), плюс полоса шириной window_size + 1 вдоль главной диагонали (включая сам токен).
        - Всё за пределами окна — False (attention нельзя).

        Args:
        -----
        max_seq_len : int
            Максимальная длина последовательности (размер будущей attention-матрицы).
        window_size : int
            Сколько предыдущих токенов доступно для внимания у каждого шага (не считая самого токена; всего window_size + 1 позиций).
        device : torch.device, опционально
            На каком устройстве (cpu/gpu) создавать маску.

        Returns:
        --------
        torch.Tensor 
            Маска внимания формы [max_seq_len, max_seq_len], где True — допускается внимание (иначе False).

        Пример:
        -------
            >>> mask = attn._create_sliding_window_mask(8, 3)
            >>> print(mask.int())
            tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1]])
        """
        row_indices = torch.arange(max_seq_len, device=device).unsqueeze(1)  # [max_seq_len, 1]
        col_indices = torch.arange(max_seq_len, device=device).unsqueeze(0)  # [1, max_seq_len]

        causal_mask = col_indices <= row_indices

        window_mask = (row_indices - col_indices) <= window_size

        mask = causal_mask & window_mask
        
        return mask