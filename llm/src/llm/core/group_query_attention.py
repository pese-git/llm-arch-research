import torch
from torch import nn
import torch.nn.functional as F

from llm.core.rope import RoPE

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)
    =============================

    Что такое Grouped Query Attention?
    ----------------------------------
    Это разновидность многоголового внимания (multi-head), где для Q (query) голов может быть больше, чем для K/V (key/value) голов:
    вместо стандартного MHA (num_q_heads == num_kv_heads) — меньшее число K/V разделяет информацию для всех Q.
    Такой подход экономит память и ускоряет инференс, сохраняя высокое качество внимания (используется например в Mistral, Llama-2, GPT-4 и др.).

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
    window_size: int — размер sliding window (макс. количество токенов в контексте внимания)
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
    - \"Self-attention with linear complexity\" (Vila et al.): https://arxiv.org/abs/2302.05442
    - Обзор: https://huggingface.co/blog/mistral

    """

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        window_size: int,
        rope: RoPE = None,
        dropout: float = 0.1,
    ):
        """
        Инициализация слоя Grouped Query Attention (GQA).

        Этот конструктор задаёт архитектуру эффективного внимания, где Q-голов может быть больше, чем KV-голов. 
        Это экономит память/вычисления и позволяет реализовать сдвигающееся "окно" внимания (Mistral-style).

        Аргументы:
        ----------
        num_q_heads : int
            Количество Query attention heads (чаще всего кратно num_kv_heads, напр. 8/2, 12/4).
            Чем больше — тем богаче контекстное окно каждой позиции.
        num_kv_heads : int
            Количество Key/Value attention heads (обычно 2-4, иногда меньше, чем Query).
            В современных LLM принято уменьшать их число для оптимизации скорости/кэша.
        emb_size : int
            Размерность входного embedding (общий размер вектора на токен).
        head_size : int
            Размерность одной головы внимания.
            Требуется: num_q_heads * head_size == emb_size (иначе ошибка).
        max_seq_len : int
            Максимальная поддерживаемая длина входной последовательности; определяет размер триангулярной (causal/sliding window) маски.
        window_size : int
            Размер "скользящего окна" истории — сколько токенов учитывается при слепом внимании (как у Mistral).
            Чем меньше значение, тем локальнее работает внимание (и меньше память/время).
        rope : RoPE, опционально
            Если задан — применяется Rotary Positional Encoding к Q и K для относительного позиционного кодирования.
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
        self._num_heads = num_q_heads
        self._num_kv_heads = num_kv_heads
        self._head_size = head_size
        self._max_seq_len = max_seq_len
        self._rope = rope
        self._window_size = window_size

        self._q = nn.Linear(emb_size, self._num_heads * head_size)
        self._k = nn.Linear(emb_size, num_kv_heads * head_size)
        self._v = nn.Linear(emb_size, num_kv_heads * head_size)

        # Создание causal маски
        mask = self._create_sliding_window_mask(max_seq_len, self._window_size)
        self.register_buffer(
            "_tril_mask", mask.bool() if hasattr(torch, "bool") else mask.byte()
        )
        
        self._layer = nn.Linear(head_size * self._num_heads, emb_size)
        self._dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        use_cache: bool = True,
        cache: list = None,
    ):
        """
        Шаг внимания в режиме Grouped Query Attention — 
        реализует эффективное многооконное внимание с раздельными Q/KV и sliding/casual mask.

        Что происходит в этом методе:
        -----------------------------
        - Преобразует входной тензор x (токеновые эмбеддинги) в Q, K, V-матрицы с учётом разного числа голов для Q и KV.
        - Формирует attention "маску" для sliding window, если нужно ограничить историю.
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
        mask : torch.Tensor, по умолчанию None
            Матричная маска для внимания (можно передать внешнюю или использовать встроенную sliding window mask)
        use_cache : bool, по умолчанию True
            Нужно ли использовать/возвращать кэш KV для быстрых автогенераций.
        cache : list, опционально
            Ранее сохранённый кэш KV (используется для инференса по одному токену)

        Возвращает:
        -----------
        - output: torch.Tensor формы [batch, seq_len, emb_size]
        - kv_cache: кэш новых KV (если use_cache=True), иначе None

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

        if seq_len > self._max_seq_len:
            raise ValueError(
                f"Длина последовательности {seq_len} превышает максимум {self._max_seq_len}"
            )

        # Пропустите тензор x через матрицы Wq, Wk , Wv, чтобы получить матрицы запроса, ключа и значения.
        k = self._k(x)  # [B, T, hs]
        q = self._q(x)  # [B, T, hs]
        v = self._v(x)  # [B, T, hs]

        # Шаг 2: Изменение формы для multi-head
        # [batch_size, seq_len, num_heads * head_size] 
        # -> [batch_size, seq_len, num_heads, head_size]
        # Измените форму запроса (query) на batch_size × num_q_heads × seq_len × head_size.
        q = q.reshape(batch_size, seq_len, self._num_heads, self._head_size)

        # Измените форму ключа (key) и значения (value) на batch_size × num_kv_heads × seq_len × head_size.
        k = k.reshape(batch_size, seq_len, self._num_kv_heads, self._head_size)
        v = v.reshape(batch_size, seq_len, self._num_kv_heads, self._head_size)
     

        # 3. Transpose: [B, T, H, hs] -> [B, H, T, hs]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        start_pos = 0
        if cache is not None:
            k_cache, v_cache = cache
            cache_len = k_cache.shape[2]
            start_pos = cache_len
        
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

        # Если use_cache == True, то сохраните матрицы ключа и значения для кэша (это нужно сделать до дублирования голов).
        #if use_cache == True:
        #    # Обрезаем до последних window_size токенов
        #    k_to_cache = k[:, :, -self._window_size:, :]
        #    v_to_cache = v[:, :, -self._window_size:, :]
        #    kv_cache = (k_to_cache, v_to_cache)

        # Продублируйте головы в тензорах ключа (key) и значения (value), чтобы получился тензор размера на batch_size × num_q_heads × seq_len × head_size.
        #k = self._repeat_kv_heads(k, self._num_heads, self._num_kv_heads)
        #v = self._repeat_kv_heads(v, self._num_heads, self._num_kv_heads)
        k_expanded = self._repeat_kv_heads(k, self._num_heads, self._num_kv_heads)
        v_expanded = self._repeat_kv_heads(v, self._num_heads, self._num_kv_heads)
        
        # Перемножим матрицы запроса и ключа (транспонированную), чтобы вычислить матрицу внимания.
        # И разделить все значения в матрице внимания на корень из head_size.
        scores = q @ k_expanded.transpose(-2, -1) / (self._head_size ** 0.5)

        # 8. Применение маски
        k_seq_len = k_expanded.size(2)  # Длина K после concat с кэшем
    
        if cache is None:
            # Случай 1: Без кэша - полная квадратная маска
            # scores: [B, H, seq_len, seq_len]
            # Применяем маску [:seq_len, :seq_len]
            scores = scores.masked_fill(
                ~self._tril_mask[:seq_len, :seq_len], 
                float("-inf")
            )

        # Применить к матрице внимания (построчно) функцию Softmax.
        weights = F.softmax(scores, dim=-1)

        # Перемножим матрицу внимания и матрицу значения.
        x_out = weights @ v_expanded  # [B, T, hs]

        # Измените форму тензора на batch_size × seq_len × num_heads*head_size.
        # Transpose обратно и concatenate heads
        x_out = x_out.transpose(1, 2)  # [B, T_q, H, hs]
        x_out = x_out.contiguous()  # Важно для reshape!
        concatenated_attention = x_out.reshape(batch_size, seq_len, self._num_heads * self._head_size)

        #concatenated_attention = x_out.reshape(batch_size, seq_len, self._num_heads * self._head_size)

        # Пропустите получившийся тензор через последний линейный слой.
        # 3. Проецируем в пространство эмбеддингов
        projected_output = self._layer(concatenated_attention)

        # 4. Применяем dropout для регуляризации
        output = self._dropout(projected_output)

        if use_cache:
            # Обрезаем оригинальный K и V (до дублирования)
            k_to_cache = k[:, :, -self._window_size:, :]
            v_to_cache = v[:, :, -self._window_size:, :]
            kv_cache = (k_to_cache, v_to_cache)
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
        - Для каждого токена mask[i, j] == True только если токен j находится СЛЕВА и не дальше, чем window_size позиций (или сам i).
        - Главное: mask всегда "нижнетреугольная" (causal), плюс полоса шириной window_size вдоль главной диагонали.
        - Всё за пределами окна — False (attention нельзя).

        Args:
        -----
        max_seq_len : int
            Максимальная длина последовательности (размер будущей attention-матрицы).
        window_size : int
            Сколько предыдущих токенов доступно для внимания у каждого шага (вкл. сам себя).
        device : torch.device, опционально
            На каком устройстве (cpu/gpu) создавать маску.

        Returns:
        --------
        torch.Tensor 
            Маска внимания формы [max_seq_len, max_seq_len], где True — допускается внимание (иначе False).

        Пример:
        -------
            >>> mask = create_sliding_window_mask(8, 3)
            >>> print(mask.int())
            tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1]])
        """
        row_indices = torch.arange(max_seq_len, device=device).unsqueeze(1)  # [max_seq_len, 1]
        col_indices = torch.arange(max_seq_len, device=device).unsqueeze(0)  # [1, max_seq_len]

        causal_mask = col_indices <= row_indices

        window_mask = (row_indices - col_indices) <= window_size

        mask = causal_mask & window_mask
        
        return mask