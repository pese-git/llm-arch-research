from torch import nn
import torch
import torch.nn.functional as F
from .rope import RoPE


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Многоголовое внимание)
    ============================================

    Что такое Multi-Head Attention?
    -------------------------------
    Это ключевой компонент трансформеров, который позволяет "смотреть" на разные части предложения
    одновременно с нескольких независимых ракурсов ("голов"). Всё, что делает Single-Head Attention — только гораздо мощнее и глубже!

    Зачем это нужно?
    ----------------
    - Модель может учиться одновременно учитывать и локальные, и глобальные взаимосвязи между токенами.
    - Каждая attention head "ловит" свой собственный смысл/зависимости, и на выходе они объединяются.
    - Это значительно улучшает понимание сложных зависимостей в тексте, особенно на длинных последовательностях.

    Как работает алгоритм? (основная схема)
    ---------------------------------------
    1. Генерируются Q, K, V (query, key, value) — по отдельной проекции для каждой головы.
    2. Для каждой головы: attention(Q, K, V) = softmax(Q·K^T / sqrt(d)) · V
    3. Все головы "склеиваются" (concatenate) и прогоняются через общий финальный линейный слой.

    Почему это работает?
    --------------------
    - Даёт трансформеру многомерное восприятие текста.
    - Позволяет эффективно обучаться на задачах, где порядок и "дальние" связи важнее, чем простое соседство.

    Что принимается на вход:
    ------------------------
    - x: shape [batch, seq_len, embed_dim] — обычный batched-embed тензор.
    - mask (опционально): shape [seq_len, seq_len] — маска для автогерерации или causal attention.

    Какие параметры важны:
    ----------------------
    - num_heads: сколько attention heads внутри (обычно 4, 8, 16...).
    - embed_dim: исходная размерность входного тензора.
    - head_size: размер одной attention-head (обычно embed_dim // num_heads).
    - max_seq_len: максимальная длина последовательности для маски.

    Что возвращает:
    ---------------
    - output: shape [batch, seq_len, embed_dim] — результат применения всех attention heads.
    - (опционально) cache: кэш для Q/K/V (нужно для генерации по одному токену).

    Особенности реализации:
    -----------------------
    - Оптимизированно работает через матричные умножения (без python for циклов!).
    - Включена поддержка causal attention (маска, предотвращающая «заглядывание в будущее»).
    - Является ядром любого трансформера (и LLM!).

    Пример использования:
    ---------------------
        >>> attn = MultiHeadAttention(num_heads=8, embed_dim=256, head_size=32, max_seq_len=1024)
        >>> x = torch.randn(2, 128, 256)  # [batch, seq_len, embed_dim]
        >>> context, _ = attn(x)
        >>> print(context.shape)  # torch.Size([2, 128, 256])

    Где прочитать подробнее:
    -------------------------
    - Attention is All You Need (Vaswani et al, 2017): https://arxiv.org/abs/1706.03762
    - Illustrated Transformer (blog): https://jalammar.github.io/illustrated-transformer/
    """

    def __init__(
        self,
        num_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        rope: RoPE = None,
        dropout: float = 0.1,
    ):
        """
        Конструктор многоголового внимания (MultiHeadAttention).

        Здесь создаются все параметры и внутренние слои для эффективного параллельного внимания (attention) сразу из нескольких "голов".

        Аргументы:
        ----------
        num_heads : int
            Сколько attention-heads будет внутри слоя.
            Каждая “голова” учится видеть уникальные зависимости в тексте. Обычно это 4, 8, 16 и т.п.
            Чем больше голов — тем богаче контекст, но и больше памяти.
        emb_size : int
            Сколько float-значений в каждом входном векторе (размерность embedding).
            Обычно это 256, 512, 768, 1024 и т.д.
        head_size : int
            Сколько компонент будет у каждой головы внимания.
            Важно: num_heads * head_size должно ровно совпадать с emb_size!
            Обычно head_size = emb_size // num_heads.
        max_seq_len : int
            Максимально допустимая длина последовательности для attention/маски/генерации.
            Определяет размер буферов для causal mask.
        rope : RoPE, по умолчанию None
            Объект Rotary Positional Encoding (если хотите привнести продвинутое позиционное кодирование в attention).
            Не обязателен, но нужен для современных LLM (Llama, Mistral и пр.).
        dropout : float, по умолчанию 0.1
            Величина dropout (регуляризации) — помогает борьбе с переобучением. Чем больше, тем сильнее регуляризация.

        Внутри конструктора происходит:
        -------------------------------
        - Создаются три линейных слоя для Q, K, V (“где смотреть” и “что вытаскивать” в attention).
        - Генерируется нижнетреугольная causal-маска (запрещает видеть будущее для автогерерации).
        - Создаётся финальный линейный слой для склейки всех голов в одно пространство emb_size.
        - Вводится dropout (случайное зануление, чтобы не было сильной зависимости внимания к отдельным "плейсам").

        Пример:
        -------
            >>> attn = MultiHeadAttention(num_heads=8, emb_size=256, head_size=32, max_seq_len=1024)
        """
        super().__init__()
        self._num_heads = num_heads
        self._head_size = head_size
        self._max_seq_len = max_seq_len
        self._rope = rope

        self._q = nn.Linear(emb_size, num_heads * head_size)
        self._k = nn.Linear(emb_size, num_heads * head_size)
        self._v = nn.Linear(emb_size, num_heads * head_size)

        # Создание causal маски
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer(
            "_tril_mask", mask.bool() if hasattr(torch, "bool") else mask.byte()
        )
        
        self._layer = nn.Linear(head_size * num_heads, emb_size)
        self._dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        use_cache: bool = True,
        cache: list = None,
    ):
        """
        Основной шаг \"многоголового внимания\": находит взаимосвязи между токенами 
        в последовательности сразу из нескольких “ракурсов” (attention heads).

        Что делает этот метод:
        ----------------------
        - Для каждого токена сравнивает его с остальными во входной последовательности.
        - Делает это одновременно через несколько attention heads (каждая head видит текст по-своему).
        - Итоговое “внимание” — это взвешенная сумма других токенов (контекста) для каждого токена.
        - Можно использовать кэш для генерации длинных последовательностей по одному токену (ускоряет инференс).

        Аргументы:
        ----------
        x : torch.Tensor
            Входной тензор формы [batch, seq_len, emb_size].
            Это ваши входные эмбеддинги (обычно после token + positional embedding).
        mask : torch.Tensor, опционально
            Матрица формы [seq_len, seq_len], задающая “разрешения” — кто может смотреть на кого (например, causal mask).
            Если не указана — используется внутренняя маска (например, для autoregressive генерации).
        use_cache : bool, по умолчанию True
            Нужно ли использовать кэш для KV attention (важно для ускорения генерации по одному токену).
        cache : list, опционально
            Предыдущий кэш Key/Value — для генерации текста по частям.

        Возвращает:
        -----------
        - output: torch.Tensor формы [batch, seq_len, emb_size] — результат применения multi-head attention.
        - kv_caches: список новых KV для кэширования при генерации (или None).

        Важно:
        -------
        - Shape входа всегда [batch, seq_len, emb_size], выход тот же.
        - При seq_len > max_seq_len выбросит ошибку (безопасно для контроля переполнения буферов).
        - При использовании use_cache=True кешируется только последние токены (актуально для LLM).

        Пример:
        -------
            >>> attn = MultiHeadAttention(num_heads=8, emb_size=256, head_size=32, max_seq_len=1024)
            >>> x = torch.randn(2, 100, 256)
            >>> y, kv_cache = attn(x)
            >>> print(y.shape)  # torch.Size([2, 100, 256])
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
        q = q.reshape(batch_size, seq_len, self._num_heads, self._head_size)
        k = k.reshape(batch_size, seq_len, self._num_heads, self._head_size)
        v = v.reshape(batch_size, seq_len, self._num_heads, self._head_size)
        

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
            # ✅ Применяем RoPE к Q и K (НЕ к V!)
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

        # Если cache пришел, то маску не накладываем. Иначе наложите на матрицу внимания треугольную маску, созданную при инициализации. Все скрытые значения должны быть приведены к минус бесконечности: float('-inf').
        if cache is None:
            scores = scores.masked_fill(
                ~self._tril_mask[:seq_len, :seq_len], float("-inf")
            )

        # Применить к матрице внимания (построчно) функцию Softmax.
        weights = F.softmax(scores, dim=-1)

        # Перемножим матрицу внимания и матрицу значения.
        x_out = weights @ v  # [B, T, hs]

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
        final_output = self._dropout(projected_output)

        if use_cache is True:
            return (final_output, (k, v))
        else:
            return (final_output, None)
