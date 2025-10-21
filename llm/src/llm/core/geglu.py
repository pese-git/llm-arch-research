import torch
from torch import nn
from llm.core.gelu import GELU

class GeGLU(nn.Module):
    """
    GeGLU (Gated GELU Linear Unit) — эффективная нелинейность для feed-forward блоков в современных трансформерах.

    Назначение:
    -----------
    GeGLU — это вариант GLU (Gated Linear Unit), где «шлюз» реализован через GELU-активацию,
    а затем поэлементно перемножается с другим линейным преобразованием. Такой gating-механизм позволяет повысить
    выразительность MLP-блока и ускорить обучение, что подтверждено экспериментами на LLM (см. PaLM, LLaMA, T5).

    Формула:
    --------
        GeGLU(x) = GELU(W_g x + b_g) ⊙ (W_u x + b_u) W_d + b_d
    (здесь W_g, W_u, W_d — матрицы весов; GELU применяется к одной ветке, ⊙ — поэлементное умножение)

    Структура блока:
    ----------------
    1. gate = GELU(Linear_gate(x))      # ветка gating-а, shape [batch, seq, 4×emb]
    2. up   = Linear_up(x)              # ветка передачи, shape [batch, seq, 4×emb]
    3. out  = gate * up                 # поэлементно, реализует динамическую фильтрацию информации
    4. out  = Linear_down(out)          # проекция обратно в исходное пространство
    5. out  = Dropout(out)              # регуляризация

    Основные преимущества:
    ----------------------
    - Позволяет эффективно обучать глубокие трансформеры (см. PaLM, LLaMA).
    - Обеспечивает плавные градиенты за счёт GELU и gating-эффекта.
    - Используется во многих современных LLM вместо обычных FFN или простых GLU.

    Аргументы конструктора:
    -----------------------
    emb_size : int
        Размер эмбеддинга (input и output).
    dropout : float, по умолчанию 0.1
        Dropout к финальному выходу (примерно 0.1-0.2 для регуляризации).

    Пример использования:
    ---------------------
        >>> geglu = GeGLU(emb_size=512, dropout=0.1)
        >>> x = torch.randn(8, 16, 512)
        >>> y = geglu(x)
        >>> print(y.shape)   # torch.Size([8, 16, 512])

    Литература:
    -----------
    - Shazeer N., "GLU Variants Improve Transformer", 2020: https://arxiv.org/abs/2002.05202
    - PaLM: https://arxiv.org/abs/2204.02311
    - LLaMA: https://arxiv.org/abs/2302.13971
    - T5: https://arxiv.org/abs/1910.10683
    """
    def __init__(self, emb_size: int, dropout: float = 0.1):
        """
        Инициализация блока GeGLU.

        Создаёт три последовательных линейных слоя и задаёт GELU в качестве активации для ветки gating,
        а также финальный dropout. Все размеры согласованы так, чтобы реализовать формулу GeGLU (см. описание класса).

        Аргументы:
        ----------
        emb_size : int
            Размерность входного и выходного скрытого пространства (hidden size).
            Данная величина определяет размерность эмбеддинга для всех внутренних вычислений.
            Обычно равна размеру скрытого слоя трансформера.

        dropout : float, по умолчанию 0.1
            Вероятность отключения нейронов после выхода из блока (регуляризация).
            Рекомендуемое значение: 0.1 (или чуть больше для небольших моделей).

        Внутри:
        -------
        - self._gate: Linear слой размерности [emb_size, 4 * emb_size], ветка gating (проходит через GELU)
        - self._up:   Linear слой размерности [emb_size, 4 * emb_size], ветка передачи ("пропускная")
        - self._down: Linear слой сжатия обратно к emb_size
        - self._activation: Активация GELU для gating-ветки
        - self._dropout: Dropout для выходного тензора

        Пример:
        -------
            >>> block = GeGLU(emb_size=256, dropout=0.1)
            >>> print(block)
        """
        super().__init__()

        self._gate = nn.Linear(emb_size, 4 * emb_size)
        self._up = nn.Linear(emb_size, 4 * emb_size)
        self._down = nn.Linear(4 * emb_size, emb_size)
        self._activation = GELU()
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Прямой проход (forward) через блок GeGLU.

        Для входного тензора скрытых состояний x реализует последовательность операций:
        1. Gating-ветка: линейное преобразование → GELU-активация
        2. Пропускная ветка: линейное преобразование
        3. Поэлементное умножение результатов обеих веток (gating)
        4. Проекция через Linear обратно к emb_size
        5. Dropout результата для регуляризации

        Математически:
        --------------
            gate = GELU(W_g·x + b_g)
            up   = W_u·x + b_u
            out  = gate * up
            out  = W_d·out + b_d
            out  = Dropout(out)

        Аргументы:
        ----------
        x : torch.Tensor
            Входной тензор формы [batch_size, seq_len, emb_size]
            (или любой совместимой формы, где последняя ось — emb_size).

        Возвращает:
        -----------
        torch.Tensor :
            Тензор той же формы [batch_size, seq_len, emb_size], прошедший через структуру GeGLU.

        Пример:
        -------
            >>> y = geglu(x)
            >>> print(y.shape)  # [batch_size, seq_len, emb_size]

        Примечания:
        -----------
        - Ветка gating строит masк для динамической фильтрации информации.
        - Такой тип блока эффективно используется как замена обычного FFN в современных LLM.
        """
        gate_out = self._gate(x)                          # [batch, seq, 4*emb]
        activation_out = self._activation(gate_out)       # [batch, seq, 4*emb]
        up_out = self._up(x)                              # [batch, seq, 4*emb]
        out = up_out * activation_out                     # поэлементное!
        out = self._down(out)                             # [batch, seq, emb]
        return self._dropout(out)

