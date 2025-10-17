"""
Rotary Positional Embeddings (RoPE)
===================================

Что такое RoPE?
----------------
RoPE — это способ "вписать" информацию о позиции токенов в скрытые вектора модели трансформера.
Вместо простого сложения с абсолютным positional embedding, RoPE использует вращения векторов (как поворот стрелки на круге) внутри каждого attention head. Каждый элемент пары (вектор четного и нечетного индекса) поворачивается на угол, зависящий от позиции токена.

Зачем это?
-----------
- RoPE реализует **относительное позиционное кодирование**: модель может сравнивать "расстояния" между токенами, а не просто помнить положение.
- Такое кодирование **улучшает генерацию длинных последовательностей** и перенос модели на тексты большей длины, чем были в обучении.
- Форма векторов и длина (норма) НЕ искажаются.

Как это работает? (главная формула)
-------------------------------------
Для каждой позиции m и пары компонент (2i, 2i+1) внутри head применяются:

    θ_i = base^(-2i / d)
    q'_{m,2i}   = q_{m,2i}   * cos(m * θ_i) - q_{m,2i+1} * sin(m * θ_i)
    q'_{m,2i+1} = q_{m,2i+1} * cos(m * θ_i) + q_{m,2i}   * sin(m * θ_i)
    
где d — размерность "головы" attention (head_size), base обычно 10_000.

То есть, берём каждый "вектор" (в рамках head), делим на четные/нечетные части и поворачиваем их на уникальный угол, связанный с позицией/частотой.

Архитектурные детали:
---------------------
- Ваш тензор должен быть строго 4-мерным: [batch, num_heads, seq_len, head_size].
- Размер head_size должен быть чётным!
- RoPE применяется отдельно к **Q** и **K** в механизме внимания (но не к V).

Где об этом читать:
-------------------
- RoFormer: Enhanced Transformer with Rotary Position Embedding  
  https://arxiv.org/abs/2104.09864
- Llama: Open and Efficient Foundation Language Models  
  https://arxiv.org/abs/2302.13971
- Визуализация позиционных кодировок:  
  https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

Пример использования:
---------------------
>>> rope = RoPE(head_size=64, max_seq_len=2048)
>>> x = torch.randn(2, 8, 128, 64)  # [batch, num_heads, seq_len, head_size]
>>> x_enc = rope(x)  # здесь вектор x обогатится позиционной информацией

"""

import torch
from torch import nn
from typing import Optional


class RoPE(nn.Module):
    """
    Реализация RoPE (Rotary Positional Embeddings) для self-attention в трансформерах.

    Этот слой добавляет позиционную информацию к векторам внимания (Q, K) — 
    не с помощью простого сложения с positional embedding, а с помощью математического 
    вращения (как если бы вы крутили стрелку на круге) для каждой пары компонент 
    (even/odd) в каждом attention head.

    Формула (для каждого токена и каждой пары компонент внутри head):
        θ_i = base^(-2i / d)
        out_{m,2i}   = x_{m,2i}   * cos(m * θ_i) - x_{m,2i+1} * sin(m * θ_i)
        out_{m,2i+1} = x_{m,2i+1} * cos(m * θ_i) + x_{m,2i}   * sin(m * θ_i)
        где d — head_size, base обычно 10_000, степень i по head axis.

    Какие входы принимает:
    ----------------------
    - x: обязательно размерности [batch, num_heads, seq_len, head_size]!
    - head_size (размер внимания) должен быть чётным.
    - start_pos: опционально, позволяет сдвигать позиционный offset для генерации с кэшем.

    Что возвращает:
    ---------------
    - Тот же тензор (x), только со встроенной позиционной информацией (“повёрнутый” RoPE-кодировкой).
    - Форма и тип выходного тензора не меняются.

    Где используется:
    -----------------
    - В любых современных LLM (Llama, Mistral, GPT-NeoX и др.) для повышения устойчивости и generalization transformer's attention.

    Пример использования:
    ---------------------
        >>> rope = RoPE(head_size=64, max_seq_len=2048)
        >>> x = torch.randn(2, 8, 128, 64)  # (batch, num_heads, seq_len, head_size)
        >>> x_encoded = rope(x)

    Подробнее про математику и примеры с визуализацией:
    ---------------------------------------------------
    - RoFormer: https://arxiv.org/abs/2104.09864
    - Llama: https://arxiv.org/abs/2302.13971
    - Демонстрация наглядно: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    """

    def __init__(self, head_size: int, max_seq_len: int, base: int = 10_000):
        """
        Инициализация объекта RoPE — настраивает и предвычисляет все необходимые
        параметры для ротационного позиционного кодирования.

        Аргументы:
        ----------
        head_size : int
            Размер одного attention head (последнего измерения вектора) — сколько компонент
            (float-значений) отвечает за одну "голову". Должен быть ЧЁТНЫМ числом, иначе RoPE не применим.
            Обычно head_size = embed_dim // num_heads.
        max_seq_len : int
            Максимальная длина последовательности, которую RoPE сможет обработать.
            Если ваш текст длиннее этого числа — будет ошибка! Например, для GPT2 обычно 1024, у LLaMA — до 4096.
            Это число определяет размер внутренних буферов cos/sin.
        base : int, по умолчанию 10_000
            База для вычисления частот вращения (θ_i) для каждой компоненты.
            В оригинальных статьях почти всегда используют base=10000.
            Менять этот параметр не нужно, если вы не исследуете математические детали.

        Что происходит внутри:
        ----------------------
        - Проверяется чётность head_size.
        - Для каждого возможного положения в пределах max_seq_len и каждой пары component высчитываются уникальные cos/sin значения (матрицы частот).
        - Эти матрицы используются далее для быстрого наложения позиционного "вращения" токенов внутри attention.
        """
        super().__init__()
        assert head_size % 2 == 0, "head_size должен быть четным"

        # Вычисление частот: θ_i = base^(-2i/d) для i ∈ [0, d/2-1]
        freqs = 1.0 / (base ** (2 * torch.arange(head_size // 2).float() / head_size))

        # Позиции от 0 до max_seq_len-1
        positions = torch.arange(max_seq_len).float()

        # Внешнее произведение: m * θ_i для всех позиций и частот
        freq_matrix = positions.unsqueeze(1) * freqs.unsqueeze(0)

        # Предвычисление матриц косинусов и синусов
        self.register_buffer("cos_matrix", torch.cos(freq_matrix))
        self.register_buffer("sin_matrix", torch.sin(freq_matrix))

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Применяет ротационное позиционное кодирование (RoPE) к входному тензору.

        Что делает эта функция:
        -----------------------
        Для каждого токена в последовательности внутри каждого attention head
        "поворачивает" его вектор в подпространстве (even/odd пар) на свой уникальный угол,
        зависящий от позиции токена. Это позволяет attention "понимать расстояния" между токенами.

        Аргументы:
        ----------
        x : torch.Tensor
            Входной тензор строго формы [batch, num_heads, seq_len, head_size].
            Это обычно либо Q, либо K из механизма внимания.
        start_pos : int, по умолчанию 0
            Сдвиг начала позиции (нужно при генерации с кэшем, почти всегда оставить 0 если не пишете автогенератор).

        Возвращает:
        -----------
        torch.Tensor с теми же формой и типом, что и x, но уже с наложенным позиционным кодированием.

        Важно:
        -------
        - Если передан тензор не 4D, будет выброшено исключение!
        - Не изменяет значения "на месте", всегда возвращает новый тензор.

        Пример:
        -------
            >>> rope = RoPE(head_size=64, max_seq_len=1024)
            >>> q = torch.randn(2, 8, 32, 64)  # batch, num_heads, seq_len, head_size
            >>> q_rope = rope(q)
        """
        assert x.ndim == 4, "RoPE поддерживает только 4D-вход [batch, num_heads, seq_len, head_size]"
        batch_size, num_heads, seq_len, head_size = x.shape

        # Берем нужную часть матриц и приводим к типу x
        cos = self.cos_matrix[start_pos:start_pos+seq_len].to(x.dtype)  # [seq_len, head_size//2]
        sin = self.sin_matrix[start_pos:start_pos+seq_len].to(x.dtype)  # [seq_len, head_size//2]

        # Явное изменение формы для broadcasting
        cos = cos.reshape(1, 1, seq_len, head_size // 2)
        sin = sin.reshape(1, 1, seq_len, head_size // 2)

        # Разделяем на четные и нечетные компоненты по ПОСЛЕДНЕМУ измерению
        x_even = x[..., 0::2]  # [batch_size, num_heads, seq_len, head_size//2]
        x_odd = x[..., 1::2]   # [batch_size, num_heads, seq_len, head_size//2]

        # Применяем поворот: q' = q * cos(mθ) + rotate(q) * sin(mθ)
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # Объединяем обратно в исходную размерность
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_rotated = x_rotated.flatten(-2)  # [batch_size, seq_len, head_size]

        return x_rotated
