from torch import nn
import torch
from .head_attention import HeadAttention

class MultiHeadAttention(nn.Module):
    """
    Реализация механизма многоголового внимания (Multi-Head Attention) из архитектуры Transformer.

    Основные характеристики:
    - Параллельная обработка входных данных несколькими головами внимания
    - Поддержка маскирования (causal mask и пользовательские маски)
    - Финальная проекция с dropout регуляризацией

    Математическое описание:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    где head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

    Примеры использования:

    1. Базовый пример:
    >>> mha = MultiHeadAttention(num_heads=8, emb_size=512, head_size=64, max_seq_len=1024)
    >>> x = torch.randn(2, 50, 512)  # [batch_size, seq_len, emb_size]
    >>> output = mha(x)  # [2, 50, 512]

    2. С использованием маски:
    >>> mask = torch.tril(torch.ones(50, 50))  # Causal mask
    >>> output = mha(x, mask)

    3. Интеграция в Transformer:
    >>> # В составе Transformer слоя
    >>> self.attention = MultiHeadAttention(...)
    >>> x = self.attention(x, mask)
    """
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, dropout: float = 0.1):
        """
        Инициализация многоголового внимания.

        Параметры:
            num_heads (int): Количество голов внимания. Типичные значения: 4-16
            emb_size (int): Размерность входных и выходных эмбеддингов
            head_size (int): Размерность каждой головы внимания (обычно emb_size // num_heads)
            max_seq_len (int): Максимальная длина последовательности
            dropout (float): Вероятность dropout (по умолчанию 0.1)

        Контрольные значения:
            - num_heads * head_size должно равняться emb_size
            - head_size обычно выбирают 32-128
            - max_seq_len зависит от задачи (512 для BERT, 2048 для GPT-3)
        """
        super().__init__()
        self._heads = nn.ModuleList([
            HeadAttention(
                emb_size=emb_size, 
                head_size=head_size, 
                max_seq_len=max_seq_len
            ) for _ in range(num_heads)
        ])
        self._layer = nn.Linear(head_size * num_heads, emb_size)
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, use_cache: bool = True, cache: list = None):
        """
        Прямой проход через слой многоголового внимания.

        Подробное описание преобразований тензоров:
        1. Входной тензор [batch_size, seq_len, emb_size] разделяется на N голов:
           - Каждая голова получает тензор [batch_size, seq_len, head_size]
        2. Каждая голова вычисляет attention:
           - Вход: [batch_size, seq_len, head_size]
           - Выход: [batch_size, seq_len, head_size]
        3. Конкатенация результатов:
           - Объединенный выход: [batch_size, seq_len, num_heads * head_size]
        4. Линейная проекция:
           - Выход: [batch_size, seq_len, emb_size]
        5. Применение dropout

        Аргументы:
            x (torch.Tensor): Входной тензор формы [batch_size, seq_len, emb_size]
            mask (torch.Tensor, optional): Маска внимания формы [seq_len, seq_len]

        Возвращает:
            torch.Tensor: Выходной тензор формы [batch_size, seq_len, emb_size]

        Пример преобразований для emb_size=512, num_heads=8:
        Вход: [4, 100, 512]
        -> Каждая голова: [4, 100, 64]
        -> После внимания: 8 x [4, 100, 64] 
        -> Конкатенация: [4, 100, 512]
        -> Проекция: [4, 100, 512]
        -> Dropout: [4, 100, 512]
        """
        # 1. Вычисляем attention для каждой головы
        attention_results = []
        for i, head in enumerate(self._heads):
            head_cache = cache[i] if cache is not None else None
            result = head(x, use_cache=use_cache, cache=head_cache)
            attention_results.append(result)
        
        outputs, caches = zip(*attention_results)
        attention_outputs = list(outputs)
        kv_caches = list(caches)
        
        # 2. Объединяем результаты всех голов
        concatenated_attention = torch.cat(attention_outputs, dim=-1)
        
        # 3. Проецируем в пространство эмбеддингов
        projected_output = self._layer(concatenated_attention)
        
        # 4. Применяем dropout для регуляризации
        final_output = self._dropout(projected_output)
        
        if use_cache is True:
            return (final_output, kv_caches)
        else:
            return (final_output, None)
