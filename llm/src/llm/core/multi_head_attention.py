from torch import nn
import torch
from .head_attention import HeadAttention
from .rope import RoPE

class MultiHeadAttention(nn.Module):
    """
    Мультиголовый (многоголовый) механизм внимания — ключевой компонент любого Transformer.

    Научная суть:
        - Модель параллельно агрегирует информацию через несколько подпространств (головы),
          чтобы видеть разные связи в последовательности (разный контекст, локально/глобально).
        - Каждый attention блок работает независимо, выход конкатенируется.
        - Механизм предложен в статье "Attention is All You Need" (Vaswani et al., 2017).
        
        Формула внимания для одной головы:
            Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))·V
        Мультиголовый:
            MultiHead(Q, K, V) = Concat([head_i])*W^O

    Args:
        num_heads (int): количество attention "голов"
        emb_size (int): размерности входа и выхода
        head_size (int): размер одной attention-головы (emb_size/num_heads)
        max_seq_len (int): максимальная длина последовательности
        rope (RoPE, optional): если задан, используется Rotary Positional Encoding
        dropout (float): вероятность регуляризации

    Пример использования:
        >>> mha = MultiHeadAttention(num_heads=8, emb_size=512, head_size=64, max_seq_len=1024)
        >>> x = torch.randn(2, 50, 512)
        >>> out, cache = mha(x)
        >>> print(out.shape)
    """
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, rope: RoPE = None, dropout: float = 0.1):
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
                max_seq_len=max_seq_len,
                rope=rope,
            ) for _ in range(num_heads)
        ])
        self._layer = nn.Linear(head_size * num_heads, emb_size)
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, use_cache: bool = True, cache: list = None):
        """
        Прямой проход (forward):
        Для каждого токена оценивает "важность" остальных токенов сразу через несколько attention-блоков.

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
        
        Args:
            x (Tensor[float]): [batch, seq_len, emb_size] — вход
            mask (Optional[Tensor[bool]]): маска позиции [seq_len, seq_len]
            use_cache (bool): использовать ли key-value кэш (для генерации)
            cache (list): предыдущие значения KV для ускорения

        Returns:
            out (Tensor[float]): [batch, seq_len, emb_size] — результат MHA
            kv_caches (list): списки новых KV-кэшей (если используется)

        Типичный паттерн:
            Вход: [batch, seq, emb] → N голов [batch, seq, head_size] →
                → concat [batch, seq, N*head_size] → проекция → dropout

        Пример преобразований для emb_size=512, num_heads=8:
        Вход: [4, 100, 512]
        -> Каждая голова: [4, 100, 64]
        -> После внимания: 8 x [4, 100, 64] 
        -> Конкатенация: [4, 100, 512]
        -> Проекция: [4, 100, 512]
        -> Dropout: [4, 100, 512]
        
        Пример:
            >>> out, caches = mha(x)
            >>> out.shape   # [batch, seq_len, emb_size]
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
