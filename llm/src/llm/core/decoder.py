from torch import nn
import torch
from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention

class Decoder(nn.Module):
    """
    Декодер трансформера - ключевой компонент архитектуры Transformer.
    
    Предназначен для:
    - Обработки последовательностей с учетом контекста (самовнимание)
    - Постепенного генерирования выходной последовательности
    - Учета масок для предотвращения "заглядывания в будущее"

    Алгоритм работы:
    1. Входной тензор (batch_size, seq_len, emb_size)
    2. Многоголовое внимание с residual connection и LayerNorm
    3. FeedForward сеть с residual connection и LayerNorm
    4. Выходной тензор (batch_size, seq_len, emb_size)

    Основные характеристики:
    - Поддержка масок внимания
    - Residual connections для стабилизации градиентов
    - Layer Normalization после каждого sub-layer
    - Конфигурируемые параметры внимания

    Примеры использования:

    1. Базовый случай:
    >>> decoder = Decoder(num_heads=8, emb_size=512, head_size=64, max_seq_len=1024)
    >>> x = torch.randn(1, 10, 512)  # [batch, seq_len, emb_size]
    >>> output = decoder(x)
    >>> print(output.shape)
    torch.Size([1, 10, 512])

    2. С маской внимания:
    >>> mask = torch.tril(torch.ones(10, 10))  # Нижнетреугольная маска
    >>> output = decoder(x, mask)

    3. Инкрементальное декодирование:
    >>> for i in range(10):
    >>>     output = decoder(x[:, :i+1, :], mask[:i+1, :i+1])
    """
    def __init__(self, 
        num_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        dropout: float = 0.1
    ):
        """
        Инициализация декодера.

        Параметры:
            num_heads: int - количество голов внимания
            emb_size: int - размерность эмбеддингов
            head_size: int - размерность каждой головы внимания
            max_seq_len: int - максимальная длина последовательности
            dropout: float (default=0.1) - вероятность dropout
        """
        super().__init__()
        self._heads = MultiHeadAttention(
            num_heads=num_heads, 
            emb_size=emb_size, 
            head_size=head_size, 
            max_seq_len=max_seq_len, 
            dropout=dropout
        )
        self._ff = FeedForward(emb_size=emb_size, dropout=dropout)
        self._norm1 = nn.LayerNorm(emb_size)
        self._norm2 = nn.LayerNorm(emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Прямой проход через декодер.

        Вход:
            x: torch.Tensor - входной тензор [batch_size, seq_len, emb_size]
            mask: torch.Tensor (optional) - маска внимания [seq_len, seq_len]

        Возвращает:
            torch.Tensor - выходной тензор [batch_size, seq_len, emb_size]

        Алгоритм forward:
        1. Применяем MultiHeadAttention к входу
        2. Добавляем residual connection и LayerNorm
        3. Применяем FeedForward сеть
        4. Добавляем residual connection и LayerNorm
        """
        # Self-Attention блок
        attention = self._heads(x, mask)
        out = self._norm1(attention + x)
        
        # FeedForward блок
        ffn_out = self._ff(out)
        return self._norm2(ffn_out + out)