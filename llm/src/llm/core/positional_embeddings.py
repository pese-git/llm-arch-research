import torch
from torch import nn, Tensor


class PositionalEmbeddings(nn.Module):
    """
    Обучаемые позиционные эмбеддинги (learnable positional embeddings).

    Позиционные эмбеддинги используются в нейросетях для передачи информации
    о позиции элементов в последовательности (например, в Transformer).

    Научная суть:
        - Трансформеры не используют рекуррентность, а значит сами по себе не различают порядок слов.
        - Позиционные эмбеддинги добавляются к токеновым, чтобы сеть понимала, в каком месте последовательности находится каждый токен.
        - Обычно реализуются как отдельная матрица (nn.Embedding), которая обучается вместе с моделью (это learnable вариант, как в GPT и BERT).

    Args:
        max_seq_len (int): максимальная длина последовательности
        emb_size (int): размер вектора позиции

    Пример использования:
        >>> pos_encoder = PositionalEmbeddings(max_seq_len=100, emb_size=256)
        >>> # Получить эмбеддинги для последовательности из 10 элементов
        >>> embeddings = pos_encoder(10)  # Tensor shape: [10, 256]
        >>> # Использование в модели
        >>> class MyModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.pos_emb = PositionalEmbeddings(100, 256)
        ...     def forward(self, x):
        ...         pos = self.pos_emb(x.size(1))
        ...         return x + pos  # Добавляем позиционную информацию
    """

    def __init__(self, max_seq_len: int, emb_size: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.embedding = nn.Embedding(
            num_embeddings=max_seq_len, embedding_dim=emb_size
        )

    def forward(self, seq_len: int, start_pos: int = 0) -> Tensor:
        """
        Возвращает позиционные эмбеддинги для заданной длины последовательности.

        Args:
            seq_len (int): Длина последовательности (1 <= seq_len <= max_seq_len)

        Returns:
            Tensor: Тензор позиционных эмбеддингов формы [seq_len, emb_size]

        Raises:
            IndexError: Если seq_len выходит за допустимые границы

        Пример:
            >>> pos_encoder = PositionalEmbeddings(100, 64)
            >>> emb = pos_encoder(10)  # Тензор 10x64
        """
        if seq_len < 1 or seq_len > self.max_seq_len:
            raise IndexError(f"Длина {seq_len} должна быть от 1 до {self.max_seq_len}")
        if start_pos == 0:
            positions = torch.arange(seq_len, device=self.embedding.weight.device)
        else:
            positions = torch.arange(
                start=start_pos,
                end=start_pos + seq_len,
                device=self.embedding.weight.device,
            )
        return self.embedding(positions)
