import torch
from torch import nn
from torch import Tensor

class TokenEmbeddings(nn.Module):
    """
    Токеновые эмбеддинги — обучаемые векторные представления для каждого токена словаря.

    Преобразует целочисленные индексы токенов в обучаемые векторные представления фиксированного размера.
    Обычно используется как первый слой в нейронных сетях для задач NLP.
    
    Научная суть:
        - Первый шаг для любого NLP-модуля: вместо индекса токена подаём его dense-вектор.
        - Эти вектора изучаются в процессе обучения и отражают скрытые взаимосвязи между токенами.
        - Позволяют обрабатывать тексты как матрицу чисел, а не как символы или индексы.
        - Аналог словарных эмбеддингов в word2vec, но обучаются энд-ту-энд с моделью.

    Args:
        vocab_size (int): размер словаря (количество уникальных токенов)
        emb_size (int): размерность эмбеддинга (длина вектора)

    Примечание:
        - Индексы должны быть в диапазоне [0, vocab_size-1]
        - Эмбеддинги инициализируются случайно и обучаются в процессе тренировки модели
    
    Пример:
        >>> emb = TokenEmbeddings(vocab_size=10000, emb_size=256)
        >>> tokens = torch.tensor([[1, 2, 3]])
        >>> vecs = emb(tokens)
        >>> vecs.shape  # torch.Size([1, 3, 256])
    """
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self._embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_size
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._embedding(x)

    @property
    def num_embeddings(self) -> int:
        """Возвращает размер словаря"""
        return self._embedding.num_embeddings

    @property
    def embedding_dim(self) -> int:
        """Возвращает размерность эмбеддингов"""
        return self._embedding.embedding_dim


if __name__ == "__main__":
    # Пример использования
    embedding = TokenEmbeddings(vocab_size=100, emb_size=128)

    # Создаем тензор с индексами в пределах vocab_size (0-99)
    tensor = torch.tensor([
        [11, 45, 76, 34],
        [34, 67, 45, 54]
    ])

    # Проверяем индексы
    if (tensor >= 100).any():
        raise ValueError("Some indices are out of vocabulary range (vocab_size=100)")

    output = embedding(tensor)
    print("Embeddings shape:", output.shape)
    print(f"{output.shape} | {output.mean().item():.11f}")  # Формат как в ТЗ