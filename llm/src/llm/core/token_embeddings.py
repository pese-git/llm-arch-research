import torch
from torch import nn
from torch import Tensor


class TokenEmbeddings(nn.Module):
    """
    TokenEmbeddings — обучаемый слой эмбеддингов для токенов (слов, сабслов, байтов и т.д.) в трансформерах.

    Назначение:
    -----------
    - Преобразует каждый целочисленный индекс-токен из словаря (vocab) в обучаемый dense-вектор фиксированной длины.
    - Это "входной слой" для любой нейросетевой языковой модели: позволяет работать с текстом как с матрицей чисел, а не с индексами/категориальными значениями.
    - Обеспечивает возможность end-to-end обучения embedding-матрицы совместно с целью модели.

    Мотивация и особенности:
    ------------------------
    - Каждый токен (индекс) получает свой learnable embedding (float-вектор).
    - Размерность слоя: [vocab_size, emb_size] (матрица эмбеддингов).
    - Веса эмбеддингов инициализируются случайно и обучаются вместе с остальной моделью.
    - Аналог таблицы эмбеддингов в word2vec/fastText, но управляется end-to-end.
    - Могут использоваться с любым токенизатором (BPE, SentencePiece, WordPiece и др.).

    Формула:
    --------
        emb(x) = W[x], где W — матрица размера [vocab_size, emb_dim], x — индексы shape [batch, seq_len]
        На выходе: тензор [batch, seq_len, emb_dim]

    Args:
    -----
    vocab_size: int — размер словаря/алфавита (количество уникальных токенов)
    emb_size: int — размерность (длина) эмбеддинговых векторов (обычно 256/512/1024...)

    Пример:
    -------
        >>> embedding = TokenEmbeddings(vocab_size=5000, emb_size=256)
        >>> tokens = torch.tensor([[12, 47, 301], [6, 88, 413]])
        >>> vecs = embedding(tokens)
        >>> print(vecs.shape)  # torch.Size([2, 3, 256])

    References:
    -----------
    - Mikolov et al., "Efficient Estimation of Word Representations in Vector Space (word2vec)", 2013
    - Vaswani et al., "Attention is All You Need", 2017: https://arxiv.org/abs/1706.03762
    - BPE, SentencePiece overviews: https://huggingface.co/docs/transformers/tokenizer_summary
    """

    def __init__(self, vocab_size: int, emb_size: int):
        """
        Инициализация слоя эмбеддингов.

        Args:
        -----
        vocab_size: int
            Размер словаря (уникальных токенов/индексов).
        emb_size: int
            Длина эмбеддингового вектора для каждого токена.

        Внутри:
        -------
        - Создаёт nn.Embedding с [vocab_size, emb_size] learnable весами.
        """
        super().__init__()
        self._embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=emb_size
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Получить эмбеддинги для входных токенов.

        Args:
        -----
        x : torch.Tensor
            Тензор shape [...], содержащий индексы токенов (каждое значение от 0 до vocab_size-1).

        Returns:
        --------
        torch.Tensor — тензор обычной формы [..., emb_size] (на каждую позицию — свой embedding-вектор).

        Пример:
        -------
            >>> embedding = TokenEmbeddings(vocab_size=100, emb_size=64)
            >>> tokens = torch.tensor([[0, 99, 5]])
            >>> vecs = embedding(tokens)  # [1, 3, 64]
        """
        return self._embedding(x)

    @property
    def num_embeddings(self) -> int:
        """Возвращает размер словаря (количество уникальных токенов)."""
        return self._embedding.num_embeddings

    @property
    def embedding_dim(self) -> int:
        """Возвращает размерность эмбеддингов (длина вектора каждого токена)."""
        return self._embedding.embedding_dim
