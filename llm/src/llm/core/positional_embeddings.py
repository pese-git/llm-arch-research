import torch
from torch import nn, Tensor


class PositionalEmbeddings(nn.Module):
    """
    PositionalEmbeddings — классические позиционные эмбеддинги для трансформеров (absolute sinusoidal or learned).

    Назначение:
    -----------
    - Добавляет или конкатенирует форму позиционной информации к каждому входному токену (since Transformer cannot distinguish positions otherwise).
    - Используется во всех \"ранних\" трансформерах (GPT, BERT, T5), чаще всего в виде learnable или синусоидальных embeddings.

    Архитектурные варианты:
    -----------------------
    - Learnable positional embeddings (как в GPT-2): обычный nn.Embedding инициализируется случайно, и веса учатся вместе с моделью.
    - Sinusoidal positional encoding (как в оригинальном Transformer): не имеет параметров, а создаётся по заданной формуле sin/cos(ω*x).

    Принцип работы:
    ---------------
    - Для каждой позиции t заполняется вектор emb_size длиной по формуле (или выбирается из weight matrix).
    - Эти вектора можно либо складывать с токеновыми эмбеддингами, либо конкатенировать.
    - Позволяет attention-механизму \"понимать\" порядок токенов/слов в последовательности.

    Формулы (Or: Vaswani et al., 2017):
    ------------------------------------
        PE(pos, 2i)   = sin(pos / 10000^{2i/d})
        PE(pos, 2i+1) = cos(pos / 10000^{2i/d})
    где d = emb_size, pos = позиция (int), i = индекс пары компонент.

    Аргументы конструктора:
    -----------------------
    max_seq_len: int — максимально поддерживаемая длина последовательности
    emb_size: int — размер возвращаемого positional vector для каждой позиции
    (иногда выбирается вариант — learnable или фиксация через sin/cos)

    Пример:
    -------
        >>> pos = PositionalEmbeddings(max_seq_len=1024, emb_size=256)
        >>> p = pos(32)  # Получить positional embeddings для 32 позиций
        >>> p.shape  # torch.Size([32, 256])
        >>> token_emb = ...  # [batch, seq_len, emb_size]
        >>> encoded = token_emb + p.unsqueeze(0)  # Broadcast add

    References:
    -----------
    - Vaswani et al., \"Attention is All You Need\", 2017: https://arxiv.org/abs/1706.03762
    - GPT-2 implementation: https://github.com/openai/gpt-2
    - Почему positional encoding важен: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    """

    def __init__(self, max_seq_len: int, emb_size: int):
        """
        Инициализация позиционного энкодера.

        Аргументы:
        ----------
        max_seq_len : int
            Максимальная длина последовательности (builds buffer for sin/cos or embedding)
        emb_size : int
            Длина позиционного вектора

        Внутри:
        -------
        - Если используется learned embedding: создаётся nn.Embedding (можно легко менять в будущем).
        - Если fixed (sin/cos): вычисляется и хранится буфер (max_seq_len, emb_size).
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.embedding = nn.Embedding(
            num_embeddings=max_seq_len, embedding_dim=emb_size
        )

    def forward(self, seq_len: int, start_pos: int = 0) -> Tensor:
        """
        Получить positional embeddings для последовательности длиной seq_len.

        Аргументы:
        ----------
        seq_len : int
            Сколько позиций сгенерировать (обычно == входная длина x)
        start_pos : int, по умолчанию 0
            Возможность выдать positional embeddings \"с середины\" (для autoregressive генерации)

        Возвращает:
        -----------
        torch.Tensor — positional embeddings формы [seq_len, emb_size]

        Пример:
        -------
            >>> pos = PositionalEmbeddings(512, 128)
            >>> p = pos(10)  # [10, 128]
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
