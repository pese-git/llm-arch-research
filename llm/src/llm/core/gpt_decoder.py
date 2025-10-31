from torch import nn
import torch
from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention


class GptDecoder(nn.Module):
    """
    Decoder — базовый transformer decoder block (pre-LN), классический строительный блок современных языковых моделей.

    Назначение:
    -----------
    - Инкапсулирует архитектуру: norm → multi-head self-attention → residual → norm → feed-forward → residual
    - Подходит как для LLM/GPT, так и для любых autoregressive sequence моделей.
    - Использует masked self-attention: каждый токен видит только предыдущие (никакого \"заглядывания в будущее\").
    - Стабильность обеспечивается через residual connections и LayerNorm после каждого sub-layer.

    Почему это важно?
    -----------------
    - Все современные языковые модели состоят из подобных блоков, соединённых в стек.
    - Алгоритм residual+norm позволяет проще обучать очень глубокие сети.
    - Разделение на attention+FFN дает и локальные, и глобальные взаимодействия между токенами.

    Формула работы (псевдокод):
    ---------------------------
        y1 = norm1(x)
        attn_out = Attention(y1)
        x2 = x + attn_out        # residual
        y2 = norm2(x2)
        ffn_out = FFN(y2)
        out = x2 + ffn_out       # residual

    Архитектурные особенности:
    --------------------------
    - Поддержка внимания с маской (causal mask или произвольная attention mask)
    - Residual connections для каждого блока (attention, FFN)
    - Pre-LN (norm перед каждым подблоком)
    - Зависит от переданных блоков self_attention и feed_forward, а не их реализации

    References:
    -----------
    - Vaswani et al., \"Attention is All You Need\" (2017): https://arxiv.org/abs/1706.03762
    - Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
    - Transformer Circuits (дружественное описание): https://transformer-circuits.pub/2021/framework/index.html

    Пример:
    -------
        >>> decoder = Decoder(num_heads=8, emb_size=512, head_size=64, max_seq_len=1024)
        >>> x = torch.randn(1, 10, 512)
        >>> out = decoder(x)
        >>> print(out.shape)  # torch.Size([1, 10, 512])
    """

    def __init__(
        self,
        num_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        """
        Инициализация стандартного decoder-блока для Transformer.

        Аргументы:
        ----------
        num_heads: int
            Количество attention голов (как делить emb_size на heads)
        emb_size: int
            Размерность эмбеддингов (и входа и выхода)
        head_size: int
            Размерность одной attention-головы (emb_size = num_heads * head_size)
        max_seq_len: int
            Максимальная длина последовательности (важно для mask)
        dropout: float, default=0.1
            Dropout после внимания и FFN

        Внутри:
        -------
        - Создаёт слой MultiHeadAttention (masked/casual)
        - Создаёт двухслойный FeedForward (SwiGLU или GELU)
        - Применяет 2 слоя LayerNorm для стабилизации градиентов
        - Все блоки реализованы как PyTorch-модули
        """
        super().__init__()
        self._heads = MultiHeadAttention(
            num_heads=num_heads,
            emb_size=emb_size,
            head_size=head_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self._ff = FeedForward(emb_size=emb_size, dropout=dropout)
        self._norm1 = nn.LayerNorm(emb_size)
        self._norm2 = nn.LayerNorm(emb_size)

    def forward(
        self, 
        x: torch.Tensor, 
        use_cache: bool = False, 
        cache: list = None, 
        attention_mask=None
    ) -> tuple:
        """
        Один прямой проход через Transformer decoder block.

        Аргументы:
        ----------
        x : torch.Tensor
            Входной тензор [batch_size, seq_len, emb_size]
        mask : torch.Tensor, optional
            Attention/causal mask (по умолчанию None, тогда будет casual mask по длине seq_len)

        Возвращает:
        -----------
        out : torch.Tensor
            Выходной тензор той же формы, что и x

        Алгоритм:
        ---------
        - Применяем attention к нормализованному входу (layernorm)
        - Добавляем residual-связь (attention + исходный вход)
        - Применяем FFN к нормализованному результату (layernorm)
        - Добавляем residual-связь (ffn + предыдущий выход)
        """

        # Self-Attention блок
        attention, kv_caches = self._heads(x, attention_mask, use_cache=use_cache, cache=cache)
        out = self._norm1(attention + x)

        # FeedForward блок
        ffn_out = self._ff(out)
        result =  self._norm2(ffn_out + out)
        
        if use_cache:
            return (result, kv_caches)
        else:
            return (result, None)
