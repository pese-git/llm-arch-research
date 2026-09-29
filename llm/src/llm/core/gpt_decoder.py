from torch import nn
import torch
from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention


class GptDecoder(nn.Module):
    """
    GptDecoder — блок декодера GPT-1 с post-LN: нормализация стоит после residual-сложения.

    Назначение:
    -----------
    - Инкапсулирует архитектуру: masked self-attention → residual → LayerNorm → feed-forward → residual → LayerNorm.
    - Использует masked self-attention: каждый токен видит только предыдущие (никакого "заглядывания в будущее").
    - Post-LN — как в оригинальном Transformer и GPT-1; начиная с GPT-2 нормализацию переносят
      перед подблоком (pre-LN, см. Gpt2Decoder), что устойчивее на глубоких стеках.

    Формула работы (псевдокод):
    ---------------------------
        attn_out = Attention(x)
        x2 = LayerNorm1(x + attn_out)       # residual, затем норма
        ffn_out = FFN(x2)
        out = LayerNorm2(x2 + ffn_out)      # residual, затем норма

    Архитектурные особенности:
    --------------------------
    - Только встроенная causal-маска; паддинг проверяется в forward модели (check_attention_mask)
    - Residual connections для каждого подблока (attention, FFN)
    - Post-LN (норма после каждого residual-сложения)
    - KV-кэш для генерации по одному токену

    References:
    -----------
    - Radford et al., "Improving Language Understanding by Generative Pre-Training" (GPT-1, 2018):
      https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
    - Vaswani et al., "Attention is All You Need" (2017): https://arxiv.org/abs/1706.03762
    - Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/

    Пример:
    -------
        >>> decoder = GptDecoder(num_heads=8, emb_size=512, head_size=64, max_seq_len=1024)
        >>> x = torch.randn(1, 10, 512)
        >>> out, _ = decoder(x)
        >>> print(out.shape)  # torch.Size([1, 10, 512])
    """

    def __init__(
        self,
        num_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = "gelu_tanh",
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
            Размерность одной attention-головы (обычно emb_size // num_heads)
        max_seq_len: int
            Максимальная длина последовательности (важно для mask)
        dropout: float, default=0.1
            Dropout после внимания и FFN
        attention_dropout: float, default=0.0
            Dropout на весах внимания после softmax (в GPT-1 — 0.1)
        activation: str, default="gelu_tanh"
            Активация в FeedForward ("gelu_tanh", "gelu", "relu").
            "gelu_tanh" — tanh-аппроксимация GELU, как в оригинальном коде GPT-1;
            "gelu" — точный GELU через erf; "relu" — упрощённый учебный вариант.

        Внутри:
        -------
        - Создаёт слой MultiHeadAttention (masked/casual)
        - Создаёт двухслойный FeedForward с заданной активацией (по умолчанию GELU)
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
            attention_dropout=attention_dropout,
        )
        # По умолчанию GELU (tanh-аппроксимация), а не ReLU (дефолт FeedForward), т.к. GPT-1 использует GELU:
        # "For the activation function, we used the Gaussian Error Linear Unit (GELU)"
        # — Radford et al., "Improving Language Understanding by Generative Pre-Training", 2018, разд. 4.1
        # https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
        # Вариант — tanh-аппроксимация, как в оригинальном коде (openai/finetune-transformer-lm, train.py).
        self._ff = FeedForward(
            emb_size=emb_size,
            dropout=dropout,
            activation=activation,
        )
        self._norm1 = nn.LayerNorm(emb_size)
        self._norm2 = nn.LayerNorm(emb_size)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        cache: list = None,
    ) -> tuple:
        """
        Один прямой проход через блок декодера.

        Аргументы:
        ----------
        x : torch.Tensor
            Входной тензор [batch_size, seq_len, emb_size]
        use_cache : bool, по умолчанию False
            Вернуть KV-кэш attention.
        cache : tuple, optional
            KV-кэш этого слоя с предыдущих шагов.

        Возвращает:
        -----------
        (out, cache) : tuple
            out — тензор той же формы, что и x; cache — новый KV-кэш или None при use_cache=False.

        Алгоритм:
        ---------
        - attention по входу, residual-связь, LayerNorm
        - FFN, residual-связь, LayerNorm
        """

        # Self-Attention блок
        attention, kv_caches = self._heads(x, use_cache=use_cache, cache=cache)
        out = self._norm1(attention + x)

        # FeedForward блок
        ffn_out = self._ff(out)
        result = self._norm2(ffn_out + out)

        if use_cache:
            return (result, kv_caches)
        else:
            return (result, None)
