from functools import partial

import torch
from torch import nn

from llm.core.base_model import BaseModel
from llm.core.config_checks import resolve_head_size
from llm.core.generation import (
    cache_start_pos,
    check_attention_mask,
    check_sequence_length,
)
from llm.core.token_embeddings import TokenEmbeddings
from llm.core.swi_glu import SwiGLU
from llm.core.rms_norm import RMSNorm
from llm.core.rope import RoPE
from llm.core.cached_decoder import CachedDecoder


class Llama(BaseModel):
    """
    LLaMA — автогерессивная большая языковая модель (Large Language Model from Meta, 2023).

    Назначение:
    -----------
    - Модель реализует архитектуру decoder-only Transformer с современными "индустриальными" трюками (RMSNorm, SwiGLU, RoPE).
    - Предназначена для генерации текста, чат-ботов, zero-/few-shot вывода, fine-tune в стиле RLHF, transfer learning и исследований в LLM.

    Архитектурные особенности:
    --------------------------
    - Токеновые эмбеддинги и позиционное кодирование с помощью Rotary Position Embedding (RoPE, https://arxiv.org/abs/2104.09864).
    - Stack из num_layers блоков CachedDecoder с обычным Multi-Head Attention (как в LLaMA-1; GQA появилась в LLaMA-2 и реализована здесь в Mistral).
    - FeedForward блоки с SwiGLU (см. https://arxiv.org/abs/2002.05202).
    - Нормализация RMSNorm перед каждым sub-layer (вот почему "Pre-RMSNorm").
    - Кэширование attention (KV cache) для быстрой autoregressive генерации.
    - Отличия от оригинала: Linear-слои (Q/K/V, выходная проекция, голова) создаются с bias, dropout применяется в attention и FFN.

    Аргументы конструктора:
    -----------------------
    config: dict с требуемыми ключами:
        vocab_size: int — размер словаря токенов
        embed_dim: int — размерность эмбеддингов
        num_heads: int — количество attention-голов (head_size = embed_dim // num_heads, если head_size не задан)
        num_layers: int — число слоёв-декодеров
        max_position_embeddings: int — максимальная длина последовательности
        dropout: float — вероятность dropout
 
    Пример использования:
    ---------------------
        >>> llama = Llama({...})
        >>> tokens = torch.tensor([[100, 56, 8]])
        >>> logits, cache = llama(tokens)
        >>> out = llama.generate(tokens, max_new_tokens=10, do_sample=True, top_k=50)

    References:
    -----------
    - "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023): https://arxiv.org/abs/2302.13971
    - "RoFormer: Enhanced Transformer with Rotary Position Embedding": https://arxiv.org/abs/2104.09864
    - Discussion of efficient LLMs: https://huggingface.co/blog/mistral

    """

    def __init__(self, config):
        """
        Инициализация LLaMA.

        Args:
            config (dict): Параметры архитектуры, см. docstring класса.
        Внутри:
        -------
        - Создаёт Embedding-слой, Rotary Position Embeddings (RoPE), стек слоёв CachedDecoder (MHA + RMSNorm + SwiGLU).
        - Финальный слой нормализации и проекции на vocabulary.
        """
        super().__init__(config)

        # Размер головы: head_size из конфига или embed_dim // num_heads (с проверками)
        head_size = resolve_head_size(config, "num_heads", rope=True)
        # eps всех RMSNorm: 1e-6 по умолчанию как в LLaMA
        norm_eps = config.get("rms_norm_eps", 1e-6)

        # Инициализация слоев
        self._max_seq_len = config["max_position_embeddings"]
        self._token_embeddings = TokenEmbeddings(
            vocab_size=config["vocab_size"], emb_size=config["embed_dim"]
        )
        self._position_embeddings = RoPE(
            head_size=head_size,
            max_seq_len=config["max_position_embeddings"],
        )

        self._dropout = nn.Dropout(config["dropout"])
        self._decoders = nn.ModuleList(
            [
                CachedDecoder(
                    norm_layer=partial(RMSNorm, eps=norm_eps),
                    num_heads=config["num_heads"],
                    emb_size=config["embed_dim"],
                    head_size=head_size,
                    feed_forward_layer=SwiGLU(
                        emb_size=config["embed_dim"],
                        dropout=config["dropout"],
                    ),
                    max_seq_len=config["max_position_embeddings"],
                    rope=self._position_embeddings,
                    dropout=config["dropout"],
                )
                for _ in range(config["num_layers"])
            ]
        )
        self._norm = RMSNorm(config["embed_dim"], eps=norm_eps)
        self._linear = nn.Linear(config["embed_dim"], config["vocab_size"])

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        cache: list = None,
        attention_mask: torch.Tensor = None,
    ) -> tuple:
        """
        Прямой проход: возвращает logits (и возможно обновлённый cache) по входным токенам.

        Args:
            x (torch.Tensor): [batch, seq_len] — индексы токенов, shape [batch, seq_len]
            use_cache (bool): использовать механизм KV cache (ускоряет autoregressive generation)
            cache (list or None): предыдущий кэш, если нужен
            attention_mask (torch.Tensor, опц.): маска [batch, seq_len] (1 — токен, 0 — паддинг).
                Поддерживается правый паддинг; на другие маски с нулями — NotImplementedError
                (см. docs/README.md, раздел «Маски»).

        Returns:
            logits: torch.Tensor [batch, seq_len, vocab_size]
            new_cache: новый кэш attention (или None)
        """
        # Длина с учётом кэша: позиции start_pos … start_pos + seq_len − 1 должны быть < max_seq_len.
        # attention_mask допускается только такая, при которой causal-маски достаточно.
        check_sequence_length(x.size(1), cache_start_pos(cache), self._max_seq_len)
        check_attention_mask(attention_mask, x, cache)

        # Эмбеддинги токенов; позиции кодируются RoPE внутри attention
        tok_out = self._token_embeddings(x)  # [batch, seq_len, emb_size]

        # Комбинирование
        out = self._dropout(tok_out)  # [batch, seq_len, emb_size]

        # Стек декодеров с передачей кэша
        new_cache = []
        for i, decoder in enumerate(self._decoders):
            decoder_cache = cache[i] if cache is not None else None
            decoder_result = decoder(out, use_cache=use_cache, cache=decoder_cache)

            # Извлекаем результат из кортежа
            if use_cache:
                out, decoder_new_cache = decoder_result
                new_cache.append(decoder_new_cache)
            else:
                out = decoder_result[0]

        out = self._norm(out)
        logits = self._linear(out)

        # Возвращаем результат с учетом use_cache
        if use_cache:
            return (logits, new_cache)
        else:
            return (logits, None)
