"""
Классическая GPT (Generative Pre-trained Transformer), OpenAI 2018.

Научная суть:
    - Первая массовая архитектура языка на основе исключительно self-attention механизмов (трансформер-декодер).
    - Обучается сначала на задаче языкового моделирования (unsupervised), далее дообучается на downstream-задачах (transfer learning).
    - Обеспечивает длинную память и “глобальный” контекст благодаря attention.

    Ключевые элементы:
    - masked self-attention (causal)
    - LayerNorm ПОСЛЕ attention и FFN (что отличает от GPT2)
    - GELU активация
    - Absolute learned positional embeddings

    Подробнее: Radford et al., "Improving Language Understanding by Generative Pre-Training" (OpenAI, 2018)
    https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

    Пример использования:
        >>> model = GPT({"vocab_size": 50257, ...})
        >>> logits, cache = model(input_ids)
        >>> out = model.generate(input_ids, max_new_tokens=30, do_sample=False)
    """

from functools import partial

import torch
import torch.nn as nn
from llm.core.base_model import BaseModel
from llm.core.config_checks import resolve_head_size
from llm.core.weight_init import DEFAULT_INITIALIZER_RANGE, init_normal_
from llm.core.generation import (
    cache_start_pos,
    check_attention_mask,
    check_sequence_length,
)
from llm.core.gpt_decoder import GptDecoder
from llm.core.token_embeddings import TokenEmbeddings, output_projection
from llm.core.positional_embeddings import PositionalEmbeddings


class GPT(BaseModel):
    """
    GPT (Generative Pretrained Transformer) — автогерессивная языковая модель по мотивам оригинального GPT-1.

    Назначение:
    -----------
    - Позволяет предсказывать и генерировать последовательности текста, обучаясь на задаче language modeling (предсказывать следующий токен).
    - Класс реализует архитектуру classic Transformer Decoder Stack с masked multi-head attention и token/positional embeddings.
    - Используется как базовая модель для генерации, zero-/few-shot, задач обучения с подкреплением и пр.

    Архитектурные особенности:
    --------------------------
    - Embedding-слои для токенов (token_embeddings) и позиций (position_embeddings).
    - Stack из N блоков GptDecoder (MultiHeadAttention + FeedForward + residual + LayerNorm, post-LN: нормализация после residual).
    - Masked self-attention — каждый токен видит только свои и предыдущие, обеспечивая автогерессию.
    - Финальной нормализации перед проекцией на словарь нет (в отличие от GPT-2).
    - Поддержка efficient KV кэша — ускоряет autoregressive inference/generation.

    Основные параметры:
    -------------------
    config: dict в формате {
        vocab_size,        # размер словаря токенов
        embed_dim,         # размерность эмбеддинга
        num_heads,         # количество attention heads
        num_layers,        # глубина модели (число блоков)
        max_position_embeddings,
        dropout,
        activation         # опционально: активация FFN, по умолчанию "gelu_tanh"
    }

    Формула и поток данных:
    -----------------------
        x -> token_embeddings -> + position_embeddings -> dropout ->
           -> stack([GptDecoder]) ->
           -> Linear(out_dim=vocab_size) -> output_logits

    Пример использования:
    ---------------------
        >>> gpt = GPT({...})
        >>> tokens = torch.tensor([[12, 123, 44]])
        >>> logits, cache = gpt(tokens)
        >>> generated = gpt.generate(tokens, max_new_tokens=10, do_sample=False)

    References:
    -----------
    - Radford et al., "Improving Language Understanding by Generative Pre-Training" (GPT-1, 2018)
      https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
    - Original BPE Tokenizer code: https://github.com/openai/gpt-2/blob/master/src/encoder.py
    - Формула masked self-attention: Vaswani et al., "Attention is All You Need", 2017
      https://arxiv.org/abs/1706.03762
    """

    def __init__(self, config):
        """
        Инициализация модели GPT.

        Args:
        -----
        config: dict
            Параметры архитектуры:
              vocab_size: int — размер словаря токенов
              embed_dim: int — размерность эмбеддинга
              num_heads: int — количество attention-heads
              num_layers: int — число Transformer блоков
              max_position_embeddings: int — макс. длина последовательности
              dropout: float — dropout
              activation: str, опционально — активация FFN ("gelu_tanh" по умолчанию —
                  tanh-аппроксимация GELU, как в оригинальном коде; "gelu" — точный GELU через erf;
                  "relu" — упрощённый учебный вариант)
              tie_word_embeddings: bool, опционально — общие веса эмбеддингов и выходной
                  проекции без bias, как в оригинальном коде (по умолчанию False)

        Внутри:
        -------
        - Создаёт слой эмбеддингов, позиционку, стек декодеров, нормализацию, линейную проекцию.
        """
        super().__init__(config)

        # Размер головы: head_size из конфига или embed_dim // num_heads (с проверками)
        head_size = resolve_head_size(config, "num_heads")

        # Инициализация слоев
        self._max_seq_len = config["max_position_embeddings"]
        self._token_embeddings = TokenEmbeddings(
            vocab_size=config["vocab_size"], emb_size=config["embed_dim"]
        )
        self._position_embeddings = PositionalEmbeddings(
            max_seq_len=config["max_position_embeddings"], emb_size=config["embed_dim"]
        )
        self._dropout = nn.Dropout(config["dropout"])
        self._decoders = nn.ModuleList(
            [
                GptDecoder(
                    num_heads=config["num_heads"],
                    emb_size=config["embed_dim"],
                    head_size=head_size,
                    max_seq_len=config["max_position_embeddings"],
                    dropout=config["dropout"],
                    attention_dropout=config.get("attention_dropout", 0.0),
                    activation=config.get("activation", "gelu_tanh"),
                )
                for _ in range(config["num_layers"])
            ]
        )
        # Выходная проекция; tie_word_embeddings=True — общие веса с эмбеддингами и без bias,
        # как в оригинале. По умолчанию False, чтобы грузились прежние чекпоинты
        self._linear = output_projection(
            self._token_embeddings, tie_weights=config.get("tie_word_embeddings", False)
        )

        # Инициализация из статьи GPT-1 (разд. 4.1): Linear и Embedding — N(0, 0.02), bias — нули
        self.apply(
            partial(init_normal_, std=config.get("initializer_range", DEFAULT_INITIALIZER_RANGE))
        )

    def forward(
        self, x: torch.Tensor, attention_mask=None, use_cache: bool = False, cache: list = None
    ) -> tuple:
        """
        Прямой проход для получения логитов по последовательности токенов.

        Args:
        -----
        x : torch.Tensor [batch, seq_len]
            Индексы входных токенов.
        use_cache : bool, optional
            Использовать ли кэш attention (ускоряет инференс, важно для генерации)
        cache : list, optional
            Список старых KV (key/value)-кэшей
        attention_mask : torch.Tensor, optional
            Маска [batch, seq_len] (1 — токен, 0 — паддинг). Поддерживается правый паддинг:
            causal-маска и так скрывает от настоящих токенов стоящий после них паддинг.
            На другие маски с нулями — NotImplementedError (см. docs/masks.md).

        Returns:
        --------
        logits: [batch, seq_len, vocab_size]   (логиты для softmax по словарю)
        new_cache: кэш KV после прохода
        """
        # Длина с учётом кэша: позиции start_pos … start_pos + seq_len − 1 должны быть < max_seq_len.
        # attention_mask допускается только такая, при которой causal-маски достаточно.
        start_pos = cache_start_pos(cache)
        check_sequence_length(x.size(1), start_pos, self._max_seq_len)
        check_attention_mask(attention_mask, x, cache)

        seq_len = x.size(1)

        # Эмбеддинги токенов и позиций
        tok_out = self._token_embeddings(x)  # [batch, seq_len, emb_size]
        pos_out = self._position_embeddings(
            seq_len, start_pos=start_pos
        )  # [seq_len, emb_size]

        # Комбинирование
        out = self._dropout(
            tok_out + pos_out.unsqueeze(0)
        )  # [batch, seq_len, emb_size]

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

        logits = self._linear(out)  # [batch, seq_len, vocab_size]

        # Возвращаем результат с учетом use_cache
        if use_cache:
            return (logits, new_cache)
        else:
            return (logits, None)
