"""
GPT-2 — масштабируемый автогерессивный языковой трансформер второго поколения от OpenAI (2019).

Научная суть:
    - В сравнении с классическим GPT, layer normalization теперь применяется ПЕРЕД attention и FFN.
    - Позволило сильно увеличить глубину и размер модели (GPT2-модели имеют от 117M до 1.5B параметров).
    - Используется GELU активация; эффективное кэширование KV attention для генерации.

Формула attention-блока:
    LN(x) → Attention → рез. связь → LN → FFN → рез. связь

Подробнее:
    Radford et al. "Language Models are Unsupervised Multitask Learners"
    https://cdn.openai.com/better-language-models/language-models.pdf

Пример использования:
    >>> model = GPT2({"vocab_size": 50257, ...})
    >>> logits, _ = model(input_ids)
    >>> out = model.generate(input_ids, max_new_tokens=30, do_sample=False)
"""
from functools import partial

import torch
from torch import nn
from llm.core.base_model import BaseModel
from llm.core.config_checks import resolve_head_size
from llm.core.weight_init import DEFAULT_INITIALIZER_RANGE, init_normal_, scale_residual_projections_
from llm.core.generation import (
    cache_start_pos,
    check_attention_mask,
    check_sequence_length,
)
from llm.core.token_embeddings import TokenEmbeddings, output_projection
from llm.core.positional_embeddings import PositionalEmbeddings
from llm.core.gpt2_decoder import Gpt2Decoder


class GPT2(BaseModel):
    """
    GPT-2 — масштабируемый автогерессивный языковой трансформер второго поколения от OpenAI (2019).

    Назначение:
    -----------
    - Позволяет предсказывать и порождать последовательности текста по одному токену, будучи обученным на задаче language modeling.
    - Модель реализует архитектуру decoder-only Transformer с Pre-LN (LayerNorm перед attention и FFN).
    - Используется для генерации, обучения с подкреплением для RLHF, zero/few-shot inference, чат-ботов и др.

    Архитектурные особенности:
    --------------------------
    - Token и positional embeddings (learnable, как в GPT-2 оригинале).
    - Stack из N блоков Gpt2Decoder (MultiHeadAttention с causal mask, Residual, Pre-LayerNorm, GELU FFN).
    - KV attention-кэш (ускоряет autoregressive generation, критически важно для LLM).
    - Использует GELU как функцию активации.
    - Поддержка dropout на каждом этапе.

    Основные параметры:
    -------------------
    config: dict — параметры модели:
        vocab_size,         # размер словаря токенов
        embed_dim,          # размерность эмбеддинга
        num_heads,          # количество attention голов
        num_layers,         # глубина модели (число блоков)
        max_position_embeddings,
        dropout

    Процессинг:
    -----------
        x (индексы токенов) → token_embeddings + position_embeddings → dropout
        → stack Decoder blocks (masked attention, pre-LN)
        → LayerNorm
        → Linear(out_dim=vocab_size) → выходные логиты

    Пример использования:
    ---------------------
        >>> gpt2 = GPT2({...})
        >>> logits, _ = gpt2(input_ids)
        >>> output = gpt2.generate(input_ids, max_new_tokens=20, do_sample=True)

    References:
    -----------
    - Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019): https://cdn.openai.com/better-language-models/language-models.pdf
    - HuggingFace GPT-2: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
    - Репликация в NanoGPT: https://github.com/karpathy/nanoGPT
    """

    def __init__(self, config):
        """
        Инициализация GPT-2.

        Args:
            config (dict): Параметры архитектуры:
                vocab_size: int — размер словаря
                embed_dim: int — размерность эмбеддинга
                num_heads: int — количество attention-голов
                num_layers: int — количество декодер-блоков
                max_position_embeddings: максимальная длина последовательности
                dropout: float — dropout
                tie_word_embeddings: bool, опционально — общие веса эмбеддингов и выходной
                    проекции без bias, как в оригинале и HF (по умолчанию False)

        Внутри:
        -------
        - Создаёт токеновые и позиционные эмбеддинги, стек декодеров, финальный LayerNorm и линейную проекцию в словарь.
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
                Gpt2Decoder(
                    num_heads=config["num_heads"],
                    emb_size=config["embed_dim"],
                    head_size=head_size,
                    max_seq_len=config["max_position_embeddings"],
                    dropout=config["dropout"],
                    attention_dropout=config.get("attention_dropout", 0.0),
                )
                for _ in range(config["num_layers"])
            ]
        )
        self._norm = nn.LayerNorm(config["embed_dim"])
        # Выходная проекция; tie_word_embeddings=True — общие веса с эмбеддингами и без bias,
        # как в оригинале. По умолчанию False, чтобы грузились прежние чекпоинты
        self._linear = output_projection(
            self._token_embeddings, tie_weights=config.get("tie_word_embeddings", False)
        )

        # Инициализация как в GPT-2: N(0, 0.02), а проекции, которые пишут в residual-поток
        # (выход attention и второй слой FFN), — N(0, 0.02 / √(2·num_layers)) (разд. 2.3 статьи)
        std = config.get("initializer_range", DEFAULT_INITIALIZER_RANGE)
        self.apply(partial(init_normal_, std=std))
        scale_residual_projections_(
            [projection for decoder in self._decoders for projection in (decoder._heads._layer, decoder._ff._layer2)],
            num_layers=config["num_layers"],
            std=std,
        )

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        cache: list = None,
        attention_mask: torch.Tensor = None,
    ) -> tuple:
        """
        Прямой проход для batch of sequences (получение логитов по токенам).

        Args:
            x (torch.Tensor): Входной тензор с токенами [batch, seq_len]
            use_cache (bool): Использовать/возвращать кэш KV attention (ускоряет генерацию)
            cache (list / None): Внешний кэш KV attention (передаётся при генерации)
            attention_mask (torch.Tensor, опц.): маска [batch, seq_len] (1 — токен, 0 — паддинг).
                Поддерживается правый паддинг; на другие маски с нулями — NotImplementedError
                (см. docs/masks.md).

        Returns:
            logits: torch.Tensor [batch, seq_len, vocab_size]
            new_cache: новый кэш KV attention (или None)

        Пример:
            >>> logits, cache = gpt2(x, use_cache=True)
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

        out = self._norm(out)
        logits = self._linear(out)

        # Возвращаем результат с учетом use_cache
        if use_cache:
            return (logits, new_cache)
        else:
            return (logits, None)
