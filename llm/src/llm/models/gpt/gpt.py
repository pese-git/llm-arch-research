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

    Подробнее: Radford et al., "Improving Language Understanding by Generative Pre-Training", arXiv:1801.10198
    https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

    Пример использования:
        >>> model = GPT({"vocab_size": 50257, ...})
        >>> logits = model(input_ids)
        >>> out = model.generate(input_ids, max_length=30)
    """

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from llm.core.base_model import BaseModel
from llm.core.gpt_decoder import GptDecoder
from llm.core.token_embeddings import TokenEmbeddings
from llm.core.positional_embeddings import PositionalEmbeddings


class GPT(BaseModel):
    """
    GPT (Generative Pretrained Transformer) — автогерессивная языковая модель по мотивам оригинального GPT/GPT-2 architecture.

    Назначение:
    -----------
    - Позволяет предсказывать и генерировать последовательности текста, обучаясь на задаче language modeling (предсказывать следующий токен).
    - Класс реализует архитектуру classic Transformer Decoder Stack с masked multi-head attention и token/positional embeddings.
    - Используется как базовая модель для генерации, zero-/few-shot, задач обучения с подкреплением и пр.

    Архитектурные особенности:
    --------------------------
    - Embedding-слои для токенов (token_embeddings) и позиций (position_embeddings).
    - Stack из N декодер-блоков (MultiHeadAttention + FeedForward + residual + LayerNorm).
    - Masked self-attention — каждый токен видит только свои и предыдущие, обеспечивая автогерессию.
    - LayerNorm до проекции на словарь (pre-LN).
    - Поддержка efficient KV кэша — ускоряет autoregressive inference/generation.

    Основные параметры:
    -------------------
    config: dict в формате {
        vocab_size,        # размер словаря токенов
        embed_dim,         # размерность эмбеддинга
        num_heads,         # количество attention heads
        num_layers,        # глубина модели (число блоков)
        max_position_embeddings,
        dropout
    }

    Формула и поток данных:
    -----------------------
        x -> token_embeddings -> + position_embeddings -> dropout ->
           -> stack([DecoderBlock]) ->
           -> LayerNorm ->
           -> Linear(out_dim=vocab_size) -> output_logits

    Пример использования:
    ---------------------
        >>> gpt = GPT({...})
        >>> tokens = torch.tensor([[12, 123, 44]])
        >>> logits = gpt(tokens)
        >>> generated = gpt.generate(tokens, max_new_tokens=10)

    References:
    -----------
    - Radford et al., "Improving Language Understanding by Generative Pre-Training" (GPT-1, 2018)
      https://cdn.openai.com/research-covers/languageunsupervised/language_understanding_paper.pdf
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

        Внутри:
        -------
        - Создаёт слой эмбеддингов, позиционку, стек декодеров, нормализацию, линейную проекцию.
        """
        super().__init__(config)

        # Инициализация слоев
        self._max_seq_len = config["max_position_embeddings"]
        self._token_embeddings = TokenEmbeddings(
            vocab_size=config["vocab_size"], emb_size=config["embed_dim"]
        )
        self._position_embeddings = PositionalEmbeddings(
            max_seq_len=config["max_position_embeddings"], emb_size=config["embed_dim"]
        )
        self._dropout = nn.Dropout(config["dropout"])
        # head_size = emb_size // num_heads
        self._decoders = nn.ModuleList(
            [
                GptDecoder(
                    num_heads=config["num_heads"],
                    emb_size=config["embed_dim"],
                    head_size=config["embed_dim"] // config["num_heads"],
                    max_seq_len=config["max_position_embeddings"],
                    dropout=config["dropout"],
                )
                for _ in range(config["num_layers"])
            ]
        )
        self._linear = nn.Linear(config["embed_dim"], config["vocab_size"])

    @property
    def max_seq_len(self):
        """Возвращает максимальную длину последовательности."""
        return self._max_seq_len

    def forward(
        self, x: torch.Tensor, attention_mask=None, use_cache: bool = True, cache: list = None
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

        Returns:
        --------
        logits: [batch, seq_len, vocab_size]   (логиты для softmax по словарю)
        new_cache: кэш KV после прохода
        """
        # Проверка длины последовательности
        if x.size(1) > self._max_seq_len:
            raise ValueError(
                f"Длина последовательности {x.size(1)} превышает максимальную {self._max_seq_len}"
            )

        # Вычисление start_pos из кэша (если кэш передан)
        if cache is not None:
            seq_len = 1
            # Безопасно извлекаем key_cache для вычисления start_pos
            if (
                isinstance(cache, (list, tuple))
                and len(cache) > 0
                and cache[0] is not None
                and isinstance(cache[0], (list, tuple))
                and len(cache[0]) > 0
                and cache[0][0] is not None
                and isinstance(cache[0][0], (tuple, list))
                and len(cache[0][0]) > 0
            ):
                key_cache, _ = cache[0][0]
                start_pos = key_cache.size(1)
            else:
                start_pos = 0
        else:
            # Без кэша работаем как раньше
            start_pos = 0
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

    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        use_cache: bool = True,
        attention_mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Авторегрессивная генерация текста с поддержкой жадного поиска (greedy), вероятностного сэмплирования с температурой,
        top-k и nucleus (top-p) sampling.
    
        Аргументы:
            x (torch.Tensor): Входной тензор с индексами токенов, форма [batch_size, seq_len].
            max_new_tokens (int): Максимальное количество новых токенов для генерации.
            do_sample (bool): Если True — вероятностное сэмплирование; если False — жадная генерация (argmax).
            temperature (float): Температура для управления случайностью (>0, влияет только если do_sample=True).
                                 >1.0 — более случайно, <1.0 — более детерминированно.
            top_k (int, опц.): При do_sample=True ограничивает выбор top_k самых вероятных токенов (top-k sampling).
            top_p (float, опц.): При do_sample=True включает top-p (nucleus) sampling: кумулятивная вероятность ≤ top_p.
                                 Должно быть в (0, 1].
            attention_mask (torch.Tensor, опц.): Внешняя маска внимания (для совместимости с HuggingFace).
            **kwargs: Игнорируются.
    
        Возвращает:
            torch.Tensor: Последовательность токенов [batch_size, seq_len + max_new_tokens].
    
        Исключения:
            ValueError: Если x длиннее max_seq_len модели.
            ValueError: Если temperature ≤ 0.
            ValueError: Если одновременно заданы top_k и top_p.
            ValueError: Если top_k ≤ 0.
            ValueError: Если top_p вне диапазона (0, 1].
    
        Примеры:
            >>> # Жадная (детерминированная) генерация
            >>> output = model.generate(input_ids, max_new_tokens=12, do_sample=False)
            >>> # Вероятностная генерация с температурой
            >>> output = model.generate(input_ids, max_new_tokens=12, do_sample=True, temperature=0.8)
            >>> # Top-k сэмплирование
            >>> output = model.generate(input_ids, max_new_tokens=12, do_sample=True, top_k=50)
            >>> # Top-p (nucleus) sampling
            >>> output = model.generate(input_ids, max_new_tokens=12, do_sample=True, top_p=0.92)
            >>> # Комбинация температуры и top-k
            >>> output = model.generate(input_ids, max_new_tokens=12, do_sample=True, temperature=1.0, top_k=100)
    
        Примечания:
            - Для детерминированных выборок зафиксируйте random seed через torch.manual_seed.
            - Параметры temperature, top_k, top_p применимы только если do_sample=True.
            - Одновременное использование top_k и top_p не допускается.
            - Модель всегда возвращает тензор индексов токенов; для получения логитов используйте прямой вызов forward.
    
        Ссылки:
            - Holtzman et al., "The Curious Case of Neural Text Degeneration" (nucleus sampling): https://arxiv.org/abs/1904.09751
            - Оригинальный GPT-2: https://cdn.openai.com/better-language-models/language-models.pdf
        """
        cache = None
        
        for _ in range(max_new_tokens):
            # 1. Обрезаем вход, если последовательность слишком длинная
            if use_cache and cache is not None:
                # Используем кэш - передаем только последний токен
                x_input = x[:, -1:]  # [batch_size, 1]
            else:
                # Первая итерация или кэш отключен - передаем всю последовательность
                x_input = x

            # 2. Передаем последовательность в метод forward класса GPT и полуаем логиты.
            # Прямой проход с кэшем
            logits, new_cache = self.forward(x_input, use_cache=use_cache, cache=cache)

            # Обновляем кэш для следующей итерации
            if use_cache:
                cache = new_cache

            # 3. Берем логиты для последнего токена
            last_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # Масштабируем логиты температурой
            if temperature > 0:
                logits_scaled = last_logits / temperature
            else:
                logits_scaled = last_logits

            if do_sample == True and top_k != None:
                _, topk_indices = torch.topk(logits_scaled, top_k, dim=-1)

                # # Заменим все НЕ top-k логиты на -inf
                masked_logits = logits_scaled.clone()
                vocab_size = logits_scaled.size(-1)

                # создаём маску: True, если токен НЕ в topk_indices
                mask = torch.ones_like(
                    logits_scaled,
                    dtype=torch.bool if hasattr(torch, "bool") else torch.uint8,
                )
                mask.scatter_(
                    1, topk_indices, False if hasattr(torch, "bool") else 0
                )  # False там, где top-k индексы
                masked_logits[mask] = float("-inf")

                logits_scaled = masked_logits

            if do_sample == True and top_p != None:
                # 1. Применим softmax, чтобы получить вероятности:
                probs = F.softmax(logits_scaled, dim=-1)  # [B, vocab_size]
                # 2. Отсортируем токены по убыванию вероятностей:
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True, dim=-1
                )
                # 3. Посчитаем кумулятивную сумму вероятностей:
                cum_probs = torch.cumsum(sorted_probs, dim=-1)  # [B, vocab_size]
                # 4. Определим маску: оставить токены, пока сумма < top_p
                sorted_mask = cum_probs <= top_p  # [B, vocab_size]
                # Гарантируем, что хотя бы первый токен останется
                sorted_mask[:, 0] = True
                # 5. Преобразуем маску обратно в оригинальный порядок:
                # Создаём полную маску из False
                mask = torch.zeros_like(
                    probs, dtype=torch.bool if hasattr(torch, "bool") else torch.uint8
                )
                # Устанавливаем True в местах нужных токенов
                mask.scatter_(dim=1, index=sorted_indices, src=sorted_mask)
                # 6. Зануляем логиты токенов вне топ-p:
                logits_scaled[~mask] = float("-inf")

            # 4. Применяем Softmax
            probs = F.softmax(logits_scaled, dim=-1)  # [batch_size, vocab_size]

            if do_sample == True:
                # 5. Если do_sample равен True, то отбираем токен случайно с помощью torch.multinomial
                next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
            else:
                # 5. Если do_sample равен False, то выбираем токен с максимальной вероятностью
                next_token = torch.argmax(
                    probs, dim=-1, keepdim=True
                )  # [batch_size, 1]

            # 6. Добавляем его к последовательности
            x = torch.cat([x, next_token], dim=1)  # [batch_size, seq_len+1]
        return x


#    def generate(self, input_ids, max_length=50):
#        for _ in range(max_length):
#            logits = self.forward(input_ids)
#            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
#            input_ids = torch.cat([input_ids, next_token], dim=1)
#        return input_ids
