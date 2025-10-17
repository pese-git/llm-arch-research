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
    >>> logits = model(input_ids)
    >>> out = model.generate(input_ids, max_length=30)
"""
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from llm.core.base_model import BaseModel
from llm.core.token_embeddings import TokenEmbeddings
from llm.core.positional_embeddings import PositionalEmbeddings
from llm.core.cached_decoder import CachedDecoder
from llm.core.feed_forward import FeedForward


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
    - Stack из N блоков Decoder (MultiHeadAttention с causal mask, Residual, Pre-LayerNorm, GELU FFN).
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
        >>> logits = gpt2(input_ids)
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

        Внутри:
        -------
        - Создаёт токеновые и позиционные эмбеддинги, стек декодеров, финальный LayerNorm и линейную проекцию в словарь.
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
                CachedDecoder(
                    num_heads=config["num_heads"],
                    emb_size=config["embed_dim"],
                    head_size=config["embed_dim"] // config["num_heads"],
                    feed_forward_layer=FeedForward(
                        emb_size=config["embed_dim"],
                        dropout=config["dropout"],
                        activation="gelu",
                    ),
                    max_seq_len=config["max_position_embeddings"],
                    dropout=config["dropout"],
                )
                for _ in range(config["num_layers"])
            ]
        )
        self._norm = nn.LayerNorm(config["embed_dim"])
        self._linear = nn.Linear(config["embed_dim"], config["vocab_size"])

    def forward(
        self, x: torch.Tensor, use_cache: bool = True, cache: list = None
    ) -> tuple:
        """
        Прямой проход для batch of sequences (получение логитов по токенам).

        Args:
            x (torch.Tensor): Входной тензор с токенами [batch, seq_len]
            use_cache (bool): Использовать/возвращать кэш KV attention (ускоряет генерацию)
            cache (list / None): Внешний кэш KV attention (передаётся при генерации)

        Returns:
            logits: torch.Tensor [batch, seq_len, vocab_size]
            new_cache: новый кэш KV attention (или None)

        Пример:
            >>> logits, cache = gpt2(x, use_cache=True)
        """
        # Проверка длины последовательности (только при отсутствии кэша)
        if cache is None and x.size(1) > self._max_seq_len:
            raise ValueError(
                f"Длина последовательности {x.size(1)} превышает максимальную {self.max_seq_len}"
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

        out = self._norm(out)
        logits = self._linear(out)

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
    ) -> torch.Tensor:
        """
        Авторегрессивная генерация токенов с поддержкой greedy, temperature, top-k, top-p sampling и KV-кэша.
    
        Аргументы:
            x (torch.Tensor): Входной тензор с индексами токенов [batch_size, seq_len].
            max_new_tokens (int): Максимальное количество новых токенов для генерации.
            do_sample (bool): Режим генерации:
                - True: вероятностное сэмплирование (random sampling)
                - False: жадный (greedy) поиск (выбор argmax на каждом шаге)
            temperature (float): Температура распределения (>0, по умолчанию 1.0).
                - >1.0 — генерация более "творческая"/приподнятая вероятность "редких" токенов;
                - <1.0 — более предсказуемый и суженный выбор.
            top_k (int, опционально): Если задан, sampling только из top_k самых вероятных токенов (top-k sampling).
            top_p (float, опционально): Если задан, sampling только из токенов, кумулятивная вероятность которых ≤ top_p (nucleus/top-p sampling, см. Holtzman et al., 2019).
            use_cache (bool, по умолчанию True): Использовать кэш attention KV для ускорения авторегрессии.
    
        Возвращает:
            torch.Tensor: Тензор индексов токенов [batch_size, seq_len + max_new_tokens].
    
        Исключения:
            ValueError: Если x длиннее максимальной длины (max_seq_len).
            ValueError: Если temperature ≤ 0.
            ValueError: Если одновременно заданы top_k и top_p.
            ValueError: Если top_k ≤ 0.
            ValueError: Если top_p не в диапазоне (0, 1].
    
        Примеры использования:
            >>> # Жадная генерация
            >>> output = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    
            >>> # Сэмплирование с температурой
            >>> output = model.generate(input_ids, max_new_tokens=20, do_sample=True, temperature=0.8)
    
            >>> # Top-k sampling
            >>> output = model.generate(input_ids, max_new_tokens=20, do_sample=True, top_k=50)
    
            >>> # Top-p (nucleus) sampling
            >>> output = model.generate(input_ids, max_new_tokens=20, do_sample=True, top_p=0.92)
    
            >>> # Комбинация температуры и top-k
            >>> output = model.generate(input_ids, max_new_tokens=20, do_sample=True, temperature=0.7, top_k=40)
    
        Примечания:
            - Для детерминированных результатов используйте torch.manual_seed.
            - temperature, top_k, top_p работают только при do_sample=True.
            - Только один из top_k/top_p может быть задан одновременно.
            - Метод всегда возвращает индексы токенов (ids); для получения логитов используйте forward.
    
        Ссылки:
            - Holtzman et al., "The Curious Case of Neural Text Degeneration" (nucleus sampling): https://arxiv.org/abs/1904.09751
            - Оригинальная статья GPT-2: https://cdn.openai.com/better-language-models/language-models.pdf
        """
        cache = None

        for _ in range(max_new_tokens):
            if use_cache and cache is not None:
                # Используем кэш - передаем только последний токен
                x_input = x[:, -1:]  # [batch_size, 1]
            else:
                # Первая итерация или кэш отключен - передаем всю последовательность
                x_input = x

            # Прямой проход с кэшем
            logits, new_cache = self.forward(x_input, use_cache=use_cache, cache=cache)

            # Обновляем кэш для следующей итерации
            if use_cache:
                cache = new_cache

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

                # создаём маску: 1, если токен НЕ в topk_indices
                mask = torch.ones_like(logits_scaled, dtype=torch.bool if hasattr(torch, "bool") else torch.uint8)
                mask.scatter_(1, topk_indices, False if hasattr(torch, "bool") else 0)  # 0 там, где top-k индексы
                masked_logits[mask.bool() if hasattr(torch, "bool") else mask.byte()] = float('-inf')
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
                sorted_mask = (cum_probs <= top_p).bool() if hasattr(torch, "bool") else  (cum_probs <= top_p).byte()  # [B, vocab_size]
                # Гарантируем, что хотя бы первый токен останется
                sorted_mask[:, 0] = True if hasattr(torch, "bool") else 1
                # 5. Преобразуем маску обратно в оригинальный порядок:
                # Создаём полную маску из 0
                mask = torch.zeros_like(probs, dtype=torch.bool if hasattr(torch, "bool") else torch.uint8)
                # Устанавливаем 1 в местах нужных токенов
                mask.scatter_(dim=1, index=sorted_indices, src=sorted_mask)
                # 6. Зануляем логиты токенов вне топ-p:
                logits_scaled[~mask] = float('-inf')

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

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len
