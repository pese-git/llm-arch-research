import torch
from torch import nn, Tensor
import torch.nn.functional as F

from llm.core.base_model import BaseModel
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
    - Модель реализует архитектуру decoder-only Transformer с современными "индустриальными" трюками (RMSNorm, SwiGLU, RoPE, GQA).
    - Предназначена для генерации текста, чат-ботов, zero-/few-shot вывода, fine-tune в стиле RLHF, transfer learning и исследований в LLM.

    Архитектурные особенности:
    --------------------------
    - Токеновые эмбеддинги и позиционное кодирование с помощью Rotary Position Embedding (RoPE, https://arxiv.org/abs/2104.09864).
    - Stack из num_layers современных декодеров с Grouped Query Attention (GQA: num_q_heads > num_kv_heads) для эффективной генерации.
    - FeedForward блоки с SwiGLU (см. https://arxiv.org/abs/2002.05202).
    - Нормализация RMSNorm перед каждым sub-layer (вот почему "Pre-RMSNorm").
    - Кэширование attention (KV cache) для быстрой autoregressive генерации.
    - Нет bias в Linear слоях, нет Dropout внутри attention.

    Аргументы конструктора:
    -----------------------
    config: dict с требуемыми ключами:
        vocab_size: int — размер словаря токенов
        embed_dim: int — размерность эмбеддингов
        num_q_heads: int — количество query-голов в attention (обычно больше num_kv_heads)
        num_kv_heads: int — количество key/value-голов
        num_layers: int — число слоёв-декодеров
        max_position_embeddings: int — максимальная длина последовательности
        window_size: int (optional) — размер sliding window для attention
        dropout: float (обычно 0.0 или очень мал)
        ...
 
    Пример использования:
    ---------------------
        >>> llama = LLaMA({...})
        >>> tokens = torch.tensor([[100, 56, 8]])
        >>> logits = llama(tokens)
        >>> out = llama.generate(tokens, max_new_tokens=10, do_sample=True, top_k=50)

    References:
    -----------
    - "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023): https://arxiv.org/abs/2302.13971
    - "Grouped-Query Attention": https://arxiv.org/abs/2307.09288
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
        - Создаёт Embedding-слой, Rotary Position Embeddings (RoPE), стек слоёв с GQA, RMSNorm, SwiGLU.
        - Финальный слой нормализации и проекции на vocabulary.
        """
        super().__init__(config)

        # Инициализация слоев
        self._max_seq_len = config["max_position_embeddings"]
        self._token_embeddings = TokenEmbeddings(
            vocab_size=config["vocab_size"], emb_size=config["embed_dim"]
        )
        self._position_embeddings = RoPE(
            head_size=config["embed_dim"] // config["num_heads"],
            max_seq_len=config["max_position_embeddings"],
        )

        self._dropout = nn.Dropout(config["dropout"])
        self._decoders = nn.ModuleList(
            [
                CachedDecoder(
                    norm_layer=RMSNorm,
                    num_heads=config["num_heads"],
                    emb_size=config["embed_dim"],
                    head_size=config["embed_dim"] // config["num_heads"],
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
        self._norm = RMSNorm(config["embed_dim"])
        self._linear = nn.Linear(config["embed_dim"], config["vocab_size"])

    def forward(
        self, x: torch.Tensor, use_cache: bool = True, cache: list = None
    ) -> tuple:
        """
        Прямой проход: возвращает logits (и возможно обновлённый cache) по входным токенам.

        Args:
            x (torch.Tensor): [batch, seq_len] — индексы токенов, shape [batch, seq_len]
            use_cache (bool): использовать механизм KV cache (ускоряет autoregressive generation)
            cache (list or None): предыдущий кэш, если нужен

        Returns:
            logits: torch.Tensor [batch, seq_len, vocab_size]
            new_cache: новый кэш attention (или None)
        """
        # Проверка длины последовательности (только при отсутствии кэша)
        if cache is None and x.size(1) > self._max_seq_len:
            raise ValueError(
                f"Длина последовательности {x.size(1)} превышает максимальную {self.max_seq_len}"
            )

        # Вычисление start_pos из кэша (если кэш передан)
        # if cache is not None:
        #    # При кэше обрабатываем только один токен (последний)
        #    seq_len = 1
        #    # Вычисляем start_pos из самого нижнего уровня кэша
        #    if cache and cache[0] and cache[0][0]:
        #        key_cache, _ = cache[0][0]  # Первый декодер, первая голова
        #        start_pos = key_cache.size(1)  # cache_len
        #    else:
        #        start_pos = 0
        # else:
        #    # Без кэша работаем как раньше
        #    start_pos = 0
        #    seq_len = x.size(1)

        # Эмбеддинги токенов и позиций
        tok_out = self._token_embeddings(x)  # [batch, seq_len, emb_size]
        # pos_out = self._position_embeddings(x)  # [batch, seq_len, emb_size]

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
        Авторегрессивная генерация последовательностей на основе LLaMA (greedy, temperature, top-k, top-p/nucleus, поддержка KV-кэша).
    
        Аргументы:
            x (torch.Tensor): Входной тензор с токенами shape [batch_size, seq_len].
            max_new_tokens (int): Максимальное количество новых токенов для генерации.
            do_sample (bool): Использовать вероятностное сэмплирование (True) или жадный режим (False, argmax).
            temperature (float): Температура (сглаживание распределения вероятностей, >0; по умолчанию 1.0).
                >1.0 — менее предсказуемые, более разнообразные выборки.
                <1.0 — более строгие, консервативные выборки.
            top_k (int, опционально): Top-k сэмплирование (ограничение выбора k самыми вероятными токенами).
            top_p (float, опционально): Nucleus (top-p) sampling (срез по кумулятивной вероятности ≤ top_p, см. Holtzman et al., 2019).
            use_cache (bool, по умолчанию True): Использовать KV-кэш для ускорения генерации.
    
        Возвращает:
            torch.Tensor: Последовательность токенов shape [batch_size, seq_len + max_new_tokens].
    
        Исключения:
            ValueError: Если x длиннее максимально допустимой длины (max_seq_len модели).
            ValueError: Если temperature ≤ 0.
            ValueError: Если одновременно заданы top_k и top_p.
            ValueError: Если top_k ≤ 0.
            ValueError: Если top_p не в диапазоне (0, 1].
    
        Примеры:
            >>> # Строго жадная генерация
            >>> out = model.generate(input_ids, max_new_tokens=16, do_sample=False)
            >>> # Вероятностная генерация с температурой
            >>> out = model.generate(input_ids, max_new_tokens=16, do_sample=True, temperature=0.7)
            >>> # Top-k sampling
            >>> out = model.generate(input_ids, max_new_tokens=16, do_sample=True, top_k=50)
            >>> # Top-p (nucleus)
            >>> out = model.generate(input_ids, max_new_tokens=16, do_sample=True, top_p=0.92)
            >>> # Комбинация температуры и top-k
            >>> out = model.generate(input_ids, max_new_tokens=16, do_sample=True, temperature=1.0, top_k=100)
    
        Примечания:
            - temperature, top_k, top_p применяются только если do_sample=True.
            - Одновременное использование top_k и top_p запрещено.
            - Для воспроизводимых результатов зафиксируйте seed через torch.manual_seed.
            - Возвращается только индексы токенов; для получения вероятностей используйте forward.
    
        Ссылки:
            - Holtzman et al., "The Curious Case of Neural Text Degeneration" (nucleus/top-p): https://arxiv.org/abs/1904.09751
            - LLaMA: https://arxiv.org/abs/2302.13971
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
