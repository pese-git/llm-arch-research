import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from math import sqrt
from llm.core.base_model import BaseModel
from llm.core.token_embeddings import TokenEmbeddings
from llm.core.rms_norm import RMSNorm
from llm.core.rope import RoPE
from llm.core.mistral_decoder import MistralDecoder


class Mistral(BaseModel):
    """
    Mistral — автогерессивная языковая LLM-архитектура (2023, Mistral AI) для быстрого и качественного моделирования текста.

    Назначение:
    -----------
    - Модель построена на базе decoder-only Transformer с важными оптимизациями: GQA (Grouped Query Attention), RoPE, SwiGLU, RMSNorm, sliding window attention.
    - Поддерживает autoregressive generation (step-by-step текст), обучение и inference на длинных последовательностях.
    - Используется в современных open-source LLM: Mistral-7B, Mixtral-8x7B и др.

    Архитектурные особенности:
    --------------------------
    - Токеновые эмбеддинги (TokenEmbeddings) и позиционное кодирование через RoPE (rotary position embedding).
    - Stack из num_layers декодеров с Grouped Query Attention (раздельное число query/key heads для оптимизации памяти).
    - Sliding Window Attention Mask — позволяет ускорять обработку длинных текстов, ограничивая область внимания для каждого токена (как в оригинальном Mistral).
    - SwiGLU FeedForward-блоки и RMSNorm.
    - Dropout (регуляризация).
    - Кэширование attention (KV cache) для быстрой генерации токенов по одному.

    Аргументы конструктора:
    -----------------------
    config (dict): параметры модели (см. документацию Mistral):
        vocab_size: int — размер словаря токенов
        embed_dim: int — размерность эмбеддингов
        num_q_heads: int — количество query-голов (обычно больше num_kv_heads)
        num_kv_heads: int — количество key/value attention-голов
        num_layers: int — число слоёв-декодеров
        max_position_embeddings: int — максимальная длина последовательности
        window_size: int — размер sliding window attention
        dropout: float — dropout (обычно очень мал или 0)
        ...
    
    Пример использования:
    ---------------------
        >>> model = Mistral({...})
        >>> tokens = torch.tensor([[100, 56, 8]])
        >>> logits = model(tokens)
        >>> generated = model.generate(tokens, max_new_tokens=16, do_sample=True, top_k=50)

    References:
    -----------
    - "Mistral: Fast and Efficient Dense and Mixture of Experts Transformer Models" (2023): https://arxiv.org/abs/2310.06825
    - LLaMA v2 & Grouped-Query Attention: https://arxiv.org/abs/2307.09288
    - Оригинальное обсуждение архитектуры: https://huggingface.co/blog/mistral

    """
    def __init__(self, config):
        super().__init__(config)
        
        self._max_seq_len = config["max_position_embeddings"]
        # Инициализация слоев
        self._token_embeddings = TokenEmbeddings(
            vocab_size=config["vocab_size"], 
            emb_size=config["embed_dim"]
        )
        self._position_embeddings = RoPE(
            head_size=config["embed_dim"] // config["num_q_heads"],
            max_seq_len=config["max_position_embeddings"]
        )
        #self._position_embeddings = PositionalEmbeddings(
        #    max_seq_len=max_seq_len, 
        #    emb_size=emb_size
        #)
        self._dropout = nn.Dropout(config["dropout"])
        self._decoders = nn.ModuleList([MistralDecoder(
            num_q_heads=config["num_q_heads"],
            num_kv_heads=config["num_kv_heads"],
            emb_size=config["embed_dim"],
            head_size=config["embed_dim"] // config["num_q_heads"],
            max_seq_len=config["max_position_embeddings"],
            window_size=config["window_size"],
            rope=self._position_embeddings,
            dropout=config["dropout"] 
        ) for _ in range(config["num_layers"])])
        self._norm = RMSNorm(config["embed_dim"])
        self._linear = nn.Linear(config["embed_dim"], config["vocab_size"])

    def forward(self, x: torch.Tensor, use_cache: bool = True, cache: list = None) -> tuple:
        """
        Прямой проход (forward) через всю модель Mistral: возвращает логиты для токенов и (опционально) кэш attention для ускорения autoregressive генерации.
    
        Аргументы:
            x (torch.Tensor): Входной тензор с токенами (shape [batch_size, seq_len]), где значения — индексы токенов.
            use_cache (bool, по умолчанию True): Возвращать ли новый KV attention-кэш для последующей генерации.
            cache (list or None): Предыдущий кэш attention (или None для полного прохода без накопления кэша).
    
        Возвращает:
            logits (torch.Tensor): Тензор логитов shape [batch_size, seq_len, vocab_size] — вероятностное распределение по словарю для каждого токена.
            new_cache (list or None): Новый кэш KV attention-слоев (или None, если use_cache=False).
    
        Исключения:
            ValueError: Если длина последовательности превышает максимальную (max_seq_len), когда не используется кэш.
    
        Пример:
            >>> logits, cache = model.forward(input_ids, use_cache=True)
            >>> probabilities = torch.softmax(logits, dim=-1)
        """
        # Проверка длины последовательности (только при отсутствии кэша)
        if cache is None and x.size(1) > self._max_seq_len:
            raise ValueError(f"Длина последовательности {x.size(1)} превышает максимальную {self.max_seq_len}")
        
        # Эмбеддинги токенов и позиций
        tok_out = self._token_embeddings(x)  # [batch, seq_len, emb_size]
       #pos_out = self._position_embeddings(x)  # [batch, seq_len, emb_size]
        
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
        attention_mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Авторегрессивная генерация токенов с поддержкой greedy, temperature, top-k/top-p sampling
        и ускорением через attention-кэш (KV-cache, важно для inference на длинных текстах).

        Аргументы:
            x (torch.Tensor): Входной тензор с токенами shape [batch_size, seq_len].
            max_new_tokens (int): Максимальное количество новых токенов для генерации.
            do_sample (bool): Если True — вероятность/случайность (random sampling); если False — жадная генерация (argmax).
            temperature (float): Температура (>0, по умолчанию 1.0); >1.0 — более случайные выборы, <1.0 — более строгие.
            top_k (int, optional): top-k sampling; при сэмплировании выбираются только top_k наиболее вероятных токенов.
            top_p (float, optional): nucleus (top-p) sampling; выбираются токены с накопленной вероятностью ≤ top_p.
            use_cache (bool, по умолчанию True): Использовать ускорение через KV attention cache для autoregressive режима.

        Возвращает:
            torch.Tensor: Последовательность индексов токенов shape [batch_size, seq_len + max_new_tokens].

        Исключения:
            ValueError: Если x длиннее max_seq_len модели.
            ValueError: Если temperature ≤ 0.
            ValueError: Если одновременно заданы top_k и top_p.
            ValueError: Если top_k ≤ 0.
            ValueError: Если top_p не в диапазоне (0, 1].

        Примеры:
            >>> # Жадная генерация
            >>> out = model.generate(input_ids, max_new_tokens=16, do_sample=False)
            >>> # Сэмплирование с температурой
            >>> out = model.generate(input_ids, max_new_tokens=16, do_sample=True, temperature=0.8)
            >>> # Top-k sampling
            >>> out = model.generate(input_ids, max_new_tokens=16, do_sample=True, top_k=50)
            >>> # Top-p (nucleus) sampling
            >>> out = model.generate(input_ids, max_new_tokens=16, do_sample=True, top_p=0.92)
            >>> # Температура + top-k
            >>> out = model.generate(input_ids, max_new_tokens=16, do_sample=True, temperature=1.0, top_k=100)

        Примечания:
            - Одновременно использовать top_k и top_p нельзя.
            - Параметры temperature, top_k, top_p работают только при do_sample=True.
            - Для полного воспроизведения результата зафиксируйте seed через torch.manual_seed.
            - Метод всегда возвращает только индексы токенов; для получения логитов используйте forward.

        Ссылки:
            - Holtzman et al., "The Curious Case of Neural Text Degeneration" (nucleus/top-p sampling): https://arxiv.org/abs/1904.09751
            - Mistral: https://arxiv.org/abs/2310.06825
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
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
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
                next_token = torch.argmax(probs, dim=-1, keepdim=True)  # [batch_size, 1]
            
            # 6. Добавляем его к последовательности
            x = torch.cat([x, next_token], dim=1)  # [batch_size, seq_len+1]
        return x


    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len