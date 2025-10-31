import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from math import sqrt
from llm.core.base_model import BaseModel
from llm.core.token_embeddings import TokenEmbeddings
from llm.core.rope import RoPE
from llm.core.rms_norm import RMSNorm
from llm.core.mixtral_decoder import MixtralDecoder





class Mixtral(BaseModel):
    """
    Mixtral — языковая модель с архитектурой Mixture-of-Experts на основе современных трансформеров (см. Mixtral 8x7B).

    Описание:
    ---------
    Данный класс реализует полностью функциональную LLM с блоками MixtralDecoder, которые используют разреженные Feed-Forward сети MoE (Mixture-of-Experts)
    и Grouped Query Attention (GQA). Позволяет масштабировать количество параметров без экспоненциального роста вычислительных затрат благодаря активации лишь части экспертов на каждый токен.
    Mixtral поддерживает автотекстогенерацию с caching, position encoding через RoPE и всё необходимое для работы и тренировки современных LLM.

    Архитектурные особенности:
    --------------------------
    - Stack из N слоёв MixtralDecoder (каждый — MoE-блок + attention + RMSNorm).
    - Dropout для регуляризации на уровне эмбеддингов и слоёв.
    - Позиционные эмбеддинги реализованы через RoPE (Rotary Positional Embeddings).
    - Финальная RMSNorm плюс Linear-проекция к словарю токенов.
    - Поддержка автогенерации с sampling (greedy, top-k, top-p), temperature и KV-cache.

    Аргументы конструктора:
    ----------------------
    config : dict
        Словарь-конфиг с основными гиперпараметрами модели:
            - vocab_size : int — размер словаря токенов
            - embed_dim : int — размер скрытого пространства
            - max_position_embeddings : int — макс. длина последовательности
            - num_layers : int — количество декодерных блоков в стеке
            - num_q_heads : int — число query-голов в attention
            - num_kv_heads : int — число kv-голов в attention
            - num_experts : int — число MoE-экспертов
            - top_k_experts : int — сколько экспертов активировать на токен
            - dropout : float — вероятность Dropout
            - window_size : int — размер окна внимания

    Основные методы:
    ----------------
    - forward(x, use_cache=True, cache=None) — прямой проход, поддерживает batched вход, caching.
    - generate(...) — авторегрессивная генерация с разными стратегиями sampling и ускорением через cache.
    - save(path)/load(path, device) — сохранение и восстановление обученной модели.

    Пример:
    -------
        >>> config = {...}  # dict с параметрами
        >>> model = Mixtral(config)
        >>> x = torch.randint(0, config["vocab_size"], (2, 16))
        >>> logits, cache = model(x, use_cache=True)
        >>> print(logits.shape)  # [2, 16, vocab_size]

        >>> # Генерация
        >>> out = model.generate(x, max_new_tokens=20, do_sample=True, top_k=10, temperature=0.9)

    Литература:
    -----------
    - Mixtral 8x7B: https://mistral.ai/news/mixtral-of-experts/
    - Switch Transformer: https://arxiv.org/abs/2101.03961
    - GShard: https://arxiv.org/abs/2006.16668
    - RoPE: https://arxiv.org/abs/2104.09864
    - Grouped Query Attention: https://arxiv.org/abs/2305.14236
    - RMSNorm: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, config):
        """
        Конструктор класса Mixtral.

        Осуществляет инициализацию всех модулей и внутренних параметров большой языковой модели с архитектурой Mixtral/MoE.
        Использует параметры из конфиг-словаря `config` для гибкой настройки модели.

        Аргументы:
        ----------
        config : dict
            Словарь с основными гиперпараметрами архитектуры. Должен содержать ключи:
                vocab_size (int): Размер словаря токенов.
                embed_dim (int): Размер скрытого пространства (эмбеддингов).
                max_position_embeddings (int): Максимальная длина токенной последовательности.
                num_layers (int): Количество декодерных блоков (слоёв) в модели.
                num_q_heads (int): Число query-голов (attention heads).
                num_kv_heads (int): Число key-value голов (attention heads).
                num_experts (int): Количество экспертов в каждом MoE-блоке.
                top_k_experts (int): Сколько экспертов активируется для одного токена.
                dropout (float): Dropout для регуляризации.
                window_size (int): Размер окна внимания (Attention Window).

        Внутри:
        -------
        - Инициализируются эмбеддинги токенов, позиционные эмбеддинги RoPE, Dropout.
        - Строится стек из num_layers модулей MixtralDecoder с заданным количеством attention heads и экспертов.
        - Финальный слой нормализации и проекция к логитам словаря (linear layer).

        Пример:
        -------
            >>> config = {
            ...     "vocab_size": 32000,
            ...     "embed_dim": 512,
            ...     "max_position_embeddings": 2048,
            ...     "num_layers": 24,
            ...     "num_q_heads": 8,
            ...     "num_kv_heads": 8,
            ...     "num_experts": 8,
            ...     "top_k_experts": 2,
            ...     "dropout": 0.1,
            ...     "window_size": 256,
            ... }
            >>> model = Mixtral(config)

        Примечания:
        -----------
        - Конфиг модели должен быть согласован: размеры должны делиться на число голов, число экспертов и top_k_experts корректно выбраны.
        - Все параметры, необходимые для построения MixtralDecoder, attention и MoE, берутся из config.
        """
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
        self._decoders = nn.ModuleList([MixtralDecoder(
            num_q_heads=config["num_q_heads"],
            num_kv_heads=config["num_kv_heads"],
            emb_size=config["embed_dim"],
            head_size=config["embed_dim"] // config["num_q_heads"],
            max_seq_len=config["max_position_embeddings"],
            num_experts=config["num_experts"],
            top_k_experts=config["top_k_experts"],
            window_size=config["window_size"],
            rope=self._position_embeddings,
            dropout=config["dropout"] 
        ) for _ in range(config["num_layers"])])
        self._norm = RMSNorm(config["embed_dim"])
        self._linear = nn.Linear(config["embed_dim"], config["vocab_size"])

    def forward(self, x: torch.Tensor, use_cache: bool = True, cache: list = None) -> tuple:
        """
        Прямой проход (forward) через всю модель Mixtral.

        Данный метод реализует трансформацию входной последовательности токенов в логиты (предсказания вероятностей токенов словаря)
        с поддержкой эффективного инференса с использованием cache (KV-кэш attention для автогенерации).

        Аргументы:
        ----------
        x : torch.Tensor
            Двумерный входной тензор shape [batch_size, seq_len], где каждое значение — ID токена.
        use_cache : bool, по умолчанию True
            Если True — в режиме генерации модель возвращает обновлённый список кэшей attention для ускорения последовательного инференса.
            Если False — attention cache не используется.
        cache : list, optional
            (Необязательно) Список (или None) с кэшем KV attention для каждого слоя. Используется для автогенерации текста.

        Возвращает:
        -----------
        tuple:
            - logits : torch.Tensor — выходной тензор shape [batch_size, seq_len, vocab_size] — массив логитов по токенам и словарю.
            - new_cache : list или None — обновлённый cache, если используется.

        Пример:
        -------
            >>> logits, new_cache = model(x, use_cache=True, cache=None)
            >>> logits.shape  # [batch_size, seq_len, vocab_size]

        Примечания:
        -----------
        - Если используется cache — эффективно для авторегрессионной генерации (token-by-token), например, при диалогах или длинной генерации.
        - Если входная последовательность длиннее max_seq_len — будет выброшено исключение.
        - Если нужен только логит последнего токена — используйте slice: logits[:, -1, :]

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




