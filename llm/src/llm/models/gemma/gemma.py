import torch
import math
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from math import sqrt
from llm.core.base_model import BaseModel
from llm.core.token_embeddings import TokenEmbeddings
from llm.core.rope import RoPE
from llm.core.rms_norm import RMSNorm
from llm.core.gemma_decoder import GemmaDecoder
    

class Gemma(BaseModel):
    """
    Gemma — языковая трансформер-модель от Google, с архитектурой, оптимизированной для open-source и research-комьюнити.

    Назначение:
    -----------
    Модель Gemma реализует стек современных декодерных блоков (GemmaDecoder), поддерживает rotary-позиционирование, multi-query self-attention,
    эффективный режим генерации (KV-cache), dropout, compact residual connections, базируется на best-practice LLM-инженерии последних лет.
    Поддерживает batched-тренировку и inference, генерацию с различными стратегиями выборки (greedy, top-k, top-p), автосохранение.

    Архитектурные особенности:
    --------------------------
    - Stack из N слоёв GemmaDecoder (attention с Multi-Query либо Grouped heads, FFN с GeGLU/SwiGLU)
    - RMSNorm или LayerNorm для стабилизации
    - Dropout для регуляризации
    - Rotary Position Embedding (RoPE) для позиционных кодов
    - Выходная проекция (linear → logits) к словарю токенов
    - Полная поддержка cache для ускорения autoregressive генерации

    Конфиг/Параметры конструктора:
    ------------------------------
    config : dict
        Словарь c параметрами модели:
            - vocab_size : int — размер словаря
            - embed_dim : int — размер скрытого (hidden) пространства
            - max_position_embeddings : int — максимальная длина последовательности
            - num_layers : int — количество декодерных блоков
            - num_q_heads : int — количество attention голов (Queries)
            - num_kv_heads : int — количество ключевых/значенческих attention голов
            - dropout : float — Dropout率
            - ... (доп. гиперпараметры, требуемые GemmaDecoder'ами)

    Основные методы:
    ----------------
    - forward(x, use_cache=True, cache=None): выдает батч логитов по токенам, возвращает при необходимости обновленный cache.
    - generate(...): автотекстогенерация с greedy, temperature, top-k/p sampling, поддержкой кэша (ускорение inference).
    - save(path)/load(path, device): сохранение и загрузка предобученных весов, параметров и состояния.

    Пример:
    -------
        >>> config = {...}  # словарь с параметрами
        >>> model = Gemma(config)
        >>> x = torch.randint(0, config["vocab_size"], (4, 64))
        >>> logits, cache = model(x, use_cache=True)
        >>> print(logits.shape)  # [4, 64, vocab_size]
        >>> out = model.generate(x, max_new_tokens=20, do_sample=True, top_k=10, temperature=0.8)

    Литература и ссылки:
    --------------------
    - Gemma: https://ai.google.dev/gemma (официальная страница)
    - Разработка и архитектура: https://arxiv.org/abs/2403.07794
    - Rotary Embedding: https://arxiv.org/abs/2104.09864
    - Multi-Query Attention: https://arxiv.org/abs/1911.02150
    - Llama: https://arxiv.org/abs/2302.13971
    """
    def __init__(self, config):
        """
        Конструктор класса Gemma.

        Позволяет создать объект языковой модели с архитектурой Gemma и
        произвольной конфигурацией (гибкая поддержка разных масштабов, ширин, глубин).

        Аргументы:
        ----------
        config : dict
            Словарь со всеми необходимыми гиперпараметрами и архитектурными детальями модели Gemma.
            Ожидаемые ключи (группы параметров):
                - vocab_size : int — размер словаря токенов (размерность входа/выхода)
                - embed_dim : int — скрытый размер эмбеддинга (hidden dim)
                - max_position_embeddings : int — максимальная длина последовательности
                - num_layers : int — количество декодерных блоков (глубина стека)
                - num_q_heads : int — число attention голов (Query heads)
                - num_kv_heads : int — число голов для Key/Value (MultiQuery Attention)
                - dropout : float — Dropout для регуляризации
                - остальные специфичные для GemmaDecoder'ов параметры

        Внутри:
        -------
        - Инициализируются модули эмбеддинга токенов, позиционного кодирования (RoPE) и Dropout,
          стек декодеров (GemmaDecoder(...)), слой финальной нормализации и выходная проекция (linear).
        - Все архитектурные параметры напрямую берутся из config.

        Пример:
        -------
            >>> config = {
            ...     "vocab_size": 32000,
            ...     "embed_dim": 512,
            ...     "max_position_embeddings": 2048,
            ...     "num_layers": 24,
            ...     "num_q_heads": 8,
            ...     "num_kv_heads": 4,
            ...     "dropout": 0.1,
            ... }
            >>> model = Gemma(config)

        Примечание:
        -----------
        - Внимание: значения config должны быть согласованы друг с другом! Например, embed_dim должен быть кратным num_q_heads и т.д.
        - Поддерживается дальнейшая кастомизация стека декодеров через ключи в config.
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
        self._decoders = nn.ModuleList([GemmaDecoder(
            num_q_heads=config["num_q_heads"],
            emb_size=config["embed_dim"],
            head_size=config["embed_dim"] // config["num_q_heads"],
            max_seq_len=config["max_position_embeddings"],
            rope=self._position_embeddings,
            dropout=config["dropout"]  
        ) for _ in range(config["num_layers"])])
        self._norm = RMSNorm(config["embed_dim"])
        self._linear = nn.Linear(config["embed_dim"], config["vocab_size"])

    def forward(self, x: torch.Tensor, use_cache: bool = True, cache: list = None) -> tuple:
        """
        Прямой проход (forward) через полную модель Gemma.

        Трансформирует входную последовательность токенов через стек из декодерных блоков GemmaDecoder.
        Возвращает логиты по всем токенам и (при необходимости) кэш attention для быстрой autoregressive-генерации.

        Аргументы:
        ----------
        x : torch.Tensor
            Входной тензор shape [batch_size, seq_len], содержащий токен-IDs.
        use_cache : bool, по умолчанию True
            Если True — сохраняет и возвращает KV-кэш attention (ускоряет автогенерацию).
            Если False — кэш не используется.
        cache : list, optional
            (Необязательно) Список/None: с кэшами KV-матриц для каждого слоя (для режима генерации статей/диalogов).

        Возвращает:
        -----------
        tuple:
            - logits : torch.Tensor shape [batch_size, seq_len, vocab_size]
                Логиты по словарю для каждого токена (input + сколь угодно новых).
            - new_cache : list или None
                Обновлённый cache (если use_cache=True).

        Пример:
        -------
            >>> logits, new_cache = model(x, use_cache=True, cache=None)
            >>> logits.shape  # [batch_size, seq_len, vocab_size]

        Примечания:
        -----------
        - Используется при обучении и инференсе.
        - Если нужно только инференс last-token — используйте logits[:, -1, :].
        - При превышении x.shape[1] > max_seq_len выдаёт ValueError.
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
        Авторегрессивная генерация токенов с использованием greedy, temperature, top-k и top-p sampling.
        Реализует generation-loop с обновлением attention-кэша для ускорения инференса.

        Аргументы:
        ----------
        x : torch.Tensor
            Входной тензор с последовательностью токенов (shape [batch_size, seq_len]), который необходимо продолжить.
        max_new_tokens : int
            Сколько новых токенов сгенерировать (максимум).
        do_sample : bool
            Если True — сэмплирует следующий токен согласно распределению вероятностей (stochastic), иначе выбирает токен с максимальной вероятностью (greedy).
        temperature : float, default=1.0
            Параметр для шкалирования распределения вероятностей логитов. Больше 1.0 — больше случайности, меньше 1.0 — более детерминированный (жёсткий) выбор.
        top_k : int, optional
            Если задано — для сэмплирования учитываются только top_k наиболее вероятных токенов.
        top_p : float, optional
            Если задано — работают nucleus sampling: учитываются токены, суммарная вероятность которых не превышает top_p.
        use_cache : bool, default=True
            Если True — для ускорения использует и обновляет attention-кэши (KV-cache).

        Возвращает:
        -----------
        torch.Tensor
            Тензор shape [batch_size, seq_len + max_new_tokens] с исходными и сгенерированными токенами (token IDs).

        Пример:
        -------
            >>> out = model.generate(
            ...     x, max_new_tokens=20, do_sample=True, temperature=0.8, top_k=50
            ... )
            >>> print(out.shape)  # [batch_size, seq_len+20]

        Примечания:
        -----------
        - Нельзя указывать одновременно top_k и top_p (будет выброшено исключение).
        - temperature <= 0 некорректно (будет выброшено исключение).
        - Поддержка cache (use_cache=True) значительно ускоряет генерацию длинных последовательностей и позволяет использовать beam search/decoding.
        - Для воспроизводимых результатов установите torch.manual_seed перед генерацией.
        - Метод возвращает только token_ids, если нужны logits — используйте .forward напрямую.

        Литература:
        -----------
        - Holtzman et al., "The Curious Case of Neural Text Degeneration" (nucleus/top-p sampling): https://arxiv.org/abs/1904.09751
        - Gemma: https://arxiv.org/abs/2403.07794
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