import torch
import math
from torch import nn
from torch import Tensor
from math import sqrt
from llm.core.base_model import BaseModel
from llm.core.config_checks import resolve_head_size
from llm.core.generation import (
    cache_start_pos,
    check_attention_mask,
    check_sequence_length,
)
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
    - forward(x, use_cache=False, cache=None): выдает батч логитов по токенам, возвращает при необходимости обновленный cache.
    - generate(...): автотекстогенерация с greedy, temperature, top-k/p sampling, поддержкой кэша (ускорение inference).
    - save(path) / Gemma.load(path, device): сохранение и загрузка весов вместе с конфигом (из BaseModel).

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

        # Размер головы: head_size из конфига или embed_dim // num_q_heads (с проверками)
        head_size = resolve_head_size(config, "num_q_heads", rope=True)

        self._max_seq_len = config["max_position_embeddings"]

        # Инициализация слоев
        self._token_embeddings = TokenEmbeddings(
            vocab_size=config["vocab_size"], 
            emb_size=config["embed_dim"]
        )
        self._position_embeddings = RoPE(
            head_size=head_size,
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
            head_size=head_size,
            max_seq_len=config["max_position_embeddings"],
            rope=self._position_embeddings,
            dropout=config["dropout"]  
        ) for _ in range(config["num_layers"])])
        self._norm = RMSNorm(config["embed_dim"])
        self._linear = nn.Linear(config["embed_dim"], config["vocab_size"])

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        cache: list = None,
        attention_mask: torch.Tensor = None,
    ) -> tuple:
        """
        Прямой проход (forward) через полную модель Gemma.

        Трансформирует входную последовательность токенов через стек из декодерных блоков GemmaDecoder.
        Возвращает логиты по всем токенам и (при необходимости) кэш attention для быстрой autoregressive-генерации.

        Аргументы:
        ----------
        x : torch.Tensor
            Входной тензор shape [batch_size, seq_len], содержащий токен-IDs.
        use_cache : bool, по умолчанию False
            Если True — сохраняет и возвращает KV-кэш attention (ускоряет автогенерацию).
            Если False — кэш не используется.
        cache : list, optional
            (Необязательно) Список/None: с кэшами KV-матриц для каждого слоя (для режима генерации статей/диalogов).
        attention_mask : torch.Tensor, optional
            Маска [batch, seq_len] (1 — токен, 0 — паддинг). Поддерживается правый паддинг:
            causal-маска и так скрывает от настоящих токенов стоящий после них паддинг.
            На другие маски с нулями — NotImplementedError (см. docs/README.md, раздел «Маски»).

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
        # Длина с учётом кэша: позиции start_pos … start_pos + seq_len − 1 должны быть < max_seq_len.
        # attention_mask допускается только такая, при которой causal-маски достаточно.
        check_sequence_length(x.size(1), cache_start_pos(cache), self._max_seq_len)
        check_attention_mask(attention_mask, x, cache)
        
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
