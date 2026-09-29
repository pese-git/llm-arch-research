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
    - forward(x, use_cache=False, cache=None) — прямой проход, поддерживает batched вход, caching.
    - generate(...) — авторегрессивная генерация с разными стратегиями sampling и ускорением через cache.
    - save(path) / Mixtral.load(path, device) — сохранение и восстановление модели с конфигом (из BaseModel).

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
    - Jiang et al., "Mixtral of Experts" (2024): https://arxiv.org/abs/2401.04088
    - Mixtral 8x7B (блог): https://mistral.ai/news/mixtral-of-experts/
    - Switch Transformer: https://arxiv.org/abs/2101.03961
    - GShard: https://arxiv.org/abs/2006.16668
    - RoPE: https://arxiv.org/abs/2104.09864
    - Grouped Query Attention (Ainslie et al., 2023): https://arxiv.org/abs/2305.13245
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

        # Размер головы: head_size из конфига или embed_dim // num_q_heads (с проверками)
        head_size = resolve_head_size(config, "num_q_heads", rope=True)
        # eps всех RMSNorm: 1e-6 по умолчанию (LLaMA, Gemma), у Mistral 7B — 1e-5
        norm_eps = config.get("rms_norm_eps", 1e-6)
        
        self._max_seq_len = config["max_position_embeddings"]

        # Инициализация слоев
        self._token_embeddings = TokenEmbeddings(
            vocab_size=config["vocab_size"], 
            emb_size=config["embed_dim"]
        )
        self._position_embeddings = RoPE(
            head_size=head_size,
            max_seq_len=config["max_position_embeddings"],
            # база частот RoPE: 10 000 по умолчанию; у Mixtral 8x7B — 1e6 (задаётся в конфиге)
            base=config.get("rope_theta", 10_000),
        )
        self._dropout = nn.Dropout(config["dropout"])
        self._decoders = nn.ModuleList([MixtralDecoder(
            num_q_heads=config["num_q_heads"],
            num_kv_heads=config["num_kv_heads"],
            emb_size=config["embed_dim"],
            head_size=head_size,
            max_seq_len=config["max_position_embeddings"],
            num_experts=config["num_experts"],
            top_k_experts=config["top_k_experts"],
            window_size=config["window_size"],
            rope=self._position_embeddings,
            dropout=config["dropout"],
            norm_eps=norm_eps
        ) for _ in range(config["num_layers"])])
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
        Прямой проход (forward) через всю модель Mixtral.

        Данный метод реализует трансформацию входной последовательности токенов в логиты (предсказания вероятностей токенов словаря)
        с поддержкой эффективного инференса с использованием cache (KV-кэш attention для автогенерации).

        Аргументы:
        ----------
        x : torch.Tensor
            Двумерный входной тензор shape [batch_size, seq_len], где каждое значение — ID токена.
        use_cache : bool, по умолчанию False
            Если True — в режиме генерации модель возвращает обновлённый список кэшей attention для ускорения последовательного инференса.
            Если False — attention cache не используется.
        cache : list, optional
            (Необязательно) Список (или None) с кэшем KV attention для каждого слоя. Используется для автогенерации текста.
        attention_mask : torch.Tensor, optional
            Маска [batch, seq_len] (1 — токен, 0 — паддинг). Поддерживается правый паддинг:
            causal-маска и так скрывает от настоящих токенов стоящий после них паддинг.
            На другие маски с нулями — NotImplementedError (см. docs/README.md, раздел «Маски»).

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
        # Длина с учётом кэша: позиции start_pos … start_pos + seq_len − 1 должны быть < max_seq_len.
        # attention_mask допускается только такая, при которой causal-маски достаточно.
        check_sequence_length(x.size(1), cache_start_pos(cache), self._max_seq_len)
        check_attention_mask(attention_mask, x, cache)
        
        # Эмбеддинги токенов и позиций
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
