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
        >>> logits, _ = model(tokens)
        >>> generated = model.generate(tokens, max_new_tokens=16, do_sample=True, top_k=50)

    References:
    -----------
    - Jiang et al., "Mistral 7B" (2023): https://arxiv.org/abs/2310.06825
    - LLaMA v2 & Grouped-Query Attention: https://arxiv.org/abs/2307.09288
    - Оригинальное обсуждение архитектуры: https://huggingface.co/blog/mistral

    """
    def __init__(self, config):
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
            # база частот RoPE: 10 000 по умолчанию — как в Mistral 7B v0.1
            base=config.get("rope_theta", 10_000),
        )
        self._dropout = nn.Dropout(config["dropout"])
        self._decoders = nn.ModuleList([MistralDecoder(
            num_q_heads=config["num_q_heads"],
            num_kv_heads=config["num_kv_heads"],
            emb_size=config["embed_dim"],
            head_size=head_size,
            max_seq_len=config["max_position_embeddings"],
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
        Прямой проход (forward) через всю модель Mistral: возвращает логиты для токенов и (опционально) кэш attention для ускорения autoregressive генерации.
    
        Аргументы:
            x (torch.Tensor): Входной тензор с токенами (shape [batch_size, seq_len]), где значения — индексы токенов.
            use_cache (bool, по умолчанию False): Возвращать ли новый KV attention-кэш для последующей генерации.
            cache (list or None): Предыдущий кэш attention (или None для полного прохода без накопления кэша).
            attention_mask (torch.Tensor, опц.): маска [batch, seq_len] (1 — токен, 0 — паддинг).
                Поддерживается правый паддинг; на другие маски с нулями — NotImplementedError
                (см. docs/README.md, раздел «Маски»).
    
        Возвращает:
            logits (torch.Tensor): Тензор логитов shape [batch_size, seq_len, vocab_size] — вероятностное распределение по словарю для каждого токена.
            new_cache (list or None): Новый кэш KV attention-слоев (или None, если use_cache=False).
    
        Исключения:
            ValueError: Если длина последовательности с учётом кэша превышает максимальную (max_seq_len).
    
        Пример:
            >>> logits, cache = model.forward(input_ids, use_cache=True)
            >>> probabilities = torch.softmax(logits, dim=-1)
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
