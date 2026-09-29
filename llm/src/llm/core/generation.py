"""
Общие вспомогательные функции прямого прохода и генерации для всех моделей.
"""

from typing import Optional

import torch


def validate_sampling_args(
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> None:
    """
    Проверяет параметры сэмплирования generate.

    temperature, top_k и top_p влияют только на сэмплирование, поэтому при
    do_sample=False (жадная генерация) не проверяются: например, temperature=0
    там допустима.

    Raises:
        ValueError: Если при do_sample=True temperature ≤ 0, одновременно заданы
            top_k и top_p, top_k ≤ 0 или top_p вне диапазона (0, 1].
    """
    if not do_sample:
        return
    if temperature <= 0:
        raise ValueError(
            f"temperature должна быть > 0 при do_sample=True, получено {temperature}"
        )
    if top_k is not None and top_p is not None:
        raise ValueError("top_k и top_p нельзя задавать одновременно")
    if top_k is not None and top_k <= 0:
        raise ValueError(f"top_k должен быть > 0, получено {top_k}")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError(f"top_p должен быть в диапазоне (0, 1], получено {top_p}")


def cache_start_pos(cache: Optional[list]) -> int:
    """
    Абсолютная позиция первого нового токена для переданного кэша.

    Кэш — список по слоям. Слой MHA/MQA хранит (K, V), и позиция равна длине
    закэшированной последовательности. Слой GQA хранит (K, V, next_pos): K и V
    обрезаны до окна внимания, поэтому позиция хранится отдельно.
    """
    if cache is None:
        return 0
    layer_cache = cache[0]
    if len(layer_cache) == 3:
        return layer_cache[2]
    return layer_cache[0].size(2)


def check_sequence_length(seq_len: int, start_pos: int, max_seq_len: int) -> None:
    """
    Проверяет, что новые токены помещаются в max_seq_len с учётом кэша.

    Raises:
        ValueError: Если start_pos + seq_len > max_seq_len.
    """
    if start_pos + seq_len > max_seq_len:
        raise ValueError(
            f"Длина последовательности {start_pos + seq_len} "
            f"(кэш {start_pos} + новые токены {seq_len}) превышает максимальную {max_seq_len}"
        )


def check_attention_mask(
    attention_mask: Optional[torch.Tensor],
    x: torch.Tensor,
    cache: Optional[list] = None,
    generating: bool = False,
) -> None:
    """
    Проверяет, что attention_mask можно обработать без явного маскирования.

    Модели накладывают только causal-маску (и скользящее окно). Этого достаточно для
    правого паддинга: pad-токены стоят после настоящих, и causal-маска и так не даёт
    настоящим токенам на них смотреть, поэтому их выход совпадает с выходом без
    паддинга. Левый паддинг требует маски ключей и сдвига позиций и не поддерживается;
    вместо молчаливо неверного результата бросается NotImplementedError.

    Допускаются:
        - None или маска из одних единиц;
        - в forward без кэша — правый паддинг: в каждой строке единицы, затем нули.

    Raises:
        ValueError: Если форма маски не совпадает с [batch, seq_len] входа
            (с кэшем допускается и полная длина [batch, cache_len + seq_len]).
        NotImplementedError: Если в маске есть нули, которые нельзя обработать:
            левый паддинг, пропуски, нули при генерации или вместе с кэшем.
    """
    if attention_mask is None:
        return
    batch_size, seq_len = x.shape
    allowed_lengths = {seq_len, cache_start_pos(cache) + seq_len}
    if (
        attention_mask.dim() != 2
        or attention_mask.size(0) != batch_size
        or attention_mask.size(1) not in allowed_lengths
    ):
        raise ValueError(
            f"attention_mask должна иметь форму [batch, seq_len] = [{batch_size}, {seq_len}], "
            f"получено {list(attention_mask.shape)}"
        )
    mask = attention_mask != 0
    if bool(mask.all()):
        return
    if generating:
        raise NotImplementedError(
            "generate не поддерживает attention_mask с нулями (паддинг в промптах): "
            "генерация батчем промптов разной длины требует левого паддинга и сдвига позиций"
        )
    if cache is not None:
        raise NotImplementedError(
            "attention_mask с нулями вместе с кэшем не поддерживается"
        )
    # Правый паддинг: вдоль строки маска не возрастает (1 … 1 0 … 0)
    if not bool((mask[:, 1:] <= mask[:, :-1]).all()):
        raise NotImplementedError(
            "поддерживается только правый паддинг (единицы, затем нули в каждой строке); "
            "левый паддинг требует маски ключей и сдвига позиций"
        )


def next_generation_input(
    x: torch.Tensor,
    cache: Optional[list],
    use_cache: bool,
    max_seq_len: int,
):
    """
    Выбирает вход и кэш для очередного шага generate.

    Пока последовательность помещается в max_seq_len, с кэшем подаётся только последний
    токен. Когда она становится длиннее, берутся последние max_seq_len токенов и кэш
    сбрасывается: окно сдвинулось, абсолютные позиции всех токенов изменились, и
    закэшированные K/V (посчитанные со старыми позициями) больше не годятся.

    Returns:
        (x_input, cache): вход для forward и кэш, который нужно ему передать.
    """
    if x.size(1) > max_seq_len:
        return x[:, -max_seq_len:], None
    if use_cache and cache is not None:
        return x[:, -1:], cache
    return x, None


def sample_next_token(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Выбирает следующий токен по логитам последней позиции.

    Args:
        logits: логиты [batch, vocab_size]; не изменяются.
        do_sample: False — жадный выбор (argmax), остальные параметры не влияют;
            True — сэмплирование из softmax(logits / temperature).
        temperature: температура сэмплирования (> 0).
        top_k: оставить только top_k самых вероятных токенов; значение больше
            размера словаря означает весь словарь.
        top_p: nucleus sampling — оставить минимальный набор самых вероятных токенов,
            суммарная вероятность которых не меньше top_p (Holtzman et al., 2019).

    Returns:
        Индексы выбранных токенов [batch, 1].
    """
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        topk_indices = torch.topk(logits, top_k, dim=-1).indices
        keep = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, topk_indices, True)
        logits = logits.masked_fill(~keep, float("-inf"))

    if top_p is not None:
        sorted_probs, sorted_indices = torch.sort(
            torch.softmax(logits, dim=-1), descending=True, dim=-1
        )
        # Токен входит в ядро, если сумма вероятностей более вероятных токенов (без него)
        # меньше top_p. Так в ядро попадает и токен, на котором сумма переходит порог, а
        # самый вероятный токен остаётся всегда (сумма до него 0 < top_p).
        prob_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
        keep_sorted = prob_before < top_p
        keep = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            -1, sorted_indices, keep_sorted
        )
        logits = logits.masked_fill(~keep, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
