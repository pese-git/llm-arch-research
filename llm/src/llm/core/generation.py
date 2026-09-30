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


def check_generation_mask(
    attention_mask: Optional[torch.Tensor], x: torch.Tensor
) -> Optional[torch.Tensor]:
    """
    Проверяет attention_mask промпта для generate.

    Паддинг в промпте должен быть левым: генерация продолжается с последнего токена
    каждой строки, и он должен быть настоящим. Правый паддинг в generate дал бы продолжение
    с pad-токена, поэтому отклоняется.

    Returns:
        None, если маски нет или в ней одни единицы (generate идёт прежним путём),
        иначе маску как bool-тензор [batch, seq_len].

    Raises:
        ValueError: Если форма маски не [batch, seq_len] или последний токен какой-то строки — паддинг.
    """
    if attention_mask is None:
        return None
    if attention_mask.dim() != 2 or attention_mask.shape != x.shape:
        raise ValueError(
            f"attention_mask должна иметь форму промпта [batch, seq_len] = {list(x.shape)}, "
            f"получено {list(attention_mask.shape)}"
        )
    mask = attention_mask != 0
    if bool(mask.all()):
        return None
    if not bool(mask[:, -1].all()):
        raise ValueError(
            "в generate паддинг должен быть слева: последний токен каждой строки промпта — "
            "настоящий (с него продолжается генерация)"
        )
    return mask


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
