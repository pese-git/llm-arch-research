"""
Общие вспомогательные функции генерации для всех моделей.
"""

from typing import Optional


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
