"""
Инициализация весов по оригинальным статьям и коду: GPT-1 и GPT-2 (OpenAI), LLaMA, Mistral,
Mixtral и Gemma (initializer_range = 0.02 в их HF-конфигах).

Инициализация по умолчанию в PyTorch рассчитана на другое: веса Linear получают std
≈ 1/√(3·fan_in) (для 256 входов ≈ 0.036), эмбеддинги — N(0, 1). GPT-1 и GPT-2 обучались
с N(0, 0.02), а GPT-2 вдобавок уменьшает веса проекций, которые пишут в residual-поток,
чтобы его дисперсия не росла с глубиной сети.

Инициализация важна только для обучения с нуля: загрузка чекпоинта её перезаписывает.
"""

import math

from torch import nn

# Стандартное отклонение из GPT-1 (разд. 4.1) и кода gpt-2; в HF — initializer_range
DEFAULT_INITIALIZER_RANGE = 0.02


def init_normal_(module: nn.Module, std: float = DEFAULT_INITIALIZER_RANGE) -> None:
    """
    Инициализирует один модуль как в GPT: Linear и Embedding — N(0, std), bias — нули,
    LayerNorm — вес 1 и сдвиг 0. Предназначена для model.apply(...).
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def scale_residual_projections_(
    residual_projections, num_layers: int, std: float = DEFAULT_INITIALIZER_RANGE
) -> None:
    """
    Переинициализирует выходные проекции residual-веток GPT-2: N(0, std / √(2·num_layers)).

    В каждом блоке в residual-поток пишут две проекции — выход attention и второй слой
    FFN, всего 2·num_layers слагаемых. Статья GPT-2 (разд. 2.3) масштабирует их веса на
    1/√N; здесь N = 2·num_layers, как в GPT2PreTrainedModel._init_weights в HuggingFace.
    """
    residual_std = std / math.sqrt(2 * num_layers)
    for projection in residual_projections:
        nn.init.normal_(projection.weight, mean=0.0, std=residual_std)
