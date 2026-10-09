r"""
Loss автогрессивного языкового моделирования — одна реализация для `Trainer`
и `llm.evaluation`.

Модель в позиции t предсказывает токен t + 1, поэтому логиты и метки сдвигаются
на одну позицию относительно друг друга, а последняя позиция логитов и первая
позиция меток не участвуют:

    L = -\frac{1}{|S|} \sum_{t \in S} \log p(w_{t+1} \mid w_1, \dots, w_t),

где S — позиции, метка которых отлична от ignore_index (паддинг помечается -100,
см. llm/datasets/lm_example.py). Сдвиг делает loss, а не датасет: метки — копия
входа, иначе модель училась бы предсказывать токен через один.
"""

import torch
import torch.nn.functional as F

from llm.datasets.lm_example import IGNORE_INDEX


def causal_lm_loss(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """
    Cross-entropy следующего токена со сдвигом логитов и меток.

    Args:
        logits: [batch, seq_len, vocab_size] — логиты модели.
        labels: [batch, seq_len] — токены входа; позиции с ignore_index не входят в loss.
        ignore_index: метка, которую пропускаем (по умолчанию -100, как в HuggingFace).

    Returns:
        Скалярный тензор — средний loss по позициям с меткой, отличной от ignore_index.
        Если таких позиций нет (батч из одного паддинга) — 0, связанный с графом,
        а не NaN: среднее по пустому множеству испортило бы веса через backward.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    if not bool((shift_labels != ignore_index).any()):
        return shift_logits.sum() * 0.0

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
