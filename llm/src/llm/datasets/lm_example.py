"""
Сборка обучающего примера языковой модели фиксированной длины.

Короткая последовательность дополняется справа до block_size. pad-позиции помечаются
по месту, а не по значению токена: pad_token_id может совпадать с настоящим токеном
(0 по умолчанию, pad = eos у GPT-2), и сравнение `input_ids == pad_token_id` задело бы
настоящие токены.
"""

from typing import Dict, List

import torch

# ignore_index в F.cross_entropy (Trainer.compute_lm_loss) и в HuggingFace
IGNORE_INDEX = -100


def lm_example(token_ids: List[int], block_size: int, pad_token_id: int) -> Dict[str, torch.Tensor]:
    """
    Обрезает или дополняет токены до block_size и строит маску и метки.

    Args:
        token_ids: токены примера.
        block_size: длина примера.
        pad_token_id: чем дополнять input_ids.

    Returns:
        dict с long-тензорами формы [block_size]:
            - input_ids: токены, дополненные pad_token_id;
            - attention_mask: 1 для настоящих токенов, 0 для паддинга;
            - labels: копия токенов, на pad-позициях -100 — они не входят в loss.
    """
    token_ids = list(token_ids[:block_size])
    num_pad = block_size - len(token_ids)
    return {
        "input_ids": torch.tensor(token_ids + [pad_token_id] * num_pad, dtype=torch.long),
        "attention_mask": torch.tensor([1] * len(token_ids) + [0] * num_pad, dtype=torch.long),
        "labels": torch.tensor(token_ids + [IGNORE_INDEX] * num_pad, dtype=torch.long),
    }
