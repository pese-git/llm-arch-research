"""
Паддинг в батче: маска ключей и позиции токенов из attention_mask.

Последовательности разной длины в одном батче дополняются pad-токенами. Чтобы результат
настоящих токенов не зависел от паддинга, нужны две вещи:

1. **Маска ключей.** Настоящие токены не смотрят на pad-токены. Сама causal-маска этого не
   даёт при левом паддинге: pad-токены стоят раньше настоящих.
2. **Позиции.** Позиция токена — его номер среди настоящих токенов строки, а не номер
   столбца: у строки `[pad, pad, a, b]` токен `a` — на позиции 0. Иначе позиционные
   эмбеддинги (GPT) и RoPE сдвинутся на число pad-токенов слева.

Позиции считаются как в HuggingFace: `cumsum(attention_mask) − 1`. Causal-маска и скользящее
окно по-прежнему строятся по столбцам (слотам) последовательности и кэша.

pad-токен как запрос видит только себя: иначе у pad-токена в начале строки не было бы ни
одного допустимого ключа, softmax дал бы NaN, а NaN через нулевой вес (0 · NaN) попал бы в
выход настоящих токенов. Выход pad-позиций смысла не имеет и дальше не используется.
"""

from typing import NamedTuple, Optional

import torch


class Padding(NamedTuple):
    """
    Паддинг текущего прохода.

    key_mask: bool [batch, start_pos + seq_len] — True для настоящих токенов, по всем слотам:
        кэш и новые токены.
    positions: long [batch, seq_len] — позиции новых токенов для позиционных эмбеддингов и RoPE.
    """

    key_mask: torch.Tensor
    positions: torch.Tensor

    def apply(self, allowed: torch.Tensor, start_pos: int, key_start: int) -> torch.Tensor:
        """
        Добавляет маску ключей к маске causal/окна.

        Args:
            allowed: bool [seq_len, num_keys] — маска causal (и окна) для новых токенов
                (слоты start_pos …) и ключей (слоты key_start … start_pos + seq_len − 1).
            start_pos: слот первого нового токена.
            key_start: слот первого ключа (0 или начало обрезанного по окну кэша).

        Returns:
            bool [batch, 1, seq_len, num_keys] — для умножения на все головы.
        """
        seq_len, num_keys = allowed.shape
        keys = self.key_mask[:, key_start:key_start + num_keys]  # [B, num_keys]
        query_slots = torch.arange(start_pos, start_pos + seq_len, device=allowed.device)
        key_slots = torch.arange(key_start, key_start + num_keys, device=allowed.device)
        itself = query_slots.unsqueeze(1) == key_slots.unsqueeze(0)  # [seq_len, num_keys]
        return (allowed & (keys.unsqueeze(1) | itself)).unsqueeze(1)


def padding_from_attention_mask(
    attention_mask: Optional[torch.Tensor], x: torch.Tensor, start_pos: int = 0
) -> Optional[Padding]:
    """
    Проверяет attention_mask и строит по ней Padding.

    Args:
        attention_mask: без кэша — [batch, seq_len], с кэшем — [batch, start_pos + seq_len]:
            1 для настоящих токенов, 0 для паддинга (как в HuggingFace). С кэшем маска всегда
            покрывает и кэш: по одним новым токенам нельзя узнать, был ли паддинг в
            закэшированных, а незамаскированный паддинг кэша молча исказил бы результат.
        x: входные токены [batch, seq_len].
        start_pos: число закэшированных позиций (слот первого нового токена).

    Returns:
        None, если маски нет или в ней одни единицы: тогда модели идут прежним путём, без
        маски ключей и с позициями start_pos, start_pos + 1, … Иначе — Padding.

    Raises:
        ValueError: Если форма маски не [batch, seq_len] без кэша или не
            [batch, start_pos + seq_len] с кэшем.
    """
    if attention_mask is None:
        return None
    batch_size, seq_len = x.shape
    total_len = start_pos + seq_len
    if attention_mask.dim() != 2 or attention_mask.shape != (batch_size, total_len):
        expected = (
            f"[batch, seq_len] = [{batch_size}, {seq_len}]"
            if start_pos == 0
            else f"с кэшем — [batch, cache_len + seq_len] = [{batch_size}, {total_len}]: "
            "маска покрывает и закэшированные токены, иначе паддинг в кэше не замаскировать"
        )
        raise ValueError(
            f"attention_mask должна иметь форму {expected}, получено {list(attention_mask.shape)}"
        )
    key_mask = attention_mask != 0
    if bool(key_mask.all()):
        return None
    positions = (key_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
    return Padding(key_mask=key_mask, positions=positions[:, start_pos:])
