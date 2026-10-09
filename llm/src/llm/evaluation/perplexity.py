r"""
Перплексия — стандартная метрика качества языковой модели.

Перплексия — экспонента среднего loss следующего токена:

    PPL = \exp\left(-\frac{1}{N} \sum_{t=1}^{N} \log p(w_{t+1} \mid w_{\le t})\right).

Её можно читать как «среди скольких равновероятных токенов модель выбирает»:
модель, которая ничего не знает, даёт PPL = V (размер словаря), идеальная — 1.
Loss считается той же функцией, что и в обучении (`causal_lm_loss`), поэтому
`perplexity(...) == exp(Trainer.evaluate())` на одном наборе: среднее берётся по
батчам, как в `Trainer.evaluate`. Для TokenBlockDataset все батчи одинаковой
длины, и это совпадает со средним по токенам.
"""

import math
from typing import Optional

import torch

from llm.training.loss import causal_lm_loss


def lm_loss(
    model: torch.nn.Module,
    loader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> float:
    """
    Средний loss следующего токена по батчам loader без градиентов.

    Args:
        model: модель с `forward(input_ids, attention_mask=...) -> (logits, cache)`
            или возвращающая логиты тензором.
        loader: итерируемое по словарям с `input_ids`, `labels` и, если есть паддинг,
            `attention_mask` — например, DataLoader над датасетом llm.datasets.
        device: куда класть батчи; None — устройство параметров модели.
        max_batches: ограничить число батчей (быстрая оценка на части набора).

    Returns:
        float — средний loss по пройденным батчам. Модель остаётся в режиме eval.

    Raises:
        ValueError: loader пуст.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if max_batches is not None and count >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                outputs = model(input_ids, attention_mask=attention_mask.to(device))
            else:
                outputs = model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            total += causal_lm_loss(logits, labels).item()
            count += 1
    if count == 0:
        raise ValueError("loader не дал ни одного батча: не на чем считать loss")
    return total / count


def perplexity(
    model: torch.nn.Module,
    loader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> float:
    """
    Перплексия модели на loader: `exp` от `lm_loss` с теми же аргументами.
    """
    return math.exp(lm_loss(model, loader, device=device, max_batches=max_batches))
