"""
causal_lm_loss — общая реализация loss для Trainer и llm.evaluation.
"""

import pytest
import torch
import torch.nn.functional as F

from llm.training.loss import causal_lm_loss


def test_shift_and_ignore_index_match_manual_cross_entropy():
    """Логит позиции t сравнивается с меткой t+1; метки -100 не входят в среднее."""
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    labels = torch.tensor([[3, 1, 4, -100, -100], [2, 6, 5, 0, 1]])
    expected = F.cross_entropy(
        torch.cat([logits[0, :2], logits[1, :4]]),
        torch.cat([labels[0, 1:3], labels[1, 1:5]]),
    )
    assert causal_lm_loss(logits, labels).item() == pytest.approx(expected.item())


def test_no_targets_gives_zero_with_graph():
    logits = torch.randn(1, 3, 4, requires_grad=True)
    labels = torch.full((1, 3), -100)
    loss = causal_lm_loss(logits, labels)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_custom_ignore_index():
    torch.manual_seed(1)
    logits = torch.randn(1, 4, 5)
    labels = torch.tensor([[1, 2, 9, 3]])
    expected = F.cross_entropy(logits[0, [0, 2]], labels[0, [1, 3]])
    assert causal_lm_loss(logits, labels, ignore_index=9).item() == pytest.approx(expected.item())


def test_uniform_logits_give_log_vocab():
    """Равномерные логиты — loss = ln V: отсюда perplexity V у модели, которая ничего не знает."""
    logits = torch.zeros(3, 6, 50)
    labels = torch.randint(0, 50, (3, 6))
    assert causal_lm_loss(logits, labels).item() == pytest.approx(torch.log(torch.tensor(50.0)).item())
