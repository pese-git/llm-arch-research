import pytest
import torch.nn as nn
from llm.training.optimizer import get_optimizer

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

def test_get_optimizer_adamw():
    model = DummyModel()
    optimizer = get_optimizer(model, lr=1e-3, weight_decay=0.02, optimizer_type="adamw")
    assert optimizer.__class__.__name__ == 'AdamW'
    assert optimizer.defaults['lr'] == 1e-3
    assert optimizer.defaults['weight_decay'] == 0.02

def test_get_optimizer_adam():
    model = DummyModel()
    optimizer = get_optimizer(model, lr=1e-4, weight_decay=0.01, optimizer_type="adam")
    assert optimizer.__class__.__name__ == 'Adam'
    assert optimizer.defaults['lr'] == 1e-4
    assert optimizer.defaults['weight_decay'] == 0.01

def test_get_optimizer_sgd():
    model = DummyModel()
    optimizer = get_optimizer(model, lr=0.1, optimizer_type="sgd")
    assert optimizer.__class__.__name__ == 'SGD'
    assert optimizer.defaults['lr'] == 0.1
    # SGD: weight_decay по умолчанию 0 для этого вызова
    assert optimizer.defaults['momentum'] == 0.9

def test_get_optimizer_invalid():
    model = DummyModel()
    with pytest.raises(ValueError):
        get_optimizer(model, optimizer_type="nonexistent")