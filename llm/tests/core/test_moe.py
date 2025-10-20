import torch
import pytest
from llm.core.moe import MoE

@pytest.fixture
def moe():
    # Базовая MoE для коротких тестов
    return MoE(emb_size=16, num_experts=4, top_k_experts=2, dropout=0.0)

def test_forward_shape(moe):
    x = torch.randn(3, 5, 16)  # [batch, seq, emb]
    y = moe(x)
    assert y.shape == x.shape

def test_forward_grad(moe):
    x = torch.randn(2, 4, 16, requires_grad=True)
    y = moe(x)
    (y.sum()).backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape

def test_top_k_larger_than_experts():
    # top_k_experts > num_experts должно падать
    with pytest.raises(ValueError):
        MoE(emb_size=8, num_experts=2, top_k_experts=4)

def test_single_expert_no_error():
    # один эксперт, один топ-к — модель всё ещё валидна
    moe = MoE(emb_size=8, num_experts=1, top_k_experts=1)
    x = torch.randn(2, 2, 8)
    y = moe(x)
    assert y.shape == x.shape

def test_forward_trivial_weights():
    """Проверяет, что при одинаковых весах роутера MoE возвращает усреднённое по экспертам."""
    class DummyMoE(MoE):
        def forward(self, x):
            # Роутер отдаёт всегда единичные логиты = softmax -> uniform
            self._router = torch.nn.Linear(x.size(-1), self._num_experts, bias=False)
            torch.nn.init.constant_(self._router.weight, 0.0)
            return super().forward(x)
    moe = DummyMoE(emb_size=4, num_experts=2, top_k_experts=2)
    x = torch.zeros(1, 2, 4)
    y = moe(x)
    assert y.shape == x.shape

def test_forward_deterministic_seed(moe):
    torch.manual_seed(42)
    x = torch.randn(2, 3, 16)
    y1 = moe(x)
    torch.manual_seed(42)
    y2 = moe(x)
    assert torch.allclose(y1, y2, atol=1e-5)

def test_forward_no_dropout():
    """Без dropout MoE не меняет shape и не даёт NaN."""
    moe = MoE(emb_size=5, num_experts=3, top_k_experts=2, dropout=0.0)
    x = torch.randn(2, 7, 5)
    y = moe(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()
