import torch
import pytest
from llm.core.silu import SiLU

def test_silu_shape_and_dtype():
    silu = SiLU()
    x = torch.randn(3, 10, 8)
    y = silu(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype

def test_silu_known_values():
    silu = SiLU()
    x = torch.tensor([-2.0, 0.0, 2.0])
    y = silu(x)
    # PyTorch эталон
    y_ref = torch.nn.functional.silu(x)
    assert torch.allclose(y, y_ref, atol=1e-6)

def test_silu_large_vs_small():
    silu = SiLU()
    x_pos = torch.tensor([100.0])
    x_neg = torch.tensor([-100.0])
    y_pos = silu(x_pos)
    y_neg = silu(x_neg)
    assert torch.allclose(y_pos, x_pos, rtol=1e-4, atol=1e-4)   # SiLU(x) ~ x для больших x>0
    assert torch.allclose(y_neg, torch.zeros_like(x_neg), rtol=1e-4, atol=1e-4) # SiLU(x) ~ 0 для x<0

def test_silu_gradients():
    silu = SiLU()
    x = torch.randn(4, 4, requires_grad=True)
    y = silu(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape

def test_silu_broadcast():
    silu = SiLU()
    x = torch.randn(3, 1, 16)
    y = silu(x)
    assert y.shape == x.shape
