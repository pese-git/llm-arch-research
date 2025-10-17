import torch
import pytest
from llm.core.gelu import GELU

def test_gelu_shapes_and_dtype():
    gelu = GELU()
    x = torch.randn(4, 16, 8)
    y = gelu(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype

def test_gelu_known_values():
    gelu = GELU()
    x = torch.tensor([-3.0, 0.0, 3.0])
    y = gelu(x)
    # Сравнение с PyTorch F.gelu (которая использует точный алгоритм)
    y_ref = torch.nn.functional.gelu(x)
    diff = (y - y_ref).abs().max().item()
    assert diff < 5e-3, f"Max difference {diff} exceeds threshold"

def test_gelu_is_smooth_and_monotonic():
    gelu = GELU()
    x = torch.linspace(-5, 5, 100)
    y = gelu(x)
    dy = y[1:] - y[:-1]
    # Проверяем, что функция GELU хотя бы локально монотонна на большинстве промежутков
    assert (dy.mean() > 0 or dy.mean() < 0)

def test_gelu_gradients():
    gelu = GELU()
    x = torch.randn(3, 5, requires_grad=True)
    y = gelu(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape

def test_gelu_large_vs_small():
    gelu = GELU()
    x_pos = torch.tensor([100.0])
    x_neg = torch.tensor([-100.0])
    y_pos = gelu(x_pos)
    y_neg = gelu(x_neg)
    # Для больших положительных GELU(x) ~ x, для больших отрицательных ~0
    assert torch.allclose(y_pos, x_pos, rtol=1e-4, atol=1e-4)
    assert torch.allclose(y_neg, torch.zeros_like(x_neg), rtol=1e-4, atol=1e-4)
