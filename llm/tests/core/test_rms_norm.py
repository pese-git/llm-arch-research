import torch
import pytest
from llm.core.rms_norm import RMSNorm

def test_rmsnorm_shape_preservation():
    norm = RMSNorm(64)
    x = torch.randn(3, 5, 64)
    y = norm(x)
    assert y.shape == x.shape

def test_rmsnorm_dtype_and_device():
    norm = RMSNorm(32)
    x = torch.randn(8, 32, device='cpu', dtype=torch.float64)
    y = norm(x)
    assert y.dtype == torch.float64
    assert y.device == x.device

def test_rmsnorm_mean_no_shift():
    norm = RMSNorm(32)
    x = torch.randn(3, 128, 32)
    y = norm(x)
    rms = torch.sqrt((y ** 2).mean(dim=-1))
    w_mean = norm._w.mean().item()
    assert torch.allclose(rms.mean(), torch.tensor(w_mean), rtol=0.2, atol=0.2)

def test_rmsnorm_backward():
    norm = RMSNorm(16)
    x = torch.randn(2, 15, 16, requires_grad=True)
    y = norm(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert norm._w.grad is not None

def test_rmsnorm_fp16():
    norm = RMSNorm(8).half()
    x = torch.randn(2, 6, 8).half()
    y = norm(x)
    assert y.shape == x.shape
    assert y.dtype == torch.float16

def test_rmsnorm_large_eps_stability():
    norm = RMSNorm(16, eps=1)
    x = torch.zeros(2, 5, 16)
    y = norm(x)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
