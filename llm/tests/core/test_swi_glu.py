import torch
import pytest
from llm.core.swi_glu import SwiGLU

def test_swiglu_shape_and_dtype():
    swiglu = SwiGLU(emb_size=32, dropout=0.1)
    x = torch.randn(4, 10, 32)
    y = swiglu(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype

def test_swiglu_forward_range():
    swiglu = SwiGLU(emb_size=16, dropout=0.0)
    x = torch.randn(3, 7, 16)
    y = swiglu(x)
    assert y.abs().max() < 20

def test_swiglu_gradients():
    swiglu = SwiGLU(emb_size=8, dropout=0.0)
    x = torch.randn(2, 5, 8, requires_grad=True)
    out = swiglu(x)
    loss = out.pow(2).sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape

def test_swiglu_fp16():
    swiglu = SwiGLU(emb_size=16, dropout=0.0).half()
    x = torch.randn(1, 8, 16).half()
    y = swiglu(x)
    assert y.shape == x.shape
    assert y.dtype == torch.float16

def test_swiglu_reproducibility():
    swiglu = SwiGLU(emb_size=8, dropout=0.0)
    x = torch.ones(2, 4, 8)
    y1 = swiglu(x)
    y2 = swiglu(x)
    assert torch.allclose(y1, y2)
