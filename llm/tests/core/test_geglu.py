import torch
import pytest
from llm.core.geglu import GeGLU

@pytest.fixture
def geglu():
    return GeGLU(emb_size=16, dropout=0.1)

def test_forward_shape(geglu):
    x = torch.randn(2, 5, 16)
    y = geglu(x)
    assert y.shape == x.shape

def test_forward_no_batch(geglu):
    x = torch.randn(1, 16)
    y = geglu(x.unsqueeze(0))
    assert y.shape == (1, 1, 16)

@pytest.mark.skip(reason="float16 not supported without parameter casting")
def test_forward_dtype_fp16():
    geglu = GeGLU(emb_size=8, dropout=0.0)
    x = torch.randn(2, 4, 8).half()
    y = geglu(x)
    assert y.shape == x.shape
    assert y.dtype == torch.float16

def test_forward_no_dropout():
    geglu = GeGLU(emb_size=4, dropout=0.0)
    x = torch.randn(3, 2, 4)
    y = geglu(x)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()

def test_gradient_flow(geglu):
    x = torch.randn(3, 8, 16, requires_grad=True)
    y = geglu(x)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape

def test_forward_repeatability():
    torch.manual_seed(42)
    geglu = GeGLU(emb_size=8, dropout=0.0)
    x = torch.randn(3, 2, 8)
    y1 = geglu(x)
    torch.manual_seed(42)
    geglu2 = GeGLU(emb_size=8, dropout=0.0)
    x2 = torch.randn(3, 2, 8)
    y2 = geglu2(x2)
    assert torch.allclose(y1, y2, atol=1e-5)

def test_edge_small_large():
    geglu = GeGLU(emb_size=2, dropout=0.0)
    x = torch.randn(2, 2, 2)
    y = geglu(x)
    assert y.shape == x.shape
    geglu = GeGLU(emb_size=256, dropout=0.0)
    x = torch.randn(1, 1, 256)
    y = geglu(x)
    assert y.shape == x.shape
