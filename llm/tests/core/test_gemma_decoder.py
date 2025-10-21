import torch
import pytest
from llm.core.gemma_decoder import GemmaDecoder
from llm.core.rope import RoPE

@pytest.fixture
def gemma_decoder():
    rope = RoPE(head_size=4, max_seq_len=32)
    return GemmaDecoder(
        num_q_heads=4,
        emb_size=16,
        head_size=4,
        max_seq_len=32,
        rope=rope,
        dropout=0.1,
    )

def test_forward_shape(gemma_decoder):
    x = torch.randn(2, 12, 16)
    out, cache = gemma_decoder(x)
    assert out.shape == (2, 12, 16)
    assert isinstance(cache, tuple) or cache is None

def test_forward_masked(gemma_decoder):
    x = torch.randn(1, 8, 16)
    mask = torch.ones(1, 8, 8, dtype=torch.bool)
    out, _ = gemma_decoder(x, mask=mask)
    assert out.shape == x.shape

def test_forward_with_cache_flag(gemma_decoder):
    x = torch.randn(2, 7, 16)
    out, cache = gemma_decoder(x, use_cache=True, cache=None)
    assert out.shape == (2, 7, 16)

def test_forward_wrong_seq_len_raises(gemma_decoder):
    x = torch.randn(1, 100, 16)
    with pytest.raises(Exception):
        gemma_decoder(x)

def test_gradient_flow(gemma_decoder):
    x = torch.randn(3, 9, 16, requires_grad=True)
    y, _ = gemma_decoder(x)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape

def test_various_shapes(gemma_decoder):
    for b, s in [(1, 1), (2, 5), (2, 32)]:
        x = torch.randn(b, s, 16)
        y, _ = gemma_decoder(x)
        assert y.shape == (b, s, 16)

def test_forward_repeatability():
    torch.manual_seed(42)
    rope = RoPE(head_size=4, max_seq_len=32)
    decoder = GemmaDecoder(
        num_q_heads=4, emb_size=16, head_size=4, max_seq_len=32, rope=rope, dropout=0.0,
    )
    x = torch.randn(2, 8, 16)
    y1, _ = decoder(x)
    torch.manual_seed(42)
    decoder2 = GemmaDecoder(
        num_q_heads=4, emb_size=16, head_size=4, max_seq_len=32, rope=rope, dropout=0.0,
    )
    x2 = torch.randn(2, 8, 16)
    y2, _ = decoder2(x2)
    assert torch.allclose(y1, y2, atol=1e-5)
