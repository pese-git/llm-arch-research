import torch
import pytest
from llm.core.rope import RoPE

def test_rope_shapes_and_dtype():
    rope = RoPE(head_size=8, max_seq_len=32)
    x = torch.randn(2, 4, 16, 8)  # [batch, num_heads, seq_len, head_size]
    y = rope(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype

def test_rope_raises_on_bad_ndim():
    rope = RoPE(head_size=8, max_seq_len=16)
    x = torch.randn(2, 16, 8)  # [batch, seq_len, head_size] (3D)
    with pytest.raises(AssertionError):
        _ = rope(x)

def test_rope_preserves_norm():
    rope = RoPE(head_size=8, max_seq_len=16)
    x = torch.randn(2, 3, 7, 8)
    x_norm = x.norm(dim=-1)
    y = rope(x)
    y_norm = y.norm(dim=-1)
    # Нормы могут немного отличаться из-за float, сравниваем с допуском
    assert torch.allclose(x_norm, y_norm, rtol=1e-5, atol=1e-7)

def test_rope_backward_pass():
    rope = RoPE(head_size=8, max_seq_len=16)
    x = torch.randn(2, 2, 8, 8, requires_grad=True)
    out = rope(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape

@pytest.mark.parametrize("batch,num_heads,seq_len,head_size", [
    (1, 1, 4, 8),
    (2, 4, 16, 8),
    (3, 2, 7, 8),
])
def test_rope_various_shapes(batch, num_heads, seq_len, head_size):
    rope = RoPE(head_size=head_size, max_seq_len=32)
    x = torch.randn(batch, num_heads, seq_len, head_size)
    y = rope(x)
    assert y.shape == x.shape

def test_rope_start_pos():
    rope = RoPE(head_size=8, max_seq_len=32)
    x_full = torch.randn(1, 2, 8, 8)
    # Сравниваем участок результата для разных start_pos
    out1 = rope(x_full)
    out2 = rope(x_full, start_pos=2)
    assert not torch.allclose(out1, out2)
    # Для одинакового start_pos и x должны совпадать
    assert torch.allclose(rope(x_full, start_pos=1), rope(x_full, start_pos=1))
