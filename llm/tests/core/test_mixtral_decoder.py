import torch
import pytest
from llm.core.mixtral_decoder import MixtralDecoder
from llm.core.rope import RoPE

@pytest.fixture
def basic_decoder():
    emb_size = 16
    num_q_heads = 4
    num_kv_heads = 2
    head_size = 4
    max_seq_len = 32
    num_experts = 4
    top_k_experts = 2
    window_size = 8
    rope = RoPE(head_size=head_size, max_seq_len=max_seq_len)
    return MixtralDecoder(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        emb_size=emb_size,
        head_size=head_size,
        max_seq_len=max_seq_len,
        num_experts=num_experts,
        top_k_experts=top_k_experts,
        window_size=window_size,
        rope=rope,
        dropout=0.0,
    )

def test_forward_shape(basic_decoder):
    x = torch.randn(2, 10, 16)
    out, cache = basic_decoder(x)
    assert out.shape == (2, 10, 16)
    assert cache is None or isinstance(cache, (tuple, list))

def test_forward_masked(basic_decoder):
    x = torch.randn(3, 7, 16)
    mask = torch.ones(3, 7, 7, dtype=torch.bool)
    out, cache = basic_decoder(x, mask=mask)
    assert out.shape == (3, 7, 16)

def test_forward_with_cache_flag(basic_decoder):
    x = torch.randn(2, 8, 16)
    out, cache = basic_decoder(x, use_cache=True, cache=None)
    assert out.shape == (2, 8, 16)
    assert isinstance(cache, (tuple, list)) or cache is None

def test_backprop_pass(basic_decoder):
    x = torch.randn(2, 5, 16, requires_grad=True)
    out, _ = basic_decoder(x)
    y = out.sum()
    y.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape

def test_seq_too_long_raises(basic_decoder):
    x = torch.randn(1, 40, 16)  # seq_len > max_seq_len
    with pytest.raises(Exception):
        basic_decoder(x)

def test_different_config():
    rope = RoPE(head_size=2, max_seq_len=12)
    decoder = MixtralDecoder(
        num_q_heads=2, num_kv_heads=2, emb_size=4, head_size=2,
        max_seq_len=12, num_experts=2, top_k_experts=1, window_size=4, rope=rope, dropout=0.1
    )
    x = torch.randn(1, 8, 4)
    out, cache = decoder(x)
    assert out.shape == x.shape

def test_forward_no_dropout():
    # Проверка на корректность shape при отсутствии Dropout
    rope = RoPE(head_size=2, max_seq_len=12)
    decoder = MixtralDecoder(
        num_q_heads=2, num_kv_heads=1, emb_size=4, head_size=2,
        max_seq_len=12, num_experts=2, top_k_experts=1, window_size=3, rope=rope, dropout=0.0
    )
    x = torch.randn(2, 3, 4)
    out, cache = decoder(x)
    assert out.shape == x.shape
