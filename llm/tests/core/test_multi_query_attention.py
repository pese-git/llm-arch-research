import torch
import pytest
from llm.core.multi_query_attention import MultiQueryAttention
from llm.core.rope import RoPE

@pytest.fixture
def mqa_rope():
    return MultiQueryAttention(
        num_q_heads=4, emb_size=16, head_size=4, max_seq_len=32, rope=RoPE(head_size=4, max_seq_len=32), dropout=0.1
    )

@pytest.fixture
def mqa_no_rope():
    return MultiQueryAttention(
        num_q_heads=2, emb_size=8, head_size=4, max_seq_len=16, rope=None, dropout=0.0
    )

def test_forward_shape(mqa_rope):
    x = torch.randn(2, 10, 16)
    out, cache = mqa_rope(x)
    assert out.shape == (2, 10, 16)
    assert isinstance(cache, tuple) and len(cache) == 2

def test_forward_masked(mqa_rope):
    x = torch.randn(2, 8, 16)
    mask = torch.ones(2, 8, 8, dtype=torch.bool)
    out, cache = mqa_rope(x, mask=mask)
    assert out.shape == (2, 8, 16)

def test_forward_cache(mqa_rope):
    x = torch.randn(1, 4, 16)
    # Первый вызов — кэша нет
    out1, cache1 = mqa_rope(x)
    # Повторяем: подаем x второй раз — теперь добавим cache
    out2, cache2 = mqa_rope(x, use_cache=True, cache=cache1)
    assert out2.shape == (1, 4, 16)
    assert isinstance(cache2, tuple) and len(cache2) == 2
    # Проверка, что длина k_cache увеличилась
    assert cache2[0].shape[2] == cache1[0].shape[2] + x.shape[1]  # по длине seq

def test_forward_no_rope(mqa_no_rope):
    x = torch.randn(3, 6, 8)
    out, _ = mqa_no_rope(x)
    assert out.shape == (3, 6, 8)

def test_forward_different_batch_seq(mqa_rope):
    for batch, seq in [(1, 1), (2, 5), (3, 32)]:
        x = torch.randn(batch, seq, 16)
        out, _ = mqa_rope(x)
        assert out.shape == (batch, seq, 16)

def test_forward_raise_on_long_seq(mqa_rope):
    x = torch.randn(2, 40, 16)  # seq_len > max_seq_len
    with pytest.raises(ValueError):
        mqa_rope(x)

def test_forward_grad(mqa_rope):
    x = torch.randn(2, 7, 16, requires_grad=True)
    out, _ = mqa_rope(x)
    y = out.sum()
    y.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape

def test_dropout_applied():
    mqa = MultiQueryAttention(num_q_heads=2, emb_size=8, head_size=4, max_seq_len=12, rope=None, dropout=0.99)
    x = torch.ones(1, 3, 8)
    mqa.train()
    y, _ = mqa(x)
    # При очень большом dropout почти всё обнуляется
    assert (torch.abs(y) < 1e-5).float().mean() > 0.6 or y.sum() < 1e-2
