# llm/tests/core/test_group_query_attention.py

import torch
import pytest
from llm.core.group_query_attention import GroupedQueryAttention
from llm.core.rope import RoPE

@pytest.fixture
def params():
    return {
        'num_q_heads': 4,
        'num_kv_heads': 2,
        'emb_size': 16,
        'head_size': 4,
        'max_seq_len': 32,
        'window_size': 8,
        'dropout': 0.0
    }

def test_initialization(params):
    attn = GroupedQueryAttention(**params)
    assert isinstance(attn, GroupedQueryAttention)

def test_forward_shape(params):
    batch, seq = 2, 10
    x = torch.randn(batch, seq, params['emb_size'])
    attn = GroupedQueryAttention(**params)
    y, cache = attn(x)
    assert y.shape == (batch, seq, params['emb_size'])
    assert cache is not None
    assert isinstance(y, torch.Tensor)

def test_forward_shape_with_mask(params):
    batch, seq = 2, 10
    x = torch.randn(batch, seq, params['emb_size'])
    mask = torch.tril(torch.ones(seq, seq)).bool()
    attn = GroupedQueryAttention(**params)
    y, _ = attn(x, mask=mask)
    assert y.shape == (batch, seq, params['emb_size'])

def test_kv_repetition(params):
    batch, seq = 1, 3
    attn = GroupedQueryAttention(**params)
    kv = torch.randn(batch, params['num_kv_heads'], seq, params['head_size'])
    rep = attn._repeat_kv_heads(kv, params['num_q_heads'], params['num_kv_heads'])
    assert rep.shape == (batch, params['num_q_heads'], seq, params['head_size'])

def test_window_mask(params):
    attn = GroupedQueryAttention(**params)
    mask = attn._create_sliding_window_mask(8, 3)
    assert mask.shape == (8, 8)
    # Проверим булеву маску окна в позиции 4
    expected = torch.tensor([True, True, True, True, False, False])
    assert torch.equal(mask[4, 1:7], expected)

def test_forward_with_rope(params):
    batch, seq = 2, 12
    x = torch.randn(batch, seq, params['emb_size'])
    rope = RoPE(head_size=params['head_size'], max_seq_len=params['max_seq_len'])
    params2 = params.copy()
    params2['rope'] = rope
    attn = GroupedQueryAttention(**params2)
    y, _ = attn(x)
    assert y.shape == (batch, seq, params['emb_size'])

def test_cache_usage(params):
    batch, seq = 1, 5
    x = torch.randn(batch, seq, params['emb_size'])
    attn = GroupedQueryAttention(**params)
    # Первый проход - получаем кэш
    _, cache = attn(x)
    # Второй проход с кэшем (имитируем автокомплит seq_len=1)
    x2 = torch.randn(batch, 1, params['emb_size'])
    y2, cache2 = attn(x2, cache=cache)
    assert cache2 is not None
    assert y2.shape == (batch, 1, params['emb_size'])

def test_gradient_backward(params):
    batch, seq = 2, 6
    x = torch.randn(batch, seq, params['emb_size'], requires_grad=True)
    attn = GroupedQueryAttention(**params)
    y, _ = attn(x)
    y.sum().backward()
    for param in attn.parameters():
        assert param.grad is not None
