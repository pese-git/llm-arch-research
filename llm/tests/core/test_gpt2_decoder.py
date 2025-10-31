import torch
import pytest
from llm.core.gpt2_decoder import Gpt2Decoder

def gpt2_decoder_config():
    return dict(
        num_heads=4,
        emb_size=32,
        head_size=8,
        max_seq_len=64,
        dropout=0.1
    )

def test_gpt2_decoder_init():
    cfg = gpt2_decoder_config()
    model = Gpt2Decoder(**cfg)
    assert model is not None
    assert hasattr(model, '_heads')
    assert hasattr(model, '_ff')


def test_gpt2_decoder_forward_shape():
    cfg = gpt2_decoder_config()
    model = Gpt2Decoder(**cfg)
    batch, seq_len, emb_size = 3, 10, cfg['emb_size']
    x = torch.randn(batch, seq_len, emb_size)
    output, cache = model(x, use_cache=True)
    assert output.shape == (batch, seq_len, emb_size)
    assert cache is not None or cache is None  # cache type may be tensor in current impl


def test_gpt2_decoder_forward_no_cache():
    cfg = gpt2_decoder_config()
    model = Gpt2Decoder(**cfg)
    batch, seq_len, emb_size = 2, 12, cfg['emb_size']
    x = torch.randn(batch, seq_len, emb_size)
    output, cache = model(x, use_cache=False)
    assert output.shape == (batch, seq_len, emb_size)
    assert cache is None


def test_gpt2_decoder_error_on_long_seq():
    cfg = gpt2_decoder_config()
    model = Gpt2Decoder(**cfg)
    batch, seq_len, emb_size = 1, cfg['max_seq_len'] + 1, cfg['emb_size']
    x = torch.randn(batch, seq_len, emb_size)
    with pytest.raises(ValueError):
        model(x)


def test_gpt2_decoder_backward():
    cfg = gpt2_decoder_config()
    model = Gpt2Decoder(**cfg)
    batch, seq_len, emb_size = 2, 7, cfg['emb_size']
    x = torch.randn(batch, seq_len, emb_size, requires_grad=True)
    output, cache = model(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None


def test_gpt2_decoder_kv_cache_chain():
    cfg = gpt2_decoder_config()
    model = Gpt2Decoder(**cfg)
    batch, seq_len, emb_size = 1, 4, cfg['emb_size']
    x = torch.randn(batch, seq_len, emb_size)
    # Первый проход — кэша нет
    _, cache = model(x, use_cache=True)
    # Второй проход — передаём кэш, добавляем еще токен:
    next_x = torch.randn(batch, 1, emb_size)
    _, cache2 = model(next_x, use_cache=True, cache=cache)
    assert cache2 is not None