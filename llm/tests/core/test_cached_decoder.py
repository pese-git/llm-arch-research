import torch
import pytest
from llm.core.cached_decoder import CachedDecoder
from llm.core.feed_forward import FeedForward

@pytest.fixture
def decoder_config():
    return dict(
        num_heads=4,
        emb_size=32,
        head_size=8,
        feed_forward_layer=FeedForward(emb_size=32, dropout=0.1, activation="gelu"),
        max_seq_len=64,
        dropout=0.1
    )

def test_cached_decoder_init(decoder_config):
    model = CachedDecoder(**decoder_config)
    assert model is not None
    # Main attention block is usually stored as _heads or _attention (which itself includes _q _k _v)
    assert hasattr(model, '_heads') or hasattr(model, '_attention')
    assert hasattr(model, '_ff') or hasattr(model, 'feed_forward_layer')

def test_cached_decoder_forward_shape(decoder_config):
    model = CachedDecoder(**decoder_config)
    batch, seq_len, emb_size = 3, 10, decoder_config['emb_size']
    x = torch.randn(batch, seq_len, emb_size)
    output, cache = model(x, use_cache=True)
    assert output.shape == (batch, seq_len, emb_size)
    assert cache is not None

def test_cached_decoder_forward_no_cache(decoder_config):
    model = CachedDecoder(**decoder_config)
    batch, seq_len, emb_size = 2, 12, decoder_config['emb_size']
    x = torch.randn(batch, seq_len, emb_size)
    output, cache = model(x, use_cache=False)
    assert output.shape == (batch, seq_len, emb_size)
    assert cache is None

def test_cached_decoder_error_on_long_seq(decoder_config):
    model = CachedDecoder(**decoder_config)
    batch, seq_len, emb_size = 1, decoder_config['max_seq_len'] + 1, decoder_config['emb_size']
    x = torch.randn(batch, seq_len, emb_size)
    with pytest.raises(ValueError):
        model(x)

def test_cached_decoder_backward(decoder_config):
    model = CachedDecoder(**decoder_config)
    batch, seq_len, emb_size = 2, 7, decoder_config['emb_size']
    x = torch.randn(batch, seq_len, emb_size, requires_grad=True)
    output, cache = model(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None

def test_cached_decoder_kv_cache_chain(decoder_config):
    model = CachedDecoder(**decoder_config)
    batch, seq_len, emb_size = 1, 4, decoder_config['emb_size']
    x = torch.randn(batch, seq_len, emb_size)
    # Первый проход — кэша нет
    _, cache = model(x, use_cache=True)
    # Второй проход — передаём кэш, добавляем еще токен:
    next_x = torch.randn(batch, 1, emb_size)
    _, cache2 = model(next_x, use_cache=True, cache=cache)
    assert cache2 is not None
