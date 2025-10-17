import torch
import pytest
from llm.core.mistral_decoder import MistralDecoder
from llm.core.rope import RoPE

@pytest.fixture
def decoder_config():
    # Current MistralDecoder is a single block (not a stack).
    return dict(
        num_q_heads=4,
        num_kv_heads=2,
        emb_size=32,
        head_size=8,
        max_seq_len=128,
        window_size=16,
        rope=RoPE(head_size=8, max_seq_len=128),
        dropout=0.0
    )

def test_mistral_decoder_init(decoder_config):
    model = MistralDecoder(**decoder_config)
    assert model is not None

def test_mistral_decoder_forward_shapes(decoder_config):
    model = MistralDecoder(**decoder_config)
    batch, seq_len, emb_size = 2, 10, decoder_config['emb_size']
    x = torch.randn(batch, seq_len, emb_size)
    output, cache = model(x, use_cache=True)
    assert output.shape == (batch, seq_len, emb_size)
    assert cache is not None

def test_mistral_decoder_forward_no_cache(decoder_config):
    model = MistralDecoder(**decoder_config)
    batch, seq_len, emb_size = 2, 10, decoder_config['emb_size']
    x = torch.randn(batch, seq_len, emb_size)
    output, cache = model(x, use_cache=False)
    assert output.shape == (batch, seq_len, emb_size)
    assert cache is None

def test_mistral_decoder_cache_shapes(decoder_config):
    model = MistralDecoder(**decoder_config)
    batch, seq_len, emb_size = 2, 8, decoder_config['emb_size']
    x = torch.randn(batch, seq_len, emb_size)
    # Первый проход — без кэша
    _, cache = model(x, use_cache=True)
    # Второй проход — заполняем кэш
    x_next = torch.randn(batch, 1, emb_size)
    _, cache2 = model(x_next, use_cache=True, cache=cache)
    # Можно проверить, что кэш не None и корректной структуры:
    assert cache2 is not None

def test_mistral_decoder_shape_error(decoder_config):
    model = MistralDecoder(**decoder_config)
    batch, seq_len, emb_size = 2, decoder_config['max_seq_len'] + 1, decoder_config['emb_size']
    x = torch.randn(batch, seq_len, emb_size)
    with pytest.raises(ValueError):
        model(x)

def test_mistral_decoder_backward(decoder_config):
    model = MistralDecoder(**decoder_config)
    batch, seq_len, emb_size = 2, 10, decoder_config['emb_size']
    x = torch.randn(batch, seq_len, emb_size, requires_grad=True)
    output, _ = model(x, use_cache=False)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
