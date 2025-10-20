# llm/tests/models/test_gemma.py

import torch
import pytest
from llm.models.gemma.gemma import Gemma

@pytest.fixture
def config():
    return {
        "vocab_size": 100,
        "embed_dim": 32,
        "num_q_heads": 4,
        "num_layers": 2,
        "max_position_embeddings": 16,
        "dropout": 0.0,
    }

@pytest.fixture
def model(config):
    return Gemma(config)

def test_forward_basic(model):
    x = torch.randint(0, 100, (2, 8))
    logits, cache = model(x)
    assert logits.shape == (2, 8, 100)
    assert isinstance(cache, list)
    assert len(cache) == model._decoders.__len__()

def test_forward_with_cache(model):
    x = torch.randint(0, 100, (2, 4))
    logits, cache = model(x, use_cache=True)
    # Второй проход с cache и одним новым токеном
    x2 = torch.randint(0, 100, (2, 1))
    logits2, cache2 = model(x2, use_cache=True, cache=cache)
    assert logits2.shape == (2, 1, 100)
    assert isinstance(cache2, list)

def test_generate_and_shape(model):
    x = torch.randint(0, 100, (1, 5))
    result = model.generate(x, max_new_tokens=3, do_sample=False)
    assert result.shape == (1, 8)

def test_forward_sequence_too_long(model, config):
    x = torch.randint(0, 100, (1, config["max_position_embeddings"] + 1))
    with pytest.raises(ValueError):
        model(x)

def test_generate_with_sampling_topk(model):
    x = torch.randint(0, 100, (1, 3))
    out = model.generate(x, max_new_tokens=2, do_sample=True, top_k=5)
    assert out.shape == (1, 5)

def test_generate_with_sampling_topp(model):
    x = torch.randint(0, 100, (1, 3))
    out = model.generate(x, max_new_tokens=2, do_sample=True, top_p=0.8)
    assert out.shape == (1, 5)
