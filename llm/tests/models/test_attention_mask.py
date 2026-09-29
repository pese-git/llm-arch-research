"""
Tests for attention_mask handling in every model.

Models apply only the causal (and sliding-window) mask. That is enough for right padding,
so such masks are accepted; masks that would need key masking and shifted positions
(left padding, padding in generate) are rejected instead of being silently ignored.
"""

import pytest
import torch

from llm.models.gemma import Gemma
from llm.models.gpt import GPT, GPT2
from llm.models.llama import Llama
from llm.models.mistral import Mistral
from llm.models.mixtral import Mixtral

BASE_CONFIG = {
    "vocab_size": 50,
    "embed_dim": 32,
    "num_layers": 2,
    "max_position_embeddings": 32,
    "dropout": 0.0,
}

MODELS = {
    "gpt": (GPT, {"num_heads": 4}),
    "gpt2": (GPT2, {"num_heads": 4}),
    "llama": (Llama, {"num_heads": 4}),
    "mistral": (Mistral, {"num_q_heads": 4, "num_kv_heads": 2, "window_size": 4}),
    "mixtral": (
        Mixtral,
        {"num_q_heads": 4, "num_kv_heads": 2, "window_size": 4, "num_experts": 4, "top_k_experts": 2},
    ),
    "gemma": (Gemma, {"num_q_heads": 4}),
}

REAL_LEN = 6
PAD_LEN = 3


@pytest.fixture(params=list(MODELS), ids=list(MODELS))
def model(request):
    torch.manual_seed(0)
    model_class, extra = MODELS[request.param]
    return model_class({**BASE_CONFIG, **extra}).eval()


@pytest.fixture
def tokens():
    return torch.randint(1, BASE_CONFIG["vocab_size"], (2, REAL_LEN))


def test_all_ones_mask_matches_no_mask(model, tokens):
    with torch.no_grad():
        expected, _ = model(tokens)
        actual, _ = model(tokens, attention_mask=torch.ones_like(tokens))
    assert torch.equal(actual, expected)


def test_right_padding_keeps_real_token_logits(model, tokens):
    """Pad tokens after the real ones are hidden by the causal mask anyway."""
    padded = torch.cat([tokens, torch.zeros(2, PAD_LEN, dtype=torch.long)], dim=1)
    mask = torch.cat([torch.ones_like(tokens), torch.zeros(2, PAD_LEN, dtype=torch.long)], dim=1)
    mask[1, REAL_LEN - 2 :] = 0  # rows may have different real lengths

    with torch.no_grad():
        expected, _ = model(tokens)
        actual, _ = model(padded, attention_mask=mask)

    assert torch.allclose(actual[0, :REAL_LEN], expected[0], atol=1e-5)
    assert torch.allclose(actual[1, : REAL_LEN - 2], expected[1, : REAL_LEN - 2], atol=1e-5)


@pytest.mark.parametrize(
    "zeros",
    [slice(0, 2), slice(2, 3)],
    ids=["left_padding", "gap"],
)
def test_unsupported_padding_raises(model, tokens, zeros):
    mask = torch.ones_like(tokens)
    mask[0, zeros] = 0
    with pytest.raises(NotImplementedError, match="правый паддинг"):
        model(tokens, attention_mask=mask)


def test_mask_with_zeros_and_cache_raises(model, tokens):
    with torch.no_grad():
        _, cache = model(tokens[:, :4])
        # Mask over the new tokens only and over cache + new tokens are both accepted when all ones
        model(tokens[:, 4:], cache=cache, attention_mask=torch.ones(2, REAL_LEN - 4))
        model(tokens[:, 4:], cache=cache, attention_mask=torch.ones(2, REAL_LEN))

        mask = torch.ones(2, REAL_LEN)
        mask[:, -1] = 0
        with pytest.raises(NotImplementedError, match="кэш"):
            model(tokens[:, 4:], cache=cache, attention_mask=mask)


@pytest.mark.parametrize("shape", [(2,), (1, REAL_LEN), (2, REAL_LEN + 1)], ids=str)
def test_wrong_mask_shape_raises(model, tokens, shape):
    with pytest.raises(ValueError, match="attention_mask"):
        model(tokens, attention_mask=torch.ones(shape))


def test_generate_accepts_all_ones_mask(model, tokens):
    with torch.no_grad():
        expected = model.generate(tokens, max_new_tokens=3, do_sample=False)
        actual = model.generate(
            tokens, max_new_tokens=3, do_sample=False, attention_mask=torch.ones_like(tokens)
        )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("zeros", [slice(0, 2), slice(REAL_LEN - 2, REAL_LEN)], ids=["left", "right"])
def test_generate_rejects_padding(model, tokens, zeros):
    """Generation after padding would continue from pad tokens, so any zero is rejected."""
    mask = torch.ones_like(tokens)
    mask[1, zeros] = 0
    with pytest.raises(NotImplementedError, match="generate"):
        model.generate(tokens, max_new_tokens=3, do_sample=False, attention_mask=mask)
