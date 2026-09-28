"""
Tests that incremental decoding with KV cache matches a full forward pass for every model.
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
    "max_position_embeddings": 64,
    "dropout": 0.0,
}

MODELS = {
    "gpt": (GPT, {"num_heads": 4}),
    "gpt2": (GPT2, {"num_heads": 4}),
    "llama": (Llama, {"num_heads": 4}),
    # window_size is smaller than the sequence so the rolling cache gets trimmed
    "mistral": (Mistral, {"num_q_heads": 4, "num_kv_heads": 2, "window_size": 4}),
    "mixtral": (
        Mixtral,
        {"num_q_heads": 4, "num_kv_heads": 2, "window_size": 4, "num_experts": 4, "top_k_experts": 2},
    ),
    "gemma": (Gemma, {"num_q_heads": 4}),
}

PROMPT_LEN = 3
SEQ_LEN = 12


@pytest.fixture(params=list(MODELS), ids=list(MODELS))
def model(request):
    torch.manual_seed(0)
    model_class, extra = MODELS[request.param]
    return model_class({**BASE_CONFIG, **extra}).eval()


def test_cached_logits_match_full_forward(model):
    """Logits for each new token decoded with cache must equal logits of a full pass."""
    tokens = torch.randint(0, BASE_CONFIG["vocab_size"], (2, SEQ_LEN))

    with torch.no_grad():
        full_logits, _ = model(tokens, use_cache=False)

        prefill_logits, cache = model(tokens[:, :PROMPT_LEN], use_cache=True)
        assert torch.allclose(prefill_logits, full_logits[:, :PROMPT_LEN], atol=1e-5)

        for pos in range(PROMPT_LEN, SEQ_LEN):
            step_logits, cache = model(tokens[:, pos : pos + 1], use_cache=True, cache=cache)
            assert torch.allclose(step_logits[:, -1], full_logits[:, pos], atol=1e-5), f"position {pos}"


def test_generate_with_cache_matches_without_cache(model):
    """Greedy generation must not depend on use_cache."""
    prompt = torch.randint(0, BASE_CONFIG["vocab_size"], (1, PROMPT_LEN))

    with torch.no_grad():
        cached = model.generate(prompt, max_new_tokens=SEQ_LEN, do_sample=False, use_cache=True)
        uncached = model.generate(prompt, max_new_tokens=SEQ_LEN, do_sample=False, use_cache=False)

    assert torch.equal(cached, uncached)
