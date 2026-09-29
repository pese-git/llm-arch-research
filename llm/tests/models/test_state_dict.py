"""
Tests for what model checkpoints contain in every model.

Buffers computed from the config (causal/sliding-window masks, RoPE cos/sin tables) are
not persistent: they are not saved, checkpoints do not depend on max_seq_len through them,
and checkpoints saved before the change (with these buffers) still load with strict=True.
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
    "max_position_embeddings": 16,
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
ROPE_MODELS = ["llama", "mistral", "mixtral", "gemma"]

COMPUTED_BUFFERS = ("_tril_mask", "cos_matrix", "sin_matrix")


def build(name, **overrides):
    torch.manual_seed(0)
    model_class, extra = MODELS[name]
    return model_class({**BASE_CONFIG, **extra, **overrides}).eval()


def legacy_state_dict(model):
    """state_dict in the old format: every computed buffer saved under every module that owns it."""
    state = model.state_dict()
    state.update(model.named_buffers(remove_duplicate=False))
    return state


@pytest.fixture
def tokens():
    torch.manual_seed(1)
    return torch.randint(0, BASE_CONFIG["vocab_size"], (2, 10))


@pytest.mark.parametrize("name", list(MODELS))
def test_computed_buffers_are_not_saved(name):
    model = build(name)
    assert any(key.endswith(COMPUTED_BUFFERS) for key, _ in model.named_buffers())
    assert not [key for key in model.state_dict() if key.endswith(COMPUTED_BUFFERS)]


@pytest.mark.parametrize("name", list(MODELS))
def test_legacy_checkpoint_loads_strictly(name, tokens):
    source = build(name)
    legacy = legacy_state_dict(source)
    assert any(key.endswith(COMPUTED_BUFFERS) for key in legacy)

    torch.manual_seed(42)  # different initial weights, so equality comes from loading
    target = MODELS[name][0]({**BASE_CONFIG, **MODELS[name][1]}).eval()
    target.load_state_dict(legacy)

    with torch.no_grad():
        assert torch.equal(target(tokens)[0], source(tokens)[0])


@pytest.mark.parametrize("name", ROPE_MODELS)
def test_rope_checkpoint_loads_with_longer_context(name, tokens):
    """Without saved cos/sin tables and masks a RoPE checkpoint is not tied to max_seq_len."""
    source = build(name)
    longer = build(name, max_position_embeddings=64)
    longer.load_state_dict(legacy_state_dict(source))

    with torch.no_grad():
        assert torch.allclose(longer(tokens)[0], source(tokens)[0], atol=1e-6)
