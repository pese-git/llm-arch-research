"""
Tests for config validation in every model: head_size from the config, divisibility of
embed_dim by the number of heads, GQA head groups and MoE top-k.

A wrong config must fail in the constructor with a clear ValueError instead of silently
shrinking the attention space or crashing in the first forward.
"""

import pytest
import torch

from llm.core.config_checks import resolve_head_size
from llm.core.moe import MoE
from llm.models.gemma import Gemma
from llm.models.gpt import GPT, GPT2
from llm.models.llama import Llama
from llm.models.mistral import Mistral
from llm.models.mixtral import Mixtral

BASE_CONFIG = {
    "vocab_size": 50,
    "embed_dim": 32,
    "num_layers": 1,
    "max_position_embeddings": 16,
    "dropout": 0.0,
}

# model class, extra config, key of the number of (query) heads, uses RoPE
MODELS = {
    "gpt": (GPT, {"num_heads": 4}, "num_heads", False),
    "gpt2": (GPT2, {"num_heads": 4}, "num_heads", False),
    "llama": (Llama, {"num_heads": 4}, "num_heads", True),
    "mistral": (Mistral, {"num_q_heads": 4, "num_kv_heads": 2, "window_size": 4}, "num_q_heads", True),
    "mixtral": (
        Mixtral,
        {"num_q_heads": 4, "num_kv_heads": 2, "window_size": 4, "num_experts": 4, "top_k_experts": 2},
        "num_q_heads",
        True,
    ),
    "gemma": (Gemma, {"num_q_heads": 4}, "num_q_heads", True),
}
GQA_MODELS = ["mistral", "mixtral"]


def build(name, **overrides):
    model_class, extra, _, _ = MODELS[name]
    return model_class({**BASE_CONFIG, **extra, **overrides}).eval()


def query_size(model):
    return model._decoders[0]._heads._q.out_features


@pytest.mark.parametrize("name", list(MODELS))
def test_default_head_size_is_embed_dim_by_heads(name):
    model = build(name)
    assert query_size(model) == BASE_CONFIG["embed_dim"]


@pytest.mark.parametrize("name", list(MODELS))
def test_head_size_is_read_from_config(name):
    """num_heads * head_size may differ from embed_dim: attention projects back to embed_dim."""
    model = build(name, head_size=16)
    assert query_size(model) == 4 * 16

    tokens = torch.randint(0, BASE_CONFIG["vocab_size"], (2, 5))
    with torch.no_grad():
        logits, _ = model(tokens)
    assert logits.shape == (2, 5, BASE_CONFIG["vocab_size"])


@pytest.mark.parametrize("name", list(MODELS))
def test_indivisible_embed_dim_raises(name):
    heads_key = MODELS[name][2]
    with pytest.raises(ValueError, match="не делится"):
        build(name, embed_dim=30, **{heads_key: 4})


@pytest.mark.parametrize("name", list(MODELS))
def test_explicit_head_size_allows_indivisible_embed_dim(name):
    heads_key = MODELS[name][2]
    model = build(name, embed_dim=30, head_size=8, **{heads_key: 4})
    assert query_size(model) == 32


@pytest.mark.parametrize("name", [name for name, spec in MODELS.items() if spec[3]])
def test_odd_head_size_with_rope_raises(name):
    with pytest.raises(ValueError, match="чётным"):
        build(name, head_size=7)


@pytest.mark.parametrize("name", GQA_MODELS)
@pytest.mark.parametrize("num_kv_heads", [3, 8, 0])
def test_query_heads_must_divide_into_kv_groups(name, num_kv_heads):
    with pytest.raises(ValueError, match="num_kv_heads"):
        build(name, num_kv_heads=num_kv_heads)


@pytest.mark.parametrize("top_k_experts", [0, -1, 5])
def test_mixtral_rejects_bad_top_k_experts(top_k_experts):
    with pytest.raises(ValueError, match="top_k_experts"):
        build("mixtral", top_k_experts=top_k_experts)


def test_moe_rejects_no_experts():
    with pytest.raises(ValueError, match="num_experts"):
        MoE(emb_size=8, num_experts=0, top_k_experts=1)


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"embed_dim": 32, "num_heads": 4}, 8),
        ({"embed_dim": 32, "num_heads": 4, "head_size": 16}, 16),
        ({"embed_dim": 30, "num_heads": 4, "head_size": 7}, 7),
    ],
)
def test_resolve_head_size(config, expected):
    assert resolve_head_size(config, "num_heads") == expected


@pytest.mark.parametrize(
    "config, rope, message",
    [
        ({"embed_dim": 32, "num_heads": 0}, False, "num_heads"),
        ({"embed_dim": 32, "num_heads": 4, "head_size": 0}, False, "head_size"),
        ({"embed_dim": 30, "num_heads": 4}, False, "не делится"),
        ({"embed_dim": 28, "num_heads": 4}, True, "чётным"),
    ],
)
def test_resolve_head_size_errors(config, rope, message):
    with pytest.raises(ValueError, match=message):
        resolve_head_size(config, "num_heads", rope=rope)
