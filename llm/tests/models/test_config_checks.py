"""
Tests for config validation in every model: head_size from the config, divisibility of
embed_dim by the number of heads, GQA head groups, MoE top-k, rms_norm_eps, rope_theta and attention_dropout.

A wrong config must fail in the constructor with a clear ValueError instead of silently
shrinking the attention space or crashing in the first forward.
"""

import pytest
import torch

from llm.core.config_checks import resolve_head_size
from llm.core.moe import MoE
from llm.core.rms_norm import RMSNorm
from llm.core.rope import RoPE
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


RMS_NORM_MODELS = [name for name, spec in MODELS.items() if spec[3]]  # все модели с RoPE используют RMSNorm


def rms_norm_eps_values(model):
    return {module._eps for module in model.modules() if isinstance(module, RMSNorm)}


@pytest.mark.parametrize("name", RMS_NORM_MODELS)
def test_rms_norm_eps_default(name):
    assert rms_norm_eps_values(build(name)) == {1e-6}


@pytest.mark.parametrize("name", RMS_NORM_MODELS)
def test_rms_norm_eps_from_config(name):
    """rms_norm_eps доходит до всех RMSNorm: в каждом блоке и финальной."""
    model = build(name, rms_norm_eps=1e-5)
    assert rms_norm_eps_values(model) == {1e-5}
    # 2 нормы на блок + финальная
    assert sum(isinstance(m, RMSNorm) for m in model.modules()) == 2 * BASE_CONFIG["num_layers"] + 1


@pytest.mark.parametrize("name", RMS_NORM_MODELS)
def test_rms_norm_eps_changes_output_not_weights(name):
    torch.manual_seed(0)
    default = build(name)
    torch.manual_seed(0)
    larger = build(name, rms_norm_eps=1e-1)
    assert default.state_dict().keys() == larger.state_dict().keys()

    tokens = torch.randint(0, BASE_CONFIG["vocab_size"], (2, 5))
    with torch.no_grad():
        assert not torch.allclose(default(tokens)[0], larger(tokens)[0])


@pytest.mark.parametrize("eps", [0.0, -1e-6])
def test_rms_norm_eps_must_be_positive(eps):
    with pytest.raises(ValueError, match="eps"):
        build("llama", rms_norm_eps=eps)


ROPE_MODELS = RMS_NORM_MODELS  # те же четыре модели: LLaMA, Mistral, Mixtral, Gemma


def rope_modules(model):
    return [module for module in model.modules() if isinstance(module, RoPE)]


def expected_cos(base, head_size=8):
    freqs = 1.0 / (base ** (2 * torch.arange(head_size // 2).float() / head_size))
    positions = torch.arange(BASE_CONFIG["max_position_embeddings"]).float()
    return torch.cos(positions.unsqueeze(1) * freqs.unsqueeze(0))


@pytest.mark.parametrize("name", ROPE_MODELS)
@pytest.mark.parametrize("theta", [None, 1e6])
def test_rope_theta_reaches_every_attention_layer(name, theta):
    """rope_theta задаёт базу частот единственного объекта RoPE, общего для всех слоёв."""
    model = build(name) if theta is None else build(name, rope_theta=theta)
    ropes = rope_modules(model)
    assert len({id(r) for r in ropes}) == 1
    assert torch.allclose(ropes[0].cos_matrix, expected_cos(10_000 if theta is None else theta))


@pytest.mark.parametrize("name", ROPE_MODELS)
def test_rope_theta_changes_output_not_weights(name):
    torch.manual_seed(0)
    default = build(name)
    torch.manual_seed(0)
    long_context = build(name, rope_theta=1e6)
    assert default.state_dict().keys() == long_context.state_dict().keys()

    tokens = torch.randint(0, BASE_CONFIG["vocab_size"], (2, 8))
    with torch.no_grad():
        assert not torch.allclose(default(tokens)[0], long_context(tokens)[0])


@pytest.mark.parametrize("theta", [1, 0.5, 0, -10])
def test_rope_theta_must_exceed_one(theta):
    with pytest.raises(ValueError, match="base"):
        build("mistral", rope_theta=theta)


@pytest.mark.parametrize("name", ["gpt", "gpt2"])
@pytest.mark.parametrize("value", [None, 0.1])
def test_attention_dropout_reaches_every_layer(name, value):
    """attention_dropout (attn_pdrop в GPT-1/GPT-2) доходит до внимания каждого блока; по умолчанию 0."""
    model = build(name) if value is None else build(name, attention_dropout=value)
    probabilities = {decoder._heads._attn_dropout.p for decoder in model._decoders}
    assert probabilities == {0.0 if value is None else value}
