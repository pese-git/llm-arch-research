"""
Tests for GPT-1 / GPT-2 weight initialization from the papers and OpenAI code.
"""

import math

import pytest
import torch
from torch import nn

from llm.core.weight_init import init_normal_, scale_residual_projections_
from llm.models.gpt import GPT, GPT2

CONFIG = {
    "vocab_size": 2000,
    "embed_dim": 128,
    "num_heads": 4,
    "num_layers": 4,
    "max_position_embeddings": 256,
    "dropout": 0.0,
}


def residual_projections(model):
    return [p for d in model._decoders for p in (d._heads._layer, d._ff._layer2)]


def assert_std(weight, expected):
    assert weight.std().item() == pytest.approx(expected, rel=0.1)
    assert weight.mean().abs().item() < expected / 10


@pytest.mark.parametrize("model_class", [GPT, GPT2], ids=["gpt", "gpt2"])
def test_linear_and_embedding_weights(model_class):
    torch.manual_seed(0)
    model = model_class(CONFIG)
    residual = {id(p) for p in residual_projections(model)} if model_class is GPT2 else set()

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)) and id(module) not in residual:
            assert_std(module.weight, 0.02)
        if isinstance(module, nn.Linear):
            assert torch.count_nonzero(module.bias) == 0
        if isinstance(module, nn.LayerNorm):
            assert torch.all(module.weight == 1) and torch.count_nonzero(module.bias) == 0


def test_gpt1_residual_projections_are_not_scaled():
    torch.manual_seed(0)
    for projection in residual_projections(GPT(CONFIG)):
        assert_std(projection.weight, 0.02)


def test_gpt2_residual_projections_are_scaled():
    """Attention output and second FFN layer: N(0, 0.02 / √(2·num_layers))."""
    torch.manual_seed(0)
    expected = 0.02 / math.sqrt(2 * CONFIG["num_layers"])
    projections = residual_projections(GPT2(CONFIG))
    assert len(projections) == 2 * CONFIG["num_layers"]
    for projection in projections:
        assert_std(projection.weight, expected)
        assert torch.count_nonzero(projection.bias) == 0


@pytest.mark.parametrize("model_class", [GPT, GPT2], ids=["gpt", "gpt2"])
def test_initializer_range_from_config(model_class):
    torch.manual_seed(0)
    model = model_class({**CONFIG, "initializer_range": 0.05})
    assert_std(model._token_embeddings._embedding.weight, 0.05)


def test_helpers_on_plain_modules():
    torch.manual_seed(0)
    linear = nn.Linear(256, 256)
    init_normal_(linear, std=0.1)
    assert_std(linear.weight, 0.1)
    assert torch.count_nonzero(linear.bias) == 0

    scale_residual_projections_([linear], num_layers=8, std=0.1)
    assert_std(linear.weight, 0.1 / 4)
