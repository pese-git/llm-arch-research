"""
Tests for LLaMA / Mistral / Mixtral / Gemma weight initialization as in HuggingFace
(_init_weights with initializer_range = 0.02; backlog item 62).
"""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from llm.core.rms_norm import RMSNorm
from llm.models.gemma import Gemma
from llm.models.llama import Llama
from llm.models.mistral import Mistral
from llm.models.mixtral import Mixtral

BASE = {
    "vocab_size": 1000,
    "embed_dim": 128,
    "num_layers": 2,
    "max_position_embeddings": 64,
    "dropout": 0.0,
}
CONFIGS = {
    "llama": (Llama, {**BASE, "num_heads": 4}),
    "mistral": (Mistral, {**BASE, "num_q_heads": 4, "num_kv_heads": 2, "window_size": 16}),
    "mixtral": (Mixtral, {**BASE, "num_q_heads": 4, "num_kv_heads": 2, "num_experts": 4, "top_k_experts": 2}),
    "gemma": (Gemma, {**BASE, "num_q_heads": 4, "num_kv_heads": 1}),
    # Как в статье Gemma: общая матрица эмбеддингов и головы, эмбеддинги × √d
    "gemma_tied_scaled": (Gemma, {**BASE, "num_q_heads": 4, "num_kv_heads": 1,
                                  "tie_word_embeddings": True, "scale_embeddings": True}),
}


def build(name, **overrides):
    model_class, config = CONFIGS[name]
    torch.manual_seed(0)
    return model_class({**config, **overrides}).eval()


def assert_std(weight, expected):
    assert weight.std().item() == pytest.approx(expected, rel=0.15)
    assert weight.mean().abs().item() < expected / 5


@pytest.mark.parametrize("name", list(CONFIGS))
def test_linear_embedding_bias_and_norm(name):
    """Linear и Embedding — N(0, 0.02), bias — нули, веса RMSNorm — единицы."""
    model = build(name)
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            assert_std(module.weight, 0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            assert torch.count_nonzero(module.bias) == 0
        if isinstance(module, RMSNorm):
            assert torch.all(module._w == 1)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_initializer_range_from_config(name):
    model = build(name, initializer_range=0.05)
    assert_std(model._token_embeddings._embedding.weight, 0.05)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_initial_loss_close_to_log_vocab(name):
    """Свежая модель почти равномерна по словарю: loss ≈ ln V. С инициализацией PyTorch
    по умолчанию Gemma с общими эмбеддингами × √d давала loss в десятки раз больше."""
    model = build(name)
    tokens = torch.randint(0, BASE["vocab_size"], (4, 32))
    with torch.no_grad():
        logits, _ = model(tokens)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, BASE["vocab_size"]), tokens[:, 1:].reshape(-1))
    assert loss.item() == pytest.approx(math.log(BASE["vocab_size"]), abs=0.1)
