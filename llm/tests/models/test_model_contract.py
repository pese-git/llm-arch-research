"""
Contract tests shared by every model: positional information, input length
limit and config handling.
"""

import pytest
import torch

from llm.models.gemma import Gemma
from llm.models.gpt import GPT, GPT2
from llm.models.llama import Llama
from llm.models.mistral import Mistral
from llm.models.mixtral import Mixtral

MAX_LEN = 16

BASE_CONFIG = {
    "vocab_size": 50,
    "embed_dim": 32,
    "num_layers": 2,
    "max_position_embeddings": MAX_LEN,
    "dropout": 0.0,
}

MODELS = {
    "gpt": (GPT, {"num_heads": 4}),
    "gpt2": (GPT2, {"num_heads": 4}),
    "llama": (Llama, {"num_heads": 4}),
    "mistral": (Mistral, {"num_q_heads": 4, "num_kv_heads": 2, "window_size": 8}),
    "mixtral": (
        Mixtral,
        {"num_q_heads": 4, "num_kv_heads": 2, "window_size": 8, "num_experts": 4, "top_k_experts": 2},
    ),
    "gemma": (Gemma, {"num_q_heads": 4}),
}


@pytest.fixture(params=list(MODELS), ids=list(MODELS))
def model_spec(request):
    model_class, extra = MODELS[request.param]
    return model_class, {**BASE_CONFIG, **extra}


def build(model_spec, **overrides):
    model_class, config = model_spec
    torch.manual_seed(0)
    return model_class({**config, **overrides}).eval()


def test_keeps_config(model_spec):
    model_class, config = model_spec
    assert build(model_spec).config == config


def test_output_depends_on_token_order(model_spec):
    """
    Модель должна учитывать позиции. В однослойной модели без позиционного
    кодирования выход последнего токена не зависит от порядка предыдущих
    (внимание суммирует по множеству ключей), поэтому перестановка двух
    первых токенов меняет логиты только при работающих позициях — обучаемых
    (GPT, GPT-2) или RoPE (остальные).

    initializer_range=0.2 (читают только GPT и GPT-2): при N(0, 0.02) из статей у свежей
    GPT-1 (post-LN) скалярные произведения Q·K почти нулевые, внимание почти равномерное,
    и порядок токенов влияет на логиты лишь на уровне 1e-8 — это свойство инициализации,
    а не отсутствие позиций.
    """
    model = build(model_spec, num_layers=1, initializer_range=0.2)
    original = torch.tensor([[3, 11, 25, 40]])
    swapped = torch.tensor([[11, 3, 25, 40]])

    with torch.no_grad():
        logits_original, _ = model(original)
        logits_swapped, _ = model(swapped)

    assert (logits_original[0, -1] - logits_swapped[0, -1]).abs().max() > 1e-4


def test_accepts_max_length(model_spec):
    model = build(model_spec)
    x = torch.randint(0, BASE_CONFIG["vocab_size"], (1, MAX_LEN))

    with torch.no_grad():
        logits, _ = model(x)

    assert logits.shape == (1, MAX_LEN, BASE_CONFIG["vocab_size"])


def test_rejects_too_long_input(model_spec):
    model = build(model_spec)
    x = torch.randint(0, BASE_CONFIG["vocab_size"], (1, MAX_LEN + 1))

    with pytest.raises(ValueError, match=f"{MAX_LEN + 1}.*{MAX_LEN}"):
        model(x)
