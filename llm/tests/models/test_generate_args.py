"""
Tests for generate() argument handling (temperature, top_k, top_p) in every model.
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
    "mistral": (Mistral, {"num_q_heads": 4, "num_kv_heads": 2, "window_size": 8}),
    "mixtral": (
        Mixtral,
        {"num_q_heads": 4, "num_kv_heads": 2, "window_size": 8, "num_experts": 4, "top_k_experts": 2},
    ),
    "gemma": (Gemma, {"num_q_heads": 4}),
}


@pytest.fixture(params=list(MODELS), ids=list(MODELS))
def model(request):
    torch.manual_seed(0)
    model_class, extra = MODELS[request.param]
    return model_class({**BASE_CONFIG, **extra}).eval()


@pytest.fixture
def prompt():
    torch.manual_seed(0)
    return torch.randint(0, BASE_CONFIG["vocab_size"], (2, 4))


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_greedy_ignores_non_positive_temperature(model, prompt, temperature):
    """При жадной генерации температура не влияет на результат, в том числе 0."""
    with torch.no_grad():
        expected = model.generate(prompt, max_new_tokens=4, do_sample=False)
        actual = model.generate(
            prompt, max_new_tokens=4, do_sample=False, temperature=temperature
        )

    assert torch.equal(actual, expected)


def test_greedy_ignores_sampling_args(model, prompt):
    """top_k/top_p проверяются только при сэмплировании."""
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=2, do_sample=False, top_k=0, top_p=2.0)

    assert out.shape == (2, 6)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"temperature": 0.0}, "temperature"),
        ({"temperature": -0.5}, "temperature"),
        ({"top_k": 5, "top_p": 0.9}, "одновременно"),
        ({"top_k": 0}, "top_k"),
        ({"top_k": -1}, "top_k"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.5}, "top_p"),
    ],
    ids=["temp_zero", "temp_negative", "top_k_and_top_p", "top_k_zero", "top_k_negative", "top_p_zero", "top_p_above_one"],
)
def test_sampling_rejects_invalid_args(model, prompt, kwargs, message):
    with pytest.raises(ValueError, match=message):
        model.generate(prompt, max_new_tokens=2, do_sample=True, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"temperature": 0.5}, {"top_k": 1}, {"top_p": 1.0}],
    ids=["temperature", "top_k_one", "top_p_one"],
)
def test_sampling_accepts_boundary_args(model, prompt, kwargs):
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=2, do_sample=True, **kwargs)

    assert out.shape == (2, 6)


def test_top_k_one_is_greedy(model, prompt):
    """Сэмплирование из одного лучшего токена совпадает с жадной генерацией."""
    with torch.no_grad():
        greedy = model.generate(prompt, max_new_tokens=4, do_sample=False)
        top1 = model.generate(prompt, max_new_tokens=4, do_sample=True, top_k=1)

    assert torch.equal(top1, greedy)


# Семантика сэмплирования: в предельных случаях сэмплирование обязано совпасть
# с жадной генерацией. Батч из двух промптов, чтобы ошибки в оси (softmax/cumsum
# по батчу вместо словаря) тоже меняли результат.
SAMPLING_LIMITS = {
    "near_zero_temperature": {"temperature": 1e-4},
    "tiny_top_p": {"top_p": 1e-6},
    "top_k_one": {"top_k": 1},
}


@pytest.mark.parametrize("kwargs", list(SAMPLING_LIMITS.values()), ids=list(SAMPLING_LIMITS))
def test_sampling_limit_equals_greedy(model, prompt, kwargs):
    with torch.no_grad():
        greedy = model.generate(prompt, max_new_tokens=6, do_sample=False)
        torch.manual_seed(1)
        sampled = model.generate(prompt, max_new_tokens=6, do_sample=True, **kwargs)

    assert torch.equal(sampled, greedy)


def test_plain_sampling_differs_from_greedy(model, prompt):
    """Контроль для теста выше: обычное сэмплирование с этим seed дает другой результат."""
    with torch.no_grad():
        greedy = model.generate(prompt, max_new_tokens=6, do_sample=False)
        torch.manual_seed(1)
        sampled = model.generate(prompt, max_new_tokens=6, do_sample=True)

    assert not torch.equal(sampled, greedy)
