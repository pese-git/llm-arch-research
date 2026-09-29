"""
Tests for generate() in every model: argument handling (temperature, top_k, top_p,
eos_token_id, unknown keys), stopping and the shared implementation in BaseModel.
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


def nucleus(probs, top_p):
    """Ядро top-p: минимальный набор самых вероятных токенов с суммой вероятностей ≥ top_p."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    keep = torch.cumsum(sorted_probs, dim=-1) - sorted_probs < top_p
    return set(sorted_indices[keep].tolist())


def test_top_p_samples_stay_in_nucleus(model, prompt):
    """Каждый сэмпл при top_p лежит в ядре распределения своего промпта."""
    top_p = 0.5
    with torch.no_grad():
        logits, _ = model(prompt, use_cache=False)
    allowed = [nucleus(torch.softmax(row, dim=-1), top_p) for row in logits[:, -1]]
    assert all(len(a) < BASE_CONFIG["vocab_size"] for a in allowed)

    with torch.no_grad():
        for seed in range(40):
            torch.manual_seed(seed)
            out = model.generate(prompt, max_new_tokens=1, do_sample=True, top_p=top_p)
            for row, token in enumerate(out[:, -1].tolist()):
                assert token in allowed[row], f"seed {seed}, row {row}"


def test_default_temperature_is_one(model, prompt):
    with torch.no_grad():
        torch.manual_seed(3)
        default = model.generate(prompt, max_new_tokens=6, do_sample=True)
        torch.manual_seed(3)
        explicit = model.generate(prompt, max_new_tokens=6, do_sample=True, temperature=1.0)

    assert torch.equal(default, explicit)


def test_top_k_larger_than_vocab_is_plain_sampling(model, prompt):
    """top_k больше словаря означает весь словарь, а не ошибку torch.topk."""
    with torch.no_grad():
        torch.manual_seed(5)
        plain = model.generate(prompt, max_new_tokens=6, do_sample=True)
        torch.manual_seed(5)
        clamped = model.generate(
            prompt, max_new_tokens=6, do_sample=True, top_k=BASE_CONFIG["vocab_size"] + 50
        )

    assert torch.equal(clamped, plain)


def test_unknown_keyword_raises(model, prompt):
    """Опечатки и аргументы из других API не проглатываются молча."""
    with pytest.raises(TypeError, match="max_lenght"):
        model.generate(prompt, max_new_tokens=2, do_sample=False, max_lenght=5)


def test_generate_builds_no_autograd_graph(model, prompt):
    """generate не строит граф autograd, даже если вызывающий не обернул его в no_grad."""
    logits_require_grad = []
    hook = model.register_forward_hook(
        lambda module, inputs, output: logits_require_grad.append(output[0].requires_grad)
    )
    model.generate(prompt, max_new_tokens=2, do_sample=False)
    hook.remove()

    assert logits_require_grad == [False, False]
    assert torch.is_grad_enabled()


def test_forward_returns_cache_only_on_request(model, prompt):
    logits, cache = model(prompt)
    assert cache is None
    _, cache = model(prompt, use_cache=True)
    assert len(cache) == BASE_CONFIG["num_layers"]


def greedy_tokens(model, prompt, steps):
    with torch.no_grad():
        return model.generate(prompt, max_new_tokens=steps, do_sample=False)[:, prompt.size(1):]


def test_eos_stops_when_all_rows_finish(model, prompt):
    free = greedy_tokens(model, prompt, 6)
    # eos — токен, который строка 0 генерирует на третьем шаге
    eos = free[0, 2].item()
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=6, do_sample=False, eos_token_id=eos)
    new = out[:, prompt.size(1):]

    finished_at = []
    for row in range(2):
        hits = (free[row] == eos).nonzero()
        finished_at.append(hits[0].item() if len(hits) else None)
    # Генерация идёт, пока не закончат все строки
    expected_len = 6 if None in finished_at else max(finished_at) + 1
    assert new.size(1) == expected_len
    for row, end in enumerate(finished_at):
        stop = expected_len if end is None else end + 1
        assert torch.equal(new[row, :stop], free[row, :stop])
        # После eos строка заполняется pad (по умолчанию — самим eos)
        assert (new[row, stop:] == eos).all()


def test_pad_token_fills_finished_rows(model, prompt):
    free = greedy_tokens(model, prompt, 6)
    eos = free[0, 0].item()
    pad = (eos + 1) % BASE_CONFIG["vocab_size"]
    with torch.no_grad():
        out = model.generate(
            prompt, max_new_tokens=6, do_sample=False, eos_token_id=eos, pad_token_id=pad
        )
    new = out[:, prompt.size(1):]
    assert new[0, 0].item() == eos
    if new.size(1) > 1:
        assert (new[0, 1:] == pad).all()
