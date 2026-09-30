"""
Tests for attention_mask handling in every model.

Padding anywhere in a row is supported: pad keys are masked and positions count only real
tokens (llm/core/padding.py), so every row gives the same result as without padding.
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
WINDOW_MODELS = {"mistral", "mixtral"}


def build(name):
    torch.manual_seed(0)
    model_class, extra = MODELS[name]
    return model_class({**BASE_CONFIG, **extra}).eval()


@pytest.fixture(params=list(MODELS), ids=list(MODELS))
def model(request):
    model = build(request.param)
    model.name = request.param
    return model


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


def left_pad(rows, pad_id=0):
    """Батч из строк разной длины с паддингом слева и его attention_mask."""
    width = max(len(row) for row in rows)
    tokens = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    for i, row in enumerate(rows):
        tokens[i, width - len(row):] = row
        mask[i, width - len(row):] = 1
    return tokens, mask


def rows_of(tokens):
    return [tokens[0], tokens[1, 2:], tokens[0, :3]]  # длины 6, 4, 3


def test_left_padding_keeps_real_token_logits(model, tokens):
    """Каждая строка с паддингом слева даёт те же логиты, что без паддинга: маска ключей и сдвиг позиций."""
    rows = rows_of(tokens)
    padded, mask = left_pad(rows)
    with torch.no_grad():
        actual, _ = model(padded, attention_mask=mask)
        for i, row in enumerate(rows):
            expected, _ = model(row.unsqueeze(0))
            assert torch.allclose(actual[i, -len(row):], expected[0], atol=1e-5)
    assert torch.isfinite(actual).all()  # у pad-позиций тоже: они видят только себя


def test_gap_in_the_middle_is_skipped(model, tokens):
    """Пропуск внутри строки: позиции считаются только по настоящим токенам."""
    if model.name in WINDOW_MODELS:
        pytest.skip("окно считается по слотам, а не по позициям: с пропуском состав окна другой")
    mask = torch.ones_like(tokens)
    mask[0, 2:4] = 0
    kept = torch.cat([tokens[0, :2], tokens[0, 4:]]).unsqueeze(0)
    with torch.no_grad():
        actual, _ = model(tokens, attention_mask=mask)
        expected, _ = model(kept)
    assert torch.allclose(actual[0, [0, 1, 4, 5]], expected[0], atol=1e-5)


def test_left_padding_with_cache(model, tokens):
    """Префилл с паддингом и шаги с кэшем: маска покрывает кэш и новые токены."""
    rows = rows_of(tokens)
    padded, mask = left_pad(rows)
    with torch.no_grad():
        full, _ = model(padded, attention_mask=mask)
        _, cache = model(padded[:, :4], use_cache=True, attention_mask=mask[:, :4])
        for t in range(4, padded.size(1)):
            step, cache = model(padded[:, t:t + 1], use_cache=True, cache=cache, attention_mask=mask[:, :t + 1])
            real = mask[:, t].bool()
            assert torch.allclose(step[real, 0], full[real, t], atol=1e-5)


def test_mask_with_cache_must_cover_cache(model, tokens):
    """С кэшем маска покрывает кэш и новые токены, даже если в ней одни единицы."""
    with torch.no_grad():
        _, cache = model(tokens[:, :4], use_cache=True)
        model(tokens[:, 4:], cache=cache, attention_mask=torch.ones(2, REAL_LEN))
        for mask in (torch.ones(2, REAL_LEN - 4), torch.tensor([[0.0, 1.0], [1.0, 1.0]])):
            with pytest.raises(ValueError, match="кэш"):
                model(tokens[:, 4:], cache=cache, attention_mask=mask)


def test_short_mask_cannot_hide_padding_in_cache(model, tokens):
    """Кэш с паддингом и маска только по новому токену: ошибка, а не молча неверные логиты."""
    padded, mask = left_pad(rows_of(tokens))
    step = padded[:, -1:]
    with torch.no_grad():
        _, cache = model(padded[:, :-1], use_cache=True, attention_mask=mask[:, :-1])
        with pytest.raises(ValueError, match="кэш"):
            model(step, use_cache=True, cache=cache, attention_mask=torch.ones_like(step))


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


@pytest.mark.parametrize("use_cache", [True, False], ids=["cache", "no_cache"])
def test_generate_left_padded_batch_matches_each_row(model, tokens, use_cache):
    """Батч промптов разной длины с паддингом слева: каждая строка — как её промпт отдельно."""
    rows = rows_of(tokens)
    padded, mask = left_pad(rows)
    new_tokens = 8
    with torch.no_grad():
        batch = model.generate(padded, max_new_tokens=new_tokens, do_sample=False,
                               use_cache=use_cache, attention_mask=mask)
        for i, row in enumerate(rows):
            alone = model.generate(row.unsqueeze(0), max_new_tokens=new_tokens, do_sample=False,
                                   use_cache=use_cache)
            assert torch.equal(batch[i, -new_tokens:], alone[0, -new_tokens:])


@pytest.mark.parametrize("use_cache", [True, False], ids=["cache", "no_cache"])
def test_generate_left_padded_past_max_seq_len(model, tokens, use_cache):
    """За пределами max_seq_len маска обрезается вместе с последовательностью: каждая
    строка батча генерирует то же, что её промпт отдельно."""
    rows = rows_of(tokens)
    padded, mask = left_pad(rows)
    steps = BASE_CONFIG["max_position_embeddings"] + 8
    with torch.no_grad():
        batch = model.generate(padded, max_new_tokens=steps, do_sample=False,
                               use_cache=use_cache, attention_mask=mask)
        assert batch.shape == (3, padded.size(1) + steps)
        for i, row in enumerate(rows):
            alone = model.generate(row.unsqueeze(0), max_new_tokens=steps, do_sample=False,
                                   use_cache=use_cache)
            assert torch.equal(batch[i, -steps:], alone[0, -steps:])


def test_generate_rejects_right_padding(model, tokens):
    """В generate последний токен строки — настоящий: с него продолжается генерация."""
    mask = torch.ones_like(tokens)
    mask[1, -2:] = 0
    with pytest.raises(ValueError, match="слева"):
        model.generate(tokens, max_new_tokens=3, do_sample=False, attention_mask=mask)


def test_generate_rejects_wrong_mask_shape(model, tokens):
    with pytest.raises(ValueError, match="attention_mask"):
        model.generate(tokens, max_new_tokens=3, do_sample=False, attention_mask=torch.ones(2, REAL_LEN + 1))
