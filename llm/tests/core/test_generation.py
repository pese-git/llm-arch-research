"""
Tests for shared forward/generate helpers in llm.core.generation.
"""

import pytest
import torch

from llm.core.generation import (
    cache_start_pos,
    check_sequence_length,
    next_generation_input,
    sample_next_token,
    validate_sampling_args,
)


@pytest.mark.parametrize(
    "temperature, top_k, top_p",
    [(1.0, None, None), (0.1, 1, None), (2.0, None, 1.0), (1.0, None, 1e-6)],
)
def test_valid_sampling_args(temperature, top_k, top_p):
    validate_sampling_args(True, temperature, top_k, top_p)


@pytest.mark.parametrize(
    "temperature, top_k, top_p, message",
    [
        (0.0, None, None, "temperature"),
        (-1.0, None, None, "temperature"),
        (1.0, 5, 0.9, "одновременно"),
        (1.0, 0, None, "top_k"),
        (1.0, None, 0.0, "top_p"),
        (1.0, None, -0.1, "top_p"),
        (1.0, None, 1.01, "top_p"),
    ],
)
def test_invalid_sampling_args(temperature, top_k, top_p, message):
    with pytest.raises(ValueError, match=message):
        validate_sampling_args(True, temperature, top_k, top_p)


def test_greedy_skips_validation():
    validate_sampling_args(False, 0.0, 0, 5.0)


def test_cache_start_pos():
    k = torch.zeros(1, 2, 5, 4)
    assert cache_start_pos(None) == 0
    # MHA/MQA: (K, V) — position is the cached length
    assert cache_start_pos([(k, k)]) == 5
    # GQA: (K, V, next_pos) — K is trimmed to the window, position is stored separately
    assert cache_start_pos([(k, k, 12)]) == 12


def test_check_sequence_length():
    check_sequence_length(seq_len=4, start_pos=12, max_seq_len=16)
    with pytest.raises(ValueError, match="кэш 12"):
        check_sequence_length(seq_len=5, start_pos=12, max_seq_len=16)


def test_next_generation_input():
    x = torch.arange(10).unsqueeze(0)
    cache = ["kv"]

    # First step or no cache: the whole sequence
    assert next_generation_input(x, None, True, 16)[0] is x
    x_input, new_cache = next_generation_input(x, cache, False, 16)
    assert x_input is x and new_cache is None

    # With cache: only the last token
    x_input, new_cache = next_generation_input(x, cache, True, 16)
    assert torch.equal(x_input, x[:, -1:]) and new_cache is cache

    # Past max_seq_len: last max_seq_len tokens, cache dropped
    x_input, new_cache = next_generation_input(x, cache, True, 8)
    assert torch.equal(x_input, x[:, -8:]) and new_cache is None


def allowed_tokens(logits, **kwargs):
    """Токены, которые sample_next_token выбирает хотя бы раз за 200 сэмплов."""
    torch.manual_seed(0)
    batch = logits.expand(200, -1)
    return set(sample_next_token(batch, True, **kwargs).flatten().tolist())


PROBS = torch.tensor([[0.5, 0.3, 0.15, 0.05]])


@pytest.mark.parametrize(
    "top_p, expected",
    [(0.4, {0}), (0.5, {0}), (0.7, {0, 1}), (0.8, {0, 1}), (0.81, {0, 1, 2}), (1.0, {0, 1, 2, 3})],
)
def test_top_p_keeps_token_crossing_threshold(top_p, expected):
    """В ядро входит и токен, на котором сумма вероятностей переходит порог."""
    assert allowed_tokens(PROBS.log(), top_p=top_p) == expected


@pytest.mark.parametrize("top_k, expected", [(1, {0}), (2, {0, 1}), (10, {0, 1, 2, 3})])
def test_top_k(top_k, expected):
    assert allowed_tokens(PROBS.log(), top_k=top_k) == expected


def test_greedy_is_argmax():
    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 0.5]])
    assert sample_next_token(logits, False).tolist() == [[1], [0]]


def test_sample_next_token_does_not_modify_logits():
    logits = torch.randn(2, 10)
    original = logits.clone()
    sample_next_token(logits, True, temperature=0.5, top_p=0.5)
    sample_next_token(logits, True, top_k=3)
    assert torch.equal(logits, original)
