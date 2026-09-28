"""
Tests for validate_sampling_args.
"""

import pytest

from llm.core.generation import validate_sampling_args


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
