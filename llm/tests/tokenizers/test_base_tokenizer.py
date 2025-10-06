"""
Tests for base tokenizer.
"""

import pytest
from llm.tokenizers import BaseTokenizer


class ConcreteTokenizer(BaseTokenizer):
    """Concrete implementation for testing BaseTokenizer."""

    def train(self, texts: list, vocab_size: int = 1000, **kwargs):
        """Dummy implementation for testing."""
        pass

    def encode(self, text: str, **kwargs) -> list:
        """Dummy implementation for testing."""
        return [1, 2, 3]

    def decode(self, tokens: list, **kwargs) -> str:
        """Dummy implementation for testing."""
        return "decoded text"


class TestBaseTokenizer:
    """Test cases for BaseTokenizer."""

    def test_initialization(self):
        """Test that BaseTokenizer can be initialized through concrete class."""
        tokenizer = ConcreteTokenizer()
        assert tokenizer is not None
        assert tokenizer.vocab == {}
        assert tokenizer.vocab_size == 0

    def test_encode_implemented(self):
        """Test that encode method works in concrete implementation."""
        tokenizer = ConcreteTokenizer()
        result = tokenizer.encode("test text")
        assert result == [1, 2, 3]

    def test_decode_implemented(self):
        """Test that decode method works in concrete implementation."""
        tokenizer = ConcreteTokenizer()
        result = tokenizer.decode([1, 2, 3])
        assert result == "decoded text"

    def test_get_vocab_size(self):
        """Test that get_vocab_size method works."""
        tokenizer = ConcreteTokenizer()
        tokenizer.vocab = {"a": 0, "b": 1, "c": 2}
        tokenizer.vocab_size = 3
        assert tokenizer.get_vocab_size() == 3

    def test_get_vocab(self):
        """Test that get_vocab method works."""
        tokenizer = ConcreteTokenizer()
        tokenizer.vocab = {"a": 0, "b": 1, "c": 2}
        assert tokenizer.get_vocab() == {"a": 0, "b": 1, "c": 2}
