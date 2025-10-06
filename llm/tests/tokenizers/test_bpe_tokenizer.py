"""
Tests for BPE tokenizer.
"""

import pytest
import tempfile
import os
from llm.tokenizers import BPETokenizer


class TestBPETokenizer:
    """Test cases for BPETokenizer."""

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for training tokenizer."""
        return [
            "Искусственный интеллект",
            "Нейронные сети",
            "Машинное обучение",
            "Глубокое обучение",
            "Трансформеры",
        ]

    @pytest.fixture
    def trained_tokenizer(self, sample_texts):
        """Create and train a BPE tokenizer."""
        tokenizer = BPETokenizer()
        tokenizer.train(
            texts=sample_texts,
            vocab_size=100,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        )
        return tokenizer

    def test_initialization(self):
        """Test that BPETokenizer can be initialized."""
        tokenizer = BPETokenizer()
        assert tokenizer is not None

    def test_train_tokenizer(self, sample_texts):
        """Test that tokenizer can be trained."""
        tokenizer = BPETokenizer()
        tokenizer.train(
            texts=sample_texts,
            vocab_size=50,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        )

        assert tokenizer.get_vocab_size() > 0
        assert len(tokenizer.get_vocab()) == tokenizer.get_vocab_size()

    def test_encode_decode(self, trained_tokenizer):
        """Test encoding and decoding text."""
        text = "Искусственный интеллект"

        # Encode text
        tokens = trained_tokenizer.encode(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, int) for token in tokens)

        # Decode tokens
        decoded_text = trained_tokenizer.decode(tokens)
        assert isinstance(decoded_text, str)
        # Decoded text should be similar to original (may have special tokens)
        assert len(decoded_text) > 0

    def test_encode_with_special_tokens(self, trained_tokenizer):
        """Test encoding with special tokens."""
        text = "Нейронные сети"

        # Without special tokens
        tokens_no_special = trained_tokenizer.encode(text, add_special_tokens=False)

        # With special tokens
        tokens_with_special = trained_tokenizer.encode(text, add_special_tokens=True)

        # Should have more tokens when special tokens are added
        assert len(tokens_with_special) >= len(tokens_no_special)

    def test_vocab_size(self, trained_tokenizer):
        """Test vocabulary size."""
        vocab_size = trained_tokenizer.get_vocab_size()
        assert isinstance(vocab_size, int)
        assert vocab_size > 0

        vocab = trained_tokenizer.get_vocab()
        assert isinstance(vocab, dict)
        assert len(vocab) == vocab_size

    def test_special_tokens(self, trained_tokenizer):
        """Test that special tokens are in vocabulary."""
        vocab = trained_tokenizer.get_vocab()

        # Check that special tokens are in vocabulary
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        for token in special_tokens:
            assert token in vocab
            assert isinstance(vocab[token], int)

    def test_save_load(self, trained_tokenizer, sample_texts):
        """Test saving and loading tokenizer."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_tokenizer.json")

            # Save tokenizer
            trained_tokenizer.save(save_path)
            assert os.path.exists(save_path)

            # Load tokenizer
            loaded_tokenizer = BPETokenizer.load(save_path)
            assert loaded_tokenizer is not None

            # Check that loaded tokenizer works the same
            original_vocab = trained_tokenizer.get_vocab()
            loaded_vocab = loaded_tokenizer.get_vocab()

            assert original_vocab == loaded_vocab
            assert (
                trained_tokenizer.get_vocab_size() == loaded_tokenizer.get_vocab_size()
            )

            # Test encoding consistency
            text = sample_texts[0]
            original_tokens = trained_tokenizer.encode(text)
            loaded_tokens = loaded_tokenizer.encode(text)

            assert original_tokens == loaded_tokens

    def test_unknown_tokens(self, trained_tokenizer):
        """Test handling of unknown tokens."""
        # Use text that likely contains unknown subwords
        text = "xyzabc123"  # Random text that shouldn't be in training data

        tokens = trained_tokenizer.encode(text)
        assert len(tokens) > 0

        # Should be able to decode back (even if it's mostly unk tokens)
        decoded = trained_tokenizer.decode(tokens)
        assert isinstance(decoded, str)

    def test_empty_text(self, trained_tokenizer):
        """Test encoding and decoding empty text."""
        tokens = trained_tokenizer.encode("")
        assert isinstance(tokens, list)

        decoded = trained_tokenizer.decode([])
        assert decoded == ""

    def test_tokenize_method(self, trained_tokenizer):
        """Test the tokenize method."""
        text = "Искусственный интеллект"
        tokens = trained_tokenizer.tokenize(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
