"""
Tests for base tokenizer.
"""

import json

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


class CharTokenizer(BaseTokenizer):
    """Minimal character-level tokenizer to exercise BaseTokenizer helpers."""

    def train(self, texts: list, vocab_size: int = 1000, **kwargs):
        chars = sorted(set("".join(texts)))
        self.vocab = {char: i for i, char in enumerate(chars)}
        self.inverse_vocab = {i: char for char, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.add_special_tokens(kwargs.get("special_tokens", []))

    def encode(self, text: str, **kwargs) -> list:
        return [self.vocab.get(char, self.unk_token_id) for char in text]

    def decode(self, tokens: list, **kwargs) -> str:
        return "".join(self.inverse_vocab.get(token, "") for token in tokens)


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


@pytest.fixture
def char_tokenizer():
    tokenizer = CharTokenizer()
    tokenizer.train(["абв", "вг"], special_tokens=SPECIAL_TOKENS)
    return tokenizer


class TestAbstractInterface:
    def test_cannot_instantiate_base_class(self):
        with pytest.raises(TypeError):
            BaseTokenizer()

    def test_subclass_must_implement_all_methods(self):
        class Partial(BaseTokenizer):
            def train(self, texts, vocab_size=1000, **kwargs):
                pass

        with pytest.raises(TypeError):
            Partial()

    def test_base_implementations_are_noops(self):
        """Абстрактные методы базового класса ничего не делают при вызове через super()."""

        class Delegating(BaseTokenizer):
            def train(self, texts, vocab_size=1000, **kwargs):
                return super().train(texts, vocab_size, **kwargs)

            def encode(self, text, **kwargs):
                return super().encode(text, **kwargs)

            def decode(self, tokens, **kwargs):
                return super().decode(tokens, **kwargs)

        tokenizer = Delegating()
        assert tokenizer.train(["x"]) is None
        assert tokenizer.encode("x") is None
        assert tokenizer.decode([0]) is None


class TestSpecialTokens:
    def test_defaults_before_training(self):
        tokenizer = ConcreteTokenizer()
        assert tokenizer.pad_token == "<pad>"
        assert tokenizer.eos_token == "<eos>"
        assert tokenizer.pad_token_id is None
        assert tokenizer.eos_token_id is None

    def test_add_special_tokens_appends_ids(self, char_tokenizer):
        # 4 символа "а", "б", "в", "г" + 4 специальных токена
        assert char_tokenizer.get_vocab_size() == 8
        assert char_tokenizer.pad_token_id == 4
        assert char_tokenizer.unk_token_id == 5
        assert char_tokenizer.bos_token_id == 6
        assert char_tokenizer.eos_token_id == 7
        assert char_tokenizer.inverse_vocab[7] == "<eos>"

    def test_add_special_tokens_skips_existing(self, char_tokenizer):
        char_tokenizer.add_special_tokens(["<pad>", "а", "<sep>"])

        assert char_tokenizer.get_vocab_size() == 9
        assert char_tokenizer.vocab["<sep>"] == 8
        assert char_tokenizer.pad_token_id == 4

    def test_partial_special_tokens(self):
        tokenizer = CharTokenizer()
        tokenizer.train(["аб"], special_tokens=["<pad>"])

        assert tokenizer.pad_token_id == 2
        assert tokenizer.unk_token_id is None
        assert tokenizer.bos_token_id is None


class TestHelpers:
    def test_tokenize_maps_ids_to_strings(self, char_tokenizer):
        assert char_tokenizer.tokenize("вба") == ["в", "б", "а"]

    def test_tokenize_unknown_character(self, char_tokenizer):
        assert char_tokenizer.tokenize("аz") == ["а", "<unk>"]

    def test_get_vocab_returns_copy(self, char_tokenizer):
        vocab = char_tokenizer.get_vocab()
        vocab["new"] = 100
        assert "new" not in char_tokenizer.vocab

    def test_len(self, char_tokenizer):
        assert len(char_tokenizer) == char_tokenizer.get_vocab_size() == 8

    def test_repr(self, char_tokenizer):
        assert repr(char_tokenizer) == "CharTokenizer(vocab_size=8)"


class TestSaveLoad:
    def test_save_writes_config(self, char_tokenizer, tmp_path):
        path = tmp_path / "tokenizer.json"
        char_tokenizer.save(str(path))

        config = json.loads(path.read_text(encoding="utf-8"))
        assert config["tokenizer_type"] == "CharTokenizer"
        assert config["vocab"] == char_tokenizer.vocab
        assert config["vocab_size"] == 8
        assert config["unk_token"] == "<unk>"

    def test_save_keeps_non_ascii(self, char_tokenizer, tmp_path):
        path = tmp_path / "tokenizer.json"
        char_tokenizer.save(str(path))
        # ensure_ascii=False: кириллица пишется как есть, а не \uXXXX
        assert "а" in path.read_text(encoding="utf-8")

    def test_roundtrip(self, char_tokenizer, tmp_path):
        path = tmp_path / "tokenizer.json"
        char_tokenizer.save(str(path))
        loaded = CharTokenizer.load(str(path))

        assert isinstance(loaded, CharTokenizer)
        assert loaded.get_vocab() == char_tokenizer.get_vocab()
        assert loaded.inverse_vocab == char_tokenizer.inverse_vocab
        assert len(loaded) == len(char_tokenizer)
        for name in ["pad", "unk", "bos", "eos"]:
            assert getattr(loaded, f"{name}_token") == getattr(char_tokenizer, f"{name}_token")
            assert getattr(loaded, f"{name}_token_id") == getattr(char_tokenizer, f"{name}_token_id")
        assert loaded.encode("абвг") == char_tokenizer.encode("абвг")
        assert loaded.decode(loaded.encode("гба")) == "гба"

    def test_roundtrip_custom_special_tokens(self, tmp_path):
        tokenizer = CharTokenizer()
        tokenizer.pad_token = "[PAD]"
        tokenizer.eos_token = "[EOS]"
        tokenizer.train(["аб"], special_tokens=["[PAD]", "[EOS]"])

        path = tmp_path / "tokenizer.json"
        tokenizer.save(str(path))
        loaded = CharTokenizer.load(str(path))

        assert loaded.pad_token == "[PAD]"
        assert loaded.pad_token_id == tokenizer.pad_token_id
        assert loaded.eos_token_id == tokenizer.eos_token_id
        assert loaded.unk_token_id is None
