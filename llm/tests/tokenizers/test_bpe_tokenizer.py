"""
Tests for BPE tokenizer.
"""

import pytest
import tempfile
import os
import json
from llm.tokenizers import BPETokenizer
from llm.tokenizers.bpe_tokenizer import pretokenize


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

    def test_decode_accepts_tensor(self, trained_tokenizer):
        """decode() should accept a torch.Tensor of ids, not just a list of int."""
        torch = pytest.importorskip("torch")
        text = "Искусственный интеллект"

        tokens = trained_tokenizer.encode(text)
        decoded_from_list = trained_tokenizer.decode(tokens)

        decoded_from_tensor = trained_tokenizer.decode(torch.tensor(tokens))
        assert decoded_from_tensor == decoded_from_list

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


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class TestPretokenize:
    @pytest.mark.parametrize(
        "text",
        ["Привет, мир!", "a  b\n\nc ", "  x", "3.14 — это π", "", "word"],
    )
    def test_lossless(self, text):
        assert "".join(pretokenize(text)) == text

    def test_space_attached_to_next_word(self):
        assert pretokenize("Привет, мир!") == ["Привет", ",", " мир", "!"]


class TestBPEWordBoundaries:
    """Слияния BPE не должны выходить за границы слов и текстов."""

    @pytest.fixture
    def texts(self):
        return [
            "Нейронные сети учатся на данных.",
            "Нейронные сети обрабатывают текст.",
            "Трансформеры изменили обработку текста.",
        ]

    def test_tokens_do_not_span_words(self, texts):
        tokenizer = BPETokenizer()
        # Большой vocab_size: обучение идет до конца, пока слова не сольются
        tokenizer.train(texts, vocab_size=10_000, special_tokens=SPECIAL_TOKENS)

        for token in tokenizer.vocab_list:
            assert len(pretokenize(token)) == 1, token

    def test_stops_when_words_fully_merged(self, texts):
        tokenizer = BPETokenizer()
        tokenizer.train(texts, vocab_size=10_000, special_tokens=SPECIAL_TOKENS)

        words = {word for text in texts for word in pretokenize(text)}
        assert words <= set(tokenizer.vocab_list)
        assert tokenizer.get_vocab_size() < 10_000

    def test_repeated_word_becomes_single_token(self):
        tokenizer = BPETokenizer()
        tokenizer.train(["мир мир мир мир"], vocab_size=100, special_tokens=SPECIAL_TOKENS)

        assert tokenizer.tokenize("мир мир") == ["мир", " мир"]

    def test_texts_are_not_merged_together(self):
        tokenizer = BPETokenizer()
        tokenizer.train(["аб", "вг"] * 10, vocab_size=100, special_tokens=SPECIAL_TOKENS)

        assert not any("б" in token and "в" in token for token in tokenizer.vocab_list)

    def test_encode_decode_roundtrip(self, texts):
        tokenizer = BPETokenizer()
        tokenizer.train(texts, vocab_size=80, special_tokens=SPECIAL_TOKENS)

        # Новая фраза из тех же символов: неизвестные символы стали бы <unk>
        for text in texts + ["Нейронные данные обрабатывают текст."]:
            assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_vocab_ids_are_unique_and_contiguous(self, texts):
        tokenizer = BPETokenizer()
        tokenizer.train(texts, vocab_size=10_000, special_tokens=SPECIAL_TOKENS)

        ids = sorted(tokenizer.get_vocab().values())
        assert ids == list(range(tokenizer.get_vocab_size()))
        assert len(tokenizer.vocab_list) == len(set(tokenizer.vocab_list))

    def test_training_is_deterministic(self, texts):
        first, second = BPETokenizer(), BPETokenizer()
        first.train(texts, vocab_size=60, special_tokens=SPECIAL_TOKENS)
        second.train(texts, vocab_size=60, special_tokens=SPECIAL_TOKENS)

        assert first.get_vocab() == second.get_vocab()
        assert first.merges == second.merges


class TestBPEMerges:
    def test_merges_recorded_in_rank_order(self):
        tokenizer = BPETokenizer()
        tokenizer.train(["мир мир мир"], vocab_size=100, special_tokens=SPECIAL_TOKENS)

        assert tokenizer.merges
        assert sorted(tokenizer.merges.values()) == list(range(len(tokenizer.merges)))
        for left, right in tokenizer.merges:
            assert left + right in tokenizer.get_vocab()

    def test_save_load_keeps_merges_with_commas(self, tmp_path):
        tokenizer = BPETokenizer()
        tokenizer.train(["a,, b,, c,, a,, b,,"], vocab_size=100, special_tokens=SPECIAL_TOKENS)
        assert any("," in left + right for left, right in tokenizer.merges)

        path = tmp_path / "tokenizer.json"
        tokenizer.save(str(path))
        loaded = BPETokenizer.load(str(path))

        assert loaded.merges == tokenizer.merges

    def test_load_legacy_merges_format(self, tmp_path):
        tokenizer = BPETokenizer()
        tokenizer.train(["мир мир"], vocab_size=100, special_tokens=SPECIAL_TOKENS)
        path = tmp_path / "tokenizer.json"
        tokenizer.save(str(path))

        config = json.loads(path.read_text(encoding="utf-8"))
        config["merges"] = {"м,и": 0, "ми,р": 1}
        path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

        loaded = BPETokenizer.load(str(path))
        assert loaded.merges == {("м", "и"): 0, ("ми", "р"): 1}


class TestBPESpecialTokenHandling:
    def test_encode_without_special_tokens_in_vocab(self):
        """add_special_tokens=True ничего не добавляет, если bos/eos нет в словаре."""
        tokenizer = BPETokenizer()
        tokenizer.train(["мир мир"], vocab_size=100, special_tokens=[])
        assert tokenizer.bos_token_id is None and tokenizer.eos_token_id is None

        with_special = tokenizer.encode("мир", add_special_tokens=True)
        assert with_special == tokenizer.encode("мир", add_special_tokens=False)

    def test_decode_keeps_special_tokens_when_asked(self):
        tokenizer = BPETokenizer()
        tokenizer.train(["мир мир"], vocab_size=100, special_tokens=SPECIAL_TOKENS)
        ids = tokenizer.encode("мир", add_special_tokens=True)

        assert tokenizer.decode(ids) == "мир"
        assert tokenizer.decode(ids, skip_special_tokens=False) == "<bos>мир<eos>"

    def test_load_legacy_merges_skips_ambiguous_keys(self, tmp_path):
        """В старом формате ключ "a,b" с запятой внутри токена неоднозначен и пропускается."""
        tokenizer = BPETokenizer()
        tokenizer.train(["мир мир"], vocab_size=100, special_tokens=SPECIAL_TOKENS)
        path = tmp_path / "tokenizer.json"
        tokenizer.save(str(path))

        config = json.loads(path.read_text(encoding="utf-8"))
        config["merges"] = {"м,и": 0, ",,,": 1}
        path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

        assert BPETokenizer.load(str(path)).merges == {("м", "и"): 0}
