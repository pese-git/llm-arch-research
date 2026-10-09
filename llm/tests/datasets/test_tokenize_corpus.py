"""
tokenize_file: содержимое .bin и .json, границы документов, выбор dtype, стыковка
с TokenBlockDataset и с BPETokenizer.
"""

import json

import numpy as np
import pytest

from llm.datasets.token_block_dataset import TokenBlockDataset
from llm.datasets.tokenize_corpus import tokenize_file
from llm.tokenizers import BPETokenizer


class CharTokenizer:
    """Детерминированный токенизатор: символ -> ord(c) - ord('a') + 10, словарь 40."""

    eos_token_id = 3

    def encode(self, text, add_special_tokens=False, **kwargs):
        assert not add_special_tokens
        return [ord(c) - ord("a") + 10 for c in text]

    def get_vocab_size(self):
        return 40


def ids(text):
    return [ord(c) - ord("a") + 10 for c in text]


def write(tmp_path, text):
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_tokens_match_manual_encode_and_meta_is_written(tmp_path):
    text = "abc\nde\n"
    out = str(tmp_path / "train.bin")
    n = tokenize_file(write(tmp_path, text), CharTokenizer(), out)
    expected = ids("abc") + ids("de")
    assert n == len(expected)
    assert np.fromfile(out, dtype=np.uint16).tolist() == expected
    meta = json.loads((tmp_path / "train.bin.json").read_text())
    assert meta == {
        "dtype": "uint16",
        "num_tokens": n,
        "num_documents": 1,
        "vocab_size": 40,
        "eos_token_id": None,
    }


def test_eos_between_documents_and_at_end(tmp_path):
    """Пустая строка — граница документа; eos после каждого документа, включая последний."""
    text = "ab\ncd\n\n\nef\n"
    out = str(tmp_path / "train.bin")
    tokenize_file(write(tmp_path, text), CharTokenizer(), out, eos_token_id=3)
    assert np.fromfile(out, dtype=np.uint16).tolist() == ids("abcd") + [3] + ids("ef") + [3]
    assert json.loads((tmp_path / "train.bin.json").read_text())["num_documents"] == 2


def test_no_eos_without_eos_token_id(tmp_path):
    out = str(tmp_path / "train.bin")
    tokenize_file(write(tmp_path, "ab\n\ncd\n"), CharTokenizer(), out)
    assert np.fromfile(out, dtype=np.uint16).tolist() == ids("abcd")


def test_chunked_writing_gives_same_result(tmp_path):
    """Запись чанками по одной строке совпадает с записью за один раз."""
    text = "\n".join(["abc", "", "de", "fgh", "", "i"]) + "\n"
    src = write(tmp_path, text)
    one = str(tmp_path / "one.bin")
    many = str(tmp_path / "many.bin")
    tokenize_file(src, CharTokenizer(), one, eos_token_id=3)
    tokenize_file(src, CharTokenizer(), many, eos_token_id=3, chunk_lines=1)
    assert np.fromfile(one, dtype=np.uint16).tolist() == np.fromfile(many, dtype=np.uint16).tolist()


def test_dtype_uint32_for_large_vocab_and_explicit_dtype(tmp_path):
    class BigVocab(CharTokenizer):
        def get_vocab_size(self):
            return 70_000

    out = str(tmp_path / "big.bin")
    tokenize_file(write(tmp_path, "ab\n"), BigVocab(), out)
    assert json.loads((tmp_path / "big.bin.json").read_text())["dtype"] == "uint32"
    assert np.fromfile(out, dtype=np.uint32).tolist() == ids("ab")

    out = str(tmp_path / "explicit.bin")
    tokenize_file(write(tmp_path, "ab\n"), CharTokenizer(), out, dtype="uint32")
    assert np.fromfile(out, dtype=np.uint32).tolist() == ids("ab")

    with pytest.raises(ValueError, match="dtype"):
        tokenize_file(write(tmp_path, "ab\n"), CharTokenizer(), out, dtype="int8")


def test_token_id_out_of_dtype_range_raises(tmp_path):
    class Overflow:
        def encode(self, text, add_special_tokens=False, **kwargs):
            return [70_000]

        def get_vocab_size(self):
            return 100  # врёт: id не влезает в uint16

    with pytest.raises(ValueError, match="uint16"):
        tokenize_file(write(tmp_path, "a\n"), Overflow(), str(tmp_path / "x.bin"))


def test_tokenizer_without_get_vocab_size_defaults_to_uint32(tmp_path):
    class Bare:
        def encode(self, text, add_special_tokens=False, **kwargs):
            return [1, 2]

    out = str(tmp_path / "bare.bin")
    tokenize_file(write(tmp_path, "x\n"), Bare(), out)
    meta = json.loads((tmp_path / "bare.bin.json").read_text())
    assert meta["dtype"] == "uint32" and meta["vocab_size"] is None


def test_roundtrip_with_token_block_dataset(tmp_path):
    text = "abcdefgh\nij\n"
    out = str(tmp_path / "train.bin")
    n = tokenize_file(write(tmp_path, text), CharTokenizer(), out)
    dataset = TokenBlockDataset(out, block_size=4)
    assert len(dataset) == n // 4
    assert dataset[0]["input_ids"].tolist() == ids("abcd")
    assert dataset[1]["input_ids"].tolist() == ids("efgh")


def test_works_with_bpe_tokenizer(tmp_path):
    """Настоящий BPETokenizer: токены файла равны encode каждой строки, eos — его eos_token_id."""
    texts = ["нейронные сети учатся", "трансформеры обрабатывают последовательности"]
    tokenizer = BPETokenizer()
    tokenizer.train(texts=texts, vocab_size=60, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"])
    out = str(tmp_path / "train.bin")
    n = tokenize_file(
        write(tmp_path, "\n\n".join(texts) + "\n"), tokenizer, out, eos_token_id=tokenizer.eos_token_id
    )
    expected = []
    for t in texts:
        expected += tokenizer.encode(t, add_special_tokens=False) + [tokenizer.eos_token_id]
    assert np.fromfile(out, dtype=np.uint16).tolist() == expected
    assert n == len(expected)
