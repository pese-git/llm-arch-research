import torch
import pytest
from llm.datasets.text_dataset import TextDataset

class DummyTokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
    def encode(self, text, **kwargs):
        return [len(w) % self.vocab_size for w in text.strip().split()]

def test_textdataset_shape_and_basic():
    texts = ["hello world", "this is a test", "Transformer model"]
    tokenizer = DummyTokenizer(50)
    block_size = 6
    dataset = TextDataset(texts, tokenizer, block_size=block_size)
    for i in range(len(dataset)):
        x = dataset[i]
        assert isinstance(x, dict)
        assert "input_ids" in x
        assert isinstance(x["input_ids"], torch.Tensor)
        assert x["input_ids"].shape == (block_size,)

def test_textdataset_truncation_and_padding():
    texts = ["one two three four five six seven", "short"]
    tokenizer = DummyTokenizer(100)
    block_size = 5
    dataset = TextDataset(texts, tokenizer, block_size=block_size)
    assert isinstance(dataset[0], dict)
    assert dataset[0]["input_ids"].shape[0] == 5
    assert dataset[1]["input_ids"].shape[0] == 5

def test_textdataset_index_error():
    texts = ["a", "b"]
    tokenizer = DummyTokenizer(10)
    dataset = TextDataset(texts, tokenizer, block_size=3)
    with pytest.raises(IndexError):
        _ = dataset[2]

def test_textdataset_encoding():
    texts = ["привет", "мир"]
    tokenizer = DummyTokenizer(20)
    block_size = 4
    dataset = TextDataset(texts, tokenizer, block_size=block_size)
    assert len(dataset) == 2
    x = dataset[0]
    assert isinstance(x, dict)
    assert "input_ids" in x
    assert isinstance(x["input_ids"], torch.Tensor)
    assert x["input_ids"].shape == (block_size,)