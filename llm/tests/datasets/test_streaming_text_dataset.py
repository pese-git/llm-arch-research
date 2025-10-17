import torch
import pytest
from llm.datasets.streaming_text_dataset import StreamingTextDataset

class DummyTokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
    def encode(self, text, **kwargs):
        return [len(w) % self.vocab_size for w in text.strip().split()]

def test_streaming_textdataset_basic_shape():
    texts = ["hello world", "big transformers are fun", "LLM test string"]
    tokenizer = DummyTokenizer(50)
    block_size = 7
    ds = StreamingTextDataset(texts, tokenizer, block_size)
    assert len(ds) == 3
    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert item["input_ids"].shape == (block_size,)
        assert "labels" in item
        assert item["labels"].shape == (block_size,)

def test_streaming_textdataset_padding_and_truncation():
    texts = ["short", "one two three four five six seven eight nine ten"]
    tokenizer = DummyTokenizer(40)
    block_size = 4
    ds = StreamingTextDataset(texts, tokenizer, block_size)
    # короткое предложение padded
    assert (ds[0]["input_ids"].shape[0] == block_size)
    # длинное предложение truncated
    assert (ds[1]["input_ids"].shape[0] == block_size)

def test_streaming_textdataset_index_error():
    texts = ["sample"]
    tokenizer = DummyTokenizer(10)
    ds = StreamingTextDataset(texts, tokenizer, block_size=5)
    with pytest.raises(IndexError):
        _ = ds[1]

def test_streaming_textdataset_content_matching():
    texts = ["foo bar baz", "abc def"]
    tokenizer = DummyTokenizer(99)
    block_size = 5
    ds = StreamingTextDataset(texts, tokenizer, block_size)
    # Проверка, что input_ids и labels совпадают точно
    for i in range(len(ds)):
        assert torch.equal(ds[i]["input_ids"], ds[i]["labels"])
