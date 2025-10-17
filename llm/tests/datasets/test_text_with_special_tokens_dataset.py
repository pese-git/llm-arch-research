import torch
import pytest
from llm.datasets.text_with_special_tokens_dataset import TextWithSpecialTokensDataset

class DummyTokenizer:
    def __init__(self):
        self.bos_token_id = 101
        self.eos_token_id = 102
        self.pad_token_id = 0
    def encode(self, text, add_special_tokens=False, add_bos_token=False, add_eos_token=False):
        ids = [ord(c) % 50 for c in text.strip()]
        if add_bos_token:
            ids = [self.bos_token_id] + ids
        if add_eos_token:
            ids = ids + [self.eos_token_id]
        return ids

def test_specialtokens_basic_bos_eos():
    texts = ["abc", "d"]
    tokenizer = DummyTokenizer()
    block_size = 6
    ds = TextWithSpecialTokensDataset(texts, tokenizer, block_size=block_size, add_bos=True, add_eos=True)
    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert item["input_ids"].shape == (block_size,)
        assert item["input_ids"][0] == tokenizer.bos_token_id
        assert item["input_ids"][item["input_ids"].ne(tokenizer.pad_token_id).sum() - 1] == tokenizer.eos_token_id

def test_specialtokens_padding_and_truncation():
    texts = ["qwertyuiop", "z"]
    tokenizer = DummyTokenizer()
    block_size = 5
    ds = TextWithSpecialTokensDataset(texts, tokenizer, block_size=block_size, add_bos=True)
    assert ds[0]["input_ids"].shape[0] == block_size
    assert ds[1]["input_ids"][-1] == tokenizer.pad_token_id

def test_specialtokens_no_bos_eos():
    texts = ["xyz"]
    tokenizer = DummyTokenizer()
    block_size = 6
    ds = TextWithSpecialTokensDataset(texts, tokenizer, block_size=block_size, add_bos=False, add_eos=False)
    item = ds[0]["input_ids"]
    assert tokenizer.bos_token_id not in item
    assert tokenizer.eos_token_id not in item
    assert item.shape == (block_size,)

def test_specialtokens_index_error():
    texts = ["sample"]
    tokenizer = DummyTokenizer()
    ds = TextWithSpecialTokensDataset(texts, tokenizer, block_size=8)
    with pytest.raises(IndexError):
        _ = ds[1]

def test_specialtokens_labels():
    texts = ["abcd"]
    tokenizer = DummyTokenizer()
    block_size = 7
    ds = TextWithSpecialTokensDataset(texts, tokenizer, block_size=block_size, add_bos=True, add_eos=True)
    item = ds[0]
    assert torch.equal(item["input_ids"], item["labels"])
