"""
Content checks for the datasets: exact token ids, padding, truncation,
special tokens and labels.
"""

import pytest
import torch

from llm.datasets.streaming_text_dataset import StreamingTextDataset
from llm.datasets.text_dataset import TextDataset
from llm.datasets.text_with_special_tokens_dataset import TextWithSpecialTokensDataset
from llm.tokenizers import BPETokenizer

BLOCK = 6


class CharTokenizer:
    """Детерминированный токенизатор: символ -> ord(c) - ord('a') + 10."""

    pad_token_id = 1
    bos_token_id = 2
    eos_token_id = 3

    def encode(self, text, add_special_tokens=False, **kwargs):
        ids = [ord(c) - ord("a") + 10 for c in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids


class NoSpecialTokensTokenizer:
    """Токенизатор без pad/bos/eos: датасеты должны взять значения по умолчанию."""

    def encode(self, text, add_special_tokens=False, **kwargs):
        return [ord(c) - ord("a") + 10 for c in text]


def ids(text):
    return [ord(c) - ord("a") + 10 for c in text]


PLAIN_DATASETS = {"text": TextDataset, "streaming": StreamingTextDataset}


@pytest.fixture(params=list(PLAIN_DATASETS), ids=list(PLAIN_DATASETS))
def plain_dataset_class(request):
    return PLAIN_DATASETS[request.param]


class TestPlainDatasets:
    def test_pads_with_tokenizer_pad_id(self, plain_dataset_class):
        item = plain_dataset_class(["abc"], CharTokenizer(), block_size=BLOCK)[0]
        assert item["input_ids"].tolist() == ids("abc") + [1, 1, 1]

    def test_no_special_tokens_added(self, plain_dataset_class):
        """Токенизатор вызывается без спецтокенов, даже если умеет их добавлять."""
        item = plain_dataset_class(["abc"], CharTokenizer(), block_size=BLOCK)[0]
        assert 2 not in item["input_ids"].tolist()
        assert 3 not in item["input_ids"].tolist()

    def test_truncates_to_block_size(self, plain_dataset_class):
        item = plain_dataset_class(["abcdefgh"], CharTokenizer(), block_size=BLOCK)[0]
        assert item["input_ids"].tolist() == ids("abcdef")

    def test_exact_block_size_unchanged(self, plain_dataset_class):
        item = plain_dataset_class(["abcdef"], CharTokenizer(), block_size=BLOCK)[0]
        assert item["input_ids"].tolist() == ids("abcdef")

    def test_default_pad_id_is_zero(self, plain_dataset_class):
        item = plain_dataset_class(["ab"], NoSpecialTokensTokenizer(), block_size=4)[0]
        assert item["input_ids"].tolist() == ids("ab") + [0, 0]

    def test_labels_equal_input_ids(self, plain_dataset_class):
        item = plain_dataset_class(["abc"], CharTokenizer(), block_size=BLOCK)[0]
        assert set(item) == {"input_ids", "labels"}
        assert torch.equal(item["labels"], item["input_ids"])
        assert item["input_ids"].dtype == torch.long

    def test_labels_are_a_copy(self, plain_dataset_class):
        item = plain_dataset_class(["abc"], CharTokenizer(), block_size=BLOCK)[0]
        item["labels"][0] = -100
        assert item["input_ids"][0] != -100

    def test_len(self, plain_dataset_class):
        assert len(plain_dataset_class(["a", "b", "c"], CharTokenizer(), block_size=BLOCK)) == 3


class TestTextWithSpecialTokensDataset:
    def make(self, text, tokenizer=None, **kwargs):
        dataset = TextWithSpecialTokensDataset(
            [text], tokenizer or CharTokenizer(), block_size=BLOCK, **kwargs
        )
        return dataset[0]["input_ids"].tolist()

    def test_no_special_tokens_by_default(self):
        assert self.make("abc") == ids("abc") + [1, 1, 1]

    def test_bos_once(self):
        assert self.make("abc", add_bos=True) == [2] + ids("abc") + [1, 1]

    def test_eos_once(self):
        assert self.make("abc", add_eos=True) == ids("abc") + [3] + [1, 1]

    def test_bos_and_eos_once(self):
        assert self.make("abc", add_bos=True, add_eos=True) == [2] + ids("abc") + [3, 1]

    def test_truncation_reserves_room_for_special_tokens(self):
        assert self.make("abcdefgh", add_bos=True, add_eos=True) == [2] + ids("abcd") + [3]

    def test_no_room_reserved_without_token_ids(self):
        """Если у токенизатора нет bos/eos, место под них не резервируется."""
        tokens = self.make("abcdefgh", tokenizer=NoSpecialTokensTokenizer(), add_bos=True, add_eos=True)
        assert tokens == ids("abcdef")

    def test_default_pad_id_is_zero(self):
        assert self.make("ab", tokenizer=NoSpecialTokensTokenizer()) == ids("ab") + [0] * 4

    def test_labels_equal_input_ids(self):
        item = TextWithSpecialTokensDataset(["abc"], CharTokenizer(), block_size=BLOCK, add_bos=True)[0]
        assert set(item) == {"input_ids", "labels"}
        assert torch.equal(item["labels"], item["input_ids"])
        assert item["input_ids"].dtype == torch.long

    def test_with_bpe_tokenizer(self):
        """С BPETokenizer библиотеки bos/eos не задваиваются."""
        tokenizer = BPETokenizer()
        tokenizer.train(["мир мир"], vocab_size=100, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"])
        vocab = tokenizer.get_vocab()

        item = TextWithSpecialTokensDataset(["мир"], tokenizer, block_size=5, add_bos=True, add_eos=True)[0]

        assert item["input_ids"].tolist() == [
            vocab["<bos>"], vocab["мир"], vocab["<eos>"], vocab["<pad>"], vocab["<pad>"]
        ]
