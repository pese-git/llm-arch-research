"""
TokenBlockDataset: непрерывные блоки без паддинга, стыковка блоков, чтение из файла.
"""

import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from llm.datasets.token_block_dataset import TokenBlockDataset

TOKENS = list(range(100, 123))  # 23 токена
BLOCK = 5


def test_len_drops_incomplete_tail():
    """23 токена по 5 — четыре полных блока, остаток из трёх токенов отбрасывается."""
    assert len(TokenBlockDataset(TOKENS, BLOCK)) == 23 // 5 == 4


def test_items_are_contiguous_and_adjacent_blocks_touch():
    """Блок i — токены с i·block по i·block + block; блок i+1 начинается там, где кончился i."""
    dataset = TokenBlockDataset(TOKENS, BLOCK)
    for i in range(len(dataset)):
        item = dataset[i]
        assert item["input_ids"].tolist() == TOKENS[i * BLOCK : (i + 1) * BLOCK]
    assert dataset[1]["input_ids"][0].item() == dataset[0]["input_ids"][-1].item() + 1


def test_labels_equal_input_ids_without_shift_and_no_mask():
    """Метки — копия входа (сдвиг делает loss), паддинга и attention_mask нет."""
    item = TokenBlockDataset(TOKENS, BLOCK)[2]
    assert set(item) == {"input_ids", "labels"}
    assert torch.equal(item["labels"], item["input_ids"])
    assert item["labels"].data_ptr() != item["input_ids"].data_ptr()
    assert item["input_ids"].dtype == torch.long and item["labels"].dtype == torch.long
    assert (item["labels"] != -100).all()


def test_index_out_of_range():
    dataset = TokenBlockDataset(TOKENS, BLOCK)
    with pytest.raises(IndexError):
        dataset[len(dataset)]
    with pytest.raises(IndexError):
        dataset[-1]


@pytest.mark.parametrize("dtype", ["uint16", "uint32"])
def test_reads_memmap_file_with_meta(tmp_path, dtype):
    """Файл + <path>.json дают те же блоки, что массив в памяти; dtype берётся из json."""
    path = tmp_path / "train.bin"
    np.asarray(TOKENS, dtype=np.dtype(dtype)).tofile(path)
    (tmp_path / "train.bin.json").write_text(json.dumps({"dtype": dtype, "num_tokens": len(TOKENS)}))

    from_file = TokenBlockDataset(str(path), BLOCK)
    from_list = TokenBlockDataset(TOKENS, BLOCK)
    assert len(from_file) == len(from_list)
    for i in range(len(from_file)):
        assert torch.equal(from_file[i]["input_ids"], from_list[i]["input_ids"])
    assert from_file[0]["input_ids"].dtype == torch.long


def test_file_without_meta_requires_dtype(tmp_path):
    path = tmp_path / "train.bin"
    np.asarray(TOKENS, dtype=np.uint16).tofile(path)
    with pytest.raises(ValueError, match="dtype"):
        TokenBlockDataset(str(path), BLOCK)
    assert len(TokenBlockDataset(str(path), BLOCK, dtype="uint16")) == 4


def test_too_short_corpus_and_bad_block_size():
    with pytest.raises(ValueError, match="block_size"):
        TokenBlockDataset(TOKENS[:4], BLOCK)
    with pytest.raises(ValueError, match="block_size"):
        TokenBlockDataset(TOKENS, 0)
    with pytest.raises(ValueError, match="одномерный"):
        TokenBlockDataset(np.zeros((2, 5), dtype=np.int64), BLOCK)


def test_dataloader_batches_without_collate_tricks():
    """Стандартный DataLoader собирает батч [batch, block] — коллатор не нужен."""
    loader = DataLoader(TokenBlockDataset(TOKENS, BLOCK), batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (2, BLOCK)
    assert batch["input_ids"][1, 0].item() == TOKENS[BLOCK]
