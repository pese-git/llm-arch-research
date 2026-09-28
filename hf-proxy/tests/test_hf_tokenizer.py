"""
Tests for HFTokenizerAdapter and tokenizer helpers.
"""

import json

import numpy as np
import pytest
import torch

from hf_proxy import HFTokenizerAdapter, convert_to_hf_format, create_hf_tokenizer


class TestInit:
    def test_create_hf_tokenizer(self, bpe_tokenizer):
        adapter = create_hf_tokenizer(bpe_tokenizer)
        assert isinstance(adapter, HFTokenizerAdapter)
        assert adapter.llm_tokenizer is bpe_tokenizer

    def test_copies_vocab_and_special_tokens(self, bpe_tokenizer, hf_tokenizer):
        assert hf_tokenizer.get_vocab() == bpe_tokenizer.get_vocab()
        assert hf_tokenizer.vocab_size == bpe_tokenizer.get_vocab_size()
        assert len(hf_tokenizer) == bpe_tokenizer.get_vocab_size()
        assert hf_tokenizer.pad_token == "<pad>"
        assert hf_tokenizer.pad_token_id == bpe_tokenizer.pad_token_id
        assert hf_tokenizer.unk_token_id == bpe_tokenizer.unk_token_id
        assert hf_tokenizer.bos_token_id == bpe_tokenizer.bos_token_id
        assert hf_tokenizer.eos_token_id == bpe_tokenizer.eos_token_id


class TestCall:
    def test_single_string_is_batched(self, bpe_tokenizer, hf_tokenizer):
        out = hf_tokenizer("hello")
        assert out["input_ids"] == [bpe_tokenizer.encode("hello", add_special_tokens=True)]

    def test_batch_of_strings(self, bpe_tokenizer, hf_tokenizer):
        out = hf_tokenizer(["hello", "world"])
        assert out["input_ids"] == [
            bpe_tokenizer.encode("hello", add_special_tokens=True),
            bpe_tokenizer.encode("world", add_special_tokens=True),
        ]

    def test_without_special_tokens(self, hf_tokenizer):
        ids = hf_tokenizer("hello", add_special_tokens=False)["input_ids"][0]
        assert hf_tokenizer.bos_token_id not in ids
        assert hf_tokenizer.eos_token_id not in ids

    def test_return_tensors_pt(self, hf_tokenizer):
        ids = hf_tokenizer("hello", return_tensors="pt")["input_ids"]
        assert isinstance(ids, torch.Tensor)
        assert ids.dim() == 2 and ids.size(0) == 1

    def test_truncation(self, hf_tokenizer):
        full = hf_tokenizer.encode("hello world")
        ids = hf_tokenizer(["hello world", "hi"], truncation=True, max_length=2)["input_ids"]
        assert ids[0] == full[:2]
        assert len(ids) == 2

    def test_padding_to_max_length(self, hf_tokenizer):
        full = hf_tokenizer.encode("hello")
        ids = hf_tokenizer("hello", padding="max_length", max_length=20)["input_ids"]
        assert ids == [full + [hf_tokenizer.pad_token_id] * (20 - len(full))]

    def test_padding_to_longest(self, hf_tokenizer):
        short, long = hf_tokenizer.encode("hello"), hf_tokenizer.encode("hello world")
        ids = hf_tokenizer(["hello", "hello world"], padding=True, max_length=50)["input_ids"]
        assert ids[0] == short + [hf_tokenizer.pad_token_id] * (len(long) - len(short))
        assert ids[1] == long

    def test_padded_batch_to_tensor(self, hf_tokenizer):
        ids = hf_tokenizer(["hello", "hello world"], padding=True, return_tensors="pt")
        assert ids["input_ids"].shape == (2, len(hf_tokenizer.encode("hello world")))


class TestEncode:
    def test_matches_llm_tokenizer(self, bpe_tokenizer, hf_tokenizer):
        assert hf_tokenizer.encode("hello world") == bpe_tokenizer.encode(
            "hello world", add_special_tokens=True
        )

    def test_text_pair_appended_without_specials(self, bpe_tokenizer, hf_tokenizer):
        ids = hf_tokenizer.encode("hello", text_pair="world")
        expected = bpe_tokenizer.encode("hello", add_special_tokens=True) + bpe_tokenizer.encode(
            "world", add_special_tokens=False
        )
        assert ids == expected

    def test_truncation(self, hf_tokenizer):
        full = hf_tokenizer.encode("hello world")
        ids = hf_tokenizer.encode("hello world", truncation=True, max_length=2)
        assert ids == full[:2]

    def test_truncation_requires_max_length(self, hf_tokenizer):
        full = hf_tokenizer.encode("hello world")
        assert hf_tokenizer.encode("hello world", truncation=True) == full

    def test_padding(self, hf_tokenizer):
        full = hf_tokenizer.encode("hello")
        ids = hf_tokenizer.encode("hello", padding=True, max_length=len(full) + 3)
        assert ids == full + [hf_tokenizer.pad_token_id] * 3

    def test_return_tensors_pt(self, hf_tokenizer):
        ids = hf_tokenizer.encode("hello", return_tensors="pt")
        assert isinstance(ids, torch.Tensor)
        assert ids.tolist() == [hf_tokenizer.encode("hello")]

    def test_return_tensors_np(self, hf_tokenizer):
        ids = hf_tokenizer.encode("hello", return_tensors="np")
        assert isinstance(ids, np.ndarray)
        assert ids.tolist() == [hf_tokenizer.encode("hello")]


class TestDecode:
    def test_roundtrip(self, hf_tokenizer):
        assert hf_tokenizer.decode(hf_tokenizer.encode("hello world")) == "hello world"

    def test_skips_special_tokens(self, bpe_tokenizer, hf_tokenizer):
        ids = [hf_tokenizer.bos_token_id] + bpe_tokenizer.encode(
            "hello", add_special_tokens=False
        ) + [hf_tokenizer.eos_token_id, hf_tokenizer.pad_token_id]
        assert hf_tokenizer.decode(ids) == "hello"

    def test_keeps_special_tokens(self, bpe_tokenizer, hf_tokenizer):
        ids = hf_tokenizer.encode("hello")
        assert hf_tokenizer.decode(ids, skip_special_tokens=False) == bpe_tokenizer.decode(ids)

    def test_accepts_tensor(self, hf_tokenizer):
        ids = hf_tokenizer.encode("hello")
        assert hf_tokenizer.decode(torch.tensor(ids)) == "hello"

    def test_accepts_batched_input(self, hf_tokenizer):
        ids = hf_tokenizer.encode("hello", return_tensors="pt")
        assert hf_tokenizer.decode(ids) == "hello"
        assert hf_tokenizer.decode(ids.tolist()) == "hello"

    def test_accepts_single_int(self, bpe_tokenizer, hf_tokenizer):
        token_id = bpe_tokenizer.encode("h", add_special_tokens=False)[0]
        assert hf_tokenizer.decode(token_id) == "h"


def test_tokenize(bpe_tokenizer, hf_tokenizer):
    assert hf_tokenizer.tokenize("hello world") == bpe_tokenizer.tokenize("hello world")


class TestPad:
    def test_returns_dict_padded_to_longest(self, hf_tokenizer):
        pad = hf_tokenizer.pad_token_id
        out = hf_tokenizer.pad([{"input_ids": [5, 6, 7]}, {"input_ids": [5]}])
        assert isinstance(out, dict)
        assert out["input_ids"] == [[5, 6, 7], [5, pad, pad]]

    def test_attention_mask_by_default(self, hf_tokenizer):
        out = hf_tokenizer.pad([{"input_ids": [5, 6, 7]}, {"input_ids": [5]}])
        assert out["attention_mask"] == [[1, 1, 1], [1, 0, 0]]

    def test_without_attention_mask(self, hf_tokenizer):
        out = hf_tokenizer.pad(
            [{"input_ids": [5, 6, 7]}, {"input_ids": [5]}], return_attention_mask=False
        )
        assert "attention_mask" not in out

    def test_existing_attention_mask_extended(self, hf_tokenizer):
        out = hf_tokenizer.pad(
            [
                {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 0]},
                {"input_ids": [5], "attention_mask": [1]},
            ]
        )
        assert out["attention_mask"] == [[1, 1, 0], [1, 0, 0]]

    def test_labels_padded_with_ignore_index(self, hf_tokenizer):
        out = hf_tokenizer.pad(
            [
                {"input_ids": [5, 6, 7], "labels": [5, 6, 7]},
                {"input_ids": [5], "labels": [5]},
            ]
        )
        assert out["labels"] == [[5, 6, 7], [5, -100, -100]]

    def test_other_keys_passed_through(self, hf_tokenizer):
        out = hf_tokenizer.pad([{"input_ids": [5, 6], "id": 1}, {"input_ids": [5], "id": 2}])
        assert out["id"] == [1, 2]

    def test_dict_of_lists(self, hf_tokenizer):
        pad = hf_tokenizer.pad_token_id
        out = hf_tokenizer.pad({"input_ids": [[5, 6, 7], [5]]})
        assert out["input_ids"] == [[5, 6, 7], [5, pad, pad]]

    def test_single_example(self, hf_tokenizer):
        out = hf_tokenizer.pad({"input_ids": [5, 6]})
        assert out["input_ids"] == [[5, 6]]

    def test_empty_batch(self, hf_tokenizer):
        assert hf_tokenizer.pad([]) == {"input_ids": []}

    def test_tensor_inputs(self, hf_tokenizer):
        pad = hf_tokenizer.pad_token_id
        out = hf_tokenizer.pad(
            [{"input_ids": torch.tensor([5, 6, 7])}, {"input_ids": torch.tensor([5])}],
            return_tensors="pt",
        )
        assert out["input_ids"].tolist() == [[5, 6, 7], [5, pad, pad]]
        assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]

    def test_int_input_ids(self, hf_tokenizer):
        pad = hf_tokenizer.pad_token_id
        out = hf_tokenizer.pad([{"input_ids": [5, 6]}, {"input_ids": 5}])
        assert out["input_ids"] == [[5, 6], [5, pad]]

    def test_padding_max_length(self, hf_tokenizer):
        pad = hf_tokenizer.pad_token_id
        out = hf_tokenizer.pad(
            [{"input_ids": [5, 6]}, {"input_ids": [5]}], padding="max_length", max_length=4
        )
        assert out["input_ids"] == [[5, 6, pad, pad], [5, pad, pad, pad]]

    def test_padding_max_length_requires_max_length(self, hf_tokenizer):
        with pytest.raises(ValueError, match="max_length"):
            hf_tokenizer.pad([{"input_ids": [5]}], padding="max_length")

    def test_no_padding(self, hf_tokenizer):
        out = hf_tokenizer.pad([{"input_ids": [5, 6]}, {"input_ids": [5]}], padding=False)
        assert out["input_ids"] == [[5, 6], [5]]
        assert out["attention_mask"] == [[1, 1], [1]]

    def test_pad_to_multiple_of(self, hf_tokenizer):
        out = hf_tokenizer.pad(
            [{"input_ids": [5, 6, 7]}, {"input_ids": [5]}], pad_to_multiple_of=4
        )
        assert [len(ids) for ids in out["input_ids"]] == [4, 4]

    def test_return_tensors_pt(self, hf_tokenizer):
        out = hf_tokenizer.pad(
            [{"input_ids": [5, 6, 7]}, {"input_ids": [5]}], return_tensors="pt"
        )
        assert out["input_ids"].shape == (2, 3)
        assert out["attention_mask"].shape == (2, 3)

    def test_data_collator_for_language_modeling(self, hf_tokenizer):
        """Адаптер работает как tokenizer в коллаторе transformers."""
        from transformers import DataCollatorForLanguageModeling

        collator = DataCollatorForLanguageModeling(
            tokenizer=hf_tokenizer, mlm=False, pad_to_multiple_of=8
        )
        long_ids, short_ids = hf_tokenizer.encode("hello world"), hf_tokenizer.encode("hi")
        batch = collator(
            [
                {"input_ids": long_ids, "labels": long_ids},
                {"input_ids": short_ids, "labels": short_ids},
            ]
        )

        assert batch["input_ids"].shape == (2, 8)
        assert batch["input_ids"][1, len(short_ids):].eq(hf_tokenizer.pad_token_id).all()
        assert batch["labels"][1, len(short_ids):].eq(-100).all()
        assert batch["labels"][0, : len(long_ids)].tolist() == long_ids
        assert batch["attention_mask"][1].tolist() == [1] * len(short_ids) + [0] * (
            8 - len(short_ids)
        )


class TestSaveLoad:
    def test_save_pretrained_writes_files(self, hf_tokenizer, tmp_path):
        hf_tokenizer.save_pretrained(str(tmp_path))

        with open(tmp_path / "tokenizer_config.json", encoding="utf-8") as f:
            config = json.load(f)
        assert config["tokenizer_class"] == "HFTokenizerAdapter"
        assert config["llm_tokenizer_type"] == "BPETokenizer"
        assert config["vocab_size"] == hf_tokenizer.vocab_size
        assert config["pad_token_id"] == hf_tokenizer.pad_token_id

        with open(tmp_path / "vocab.json", encoding="utf-8") as f:
            assert json.load(f) == hf_tokenizer.get_vocab()

    def test_convert_to_hf_format(self, bpe_tokenizer, tmp_path):
        adapter = convert_to_hf_format(bpe_tokenizer, str(tmp_path))
        assert isinstance(adapter, HFTokenizerAdapter)
        assert (tmp_path / "tokenizer_config.json").is_file()
        assert (tmp_path / "vocab.json").is_file()

    def test_from_pretrained_directory(self, hf_tokenizer, tmp_path):
        hf_tokenizer.save_pretrained(str(tmp_path))
        loaded = HFTokenizerAdapter.from_pretrained(str(tmp_path))

        assert loaded.get_vocab() == hf_tokenizer.get_vocab()
        assert loaded.vocab_size == hf_tokenizer.vocab_size
        assert loaded.pad_token_id == hf_tokenizer.pad_token_id
        assert loaded.eos_token_id == hf_tokenizer.eos_token_id
        ids = hf_tokenizer.encode("hello world")
        assert loaded.decode(ids) == "hello world"

    def test_from_pretrained_directory_encode_matches(self, hf_tokenizer, tmp_path):
        hf_tokenizer.save_pretrained(str(tmp_path))
        loaded = HFTokenizerAdapter.from_pretrained(str(tmp_path))
        assert loaded.encode("hello world") == hf_tokenizer.encode("hello world")

    def test_from_pretrained_legacy_directory_without_vocab_list(
        self, hf_tokenizer, tmp_path
    ):
        """Сохранения старого формата (без vocab_list) тоже кодируются корректно."""
        hf_tokenizer.save_pretrained(str(tmp_path))
        config_path = tmp_path / "tokenizer_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        del config["vocab_list"]
        config_path.write_text(json.dumps(config), encoding="utf-8")

        loaded = HFTokenizerAdapter.from_pretrained(str(tmp_path))
        assert loaded.encode("hello world") == hf_tokenizer.encode("hello world")

    def test_from_pretrained_directory_missing_files(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            HFTokenizerAdapter.from_pretrained(str(tmp_path))

    def test_from_pretrained_unsupported_type(self, hf_tokenizer, tmp_path):
        hf_tokenizer.save_pretrained(str(tmp_path))
        config_path = tmp_path / "tokenizer_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["llm_tokenizer_type"] = "WordPiece"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        with pytest.raises(ValueError, match="WordPiece"):
            HFTokenizerAdapter.from_pretrained(str(tmp_path))

    def test_from_pretrained_llm_tokenizer_file(self, bpe_tokenizer, tmp_path):
        path = tmp_path / "tokenizer.json"
        bpe_tokenizer.save(str(path))
        loaded = HFTokenizerAdapter.from_pretrained(str(path))
        assert loaded.get_vocab() == bpe_tokenizer.get_vocab()
        assert loaded.encode("hello") == bpe_tokenizer.encode("hello", add_special_tokens=True)

    def test_from_pretrained_bad_path(self, tmp_path):
        with pytest.raises(ValueError):
            HFTokenizerAdapter.from_pretrained(str(tmp_path / "missing.json"))
