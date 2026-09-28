"""
Tests for HFUtils, TokenizerWrapper and create_hf_pipeline.

Всё, что обращается к HuggingFace Hub, подменяется моками.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import huggingface_hub
import pytest
import torch
import transformers

from hf_proxy import (
    HFAdapter,
    HFGPTAdapter,
    HFPretrainedConfig,
    HFTokenizerAdapter,
    HFUtils,
    TokenizerWrapper,
    create_hf_pipeline,
)
from hf_proxy import hf_utils


class TestCreateHFConfig:
    def test_from_llm_config(self, llm_config):
        config = HFUtils.create_hf_config_from_llm(llm_config)
        assert isinstance(config, HFPretrainedConfig)
        assert config.hidden_size == llm_config["embed_dim"]
        assert config.num_hidden_layers == llm_config["num_layers"]
        assert config.num_attention_heads == llm_config["num_heads"]
        assert config.intermediate_size == 4 * llm_config["embed_dim"]


class TestConvertToHFFormat:
    def test_wraps_bpe_tokenizer(self, gpt_model, bpe_tokenizer):
        model, tokenizer = HFUtils.convert_to_hf_format(gpt_model, bpe_tokenizer)
        assert isinstance(model, HFGPTAdapter)
        assert model.llm_model is gpt_model
        assert isinstance(tokenizer, HFTokenizerAdapter)
        assert tokenizer.llm_tokenizer is bpe_tokenizer

    def test_passes_through_other_tokenizer(self, gpt_model):
        tokenizer = object()
        _, result = HFUtils.convert_to_hf_format(gpt_model, tokenizer)
        assert result is tokenizer

    def test_default_tokenizer_is_gpt2(self, gpt_model, monkeypatch):
        fake = SimpleNamespace(pad_token=None, eos_token="<|endoftext|>")
        from_pretrained = MagicMock(return_value=fake)
        monkeypatch.setattr(
            transformers, "AutoTokenizer", SimpleNamespace(from_pretrained=from_pretrained)
        )

        _, tokenizer = HFUtils.convert_to_hf_format(gpt_model)

        from_pretrained.assert_called_once_with("gpt2")
        assert tokenizer is fake
        assert tokenizer.pad_token == "<|endoftext|>"

    def test_default_tokenizer_keeps_pad_token(self, gpt_model, monkeypatch):
        fake = SimpleNamespace(pad_token="<pad>", eos_token="<eos>")
        monkeypatch.setattr(
            transformers,
            "AutoTokenizer",
            SimpleNamespace(from_pretrained=MagicMock(return_value=fake)),
        )
        _, tokenizer = HFUtils.convert_to_hf_format(gpt_model)
        assert tokenizer.pad_token == "<pad>"


class TestPushToHub:
    @pytest.fixture
    def hub(self, monkeypatch):
        """Мок huggingface_hub; запоминает файлы, которые ушли бы в upload."""
        uploaded = {}

        def upload_folder(folder_path, repo_id, commit_message):
            uploaded["files"] = sorted(os.listdir(folder_path))
            uploaded["repo_id"] = repo_id

        api = MagicMock()
        api.upload_folder.side_effect = upload_folder
        card = MagicMock()
        card.save.side_effect = lambda path: open(path, "w").close()

        mocks = SimpleNamespace(
            create_repo=MagicMock(),
            ModelCard=MagicMock(),
            HfApi=MagicMock(return_value=api),
            card=card,
            uploaded=uploaded,
        )
        mocks.ModelCard.from_template.return_value = card
        monkeypatch.setattr(huggingface_hub, "create_repo", mocks.create_repo)
        monkeypatch.setattr(huggingface_hub, "ModelCard", mocks.ModelCard)
        monkeypatch.setattr(huggingface_hub, "HfApi", mocks.HfApi)
        return mocks

    def test_uploads_model(self, gpt_model, hf_tokenizer, hub):
        model = HFAdapter.from_llm_model(gpt_model)
        HFUtils.push_to_hub(model, hf_tokenizer, "my-model")

        hub.create_repo.assert_called_once_with("my-model", private=False, exist_ok=True)
        assert hub.uploaded["repo_id"] == "my-model"
        assert {"config.json", "pytorch_model.bin", "README.md"} <= set(
            hub.uploaded["files"]
        )

    def test_organization_and_private(self, gpt_model, hf_tokenizer, hub):
        model = HFAdapter.from_llm_model(gpt_model)
        HFUtils.push_to_hub(model, hf_tokenizer, "my-model", organization="org", private=True)

        hub.create_repo.assert_called_once_with("org/my-model", private=True, exist_ok=True)
        assert hub.uploaded["repo_id"] == "org/my-model"

    def test_missing_huggingface_hub(self, gpt_model, hf_tokenizer, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        model = HFAdapter.from_llm_model(gpt_model)
        with pytest.raises(ImportError, match="huggingface_hub"):
            HFUtils.push_to_hub(model, hf_tokenizer, "my-model")


class TestLoadFromHub:
    def test_loads_from_local_repo(
        self, gpt_model, llm_config, input_ids, tmp_path, monkeypatch
    ):
        model = HFAdapter.from_llm_model(gpt_model)
        HFAdapter.save_pretrained(model, str(tmp_path))

        fake_tokenizer = object()
        monkeypatch.setattr(
            transformers,
            "AutoTokenizer",
            SimpleNamespace(from_pretrained=MagicMock(return_value=fake_tokenizer)),
        )
        hf_config = SimpleNamespace(
            vocab_size=llm_config["vocab_size"],
            hidden_size=llm_config["embed_dim"],
            num_attention_heads=llm_config["num_heads"],
            num_hidden_layers=llm_config["num_layers"],
            max_position_embeddings=llm_config["max_position_embeddings"],
            hidden_dropout_prob=llm_config["dropout"],
        )
        monkeypatch.setattr(
            hf_utils,
            "AutoConfig",
            SimpleNamespace(from_pretrained=MagicMock(return_value=hf_config)),
        )

        loaded, tokenizer = HFUtils.load_from_hub(str(tmp_path))

        assert tokenizer is fake_tokenizer
        assert isinstance(loaded, HFGPTAdapter)
        loaded.eval()
        with torch.no_grad():
            assert torch.allclose(loaded(input_ids).logits, model(input_ids).logits)


class TestCompareWithHFModel:
    @pytest.fixture
    def reference(self, llm_config, input_ids, monkeypatch):
        """Подменяет эталонную HF-модель и её токенизатор."""
        torch.manual_seed(1)
        logits = torch.randn(1, input_ids.size(1), llm_config["vocab_size"])

        tokenizer = MagicMock(return_value={"input_ids": input_ids[:1]})
        model = MagicMock(return_value=SimpleNamespace(logits=logits))
        monkeypatch.setattr(
            transformers,
            "AutoTokenizer",
            SimpleNamespace(from_pretrained=MagicMock(return_value=tokenizer)),
        )
        monkeypatch.setattr(
            transformers,
            "AutoModelForCausalLM",
            SimpleNamespace(from_pretrained=MagicMock(return_value=model)),
        )
        return logits

    def test_identical_models(self, reference):
        result = HFUtils.compare_with_hf_model(lambda ids: reference.clone())

        assert result["kl_divergence"] == pytest.approx(0.0, abs=1e-5)
        assert result["cosine_similarity"] == pytest.approx(1.0, abs=1e-5)
        assert result["hf_top_tokens"] == result["llm_top_tokens"]
        assert len(result["hf_top_tokens"]) == 5

    def test_different_models(self, reference):
        result = HFUtils.compare_with_hf_model(lambda ids: -reference)
        assert result["kl_divergence"] > 0
        assert result["cosine_similarity"] == pytest.approx(-1.0, abs=1e-5)

    @pytest.mark.xfail(
        strict=True,
        reason="GPT.forward возвращает (logits, cache), а compare_with_hf_model "
        "индексирует результат как тензор",
    )
    def test_with_llm_gpt(self, gpt_model, reference):
        result = HFUtils.compare_with_hf_model(gpt_model)
        assert "kl_divergence" in result


class TestTokenizerWrapper:
    def test_encode_batch_passes_hf_options(self):
        tokenizer = MagicMock(return_value="encoded")
        wrapper = TokenizerWrapper(tokenizer)

        assert wrapper.encode_batch(["a", "b"], max_length=4) == "encoded"
        tokenizer.assert_called_once_with(
            ["a", "b"], padding=True, truncation=True, return_tensors="pt", max_length=4
        )

    def test_decode_batch_2d(self, hf_tokenizer):
        wrapper = TokenizerWrapper(hf_tokenizer)
        texts = ["hello", "hello world"]
        max_length = max(len(hf_tokenizer.encode(t)) for t in texts)
        ids = torch.tensor(
            [hf_tokenizer.encode(t, padding=True, max_length=max_length) for t in texts]
        )
        assert wrapper.decode_batch(ids) == texts

    def test_decode_batch_1d(self, hf_tokenizer):
        wrapper = TokenizerWrapper(hf_tokenizer)
        assert wrapper.decode_batch(torch.tensor(hf_tokenizer.encode("hello"))) == ["hello"]

    def test_get_vocab_size(self, hf_tokenizer):
        assert TokenizerWrapper(hf_tokenizer).get_vocab_size() == hf_tokenizer.vocab_size

    def test_get_special_tokens(self, hf_tokenizer):
        assert TokenizerWrapper(hf_tokenizer).get_special_tokens() == {
            "pad_token": hf_tokenizer.pad_token_id,
            "eos_token": hf_tokenizer.eos_token_id,
            "bos_token": hf_tokenizer.bos_token_id,
            "unk_token": hf_tokenizer.unk_token_id,
        }


def test_create_hf_pipeline(gpt_model, bpe_tokenizer, monkeypatch):
    pipeline = MagicMock(return_value="pipe")
    monkeypatch.setattr(transformers, "pipeline", pipeline)

    assert create_hf_pipeline(gpt_model, bpe_tokenizer, device="cpu", batch_size=2) == "pipe"

    args, kwargs = pipeline.call_args
    assert args == ("text-generation",)
    assert isinstance(kwargs["model"], HFGPTAdapter)
    assert isinstance(kwargs["tokenizer"], HFTokenizerAdapter)
    assert kwargs["device"] == "cpu"
    assert kwargs["batch_size"] == 2
