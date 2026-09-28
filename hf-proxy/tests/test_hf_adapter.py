"""
Tests for HFGPTAdapter and HFAdapter.
"""

import json
import os
from unittest.mock import MagicMock

import pytest
import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from llm.models.gpt import GPT
from hf_proxy import HFAdapter, HFGPTAdapter, HFPretrainedConfig


class TestHFGPTAdapter:
    def test_is_pretrained_model(self, pretrained_config, gpt_model):
        adapter = HFGPTAdapter(pretrained_config, gpt_model)
        assert isinstance(adapter, PreTrainedModel)
        assert adapter.llm_model is gpt_model

    def test_creates_llm_model_from_config(self, pretrained_config):
        adapter = HFGPTAdapter(pretrained_config)
        assert isinstance(adapter.llm_model, GPT)
        assert len(adapter.llm_model._decoders) == pretrained_config.num_hidden_layers

    def test_hf_to_llm_config(self, pretrained_config, gpt_model, llm_config):
        adapter = HFGPTAdapter(pretrained_config, gpt_model)
        assert adapter._hf_to_llm_config(pretrained_config) == llm_config

    def test_loads_state_dict_from_config(self, pretrained_config, gpt_model):
        pretrained_config.state_dict = gpt_model.state_dict()
        adapter = HFGPTAdapter(pretrained_config)
        for key, value in gpt_model.state_dict().items():
            assert torch.equal(adapter.llm_model.state_dict()[key], value)

    def test_forward_returns_logits(self, pretrained_config, gpt_model, input_ids):
        adapter = HFGPTAdapter(pretrained_config, gpt_model)
        out = adapter(input_ids)
        assert isinstance(out, CausalLMOutputWithCrossAttentions)
        assert out.loss is None
        assert out.logits.shape == (2, 8, pretrained_config.vocab_size)
        assert out.past_key_values is None

    def test_forward_matches_llm_model(self, pretrained_config, gpt_model, input_ids):
        adapter = HFGPTAdapter(pretrained_config, gpt_model)
        with torch.no_grad():
            expected, _ = gpt_model(input_ids)
            actual = adapter(input_ids).logits
        assert torch.allclose(actual, expected)

    def test_forward_with_labels_computes_shifted_loss(
        self, pretrained_config, gpt_model, input_ids
    ):
        adapter = HFGPTAdapter(pretrained_config, gpt_model)
        out = adapter(input_ids, labels=input_ids)

        logits = out.logits
        expected = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1)
        )
        assert out.loss.dim() == 0
        assert torch.allclose(out.loss, expected)

    def test_loss_is_differentiable(self, pretrained_config, gpt_model, input_ids):
        adapter = HFGPTAdapter(pretrained_config, gpt_model)
        adapter.train()
        adapter(input_ids, labels=input_ids).loss.backward()
        grads = [p.grad for p in adapter.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)

    def test_forward_tuple_output(self, pretrained_config, gpt_model, input_ids):
        adapter = HFGPTAdapter(pretrained_config, gpt_model)

        out = adapter(input_ids, return_dict=False)
        assert isinstance(out, tuple) and len(out) == 1

        out = adapter(input_ids, labels=input_ids, return_dict=False)
        assert len(out) == 2
        loss, logits = out
        assert loss.dim() == 0
        assert logits.shape == (2, 8, pretrained_config.vocab_size)

    def test_forward_with_tensor_output_model(
        self, pretrained_config, gpt_model, input_ids, monkeypatch
    ):
        """Модель может возвращать просто тензор логитов, а не кортеж."""
        logits = torch.randn(2, 8, pretrained_config.vocab_size)
        monkeypatch.setattr(gpt_model, "forward", MagicMock(return_value=logits))
        adapter = HFGPTAdapter(pretrained_config, gpt_model)
        assert adapter(input_ids).logits is logits

    def test_prepare_inputs_for_generation(self, pretrained_config, gpt_model, input_ids):
        adapter = HFGPTAdapter(pretrained_config, gpt_model)
        prepared = adapter.prepare_inputs_for_generation(input_ids, past_key_values=())
        assert prepared.keys() == {"input_ids"}
        assert prepared["input_ids"] is input_ids

    def test_can_generate(self, pretrained_config, gpt_model):
        assert HFGPTAdapter(pretrained_config, gpt_model).can_generate()

    def test_generate_greedy(self, pretrained_config, gpt_model, input_ids):
        adapter = HFGPTAdapter(pretrained_config, gpt_model)
        out = adapter.generate(input_ids, max_new_tokens=4, do_sample=False)
        assert out.shape == (2, 12)
        assert torch.equal(out[:, :8], input_ids)

        expected = gpt_model.generate(input_ids, max_new_tokens=4, do_sample=False)
        assert torch.equal(out, expected)

    def test_generate_defaults(self, pretrained_config, gpt_model, input_ids, monkeypatch):
        generate = MagicMock()
        monkeypatch.setattr(gpt_model, "generate", generate)
        adapter = HFGPTAdapter(pretrained_config, gpt_model)
        adapter.generate(input_ids, top_k=5)
        kwargs = generate.call_args.kwargs
        assert kwargs["max_new_tokens"] == 50
        assert kwargs["do_sample"] is True
        assert kwargs["top_k"] == 5
        assert kwargs["x"] is input_ids


class TestHFAdapter:
    def test_from_llm_model_infers_config(self, gpt_model, llm_config):
        adapter = HFAdapter.from_llm_model(gpt_model)
        assert isinstance(adapter, HFGPTAdapter)
        assert adapter.llm_model is gpt_model
        assert isinstance(adapter.config, HFPretrainedConfig)
        assert adapter.config.hidden_size == llm_config["embed_dim"]
        assert adapter.config.num_hidden_layers == llm_config["num_layers"]

    def test_from_llm_model_with_explicit_config(self, gpt_model, adapter_config):
        adapter_config.eos_token_id = 7
        adapter = HFAdapter.from_llm_model(gpt_model, adapter_config)
        assert adapter.config.eos_token_id == 7

    def test_save_pretrained_writes_files(self, gpt_model, tmp_path):
        adapter = HFAdapter.from_llm_model(gpt_model)
        save_dir = tmp_path / "model"
        HFAdapter.save_pretrained(adapter, str(save_dir))

        assert (save_dir / "config.json").is_file()
        assert (save_dir / "pytorch_model.bin").is_file()

        with open(save_dir / "config.json", encoding="utf-8") as f:
            config = json.load(f)
        assert config["hidden_size"] == adapter.config.hidden_size

        state_dict = torch.load(save_dir / "pytorch_model.bin")
        assert state_dict.keys() == gpt_model.state_dict().keys()

    def test_save_pretrained_saves_tokenizer(self, gpt_model, tmp_path):
        adapter = HFAdapter.from_llm_model(gpt_model)
        tokenizer = MagicMock()
        HFAdapter.save_pretrained(adapter, str(tmp_path), tokenizer=tokenizer)
        tokenizer.save_pretrained.assert_called_once_with(str(tmp_path))

    def test_save_and_load_roundtrip(self, gpt_model, adapter_config, input_ids, tmp_path):
        adapter = HFAdapter.from_llm_model(gpt_model, adapter_config)
        HFAdapter.save_pretrained(adapter, str(tmp_path))

        loaded = HFAdapter.from_pretrained(
            os.path.join(tmp_path, "pytorch_model.bin"), adapter_config
        )
        assert isinstance(loaded, HFGPTAdapter)
        loaded.eval()
        with torch.no_grad():
            assert torch.allclose(loaded(input_ids).logits, adapter(input_ids).logits)

    def test_from_pretrained_reads_saved_config(
        self, gpt_model, llm_config, input_ids, tmp_path
    ):
        adapter = HFAdapter.from_llm_model(gpt_model)
        HFAdapter.save_pretrained(adapter, str(tmp_path))

        loaded = HFAdapter.from_pretrained(os.path.join(tmp_path, "pytorch_model.bin"))
        assert loaded.config.num_hidden_layers == llm_config["num_layers"]
        assert loaded.config.num_attention_heads == llm_config["num_heads"]
        assert loaded.config.max_position_embeddings == llm_config["max_position_embeddings"]
        loaded.eval()
        with torch.no_grad():
            assert torch.allclose(loaded(input_ids).logits, adapter(input_ids).logits)

    def test_from_pretrained_infers_config_from_weights(self, tmp_path):
        llm_config = {
            "vocab_size": 40,
            "embed_dim": 24,
            "num_heads": 12,
            "num_layers": 3,
            "max_position_embeddings": 20,
            "dropout": 0.0,
        }
        path = tmp_path / "model.bin"
        torch.save(GPT(llm_config).state_dict(), path)

        with pytest.warns(UserWarning, match="num_attention_heads"):
            loaded = HFAdapter.from_pretrained(str(path))

        assert loaded.config.vocab_size == 40
        assert loaded.config.hidden_size == 24
        assert loaded.config.num_hidden_layers == 3
        assert loaded.config.max_position_embeddings == 20

    def test_from_pretrained_without_config_bad_heads(self, gpt_model, tmp_path):
        # embed_dim=16 не делится на 12 голов по умолчанию
        path = tmp_path / "model.bin"
        torch.save(gpt_model.state_dict(), path)
        with pytest.raises(ValueError, match="hf_config"):
            HFAdapter.from_pretrained(str(path))

    def test_from_pretrained_unknown_checkpoint(self, tmp_path):
        path = tmp_path / "model.bin"
        torch.save({"weight": torch.zeros(1)}, path)
        with pytest.raises(ValueError, match="hf_config"):
            HFAdapter.from_pretrained(str(path))
