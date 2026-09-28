"""
Tests for HFAdapterConfig and HFPretrainedConfig.
"""

from transformers import PretrainedConfig

from hf_proxy import HFAdapterConfig, HFPretrainedConfig


class TestHFAdapterConfig:
    def test_defaults_match_gpt2_small(self):
        config = HFAdapterConfig()
        assert config.model_type == "gpt"
        assert config.vocab_size == 50257
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.architectures == ["GPT2LMHeadModel"]

    def test_architectures_not_shared_between_instances(self):
        a = HFAdapterConfig()
        b = HFAdapterConfig()
        a.architectures.append("Other")
        assert b.architectures == ["GPT2LMHeadModel"]

    def test_to_dict_contains_all_fields(self):
        config = HFAdapterConfig(vocab_size=100, hidden_size=32)
        d = config.to_dict()
        assert d["vocab_size"] == 100
        assert d["hidden_size"] == 32
        assert d["model_type"] == "gpt"
        assert "architectures" in d
        assert not any(k.startswith("_") for k in d)

    def test_from_llm_config_maps_keys(self, llm_config):
        config = HFAdapterConfig.from_llm_config(llm_config)
        assert config.vocab_size == llm_config["vocab_size"]
        assert config.hidden_size == llm_config["embed_dim"]
        assert config.num_hidden_layers == llm_config["num_layers"]
        assert config.num_attention_heads == llm_config["num_heads"]
        assert config.max_position_embeddings == llm_config["max_position_embeddings"]
        assert config.hidden_dropout_prob == llm_config["dropout"]

    def test_from_llm_config_sets_intermediate_size(self, llm_config):
        config = HFAdapterConfig.from_llm_config(llm_config)
        assert config.intermediate_size == 4 * llm_config["embed_dim"]

    def test_from_llm_config_partial_keeps_defaults(self):
        config = HFAdapterConfig.from_llm_config({"vocab_size": 123, "unknown": 1})
        assert config.vocab_size == 123
        assert config.hidden_size == 768
        assert config.intermediate_size == 3072


class TestHFPretrainedConfig:
    def test_is_pretrained_config(self):
        config = HFPretrainedConfig()
        assert isinstance(config, PretrainedConfig)
        assert config.model_type == "gpt"

    def test_stores_parameters(self):
        config = HFPretrainedConfig(
            vocab_size=10,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            pad_token_id=0,
            eos_token_id=1,
            bos_token_id=2,
        )
        assert config.vocab_size == 10
        assert config.hidden_size == 8
        assert config.num_hidden_layers == 1
        assert config.num_attention_heads == 2
        assert config.pad_token_id == 0
        assert config.eos_token_id == 1
        assert config.bos_token_id == 2

    def test_accepts_adapter_config_dict(self, adapter_config):
        config = HFPretrainedConfig(**adapter_config.to_dict())
        assert config.hidden_size == adapter_config.hidden_size
        assert config.num_hidden_layers == adapter_config.num_hidden_layers

    def test_to_dict_roundtrip(self, pretrained_config):
        restored = HFPretrainedConfig(**pretrained_config.to_dict())
        assert restored.vocab_size == pretrained_config.vocab_size
        assert restored.hidden_size == pretrained_config.hidden_size
        assert restored.max_position_embeddings == pretrained_config.max_position_embeddings
