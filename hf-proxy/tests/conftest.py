"""
Pytest configuration for hf-proxy tests.
"""

import os

# Тесты не должны ходить в сеть: всё, что обращается к Hub, мокается.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest
import torch

from llm.models.gpt import GPT
from llm.tokenizers import BPETokenizer
from hf_proxy import HFAdapterConfig, HFPretrainedConfig, create_hf_tokenizer


@pytest.fixture
def llm_config():
    """Маленькая конфигурация GPT, чтобы тесты шли быстро."""
    return {
        "vocab_size": 50,
        "embed_dim": 16,
        "num_heads": 2,
        "num_layers": 2,
        "max_position_embeddings": 32,
        "dropout": 0.0,
    }


@pytest.fixture
def gpt_model(llm_config):
    torch.manual_seed(0)
    model = GPT(llm_config)
    model.eval()
    return model


@pytest.fixture
def adapter_config(llm_config):
    """HFAdapterConfig, совпадающий с llm_config."""
    return HFAdapterConfig(
        vocab_size=llm_config["vocab_size"],
        hidden_size=llm_config["embed_dim"],
        num_attention_heads=llm_config["num_heads"],
        num_hidden_layers=llm_config["num_layers"],
        max_position_embeddings=llm_config["max_position_embeddings"],
        hidden_dropout_prob=llm_config["dropout"],
    )


@pytest.fixture
def pretrained_config(adapter_config):
    return HFPretrainedConfig(**adapter_config.to_dict())


@pytest.fixture
def input_ids(llm_config):
    torch.manual_seed(0)
    return torch.randint(0, llm_config["vocab_size"], (2, 8))


@pytest.fixture
def bpe_tokenizer():
    tokenizer = BPETokenizer()
    tokenizer.train(
        texts=["hello world", "hello there", "world of words"],
        vocab_size=30,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )
    return tokenizer


@pytest.fixture
def hf_tokenizer(bpe_tokenizer):
    return create_hf_tokenizer(bpe_tokenizer)
