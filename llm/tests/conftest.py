"""
Pytest configuration for llm tests.
"""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    """Return the device to run tests on."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Return a standard batch size for tests."""
    return 2


@pytest.fixture
def seq_len():
    """Return a standard sequence length for tests."""
    return 64


@pytest.fixture
def vocab_size():
    """Return a standard vocabulary size for tests."""
    return 1000


@pytest.fixture
def embed_dim():
    """Return a standard embedding dimension for tests."""
    return 256


@pytest.fixture
def num_heads():
    """Return a standard number of attention heads."""
    return 4


@pytest.fixture
def num_layers():
    """Return a standard number of layers."""
    return 2


@pytest.fixture
def gpt_config(vocab_size, embed_dim, num_heads, num_layers):
    """Return a standard GPT configuration for tests."""
    return {
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "max_position_embeddings": 1024,
        "dropout": 0.1
    }


@pytest.fixture
def random_inputs(batch_size, seq_len, vocab_size):
    """Generate random input tensors for testing."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    return input_ids

@pytest.fixture
def random_float_inputs(batch_size, seq_len, embed_dim):
    """Generate random floating point input tensors for testing feed forward."""
    inputs = torch.randn(batch_size, seq_len, embed_dim)
    return inputs

@pytest.fixture
def random_embeddings(batch_size, seq_len, embed_dim):
    """Generate random embedding tensors for testing attention modules."""
    embeddings = torch.randn(batch_size, seq_len, embed_dim)
    return embeddings


@pytest.fixture
def attention_mask(batch_size, seq_len):
    """Generate a random attention mask for testing."""
    mask = torch.ones(batch_size, seq_len)
    # Randomly mask some positions
    for i in range(batch_size):
        mask_positions = torch.randint(1, seq_len, (1,)).item()
        mask[i, mask_positions:] = 0
    return mask


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
