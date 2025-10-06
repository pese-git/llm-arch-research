"""
Tests for token embeddings.
"""

import pytest
import torch
from llm.core.token_embeddings import TokenEmbeddings


class TestTokenEmbeddings:
    """Test cases for TokenEmbeddings."""

    def test_initialization(self, vocab_size, embed_dim):
        """Test that TokenEmbeddings can be initialized."""
        embeddings = TokenEmbeddings(vocab_size, embed_dim)
        assert embeddings is not None

        # Check embedding layer
        assert hasattr(embeddings, "_embedding")
        assert embeddings._embedding.weight.shape == (vocab_size, embed_dim)

    def test_forward_pass(self, vocab_size, embed_dim, random_inputs):
        """Test forward pass of TokenEmbeddings."""
        embeddings = TokenEmbeddings(vocab_size, embed_dim)

        # Forward pass
        output = embeddings(random_inputs)

        # Check output shape
        assert output.shape == (
            random_inputs.shape[0],
            random_inputs.shape[1],
            embed_dim,
        )
        assert isinstance(output, torch.Tensor)

    def test_embedding_weights(self, vocab_size, embed_dim):
        """Test that embedding weights are properly initialized."""
        embeddings = TokenEmbeddings(vocab_size, embed_dim)

        weights = embeddings._embedding.weight
        assert weights.requires_grad is True

        # Check that weights are not all zeros
        assert not torch.allclose(weights, torch.zeros_like(weights))

    def test_different_vocab_sizes(self):
        """Test TokenEmbeddings with different vocabulary sizes."""
        test_cases = [(100, 128), (1000, 256), (50000, 512)]

        for vocab_size, embed_dim in test_cases:
            embeddings = TokenEmbeddings(vocab_size, embed_dim)
            assert embeddings._embedding.weight.shape == (vocab_size, embed_dim)

    def test_gradient_flow(self, vocab_size, embed_dim, random_inputs):
        """Test that gradients flow through TokenEmbeddings."""
        embeddings = TokenEmbeddings(vocab_size, embed_dim)

        # Forward pass
        output = embeddings(random_inputs)

        # Create a dummy loss and backward pass
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        assert embeddings._embedding.weight.grad is not None
        assert not torch.allclose(
            embeddings._embedding.weight.grad,
            torch.zeros_like(embeddings._embedding.weight.grad),
        )

    def test_device_consistency(self, vocab_size, embed_dim, random_inputs, device):
        """Test that TokenEmbeddings works on correct device."""
        embeddings = TokenEmbeddings(vocab_size, embed_dim).to(device)
        inputs = random_inputs.to(device)

        # Forward pass
        output = embeddings(inputs)

        # Check device consistency
        assert output.device == device
        assert embeddings._embedding.weight.device == device

    def test_embedding_lookup(self, vocab_size, embed_dim):
        """Test specific embedding lookups."""
        embeddings = TokenEmbeddings(vocab_size, embed_dim)

        # Test lookup for specific tokens
        test_tokens = torch.tensor(
            [[0, 1, 2], [vocab_size - 1, vocab_size - 2, vocab_size - 3]]
        )

        output = embeddings(test_tokens)

        # Check shape
        assert output.shape == (2, 3, embed_dim)

        # Check that different tokens have different embeddings
        # (with high probability due to random initialization)
        assert not torch.allclose(output[0, 0], output[0, 1], rtol=1e-4)

    @pytest.mark.parametrize("batch_size,seq_len", [(1, 1), (2, 10), (8, 64)])
    def test_different_input_shapes(self, vocab_size, embed_dim, batch_size, seq_len):
        """Test TokenEmbeddings with different input shapes."""
        embeddings = TokenEmbeddings(vocab_size, embed_dim)

        inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = embeddings(inputs)

        assert output.shape == (batch_size, seq_len, embed_dim)
