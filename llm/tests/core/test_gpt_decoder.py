"""
Tests for decoder block.
"""

import pytest
import torch
from llm.core.gpt_decoder import GptDecoder


class TestGptDecoder:
    """Test cases for Decoder."""

    def test_initialization(self, embed_dim, num_heads):
        """Test that Decoder can be initialized."""
        head_size = embed_dim // num_heads
        max_seq_len = 1024
        decoder = GptDecoder(
            num_heads=num_heads,
            emb_size=embed_dim,
            head_size=head_size,
            max_seq_len=max_seq_len,
        )
        assert decoder is not None

        # Check internal components
        assert hasattr(decoder, "_heads")
        assert hasattr(decoder, "_ff")
        assert hasattr(decoder, "_norm1")
        assert hasattr(decoder, "_norm2")

    def test_forward_pass(self, embed_dim, num_heads, random_embeddings):
        """Test forward pass of Decoder."""
        head_size = embed_dim // num_heads
        max_seq_len = 1024
        decoder = GptDecoder(
            num_heads=num_heads,
            emb_size=embed_dim,
            head_size=head_size,
            max_seq_len=max_seq_len,
        )

        # Forward pass
        output, _ = decoder(random_embeddings)

        # Check output shape
        assert output.shape == random_embeddings.shape
        assert isinstance(output, torch.Tensor)

    def test_forward_with_causal_mask(self, embed_dim, num_heads, random_embeddings):
        """Test forward pass with causal mask."""
        head_size = embed_dim // num_heads
        max_seq_len = 1024
        decoder = GptDecoder(
            num_heads=num_heads,
            emb_size=embed_dim,
            head_size=head_size,
            max_seq_len=max_seq_len,
        )

        batch_size, seq_len = random_embeddings.shape[:2]
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len))

        # Forward pass with causal mask
        output, _ = decoder(random_embeddings, attention_mask=mask)

        # Check output shape
        assert output.shape == random_embeddings.shape

    def test_residual_connections(self, embed_dim, num_heads, random_embeddings):
        """Test that residual connections are properly applied."""
        head_size = embed_dim // num_heads
        max_seq_len = 1024
        decoder = GptDecoder(
            num_heads=num_heads,
            emb_size=embed_dim,
            head_size=head_size,
            max_seq_len=max_seq_len,
        )

        output, _ = decoder(random_embeddings)

        # With residual connections and layer norm, the output shouldn't be
        # too different from input (in terms of scale/distribution)
        input_norm = random_embeddings.norm(dim=-1).mean()
        output_norm = output.norm(dim=-1).mean()

        # Norms should be of similar magnitude (not exact due to transformations)
        assert 0.1 < (output_norm / input_norm) < 10.0

    def test_layer_norm(self, embed_dim, num_heads, random_embeddings):
        """Test that layer normalization is applied."""
        head_size = embed_dim // num_heads
        max_seq_len = 1024
        decoder = GptDecoder(
            num_heads=num_heads,
            emb_size=embed_dim,
            head_size=head_size,
            max_seq_len=max_seq_len,
        )

        output, _ = decoder(random_embeddings)

        # Check that output has reasonable statistics (due to layer norm)
        # Mean should be close to 0, std close to 1 for each sequence position
        output_mean = output.mean(dim=-1)
        output_std = output.std(dim=-1)

        # These are approximate checks since the data goes through multiple transformations
        assert torch.allclose(output_mean, torch.zeros_like(output_mean), atol=1.0)
        assert torch.allclose(output_std, torch.ones_like(output_std), atol=2.0)

    def test_gradient_flow(self, embed_dim, num_heads, random_embeddings):
        """Test that gradients flow through Decoder."""
        head_size = embed_dim // num_heads
        max_seq_len = 1024
        decoder = GptDecoder(
            num_heads=num_heads,
            emb_size=embed_dim,
            head_size=head_size,
            max_seq_len=max_seq_len,
        )

        # Forward pass
        output, _ = decoder(random_embeddings)

        # Create a dummy loss and backward pass
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed for learnable parameters
        # in attention and feed forward components
        assert decoder._heads._layer.weight.grad is not None
        assert decoder._ff._layer1.weight.grad is not None
        assert decoder._norm1.weight.grad is not None
        assert decoder._norm2.weight.grad is not None

    def test_device_consistency(self, embed_dim, num_heads, random_embeddings, device):
        """Test that Decoder works on correct device."""
        head_size = embed_dim // num_heads
        max_seq_len = 1024
        decoder = GptDecoder(
            num_heads=num_heads,
            emb_size=embed_dim,
            head_size=head_size,
            max_seq_len=max_seq_len,
        ).to(device)
        inputs = random_embeddings.to(device)

        # Forward pass
        output, _ = decoder(inputs)

        # Check device consistency
        assert output.device == device
        assert decoder._heads._layer.weight.device == device

    def test_different_configurations(self):
        """Test Decoder with different configurations."""
        test_cases = [
            (64, 2),  # embed_dim=64, num_heads=2
            (128, 4),  # embed_dim=128, num_heads=4
            (256, 8),  # embed_dim=256, num_heads=8
        ]

        for embed_dim, num_heads in test_cases:
            head_size = embed_dim // num_heads
            max_seq_len = 1024
            decoder = GptDecoder(
                num_heads=num_heads,
                emb_size=embed_dim,
                head_size=head_size,
                max_seq_len=max_seq_len,
            )
            batch_size, seq_len = 2, 16
            inputs = torch.randn(batch_size, seq_len, embed_dim)

            output, _ = decoder(inputs)

            assert output.shape == inputs.shape

    @pytest.mark.parametrize("batch_size,seq_len", [(1, 8), (2, 16), (4, 32)])
    def test_different_input_shapes(self, embed_dim, num_heads, batch_size, seq_len):
        """Test Decoder with different input shapes."""
        head_size = embed_dim // num_heads
        max_seq_len = 1024
        decoder = GptDecoder(
            num_heads=num_heads,
            emb_size=embed_dim,
            head_size=head_size,
            max_seq_len=max_seq_len,
        )

        inputs = torch.randn(batch_size, seq_len, embed_dim)
        output, _ = decoder(inputs)

        assert output.shape == (batch_size, seq_len, embed_dim)

    def test_training_vs_evaluation(self, embed_dim, num_heads, random_embeddings):
        """Test that Decoder behaves differently in train vs eval mode."""
        head_size = embed_dim // num_heads
        max_seq_len = 1024
        decoder = GptDecoder(
            num_heads=num_heads,
            emb_size=embed_dim,
            head_size=head_size,
            max_seq_len=max_seq_len,
            dropout=0.5,
        )

        # Training mode
        decoder.train()
        output_train, _ = decoder(random_embeddings)

        # Evaluation mode
        decoder.eval()
        output_eval, _ = decoder(random_embeddings)

        # Outputs should be different due to dropout
        assert not torch.allclose(output_train, output_eval)

    def test_parameter_initialization(self, embed_dim, num_heads):
        """Test that parameters are properly initialized."""
        head_size = embed_dim // num_heads
        max_seq_len = 1024
        decoder = GptDecoder(
            num_heads=num_heads,
            emb_size=embed_dim,
            head_size=head_size,
            max_seq_len=max_seq_len,
        )

        # Check that various components have non-zero parameters
        assert not torch.allclose(
            decoder._heads._layer.weight, torch.zeros_like(decoder._heads._layer.weight)
        )
        assert not torch.allclose(
            decoder._ff._layer1.weight, torch.zeros_like(decoder._ff._layer1.weight)
        )
        assert not torch.allclose(
            decoder._norm1.weight, torch.zeros_like(decoder._norm1.weight)
        )
