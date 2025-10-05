"""
Tests for positional embeddings.
"""

import pytest
import torch
import math
from llm.core.positional_embeddings import PositionalEmbeddings


class TestPositionalEmbeddings:
    """Test cases for PositionalEmbeddings."""
    
    def test_initialization(self, embed_dim):
        """Test that PositionalEmbeddings can be initialized."""
        max_seq_len = 1024
        embeddings = PositionalEmbeddings(max_seq_len, embed_dim)
        assert embeddings is not None
        
        # Check that positional embeddings are created
        assert hasattr(embeddings, 'embedding')
        assert embeddings.embedding.weight.shape == (max_seq_len, embed_dim)
    
    def test_forward_pass(self, embed_dim):
        """Test forward pass of PositionalEmbeddings."""
        max_seq_len = 1024
        seq_len = 64
        embeddings = PositionalEmbeddings(max_seq_len, embed_dim)
        
        # Forward pass - takes sequence length, not input tensor
        output = embeddings(seq_len)
        
        # Check output shape
        expected_shape = (seq_len, embed_dim)
        assert output.shape == expected_shape
        assert isinstance(output, torch.Tensor)
    
    def test_positional_encoding_values(self, embed_dim):
        """Test that positional encoding values are computed correctly."""
        max_seq_len = 10
        embeddings = PositionalEmbeddings(max_seq_len, embed_dim)
        
        # Get embeddings for all positions
        pe = embeddings(max_seq_len)  # Shape: [max_seq_len, embed_dim]
        
        # Check that different positions have different embeddings
        # (since these are learnable embeddings, not fixed sine/cosine)
        for pos in range(max_seq_len):
            for i in range(pos + 1, max_seq_len):
                assert not torch.allclose(pe[pos], pe[i], rtol=1e-4)
    
    def test_different_sequence_lengths(self, embed_dim):
        """Test PositionalEmbeddings with different sequence lengths."""
        test_cases = [
            (10, 5),   # seq_len < max_seq_len
            (10, 10),  # seq_len == max_seq_len
        ]
        
        for max_seq_len, seq_len in test_cases:
            embeddings = PositionalEmbeddings(max_seq_len, embed_dim)
            
            # Get embeddings for specific sequence length
            output = embeddings(seq_len)
            
            # Output should have shape [seq_len, embed_dim]
            assert output.shape == (seq_len, embed_dim)
    
    def test_gradient_flow(self, embed_dim):
        """Test that gradients flow through PositionalEmbeddings."""
        max_seq_len = 64
        seq_len = 32
        embeddings = PositionalEmbeddings(max_seq_len, embed_dim)
        
        # Forward pass
        output = embeddings(seq_len)
        
        # Create a dummy loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Positional embeddings should have gradients (they're learnable)
        assert embeddings.embedding.weight.grad is not None
        assert not torch.allclose(embeddings.embedding.weight.grad, 
                                torch.zeros_like(embeddings.embedding.weight.grad))
        
    def test_device_consistency(self, embed_dim, device):
        """Test that PositionalEmbeddings works on correct device."""
        max_seq_len = 64
        seq_len = 32
        embeddings = PositionalEmbeddings(max_seq_len, embed_dim).to(device)
        
        # Forward pass
        output = embeddings(seq_len)
        
        # Check device consistency
        assert output.device == device
        assert embeddings.embedding.weight.device == device
    
    def test_reproducibility(self, embed_dim):
        """Test that positional embeddings are reproducible."""
        max_seq_len = 100
        embeddings1 = PositionalEmbeddings(max_seq_len, embed_dim)
        embeddings2 = PositionalEmbeddings(max_seq_len, embed_dim)
        
        # Different instances should have different embeddings (random initialization)
        assert not torch.allclose(embeddings1.embedding.weight, embeddings2.embedding.weight)
        
        # But same instance should produce same output for same input
        seq_len = 50
        output1 = embeddings1(seq_len)
        output2 = embeddings1(seq_len)  # Same instance, same input
        assert torch.allclose(output1, output2)
    
    def test_positional_pattern(self, embed_dim):
        """Test that positional embeddings create a meaningful pattern."""
        max_seq_len = 50
        embeddings = PositionalEmbeddings(max_seq_len, embed_dim)
        pe = embeddings(max_seq_len)  # Get all positional embeddings
        
        # Check that different positions have different embeddings
        # (with high probability due to random initialization)
        assert not torch.allclose(pe[0], pe[1], rtol=1e-4)
        assert not torch.allclose(pe[10], pe[20], rtol=1e-4)
    
    @pytest.mark.parametrize("max_seq_len,seq_len,embed_dim", [
        (64, 10, 64),
        (128, 50, 128), 
        (256, 100, 256),
    ])
    def test_different_configurations(self, max_seq_len, seq_len, embed_dim):
        """Test PositionalEmbeddings with different configurations."""
        embeddings = PositionalEmbeddings(max_seq_len, embed_dim)
        
        output = embeddings(seq_len)
        
        assert output.shape == (seq_len, embed_dim)
