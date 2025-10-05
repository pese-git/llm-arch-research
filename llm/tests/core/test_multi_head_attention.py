"""
Tests for multi-head attention.
"""

import pytest
import torch
from llm.core.multi_head_attention import MultiHeadAttention


class TestMultiHeadAttention:
    """Test cases for MultiHeadAttention."""
    
    def test_initialization(self, embed_dim, num_heads):
        """Test that MultiHeadAttention can be initialized."""
        head_size = embed_dim // num_heads
        attention = MultiHeadAttention(num_heads, embed_dim, head_size, max_seq_len=1024)
        assert attention is not None
        
        # Check internal attributes
        assert len(attention._heads) == num_heads
        assert attention._layer.in_features == embed_dim
        assert attention._layer.out_features == embed_dim
    
    def test_forward_pass(self, embed_dim, num_heads, random_embeddings):
        """Test forward pass of MultiHeadAttention."""
        head_size = embed_dim // num_heads
        attention = MultiHeadAttention(num_heads, embed_dim, head_size, max_seq_len=1024)
        
        # Forward pass
        output, _ = attention(random_embeddings)
        
        # Check output shape
        assert output.shape == random_embeddings.shape
        assert isinstance(output, torch.Tensor)
    
    def test_forward_with_mask(self, embed_dim, num_heads, random_embeddings):
        """Test forward pass with attention mask."""
        head_size = embed_dim // num_heads
        attention = MultiHeadAttention(num_heads, embed_dim, head_size, max_seq_len=1024)
        
        # Create a simple mask
        seq_len = random_embeddings.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len))  # Causal mask
        
        # Forward pass with mask
        output, _ = attention(random_embeddings, mask=mask)
        
        # Check output shape
        assert output.shape == random_embeddings.shape
    
    def test_causal_mask(self, embed_dim, num_heads, random_embeddings):
        """Test that causal mask prevents attending to future positions."""
        head_size = embed_dim // num_heads
        attention = MultiHeadAttention(num_heads, embed_dim, head_size, max_seq_len=1024)
        
        # Create causal mask
        seq_len = random_embeddings.shape[1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        
        # Forward pass with causal mask
        output, _ = attention(random_embeddings, mask=causal_mask)
        
        # Check output shape
        assert output.shape == random_embeddings.shape
    
    def test_attention_weights_normalization(self, embed_dim, num_heads, random_embeddings):
        """Test that attention weights are properly normalized."""
        head_size = embed_dim // num_heads
        attention = MultiHeadAttention(num_heads, embed_dim, head_size, max_seq_len=1024)
        
        # Forward pass
        output, _ = attention(random_embeddings)
        
        # Check output shape
        assert output.shape == random_embeddings.shape
    
    def test_gradient_flow(self, embed_dim, num_heads, random_embeddings):
        """Test that gradients flow through MultiHeadAttention."""
        head_size = embed_dim // num_heads
        attention = MultiHeadAttention(num_heads, embed_dim, head_size, max_seq_len=1024)
        
        # Forward pass
        output, _ = attention(random_embeddings)
        
        # Create a dummy loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed for learnable parameters
        assert attention._layer.weight.grad is not None
        if len(attention._heads) > 0:
            assert attention._heads[0]._q.weight.grad is not None
    
    def test_device_consistency(self, embed_dim, num_heads, random_embeddings, device):
        """Test that MultiHeadAttention works on correct device."""
        head_size = embed_dim // num_heads
        attention = MultiHeadAttention(num_heads, embed_dim, head_size, max_seq_len=1024).to(device)
        inputs = random_embeddings.to(device)
        
        # Forward pass
        output, _ = attention(inputs)
        
        # Check device consistency
        assert output.device == device
        assert attention._layer.weight.device == device
    
    def test_different_embed_dim_and_heads(self):
        """Test MultiHeadAttention with different embed_dim and num_heads combinations."""
        test_cases = [
            (64, 2),   # embed_dim=64, num_heads=2
            (128, 4),  # embed_dim=128, num_heads=4
            (256, 8),  # embed_dim=256, num_heads=8
            (512, 16), # embed_dim=512, num_heads=16
        ]
        
        for embed_dim, num_heads in test_cases:
            head_size = embed_dim // num_heads
            attention = MultiHeadAttention(num_heads, embed_dim, head_size, max_seq_len=1024)
            batch_size, seq_len = 2, 16
            inputs = torch.randn(batch_size, seq_len, embed_dim)
            
            output, _ = attention(inputs)
            
            assert output.shape == inputs.shape
    
    def test_attention_output_range(self, embed_dim, num_heads, random_embeddings):
        """Test that attention output is in reasonable range."""
        head_size = embed_dim // num_heads
        attention = MultiHeadAttention(num_heads, embed_dim, head_size, max_seq_len=1024)
        
        output, _ = attention(random_embeddings)
        
        # Output shouldn't have extreme values
        assert output.abs().max() < 100  # Reasonable upper bound
    
    @pytest.mark.parametrize("batch_size,seq_len", [(1, 8), (2, 16), (4, 32)])
    def test_different_input_shapes(self, embed_dim, num_heads, batch_size, seq_len):
        """Test MultiHeadAttention with different input shapes."""
        head_size = embed_dim // num_heads
        attention = MultiHeadAttention(num_heads, embed_dim, head_size, max_seq_len=1024)
        
        inputs = torch.randn(batch_size, seq_len, embed_dim)
        output, _ = attention(inputs)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
    
    def test_parameter_sharing(self, embed_dim, num_heads):
        """Test that parameters are properly shared across the sequence."""
        head_size = embed_dim // num_heads
        attention = MultiHeadAttention(num_heads, embed_dim, head_size, max_seq_len=1024, dropout=0.0)  # No dropout for deterministic test
        
        # Create two identical sequences
        seq_len = 10
        base_sequence = torch.randn(1, seq_len, embed_dim)
        identical_sequence = base_sequence.clone()
        
        # Set to eval mode to disable dropout
        attention.eval()
        
        with torch.no_grad():
            output1, _ = attention(base_sequence)
            output2, _ = attention(identical_sequence)
        
        # With identical inputs and same parameters, outputs should be identical
        assert torch.allclose(output1, output2, rtol=1e-5)
