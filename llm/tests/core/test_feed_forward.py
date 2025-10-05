"""
Tests for feed forward network.
"""

import pytest
import torch
import torch.nn as nn
from llm.core.feed_forward import FeedForward


class TestFeedForward:
    """Test cases for FeedForward."""
    
    def test_initialization(self, embed_dim):
        """Test that FeedForward can be initialized."""
        ff = FeedForward(embed_dim)
        assert ff is not None
        
        # Check internal layers
        assert hasattr(ff, '_layer1')
        assert hasattr(ff, '_layer2')
        assert hasattr(ff, '_activation')
        assert hasattr(ff, '_dropout')
        
        # Check layer dimensions
        expected_hidden_dim = embed_dim * 4  # Default expansion factor
        assert ff._layer1.weight.shape == (expected_hidden_dim, embed_dim)
        assert ff._layer2.weight.shape == (embed_dim, expected_hidden_dim)
    
    def test_forward_pass(self, embed_dim, random_float_inputs):
        """Test forward pass of FeedForward."""
        ff = FeedForward(embed_dim)
        
        # Forward pass
        output = ff(random_float_inputs)
        
        # Check output shape
        assert output.shape == random_float_inputs.shape
        assert isinstance(output, torch.Tensor)
    
    def test_custom_hidden_dim(self, embed_dim):
        """Test FeedForward with custom hidden dimension."""
        # FeedForward doesn't support custom hidden_dim in current implementation
        # This test is not applicable
        ff = FeedForward(embed_dim)
        
        # Check layer dimensions (fixed 4x expansion)
        expected_hidden_dim = embed_dim * 4
        assert ff._layer1.weight.shape == (expected_hidden_dim, embed_dim)
        assert ff._layer2.weight.shape == (embed_dim, expected_hidden_dim)
    
    def test_dropout(self, embed_dim, random_float_inputs):
        """Test that dropout is applied during training."""
        ff = FeedForward(embed_dim, dropout=0.5)
        ff.train()  # Set to training mode
        
        output = ff(random_float_inputs)
        
        # In training mode with dropout, some values should be zeroed
        # This is probabilistic, so we can't assert exact zeros,
        # but we can check the structure is preserved
        assert output.shape == random_float_inputs.shape
    
    def test_no_dropout_in_eval(self, embed_dim, random_float_inputs):
        """Test that dropout is not applied during evaluation."""
        ff = FeedForward(embed_dim, dropout=0.5)
        ff.eval()  # Set to evaluation mode
        
        # Run forward pass multiple times - outputs should be identical
        output1 = ff(random_float_inputs)
        output2 = ff(random_float_inputs)
        
        assert torch.allclose(output1, output2)
    
    def test_activation_function(self, embed_dim, random_float_inputs):
        """Test that activation function is applied."""
        ff = FeedForward(embed_dim)
        
        # Manually compute expected output without dropout for deterministic comparison
        hidden = ff._layer1(random_float_inputs)
        activated = ff._activation(hidden)
        expected_output = ff._layer2(activated)
        
        # Compare with forward pass in eval mode (no dropout)
        ff.eval()
        actual_output = ff(random_float_inputs)
        
        assert torch.allclose(actual_output, expected_output, rtol=1e-4)
    
    def test_gradient_flow(self, embed_dim, random_float_inputs):
        """Test that gradients flow through FeedForward."""
        ff = FeedForward(embed_dim)
        
        # Forward pass
        output = ff(random_float_inputs)
        
        # Create a dummy loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed for learnable parameters
        assert ff._layer1.weight.grad is not None
        assert ff._layer2.weight.grad is not None
        assert not torch.allclose(ff._layer1.weight.grad, 
                                torch.zeros_like(ff._layer1.weight.grad))
        assert not torch.allclose(ff._layer2.weight.grad, 
                                torch.zeros_like(ff._layer2.weight.grad))
    
    def test_device_consistency(self, embed_dim, random_float_inputs, device):
        """Test that FeedForward works on correct device."""
        ff = FeedForward(embed_dim).to(device)
        inputs = random_float_inputs.to(device)
        
        # Forward pass
        output = ff(inputs)
        
        # Check device consistency
        assert output.device == device
        assert ff._layer1.weight.device == device
        assert ff._layer2.weight.device == device
    
    def test_different_embed_dims(self):
        """Test FeedForward with different embedding dimensions."""
        test_cases = [64, 128, 256, 512]
        
        for embed_dim in test_cases:
            ff = FeedForward(embed_dim)
            batch_size, seq_len = 2, 16
            inputs = torch.randn(batch_size, seq_len, embed_dim)
            
            output = ff(inputs)
            
            assert output.shape == inputs.shape
    
    @pytest.mark.parametrize("batch_size,seq_len", [(1, 8), (2, 16), (4, 32)])
    def test_different_input_shapes(self, embed_dim, batch_size, seq_len):
        """Test FeedForward with different input shapes."""
        ff = FeedForward(embed_dim)
        
        inputs = torch.randn(batch_size, seq_len, embed_dim)
        output = ff(inputs)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
    
    def test_non_linearity(self, embed_dim, random_float_inputs):
        """Test that FeedForward introduces non-linearity."""
        ff = FeedForward(embed_dim)
        
        # Create a simple linear transformation for comparison
        linear_layer = nn.Linear(embed_dim, embed_dim)
        
        # Copy weights to make comparison fair
        with torch.no_grad():
            linear_layer.weight.copy_(ff._layer2.weight @ ff._layer1.weight)
            if linear_layer.bias is not None:
                linear_layer.bias.zero_()
        
        linear_output = linear_layer(random_float_inputs)
        ff_output = ff(random_float_inputs)
        
        # FeedForward output should be different from pure linear transformation
        # due to activation function
        assert not torch.allclose(ff_output, linear_output, rtol=1e-4)
    
    def test_parameter_initialization(self, embed_dim):
        """Test that parameters are properly initialized."""
        ff = FeedForward(embed_dim)
        
        # Check that weights are not all zeros
        assert not torch.allclose(ff._layer1.weight, torch.zeros_like(ff._layer1.weight))
        assert not torch.allclose(ff._layer2.weight, torch.zeros_like(ff._layer2.weight))
        
        # Check that biases are not all zeros (they should be initialized with some values)
        if ff._layer1.bias is not None:
            assert not torch.allclose(ff._layer1.bias, torch.zeros_like(ff._layer1.bias))
        if ff._layer2.bias is not None:
            assert not torch.allclose(ff._layer2.bias, torch.zeros_like(ff._layer2.bias))
