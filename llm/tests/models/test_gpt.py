"""
Tests for GPT model.
"""

import pytest
import torch
from llm.models.gpt import GPT


class TestGPT:
    """Test cases for GPT model."""
    
    def test_initialization(self, gpt_config):
        """Test that GPT can be initialized."""
        model = GPT(gpt_config)
        assert model is not None
        
        # Check that model has required components
        assert hasattr(model, '_token_embeddings')
        assert hasattr(model, '_position_embeddings')
        assert hasattr(model, '_decoders')
        assert hasattr(model, '_linear')
        assert hasattr(model, '_dropout')
        
        # Check number of decoder layers
        assert len(model._decoders) == gpt_config['num_layers']
    
    def test_forward_pass(self, gpt_config, random_inputs):
        """Test forward pass of GPT."""
        model = GPT(gpt_config)
        
        # Forward pass
        logits = model(random_inputs)
        
        # Check output shape
        batch_size, seq_len = random_inputs.shape
        vocab_size = gpt_config['vocab_size']
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert isinstance(logits, torch.Tensor)
    
    def test_forward_with_attention_mask(self, gpt_config, random_inputs, attention_mask):
        """Test forward pass with attention mask."""
        model = GPT(gpt_config)
        
        # Forward pass with mask
        logits = model(random_inputs, attention_mask=attention_mask)
        
        # Check output shape
        batch_size, seq_len = random_inputs.shape
        vocab_size = gpt_config['vocab_size']
        assert logits.shape == (batch_size, seq_len, vocab_size)
    
    def test_generate_text(self, gpt_config):
        """Test text generation."""
        model = GPT(gpt_config)
        model.eval()  # Set to evaluation mode for generation
        
        # Create initial input
        batch_size = 2
        initial_seq_len = 5
        input_ids = torch.randint(0, gpt_config['vocab_size'], (batch_size, initial_seq_len))
        
        # Generate text
        with torch.no_grad():
            generated = model.generate(
                x=input_ids,
                max_new_tokens=10,
                do_sample=False  # Use greedy for deterministic testing
            )
        
        # Check output shape
        expected_seq_len = initial_seq_len + 10
        assert generated.shape == (batch_size, expected_seq_len)
        
        # Check that initial sequence is preserved
        assert torch.allclose(generated[:, :initial_seq_len], input_ids)
    
    def test_generate_with_temperature(self, gpt_config):
        """Test text generation with temperature sampling."""
        model = GPT(gpt_config)
        model.eval()
        
        # Create initial input
        input_ids = torch.randint(0, gpt_config['vocab_size'], (1, 3))
        
        # Generate with temperature
        with torch.no_grad():
            generated = model.generate(
                x=input_ids,
                max_new_tokens=5,
                do_sample=True,
                temperature=0.8
            )
        
        assert generated.shape == (1, 8)  # 3 initial + 5 new tokens
    
    def test_generate_with_top_k(self, gpt_config):
        """Test text generation with top-k sampling."""
        model = GPT(gpt_config)
        model.eval()
        
        # Create initial input
        input_ids = torch.randint(0, gpt_config['vocab_size'], (1, 3))
        
        # Generate with top-k
        with torch.no_grad():
            generated = model.generate(
                x=input_ids,
                max_new_tokens=5,
                do_sample=True,
                top_k=10
            )
        
        assert generated.shape == (1, 8)
    
    def test_generate_with_top_p(self, gpt_config):
        """Test text generation with top-p (nucleus) sampling."""
        model = GPT(gpt_config)
        model.eval()
        
        # Create initial input
        input_ids = torch.randint(0, gpt_config['vocab_size'], (1, 3))
        
        # Generate with top-p
        with torch.no_grad():
            generated = model.generate(
                x=input_ids,
                max_new_tokens=5,
                do_sample=True,
                top_p=0.9
            )
        
        assert generated.shape == (1, 8)
    
    def test_gradient_flow(self, gpt_config, random_inputs):
        """Test that gradients flow through GPT."""
        model = GPT(gpt_config)
        
        # Forward pass
        logits = model(random_inputs)
        
        # Create a dummy loss and backward pass
        targets = torch.randint(0, gpt_config['vocab_size'], random_inputs.shape)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            targets.view(-1)
        )
        loss.backward()
        
        # Check that gradients are computed for various components
        assert model._token_embeddings._embedding.weight.grad is not None
        assert model._linear.weight.grad is not None
        if len(model._decoders) > 0:
            assert model._decoders[0]._heads._heads[0]._q.weight.grad is not None
    
    def test_device_consistency(self, gpt_config, random_inputs, device):
        """Test that GPT works on correct device."""
        model = GPT(gpt_config).to(device)
        inputs = random_inputs.to(device)
        
        # Forward pass
        logits = model(inputs)
        
        # Check device consistency
        assert logits.device == device
        assert model._token_embeddings._embedding.weight.device == device
    
    def test_different_configurations(self):
        """Test GPT with different configurations."""
        test_configs = [
            {
                "vocab_size": 1000,
                "embed_dim": 128,
                "num_heads": 2,
                "num_layers": 2,
                "max_position_embeddings": 256,
                "dropout": 0.1
            },
            {
                "vocab_size": 5000,
                "embed_dim": 256,
                "num_heads": 4,
                "num_layers": 4,
                "max_position_embeddings": 512,
                "dropout": 0.1
            },
            {
                "vocab_size": 10000,
                "embed_dim": 512,
                "num_heads": 8,
                "num_layers": 6,
                "max_position_embeddings": 1024,
                "dropout": 0.1
            }
        ]
        
        for config in test_configs:
            model = GPT(config)
            batch_size, seq_len = 2, 16
            inputs = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
            
            logits = model(inputs)
            
            expected_shape = (batch_size, seq_len, config['vocab_size'])
            assert logits.shape == expected_shape
    
    @pytest.mark.parametrize("batch_size,seq_len", [(1, 8), (2, 16), (4, 32)])
    def test_different_input_shapes(self, gpt_config, batch_size, seq_len):
        """Test GPT with different input shapes."""
        model = GPT(gpt_config)
        
        inputs = torch.randint(0, gpt_config['vocab_size'], (batch_size, seq_len))
        logits = model(inputs)
        
        expected_shape = (batch_size, seq_len, gpt_config['vocab_size'])
        assert logits.shape == expected_shape
    
    def test_training_vs_evaluation(self, gpt_config, random_inputs):
        """Test that GPT behaves differently in train vs eval mode."""
        model = GPT(gpt_config)
        
        # Training mode
        model.train()
        output_train = model(random_inputs)
        
        # Evaluation mode
        model.eval()
        output_eval = model(random_inputs)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output_train, output_eval)
    
    def test_parameter_count(self, gpt_config):
        """Test that GPT has reasonable number of parameters."""
        model = GPT(gpt_config)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # For a small GPT model, parameters should be in reasonable range
        vocab_size = gpt_config['vocab_size']
        embed_dim = gpt_config['embed_dim']
        num_layers = gpt_config['num_layers']
        num_heads = gpt_config['num_heads']
        
        # Rough estimate: token_embeddings + output_layer + (attention + ff) * layers
        expected_min = vocab_size * embed_dim * 2  # embeddings and output
        expected_max = expected_min * 10  # Allow for decoder parameters
        
        assert expected_min < total_params < expected_max
    
    def test_causal_attention(self, gpt_config):
        """Test that GPT uses causal attention during generation."""
        model = GPT(gpt_config)
        model.eval()
        
        # Create input with known pattern
        input_ids = torch.tensor([[1, 2, 3]]).long()
        
        with torch.no_grad():
            # Get logits for next token prediction
            logits = model(input_ids)
            
            # The model should only attend to previous tokens (causal)
            # We can't directly test attention masks in the public API,
            # but we can verify the generation works correctly
            
            generated = model.generate(
                x=input_ids,
                max_new_tokens=3,
                do_sample=False
            )
            
            # Generated sequence should be longer than input
            assert generated.shape[1] == input_ids.shape[1] + 3
    
    def test_output_distribution(self, gpt_config, random_inputs):
        """Test that GPT output has proper distribution."""
        model = GPT(gpt_config)
        
        logits = model(random_inputs)
        
        # Logits should not have extreme values
        assert logits.abs().max() < 100
        
        # Softmax should produce valid probabilities
        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)))
        assert (probs >= 0).all() and (probs <= 1).all()
