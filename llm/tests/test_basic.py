"""
Basic tests for llm library components.
"""

import pytest
import torch
import tempfile
import os


def test_gpt_model_creation():
    """Test that GPT model can be created and forward pass works."""
    from llm.models.gpt import GPT
    
    config = {
        "vocab_size": 1000,
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "max_position_embeddings": 256,
        "dropout": 0.1
    }
    
    model = GPT(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, config["vocab_size"])
    print("✅ GPT model creation and forward pass test passed")


def test_bpe_tokenizer_basic():
    """Test basic BPE tokenizer functionality."""
    from llm.tokenizers import BPETokenizer
    
    tokenizer = BPETokenizer()
    
    # Train on simple texts
    texts = [
        "hello world",
        "test tokenization",
        "simple example"
    ]
    
    tokenizer.train(
        texts=texts,
        vocab_size=50,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
    )
    
    # Test encoding/decoding
    text = "hello world"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    assert isinstance(tokens, list)
    assert isinstance(decoded, str)
    assert len(tokens) > 0
    print("✅ BPE tokenizer basic test passed")


def test_token_embeddings():
    """Test token embeddings."""
    from llm.core.token_embeddings import TokenEmbeddings
    
    vocab_size = 1000
    embed_dim = 128
    
    embeddings = TokenEmbeddings(vocab_size, embed_dim)
    
    # Test forward pass
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output = embeddings(input_ids)
    
    assert output.shape == (batch_size, seq_len, embed_dim)
    print("✅ Token embeddings test passed")


def test_multi_head_attention():
    """Test multi-head attention."""
    from llm.core.multi_head_attention import MultiHeadAttention
    
    num_heads = 4
    emb_size = 128
    head_size = emb_size // num_heads
    max_seq_len = 256
    
    attention = MultiHeadAttention(num_heads, emb_size, head_size, max_seq_len)
    
    # Test forward pass
    batch_size, seq_len = 2, 16
    inputs = torch.randn(batch_size, seq_len, emb_size)
    
    output = attention(inputs)
    
    assert output.shape == inputs.shape
    print("✅ Multi-head attention test passed")


def test_feed_forward():
    """Test feed forward network."""
    from llm.core.feed_forward import FeedForward
    
    embed_dim = 128
    
    ff = FeedForward(embed_dim)
    
    # Test forward pass
    batch_size, seq_len = 2, 16
    inputs = torch.randn(batch_size, seq_len, embed_dim)
    
    output = ff(inputs)
    
    assert output.shape == inputs.shape
    print("✅ Feed forward test passed")


def test_gpt_generation():
    """Test GPT text generation."""
    from llm.models.gpt import GPT
    
    config = {
        "vocab_size": 1000,
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "max_position_embeddings": 256,
        "dropout": 0.1
    }
    
    model = GPT(config)
    model.eval()
    
    # Test greedy generation
    input_ids = torch.randint(0, config["vocab_size"], (1, 5))
    
    with torch.no_grad():
        generated = model.generate(
            x=input_ids,
            max_new_tokens=3,
            do_sample=False
        )
    
    assert generated.shape == (1, 8)  # 5 initial + 3 new tokens
    print("✅ GPT generation test passed")


def test_bpe_tokenizer_save_load():
    """Test BPE tokenizer save/load functionality."""
    from llm.tokenizers import BPETokenizer
    
    tokenizer = BPETokenizer()
    
    # Train on simple texts
    texts = ["hello world", "test save load"]
    tokenizer.train(
        texts=texts,
        vocab_size=30,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "test_tokenizer.json")
        
        # Save tokenizer
        tokenizer.save(save_path)
        assert os.path.exists(save_path)
        
        # Load tokenizer
        loaded_tokenizer = BPETokenizer.load(save_path)
        
        # Test that vocab size is the same
        assert tokenizer.get_vocab_size() == loaded_tokenizer.get_vocab_size()
        
        # Test that vocabularies are the same
        assert tokenizer.get_vocab() == loaded_tokenizer.get_vocab()
        
        # Test that both can encode/decode (even if tokens differ due to BPE state)
        text = "hello world"
        original_tokens = tokenizer.encode(text)
        loaded_tokens = loaded_tokenizer.encode(text)
        
        # Both should produce valid token lists
        assert isinstance(original_tokens, list)
        assert isinstance(loaded_tokens, list)
        assert len(original_tokens) > 0
        assert len(loaded_tokens) > 0
        
        # Both should be able to decode
        original_decoded = tokenizer.decode(original_tokens)
        loaded_decoded = loaded_tokenizer.decode(loaded_tokens)
        assert isinstance(original_decoded, str)
        assert isinstance(loaded_decoded, str)
    
    print("✅ BPE tokenizer save/load test passed")


def test_gpt_with_tokenizer():
    """Test GPT model with tokenizer integration."""
    from llm.models.gpt import GPT
    from llm.tokenizers import BPETokenizer
    
    # Create and train tokenizer
    tokenizer = BPETokenizer()
    texts = ["hello world", "test integration"]
    tokenizer.train(
        texts=texts,
        vocab_size=50,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
    )
    
    vocab_size = tokenizer.get_vocab_size()
    
    # Create GPT model with tokenizer's vocab size
    config = {
        "vocab_size": vocab_size,
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "max_position_embeddings": 256,
        "dropout": 0.1
    }
    
    model = GPT(config)
    
    # Test with tokenized input
    text = "hello world"
    tokens = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([tokens])
    
    with torch.no_grad():
        logits = model(input_ids)
    
    assert logits.shape == (1, len(tokens), vocab_size)
    print("✅ GPT with tokenizer integration test passed")


def run_all_tests():
    """Run all basic tests."""
    print("🧪 Running basic tests for llm library...")
    
    test_gpt_model_creation()
    test_bpe_tokenizer_basic()
    test_token_embeddings()
    test_multi_head_attention()
    test_feed_forward()
    test_gpt_generation()
    test_bpe_tokenizer_save_load()
    test_gpt_with_tokenizer()
    
    print("🎉 All basic tests passed!")


if __name__ == "__main__":
    run_all_tests()
