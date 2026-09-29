"""
Left padding against HuggingFace: a left-padded batch gives the same logits of real tokens
and the same greedy generation as HF models with the same weights.

HF computes positions from attention_mask the same way (cumsum − 1) in generate; for the
forward pass they are passed explicitly as position_ids.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from llm.models import gemma, gpt, llama, mistral  # noqa: E402

VOCAB, EMBED, HEADS, LAYERS, MAX_LEN = 100, 64, 4, 2, 32


def randomize_(hf_model):
    torch.manual_seed(0)
    with torch.no_grad():
        for name, parameter in hf_model.named_parameters():
            noise = torch.randn_like(parameter)
            if "norm" in name and "gemma" not in type(hf_model).__name__.lower():
                parameter.copy_(1 + 0.3 * noise)
            else:
                parameter.copy_(0.2 * noise)
    hf_model.generation_config.eos_token_id = None
    return hf_model.eval()


def gpt2_pair():
    hf = randomize_(transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=VOCAB, n_positions=MAX_LEN, n_embd=EMBED, n_layer=LAYERS, n_head=HEADS,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)))
    ours = gpt.GPT2({"vocab_size": VOCAB, "embed_dim": EMBED, "num_heads": HEADS, "num_layers": LAYERS,
                     "max_position_embeddings": MAX_LEN, "dropout": 0.0, "tie_word_embeddings": True})
    ours.load_state_dict(gpt.convert_hf_state_dict(hf.state_dict()))
    return hf, ours


def llama_pair():
    hf = randomize_(transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=VOCAB, hidden_size=EMBED, intermediate_size=176, num_hidden_layers=LAYERS,
        num_attention_heads=HEADS, num_key_value_heads=HEADS, max_position_embeddings=MAX_LEN,
        rms_norm_eps=1e-5, tie_word_embeddings=False, attn_implementation="eager")))
    ours = llama.Llama({"vocab_size": VOCAB, "embed_dim": EMBED, "num_heads": HEADS, "num_layers": LAYERS,
                        "max_position_embeddings": MAX_LEN, "dropout": 0.0, "rms_norm_eps": 1e-5,
                        "intermediate_size": 176, "bias": False})
    ours.load_state_dict(llama.convert_hf_state_dict(hf.state_dict(), num_heads=HEADS))
    return hf, ours


def mistral_pair():
    hf = randomize_(transformers.MistralForCausalLM(transformers.MistralConfig(
        vocab_size=VOCAB, hidden_size=EMBED, intermediate_size=224, num_hidden_layers=LAYERS,
        num_attention_heads=HEADS, num_key_value_heads=2, max_position_embeddings=MAX_LEN,
        sliding_window=5, rms_norm_eps=1e-5, tie_word_embeddings=False, attn_implementation="eager")))
    ours = mistral.Mistral({"vocab_size": VOCAB, "embed_dim": EMBED, "num_q_heads": HEADS, "num_kv_heads": 2,
                            "num_layers": LAYERS, "max_position_embeddings": MAX_LEN, "dropout": 0.0,
                            "window_size": 4, "rms_norm_eps": 1e-5, "intermediate_size": 224, "bias": False})
    ours.load_state_dict(mistral.convert_hf_state_dict(hf.state_dict(), num_heads=HEADS, num_kv_heads=2))
    return hf, ours


def gemma_pair():
    hf = randomize_(transformers.GemmaForCausalLM(transformers.GemmaConfig(
        vocab_size=VOCAB, hidden_size=EMBED, intermediate_size=512, num_hidden_layers=LAYERS,
        num_attention_heads=HEADS, num_key_value_heads=1, head_dim=16, max_position_embeddings=MAX_LEN,
        rms_norm_eps=1e-6, attn_implementation="eager")))
    ours = gemma.Gemma({"vocab_size": VOCAB, "embed_dim": EMBED, "num_q_heads": HEADS, "num_kv_heads": 1,
                        "head_size": 16, "num_layers": LAYERS, "max_position_embeddings": MAX_LEN,
                        "dropout": 0.0, "intermediate_size": 512, "bias": False,
                        "tie_word_embeddings": True, "scale_embeddings": True})
    ours.load_state_dict(gemma.convert_hf_state_dict(hf.state_dict(), num_heads=HEADS, num_kv_heads=1))
    return hf, ours


PAIRS = {"gpt2": gpt2_pair, "llama": llama_pair, "mistral-window": mistral_pair, "gemma-mqa": gemma_pair}


def left_padded_batch():
    torch.manual_seed(1)
    lengths = [9, 5, 2]
    tokens = torch.zeros(len(lengths), max(lengths), dtype=torch.long)
    mask = torch.zeros_like(tokens)
    for i, n in enumerate(lengths):
        tokens[i, -n:] = torch.randint(1, VOCAB, (n,))
        mask[i, -n:] = 1
    return tokens, mask


@pytest.mark.parametrize("name", list(PAIRS))
def test_left_padded_batch_matches_hf(name):
    hf, ours = PAIRS[name]()
    ours.eval()
    tokens, mask = left_padded_batch()
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        expected = hf(tokens, attention_mask=mask, position_ids=position_ids).logits
        logits, _ = ours(tokens, attention_mask=mask)
        real = mask.bool()
        assert torch.allclose(logits[real], expected[real], atol=1e-4)

        hf_greedy = hf.generate(tokens, attention_mask=mask, max_new_tokens=12, do_sample=False, pad_token_id=0)
        greedy = ours.generate(tokens, attention_mask=mask, max_new_tokens=12, do_sample=False)
    assert torch.equal(greedy, hf_greedy)
