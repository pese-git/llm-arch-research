"""
Tests for Gemma as in the paper (√d embedding scale, weight tying, no bias, 8·d GeGLU,
K/V heads from the config), RMSNorm in float32 for half precision, and loading HuggingFace weights.
"""

import math

import pytest
import torch
from torch import nn

from llm.core.group_query_attention import GroupedQueryAttention
from llm.core.multi_query_attention import MultiQueryAttention
from llm.core.rms_norm import RMSNorm
from llm.core.rope import RoPE
from llm.models.gemma import Gemma, convert_hf_state_dict
from llm.models.llama import convert_hf_state_dict as convert_llama_state_dict

CONFIG = {
    "vocab_size": 100,
    "embed_dim": 64,
    "num_q_heads": 4,
    "num_layers": 2,
    "max_position_embeddings": 32,
    "dropout": 0.0,
}
# Gemma как в статье: все флаги включены
PAPER = {"bias": False, "tie_word_embeddings": True, "scale_embeddings": True}


def build(**overrides):
    torch.manual_seed(0)
    return Gemma({**CONFIG, **overrides}).eval()


def linears(model):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def test_defaults_keep_old_structure():
    """Без новых ключей — прежние MQA, GeGLU 4·d, bias, отдельная голова и эмбеддинги без множителя."""
    model = build()
    heads, ff = model._decoders[0]._heads, model._decoders[0]._ff
    head_size = CONFIG["embed_dim"] // CONFIG["num_q_heads"]
    assert heads._k.out_features == heads._v.out_features == head_size
    assert ff._gate.out_features == 4 * CONFIG["embed_dim"]
    assert all(linear.bias is not None for linear in linears(model))
    assert model._linear.weight is not model._token_embeddings._embedding.weight


def test_num_kv_heads_from_config():
    """Gemma 7B — MHA: num_kv_heads = num_q_heads, head_size из конфига."""
    model = build(num_kv_heads=4, head_size=24)
    heads = model._decoders[0]._heads
    assert (heads._q.out_features, heads._k.out_features, heads._v.out_features) == (96, 96, 96)
    with torch.no_grad():
        logits, _ = model(torch.randint(0, CONFIG["vocab_size"], (2, 5)))
    assert logits.shape == (2, 5, CONFIG["vocab_size"])


def test_intermediate_size_bias_and_tying():
    model = build(intermediate_size=512, **PAPER)
    ff = model._decoders[0]._ff
    assert (ff._gate.out_features, ff._up.out_features, ff._down.in_features) == (512, 512, 512)
    assert all(linear.bias is None for linear in linears(model))
    assert model._linear.weight is model._token_embeddings._embedding.weight


def test_bias_false_without_tying_keeps_bias_free_head():
    model = build(bias=False)
    assert model._linear.bias is None
    assert model._linear.weight is not model._token_embeddings._embedding.weight


def test_scale_embeddings_multiplies_by_sqrt_embed_dim():
    inputs = {}
    tokens = torch.randint(0, CONFIG["vocab_size"], (2, 5))
    for scale in (False, True):
        model = build(scale_embeddings=scale)
        model._decoders[0].register_forward_hook(lambda m, args, out, s=scale: inputs.__setitem__(s, args[0]))
        with torch.no_grad():
            model(tokens)
    assert torch.allclose(inputs[True], inputs[False] * math.sqrt(CONFIG["embed_dim"]))


@pytest.mark.parametrize("value", [0, -4])
def test_intermediate_size_must_be_positive(value):
    with pytest.raises(ValueError, match="hidden_dim"):
        build(intermediate_size=value)


def test_gqa_with_one_kv_head_is_bit_identical_to_mqa():
    """Gemma перешла с MultiQueryAttention на GroupedQueryAttention: при одной K/V-голове результат побитово тот же."""
    rope = RoPE(head_size=16, max_seq_len=32)
    torch.manual_seed(0)
    mqa = MultiQueryAttention(num_q_heads=4, emb_size=64, head_size=16, max_seq_len=32, rope=rope, dropout=0.0)
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(num_q_heads=4, num_kv_heads=1, emb_size=64, head_size=16, max_seq_len=32,
                                rope=rope, dropout=0.0)
    assert mqa.state_dict().keys() == gqa.state_dict().keys()
    x = torch.randn(2, 20, 64)
    with torch.no_grad():
        out_mqa, cache_mqa = mqa(x[:, :7], use_cache=True)
        out_gqa, cache_gqa = gqa(x[:, :7], use_cache=True)
        assert torch.equal(out_mqa, out_gqa)
        for t in range(7, 20):
            out_mqa, cache_mqa = mqa(x[:, t:t + 1], cache=cache_mqa)
            out_gqa, cache_gqa = gqa(x[:, t:t + 1], cache=cache_gqa)
            assert torch.equal(out_mqa, out_gqa)


def test_rms_norm_half_precision_is_computed_in_float32():
    """Сумма квадратов 300² · 64 во float16 переполняется (> 65504); во float32 — нет."""
    torch.manual_seed(0)
    norm = RMSNorm(64)
    x = 300 * torch.randn(2, 64)
    expected = norm(x)
    for dtype in (torch.float16, torch.bfloat16):
        out = norm.to(dtype)(x.to(dtype))
        assert out.dtype == dtype
        assert torch.isfinite(out).all()
        assert torch.allclose(out.float(), expected, atol=1e-2, rtol=1e-2)
        norm.float()


def test_rms_norm_float32_formula_unchanged():
    torch.manual_seed(0)
    norm = RMSNorm(16, eps=1e-6)
    with torch.no_grad():
        norm._w.copy_(torch.randn(16))
    x = torch.randn(3, 16)
    expected = norm._w * (x / (x.pow(2).mean(-1, keepdim=True) + 1e-6) ** 0.5)
    assert torch.equal(norm(x), expected)


def random_hf_gemma(num_kv_heads, head_dim):
    transformers = pytest.importorskip("transformers")
    config = transformers.GemmaConfig(
        vocab_size=CONFIG["vocab_size"], hidden_size=CONFIG["embed_dim"], intermediate_size=8 * CONFIG["embed_dim"],
        num_hidden_layers=CONFIG["num_layers"], num_attention_heads=CONFIG["num_q_heads"],
        num_key_value_heads=num_kv_heads, head_dim=head_dim, max_position_embeddings=CONFIG["max_position_embeddings"],
        rms_norm_eps=1e-6, rope_theta=500.0, attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = transformers.GemmaForCausalLM(config).eval()
    # HF инициализирует веса RMSNorm нулями (то есть множитель 1); случайные значения делают сверку строже
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(0.2 * torch.randn_like(parameter))
    model.generation_config.eos_token_id = None  # сравниваются все шаги генерации
    return model


def like_hf(hf_model, convert=convert_hf_state_dict, **overrides):
    c = hf_model.config
    model = build(num_kv_heads=c.num_key_value_heads, head_size=c.head_dim, intermediate_size=c.intermediate_size,
                  rms_norm_eps=c.rms_norm_eps, rope_theta=c.rope_theta, **{**PAPER, **overrides})
    model.load_state_dict(convert(hf_model.state_dict(), num_heads=c.num_attention_heads,
                                  num_kv_heads=c.num_key_value_heads))
    return model


# Gemma 2B — MQA с head_dim = hidden / heads; Gemma 7B — MHA с head_dim ≠ hidden / heads
HF_SHAPES = {"mqa-2b-like": (1, 16), "mha-7b-like": (4, 24)}


@pytest.mark.parametrize("num_kv_heads, head_dim", list(HF_SHAPES.values()), ids=list(HF_SHAPES))
def test_hf_weights_give_same_logits(num_kv_heads, head_dim):
    hf_model = random_hf_gemma(num_kv_heads, head_dim)
    model = like_hf(hf_model)
    assert model._linear.weight is model._token_embeddings._embedding.weight

    torch.manual_seed(1)
    tokens = torch.randint(0, CONFIG["vocab_size"], (2, CONFIG["max_position_embeddings"]))
    with torch.no_grad():
        expected = hf_model(tokens).logits
        logits, _ = model(tokens)
        hf_greedy = hf_model.generate(tokens[:1, :6], max_new_tokens=20, do_sample=False, pad_token_id=0)
        greedy = model.generate(tokens[:1, :6], max_new_tokens=20, do_sample=False, use_cache=True)
    assert torch.allclose(logits, expected, atol=1e-4)
    assert torch.equal(greedy, hf_greedy)


@pytest.mark.parametrize(
    "overrides, convert",
    [({"scale_embeddings": False}, convert_hf_state_dict), ({}, convert_llama_state_dict)],
    ids=["no-embedding-scale", "no-norm-offset"],
)
def test_hf_parity_needs_embedding_scale_and_norm_offset(overrides, convert):
    """Без √d или без +1 к весам RMSNorm результат HF не воспроизводится."""
    hf_model = random_hf_gemma(1, 16)
    model = like_hf(hf_model, convert=convert, **overrides)
    tokens = torch.randint(0, CONFIG["vocab_size"], (1, 16))
    with torch.no_grad():
        assert not torch.allclose(model(tokens)[0], hf_model(tokens).logits, atol=1e-2)
