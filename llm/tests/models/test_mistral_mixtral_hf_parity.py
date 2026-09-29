"""
Tests for Mistral / Mixtral as in the originals (intermediate_size, bias, optional sliding window)
and loading HuggingFace weights.
"""

import pytest
import torch
from torch import nn

from llm.models.mistral import Mistral, convert_hf_state_dict
from llm.models.mixtral import Mixtral

BASE = {
    "vocab_size": 100,
    "embed_dim": 64,
    "num_q_heads": 4,
    "num_kv_heads": 2,
    "num_layers": 2,
    "max_position_embeddings": 32,
    "dropout": 0.0,
}
MODELS = {
    "mistral": (Mistral, {}),
    "mixtral": (Mixtral, {"num_experts": 4, "top_k_experts": 2}),
}


def build(name, **overrides):
    model_class, extra = MODELS[name]
    torch.manual_seed(0)
    return model_class({**BASE, **extra, **overrides}).eval()


def linears(model):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def swiglus(model):
    return [m for m in model.modules() if type(m).__name__ == "SwiGLU"]


@pytest.mark.parametrize("name", list(MODELS))
def test_defaults_keep_old_structure(name):
    model = build(name, window_size=8)
    assert all(ff._gate.out_features == 4 * BASE["embed_dim"] for ff in swiglus(model))
    assert all(linear.bias is not None for linear in linears(model))


@pytest.mark.parametrize("name", list(MODELS))
def test_intermediate_size_and_bias_from_config(name):
    model = build(name, intermediate_size=224, bias=False)
    assert all(
        (ff._gate.out_features, ff._up.out_features, ff._down.in_features) == (224, 224, 224)
        for ff in swiglus(model)
    )
    # Q/K/V, выход attention, SwiGLU (у Mixtral — у каждого эксперта), роутер и голова
    assert all(linear.bias is None for linear in linears(model))
    assert not any(key.endswith(".bias") for key in model.state_dict())


@pytest.mark.parametrize("name", list(MODELS))
def test_no_window_is_plain_causal_attention(name):
    """Без window_size — обычное causal-внимание: как окно шире последовательности."""
    tokens = torch.randint(0, BASE["vocab_size"], (2, 20))
    with torch.no_grad():
        dense, _ = build(name)(tokens)
        wide, _ = build(name, window_size=BASE["max_position_embeddings"])(tokens)
        narrow, _ = build(name, window_size=4)(tokens)
    assert torch.equal(dense, wide)
    assert not torch.allclose(dense, narrow)
    assert build(name)._decoders[0]._heads._tril_mask.equal(
        torch.tril(torch.ones(BASE["max_position_embeddings"], BASE["max_position_embeddings"])).bool()
    )


@pytest.mark.parametrize("name", list(MODELS))
def test_no_window_cache_keeps_every_position(name):
    model = build(name)
    tokens = torch.randint(0, BASE["vocab_size"], (1, 12))
    with torch.no_grad():
        full, _ = model(tokens)
        _, cache = model(tokens[:, :10], use_cache=True)
        assert cache[0][0].size(2) == 10
        step, cache = model(tokens[:, 10:], use_cache=True, cache=cache)
    assert cache[0][0].size(2) == 12
    assert torch.allclose(step, full[:, 10:], atol=1e-5)


def randomize_(hf_model):
    """HF инициализирует bias нулями, а RMSNorm — единицами; случайные значения делают сверку строже."""
    torch.manual_seed(0)
    with torch.no_grad():
        for name, parameter in hf_model.named_parameters():
            noise = torch.randn_like(parameter)
            parameter.copy_(1 + 0.3 * noise if "norm" in name else 0.2 * noise)
    return hf_model.eval()


def hf_mistral(**overrides):
    transformers = pytest.importorskip("transformers")
    config = transformers.MistralConfig(
        vocab_size=BASE["vocab_size"], hidden_size=BASE["embed_dim"], intermediate_size=224,
        num_hidden_layers=BASE["num_layers"], num_attention_heads=BASE["num_q_heads"],
        num_key_value_heads=BASE["num_kv_heads"], max_position_embeddings=BASE["max_position_embeddings"],
        rms_norm_eps=1e-5, rope_theta=500.0, tie_word_embeddings=False, attn_implementation="eager",
        **overrides,
    )
    return randomize_(transformers.MistralForCausalLM(config))


def hf_mixtral(**overrides):
    transformers = pytest.importorskip("transformers")
    config = transformers.MixtralConfig(
        vocab_size=BASE["vocab_size"], hidden_size=BASE["embed_dim"], intermediate_size=224,
        num_hidden_layers=BASE["num_layers"], num_attention_heads=BASE["num_q_heads"],
        num_key_value_heads=BASE["num_kv_heads"], max_position_embeddings=BASE["max_position_embeddings"],
        num_local_experts=4, num_experts_per_tok=2, rms_norm_eps=1e-5, rope_theta=500.0,
        tie_word_embeddings=False, attn_implementation="eager", **overrides,
    )
    return randomize_(transformers.MixtralForCausalLM(config))


def like_hf(name, hf_model, **overrides):
    """Конфиг под HF: окно здесь на одну позицию шире (W + 1), поэтому window_size = sliding_window − 1."""
    c = hf_model.config
    config = {"intermediate_size": c.intermediate_size, "bias": False,
              "rms_norm_eps": c.rms_norm_eps, "rope_theta": c.rope_theta}
    if c.sliding_window is not None:
        config["window_size"] = c.sliding_window - 1
    model = build(name, **{**config, **overrides})
    model.load_state_dict(convert_hf_state_dict(
        hf_model.state_dict(), num_heads=c.num_attention_heads, num_kv_heads=c.num_key_value_heads
    ))
    return model


def assert_same_as_hf(model, hf_model, prompt_len=6, new_tokens=20):
    # HF останавливает генерацию на eos_token_id из конфига; здесь сравниваются все шаги
    hf_model.generation_config.eos_token_id = None
    torch.manual_seed(1)
    tokens = torch.randint(0, BASE["vocab_size"], (2, BASE["max_position_embeddings"]))
    with torch.no_grad():
        expected = hf_model(tokens).logits
        logits, _ = model(tokens)
        # генерация с KV-кэшем дольше окна: кэш обрезается, позиции RoPE идут из next_pos
        hf_greedy = hf_model.generate(tokens[:1, :prompt_len], max_new_tokens=new_tokens,
                                      do_sample=False, pad_token_id=0)
        greedy = model.generate(tokens[:1, :prompt_len], max_new_tokens=new_tokens,
                                do_sample=False, use_cache=True)
    assert torch.allclose(logits, expected, atol=1e-4)
    assert torch.equal(greedy, hf_greedy)


@pytest.mark.parametrize("sliding_window", [6, None], ids=["window", "dense"])
def test_mistral_hf_weights_give_same_logits(sliding_window):
    hf_model = hf_mistral(sliding_window=sliding_window)
    assert_same_as_hf(like_hf("mistral", hf_model), hf_model)


def test_mistral_hf_head_dim_differs_from_embed_dim_by_heads():
    """head_dim в HF — head_size здесь; K/V-головы переставляются по num_kv_heads."""
    hf_model = hf_mistral(sliding_window=None, head_dim=8)
    assert_same_as_hf(like_hf("mistral", hf_model, head_size=8), hf_model)


def test_mistral_hf_window_off_by_one():
    """С тем же числом (window_size = sliding_window) окно здесь шире на позицию — результат другой."""
    hf_model = hf_mistral(sliding_window=6)
    model = like_hf("mistral", hf_model, window_size=6)
    tokens = torch.randint(0, BASE["vocab_size"], (1, 20))
    with torch.no_grad():
        assert not torch.allclose(model(tokens)[0], hf_model(tokens).logits, atol=1e-2)


def test_mixtral_hf_weights_give_same_logits():
    hf_model = hf_mixtral()
    assert hf_model.config.sliding_window is None  # Mixtral 8x7B — без окна
    assert_same_as_hf(like_hf("mixtral", hf_model), hf_model)


def test_mixtral_hf_router_and_experts_are_mapped():
    hf_model = hf_mixtral()
    model = like_hf("mixtral", hf_model)
    hf_moe = hf_model.model.layers[1].block_sparse_moe
    moe = model._decoders[1]._ff
    assert torch.equal(moe._router.weight, hf_moe.gate.weight)
    for expert, hf_expert in zip(moe._experts, hf_moe.experts):
        assert torch.equal(expert._gate.weight, hf_expert.w1.weight)
        assert torch.equal(expert._up.weight, hf_expert.w3.weight)
        assert torch.equal(expert._down.weight, hf_expert.w2.weight)
