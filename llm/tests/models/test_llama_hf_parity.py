"""
Tests for LLaMA as in the paper (intermediate_size, bias) and loading HuggingFace weights.
"""

import pytest
import torch
from torch import nn

from llm.models.llama import Llama, convert_hf_state_dict, llama_intermediate_size

CONFIG = {
    "vocab_size": 100,
    "embed_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "max_position_embeddings": 32,
    "dropout": 0.0,
}


def build(**overrides):
    torch.manual_seed(0)
    return Llama({**CONFIG, **overrides}).eval()


def linears(model):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def test_defaults_keep_old_structure():
    """Без новых ключей — прежние 4·embed_dim и bias везде: старые чекпоинты загружаются."""
    model = build()
    ff = model._decoders[0]._ff
    assert ff._gate.out_features == ff._up.out_features == ff._down.in_features == 4 * CONFIG["embed_dim"]
    assert all(linear.bias is not None for linear in linears(model))


def test_intermediate_size_from_config():
    model = build(intermediate_size=176)
    for decoder in model._decoders:
        ff = decoder._ff
        assert (ff._gate.out_features, ff._up.out_features, ff._down.in_features) == (176, 176, 176)
    with torch.no_grad():
        logits, _ = model(torch.randint(0, CONFIG["vocab_size"], (2, 5)))
    assert logits.shape == (2, 5, CONFIG["vocab_size"])


@pytest.mark.parametrize("value", [0, -8])
def test_intermediate_size_must_be_positive(value):
    with pytest.raises(ValueError, match="hidden_dim"):
        build(intermediate_size=value)


def test_bias_false_removes_every_bias():
    """Q/K/V, выход attention, три матрицы SwiGLU и голова — 4 + 3 на блок и одна голова."""
    model = build(bias=False)
    assert len(linears(model)) == 7 * CONFIG["num_layers"] + 1
    assert all(linear.bias is None for linear in linears(model))
    assert not any(key.endswith(".bias") for key in model.state_dict())


@pytest.mark.parametrize(
    "embed_dim, kwargs, expected",
    [
        (4096, {}, 11008),  # LLaMA 7B
        (5120, {}, 13824),  # LLaMA 13B
        (6656, {}, 17920),  # LLaMA 33B
        (8192, {"ffn_dim_multiplier": 1.3, "multiple_of": 4096}, 28672),  # LLaMA 2 70B
        (288, {"multiple_of": 32}, 768),  # llama2.c stories15M
    ],
)
def test_llama_intermediate_size(embed_dim, kwargs, expected):
    assert llama_intermediate_size(embed_dim, **kwargs) == expected


def random_hf_llama(**overrides):
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=CONFIG["vocab_size"], hidden_size=CONFIG["embed_dim"], intermediate_size=176,
        num_hidden_layers=CONFIG["num_layers"], num_attention_heads=CONFIG["num_heads"],
        num_key_value_heads=CONFIG["num_heads"], max_position_embeddings=CONFIG["max_position_embeddings"],
        rms_norm_eps=1e-5, rope_theta=500.0, tie_word_embeddings=False, attn_implementation="eager",
        **overrides,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).eval()
    # HF инициализирует bias нулями, а RMSNorm — единицами; случайные значения делают сверку строже
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            noise = torch.randn_like(parameter)
            parameter.copy_(1 + 0.3 * noise if "norm" in name else 0.2 * noise)
    return model


def llama_like(hf_model, **overrides):
    c = hf_model.config
    return build(intermediate_size=c.intermediate_size, rms_norm_eps=c.rms_norm_eps,
                 rope_theta=c.rope_theta, **overrides)


def assert_same_as_hf(model, hf_model):
    tokens = torch.randint(0, CONFIG["vocab_size"], (2, CONFIG["max_position_embeddings"]))
    with torch.no_grad():
        expected = hf_model(tokens).logits
        logits, _ = model(tokens)
        # генерация с KV-кэшем проходит через RoPE со сдвигом start_pos
        hf_greedy = hf_model.generate(tokens[:1, :6], max_new_tokens=10, do_sample=False, pad_token_id=0)
        greedy = model.generate(tokens[:1, :6], max_new_tokens=10, do_sample=False, use_cache=True)
    assert torch.allclose(logits, expected, atol=1e-4)
    assert torch.equal(greedy, hf_greedy)


def test_hf_weights_give_same_logits():
    hf_model = random_hf_llama()
    model = llama_like(hf_model, bias=False)
    model.load_state_dict(convert_hf_state_dict(hf_model.state_dict(), num_heads=CONFIG["num_heads"]))
    assert_same_as_hf(model, hf_model)


def test_hf_weights_with_attention_and_mlp_bias():
    """HF с attention_bias и mlp_bias: bias Q и K переставляются так же, как строки весов."""
    hf_model = random_hf_llama(attention_bias=True, mlp_bias=True)
    model = llama_like(hf_model)  # bias=True: у головы тоже bias, в HF его нет — нули
    state_dict = convert_hf_state_dict(hf_model.state_dict(), num_heads=CONFIG["num_heads"])
    state_dict["_linear.bias"] = torch.zeros(CONFIG["vocab_size"])
    model.load_state_dict(state_dict)
    assert_same_as_hf(model, hf_model)


def test_q_k_rows_must_be_permuted():
    """Без перестановки строк Q/K результат другой: RoPE здесь и в HF поворачивает разные пары."""
    hf_model = random_hf_llama()
    model = llama_like(hf_model, bias=False)
    state_dict = convert_hf_state_dict(hf_model.state_dict(), num_heads=CONFIG["num_heads"])
    hf_state_dict = hf_model.state_dict()
    for i in range(CONFIG["num_layers"]):
        state_dict[f"_decoders.{i}._heads._q.weight"] = hf_state_dict[f"model.layers.{i}.self_attn.q_proj.weight"]
        state_dict[f"_decoders.{i}._heads._k.weight"] = hf_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"]
    model.load_state_dict(state_dict)
    tokens = torch.randint(0, CONFIG["vocab_size"], (1, 16))
    with torch.no_grad():
        assert not torch.allclose(model(tokens)[0], hf_model(tokens).logits, atol=1e-2)


def test_tied_hf_checkpoint_copies_embeddings_to_head():
    hf_model = random_hf_llama()
    state_dict = {k: v for k, v in hf_model.state_dict().items() if k != "lm_head.weight"}
    converted = convert_hf_state_dict(state_dict, num_heads=CONFIG["num_heads"])
    assert torch.equal(converted["_linear.weight"], converted["_token_embeddings._embedding.weight"])


def test_convert_rejects_unknown_keys():
    with pytest.raises(KeyError, match="self_attn.rotary"):
        convert_hf_state_dict({"model.layers.0.self_attn.rotary.weight": torch.zeros(1)}, num_heads=1)
