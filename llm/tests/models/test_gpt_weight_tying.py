"""
Tests for weight tying in GPT-1 / GPT-2 (tie_word_embeddings) and loading HuggingFace weights.
"""

import pytest
import torch

from llm.models.gpt import GPT, GPT2, convert_hf_state_dict

CONFIG = {
    "vocab_size": 60,
    "embed_dim": 32,
    "num_heads": 4,
    "num_layers": 2,
    "max_position_embeddings": 16,
    "dropout": 0.0,
}
MODELS = {"gpt": GPT, "gpt2": GPT2}


def build(name, **overrides):
    torch.manual_seed(0)
    return MODELS[name]({**CONFIG, **overrides})


def num_parameters(model):
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("name", list(MODELS))
def test_untied_by_default(name):
    model = build(name)
    assert model._linear.bias is not None
    assert model._linear.weight is not model._token_embeddings._embedding.weight


@pytest.mark.parametrize("name", list(MODELS))
def test_tied_projection_shares_embedding_and_has_no_bias(name):
    model = build(name, tie_word_embeddings=True)
    assert model._linear.weight is model._token_embeddings._embedding.weight
    assert model._linear.bias is None

    saved = CONFIG["vocab_size"] * CONFIG["embed_dim"] + CONFIG["vocab_size"]
    assert num_parameters(build(name)) - num_parameters(model) == saved


@pytest.mark.parametrize("name", list(MODELS))
def test_tied_logits_are_hidden_times_embedding(name):
    model = build(name, tie_word_embeddings=True).eval()
    hidden = []
    model._linear.register_forward_hook(lambda module, inputs, output: hidden.append(inputs[0]))
    tokens = torch.randint(0, CONFIG["vocab_size"], (2, 5))
    with torch.no_grad():
        logits, _ = model(tokens)
    expected = hidden[0] @ model._token_embeddings._embedding.weight.t()
    assert torch.allclose(logits, expected, atol=1e-6)


@pytest.mark.parametrize("name", list(MODELS))
def test_shared_weight_gets_gradient_from_input_and_output(name):
    """Градиент общей матрицы — сумма градиентов от эмбеддингов и от выходной проекции."""
    model = build(name, tie_word_embeddings=True)
    tokens = torch.randint(0, CONFIG["vocab_size"], (2, 5))
    logits, _ = model(tokens)
    logits.sum().backward()
    grad = model._token_embeddings._embedding.weight.grad

    unused = sorted(set(range(CONFIG["vocab_size"])) - set(tokens.flatten().tolist()))
    # Строки токенов, которых нет во входе, получают градиент только от выходной проекции
    assert grad[unused].abs().sum() > 0


@pytest.mark.parametrize("name", list(MODELS))
def test_save_load_keeps_tying(name, tmp_path):
    model = build(name, tie_word_embeddings=True).eval()
    path = tmp_path / "model.pt"
    model.save(str(path))
    loaded = MODELS[name].load(str(path)).eval()

    assert loaded._linear.weight is loaded._token_embeddings._embedding.weight
    tokens = torch.randint(0, CONFIG["vocab_size"], (2, 5))
    with torch.no_grad():
        assert torch.equal(model(tokens)[0], loaded(tokens)[0])


@pytest.mark.parametrize("name", list(MODELS))
def test_untied_checkpoint_does_not_load_into_tied_model(name):
    state_dict = build(name).state_dict()
    with pytest.raises(RuntimeError, match="_linear.bias"):
        build(name, tie_word_embeddings=True).load_state_dict(state_dict)


def hf_models():
    transformers = pytest.importorskip("transformers")
    common = dict(vocab_size=CONFIG["vocab_size"], n_positions=CONFIG["max_position_embeddings"],
                  n_embd=CONFIG["embed_dim"], n_layer=CONFIG["num_layers"], n_head=CONFIG["num_heads"],
                  resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
                  # Крупные веса: при std 0.02 точный GELU и tanh-аппроксимация неотличимы
                  initializer_range=0.5)
    return {
        # afn="gelu" в HF OpenAIGPT — tanh-аппроксимация (свой ACT_FNS в modeling_openai),
        # т. е. activation="gelu_tanh" по умолчанию здесь
        "gpt": transformers.OpenAIGPTLMHeadModel(transformers.OpenAIGPTConfig(afn="gelu", **common)),
        "gpt2": transformers.GPT2LMHeadModel(transformers.GPT2Config(**common)),
    }


@pytest.mark.parametrize("name", list(MODELS))
def test_hf_weights_give_same_logits(name):
    torch.manual_seed(0)
    hf_model = hf_models()[name].eval()
    model = build(name, tie_word_embeddings=True).eval()
    model.load_state_dict(convert_hf_state_dict(hf_model.state_dict()))
    assert model._linear.weight is model._token_embeddings._embedding.weight

    tokens = torch.randint(0, CONFIG["vocab_size"], (2, CONFIG["max_position_embeddings"]))
    with torch.no_grad():
        expected = hf_model(tokens).logits
        logits, _ = model(tokens)
    assert torch.allclose(logits, expected, atol=1e-4)


def test_convert_rejects_unknown_keys():
    with pytest.raises(KeyError, match="h.0.attn.rotary"):
        convert_hf_state_dict({"wte.weight": torch.zeros(1), "h.0.attn.rotary.weight": torch.zeros(1)})
