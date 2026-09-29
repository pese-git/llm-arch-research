"""
Tests for BaseModel.save / BaseModel.load in every model.
"""

import pytest
import torch

from llm.models.gemma import Gemma
from llm.models.gpt import GPT, GPT2
from llm.models.llama import Llama
from llm.models.mistral import Mistral
from llm.models.mixtral import Mixtral

BASE_CONFIG = {
    "vocab_size": 50,
    "embed_dim": 32,
    "num_layers": 2,
    "max_position_embeddings": 16,
    "dropout": 0.0,
}

MODELS = {
    "gpt": (GPT, {"num_heads": 4, "activation": "relu"}),
    "gpt2": (GPT2, {"num_heads": 4}),
    "llama": (Llama, {"num_heads": 4}),
    "mistral": (Mistral, {"num_q_heads": 4, "num_kv_heads": 2, "window_size": 4, "head_size": 16}),
    "mixtral": (
        Mixtral,
        {"num_q_heads": 4, "num_kv_heads": 2, "window_size": 4, "num_experts": 4, "top_k_experts": 2},
    ),
    "gemma": (Gemma, {"num_q_heads": 4}),
}


@pytest.fixture(params=list(MODELS), ids=list(MODELS))
def model_class_and_config(request):
    model_class, extra = MODELS[request.param]
    return model_class, {**BASE_CONFIG, **extra}


@pytest.fixture
def tokens():
    torch.manual_seed(1)
    return torch.randint(0, BASE_CONFIG["vocab_size"], (2, 6))


def test_roundtrip_restores_model(model_class_and_config, tokens, tmp_path):
    """load recreates the model from the saved config: no constructor arguments needed."""
    model_class, config = model_class_and_config
    torch.manual_seed(0)
    model = model_class(config).eval()
    path = tmp_path / "model.pt"
    model.save(str(path))

    torch.manual_seed(42)  # different initial weights, so equality comes from loading
    restored = model_class.load(str(path))

    assert type(restored) is model_class
    assert restored.config == config
    assert not restored.training
    with torch.no_grad():
        assert torch.equal(restored(tokens)[0], model(tokens)[0])
        assert torch.equal(
            restored.generate(tokens, max_new_tokens=4, do_sample=False),
            model.generate(tokens, max_new_tokens=4, do_sample=False),
        )


def test_saved_file_contains_class_config_and_weights_only(model_class_and_config, tmp_path):
    model_class, config = model_class_and_config
    model = model_class(config)
    path = tmp_path / "model.pt"
    model.save(str(path))

    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["model_class"] == model_class.__name__
    assert checkpoint["config"] == config
    assert checkpoint["state_dict"].keys() == model.state_dict().keys()


def test_load_moves_model_to_device(model_class_and_config, tmp_path):
    model_class, config = model_class_and_config
    path = tmp_path / "model.pt"
    model_class(config).save(str(path))

    restored = model_class.load(str(path), device="meta")
    assert all(p.device.type == "meta" for p in restored.parameters())


def test_load_rejects_other_model_class(tmp_path):
    path = tmp_path / "model.pt"
    GPT({**BASE_CONFIG, "num_heads": 4}).save(str(path))
    with pytest.raises(ValueError, match="GPT"):
        GPT2.load(str(path))


def test_load_rejects_plain_state_dict(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save(GPT({**BASE_CONFIG, "num_heads": 4}).state_dict(), path)
    with pytest.raises(ValueError, match="state_dict"):
        GPT.load(str(path))
