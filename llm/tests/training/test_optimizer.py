import pytest
import torch.nn as nn
from llm.training.optimizer import get_optimizer

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

def test_get_optimizer_adamw():
    model = DummyModel()
    optimizer = get_optimizer(model, lr=1e-3, weight_decay=0.02, optimizer_type="adamw")
    assert optimizer.__class__.__name__ == 'AdamW'
    assert optimizer.defaults['lr'] == 1e-3
    assert optimizer.defaults['weight_decay'] == 0.02

def test_get_optimizer_adam():
    model = DummyModel()
    optimizer = get_optimizer(model, lr=1e-4, weight_decay=0.01, optimizer_type="adam")
    assert optimizer.__class__.__name__ == 'Adam'
    assert optimizer.defaults['lr'] == 1e-4
    assert optimizer.defaults['weight_decay'] == 0.01

def test_get_optimizer_sgd():
    model = DummyModel()
    optimizer = get_optimizer(model, lr=0.1, optimizer_type="sgd")
    assert optimizer.__class__.__name__ == 'SGD'
    assert optimizer.defaults['lr'] == 0.1
    # SGD: weight_decay по умолчанию 0 для этого вызова
    assert optimizer.defaults['momentum'] == 0.9

def test_get_optimizer_invalid():
    model = DummyModel()
    with pytest.raises(ValueError):
        get_optimizer(model, optimizer_type="nonexistent")

# --- Группы weight decay (бэклог, пункт 61) ---

import torch

from llm.core.rms_norm import RMSNorm
from llm.models.gpt import GPT
from llm.models.llama import Llama
from llm.training.optimizer import weight_decay_param_groups


class NormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 8)
        self.linear = nn.Linear(8, 8)
        self.layer_norm = nn.LayerNorm(8)
        self.rms_norm = RMSNorm(8)


def group_ids(optimizer):
    return [{id(p) for p in group["params"]} for group in optimizer.param_groups]


@pytest.mark.parametrize("optimizer_type", ["adamw", "adam", "sgd"])
def test_matrices_decay_bias_and_norms_do_not(optimizer_type):
    """Матрицы (Linear, Embedding) — с weight_decay; bias, LayerNorm и RMSNorm — без."""
    model = NormModel()
    optimizer = get_optimizer(model, weight_decay=0.1, optimizer_type=optimizer_type)
    decay, no_decay = group_ids(optimizer)
    assert [g["weight_decay"] for g in optimizer.param_groups] == [0.1, 0.0]
    assert decay == {id(model.embedding.weight), id(model.linear.weight)}
    assert no_decay == {id(model.linear.bias), id(model.layer_norm.weight),
                        id(model.layer_norm.bias), id(model.rms_norm._w)}


@pytest.mark.parametrize("config", [
    {"tie_word_embeddings": True},
    {"tie_word_embeddings": False},
])
def test_every_parameter_once(config):
    """Все параметры модели в группах ровно по одному разу, в том числе общая матрица
    эмбеддингов и головы при weight tying (она — в группе с decay)."""
    model = GPT({"vocab_size": 50, "embed_dim": 16, "num_heads": 2, "num_layers": 2,
                 "max_position_embeddings": 8, "dropout": 0.0, **config})
    groups = weight_decay_param_groups(model, 0.01)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert id(model._token_embeddings._embedding.weight) in {id(p) for p in groups[0]["params"]}


def test_llama_norms_without_decay():
    model = Llama({"vocab_size": 50, "embed_dim": 16, "num_heads": 2, "num_layers": 2,
                   "max_position_embeddings": 8, "dropout": 0.0, "bias": False})
    decay, no_decay = weight_decay_param_groups(model, 0.01)
    assert all(p.dim() >= 2 for p in decay["params"])
    rms = {id(m._w) for m in model.modules() if isinstance(m, RMSNorm)}
    assert {id(p) for p in no_decay["params"]} == rms  # без bias — только веса RMSNorm


def test_empty_group_skipped():
    model = nn.Linear(4, 4, bias=False)
    groups = weight_decay_param_groups(model, 0.01)
    assert len(groups) == 1 and groups[0]["weight_decay"] == 0.01


def test_adamw_step_decays_weights_but_not_bias():
    """Шаг AdamW при нулевом градиенте: веса затухают на lr·λ, bias не меняется."""
    model = NormModel()
    optimizer = get_optimizer(model, lr=0.1, weight_decay=0.5, optimizer_type="adamw")
    weight, bias = model.linear.weight.detach().clone(), model.linear.bias.detach().clone()
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()
    assert torch.allclose(model.linear.weight, weight * (1 - 0.1 * 0.5))
    assert torch.equal(model.linear.bias, bias)
    assert torch.all(model.rms_norm._w == 1)


def test_sgd_uses_weight_decay():
    """Раньше SGD молча игнорировал weight_decay."""
    optimizer = get_optimizer(NormModel(), lr=0.1, weight_decay=0.02, optimizer_type="sgd")
    assert optimizer.defaults["weight_decay"] == 0.02
    assert optimizer.param_groups[0]["weight_decay"] == 0.02
