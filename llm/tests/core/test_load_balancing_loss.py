"""
Tests for load_balancing_loss and Mixtral.auxiliary_loss (backlog item 37).
"""

import pytest
import torch

from llm.core.moe import load_balancing_loss
from llm.models.gpt import GPT
from llm.models.mixtral import Mixtral

E, K = 8, 2
MIXTRAL = {
    "vocab_size": 50,
    "embed_dim": 32,
    "num_layers": 2,
    "max_position_embeddings": 16,
    "dropout": 0.0,
    "num_q_heads": 4,
    "num_kv_heads": 2,
    "window_size": 4,
    "num_experts": E,
    "top_k_experts": K,
}


def reference(router_logits, num_experts, top_k, token_mask=None):
    """Поэлементный эталон: E · Σ_k Σ_i f_{k,i} · P_i по всем слоям и (настоящим) токенам."""
    rows = torch.cat(router_logits).float()
    keep = torch.ones(len(rows)) if token_mask is None else token_mask.float().repeat(len(router_logits))
    probs = torch.softmax(rows, dim=-1)
    top = torch.topk(probs, top_k, dim=-1).indices
    n = keep.sum()
    f = torch.zeros(top_k, num_experts)
    p = torch.zeros(num_experts)
    for row in range(len(rows)):
        if keep[row]:
            p += probs[row] / n
            for k in range(top_k):
                f[k, top[row, k]] += 1 / n
    return num_experts * (f * p).sum()


def test_matches_reference():
    torch.manual_seed(0)
    logits = [torch.randn(21, E) for _ in range(3)]
    assert torch.allclose(load_balancing_loss(logits, E, K), reference(logits, E, K))


def test_uniform_routing_gives_top_k():
    assert load_balancing_loss([torch.zeros(10, E)], E, K).item() == pytest.approx(K)


def test_collapsed_routing_is_penalized():
    """Все токены у одних и тех же экспертов — loss больше, чем при равномерной загрузке."""
    collapsed = torch.full((32, E), -10.0)
    collapsed[:, :K] = 10.0
    assert load_balancing_loss([collapsed], E, K).item() > 2 * K


def test_token_mask_ignores_padding():
    torch.manual_seed(0)
    real = torch.randn(10, E)
    padded = torch.cat([real, torch.randn(4, E) * 50])
    mask = torch.cat([torch.ones(10), torch.zeros(4)])
    expected = load_balancing_loss([real], E, K)
    assert torch.allclose(load_balancing_loss([padded], E, K, token_mask=mask), expected)
    assert torch.allclose(load_balancing_loss([padded], E, K, token_mask=mask), reference([padded], E, K, mask))


def test_disabled_by_default():
    model = Mixtral(MIXTRAL)
    model(torch.randint(0, 50, (2, 6)))
    assert model.auxiliary_loss() is None
    assert GPT({**MIXTRAL, "num_heads": 4}).auxiliary_loss() is None


def test_mixtral_auxiliary_loss():
    torch.manual_seed(0)
    model = Mixtral({**MIXTRAL, "router_aux_loss_coef": 0.02}).train()
    tokens = torch.randint(0, 50, (2, 6))
    model(tokens)

    aux = model.auxiliary_loss()
    router_logits = [decoder._ff.router_logits for decoder in model._decoders]
    assert len(router_logits) == MIXTRAL["num_layers"]
    assert torch.allclose(aux, 0.02 * load_balancing_loss(router_logits, E, K))

    aux.backward()
    assert model._decoders[0]._ff._router.weight.grad.abs().sum() > 0


def test_mixtral_auxiliary_loss_ignores_right_padding():
    torch.manual_seed(0)
    model = Mixtral({**MIXTRAL, "router_aux_loss_coef": 1.0}).eval()
    tokens = torch.randint(1, 50, (2, 6))
    with torch.no_grad():
        model(tokens[:, :4])
        expected = model.auxiliary_loss()
        mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]])
        model(tokens, attention_mask=mask)
        assert torch.allclose(model.auxiliary_loss(), expected, atol=1e-6)


def test_negative_coefficient_rejected():
    with pytest.raises(ValueError, match="router_aux_loss_coef"):
        Mixtral({**MIXTRAL, "router_aux_loss_coef": -0.1})
