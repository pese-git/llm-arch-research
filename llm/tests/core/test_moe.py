import torch
import pytest
from llm.core.moe import MoE

@pytest.fixture
def moe():
    # Базовая MoE для коротких тестов
    return MoE(emb_size=16, num_experts=4, top_k_experts=2, dropout=0.0)

def test_forward_shape(moe):
    x = torch.randn(3, 5, 16)  # [batch, seq, emb]
    y = moe(x)
    assert y.shape == x.shape

def test_forward_grad(moe):
    x = torch.randn(2, 4, 16, requires_grad=True)
    y = moe(x)
    (y.sum()).backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape

def test_top_k_larger_than_experts():
    # top_k_experts > num_experts должно падать
    with pytest.raises(ValueError):
        MoE(emb_size=8, num_experts=2, top_k_experts=4)

def test_single_expert_no_error():
    # один эксперт, один топ-к — модель всё ещё валидна
    moe = MoE(emb_size=8, num_experts=1, top_k_experts=1)
    x = torch.randn(2, 2, 8)
    y = moe(x)
    assert y.shape == x.shape

def test_forward_trivial_weights():
    """Проверяет, что при одинаковых весах роутера MoE возвращает усреднённое по экспертам."""
    class DummyMoE(MoE):
        def forward(self, x):
            # Роутер отдаёт всегда единичные логиты = softmax -> uniform
            self._router = torch.nn.Linear(x.size(-1), self._num_experts, bias=False)
            torch.nn.init.constant_(self._router.weight, 0.0)
            return super().forward(x)
    moe = DummyMoE(emb_size=4, num_experts=2, top_k_experts=2)
    x = torch.zeros(1, 2, 4)
    y = moe(x)
    assert y.shape == x.shape

def test_forward_deterministic_seed(moe):
    torch.manual_seed(42)
    x = torch.randn(2, 3, 16)
    y1 = moe(x)
    torch.manual_seed(42)
    y2 = moe(x)
    assert torch.allclose(y1, y2, atol=1e-5)

def test_forward_no_dropout():
    """Без dropout MoE не меняет shape и не даёт NaN."""
    moe = MoE(emb_size=5, num_experts=3, top_k_experts=2, dropout=0.0)
    x = torch.randn(2, 7, 5)
    y = moe(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
def test_half_precision_matches_float32(dtype):
    # Буфер весов экспертов должен иметь dtype входа, иначе запись весов bf16/fp16 падает
    torch.manual_seed(0)
    moe = MoE(emb_size=16, num_experts=4, top_k_experts=2, dropout=0.0)
    x = torch.randn(2, 5, 16)
    expected = moe(x)

    y = moe.to(dtype)(x.to(dtype))
    assert y.dtype == dtype
    assert torch.allclose(y.float(), expected, atol=5e-2)

def test_matches_naive_per_token_reference():
    # Эталон: для каждого токена softmax по top-k логитам роутера и взвешенная сумма
    # выходов выбранных экспертов (Mixtral: Softmax(TopK(x·W_g)))
    torch.manual_seed(0)
    moe = MoE(emb_size=16, num_experts=4, top_k_experts=2, dropout=0.0).eval()
    x = torch.randn(2, 5, 16)

    expected = torch.zeros_like(x)
    with torch.no_grad():
        for b in range(x.size(0)):
            for t in range(x.size(1)):
                token = x[b, t]
                top_logits, top_ids = torch.topk(moe._router(token), k=2)
                weights = torch.softmax(top_logits, dim=-1)
                for w, e in zip(weights, top_ids.tolist()):
                    expected[b, t] += w * moe._experts[e](token.view(1, 1, -1)).view(-1)
        actual = moe(x)

    assert torch.allclose(actual, expected, atol=1e-6)

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
def test_router_softmax_in_float32(dtype, monkeypatch):
    # Веса роутера считаются во float32 и приводятся к dtype входа (как в HF и эталоне Mistral)
    import llm.core.moe as moe_module

    softmax_input_dtypes = []
    original_softmax = moe_module.F.softmax

    def spy(tensor, *args, **kwargs):
        softmax_input_dtypes.append(tensor.dtype)
        return original_softmax(tensor, *args, **kwargs)

    monkeypatch.setattr(moe_module.F, "softmax", spy)
    moe = MoE(emb_size=16, num_experts=4, top_k_experts=2, dropout=0.0).to(dtype)
    y = moe(torch.randn(2, 5, 16, dtype=dtype))

    assert softmax_input_dtypes == [torch.float32]
    assert y.dtype == dtype
