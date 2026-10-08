# llm/tests/core/test_group_query_attention.py

import torch
import pytest
from llm.core.group_query_attention import GroupedQueryAttention
from llm.core.padding import padding_from_attention_mask
from llm.core.rope import RoPE

@pytest.fixture
def params():
    return {
        'num_q_heads': 4,
        'num_kv_heads': 2,
        'emb_size': 16,
        'head_size': 4,
        'max_seq_len': 32,
        'window_size': 8,
        'dropout': 0.0
    }

def test_initialization(params):
    attn = GroupedQueryAttention(**params)
    assert isinstance(attn, GroupedQueryAttention)

def test_forward_shape(params):
    batch, seq = 2, 10
    x = torch.randn(batch, seq, params['emb_size'])
    attn = GroupedQueryAttention(**params)
    y, cache = attn(x)
    assert y.shape == (batch, seq, params['emb_size'])
    assert cache is not None
    assert isinstance(y, torch.Tensor)

def test_forward_is_causal(params):
    """Выход позиций < t не зависит от входа на позициях ≥ t (causal + окно)."""
    attn = GroupedQueryAttention(**{**params, "dropout": 0.0}).eval()
    x = torch.randn(2, 10, params['emb_size'])
    t = 5
    changed = x.clone()
    changed[:, t:] = torch.randn_like(changed[:, t:])
    with torch.no_grad():
        y, _ = attn(x)
        y_changed, _ = attn(changed)
    assert torch.allclose(y[:, :t], y_changed[:, :t], atol=1e-6)
    assert not torch.allclose(y[:, t:], y_changed[:, t:])

def test_kv_repetition(params):
    batch, seq = 1, 3
    attn = GroupedQueryAttention(**params)
    kv = torch.randn(batch, params['num_kv_heads'], seq, params['head_size'])
    rep = attn._repeat_kv_heads(kv, params['num_q_heads'], params['num_kv_heads'])
    assert rep.shape == (batch, params['num_q_heads'], seq, params['head_size'])

def test_window_mask(params):
    attn = GroupedQueryAttention(**params)
    mask = attn._create_sliding_window_mask(8, 3)
    assert mask.shape == (8, 8)
    # Проверим булеву маску окна в позиции 4
    expected = torch.tensor([True, True, True, True, False, False])
    assert torch.equal(mask[4, 1:7], expected)

def test_forward_with_rope(params):
    batch, seq = 2, 12
    x = torch.randn(batch, seq, params['emb_size'])
    rope = RoPE(head_size=params['head_size'], max_seq_len=params['max_seq_len'])
    params2 = params.copy()
    params2['rope'] = rope
    attn = GroupedQueryAttention(**params2)
    y, _ = attn(x)
    assert y.shape == (batch, seq, params['emb_size'])

def test_cache_usage(params):
    batch, seq = 1, 5
    x = torch.randn(batch, seq, params['emb_size'])
    attn = GroupedQueryAttention(**params)
    # Первый проход - получаем кэш
    _, cache = attn(x)
    # Второй проход с кэшем (имитируем автокомплит seq_len=1)
    x2 = torch.randn(batch, 1, params['emb_size'])
    y2, cache2 = attn(x2, cache=cache)
    assert cache2 is not None
    assert y2.shape == (batch, 1, params['emb_size'])

def test_gradient_backward(params):
    batch, seq = 2, 6
    x = torch.randn(batch, seq, params['emb_size'], requires_grad=True)
    attn = GroupedQueryAttention(**params)
    y, _ = attn(x)
    y.sum().backward()
    for param in attn.parameters():
        assert param.grad is not None


# --- Маска вместе с кэшем ---------------------------------------------------------------------


def make_attn(params, window_size, rope=True):
    """Слой в eval (без dropout); rope=True — с RoPE, чтобы позиции кэша влияли на результат."""
    torch.manual_seed(0)
    kwargs = {**params, "window_size": window_size}
    if rope:
        kwargs["rope"] = RoPE(head_size=params["head_size"], max_seq_len=params["max_seq_len"])
    return GroupedQueryAttention(**kwargs).eval()


def run_in_chunks(attn, x, chunks):
    """Прогоняет x кусками длины chunks, передавая кэш; возвращает склеенный выход и кэш."""
    outs, cache = [], None
    for piece in x.split(chunks, dim=1):
        out, cache = attn(piece, cache=cache)
        outs.append(out)
    return torch.cat(outs, dim=1), cache


@pytest.mark.parametrize("window_size", [None, 4, 8])
@pytest.mark.parametrize("chunks", [1, 3, 7])
def test_chunked_forward_matches_full(params, window_size, chunks):
    """Кусками с кэшем получается то же, что одним проходом: каждой строке — своя маска окна."""
    attn = make_attn(params, window_size)
    x = torch.randn(2, 20, params["emb_size"])
    with torch.no_grad():
        full, _ = attn(x)
        chunked, _ = run_in_chunks(attn, x, chunks)
    assert torch.allclose(full, chunked, atol=1e-5)


def test_cache_is_trimmed_to_window_and_keeps_absolute_position(params):
    attn = make_attn(params, window_size=4)
    x = torch.randn(1, 11, params["emb_size"])
    with torch.no_grad():
        _, (k, v, next_pos) = attn(x)
    # В кэше последние window_size позиций, K/V — до дублирования голов; позиция для RoPE абсолютная
    assert k.shape == v.shape == (1, params["num_kv_heads"], 4, params["head_size"])
    assert next_pos == 11


def test_cache_without_window_keeps_everything(params):
    attn = make_attn(params, window_size=None)
    x = torch.randn(1, 11, params["emb_size"])
    with torch.no_grad():
        _, (k, _, next_pos) = attn(x)
    assert k.shape[2] == 11 and next_pos == 11


def test_window_hides_tokens_outside_it_with_cache(params):
    """Ключ дальше window_size позиций назад на выход не влияет — ни в проходе, ни из кэша."""
    attn = make_attn(params, window_size=3)
    x = torch.randn(1, 12, params["emb_size"])
    changed = x.clone()
    changed[:, 0] = torch.randn_like(changed[:, 0])
    with torch.no_grad():
        # Последний токен (позиция 11) видит позиции 8…11; позиция 0 далеко за окном
        out, _ = run_in_chunks(attn, x, 4)
        out_changed, _ = run_in_chunks(attn, changed, 4)
    assert torch.allclose(out[:, 4:], out_changed[:, 4:], atol=1e-6)
    assert not torch.allclose(out[:, 0], out_changed[:, 0])


def test_cache_overflow_raises(params):
    attn = make_attn(params, window_size=None)
    x = torch.randn(1, params["max_seq_len"], params["emb_size"])
    with torch.no_grad():
        _, cache = attn(x)
        with pytest.raises(ValueError, match="превышает максимум"):
            attn(torch.randn(1, 1, params["emb_size"]), cache=cache)


def test_use_cache_false_returns_no_cache(params):
    attn = make_attn(params, window_size=4)
    _, cache = attn(torch.randn(1, 5, params["emb_size"]), use_cache=False)
    assert cache is None


def left_padding(real_lens, total_len):
    """Padding для батча с левым паддингом: настоящие токены — последние real_len в строке."""
    mask = torch.zeros(len(real_lens), total_len, dtype=torch.long)
    for row, n in enumerate(real_lens):
        mask[row, total_len - n:] = 1
    return padding_from_attention_mask(mask, torch.zeros(len(real_lens), total_len))


@pytest.mark.parametrize("rope", [True, False])
def test_left_padding_does_not_change_real_tokens(params, rope):
    """Настоящие токены строки с левым паддингом дают то же, что строка без паддинга."""
    attn = make_attn(params, window_size=None, rope=rope)
    real, pad = 6, 3
    x = torch.randn(1, real, params["emb_size"])
    padded = torch.cat([torch.randn(1, pad, params["emb_size"]), x], dim=1)
    with torch.no_grad():
        plain, _ = attn(x)
        masked, _ = attn(padded, padding=left_padding([real], real + pad))
    assert torch.allclose(plain, masked[:, pad:], atol=1e-5)


def test_left_padding_with_cache_matches_unpadded(params):
    """Префилл с левым паддингом и дальше шаги с кэшем: настоящие токены как без паддинга."""
    attn = make_attn(params, window_size=None)
    real, pad = 5, 2
    x = torch.randn(1, real + 3, params["emb_size"])
    padded_prefill = torch.cat([torch.randn(1, pad, params["emb_size"]), x[:, :real]], dim=1)
    with torch.no_grad():
        plain, _ = run_in_chunks(attn, x, real)
        _, cache = attn(padded_prefill, padding=left_padding([real], real + pad))
        outs = []
        for t in range(real, real + 3):
            mask = torch.ones(1, pad + t + 1, dtype=torch.long)
            mask[:, :pad] = 0
            step = x[:, t:t + 1]
            padding = padding_from_attention_mask(mask, step[..., 0], start_pos=pad + t)
            out, cache = attn(step, cache=cache, padding=padding)
            outs.append(out)
    assert torch.allclose(plain[:, real:], torch.cat(outs, dim=1), atol=1e-5)


def test_padding_in_batch_rows_are_independent(params):
    """В батче с разной длиной каждая строка считается как сама по себе."""
    attn = make_attn(params, window_size=None)
    lens, total = [6, 3], 6
    rows = [torch.randn(1, n, params["emb_size"]) for n in lens]
    batch = torch.zeros(2, total, params["emb_size"])
    for i, (r, n) in enumerate(zip(rows, lens)):
        batch[i, total - n:] = r[0]
    with torch.no_grad():
        out, _ = attn(batch, padding=left_padding(lens, total))
        for i, (r, n) in enumerate(zip(rows, lens)):
            alone, _ = attn(r)
            assert torch.allclose(out[i:i + 1, total - n:], alone, atol=1e-5)
            assert torch.isfinite(out).all()
