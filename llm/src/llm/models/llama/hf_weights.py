"""
Перенос весов LLaMA из формата HuggingFace (`LlamaForCausalLM`).

Отличия формата:
- у HF и у Meta одинаковые матрицы, но разный порядок строк `q_proj` и `k_proj`. RoPE здесь,
  как в коде Meta, поворачивает соседние пары координат (2i, 2i+1), а HF (`rotate_half`) —
  пары (i, i + head_size/2). Скрипт конвертации HF переставляет строки при переходе от Meta,
  здесь — обратная перестановка;
- имена слоёв: `model.layers.N.self_attn.q_proj` → `_decoders.N._heads._q` и т. д.

Подходят модели с обычным multi-head attention (num_key_value_heads == num_attention_heads),
без rope_scaling. Конфиг должен совпадать с HF: "intermediate_size" — как у модели,
"bias": False (если в HF нет attention_bias и mlp_bias), rms_norm_eps и rope_theta — как в HF.

Пример:
    >>> from transformers import LlamaForCausalLM
    >>> hf = LlamaForCausalLM.from_pretrained("nickypro/tinyllama-15M")
    >>> model = Llama({..., "intermediate_size": 768, "bias": False, "rms_norm_eps": 1e-5})
    >>> model.load_state_dict(convert_hf_state_dict(hf.state_dict(), num_heads=6))
"""

import re

import torch

_LAYER_KEYS = {
    "self_attn.q_proj": "_heads._q",
    "self_attn.k_proj": "_heads._k",
    "self_attn.v_proj": "_heads._v",
    "self_attn.o_proj": "_heads._layer",
    "mlp.gate_proj": "_ff._gate",
    "mlp.up_proj": "_ff._up",
    "mlp.down_proj": "_ff._down",
    "input_layernorm": "_norm1",
    "post_attention_layernorm": "_norm2",
}


def _hf_to_meta_rows(value: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Строки головы в HF: [x_0 … x_{d/2−1} | y_0 … y_{d/2−1}], здесь: [x_0, y_0, x_1, y_1, …].
    Годится и для веса [out, in], и для bias [out].
    """
    head_size = value.shape[0] // num_heads
    rest = value.shape[1:]
    return (
        value.reshape(num_heads, 2, head_size // 2, *rest)
        .transpose(1, 2)
        .reshape(value.shape)
        .contiguous()
    )


def convert_hf_state_dict(hf_state_dict: dict, num_heads: int) -> dict:
    """
    Преобразует state_dict `LlamaForCausalLM` в state_dict `Llama`.

    num_heads — число attention-голов (config.num_attention_heads): нужно для перестановки
    строк Q и K. Если в чекпоинте нет `lm_head.weight` (эмбеддинги привязаны), голова
    получает копию эмбеддингов: здесь у LLaMA weight tying нет, но на выходе это то же самое.
    Буферы `rotary_emb.inv_freq` из старых чекпоинтов пропускаются.
    """
    result = {}
    for key, value in hf_state_dict.items():
        if key == "model.embed_tokens.weight":
            result["_token_embeddings._embedding.weight"] = value
        elif key == "model.norm.weight":
            result["_norm._w"] = value
        elif key.startswith("lm_head."):
            result["_linear." + key.removeprefix("lm_head.")] = value
        elif key.endswith("rotary_emb.inv_freq"):
            continue
        else:
            match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)\.(weight|bias)", key)
            if match is None or match.group(2) not in _LAYER_KEYS:
                raise KeyError(f"Неизвестный ключ HF-чекпоинта: {key}")
            layer, name, kind = match.groups()
            if name in ("self_attn.q_proj", "self_attn.k_proj"):
                value = _hf_to_meta_rows(value, num_heads)
            # у RMSNorm здесь параметр называется _w
            kind = "_w" if name.endswith("layernorm") else kind
            result[f"_decoders.{layer}.{_LAYER_KEYS[name]}.{kind}"] = value

    if "_linear.weight" not in result:
        result["_linear.weight"] = result["_token_embeddings._embedding.weight"].clone()
    return result
