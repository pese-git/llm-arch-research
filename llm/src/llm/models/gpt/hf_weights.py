"""
Перенос весов GPT-1 и GPT-2 из формата HuggingFace (`openai-community/openai-gpt`, `gpt2`).

Веса оригинальных моделей подходят только к конфигурации с `tie_word_embeddings=True`:
у них нет отдельной матрицы и bias выходной проекции. Остальные отличия формата:
- HF хранит Q, K и V одной матрицей `c_attn`, здесь это три Linear;
- слои HF — `Conv1D` с весом [in, out], а у `nn.Linear` вес [out, in].

Пример:
    >>> from transformers import GPT2LMHeadModel
    >>> hf = GPT2LMHeadModel.from_pretrained("gpt2")
    >>> model = GPT2({..., "tie_word_embeddings": True})
    >>> model.load_state_dict(convert_hf_state_dict(hf.state_dict()))
"""

import re

import torch

# Имена HF внутри блока → имена здесь; Conv1D-веса транспонируются
_BLOCK_KEYS = {
    "attn.c_proj": "_heads._layer",
    "mlp.c_fc": "_ff._layer1",
    "mlp.c_proj": "_ff._layer2",
    "ln_1": "_norm1",
    "ln_2": "_norm2",
}
_CONV1D = {"attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"}

_TOP_KEYS = {
    "tokens_embed.weight": "_token_embeddings._embedding.weight",  # GPT-1
    "positions_embed.weight": "_position_embeddings.embedding.weight",
    "wte.weight": "_token_embeddings._embedding.weight",  # GPT-2
    "wpe.weight": "_position_embeddings.embedding.weight",
    "ln_f.weight": "_norm.weight",
    "ln_f.bias": "_norm.bias",
}


def convert_hf_state_dict(hf_state_dict: dict) -> dict:
    """
    Преобразует state_dict `OpenAIGPTLMHeadModel` / `GPT2LMHeadModel` (или их базовых
    моделей) в state_dict `GPT` / `GPT2` с `tie_word_embeddings=True`.

    `lm_head.weight` пропускается: он совпадает с эмбеддингами. Пропускаются и буферы
    causal-маски (`attn.bias`, `attn.masked_bias`), которые есть в старых чекпоинтах.
    """
    result = {}
    for key, value in hf_state_dict.items():
        key = key.removeprefix("transformer.")
        if key in _TOP_KEYS:
            result[_TOP_KEYS[key]] = value
            continue
        if key == "lm_head.weight":
            continue

        match = re.fullmatch(r"h\.(\d+)\.(.+)\.(weight|bias)", key)
        if match is None:
            if re.fullmatch(r"h\.\d+\.attn\.(masked_)?bias", key):
                continue
            raise KeyError(f"Неизвестный ключ HF-чекпоинта: {key}")
        layer, name, kind = match.groups()
        prefix = f"_decoders.{layer}."
        if name in _CONV1D and kind == "weight":
            value = value.t()

        if name == "attn.c_attn":
            # [3·d, d] (вес) или [3·d] (bias): Q, K, V подряд
            for part, chunk in zip(("_q", "_k", "_v"), torch.chunk(value, 3, dim=0)):
                result[f"{prefix}_heads.{part}.{kind}"] = chunk.contiguous()
        elif name in _BLOCK_KEYS:
            result[f"{prefix}{_BLOCK_KEYS[name]}.{kind}"] = value.contiguous()
        else:
            raise KeyError(f"Неизвестный ключ HF-чекпоинта: {key}")

    # Выходная проекция делит веса с эмбеддингами: load_state_dict ждёт оба ключа
    result["_linear.weight"] = result["_token_embeddings._embedding.weight"]
    return result
